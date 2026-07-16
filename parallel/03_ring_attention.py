import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup() -> tuple[int, int]:
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("gloo")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def ring_send_recv(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size
    recv = torch.empty_like(x)
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, x.contiguous(), send_to),
        dist.P2POp(dist.irecv, recv, recv_from),
    ])
    for req in reqs:
        req.wait()
    return recv


def all_gather_seq(local_x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_x
    parts = [torch.empty_like(local_x) for _ in range(world_size)]
    dist.all_gather(parts, local_x)
    return torch.cat(parts, dim=1)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return F.softmax(score, dim=-1) @ v


def ring_attention(local_q: torch.Tensor, local_k: torch.Tensor, local_v: torch.Tensor,
                   rank: int, world_size: int) -> torch.Tensor:
    b, q_len, dim = local_q.shape
    scale = 1.0 / math.sqrt(dim)
    m = torch.full((b, q_len, 1), -torch.inf)
    l = torch.zeros((b, q_len, 1))
    out = torch.zeros_like(local_q)

    cur_k, cur_v = local_k, local_v
    for _ in range(world_size):
        # 在线 softmax：每次只处理一个 KV 块，但累计结果等价于 full attention。
        score = local_q @ cur_k.transpose(-2, -1) * scale
        block_m = score.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(m, block_m)
        old_scale = torch.exp(m - new_m)
        block_exp = torch.exp(score - new_m)

        out = out * old_scale + block_exp @ cur_v
        l = l * old_scale + block_exp.sum(dim=-1, keepdim=True)
        m = new_m

        # KV 沿 ring 传给下一个 rank；每个 rank 不必一次保存完整 KV。
        cur_k = ring_send_recv(cur_k, rank, world_size)
        cur_v = ring_send_recv(cur_v, rank, world_size)

    return out / l


def main() -> None:
    parser = argparse.ArgumentParser(description="Ring Attention 本质演示")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4)
    args = parser.parse_args()

    rank, world_size = setup()
    assert args.seq_len % world_size == 0, "seq-len 必须能整除进程数"

    torch.manual_seed(2)
    full_q = torch.randn(args.batch_size, args.seq_len, args.dim)
    full_k = torch.randn(args.batch_size, args.seq_len, args.dim)
    full_v = torch.randn(args.batch_size, args.seq_len, args.dim)

    local_len = args.seq_len // world_size
    start, end = rank * local_len, (rank + 1) * local_len
    local_q = full_q[:, start:end].contiguous()
    local_k = full_k[:, start:end].contiguous()
    local_v = full_v[:, start:end].contiguous()

    local_out = ring_attention(local_q, local_k, local_v, rank, world_size)
    global_k = all_gather_seq(local_k, world_size)
    global_v = all_gather_seq(local_v, world_size)
    reference = dense_attention(local_q, global_k, global_v)
    max_error = (local_out - reference).abs().max()
    if world_size > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"world_size={world_size}")
        print(f"local_q={tuple(local_q.shape)} local_kv={tuple(local_k.shape)}")
        print(f"max_error_vs_dense_attention={float(max_error):.8f}")
        print("Ring Attention 本质：KV 块环形流动，用在线 softmax 避免 all_gather 完整 KV。")

    cleanup()


if __name__ == "__main__":
    main()
