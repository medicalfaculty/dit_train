import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup() -> tuple[int, int]:
    if "RANK" not in os.environ:
        return 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def send_recv_ring(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size
    recv = torch.empty_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor.contiguous(), send_to),
        dist.P2POp(dist.irecv, recv, recv_from),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv


def all_gather_sequence(local_x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_x
    chunks = [torch.empty_like(local_x) for _ in range(world_size)]
    dist.all_gather(chunks, local_x)
    return torch.cat(chunks, dim=1)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return F.softmax(score, dim=-1) @ v


def ring_attention(local_q: torch.Tensor, local_k: torch.Tensor, local_v: torch.Tensor,
                   rank: int, world_size: int) -> torch.Tensor:
    # 在线 softmax：逐块扫描 K/V，维护每个 query 的 max、sum(exp)、加权输出。
    bsz, q_len, dim = local_q.shape
    running_max = torch.full((bsz, q_len, 1), -torch.inf, device=local_q.device)
    running_sum = torch.zeros((bsz, q_len, 1), device=local_q.device)
    running_out = torch.zeros_like(local_q)
    scale = 1.0 / math.sqrt(dim)

    cur_k = local_k
    cur_v = local_v
    for _ in range(world_size):
        score = local_q @ cur_k.transpose(-2, -1) * scale
        block_max = score.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, block_max)

        old_scale = torch.exp(running_max - new_max)
        block_exp = torch.exp(score - new_max)
        block_sum = block_exp.sum(dim=-1, keepdim=True)

        running_out = running_out * old_scale + block_exp @ cur_v
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

        cur_k = send_recv_ring(cur_k, rank, world_size)
        cur_v = send_recv_ring(cur_v, rank, world_size)

    return running_out / running_sum


def main() -> None:
    parser = argparse.ArgumentParser(description="Ring Attention 教学脚本")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    rank, world_size = setup()
    assert args.seq_len % world_size == 0, "seq-len 必须能整除 world_size"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(100)
    full_q = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
    full_k = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
    full_v = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)

    local_seq = args.seq_len // world_size
    sl = slice(rank * local_seq, (rank + 1) * local_seq)
    local_q = full_q[:, sl].contiguous()
    local_k = full_k[:, sl].contiguous()
    local_v = full_v[:, sl].contiguous()

    local_out = ring_attention(local_q, local_k, local_v, rank, world_size)
    gathered_k = all_gather_sequence(local_k, world_size)
    gathered_v = all_gather_sequence(local_v, world_size)
    reference_local = dense_attention(local_q, gathered_k, gathered_v)

    max_error = (local_out - reference_local).abs().max()
    if world_size > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"world_size={world_size} device={device}")
        print(f"local_q={tuple(local_q.shape)} local_kv={tuple(local_k.shape)}")
        print(f"max_error_vs_dense_attention={float(max_error.cpu()):.8f}")
        print("Ring Attention 的重点：K/V 块沿 ring 传递；每个 rank 边收边算，不一次保存完整 K/V。")

    cleanup()


if __name__ == "__main__":
    main()
