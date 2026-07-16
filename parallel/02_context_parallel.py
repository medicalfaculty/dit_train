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


def all_gather_seq(local_x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_x
    parts = [torch.empty_like(local_x) for _ in range(world_size)]
    dist.all_gather(parts, local_x)
    return torch.cat(parts, dim=1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return F.softmax(score, dim=-1) @ v


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Parallel 本质演示")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4)
    args = parser.parse_args()

    rank, world_size = setup()
    assert args.seq_len % world_size == 0, "seq-len 必须能整除进程数"

    torch.manual_seed(1)
    full_x = torch.randn(args.batch_size, args.seq_len, args.dim)
    wq = torch.randn(args.dim, args.dim)
    wk = torch.randn(args.dim, args.dim)
    wv = torch.randn(args.dim, args.dim)

    local_len = args.seq_len // world_size
    start, end = rank * local_len, (rank + 1) * local_len
    local_x = full_x[:, start:end].contiguous()

    # CP 的核心：每个 rank 只负责本地 query，但 query 需要看全局 key/value。
    local_q = local_x @ wq
    global_x = all_gather_seq(local_x, world_size)
    global_k = global_x @ wk
    global_v = global_x @ wv
    local_out = attention(local_q, global_k, global_v)

    reference = attention(full_x @ wq, full_x @ wk, full_x @ wv)
    max_error = (local_out - reference[:, start:end]).abs().max()
    if world_size > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"world_size={world_size}")
        print(f"local_q={tuple(local_q.shape)} global_kv={tuple(global_k.shape)}")
        print(f"max_error_vs_full_attention={float(max_error):.8f}")
        print("CP 本质：序列分片常驻本地；attention 时让本地 Q 看到全局 K/V。")

    cleanup()


if __name__ == "__main__":
    main()
