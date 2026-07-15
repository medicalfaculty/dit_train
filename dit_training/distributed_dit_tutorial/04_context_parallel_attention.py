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


def all_gather_sequence(local_x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_x
    chunks = [torch.empty_like(local_x) for _ in range(world_size)]
    dist.all_gather(chunks, local_x)
    return torch.cat(chunks, dim=1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return F.softmax(score, dim=-1) @ v


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Parallel Attention 教学脚本")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    rank, world_size = setup()
    assert args.seq_len % world_size == 0, "seq-len 必须能整除 world_size"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(2026)
    full_x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
    local_seq = args.seq_len // world_size
    local_x = full_x[:, rank * local_seq:(rank + 1) * local_seq].contiguous()

    torch.manual_seed(7)
    wq = torch.randn(args.dim, args.dim, device=device) / math.sqrt(args.dim)
    wk = torch.randn(args.dim, args.dim, device=device) / math.sqrt(args.dim)
    wv = torch.randn(args.dim, args.dim, device=device) / math.sqrt(args.dim)

    local_q = local_x @ wq
    gathered_x = all_gather_sequence(local_x, world_size)
    full_k = gathered_x @ wk
    full_v = gathered_x @ wv
    local_out = attention(local_q, full_k, full_v)

    full_q = full_x @ wq
    reference = attention(full_q, full_k, full_v)
    reference_local = reference[:, rank * local_seq:(rank + 1) * local_seq].contiguous()
    max_error = (local_out - reference_local).abs().max()
    if world_size > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"world_size={world_size} device={device}")
        print(f"full_sequence={tuple(full_x.shape)} local_sequence={tuple(local_x.shape)}")
        print(f"max_error_vs_full_attention={float(max_error.cpu()):.8f}")
        print("CP 的重点：序列维切分；每个 rank 只算本地 query，但 attention 需要全局 key/value。")

    cleanup()


if __name__ == "__main__":
    main()
