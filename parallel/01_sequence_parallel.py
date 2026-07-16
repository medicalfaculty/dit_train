import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence Parallel 本质演示")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4)
    args = parser.parse_args()

    rank, world_size = setup()
    assert args.seq_len % world_size == 0, "seq-len 必须能整除进程数"

    torch.manual_seed(0)
    full_x = torch.randn(args.batch_size, args.seq_len, args.dim)
    full_weight = torch.randn(args.dim, args.dim)

    local_len = args.seq_len // world_size
    start, end = rank * local_len, (rank + 1) * local_len
    local_x = full_x[:, start:end].contiguous()

    # SP 的核心：逐 token 算子可以只在本地序列片段上算，不需要全序列。
    local_y = F.layer_norm(local_x, (args.dim,)) @ full_weight

    # 如果下一层需要完整序列，就 all_gather；如果下一层仍可按序列切分，就继续保留 local_y。
    gathered_y = all_gather_seq(local_y, world_size)
    reference_y = F.layer_norm(full_x, (args.dim,)) @ full_weight
    max_error = (gathered_y - reference_y).abs().max()
    if world_size > 1:
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"world_size={world_size}")
        print(f"full_x={tuple(full_x.shape)} local_x={tuple(local_x.shape)}")
        print(f"max_error_vs_full={float(max_error):.8f}")
        print("SP 本质：把 sequence 维切开，LayerNorm/MLP 这类逐 token 计算天然可并行。")

    cleanup()


if __name__ == "__main__":
    main()
