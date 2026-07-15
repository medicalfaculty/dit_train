import argparse
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


def shard_rows(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    rows = tensor.shape[0]
    assert rows % world_size == 0, "为了教学代码简洁，out_features 需要能整除 world_size"
    per_rank = rows // world_size
    return tensor[rank * per_rank:(rank + 1) * per_rank].contiguous()


def all_gather_rows(local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local)
    return torch.cat(parts, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="用一个 Linear 层演示 FSDP 的核心通信步骤")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=8)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    rank, world_size = setup()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(1234)

    full_weight_init = torch.randn(args.out_features, args.in_features, device=device) * 0.02
    local_weight = shard_rows(full_weight_init, rank, world_size).clone()

    if rank == 0:
        print(f"world_size={world_size} device={device}")
        print("这个脚本只演示核心思想：参数按 rank 分片，前向前 all_gather 成完整参数，反向后只保留本 rank 的梯度分片。")

    for step in range(1, args.steps + 1):
        torch.manual_seed(9000 + step)
        x = torch.randn(args.batch_size, args.in_features, device=device)
        target = torch.randn(args.batch_size, args.out_features, device=device)

        full_weight = all_gather_rows(local_weight, world_size)
        pred = F.linear(x, full_weight)
        loss = F.mse_loss(pred, target)
        # 教学版显式写出 Linear 的梯度：dL/dW = dL/dY^T @ X。
        # 真实 FSDP 会通过 autograd hook 和 reduce-scatter 把梯度归约并切回本地分片。
        grad_pred = 2.0 * (pred - target) / pred.numel()
        local_grad = shard_rows(grad_pred.t().matmul(x), rank, world_size)

        with torch.no_grad():
            local_weight -= args.lr * local_grad

        shown_loss = loss.detach()
        if world_size > 1:
            dist.all_reduce(shown_loss, op=dist.ReduceOp.SUM)
            shown_loss /= world_size
        if rank == 0:
            print(f"step={step:03d} loss={float(shown_loss.cpu()):.4f} local_shard_shape={tuple(local_weight.shape)}")

    cleanup()


if __name__ == "__main__":
    main()
