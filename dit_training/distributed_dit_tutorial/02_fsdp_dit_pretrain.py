import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from dit_common import DiTConfig, DiffusionSchedule, TinyDiT, count_parameters, seed_everything, train_step


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="DiT 预训练：FSDP 版本")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="每个 rank 的 batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = DiTConfig()
    model = TinyDiT(cfg)
    base_param_count = count_parameters(model)

    using_fsdp = world_size > 1 and torch.cuda.is_available()
    if using_fsdp:
        model = FSDP(model, device_id=local_rank, use_orig_params=True)
    elif world_size > 1 and rank == 0:
        print("当前是 CPU 多进程环境，PyTorch FSDP 通常需要 CUDA/NCCL；本脚本将运行非 FSDP smoke test。")
        model = model.to(device)
    else:
        model = model.to(device)

    schedule = DiffusionSchedule(cfg.timesteps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        mode = "FSDP" if using_fsdp else "single-process/plain"
        print(f"mode={mode} world_size={world_size} device={device} parameters={base_param_count:,}")

    for step in range(1, args.steps + 1):
        loss = train_step(model, optimizer, schedule, cfg, args.batch_size, device)
        if dist.is_initialized():
            loss_tensor = torch.tensor(loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= world_size
            loss = float(loss_tensor.cpu())
        if rank == 0 and (step == 1 or step % 5 == 0 or step == args.steps):
            print(f"step={step:03d} avg_loss={loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
