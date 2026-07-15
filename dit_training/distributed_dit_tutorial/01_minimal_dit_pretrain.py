import argparse

import torch

from dit_common import DiTConfig, DiffusionSchedule, TinyDiT, count_parameters, seed_everything, train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="最简 DiT 预训练：单进程版本")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg = DiTConfig()
    model = TinyDiT(cfg).to(device)
    schedule = DiffusionSchedule(cfg.timesteps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device} parameters={count_parameters(model):,}")
    for step in range(1, args.steps + 1):
        loss = train_step(model, optimizer, schedule, cfg, args.batch_size, device)
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step:03d} loss={loss:.4f}")


if __name__ == "__main__":
    main()
