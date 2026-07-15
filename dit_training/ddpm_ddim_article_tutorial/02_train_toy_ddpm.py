import argparse
from pathlib import Path

import torch

from diffusion_toy import DiffusionConfig, Schedule, TinyEpsNet, seed_everything, train_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="训练一个极小 DDPM 噪声预测网络")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save", default="outputs/tiny_eps_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DiffusionConfig(timesteps=100)
    schedule = Schedule(cfg, device)
    model = TinyEpsNet(cfg.timesteps).to(device)

    losses = train_steps(model, schedule, args.steps, args.batch_size, args.lr, device)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "losses": losses}, args.save)

    print(f"device={device} steps={args.steps} first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}")
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
