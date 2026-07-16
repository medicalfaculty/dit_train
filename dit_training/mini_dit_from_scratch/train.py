import argparse
from pathlib import Path

import torch

from mini_dit.data import make_shape_batch
from mini_dit.diffusion import Diffusion, DiffusionConfig
from mini_dit.model import MiniDiT, MiniDiTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="从 0 写的 mini DiT 训练脚本")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save", default="outputs/mini_dit.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MiniDiTConfig()
    diffusion_cfg = DiffusionConfig()
    model = MiniDiT(model_cfg).to(device)
    diffusion = Diffusion(diffusion_cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x0, y = make_shape_batch(args.batch_size, model_cfg.image_size, device)
        loss = diffusion.training_loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"step={step:04d} loss={float(loss.detach().cpu()):.4f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "diffusion_cfg": diffusion_cfg.__dict__,
        },
        args.save,
    )
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
