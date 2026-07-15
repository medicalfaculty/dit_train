import argparse
import time
from pathlib import Path

import torch

from diffusion_toy import DiffusionConfig, Schedule, TinyEpsNet, ddpm_sample, ddim_sample
from diffusion_toy import save_pgm_grid, seed_everything, train_steps


def load_or_train_model(path: Path, cfg: DiffusionConfig, schedule: Schedule,
                        device: torch.device, warmup_steps: int) -> TinyEpsNet:
    model = TinyEpsNet(cfg.timesteps).to(device)
    if path.exists():
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"loaded={path}")
    else:
        print(f"{path} not found, train {warmup_steps} quick steps first")
        losses = train_steps(model, schedule, warmup_steps, 64, 2e-3, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "losses": losses}, path)
        print(f"warmup_first_loss={losses[0]:.4f} warmup_last_loss={losses[-1]:.4f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 DDPM 逐步采样和 DDIM 跳步采样")
    parser.add_argument("--checkpoint", default="outputs/tiny_eps_model.pt")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DiffusionConfig(timesteps=100)
    schedule = Schedule(cfg, device)
    model = load_or_train_model(Path(args.checkpoint), cfg, schedule, device, args.warmup_steps)
    model.eval()

    shape = (args.sample_size, cfg.channels, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        start = time.perf_counter()
        ddpm = ddpm_sample(model, schedule, shape, device)
        ddpm_sec = time.perf_counter() - start

        start = time.perf_counter()
        ddim = ddim_sample(model, schedule, shape, device, sample_steps=args.ddim_steps, eta=0.0)
        ddim_sec = time.perf_counter() - start

    save_pgm_grid(ddpm, "outputs/ddpm_samples.pgm")
    save_pgm_grid(ddim, "outputs/ddim_samples.pgm")
    print(f"DDPM steps={cfg.timesteps} time={ddpm_sec:.3f}s saved=outputs/ddpm_samples.pgm")
    print(f"DDIM steps={args.ddim_steps} time={ddim_sec:.3f}s saved=outputs/ddim_samples.pgm")
    print("提示：toy 模型训练很少，图片质量不是重点；重点是 DDIM 用更少步数复用同一个噪声预测模型。")


if __name__ == "__main__":
    main()
