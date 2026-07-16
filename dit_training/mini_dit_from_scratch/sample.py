import argparse
from pathlib import Path

import torch

from mini_dit.diffusion import Diffusion, DiffusionConfig
from mini_dit.image_io import save_pgm_grid
from mini_dit.model import MiniDiT, MiniDiTConfig


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 0 写的 mini DiT 采样脚本")
    parser.add_argument("--ckpt", default="outputs/mini_dit.pt")
    parser.add_argument("--out", default="outputs/samples.pgm")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"找不到 checkpoint：{args.ckpt}，请先运行 train.py")

    ckpt = load_checkpoint(args.ckpt, device)
    model_cfg = MiniDiTConfig(**ckpt["model_cfg"])
    diffusion_cfg = DiffusionConfig(**ckpt["diffusion_cfg"])
    model = MiniDiT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = Diffusion(diffusion_cfg, device)
    y = torch.arange(args.num_samples, device=device) % model_cfg.num_classes
    shape = (args.num_samples, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size)
    images = diffusion.p_sample_loop(model, shape, y, device)
    save_pgm_grid(images, args.out)
    print(f"saved={args.out}")
    print("前几张类别标签:", y[: min(8, args.num_samples)].detach().cpu().tolist())


if __name__ == "__main__":
    main()
