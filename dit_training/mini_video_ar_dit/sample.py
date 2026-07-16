import argparse
from pathlib import Path

import torch

from mini_video_ar_dit.config import DataConfig, ModelConfig
from mini_video_ar_dit.model import ARVideoDiT
from mini_video_ar_dit.toy_modules import ToyT5Encoder, ToyVideoVAE, save_video_strip


def load_ckpt(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 mini 自回归视频 DiT 采样")
    parser.add_argument("--ckpt", default="outputs/ar_video_dit.pt")
    parser.add_argument("--prompt", default="moving square right")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--out", default="outputs/sample_strip.pgm")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"找不到 checkpoint: {args.ckpt}，请先运行 train.py")

    ckpt = load_ckpt(args.ckpt, device)
    data_cfg = DataConfig(**ckpt["data_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = ARVideoDiT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    t5 = ToyT5Encoder(data_cfg)
    vae = ToyVideoVAE(data_cfg)
    text = t5.encode(args.prompt).unsqueeze(0).to(device)
    steps = args.frames * data_cfg.tokens_per_frame
    latents = model.generate(text, steps=steps)[0].cpu()
    video = vae.decode(latents, frames=args.frames)
    save_video_strip(video, args.out)
    print(f"saved={args.out} prompt={args.prompt!r} frames={args.frames}")


if __name__ == "__main__":
    main()
