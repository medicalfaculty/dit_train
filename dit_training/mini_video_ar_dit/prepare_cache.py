import argparse

from mini_video_ar_dit.config import DataConfig
from mini_video_ar_dit.data import build_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="离线准备 T5/VAE 缓存，训练阶段只读 latent/text token")
    parser.add_argument("--out", default="outputs/cache.pt")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = DataConfig()
    build_cache(args.out, args.num_samples, cfg, args.seed)
    print(f"saved={args.out} num_samples={args.num_samples}")
    print("离线阶段完成：视频解码、Toy VAE、Toy T5 已经不在训练链路里。")


if __name__ == "__main__":
    main()
