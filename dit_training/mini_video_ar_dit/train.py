import argparse
from pathlib import Path

import torch

from mini_video_ar_dit.config import DataConfig, ModelConfig
from mini_video_ar_dit.data import build_cache, collate_pack, load_cache, make_packs
from mini_video_ar_dit.model import ARVideoDiT


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 mini 自回归视频 DiT")
    parser.add_argument("--cache", default="outputs/cache.pt")
    parser.add_argument("--save", default="outputs/ar_video_dit.pt")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--token-budget", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--attention", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.cache).exists():
        build_cache(args.cache, num_samples=32, cfg=DataConfig())

    data_cfg, records = load_cache(args.cache)
    max_latent_tokens = data_cfg.max_frames * data_cfg.tokens_per_frame
    model_cfg = ModelConfig(
        latent_dim=data_cfg.latent_dim,
        text_dim=data_cfg.latent_dim,
        max_tokens=data_cfg.text_len + max_latent_tokens + 1,
        attention=args.attention,
    )
    model = ARVideoDiT(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    while step < args.steps:
        for pack in make_packs(records, args.token_budget, shuffle=True):
            batch = collate_pack(pack, device)
            loss = model.loss(batch["latents"], batch["text"], batch["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step == 1 or step % 10 == 0 or step == args.steps:
                packed_tokens = int(batch["mask"].sum().item())
                print(f"step={step:04d} loss={float(loss.detach().cpu()):.4f} packed_tokens={packed_tokens}")
            if step >= args.steps:
                break

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg.__dict__, "data_cfg": data_cfg.__dict__}, args.save)
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
