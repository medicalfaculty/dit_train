import argparse

import torch

from diffusion_toy import DiffusionConfig, Schedule, make_toy_images, q_sample, save_pgm_grid, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM 前向加噪公式演示")
    parser.add_argument("--out", default="outputs/forward_noise.pgm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DiffusionConfig(timesteps=100)
    schedule = Schedule(cfg, device)

    x0 = make_toy_images(1, cfg.image_size, device)
    noise = torch.randn_like(x0)
    show_steps = torch.tensor([0, 10, 30, 60, 99], device=device)
    imgs = [x0]
    for step in show_steps:
        t = step.view(1)
        imgs.append(q_sample(x0, t, noise, schedule))

    save_pgm_grid(torch.cat(imgs, dim=0), args.out, nrow=len(imgs))
    print(f"saved={args.out}")
    for step in show_steps.tolist():
        print(f"t={step:03d} alpha_bar={schedule.alphas_bar[step].item():.4f}")


if __name__ == "__main__":
    main()
