import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from toy_common import TimeMLP, seed_everything, sample_data, save_scatter_svg


class NoiseSchedule:
    def __init__(self, steps: int, device: torch.device):
        beta = torch.linspace(1e-4, 0.02, steps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.steps = steps
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_omab = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_ab = torch.sqrt(1.0 / alpha_bar)
        self.sqrt_recipm1_ab = torch.sqrt(1.0 / alpha_bar - 1.0)

    def gather(self, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return v.gather(0, t).view(t.shape[0], 1).expand_as(x)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self.gather(self.sqrt_ab, t, x0) * x0 + self.gather(self.sqrt_omab, t, x0) * eps

    def pred_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self.gather(self.sqrt_recip_ab, t, xt) * xt - self.gather(self.sqrt_recipm1_ab, t, xt) * eps


def train(args: argparse.Namespace, device: torch.device) -> TimeMLP:
    schedule = NoiseSchedule(args.diffusion_steps, device)
    model = TimeMLP(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        x0 = sample_data(args.batch_size, device)
        t_idx = torch.randint(0, args.diffusion_steps, (args.batch_size,), device=device)
        eps = torch.randn_like(x0)
        xt = schedule.add_noise(x0, t_idx, eps)
        # 噪声预测：模型目标是 eps，而不是 x0 或速度场。
        pred_eps = model(xt, t_idx.float() / args.diffusion_steps)
        loss = F.mse_loss(pred_eps, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"noise step={step:04d} loss={float(loss.detach().cpu()):.4f}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": args.diffusion_steps, "hidden": args.hidden}, args.save)
    return model


@torch.no_grad()
def sample(model: TimeMLP, n: int, diffusion_steps: int, device: torch.device) -> torch.Tensor:
    schedule = NoiseSchedule(diffusion_steps, device)
    x = torch.randn(n, 2, device=device)
    # 教学版 DDIM-like 更新：每步先预测 x0，再按较粗略公式回到前一时刻。
    for i in reversed(range(diffusion_steps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t.float() / diffusion_steps)
        x0 = schedule.pred_x0(x, t, eps).clamp(-3, 3)
        if i > 0:
            prev = torch.full((n,), i - 1, device=device, dtype=torch.long)
            x = schedule.gather(schedule.sqrt_ab, prev, x) * x0 + schedule.gather(schedule.sqrt_omab, prev, x) * eps
        else:
            x = x0
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并采样：DDPM 风格噪声预测")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save", default="outputs/noise_model.pt")
    parser.add_argument("--svg", default="outputs/noise_samples.svg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(args, device)
    pts = sample(model.eval(), 512, args.diffusion_steps, device)
    save_scatter_svg(pts, args.svg, "noise prediction samples")
    print(f"saved={args.save} svg={args.svg}")


if __name__ == "__main__":
    main()
