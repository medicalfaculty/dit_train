import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from toy_common import TimeMLP, seed_everything, sample_data, save_scatter_svg


def make_flow_batch(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.randn(batch, 2, device=device)       # 源分布：高斯噪声
    x1 = sample_data(batch, device)                # 目标分布：数据
    t = torch.rand(batch, device=device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1   # 直线路径
    v = x1 - x0                                    # Flow Matching 目标：速度场
    return xt, t, v


def train(args: argparse.Namespace, device: torch.device) -> TimeMLP:
    model = TimeMLP(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        xt, t, target_v = make_flow_batch(args.batch_size, device)
        # Flow Matching：模型直接预测 dx/dt，也就是把噪声搬到数据的速度。
        pred_v = model(xt, t)
        loss = F.mse_loss(pred_v, target_v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"flow step={step:04d} loss={float(loss.detach().cpu()):.4f}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden": args.hidden}, args.save)
    return model


@torch.no_grad()
def sample_flow(model: TimeMLP, n: int, euler_steps: int, device: torch.device) -> torch.Tensor:
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / euler_steps
    for i in range(euler_steps):
        t = torch.full((n,), i / euler_steps, device=device)
        x = x + model(x, t) * dt
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并采样：Flow Matching 速度场预测")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--euler-steps", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save", default="outputs/flow_model.pt")
    parser.add_argument("--svg", default="outputs/flow_samples.svg")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(args, device)
    pts = sample_flow(model.eval(), 512, args.euler_steps, device)
    save_scatter_svg(pts, args.svg, "flow matching samples")
    print(f"saved={args.save} svg={args.svg}")


if __name__ == "__main__":
    main()
