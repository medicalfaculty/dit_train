import argparse
from pathlib import Path

import torch

from distill_flow import load_or_train_teacher
from flow_matching import sample_flow
from noise_prediction import sample as sample_noise
from toy_common import TimeMLP, save_scatter_svg, seed_everything


def load_time_mlp(path: str, hidden: int, device: torch.device) -> TimeMLP:
    model = TimeMLP(hidden=hidden).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="加载已有模型，生成三个 SVG 对比")
    parser.add_argument("--noise", default="outputs/noise_model.pt")
    parser.add_argument("--flow", default="outputs/flow_model.pt")
    parser.add_argument("--student", default="outputs/distilled_student.pt")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(args.noise).exists():
        noise = load_time_mlp(args.noise, args.hidden, device)
        pts = sample_noise(noise, 512, 100, device)
        save_scatter_svg(pts, "outputs/compare_noise.svg", "noise prediction")
    if Path(args.flow).exists():
        flow = load_time_mlp(args.flow, args.hidden, device)
        pts = sample_flow(flow, 512, 20, device)
        save_scatter_svg(pts, "outputs/compare_flow.svg", "flow matching")
    if Path(args.student).exists():
        student = load_time_mlp(args.student, args.hidden, device)
        z = torch.randn(512, 2, device=device)
        pts = student(z, torch.zeros(512, device=device))
        save_scatter_svg(pts, "outputs/compare_distilled.svg", "distilled one-step")
    print("saved outputs/compare_*.svg for existing checkpoints")


if __name__ == "__main__":
    main()
