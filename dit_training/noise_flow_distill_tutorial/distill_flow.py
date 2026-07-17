import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from flow_matching import sample_flow, train as train_flow
from toy_common import TimeMLP, seed_everything, save_scatter_svg


def load_or_train_teacher(args: argparse.Namespace, device: torch.device) -> TimeMLP:
    teacher = TimeMLP(hidden=args.hidden).to(device)
    if Path(args.teacher).exists():
        ckpt = torch.load(args.teacher, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt["model"])
        print(f"loaded teacher={args.teacher}")
        return teacher.eval()

    print("teacher checkpoint not found, train a small flow teacher first")
    flow_args = argparse.Namespace(
        steps=args.teacher_steps,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        save=args.teacher,
        svg="outputs/_teacher.svg",
        seed=args.seed,
    )
    return train_flow(flow_args, device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="把多步 Flow teacher 蒸馏成一步 student")
    parser.add_argument("--teacher", default="outputs/flow_model.pt")
    parser.add_argument("--save", default="outputs/distilled_student.pt")
    parser.add_argument("--svg", default="outputs/distilled_samples.svg")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--teacher-euler-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_or_train_teacher(args, device)
    student = TimeMLP(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        z = torch.randn(args.batch_size, 2, device=device)
        with torch.no_grad():
            # 蒸馏目标：teacher 用多步 ODE 从同一个 z 生成样本，student 学一步到位。
            target = z.clone()
            dt = 1.0 / args.teacher_euler_steps
            for i in range(args.teacher_euler_steps):
                t = torch.full((args.batch_size,), i / args.teacher_euler_steps, device=device)
                target = target + teacher(target, t) * dt
        pred = student(z, torch.zeros(args.batch_size, device=device))
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"distill step={step:04d} loss={float(loss.detach().cpu()):.4f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict(), "hidden": args.hidden}, args.save)
    with torch.no_grad():
        pts = student(torch.randn(512, 2, device=device), torch.zeros(512, device=device))
    save_scatter_svg(pts, args.svg, "one-step distilled samples")
    print(f"saved={args.save} svg={args.svg}")


if __name__ == "__main__":
    main()
