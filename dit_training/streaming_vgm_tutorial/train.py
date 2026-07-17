import argparse
from pathlib import Path

import torch

from streaming_vgm.data import make_batch
from streaming_vgm.model import StreamingVGM


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 mini streaming VGM")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save", default="outputs/streaming_vgm.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreamingVGM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        latents, prompt_id = make_batch(args.batch_size, args.frames, device)
        loss = model.loss(latents, prompt_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step:04d} loss={float(loss.detach().cpu()):.5f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "frames": args.frames}, args.save)
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
