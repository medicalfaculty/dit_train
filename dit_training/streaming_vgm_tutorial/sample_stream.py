import argparse
from pathlib import Path

import torch

from streaming_vgm.data import PROMPTS, decode_latents_to_video, save_video_strip
from streaming_vgm.model import StreamingVGM


def main() -> None:
    parser = argparse.ArgumentParser(description="用 KV cache 流式采样 mini VGM")
    parser.add_argument("--ckpt", default="outputs/streaming_vgm.pt")
    parser.add_argument("--prompt", default="move right", choices=PROMPTS)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--out", default="outputs/stream_sample.pgm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"找不到 checkpoint: {args.ckpt}，请先运行 train.py")
    model = StreamingVGM().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    prompt_id = torch.tensor([PROMPTS.index(args.prompt)], device=device)
    latents = model.generate_stream(prompt_id, args.frames)[0]
    video = decode_latents_to_video(latents)
    save_video_strip(video, args.out)
    print(f"saved={args.out} prompt={args.prompt!r} frames={args.frames}")


if __name__ == "__main__":
    main()
