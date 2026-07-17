import argparse
import time
from pathlib import Path

import torch

from streaming_vgm.data import PROMPTS
from streaming_vgm.model import StreamingVGM


@torch.no_grad()
def generate_naive(model: StreamingVGM, prompt_id: torch.Tensor, frames: int) -> torch.Tensor:
    """朴素推理：每生成一帧，都把完整历史重新喂给模型。"""
    latents = torch.zeros(prompt_id.shape[0], 0, model.latent_dim, device=prompt_id.device)
    for _ in range(frames):
        padded = torch.cat([latents, torch.zeros(prompt_id.shape[0], 1, model.latent_dim, device=prompt_id.device)], dim=1)
        pred = model(padded, prompt_id)[:, -1:]
        latents = torch.cat([latents, pred], dim=1)
    return latents


def load_model(ckpt: str, device: torch.device) -> StreamingVGM:
    model = StreamingVGM().to(device)
    if Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    else:
        print(f"warning: {ckpt} not found, benchmark an untrained model")
    return model.eval()


def time_call(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def attention_score_counts(frames: int) -> tuple[int, int]:
    # 只数每层每头的 attention score 个数：naive 每步重算完整历史，cache 每步只算新增 token。
    naive = sum((i + 2) * (i + 2) for i in range(frames))
    cached = 1 + sum(i + 2 for i in range(frames))
    return naive, cached


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 streaming VGM naive 推理和 KV cache 推理")
    parser.add_argument("--ckpt", default="outputs/streaming_vgm.pt")
    parser.add_argument("--prompt", default="move right", choices=PROMPTS)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    prompt_id = torch.tensor([PROMPTS.index(args.prompt)], device=device)

    naive = time_call(lambda: generate_naive(model, prompt_id, args.frames), warmup=1, repeat=args.repeat)
    cached = time_call(lambda: model.generate_stream(prompt_id, args.frames), warmup=1, repeat=args.repeat)
    speedup = naive / cached if cached > 0 else float("inf")
    naive_scores, cached_scores = attention_score_counts(args.frames)
    score_reduction = naive_scores / cached_scores

    print(f"device={device} frames={args.frames} repeat={args.repeat}")
    print(f"naive_full_history={naive:.6f}s")
    print(f"kv_cache_streaming={cached:.6f}s")
    print(f"speedup={speedup:.2f}x")
    print(f"attention_scores_naive={naive_scores} attention_scores_cache={cached_scores}")
    print(f"theoretical_attention_reduction={score_reduction:.2f}x")
    print("说明：tiny CPU 模型可能因 Python 循环开销导致 cache 实测不快；真实大模型/GPU decode 更看重 attention 计算量和显存增长。")


if __name__ == "__main__":
    main()
