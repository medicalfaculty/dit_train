from dataclasses import asdict
from pathlib import Path

import torch

from .config import DataConfig
from .toy_modules import ToyT5Encoder, ToyVideoVAE, make_toy_video


PROMPTS = [
    "moving square right",
    "moving square down",
    "horizontal bright block",
    "vertical bright block",
]


def build_cache(path: str, num_samples: int, cfg: DataConfig, seed: int = 0) -> None:
    """离线阶段：视频解码、VAE、T5 都在这里做，训练脚本不再碰这些重活。"""
    torch.manual_seed(seed)
    t5 = ToyT5Encoder(cfg)
    vae = ToyVideoVAE(cfg)
    records = []
    for idx in range(num_samples):
        prompt = PROMPTS[idx % len(PROMPTS)]
        frames = int(torch.randint(cfg.min_frames, cfg.max_frames + 1, (1,)))
        video = make_toy_video(prompt, frames, cfg)
        records.append(
            {
                "prompt": prompt,
                "frames": frames,
                "text": t5.encode(prompt),
                "latents": vae.encode(video),
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(cfg), "records": records}, path)


def load_cache(path: str) -> tuple[DataConfig, list[dict]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return DataConfig(**ckpt["cfg"]), ckpt["records"]


def make_packs(records: list[dict], token_budget: int, shuffle: bool = True) -> list[list[dict]]:
    """动态 packing：按 token 总数凑 batch，降低长短视频混在一起造成的 padding 浪费。"""
    order = torch.randperm(len(records)).tolist() if shuffle else list(range(len(records)))
    packs, cur, cur_tokens = [], [], 0
    for i in order:
        n = int(records[i]["latents"].shape[0])
        if cur and cur_tokens + n > token_budget:
            packs.append(cur)
            cur, cur_tokens = [], 0
        cur.append(records[i])
        cur_tokens += n
    if cur:
        packs.append(cur)
    return packs


def collate_pack(pack: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(x["latents"].shape[0] for x in pack)
    text_len, text_dim = pack[0]["text"].shape
    latent_dim = pack[0]["latents"].shape[-1]
    bsz = len(pack)
    latents = torch.zeros(bsz, max_len, latent_dim, device=device)
    text = torch.zeros(bsz, text_len, text_dim, device=device)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool, device=device)
    frames = torch.tensor([x["frames"] for x in pack], device=device)
    for i, item in enumerate(pack):
        n = item["latents"].shape[0]
        latents[i, :n] = item["latents"].to(device)
        text[i] = item["text"].to(device)
        mask[i, :n] = True
    return {"latents": latents, "text": text, "mask": mask, "frames": frames}
