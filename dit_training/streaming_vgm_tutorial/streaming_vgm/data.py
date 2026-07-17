from pathlib import Path

import torch


PROMPTS = ["move right", "move down", "move left", "move up"]


def make_batch(batch: int, frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 toy 视频 latent：每帧只有一个二维位置 [x, y]，范围约为 [-1, 1]。"""
    prompt_id = torch.randint(0, len(PROMPTS), (batch,), device=device)
    pos = torch.rand(batch, 2, device=device) * 1.2 - 0.6
    velocity = torch.zeros(batch, 2, device=device)
    velocity[prompt_id == 0, 0] = 0.18
    velocity[prompt_id == 1, 1] = 0.18
    velocity[prompt_id == 2, 0] = -0.18
    velocity[prompt_id == 3, 1] = -0.18
    steps = torch.arange(frames, device=device).float().view(1, frames, 1)
    latents = (pos[:, None] + steps * velocity[:, None]).clamp(-1, 1)
    latents = latents + 0.01 * torch.randn_like(latents)
    return latents, prompt_id


def decode_latents_to_video(latents: torch.Tensor, image_size: int = 32) -> torch.Tensor:
    """把二维位置 latent 解码成移动亮块视频，返回 [T, 1, H, W]。"""
    latents = latents.detach().cpu().clamp(-1, 1)
    frames = torch.full((latents.shape[0], 1, image_size, image_size), -1.0)
    block = max(3, image_size // 8)
    xy = ((latents + 1) * 0.5 * (image_size - block - 1)).long()
    for t, (x, y) in enumerate(xy.tolist()):
        frames[t, :, y:y + block, x:x + block] = 1.0
    return frames


def save_video_strip(video: torch.Tensor, path: str) -> None:
    """保存 PGM 帧条带，避免依赖 torchvision/imageio。"""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frames = ((video.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    strip = frames[:, 0].permute(1, 0, 2).reshape(frames.shape[-2], frames.shape[0] * frames.shape[-1])
    with path_obj.open("wb") as f:
        f.write(f"P5\n{strip.shape[1]} {strip.shape[0]}\n255\n".encode("ascii"))
        f.write(strip.numpy().tobytes())
