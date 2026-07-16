import math

import torch
import torch.nn.functional as F

from .config import DataConfig


class ToyT5Encoder:
    """T5 的教学替身：输入 prompt，输出固定长度文本 token。真实工程中这里换成 T5。"""
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def encode(self, prompt: str) -> torch.Tensor:
        ids = [ord(c) % 97 for c in prompt[: self.cfg.text_len]]
        ids += [0] * (self.cfg.text_len - len(ids))
        x = torch.tensor(ids, dtype=torch.float32)[:, None]
        freqs = torch.arange(self.cfg.latent_dim, dtype=torch.float32)[None]
        return torch.sin(x / 10.0 + freqs / 7.0)


class ToyVideoVAE:
    """VAE 的教学替身：把视频帧切 patch 均值，再用固定投影变成 latent token。"""
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        base = torch.linspace(-1, 1, cfg.latent_dim)
        self.proj = torch.stack([torch.cos(base), torch.sin(base)], dim=0)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        # video: [T, C, H, W] -> latents: [T * num_patches, latent_dim]
        p = self.cfg.patch_size
        t, c, h, w = video.shape
        patches = video.reshape(t, c, h // p, p, w // p, p).mean(dim=(1, 3, 5))
        patch_values = patches.reshape(t * self.cfg.tokens_per_frame, 1)
        basis = torch.linspace(0.5, 1.5, self.cfg.latent_dim).view(1, -1)
        return torch.tanh(patch_values * basis)

    def decode(self, latents: torch.Tensor, frames: int) -> torch.Tensor:
        # latents: [T * num_patches, latent_dim] -> video: [T, 1, H, W]
        p = self.cfg.patch_size
        side = self.cfg.image_size // p
        values = latents[:, :1].reshape(frames, 1, side, side).clamp(-1, 1)
        return values.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)


def make_toy_video(prompt: str, frames: int, cfg: DataConfig) -> torch.Tensor:
    """合成一个移动亮块视频，prompt 只影响移动方向，方便端到端跑通。"""
    video = torch.full((frames, cfg.channels, cfg.image_size, cfg.image_size), -1.0)
    horizontal = "right" in prompt or "horizontal" in prompt
    for i in range(frames):
        size = cfg.image_size // 4
        if horizontal:
            top = cfg.image_size // 2 - size // 2
            left = 1 + i * max(1, (cfg.image_size - size - 2) // max(1, frames - 1))
        else:
            top = 1 + i * max(1, (cfg.image_size - size - 2) // max(1, frames - 1))
            left = cfg.image_size // 2 - size // 2
        video[i, :, top:top + size, left:left + size] = 1.0
    return video


def save_video_strip(video: torch.Tensor, path: str) -> None:
    """保存 PGM 帧条带，避免引入 imageio/torchvision。"""
    import pathlib

    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frames = ((video.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    strip = frames[:, 0].permute(1, 0, 2).reshape(frames.shape[-2], frames.shape[0] * frames.shape[-1])
    with path_obj.open("wb") as f:
        f.write(f"P5\n{strip.shape[1]} {strip.shape[0]}\n255\n".encode("ascii"))
        f.write(strip.numpy().tobytes())
