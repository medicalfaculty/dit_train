from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn


@dataclass
class ToyConfig:
    hidden: int = 64
    data_dim: int = 2
    diffusion_steps: int = 100


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_data(batch: int, device: torch.device) -> torch.Tensor:
    """8 个高斯团组成的 2D toy 数据，方便观察生成分布。"""
    k = torch.randint(0, 8, (batch,), device=device)
    angle = 2 * math.pi * k.float() / 8
    centers = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * 2.0
    return centers + 0.15 * torch.randn(batch, 2, device=device)


def time_features(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    """所有示例共用的小网络：输入 x 和连续时间 t，输出 2D 向量。"""
    def __init__(self, out_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, time_features(t)], dim=1))


def save_scatter_svg(points: torch.Tensor, path: str, title: str) -> None:
    """不用 matplotlib，直接写一个简单 SVG 散点图。"""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    pts = points.detach().cpu().clamp(-3, 3)
    width = height = 360
    scale = 52
    dots = []
    for x, y in pts.tolist():
        cx = width / 2 + x * scale
        cy = height / 2 - y * scale
        dots.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.2" fill="#2563eb" opacity="0.65"/>')
    body = "\n".join(dots)
    path_obj.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="16" y="24" font-size="16" font-family="monospace">{title}</text>\n'
        f'<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="#ddd"/>\n'
        f'<line x1="{width/2}" y1="0" x2="{width/2}" y2="{height}" stroke="#ddd"/>\n'
        f'{body}\n</svg>\n',
        encoding="utf-8",
    )
