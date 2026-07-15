import math
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    image_size: int = 16
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 64
    depth: int = 2
    heads: int = 4
    timesteps: int = 1000

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freq[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.cond(c).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.mlp(h)


class TinyDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Linear(cfg.patch_dim, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.dim))
        self.time_mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.class_emb = nn.Embedding(cfg.num_classes, cfg.dim)
        self.blocks = nn.ModuleList([DiTBlock(cfg.dim, cfg.heads) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.patch_dim)
        nn.init.normal_(self.pos, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, -1, c * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, n, d = x.shape
        side = int(math.sqrt(n))
        x = x.reshape(b, side, side, self.cfg.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, self.cfg.in_channels, side * p, side * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.patch(self.patchify(x)) + self.pos
        c = self.time_mlp(sinusoidal_embedding(t, self.cfg.dim)) + self.class_emb(y)
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.out(self.final_norm(h)))


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise


def synthetic_batch(batch_size: int, cfg: DiTConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(batch_size, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
    labels = torch.randint(0, cfg.num_classes, (batch_size,), device=device)
    return images, labels


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, schedule: DiffusionSchedule,
               cfg: DiTConfig, batch_size: int, device: torch.device) -> float:
    x0, y = synthetic_batch(batch_size, cfg, device)
    t = torch.randint(0, cfg.timesteps, (batch_size,), device=device)
    noise = torch.randn_like(x0)
    xt = schedule.add_noise(x0, t, noise)
    pred = model(xt, t, y)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
