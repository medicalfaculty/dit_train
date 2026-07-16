import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """把整数时间步 t 变成 sin/cos 向量，形状 [B, dim]。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.hidden_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # 最后一个 id 作为“空类别”，用于 classifier-free guidance。
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, y: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            drop = torch.rand(y.shape, device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)


def fixed_2d_sincos_pos_embed(grid_size: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    """生成固定二维位置编码，返回 [1, grid_size*grid_size, hidden_size]。"""
    assert hidden_size % 4 == 0, "hidden_size 需要能被 4 整除"
    y, x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    pos = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1).float()
    quarter = hidden_size // 4
    omega = torch.exp(-math.log(10000) * torch.arange(quarter, device=device).float() / quarter)
    out = []
    for axis in [pos[:, 0], pos[:, 1]]:
        v = axis[:, None] * omega[None]
        out.extend([torch.sin(v), torch.cos(v)])
    return torch.cat(out, dim=1).unsqueeze(0)
