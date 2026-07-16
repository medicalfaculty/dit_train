from dataclasses import dataclass

import torch
from torch import nn

from .blocks import DiTBlock, FinalLayer
from .embeddings import LabelEmbedder, TimestepEmbedder, fixed_2d_sincos_pos_embed


@dataclass
class MiniDiTConfig:
    image_size: int = 16
    patch_size: int = 4
    in_channels: int = 1
    hidden_size: int = 64
    depth: int = 2
    num_heads: int = 4
    num_classes: int = 2
    class_dropout_prob: float = 0.1


class MiniDiT(nn.Module):
    """从 0 写的迷你 DiT：输入噪声图 x_t、时间 t、类别 y，输出预测噪声。"""
    def __init__(self, cfg: MiniDiTConfig = MiniDiTConfig()):
        super().__init__()
        self.cfg = cfg
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size

        self.x_embedder = nn.Linear(patch_dim, cfg.hidden_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.y_embedder = LabelEmbedder(cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob)
        self.blocks = nn.ModuleList([DiTBlock(cfg.hidden_size, cfg.num_heads) for _ in range(cfg.depth)])
        self.final = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)
        self.register_buffer(
            "pos_embed",
            fixed_2d_sincos_pos_embed(cfg.image_size // cfg.patch_size, cfg.hidden_size, torch.device("cpu")),
            persistent=False,
        )
        self.initialize()

    def initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, -1, c * p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, n, d = x.shape
        side = int(n ** 0.5)
        x = x.reshape(b, side, side, self.cfg.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, self.cfg.in_channels, side * p, side * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.x_embedder(self.patchify(x)) + self.pos_embed.to(x.device)
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.final(h, c))
