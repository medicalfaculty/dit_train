import torch
from torch import nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, build_causal_mask
from .config import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg.hidden_size, cfg.num_heads)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
        )

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), allowed, key_padding)
        return x + self.mlp(self.norm2(x))


class ARVideoDiT(nn.Module):
    """自回归视频 DiT 教学版：文本 token 作为 prefix，逐 token 预测下一个 VAE latent。"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.hidden_size)
        self.bos = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_tokens, cfg.hidden_size))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.latent_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, latents: torch.Tensor, text: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        b, latent_len, _ = latents.shape
        text_h = self.text_proj(text)
        # teacher forcing：输入 BOS + 前 n-1 个 latent，输出位置对齐预测第 1..n 个 latent。
        prev_latents = torch.cat([torch.zeros_like(latents[:, :1]), latents[:, :-1]], dim=1)
        latent_h = self.latent_proj(prev_latents) + self.bos
        x = torch.cat([text_h, latent_h], dim=1)
        if x.shape[1] > self.cfg.max_tokens:
            raise ValueError("序列超过 max_tokens，请调大 ModelConfig.max_tokens")
        x = x + self.pos[:, : x.shape[1]]

        text_len = text.shape[1]
        text_mask = torch.ones(b, text_len, dtype=torch.bool, device=latents.device)
        key_padding = torch.cat([text_mask, latent_mask], dim=1)
        allowed = build_causal_mask(x.shape[1], text_len, latents.device, self.cfg.attention, self.cfg.sparse_window)
        for block in self.blocks:
            x = block(x, allowed, key_padding)
        pred = self.head(self.norm(x[:, text_len:]))
        return pred

    def loss(self, latents: torch.Tensor, text: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        pred = self(latents, text, latent_mask)
        per_token = F.mse_loss(pred, latents, reduction="none").mean(dim=-1)
        return (per_token * latent_mask.float()).sum() / latent_mask.float().sum().clamp_min(1)

    @torch.no_grad()
    def generate(self, text: torch.Tensor, steps: int) -> torch.Tensor:
        self.eval()
        b = text.shape[0]
        latents = torch.zeros(b, 0, self.cfg.latent_dim, device=text.device)
        for _ in range(steps):
            # 当前已有 token 全部有效，预测最后一个位置的下一个 latent。
            padded = torch.cat([latents, torch.zeros(b, 1, self.cfg.latent_dim, device=text.device)], dim=1)
            mask = torch.ones(b, padded.shape[1], dtype=torch.bool, device=text.device)
            pred = self(padded, text, mask)[:, -1:]
            latents = torch.cat([latents, pred], dim=1)
        return latents
