import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """支持训练时 causal attention，也支持推理时追加 KV cache。"""
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, past_kv=None):
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qi = torch.arange(n, device=x.device)[:, None] + past_len
        ki = torch.arange(past_len + n, device=x.device)[None]
        causal = ki <= qi
        score = score.masked_fill(~causal[None, None], -1e4)
        out = F.softmax(score, dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor, past_kv=None):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class StreamingVGM(nn.Module):
    """Mini streaming VGM：prompt 作为前缀，逐帧自回归预测下一个 video latent。"""
    def __init__(self, latent_dim: int = 2, prompts: int = 4, hidden: int = 64,
                 depth: int = 2, heads: int = 4, max_tokens: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.prompt = nn.Embedding(prompts, hidden)
        self.bos = nn.Parameter(torch.zeros(1, 1, hidden))
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        self.blocks = nn.ModuleList([Block(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, latents: torch.Tensor, prompt_id: torch.Tensor) -> torch.Tensor:
        b, t, _ = latents.shape
        prefix = self.prompt(prompt_id)[:, None]
        prev = torch.cat([torch.zeros_like(latents[:, :1]), latents[:, :-1]], dim=1)
        x = torch.cat([prefix, self.in_proj(prev) + self.bos], dim=1)
        x = x + self.pos[:, : t + 1]
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.norm(x[:, 1:]))

    def loss(self, latents: torch.Tensor, prompt_id: torch.Tensor) -> torch.Tensor:
        pred = self(latents, prompt_id)
        return F.mse_loss(pred, latents)

    @torch.no_grad()
    def prefill_prompt(self, prompt_id: torch.Tensor):
        x = self.prompt(prompt_id)[:, None] + self.pos[:, :1]
        cache = []
        for block in self.blocks:
            x, kv = block(x, None)
            cache.append(kv)
        return cache

    @torch.no_grad()
    def stream_step(self, prev_latent: torch.Tensor, cache: list, index: int) -> tuple[torch.Tensor, list]:
        """流式生成一步：只输入上一帧 latent，复用过去 KV，不重新算全部历史。"""
        x = self.in_proj(prev_latent[:, None]) + self.bos + self.pos[:, index:index + 1]
        new_cache = []
        for block, kv in zip(self.blocks, cache):
            x, new_kv = block(x, kv)
            new_cache.append(new_kv)
        next_latent = self.head(self.norm(x[:, -1]))
        return next_latent, new_cache

    @torch.no_grad()
    def generate_stream(self, prompt_id: torch.Tensor, frames: int) -> torch.Tensor:
        self.eval()
        cache = self.prefill_prompt(prompt_id)
        prev = torch.zeros(prompt_id.shape[0], self.latent_dim, device=prompt_id.device)
        outs = []
        for i in range(frames):
            pred, cache = self.stream_step(prev, cache, index=i + 1)
            outs.append(pred)
            prev = pred
        return torch.stack(outs, dim=1)
