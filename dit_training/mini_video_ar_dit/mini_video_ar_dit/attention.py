import torch
from torch import nn
import torch.nn.functional as F


def build_causal_mask(total_len: int, text_len: int, device: torch.device,
                      mode: str, sparse_window: int) -> torch.Tensor:
    """返回 [total_len, total_len]，True 表示允许 attention。"""
    q = torch.arange(total_len, device=device)[:, None]
    k = torch.arange(total_len, device=device)[None]
    causal = k <= q
    if mode == "dense":
        return causal
    if mode != "sparse":
        raise ValueError(f"unknown attention mode: {mode}")

    # sparse 教学版：文本 prefix 永远可见，latent token 只看最近 window 个历史 latent。
    text_visible = k < text_len
    recent = (q - k) <= sparse_window
    return causal & (text_visible | recent)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = allowed[None, None] & key_padding[:, None, None, :]
        score = score.masked_fill(~mask, -1e4)
        out = F.softmax(score, dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(b, n, c))
