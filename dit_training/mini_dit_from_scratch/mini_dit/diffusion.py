from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """从长度为 T 的系数表里按 batch 的 t 取值，并 reshape 成可广播形状。"""
    out = v.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_1: float = 1e-4
    beta_T: float = 0.02


class Diffusion:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.betas = torch.linspace(cfg.beta_1, cfg.beta_T, cfg.timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1.0)

        self.posterior_var = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_log_var = torch.log(torch.clamp(self.posterior_var, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_bar, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_bar, t, x0.shape) * noise
        )

    def training_loss(self, model: nn.Module, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.cfg.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t, y)
        return F.mse_loss(pred_noise, noise)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple[int, int, int, int],
                      y: torch.Tensor, device: torch.device) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        for step in reversed(range(self.cfg.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = model(x_t, t, y)
            x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x0
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            noise = torch.randn_like(x_t) if step > 0 else 0.0
            x_t = mean + torch.exp(0.5 * extract(self.posterior_log_var, t, x_t.shape)) * noise
        return x_t.clamp(-1, 1)
