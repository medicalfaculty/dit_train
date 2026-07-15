import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    image_size: int = 16
    channels: int = 1
    timesteps: int = 100
    beta_1: float = 1e-4
    beta_T: float = 0.02


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = v.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


class Schedule:
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
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    return (
        extract(schedule.sqrt_alphas_bar, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alphas_bar, t, x0.shape) * noise
    )


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
                        schedule: Schedule) -> torch.Tensor:
    return (
        extract(schedule.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
        - extract(schedule.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
    )


def ddpm_posterior_mean_var(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
                            schedule: Schedule) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = predict_x0_from_eps(x_t, t, eps, schedule).clamp(-1, 1)
    mean = (
        extract(schedule.posterior_mean_coef1, t, x_t.shape) * x0
        + extract(schedule.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    log_var = extract(schedule.posterior_log_var, t, x_t.shape)
    return mean, log_var


def ddpm_sample(model: nn.Module, schedule: Schedule, shape: tuple[int, int, int, int],
                device: torch.device) -> torch.Tensor:
    x_t = torch.randn(shape, device=device)
    for time_step in reversed(range(schedule.cfg.timesteps)):
        t = torch.full((shape[0],), time_step, device=device, dtype=torch.long)
        eps = model(x_t, t)
        mean, log_var = ddpm_posterior_mean_var(x_t, t, eps, schedule)
        noise = torch.randn_like(x_t) if time_step > 0 else 0.0
        x_t = mean + torch.exp(0.5 * log_var) * noise
    return x_t.clamp(-1, 1)


def ddim_sample(model: nn.Module, schedule: Schedule, shape: tuple[int, int, int, int],
                device: torch.device, sample_steps: int = 10, eta: float = 0.0) -> torch.Tensor:
    x_t = torch.randn(shape, device=device)
    times = torch.linspace(schedule.cfg.timesteps - 1, 0, sample_steps, device=device).long()
    prev_times = torch.cat([times[1:], torch.zeros(1, device=device, dtype=torch.long)])

    for cur, prev in zip(times, prev_times):
        t = torch.full((shape[0],), int(cur), device=device, dtype=torch.long)
        prev_t = torch.full((shape[0],), int(prev), device=device, dtype=torch.long)
        eps = model(x_t, t)
        x0 = predict_x0_from_eps(x_t, t, eps, schedule).clamp(-1, 1)

        alpha_t = extract(schedule.alphas_bar, t, x_t.shape)
        alpha_prev = extract(schedule.alphas_bar, prev_t, x_t.shape)
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev).clamp(min=0)
        )
        direction = torch.sqrt((1 - alpha_prev - sigma ** 2).clamp(min=0)) * eps
        noise = torch.randn_like(x_t) if int(prev) > 0 else 0.0
        x_t = torch.sqrt(alpha_prev) * x0 + direction + sigma * noise
    return x_t.clamp(-1, 1)


class TinyEpsNet(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.net = nn.Sequential(
            nn.Conv2d(1 + 32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time((t.float() / self.timesteps).view(-1, 1))
        temb = temb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.net(torch.cat([x, temb], dim=1))


def make_toy_images(batch: int, image_size: int, device: torch.device) -> torch.Tensor:
    x = torch.full((batch, 1, image_size, image_size), -1.0, device=device)
    for i in range(batch):
        size = int(torch.randint(4, 9, (1,), device=device))
        top = int(torch.randint(1, image_size - size - 1, (1,), device=device))
        left = int(torch.randint(1, image_size - size - 1, (1,), device=device))
        x[i, :, top:top + size, left:left + size] = 1.0
    return x


def train_steps(model: nn.Module, schedule: Schedule, steps: int, batch_size: int,
                lr: float, device: torch.device) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x0 = make_toy_images(batch_size, schedule.cfg.image_size, device)
        t = torch.randint(0, schedule.cfg.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise, schedule)
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def save_pgm_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = ((images.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    b, _, h, w = images.shape
    rows = math.ceil(b / nrow)
    grid = torch.zeros(rows * h, nrow * w, dtype=torch.uint8)
    for idx in range(b):
        r, c = divmod(idx, nrow)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[idx, 0]
    with path.open("wb") as f:
        f.write(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode("ascii"))
        f.write(grid.numpy().tobytes())
