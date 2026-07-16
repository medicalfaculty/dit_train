import math
from pathlib import Path

import torch


def save_pgm_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """不用 torchvision，直接保存灰度 PGM 图片网格。"""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    images = ((images.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    b, _, h, w = images.shape
    rows = math.ceil(b / nrow)
    grid = torch.zeros(rows * h, nrow * w, dtype=torch.uint8)
    for i in range(b):
        r, c = divmod(i, nrow)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i, 0]
    with path_obj.open("wb") as f:
        f.write(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode("ascii"))
        f.write(grid.numpy().tobytes())
