import torch


def make_shape_batch(batch_size: int, image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """生成两类合成图：类别 0 是方块，类别 1 是十字。像素范围 [-1, 1]。"""
    x = torch.full((batch_size, 1, image_size, image_size), -1.0, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)
    for i in range(batch_size):
        if int(y[i]) == 0:
            size = int(torch.randint(4, 8, (1,), device=device))
            top = int(torch.randint(2, image_size - size - 2, (1,), device=device))
            left = int(torch.randint(2, image_size - size - 2, (1,), device=device))
            x[i, :, top:top + size, left:left + size] = 1.0
        else:
            center = image_size // 2 + int(torch.randint(-2, 3, (1,), device=device))
            x[i, :, center - 1:center + 1, 3:image_size - 3] = 1.0
            x[i, :, 3:image_size - 3, center - 1:center + 1] = 1.0
    return x, y
