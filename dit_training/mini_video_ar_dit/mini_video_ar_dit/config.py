from dataclasses import dataclass


@dataclass
class DataConfig:
    image_size: int = 16
    patch_size: int = 8
    channels: int = 1
    latent_dim: int = 8
    text_len: int = 4
    min_frames: int = 3
    max_frames: int = 8

    @property
    def tokens_per_frame(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class ModelConfig:
    latent_dim: int = 8
    text_dim: int = 8
    hidden_size: int = 64
    depth: int = 3
    num_heads: int = 4
    max_tokens: int = 128
    attention: str = "dense"  # dense 或 sparse
    sparse_window: int = 16
