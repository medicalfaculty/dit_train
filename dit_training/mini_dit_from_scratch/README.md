# 从 0 写一个 Mini DiT

这个目录是 `/DiT/` 的 mini 教学版。目标不是复现论文效果，而是把原项目拆成最少几个清楚的文件，让你知道从 scratch 写 DiT 时每一块该放在哪里、负责什么。

原始 `/DiT/` 大致包含：

- `models.py`：DiT 模型，包括 patch embedding、timestep embedding、label embedding、Transformer block、final layer。
- `diffusion/`：DDPM 训练 loss 和反向采样。
- `train.py`：数据、模型、扩散过程、优化器、checkpoint。
- `sample.py`：加载 checkpoint，从高斯噪声逐步采样。

本目录对应成：

- `mini_dit/model.py`：MiniDiT 主体，负责 patchify、unpatchify 和整体 forward。
- `mini_dit/embeddings.py`：时间步、类别、二维位置编码。
- `mini_dit/blocks.py`：Self-Attention、MLP、DiTBlock、FinalLayer。
- `mini_dit/diffusion.py`：前向加噪、噪声预测 loss、DDPM 反向采样。
- `mini_dit/data.py`：合成两类小图，避免下载数据集。
- `mini_dit/image_io.py`：保存 PGM 图片，避免依赖 torchvision。
- `train.py`：训练入口。
- `sample.py`：采样入口。

## 安装

```bash
cd /data/home/sheshuchen/dit_train/dit_training/mini_dit_from_scratch
python -m pip install -r requirements.txt
```

## 跑通训练

```bash
python train.py --steps 100 --batch-size 32
```

你会看到类似：

```text
step=0001 loss=1.42
step=0020 loss=0.98
...
saved=outputs/mini_dit.pt
```

训练过程对应 DDPM 最常用的噪声预测目标：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
loss = MSE(model(x_t, t, y), noise)
```

这里的 `x_0` 不是真实图片数据集，而是 `mini_dit/data.py` 生成的方块和十字。类别 `y=0` 表示方块，`y=1` 表示十字。

## 跑通采样

```bash
python sample.py --ckpt outputs/mini_dit.pt --num-samples 16
```

输出：

```text
saved=outputs/samples.pgm
前几张类别标签: [0, 1, 0, 1, 0, 1, 0, 1]
```

`outputs/samples.pgm` 是灰度图片网格，可以用常见图片查看器打开。训练步数很少时，图片可能不清晰，这是正常的；这个教程关注代码结构和数据流。

## 从 0 写 DiT 的顺序

1. 先写 `patchify/unpatchify`：把 `[B,C,H,W]` 图片变成 `[B,N,D]` token，再还原回图片。
2. 写 `TimestepEmbedder`：让模型知道当前去噪处在哪个时间步。
3. 写 `LabelEmbedder`：让模型知道要生成哪一类图。
4. 写 `SelfAttention + MLP`：这是 Transformer 的基本计算。
5. 写 `DiTBlock`：用时间和类别条件调制 LayerNorm，也就是 adaLN。
6. 写 `Diffusion.training_loss`：随机采样 `t`，加噪，让模型预测噪声。
7. 写 `Diffusion.p_sample_loop`：从纯噪声开始，循环去噪得到样本。
8. 最后写 `train.py` 和 `sample.py`：把模块串起来。

## 和原始 DiT 的主要差异

- 原始 DiT 在 VAE latent 空间训练；这里直接在 `16x16` 灰度图上训练。
- 原始 DiT 可用 ImageNet 类别和 classifier-free guidance；这里只有两类 toy 图。
- 原始项目有 DDP、EMA、日志、真实数据加载；这里保留最核心训练闭环。
- 原始模型依赖 `timm` 的 Attention/PatchEmbed；这里手写，方便读懂。

理解完这个目录后，再回头看 `/DiT/models.py`、`/DiT/diffusion/gaussian_diffusion.py`、`/DiT/train.py` 会容易很多。
