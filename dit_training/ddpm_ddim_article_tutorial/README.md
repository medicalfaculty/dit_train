# DDPM 与 DDIM 精简代码导读

写在最前面：此文所要介绍的两个模型是AIGC领域大火的Diffusion模型原版及其变种，其中DDPM是开山之作，DDIM是基于DDPM的。DDPM原始论文的原理推导非常复杂，对于刚入门的新手非常不友好。本文将对原理进行一个简单梳理，过程可能不严谨，适用于有一定概率论和深度学习基础的新手小白，并结合pytorch代码实现，若有错误希望能够得到指正。

本文仅作学习使用，不作商业用途，若有版权问题请联系笔者，代码是在他人基础上进行改的，感谢开源社区提供的pytorch代码实现。笔者本人学习中参考的主要资料为：

- 大白话DDPM：https://www.bilibili.com/video/BV1tz4y1h7q1
- DDPM和DDIM公式推导：https://www.bilibili.com/video/BV1Zh411A72y
- 迪哥讲Diffusion（含原理和代码）：https://www.bilibili.com/video/BV1pD4y1179T
- 代码来源：https://github.com/w86763777/pytorch-ddpm
- 文章链接：https://zhuanlan.zhihu.com/p/666552214

这个目录不是复刻完整 CIFAR10 训练，而是把文章里的核心公式变成三个能跑的小脚本。所有图片都是程序生成的 `16x16` 方块图，只用于理解 DDPM/DDIM 的机制。

## 文件

- `diffusion_toy.py`：DDPM/DDIM 公式、极小噪声预测网络、toy 数据、PGM 图片保存。
- `01_forward_noise_demo.py`：演示 `x_0 -> x_t` 前向加噪。
- `02_train_toy_ddpm.py`：训练一个小 CNN 学习预测噪声 `epsilon`。
- `03_compare_ddpm_ddim_sampling.py`：用同一个模型对比 DDPM 和 DDIM 采样。

## 安装

```bash
cd /data/home/sheshuchen/dit_train/dit_training/ddpm_ddim_article_tutorial
python -m pip install -r requirements.txt
```

如果当前环境已有 PyTorch，可以直接运行。

## 1. 前向加噪

```bash
python 01_forward_noise_demo.py
```

输出类似：

```text
saved=outputs/forward_noise.pgm
t=000 alpha_bar=0.9999
t=010 alpha_bar=0.9879
t=030 alpha_bar=0.9078
t=060 alpha_bar=0.6869
t=099 alpha_bar=0.3636
```

代码对应文章里的公式：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

`alpha_bar_t` 越小，原图权重越低，噪声权重越高。

## 2. 训练 DDPM 噪声预测网络

```bash
python 02_train_toy_ddpm.py --steps 100
```

输出类似：

```text
device=cuda steps=100 first_loss=1.02 last_loss=0.55
saved=outputs/tiny_eps_model.pt
```

训练逻辑只有四步：

1. 生成 toy 原图 `x_0`。
2. 随机采样时间步 `t` 和噪声 `epsilon`。
3. 用前向公式得到 `x_t`。
4. 模型输入 `(x_t, t)`，用 MSE 学习预测真实噪声 `epsilon`。

这就是文章 3.3 中 `GaussianDiffusionTrainer` 的最小版本。

## 3. DDPM 与 DDIM 采样

```bash
python 03_compare_ddpm_ddim_sampling.py --ddim-steps 10
```

如果没有 checkpoint，脚本会先快速训练几十步。输出类似：

```text
DDPM steps=100 time=0.210s saved=outputs/ddpm_samples.pgm
DDIM steps=10 time=0.025s saved=outputs/ddim_samples.pgm
```

DDPM 从纯噪声开始，每次只走 `t -> t-1`，所以这里走 100 步。DDIM 使用同一个模型预测噪声，但允许跳步，例如只走 10 步，所以推理更快。

## 重点公式对应

前向加噪：

```text
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

模型训练目标：

```text
loss = MSE(epsilon_theta(x_t, t), epsilon)
```

由噪声反推原图：

```text
x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)
```

DDPM 采样：

```text
x_{t-1} = posterior_mean(x_t, predicted_x_0) + sigma_t * z
```

DDIM 采样：

```text
x_prev = sqrt(alpha_prev) * predicted_x_0
       + sqrt(1 - alpha_prev - sigma_t^2) * epsilon_theta
       + sigma_t * z
```

当 `eta=0` 时，DDIM 采样是确定性的；当 `eta>0` 时，会重新引入随机噪声。

## 读代码顺序

先看 `diffusion_toy.py` 里的 `q_sample`、`predict_x0_from_eps`、`ddpm_sample`、`ddim_sample`。再运行三个脚本，对照输出和 `outputs/*.pgm` 图片理解文章里的公式。
