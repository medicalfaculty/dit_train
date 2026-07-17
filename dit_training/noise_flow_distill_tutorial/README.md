# 预测噪声、Flow Matching 与蒸馏：Mini 教学项目

这个目录用 2D toy 数据说明三件事：

1. **预测噪声**：DDPM 常见训练目标，模型输入 `x_t,t`，预测加入的噪声 `epsilon`。
2. **Flow Matching**：模型输入 `x_t,t`，预测从噪声分布流向数据分布的速度 `dx/dt`。
3. **蒸馏**：用多步 teacher 生成目标，让 student 学会一步生成，减少采样步数。

代码只依赖 PyTorch，输出是 SVG 散点图，不需要 matplotlib。

## 参考动向

- Flow Matching 原始框架提出了“回归固定概率路径的向量场”的训练方式：https://arxiv.org/abs/2210.02747
- Meta/FAIR 的 Flow Matching Guide 总结了 FM 的数学基础、路径设计和扩展，并提供 PyTorch 代码库：https://arxiv.org/abs/2412.06264
- 对应 GitHub 实现可作为工业化代码组织参考：https://github.com/facebookresearch/flow_matching
- Diffusion 与 Flow Matching 的关系常被讨论，一个实用结论是：很多差异来自网络参数化和采样路径，而不是二者完全割裂：https://diffusionflow.github.io/
- Progressive Distillation 的核心目标是把多步 deterministic sampler 蒸馏成更少步数：https://arxiv.org/abs/2202.00512
- Consistency Models / LCM 系列把生成过程压到一步或少步，是扩散蒸馏的重要路线：https://arxiv.org/abs/2303.01469
- 2025 年以后 Flow Matching 的研究热点继续扩展到 guidance、离散数据、Riemannian/生物分子等方向，可参考综述型列表：https://github.com/dongzhuoyao/awesome-flow-matching

## 跑通

```bash
cd /data/home/sheshuchen/dit_train/dit_training/noise_flow_distill_tutorial
python noise_prediction.py --steps 200
python flow_matching.py --steps 200
python distill_flow.py --steps 200 --teacher-steps 200
python sample_compare.py
```

生成文件：

- `outputs/noise_samples.svg`
- `outputs/flow_samples.svg`
- `outputs/distilled_samples.svg`
- `outputs/compare_*.svg`

## 预测噪声是什么

DDPM 前向过程：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

训练时我们知道真实噪声 `epsilon`，所以可以训练：

```text
epsilon_theta(x_t, t) ≈ epsilon
```

采样时从纯噪声开始，反复预测噪声，再把 `x_t` 往更干净的 `x_{t-1}` 推。它的特点是训练目标稳定，但采样通常需要很多步。

对应代码：

```text
noise_prediction.py
```

## Flow Matching 是什么

本教程使用最简单的 rectified flow 路径：

```text
x_t = (1 - t) * x_0 + t * x_1
```

其中：

- `x_0` 来自高斯噪声。
- `x_1` 来自真实数据。
- 目标速度是 `v = x_1 - x_0`。

训练：

```text
v_theta(x_t, t) ≈ x_1 - x_0
```

采样时从噪声出发，用 Euler ODE 积分：

```text
x = x + v_theta(x, t) * dt
```

它和噪声预测最大的教学差别是：**噪声预测学习“这个 noisy sample 里混了多少噪声”，Flow Matching 学“这个点下一瞬间该往哪里走”。**

对应代码：

```text
flow_matching.py
```

## 蒸馏是什么

蒸馏的目的不是换数据分布，而是减少采样计算。

本教程用 flow teacher 做多步采样：

```text
z -> teacher Euler 20 steps -> x_teacher
```

student 学一步：

```text
student(z, t=0) ≈ x_teacher
```

这样 student 采样只需要一次前向。真实项目里的蒸馏会更复杂，比如 progressive distillation、consistency distillation、score distillation，但核心都是：**让小步数 student 模仿大步数 teacher 的轨迹或终点。**

对应代码：

```text
distill_flow.py
```

## 一句话对比

- 噪声预测：预测 `epsilon`，通过反复去噪采样。
- Flow Matching：预测速度 `v=dx/dt`，通过 ODE 积分采样。
- 蒸馏：把多步 teacher 的采样能力压缩到少步或一步 student。

## 学完后怎么迁移到 DiT

把 `TimeMLP` 换成 DiT backbone：

```text
2D x/t -> patch tokens + timestep embedding + text conditioning -> transformer -> target
```

如果是 DDPM/EDM 类训练，target 可以是 `epsilon`、`x0` 或 `v`。如果是 rectified flow / flow matching，target 通常是速度场。蒸馏时保留 teacher 的采样器，训练 student 去拟合 teacher 的少步轨迹或最终输出。
