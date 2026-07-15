# DiT 官方训练实验记录

实验日期：2026-07-15

结论：训练成功。本次不是随机 latent smoke test，而是跑通了官方 `train.py` 路径，包括 `ImageFolder`、图像 transform、Stable Diffusion VAE encode、`diffusion.training_losses`、backward、optimizer step、EMA 更新和 checkpoint 保存。

## 环境

使用解释器：

```bash
/data/home/sheshuchen/.conda/envs/aiinfra-interview/bin/python
```

关键依赖：

```text
torch 2.13.0+cu132
torchvision 0.28.0+cu132
timm 1.0.28
diffusers 0.39.0
accelerate 1.14.0
```

GPU：

```text
NVIDIA RTX A6000
```

本机默认 `base` Python 没有 PyTorch，所以不要直接用 `python train.py`。

## 临时数据集

为了验证训练链路，创建了临时 ImageFolder 数据集：

```text
/tmp/dit_tiny_imagefolder/
  class0/
    000.png
    ...
    009.png
```

共 10 张临时 RGB 图片。该数据集只用于 smoke test，不用于评估生成质量。

## 第一次失败

第一次运行官方 `train.py` 时失败在 VAE 下载阶段：

```text
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

原因是 `huggingface_hub/httpx` 检测到当前环境使用 SOCKS proxy，但 Python 环境缺少 `socksio`。

修复：

```bash
/data/home/sheshuchen/.conda/envs/aiinfra-interview/bin/python -m pip install socksio
```

修复后，`AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")` 可以正常下载/加载。

## 成功训练命令

```bash
cd /data/home/sheshuchen/dit_train/DiT

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
/data/home/sheshuchen/.conda/envs/aiinfra-interview/bin/python \
  -m torch.distributed.run --nproc_per_node=1 train.py \
  --data-path /tmp/dit_tiny_imagefolder \
  --results-dir /tmp/dit_train_results \
  --model DiT-S/2 \
  --image-size 256 \
  --global-batch-size 2 \
  --epochs 20 \
  --num-workers 0 \
  --log-every 10 \
  --ckpt-every 100
```

为什么是 20 个 epoch：

- 临时数据集 10 张图。
- batch size 为 2。
- 每个 epoch 5 step。
- 20 个 epoch 正好 100 step。

## 成功日志

关键日志：

```text
Starting rank=0, seed=0, world_size=1.
Experiment directory created at /tmp/dit_train_results/001-DiT-S-2
DiT Parameters: 32,963,360
Dataset contains 10 images (/tmp/dit_tiny_imagefolder)
Training for 20 epochs...
```

loss 记录：

```text
step=0000010 Train Loss: 0.9889, Train Steps/Sec: 4.53
step=0000020 Train Loss: 0.9193, Train Steps/Sec: 6.72
step=0000030 Train Loss: 0.7645, Train Steps/Sec: 6.68
step=0000040 Train Loss: 0.5972, Train Steps/Sec: 6.66
step=0000050 Train Loss: 0.4623, Train Steps/Sec: 6.66
step=0000060 Train Loss: 0.2354, Train Steps/Sec: 6.64
step=0000070 Train Loss: 0.1335, Train Steps/Sec: 6.58
step=0000080 Train Loss: 0.1498, Train Steps/Sec: 6.72
step=0000090 Train Loss: 0.2945, Train Steps/Sec: 6.62
step=0000100 Train Loss: 0.3163, Train Steps/Sec: 6.71
```

checkpoint：

```text
Saved checkpoint to /tmp/dit_train_results/001-DiT-S-2/checkpoints/0000100.pt
Done!
```

## 训练链路确认

这次成功覆盖了官方训练脚本中的关键路径：

- DDP 初始化：`dist.init_process_group("nccl")`
- 模型实例化：`DiT_models["DiT-S/2"](input_size=32, num_classes=1000)`
- VAE 加载：`AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")`
- 图像编码：`vae.encode(x).latent_dist.sample().mul_(0.18215)`
- timestep 采样：`torch.randint(0, diffusion.num_timesteps, ...)`
- loss：`diffusion.training_losses(model, x, t, {"y": y})`
- 反向传播：`loss.backward()`
- 参数更新：`opt.step()`
- EMA 更新：`update_ema(ema, model.module)`
- checkpoint 保存：`0000100.pt`

## 注意事项

- 这是最小训练链路验证，不代表模型收敛质量。
- 数据集太小，loss 下降主要说明模型能在临时数据上过拟合。
- 本机多卡 DDP 之前直接使用 NCCL 默认路径会卡住，建议保留：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
```

- `train.py` 原生没有 `--max-steps` 参数；要精确跑 100 step，可以控制 `数据量 / batch size * epochs = 100`，或者给训练循环加 `max_steps`。
