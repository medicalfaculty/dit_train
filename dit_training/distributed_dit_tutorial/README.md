# DiT 分布式训练最小教程

这个目录用最小代码讲清楚五件事：

1. `01_minimal_dit_pretrain.py`：一个能跑的最简 DiT 扩散预训练循环。
2. `02_fsdp_dit_pretrain.py`：把同一个 DiT 训练循环接到 PyTorch FSDP。
3. `03_how_torch_fsdp_works.py`：用一个 Linear 层手写演示 FSDP 背后的核心动作。
4. `04_context_parallel_attention.py`：演示 CP 并行下 attention 如何保持和完整 attention 等价。
5. `05_ring_attention.py`：演示 Ring Attention 如何边传 K/V 块边做在线 softmax。

数据不是 ImageNet，而是随机生成的小图片。这样做的目的不是训练出好模型，而是把 DiT、扩散训练目标、分布式启动、FSDP 参数分片这些概念压到可以直接运行和调试的大小。

## 环境

当前机器的默认 Python 没有安装 PyTorch。先进入本目录，然后安装依赖：

```bash
cd /data/home/sheshuchen/dit_training/distributed_dit_tutorial
python -m pip install -r requirements.txt
```

如果你有 CUDA，建议按 PyTorch 官网给出的 CUDA 版本命令安装 `torch`。CPU 也可以跑第 1 和第 3 个脚本；真正的 FSDP 训练建议使用多张 CUDA GPU。

## 代码结构

- `dit_common.py`：共享的最小 DiT、扩散噪声调度、随机 batch、单步训练函数。
- `01_minimal_dit_pretrain.py`：单进程预训练。
- `02_fsdp_dit_pretrain.py`：FSDP 预训练。
- `03_how_torch_fsdp_works.py`：解释 FSDP 通信流程的教学脚本。
- `04_context_parallel_attention.py`：解释 Context Parallel 的教学脚本。
- `05_ring_attention.py`：解释 Ring Attention 的教学脚本。
- `CP_AND_RING_ATTENTION.md`：CP 与 Ring Attention 的中文细节文档。

## 1. 最简 DiT 预训练

启动：

```bash
python 01_minimal_dit_pretrain.py --steps 20 --batch-size 8
```

你会看到类似输出：

```text
device=cpu parameters=138,480
step=001 loss=1.28
step=005 loss=1.11
step=010 loss=1.05
...
```

每一步做的事情：

1. 随机生成一批 `16x16` RGB 图片 `x0` 和类别标签 `y`。
2. 随机采样扩散时间步 `t`。
3. 按 DDPM 公式得到加噪图片 `xt = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise`。
4. DiT 输入 `xt, t, y`，预测噪声 `pred_noise`。
5. 用 `MSE(pred_noise, real_noise)` 作为 loss。

这个 loss 不一定单调下降，因为数据每步都是随机噪声；正常结果是脚本能稳定前向、反向、更新参数，并打印有限数值的 loss。

## 2. FSDP 版本

单进程 smoke test：

```bash
python 02_fsdp_dit_pretrain.py --steps 10 --batch-size 4
```

多 GPU FSDP 启动示例：

```bash
torchrun --standalone --nproc_per_node=2 02_fsdp_dit_pretrain.py --steps 20 --batch-size 4
```

这里 `--batch-size` 是每个 rank 的 batch size。上面命令的全局 batch size 是 `2 * 4 = 8`。

脚本逻辑：

1. `torchrun` 会给每个进程注入 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。
2. 脚本用这些环境变量初始化 `torch.distributed`。
3. 如果检测到多进程 CUDA 环境，就用 `FullyShardedDataParallel` 包住 DiT。
4. 每个 rank 各自生成一份随机数据，前向和反向由 FSDP 接管参数通信。
5. 打印 loss 前，用 `dist.all_reduce(..., AVG)` 求所有 rank 的平均 loss。

FSDP 的关键收益是显存。普通 DDP 每张卡都有完整参数、梯度、优化器状态；FSDP 会把它们切成分片，每个 rank 只长期保存自己的一片，需要计算某层时再临时聚合完整参数。

如果你在 CPU 多进程下运行这个脚本，它会退化成非 FSDP smoke test，因为 PyTorch 的 FSDP 实战路径通常依赖 CUDA/NCCL。

## 3. Torch FSDP 大致怎么实现

启动单进程：

```bash
python 03_how_torch_fsdp_works.py --steps 5
```

启动多进程 CPU 演示：

```bash
torchrun --standalone --nproc_per_node=2 03_how_torch_fsdp_works.py --steps 5
```

你会看到类似：

```text
world_size=2 device=cpu
这个脚本只演示核心思想：参数按 rank 分片，前向前 all_gather 成完整参数，反向后只保留本 rank 的梯度分片。
step=001 loss=0.93 local_shard_shape=(4, 8)
...
```

这个脚本不是 PyTorch FSDP 源码复刻，而是把 FSDP 最核心的几步摊开：

1. 初始化一个完整 Linear 权重。
2. 每个 rank 只保留一部分行，也就是参数分片。
3. 前向计算前，所有 rank 通过 `all_gather` 临时拿到完整权重。
4. 完成前向、loss、反向后，每个 rank 只更新自己那一片参数。
5. 下一步训练再次临时聚合完整权重。

真实 PyTorch FSDP 还会做更多事情：

- 把多个参数 flatten 成大块，减少通信调用次数。
- 根据 module 包裹边界决定何时 all-gather、何时释放完整参数。
- 反向时用 reduce-scatter / reduce 通信把梯度归约并切回分片。
- 管理 mixed precision、state dict、checkpoint、CPU offload、prefetch 等工程细节。

## 推荐学习顺序

先跑 `01_minimal_dit_pretrain.py`，确认你理解 DiT 训练目标。然后跑 `03_how_torch_fsdp_works.py`，理解“参数常驻分片、计算时临时聚合”的思想。再用多 GPU 跑 `02_fsdp_dit_pretrain.py`，观察真实 FSDP 训练入口和 `torchrun` 启动方式。

继续学习长序列训练时，阅读 `CP_AND_RING_ATTENTION.md`，然后运行：

```bash
torchrun --standalone --nproc_per_node=2 04_context_parallel_attention.py
torchrun --standalone --nproc_per_node=2 05_ring_attention.py
```
