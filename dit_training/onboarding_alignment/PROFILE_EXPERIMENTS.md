# 显存、时间与 nsys Profiling 实验文档

这份文档给出一套可以直接执行或改造的 profiling 实验流程，用来分析 DiT、Open-Sora、FastVideo 等视频生成训练项目的显存占用、step time、通信、数据链路和 CUDA kernel 时间。

目标不是一次性跑完整大训练，而是建立可重复的小实验方法：先用短 step 找瓶颈，再用 nsys 抓稳定阶段，最后把结论写成表格。

## 1. 实验目标

每次 profiling 至少回答五个问题：

1. 单步训练时间是多少，forward/backward/optimizer/communication 各占多少。
2. 峰值显存是多少，主要来自参数、activation、optimizer state、attention workspace 还是数据 buffer。
3. GPU 是否有空洞，空洞来自 DataLoader、H2D、Python 同步、checkpoint 还是通信等待。
4. NCCL 通信是否能和 backward overlap，是否有 rank 间负载不均。
5. 长短视频、不同 token 数、dense/sparse attention、packing/offload 是否显著改变吞吐。

## 2. 实验目录建议

建议每次实验单独建目录：

```bash
mkdir -p profile_runs/$(date +%Y%m%d_%H%M%S)
```

目录中保存：

```text
config.yaml 或命令行参数
git_commit.txt
env.txt
memory_summary.txt
torch_profiler.json
nsys_rank*.nsys-rep
nsys_stats.txt
result_table.md
notes.md
```

记录 commit：

```bash
git rev-parse HEAD > profile_runs/exp/git_commit.txt
```

记录环境：

```bash
nvidia-smi > profile_runs/exp/env.txt
python - <<'PY' >> profile_runs/exp/env.txt
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## 3. Profiling 分层策略

不要一开始就用 nsys 跑完整训练，建议分三层：

### 第一层：轻量日志

在训练脚本里打印：

```text
step_time
data_time
forward_time
backward_time
optimizer_time
tokens_per_step
max_memory_allocated
max_memory_reserved
```

这一层 overhead 最低，适合长时间观察吞吐稳定性。

### 第二层：torch profiler

用 PyTorch profiler 看 PyTorch op、CUDA op、显存和调用栈，适合快速定位某类算子。

### 第三层：nsys

用 nsys 看端到端 timeline，适合确认 GPU 空洞、NCCL overlap、H2D、CPU 阻塞和多 rank 对齐问题。

## 4. 最小 PyTorch 计时代码

可以在训练 loop 中加入：

```python
import time
import torch

for step, batch in enumerate(loader):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    data_end = time.perf_counter()

    loss = model_forward_loss(batch)
    fwd_end = time.perf_counter()

    loss.backward()
    bwd_end = time.perf_counter()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    opt_end = time.perf_counter()

    torch.cuda.synchronize()
    end = time.perf_counter()

    if step % 10 == 0:
        print({
            "step": step,
            "total": end - t0,
            "data": data_end - t0,
            "forward": fwd_end - data_end,
            "backward": bwd_end - fwd_end,
            "optimizer": opt_end - bwd_end,
            "max_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        })
```

注意：`torch.cuda.synchronize()` 会影响性能，因此只在 profiling 模式使用。

## 5. PyTorch Memory Snapshot

查看当前显存摘要：

```python
print(torch.cuda.memory_summary())
```

保存 allocator snapshot：

```python
torch.cuda.memory._record_memory_history()
# run several steps
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

用法：

```bash
python -m torch.cuda._memory_viz trace_plot memory_snapshot.pickle -o memory_trace.html
```

适合分析：

- reserved 和 allocated 差距。
- 大 tensor 生命周期。
- 是否有显存碎片。
- VAE/T5/训练混链路导致的额外峰值。

## 6. PyTorch Profiler 示例

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=4, repeat=1),
    on_trace_ready=tensorboard_trace_handler("profile_runs/torch_profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if step >= 10:
            break
```

查看：

```bash
tensorboard --logdir profile_runs/torch_profiler
```

如果没有 TensorBoard，也可以打印表：

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))
```

## 7. nsys 基础命令

单进程：

```bash
nsys profile \
  -o profile_runs/exp/train \
  --trace=cuda,nvtx,osrt \
  --capture-range=none \
  python train.py --steps 20
```

多卡 torchrun：

```bash
nsys profile \
  -o profile_runs/exp/rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  --capture-range=none \
  torchrun --standalone --nproc_per_node=8 train.py --steps 20
```

只采稳定阶段：

```bash
nsys profile \
  -o profile_runs/exp/stable_rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --stop-on-range-end=true \
  torchrun --standalone --nproc_per_node=8 train.py
```

代码里在第 N 步开始和结束：

```python
if step == 10:
    torch.cuda.cudart().cudaProfilerStart()
if step == 20:
    torch.cuda.cudart().cudaProfilerStop()
    break
```

## 8. nsys stats 常用命令

```bash
nsys stats profile_runs/exp/rank_0.nsys-rep > profile_runs/exp/nsys_stats.txt
```

CUDA kernel 汇总：

```bash
nsys stats --report cuda_gpu_kern_sum profile_runs/exp/rank_0.nsys-rep
```

CUDA memcpy 汇总：

```bash
nsys stats --report cuda_gpu_mem_time_sum profile_runs/exp/rank_0.nsys-rep
```

OS runtime 汇总：

```bash
nsys stats --report osrt_sum profile_runs/exp/rank_0.nsys-rep
```

导出 SQLite：

```bash
nsys export --type sqlite --output profile_runs/exp/rank_0.sqlite profile_runs/exp/rank_0.nsys-rep
```

## 9. 必须加的 NVTX 区间

训练代码建议加：

```python
from torch.cuda import nvtx

with nvtx.range("data_load"):
    batch = next(data_iter)

with nvtx.range("h2d"):
    batch = move_to_cuda(batch)

with nvtx.range("forward"):
    loss = model_forward_loss(batch)

with nvtx.range("backward"):
    loss.backward()

with nvtx.range("optimizer"):
    optimizer.step()
```

分布式训练还建议标：

```text
fsdp_all_gather
grad_reduce_scatter
tp_all_reduce
cp_all_gather_kv
checkpoint_save
validation_sample
```

没有 NVTX 时，nsys timeline 只能看到底层 kernel 名，难以对应业务阶段。

## 10. DiT 项目实验

本地路径：

```text
/data/home/sheshuchen/dit_train/DiT
```

训练入口：

```text
DiT/train.py
```

采样入口：

```text
DiT/sample.py
```

DiT 原始训练依赖 ImageFolder 数据和 VAE，因此建议先做两类实验：

### 10.1 采样 profiling

如果已有 checkpoint：

```bash
cd /data/home/sheshuchen/dit_train/DiT
nsys profile \
  -o ../profile_runs/dit_sample \
  --trace=cuda,nvtx,osrt \
  python sample.py --ckpt pretrained_models/DiT-XL-2-256x256.pt --num-sampling-steps 20
```

观察：

- DiT forward kernel 时间。
- VAE decode 时间。
- classifier-free guidance 是否导致 batch 翻倍。
- 采样循环中每步 kernel 是否重复且稳定。

### 10.2 训练 profiling

小数据集 debug：

```bash
cd /data/home/sheshuchen/dit_train/DiT
torchrun --standalone --nproc_per_node=1 train.py \
  --data-path /path/to/small_imagefolder \
  --results-dir ../profile_runs/dit_train \
  --global-batch-size 8
```

nsys：

```bash
nsys profile \
  -o ../profile_runs/dit_train/rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  torchrun --standalone --nproc_per_node=1 train.py \
  --data-path /path/to/small_imagefolder \
  --results-dir ../profile_runs/dit_train \
  --global-batch-size 8
```

重点看：

- VAE encode 是否在训练 step 内占用大量时间和显存。
- DiT forward/backward 占比。
- DDP gradient all-reduce 时间。
- dataloader resize/crop 是否拖慢。

## 11. Open-Sora 实验

本地路径：

```text
/data/home/sheshuchen/Open-Sora
```

入口：

```text
scripts/diffusion/train.py
scripts/diffusion/inference.py
scripts/vae/train.py
scripts/vae/inference.py
```

Open-Sora 的重点是视频数据、bucket、不同帧数和并行策略。

### 11.1 demo 训练 profiling

先使用小配置或 demo 配置：

```bash
cd /data/home/sheshuchen/Open-Sora
torchrun --standalone --nproc_per_node=1 \
  scripts/diffusion/train.py configs/diffusion/train/demo.py \
  --dataset.data-path /path/to/small_meta.csv
```

nsys：

```bash
nsys profile \
  -o profile_runs/opensora_demo/rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  torchrun --standalone --nproc_per_node=1 \
  scripts/diffusion/train.py configs/diffusion/train/demo.py \
  --dataset.data-path /path/to/small_meta.csv
```

重点看：

- video read/decode 是否进入 step。
- bucket 后每步 token 数是否稳定。
- VAE/T5 是否在训练进程中执行。
- dataloader 是否和 GPU 计算 overlap。
- attention kernel 是否占主要时间。

### 11.2 VAE 单独 profiling

```bash
cd /data/home/sheshuchen/Open-Sora
nsys profile \
  -o profile_runs/opensora_vae_infer \
  --trace=cuda,nvtx,osrt \
  python scripts/vae/inference.py [配置参数按项目要求补齐]
```

目的：

- 单独测 VAE encode/decode 显存。
- 确认 VAE 是否应该离线缓存。
- 估算远程 VAE 服务吞吐需求。

### 11.3 bucket / packing 实验

对比三组：

```text
固定 batch size
按帧数 bucket
按 token budget dynamic packing
```

记录：

```text
tokens_per_step
padding_ratio
step_time_p50/p90/p99
max_memory_allocated
gpu_idle_time
```

预期：

- bucket 会降低 step time 抖动。
- dynamic packing 会提高有效 token 吞吐。
- 极长视频仍可能成为长尾，需要单独 bucket 或降低 batch。

## 12. FastVideo 实验

本地路径：

```text
/data/home/sheshuchen/FastVideo
```

训练入口：

```text
fastvideo/train/entrypoint/train.py
```

FastVideo 值得重点看：

- attention backend。
- video sparse attention。
- dataset parquet/map-style/iterable-style。
- VAE/T5 encoder 分离。
- training pipeline callback。

### 12.1 训练入口 profiling

示例命令需要按 FastVideo 配置补齐：

```bash
cd /data/home/sheshuchen/FastVideo
nsys profile \
  -o profile_runs/fastvideo_train/rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  torchrun --standalone --nproc_per_node=1 \
  fastvideo/train/entrypoint/train.py [config_or_args]
```

重点看：

- attention backend 最终调用哪个 kernel。
- sparse attention 是否真的减少 kernel 时间。
- dataset parquet 读取是否成为瓶颈。
- T5/VAE 是否和训练主链路隔离。
- callbacks 是否插入长耗时操作。

### 12.2 Attention backend 对比

建议对比：

```text
sdpa
flash_attn
video_sparse_attn
sage_attn
```

记录：

```text
attention_backend
seq_len
heads
hidden
step_time
attention_cuda_time
max_memory_allocated
生成质量或训练 loss 是否异常
```

重要判断：

- sparse attention 如果只是 mask 变稀疏，但底层仍跑 dense kernel，未必更快。
- 真正收益来自 block sparse / video sparse kernel 避免计算无用 score。
- 训练端变快但推理端不支持 streaming，也可能不是最终方案。

## 13. 显存实验矩阵

建议跑下面矩阵，每组只跑 20 到 50 个 step：

| 实验 | batch/token | attention | precision | checkpoint | offload | 目标 |
|---|---|---|---|---|---|---|
| baseline | 固定 | dense | BF16 | off | off | 获取基线 |
| ckpt | 固定 | dense | BF16 | on | off | 看 activation 节省和时间代价 |
| packing | token budget | dense | BF16 | off | off | 看 padding 和负载均衡 |
| sparse | token budget | sparse | BF16 | off | off | 看 attention 时间变化 |
| fp8 | token budget | dense | FP8 | off | off | 看低精度收益和稳定性 |
| offload | 固定 | dense | BF16 | off | optimizer | 看 offload 代价 |

每组记录：

```text
max_memory_allocated_gb
max_memory_reserved_gb
step_time_avg
step_time_p90
tokens_per_second
samples_per_second
MFU
loss
是否 OOM
是否 NCCL timeout
```

## 14. 时间实验矩阵

建议把 step time 拆成：

```text
data_time
h2d_time
forward_time
backward_time
communication_time
optimizer_time
checkpoint_time
```

如果没有精确通信计时，先用 nsys 中 NCCL kernel 区间近似。

记录模板：

```markdown
| 项目 | 配置 | data | h2d | fwd | bwd | nccl | opt | total | token/s | mem GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiT | bs=8 dense | | | | | | | | | |
| Open-Sora | bucket 33f | | | | | | | | | |
| FastVideo | sparse backend | | | | | | | | | |
```

## 15. nsys 判断清单

打开 nsys 后按顺序检查：

- GPU lane 是否有空白。
- CPU 是否在 Python/DataLoader/I/O 上阻塞。
- CUDA memcpy 是否进入关键路径。
- NCCL kernel 是否长且无法 overlap。
- attention kernel 是否占比最高。
- optimizer kernel 是否有大量 HBM 读写。
- 是否有很多小 kernel。
- rank 之间 step 边界是否对齐。
- checkpoint 是否卡住所有 GPU。

## 16. 常见结论写法

示例：

```text
结论 1：当前瓶颈不是 DiT forward，而是 VAE encode 混在训练 step 中，导致 GPU 显存峰值增加 18GB，并产生 120ms 的额外 CUDA kernel。
建议：将 VAE encode 离线缓存，训练只读取 latent。
```

```text
结论 2：长视频 batch 的 token 数是短视频的 2.3 倍，step time p99 比 p50 高 68%。
建议：按 token budget dynamic packing，并把超长视频单独 bucket。
```

```text
结论 3：Sparse Attention mask 已启用，但 nsys 显示仍调用 dense attention kernel，attention 时间没有下降。
建议：接入真正 block sparse kernel 或 video sparse backend。
```

```text
结论 4：FSDP all-gather 与 backward 没有 overlap，NCCL kernel 位于 step 尾部形成长阻塞。
建议：调整 bucket、prefetch、wrap policy 或 parallel group。
```

## 17. 实验注意事项

- 每次只改一个变量。
- 先 warmup，再 profile。
- 记录有效 token 数，而不只是 batch size。
- profile 时关闭不必要 validation。
- 多 rank 报告文件必须带 rank id。
- 不要把 nsys overhead 当作真实吞吐。
- 关注 p50/p90/p99，而不只看平均 step time。
- 确认 loss 正常，否则性能数字没有意义。

## 18. 推荐入职第一批实验

1. 跑一个最小 DiT sample nsys，理解采样循环和 VAE decode。
2. 跑一个 Open-Sora demo train nsys，观察视频数据链路和 bucket。
3. 跑一个 FastVideo attention backend 对比，确认 sparse attention 是否真实节省时间。
4. 对同一模型跑 fixed batch 与 token budget packing，比较 step time 抖动。
5. 对同一配置跑 checkpoint on/off，比较 activation 显存和训练时间。
6. 对同一配置跑 BF16 与 FP8，比较吞吐和 loss 稳定性。

## 19. 最终报告模板

```markdown
# Profiling Report

## 实验配置
- commit:
- hardware:
- model:
- dataset:
- precision:
- parallel:
- attention:
- batch/token budget:

## 关键结果
| 指标 | 数值 |
|---|---:|
| step_time_avg | |
| step_time_p90 | |
| tokens/s | |
| max_memory_allocated | |
| max_memory_reserved | |
| MFU | |

## nsys 观察
- GPU idle:
- NCCL:
- H2D:
- attention:
- optimizer:
- checkpoint:

## 结论
1.
2.
3.

## 下一步
1.
2.
3.
```

## 20. 一句话总结

Profiling 的目标不是生成漂亮 trace，而是把“慢”和“占显存”拆成可归因、可复现、可对比的数字，然后指导数据链路、并行策略、attention kernel 和训练配置的取舍。
