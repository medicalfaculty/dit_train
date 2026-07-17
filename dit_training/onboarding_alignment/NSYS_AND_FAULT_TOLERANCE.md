# nsys 使用与 GPU 故障容错

这份文档说明两件事：

1. 如何用 NVIDIA Nsight Systems 的命令行工具 `nsys` 分析视频 DiT 训练瓶颈。
2. 当 GPU、节点、NCCL、DataLoader 或远程预处理服务出问题时，训练框架应该如何容错。

## 1. nsys 是什么

`nsys` 是 Nsight Systems 的命令行工具，用来采集 CPU、CUDA kernel、CUDA memcpy、NCCL、NVTX、OS runtime 等端到端 timeline。

它适合回答：

- GPU 是否空闲。
- DataLoader 是否拖慢训练。
- H2D 拷贝是否和计算重叠。
- NCCL 通信是否阻塞 backward。
- 是否有大量小 kernel。
- VAE/T5/视频解码是否混入训练 step。
- checkpoint 或 logging 是否造成长尾停顿。

一句话：`nsys` 看的是“整个训练系统什么时候在干什么”，不是单个 kernel 的微观指标。

## 2. nsys 与 ncu 的区别

- `nsys`：看端到端 timeline，定位瓶颈属于数据、计算、通信还是等待。
- `ncu`：看单个 CUDA kernel 的细节，例如 occupancy、memory throughput、Tensor Core 利用率。

建议顺序：

```text
先 nsys 找到慢在哪里
再 ncu 深挖某个慢 kernel 为什么慢
```

## 3. 最小采集命令

单进程训练：

```bash
nsys profile \
  -o reports/train_step \
  --trace=cuda,nvtx,osrt \
  --capture-range=none \
  python train.py --steps 20
```

多卡 `torchrun`：

```bash
nsys profile \
  -o reports/torchrun_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  --capture-range=none \
  torchrun --standalone --nproc_per_node=8 train.py --steps 20
```

说明：

- `-o` 指定输出文件前缀。
- `%q{RANK}` 会把环境变量 `RANK` 放进文件名，避免多进程互相覆盖。
- `--trace=cuda,nvtx,osrt` 采集 CUDA、NVTX 标记和 OS runtime。
- `--capture-range=none` 表示从进程开始采到结束，简单但报告可能很大。

## 4. 推荐采集：只采稳定 step

训练刚启动时会有编译、cache warmup、CUDA 初始化和数据预取，通常不代表稳定性能。

更好的方式是在代码里加入 NVTX，并只采中间几个 step。

代码示例：

```python
import torch

for step, batch in enumerate(loader):
    if step == 10:
        torch.cuda.cudart().cudaProfilerStart()

    with torch.cuda.nvtx.range(f"train_step_{step}"):
        loss = train_step(batch)

    if step == 20:
        torch.cuda.cudart().cudaProfilerStop()
        break
```

命令：

```bash
nsys profile \
  -o reports/stable_steps \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --stop-on-range-end=true \
  python train.py
```

这样报告只包含第 10 到第 20 步，体积更小，也更接近真实 steady state。

## 5. 分布式训练推荐命令

8 卡单机：

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile \
  -o reports/rank_%q{RANK} \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --stop-on-range-end=true \
  torchrun --standalone --nproc_per_node=8 train.py
```

如果报告太大，可以只采 rank 0：

```bash
if [ "$LOCAL_RANK" = "0" ]; then
  nsys profile -o reports/rank0 --trace=cuda,nvtx,osrt python train.py
else
  python train.py
fi
```

但注意：只采 rank 0 可能漏掉负载不均衡问题，因为最慢 rank 不一定是 rank 0。

## 6. 生成统计摘要

`nsys profile` 会生成 `.nsys-rep` 文件，可以用 GUI 打开，也可以用命令行导出统计。

常用命令：

```bash
nsys stats reports/stable_steps.nsys-rep
```

导出 SQLite：

```bash
nsys export --type sqlite --output reports/stable_steps.sqlite reports/stable_steps.nsys-rep
```

如果只想快速看 kernel 时间，可以先看：

```bash
nsys stats --report cuda_gpu_kern_sum reports/stable_steps.nsys-rep
```

如果想看 CUDA memcpy：

```bash
nsys stats --report cuda_gpu_mem_time_sum reports/stable_steps.nsys-rep
```

## 7. 训练代码里应该加哪些 NVTX

建议至少标记：

```text
data_load
h2d
forward
loss
backward
grad_comm
optimizer
checkpoint
validation
```

示例：

```python
from torch.cuda import nvtx

nvtx.range_push("forward")
pred = model(batch)
nvtx.range_pop()
```

更推荐用 context manager：

```python
with torch.cuda.nvtx.range("forward"):
    pred = model(batch)
```

一句话：没有 NVTX 的 timeline 很难读，有 NVTX 才能把业务阶段和底层 kernel 对齐。

## 8. nsys 看报告时的顺序

打开报告后按这个顺序看：

1. 看 GPU lane 是否有大段空白。
2. 看 CPU thread 是否在 DataLoader、Python、I/O 或同步上卡住。
3. 看 CUDA memcpy 是否挡在 forward 前面。
4. 看 NCCL 是否和 backward 重叠。
5. 看 attention、MLP、optimizer 哪类 kernel 占大头。
6. 看是否有大量极短 kernel 造成 launch overhead。
7. 看不同 rank 的 step 是否对齐。
8. 看 checkpoint、validation、logging 是否插入训练关键路径。

## 9. 常见 timeline 现象与判断

### GPU 大段空白

可能原因：

- DataLoader 跟不上。
- 视频解码太慢。
- H2D 拷贝没有 overlap。
- Python 侧同步或阻塞。
- checkpoint 保存阻塞。

处理方向：

- 提前离线 VAE/T5。
- 增加 prefetch。
- 使用 pinned memory。
- 把 heavy validation 移出训练进程。
- 检查是否有 `.item()`、`torch.cuda.synchronize()` 等同步点。

### NCCL 很长

可能原因：

- 并行切分不合理。
- bucket 太大或太小。
- 通信不能与 backward overlap。
- rank mapping 不符合 NVLink/NIC 拓扑。
- 跨机 RDMA/NCCL 没走预期路径。

处理方向：

- 看 NCCL log。
- 调整 bucket size。
- 调整 parallel group。
- 优化 rank placement。
- 检查 IB/RDMA/NVLink 状态。

### 小 kernel 很多

可能原因：

- 算子没有融合。
- 动态 shape 导致编译器难优化。
- Python 层循环太多。
- attention mask 或 packing 逻辑散在 step 内。

处理方向：

- kernel fusion。
- torch.compile 或自定义 kernel。
- 固定 bucket shape。
- 把 preprocessing 移出 step。

### memcpy 很多

可能原因：

- CPU/GPU 来回搬数据。
- VAE/T5/训练混用不同 device。
- logging 或 debug 把 tensor 搬回 CPU。
- offload 过多。

处理方向：

- 减少 `.cpu()`、`.numpy()`、`.item()`。
- 固定 device ownership。
- 只 offload 必须 offload 的内容。

## 10. nsys 采集注意事项

- 不要一上来采完整训练，报告会很大。
- 先采 5 到 10 个稳定 step。
- 多卡训练要避免报告文件互相覆盖。
- 采集本身会带来一定 overhead，不要把 profile 结果当作绝对吞吐。
- profile 前先关掉不必要的 validation 和 debug logging。
- 如果使用 CUDA Graph 或 torch.compile，要区分 warmup 和稳定阶段。
- 如果 timeline 看不到 NCCL，确认 `--trace` 和环境配置是否正确。

## 11. GPU “坏了”可能指什么

工程里说 GPU 坏了，不一定是物理损坏，可能包括：

- Xid error。
- ECC error。
- GPU lost / reset。
- GPU 温度或功耗异常。
- 某张卡显存分配失败。
- 某张卡 kernel hang。
- 某个 rank NCCL timeout。
- 某节点网络故障。
- 进程被 OOM killer 杀掉。

一句话：先判断是硬件故障、驱动故障、通信故障、显存 OOM，还是数据/代码触发的单 rank 崩溃。

## 12. GPU 故障时训练框架要做什么

最小容错闭环：

```text
检测错误
  -> 停止所有 rank
  -> 保存或确认最近 checkpoint
  -> 释放坏节点/坏卡
  -> 重新拉起 job
  -> 从 checkpoint 恢复 model/optimizer/dataloader/rng
```

不能只重启某一个 rank，因为分布式训练要求所有 rank 的 collective 顺序一致。

## 13. 必须保存哪些状态才能恢复训练

完整 resume 至少需要：

- model weights。
- optimizer state。
- lr scheduler state。
- grad scaler state。
- dataloader/sampler state。
- global step / epoch。
- random seed / RNG state。
- EMA state。
- mixed precision 配置。
- parallel/sharding 配置。
- 数据版本和 tokenizer/VAE/T5 版本。

如果缺 dataloader state，训练还能继续，但可能重复或跳过部分数据。

如果缺 optimizer state，loss 可能抖动，等价于不完整恢复。

## 14. Checkpoint 设计建议

建议采用：

```text
step_000100/
  model shards
  optimizer shards
  scheduler.pt
  sampler_state.pt
  metadata.json
  COMMIT_DONE
```

关键点：

- 先写临时目录，写完后原子 rename 或写 `COMMIT_DONE`。
- resume 时只读取有完成标记的 checkpoint。
- 分片 checkpoint 要记录 world size 和 shard 规则。
- 定期保留多个 checkpoint，避免最新 checkpoint 损坏。
- checkpoint 保存不要长时间阻塞训练主循环。

## 15. torchrun elastic 的基本思路

PyTorch Elastic 支持 worker 失败后由 agent 重启 worker group。

常见启动形式：

```bash
torchrun \
  --nnodes=2:4 \
  --nproc_per_node=8 \
  --max_restarts=3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29400 \
  train.py --resume auto
```

含义：

- `--max_restarts` 控制最多重启次数。
- elastic 会重启 worker group，而不是只修一个 rank。
- 训练脚本必须支持从 checkpoint 自动恢复。
- 如果节点数量变化，数据并行和 sharding 逻辑必须能处理 world size 改变。

一句话：elastic 负责重启进程，训练代码负责正确恢复状态。

## 16. NCCL hang 和 timeout 怎么处理

NCCL hang 常见原因：

- 某个 rank 先报错退出，其他 rank 还在 collective 中等待。
- 各 rank collective 顺序不一致。
- 某张 GPU 或网卡异常。
- 网络拥塞或 RDMA 退化。
- tensor shape 在不同 rank 不一致。

排查建议：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
```

处理策略：

- 发现单 rank fatal error 后尽快让所有 rank fail fast。
- collective 前检查 shape 和 dtype。
- 对动态 packing 的 batch metadata 做 all-gather 校验。
- 设定合理 timeout，避免无限挂住。
- 保存最近 checkpoint 后整体重启。

## 17. OOM 容错

OOM 不一定是 batch 太大，也可能是某个 batch token 特别多、attention workspace 突增或显存碎片。

可做策略：

- token budget 而不是固定样本数。
- 长视频单独 bucket。
- OOM 后降低当前 bucket batch size。
- OOM batch 记录到 bad batch log。
- 周期性清理无用引用。
- 禁止训练 step 中跑 VAE/T5。

不建议在 OOM 后盲目 `torch.cuda.empty_cache()` 继续训练，因为可能已经破坏了分布式同步顺序。

更稳妥做法是 fail fast 后从 checkpoint 恢复，并调整 batch/bucket 配置。

## 18. 数据错误容错

视频数据常见问题：

- 文件损坏。
- 解码失败。
- 帧数不足。
- fps 或分辨率元信息错误。
- prompt 缺失或乱码。
- VAE/T5 缓存文件损坏。

处理策略：

- 预处理阶段生成 manifest 和 checksum。
- 失败样本写入 bad list。
- 训练阶段只读验证过的缓存。
- 每条缓存带版本号和 shape metadata。
- collate 前校验 token 数、dtype 和 NaN。

一句话：坏数据要在训练前处理掉，不要让训练进程承担视频清洗责任。

## 19. 远程 VAE/T5 服务容错

如果使用远程服务处理 VAE/T5：

- 请求必须有 timeout。
- 结果必须有 retry。
- 输出必须带 checksum。
- 服务版本必须写入 metadata。
- 失败样本不能阻塞整个训练。
- 训练节点最好只消费已完成缓存，不直接依赖在线服务。

推荐链路：

```text
raw video/prompt
  -> preprocessing service
  -> verified latent/text cache
  -> training job
```

不推荐：

```text
training step
  -> remote VAE/T5 RPC
  -> wait
  -> train
```

## 20. 节点故障容错

多机训练中节点故障可能来自：

- 机器断电或重启。
- GPU Xid。
- NIC/RDMA 故障。
- 本地盘故障。
- 进程被调度系统杀掉。

框架层策略：

- 周期性 checkpoint。
- job launcher 支持重试。
- rank membership 变化时能重新初始化 process group。
- 数据 sampler 支持从 global step 恢复。
- checkpoint 在共享存储或对象存储中，而不是只在坏节点本地盘。

## 21. 如何判断是否需要降级训练

遇到故障后可以考虑降级：

- 移除坏 GPU。
- 减少节点数。
- 降低 token budget。
- 关闭 sparse/dynamic experimental kernel。
- 从 FP8 回退 BF16。
- 关闭 validation。

降级的前提是：

- checkpoint 可以在新 world size 下加载。
- sharding 规则支持 reshard。
- batch size / lr 策略重新计算。
- 实验记录清楚标记配置变化。

## 22. 容错和性能的冲突

更强容错通常会牺牲性能：

- 更频繁 checkpoint 会增加 I/O。
- 更严格校验会增加 CPU 时间。
- 更小 bucket 会降低吞吐。
- 更保守 timeout 会更容易重启。
- 保存完整 optimizer state 会占更多存储。

因此工业训练通常按任务阶段调整：

- debug 阶段：更强校验和更频繁 checkpoint。
- 长训阶段：稳定配置、适中 checkpoint 间隔、尽量减少主链路开销。
- 冲刺阶段：优先吞吐，但保留关键故障恢复能力。

## 23. 推荐的最小容错实现清单

- 训练脚本支持 `--resume auto`。
- 每 N step 保存完整 checkpoint。
- checkpoint 写入有完成标记。
- dataloader/sampler 可恢复。
- 所有 rank 遇到 fatal error 一起退出。
- 记录 bad data，不在训练 step 中反复尝试。
- NCCL 设置 async error handling。
- 日志记录 hostname、rank、local_rank、GPU UUID。
- 启动前检查 GPU ECC/Xid 状态。
- profile 和故障日志保留到实验目录。

## 24. 入职后应该能说清楚的几句话

- 我会先用 nsys 判断训练是数据、计算、通信还是同步瓶颈，而不是直接猜。
- 我会给训练代码加 NVTX，把 forward、backward、NCCL、optimizer 和 checkpoint 在 timeline 中标出来。
- GPU 出错时不能只重启单个 rank，而要让 worker group 一致退出并从 checkpoint 恢复。
- 真正可靠的 resume 需要 model、optimizer、scheduler、sampler、RNG 和 global step。
- 视频解码、VAE 和 T5 最好离线或服务化，训练进程只消费验证过的缓存。
- 容错策略要和性能权衡，debug 阶段更严格，长训阶段更轻量。

## 25. 资料来源

- Nsight Systems User Guide：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Systems CLI / nsys 文档：https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-command-switch-options
- Nsight Systems Stats 文档：https://docs.nvidia.com/nsight-systems/UserGuide/index.html#statistics
- PyTorch Elastic torchrun 文档：https://docs.pytorch.org/docs/stable/elastic/run.html
- PyTorch Distributed Elastic 文档：https://docs.pytorch.org/docs/stable/distributed.elastic.html
