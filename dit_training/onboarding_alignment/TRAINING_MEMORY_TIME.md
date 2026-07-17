# 训练时显存和耗时构成：从一个 Step 看懂瓶颈

这份文档解释一次视频 DiT 训练 step 中各部分显存和时间花在哪里，目标是让你能做基本估算、读 profiler，并知道优化方案为什么有效。

## 1. 一个训练 step 的典型流程

一个纯训练 step 通常是：

```text
读取缓存 batch
  -> H2D / GPU batch 准备
  -> forward
  -> loss
  -> backward
  -> gradient communication
  -> optimizer step
  -> logging / checkpoint
```

如果训练 step 里还混入视频解码、VAE encode 或 T5 encode，那么它就不再是纯训练链路，显存峰值和耗时都会更难解释。

## 2. 显存总账

训练显存可以粗略拆成：

```text
显存峰值 =
  参数
  + 梯度
  + optimizer state
  + activation
  + attention 临时张量
  + 通信 buffer
  + dataloader / batch buffer
  + CUDA workspace / allocator cache
  + 其他模型或服务残留
```

优化前先要确认哪一项最大，否则容易用错方案。

## 3. 参数显存

参数显存是模型权重本身占用的显存，大小约为：

```text
num_parameters * bytes_per_parameter
```

常见精度：

- FP32：每参数 4 bytes。
- BF16/FP16：每参数 2 bytes。
- FP8：每参数 1 byte，但训练中通常还会保留 scale 或 master weight。

例如 10B 参数模型只存 BF16 权重就约 20GB，但训练时绝不只存权重。

优化方法：

- FSDP / ZeRO 切分参数。
- CPU/NVMe offload 参数。
- 更低精度权重。
- 模型结构裁剪或共享。

时间影响：

- 参数本身不直接耗时，但 FSDP all-gather 参数会增加通信时间。
- 权重精度和 layout 会影响 GEMM/Tensor Core 性能。

## 4. 梯度显存

梯度显存通常与可训练参数同规模，BF16 梯度约为：

```text
num_trainable_parameters * 2 bytes
```

如果梯度用 FP32 保存，则翻倍。

优化方法：

- FSDP / ZeRO 切分梯度。
- gradient accumulation 降低单 step batch，但不会减少单次 backward 的参数梯度规模。
- reduce-scatter 让梯度通信后只保留本 rank 分片。

时间影响：

- 梯度产生在 backward。
- 梯度同步是分布式训练中的主要通信耗时之一。
- 通信能否与 backward overlap 会明显影响 step time。

## 5. Optimizer State 显存

AdamW 通常保存一阶动量 `m` 和二阶动量 `v`，如果用 FP32，每个参数额外约 8 bytes。

典型 AdamW 总账：

```text
BF16 参数 2 bytes
BF16/FP16 梯度 2 bytes
FP32 master weight 4 bytes
FP32 m 4 bytes
FP32 v 4 bytes
合计约 16 bytes / parameter
```

不同框架会有差异，但这个估算能帮助你理解 optimizer 为什么非常吃显存。

优化方法：

- ZeRO / FSDP sharding optimizer state。
- 8-bit optimizer。
- CPU offload optimizer state。
- 使用更省状态的优化器。

时间影响：

- optimizer step 会读写大量参数和 state，常常是 HBM bandwidth-bound。
- offload optimizer 会增加 PCIe/NVLink/CPU 内存传输时间。

## 6. Activation 显存

Activation 是 forward 中为了 backward 保存的中间结果，通常与以下因素成正比：

```text
batch_size * sequence_length * hidden_size * num_layers
```

视频 DiT 中 sequence length 来自：

```text
frames * height_patches * width_patches + text_tokens + reference_tokens
```

因此视频帧数翻倍，activation 显存和 attention 计算通常都会明显上升。

优化方法：

- gradient checkpointing / recompute。
- sequence parallel。
- context parallel。
- dynamic packing 降低 padding。
- 降低 micro-batch。
- 更稀疏的 attention。

时间影响：

- 不 checkpoint 时 forward 更快但显存高。
- checkpoint 会在 backward 中重跑部分 forward，通常增加计算时间。
- activation offload 会显著增加数据搬运时间。

## 7. Attention 显存

Dense attention 的 score 形状约为：

```text
batch * heads * query_tokens * key_tokens
```

如果 `N` 是 token 数，dense attention 的 score 规模约为 `O(N^2)`。

视频任务中这非常危险，因为：

- 帧数增加会线性增加 token。
- score 矩阵会按 token 平方增长。
- mask、softmax、dropout 也会产生中间张量。

优化方法：

- FlashAttention：不显式存完整 attention matrix。
- Sparse Attention：减少每个 query 看的 key 数。
- Ring Attention：KV 分块流动，降低单 rank KV 峰值。
- Context Parallel：序列切分后跨卡获取上下文。
- Sliding window / block sparse / key frame attention。

时间影响：

- Dense attention 可能同时是计算瓶颈和显存瓶颈。
- FlashAttention 主要减少 HBM 读写和中间存储。
- Sparse attention 只有配合高效 kernel 才能真正变快。

## 8. 通信 Buffer 显存

分布式训练需要额外 buffer 存放通信中的张量，例如：

- DDP gradient bucket。
- FSDP parameter all-gather buffer。
- FSDP reduce-scatter buffer。
- TP all-reduce buffer。
- CP all-gather K/V buffer。
- pipeline P2P activation buffer。

这些 buffer 的显存峰值经常被忽略，但在大模型和长序列下可能很明显。

优化方法：

- 调整 bucket size。
- 通信和计算 overlap。
- 减少不必要 all-gather。
- 使用 reduce-scatter 替代 all-reduce。
- 对 CP/Ring Attention 做分块通信。

时间影响：

- 通信时间由数据量、拓扑、带宽、延迟和 NCCL 算法共同决定。
- 通信不能 overlap 时会直接增加 step time。

## 9. 临时 Workspace 和 CUDA Allocator Cache

很多算子会申请临时 workspace，例如 cuBLAS、cuDNN、attention kernel、编译器生成 kernel 和排序/packing 操作。

PyTorch CUDA caching allocator 还会保留已释放显存以便复用，所以 `nvidia-smi` 看到的显存不等于真实活跃 tensor 显存。

优化方法：

- 用 PyTorch memory summary / snapshot 区分 allocated 和 reserved。
- 固定 shape 或 bucket shape。
- 避免训练中频繁创建大临时 tensor。
- 让 VAE/T5/采样验证和训练进程隔离。

时间影响：

- 频繁申请释放会造成 allocator overhead。
- workspace 不够可能导致 kernel 选择更慢算法。

## 10. Data / Batch Buffer 显存

训练前的 batch 可能包括：

- raw frames。
- VAE latents。
- text embeddings。
- attention mask。
- position ids。
- packed sequence metadata。
- reference frame inputs。

如果 raw video、VAE、T5 和 DiT 同时在 GPU 上，batch buffer 会显著扩大显存峰值。

优化方法：

- 离线缓存 latent/text。
- CPU pinned memory + async H2D。
- 只把当前 step 需要的 tensor 搬到 GPU。
- 数据预取与训练计算 overlap。
- bucketing 减少 padding。

时间影响：

- DataLoader 慢会导致 GPU 空洞。
- H2D 没有 overlap 会进入 step time。
- 远程服务传输慢会拖慢训练入口。

## 11. Forward 时间

Forward 时间主要由以下部分组成：

- embedding 和 projection。
- DiT block 中 attention。
- MLP。
- norm、residual、dropout。
- loss 计算。

视频 DiT 中 forward 的大头通常是 attention 和 MLP，其中 attention 对 token 数更敏感，MLP 对 hidden size 和层数更敏感。

优化方法：

- FlashAttention / sparse attention。
- fused MLP。
- fused norm。
- 低精度 Tensor Core。
- 减少 padding token。
- 编译优化。

## 12. Backward 时间

Backward 通常比 forward 更慢，因为它要计算输入梯度和参数梯度，并可能触发通信。

Backward 时间包含：

- attention backward。
- MLP backward。
- norm backward。
- activation recompute。
- gradient reduce / reduce-scatter。

优化方法：

- checkpoint 选择性开启，而不是全开。
- backward 通信 overlap。
- fused backward kernel。
- 减少无效 padding。
- 合理设置 micro-batch。

## 13. Optimizer Step 时间

Optimizer step 会读写参数、梯度和 optimizer state，因此经常受 HBM 带宽限制。

AdamW 每步大致要：

- 读参数。
- 读梯度。
- 读写一阶动量。
- 读写二阶动量。
- 写回更新后的参数。

优化方法：

- fused optimizer。
- 8-bit optimizer。
- sharded optimizer。
- 减少参数规模。
- optimizer step 与其他工作重叠。

## 14. Checkpoint 时间

Checkpoint 保存会影响训练吞吐，尤其在多机大模型中可能造成明显停顿。

Checkpoint 内容可能包括：

- model weights。
- optimizer state。
- scheduler state。
- dataloader state。
- EMA。
- random seed。

优化方法：

- 异步 checkpoint。
- 分片 checkpoint。
- 只保存必要状态。
- 降低保存频率。
- 使用高吞吐存储或专门 checkpoint 服务。

## 15. Logging / Validation 时间

训练中穿插采样验证、VAE decode 或指标计算会污染 step time。

优化方法：

- 训练进程只记录轻量标量。
- heavy validation 放到独立进程或独立节点。
- 固定 validation 频率。
- 避免在训练 GPU 上长时间解码视频。

## 16. 动态长度视频带来的额外显存和时间问题

不同视频长度会导致：

- token 数不同。
- attention matrix 大小不同。
- activation 峰值不同。
- 每个 GPU 计算量不同。
- padding 浪费不同。
- NCCL 等待最慢 rank。

优化方法：

- dynamic packing。
- bucketing。
- token-based batch size。
- length-aware sampler。
- 按 token 而不是按样本统计吞吐。

一句话：视频训练不要只说 batch size，要说 token budget。

## 17. 常见优化手段对显存和时间的影响

| 方法 | 省显存 | 对时间的影响 | 适用场景 |
|---|---:|---:|---|
| FSDP / ZeRO | 高 | 增加通信，可 overlap | 参数和 optimizer state 太大 |
| Gradient checkpointing | 中到高 | 增加重计算 | activation 太大 |
| Activation offload | 高 | 通常明显变慢 | 显存实在不够 |
| Optimizer offload | 高 | optimizer step 变慢 | optimizer state 放不下 |
| FP8/FP4 | 中到高 | 可能显著变快 | 精度稳定时 |
| FlashAttention | 中 | 通常变快 | attention HBM 压力大 |
| Sparse Attention | 高潜力 | 取决于 kernel | 长视频 token 很多 |
| Dynamic packing | 中 | 通常变快 | 样本长度差异大 |
| Bucketing | 中 | 通常变稳 | 分辨率/帧数变化大 |

## 18. 一个视频 DiT step 的排查顺序

1. 先看 GPU utilization 是否有空洞。
2. 再看 step time 是否随视频长度波动。
3. 看显存峰值来自 activation、optimizer 还是 attention workspace。
4. 看 DataLoader/H2D 是否进入关键路径。
5. 看 NCCL 是否阻塞 backward。
6. 看 attention 和 MLP 的 kernel 是否占主要时间。
7. 看 padding token 比例是否过高。
8. 看 optimizer step 是否 HBM-bound。
9. 最后再决定并行、低精度、packing、sparse attention 或 offload。

## 19. 面试或入职时可以这样表达

- 我会先把训练 step 拆成数据、forward、backward、通信、optimizer 和 checkpoint，而不是直接猜瓶颈。
- 我会先算显存总账，区分参数、梯度、optimizer state、activation 和临时 workspace。
- 我会用 token throughput、step time、MFU 和端到端时间一起评估优化效果。
- 我理解 offload 是最后手段之一，因为它通常牺牲吞吐换显存。
- 我理解视频训练要按 token budget 做 packing，而不是只看样本 batch size。
- 我理解 sparse attention 只有配合高效 kernel 和推理约束，才会真正产生系统收益。

## 20. 需要背下来的估算公式

```text
参数显存 ≈ 参数量 * 参数字节数
梯度显存 ≈ 可训练参数量 * 梯度字节数
AdamW 状态 ≈ 参数量 * 8 bytes  # FP32 m/v，不含 master weight
Activation ≈ batch * seq_len * hidden * layers * 若干常数
Dense Attention score ≈ batch * heads * seq_len^2 * bytes
Token 数 ≈ frames * H_patches * W_patches + text_tokens + extra_tokens
吞吐 ≈ processed_tokens / wall_time
MFU ≈ actual_model_flops_per_second / hardware_peak_flops
```

这些公式不精确，但足够帮你在 profiling 前建立数量级判断。
