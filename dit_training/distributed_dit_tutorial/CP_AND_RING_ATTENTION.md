# CP 并行与 Ring Attention

这份文档继续补充 DiT/Transformer 训练里常见的长序列并行方法：Context Parallel，简称 CP，以及 Ring Attention。

## 为什么需要 CP

Transformer 训练的显存压力不只来自参数，还来自激活。注意力层的输入通常是：

```text
x: [batch, seq_len, hidden_dim]
```

当 `seq_len` 很长时，比如视频 DiT、长文本、多帧 latent token，单卡保存完整序列激活会很贵。CP 的做法是沿序列维切分：

```text
rank0: x[:, 0:seq/2]
rank1: x[:, seq/2:seq]
```

每张卡只常驻一段 token 的激活。这样 MLP、LayerNorm、Q projection 这类逐 token 计算天然可以在本地完成。

## CP Attention 的核心矛盾

注意力不是完全本地计算。第 `i` 个 query token 理论上要看全序列的 key/value：

```text
Attention(Q_i, K_all, V_all)
```

所以 CP attention 的关键是：

1. Query 可以只保留本 rank 的本地 query。
2. Key/Value 必须让每个 rank 能看到全局信息。
3. 输出仍然只保留本 rank 对应的序列片段。

`04_context_parallel_attention.py` 使用最直接的实现：先 `all_gather` 所有 rank 的本地序列，再计算全局 K/V，然后每个 rank 只计算自己的 query 输出。

启动：

```bash
python 04_context_parallel_attention.py
torchrun --standalone --nproc_per_node=2 04_context_parallel_attention.py --seq-len 16 --dim 32
```

预期输出：

```text
world_size=2 device=cpu
full_sequence=(2, 16, 32) local_sequence=(2, 8, 32)
max_error_vs_full_attention=0.00000000
CP 的重点：序列维切分；每个 rank 只算本地 query，但 attention 需要全局 key/value。
```

`max_error` 接近 0，说明 CP 版本每个 rank 的本地输出，和单卡完整 attention 的对应切片一致。

## Ring Attention 解决什么问题

直接 `all_gather K/V` 容易理解，但它会让每个 rank 临时保存完整 K/V。序列很长时，这个临时峰值显存也可能很大。

Ring Attention 的思路是：不要一次拿到完整 K/V，而是让 K/V 块沿 ring 传递。

以 4 个 rank 为例：

```text
初始：
rank0 持有 KV0
rank1 持有 KV1
rank2 持有 KV2
rank3 持有 KV3

第 1 轮：
每个 rank 用本地 Q 计算当前 KV 块
然后 KV0 -> rank1, KV1 -> rank2, KV2 -> rank3, KV3 -> rank0

第 2 轮：
每个 rank 计算收到的新 KV 块
继续传递

循环 world_size 轮后：
每个 rank 的本地 Q 都看过所有 KV 块
```

每个 rank 始终只需要保存：

- 本地 Q。
- 当前正在处理的一个 K 块。
- 当前正在处理的一个 V 块。
- 在线 softmax 的中间状态。

## 在线 Softmax 细节

普通 attention 是：

```text
softmax(QK^T / sqrt(d)) V
```

如果 K/V 分块处理，不能对每一块单独 softmax 后直接相加，因为 softmax 的分母必须覆盖所有 key。

Ring Attention 使用在线 softmax。每处理一个新块，就维护三样东西：

- `running_max`：到目前为止所有 score 的最大值，用来做数值稳定。
- `running_sum`：到目前为止 `exp(score - running_max)` 的累计分母。
- `running_out`：到目前为止 `exp(score - running_max) @ V` 的累计分子。

当新块来了：

```text
block_score = Q @ K_block^T / sqrt(d)
block_max = max(block_score)
new_max = max(running_max, block_max)

旧累计值要乘 exp(running_max - new_max)
新块 exp 要用 exp(block_score - new_max)
```

最后：

```text
output = running_out / running_sum
```

这和一次性对完整 `K_all/V_all` 做 softmax attention 在数学上等价。

## Ring Attention 教学脚本

启动：

```bash
python 05_ring_attention.py
torchrun --standalone --nproc_per_node=2 05_ring_attention.py --seq-len 16 --dim 32
```

预期输出：

```text
world_size=2 device=cpu
local_q=(2, 8, 32) local_kv=(2, 8, 32)
max_error_vs_dense_attention=0.00000012
Ring Attention 的重点：K/V 块沿 ring 传递；每个 rank 边收边算，不一次保存完整 K/V。
```

误差通常是 `1e-6` 以内，来自浮点计算顺序差异。

## CP、Ring Attention、FSDP 的关系

- FSDP：主要切参数、梯度、优化器状态，解决模型参数显存。
- CP：主要切序列激活，解决长序列激活显存。
- Ring Attention：是 CP attention 的一种通信与计算方式，避免每个 rank 一次性保存完整 K/V。

大模型训练里这些技术经常组合使用：

```text
FSDP / ZeRO: 参数维度省显存
Tensor Parallel: hidden/head 维度并行
Context Parallel: sequence 维度并行
Ring Attention: CP 下更省峰值显存的 attention 实现
```

## 在 DiT 里的位置

DiT 把图片或视频 latent 切成 token 序列，然后用 Transformer block 处理。图片较小时 CP 意义不大；视频、多帧、高分辨率 latent 会产生很长的 token 序列，这时 CP 和 Ring Attention 就变得重要。

在真实 DiT 工程中，接入点通常在 attention 层：

1. 输入 hidden states 已经按序列维分到不同 rank。
2. 本 rank 计算本地 `Q/K/V projection`。
3. Attention 不再直接 `Q @ K_all^T`，而是调用 CP/Ring Attention kernel。
4. 输出仍是本 rank 的本地序列片段。
5. 后续 MLP 继续本地计算。

本教程没有把 CP/Ring 写进 `TinyDiT`，是为了让并行算法本身更清楚；真实工程会把 `DiTBlock.attn` 替换成支持 CP 的 attention 实现。
