# SP、CP、Ring Attention 最小 PyTorch 例子

这三个脚本都能用 CPU 跑通：

```bash
torchrun --standalone --nproc_per_node=2 01_sequence_parallel.py
torchrun --standalone --nproc_per_node=2 02_context_parallel.py
torchrun --standalone --nproc_per_node=2 03_ring_attention.py
```

- `01_sequence_parallel.py`：SP，把 sequence 维切开，逐 token 算子在本地分片上计算。
- `02_context_parallel.py`：CP，本地 rank 只算本地 query，但 attention 需要全局 key/value。
- `03_ring_attention.py`：Ring Attention，不 all-gather 完整 KV，而是让 KV 块沿 ring 传递，并用在线 softmax 累计结果。

输出里的 `max_error` 应该接近 0，表示并行写法和单进程 full attention / full sequence 参考结果一致。
