# How to Optimize Streaming VGM Inference

Streaming VGM 推理优化的目标是：在持续生成视频时，让每一帧的延迟稳定、显存可控，并且生成越久不会越来越慢。

## 1. 先明确推理形态

Streaming VGM 的推理不是：

```text
prompt -> 一次性生成完整视频
```

而是：

```text
prompt + 历史状态 -> 生成下一帧/chunk -> 更新状态 -> 继续生成
```

所以优化目标不是单次生成总耗时，而是：

- 首帧延迟。
- 每帧或每 chunk 延迟。
- 长时间生成时的延迟稳定性。
- KV cache 显存增长。
- 支持用户交互插入新条件。

## 2. 最重要优化：KV Cache

朴素推理每一步都会重算完整历史：

```text
step 1: prompt + frame_0
step 2: prompt + frame_0 + frame_1
step 3: prompt + frame_0 + frame_1 + frame_2
```

这会让生成越久越慢。

KV cache 做法是：

```text
历史 token 的 K/V 保存下来
每一步只计算新增 token 的 Q/K/V
新增 Q 读取历史 K/V + 当前 K/V
```

代码位置：

```text
streaming_vgm/model.py
  CausalSelfAttention.forward
  StreamingVGM.prefill_prompt
  StreamingVGM.stream_step
  StreamingVGM.generate_stream
```

## 3. 跑推理优化对比

先训练一个小模型：

```bash
python train.py --steps 100 --frames 12
```

再 benchmark：

```bash
python benchmark_inference.py --ckpt outputs/streaming_vgm.pt --frames 32 --repeat 5
```

输出类似：

```text
device=cpu frames=32 repeat=5
naive_full_history=0.120000s
kv_cache_streaming=0.045000s
speedup=2.67x
attention_scores_naive=12529 attention_scores_cache=561
theoretical_attention_reduction=22.33x
```

toy 模型很小，CPU 上 Python 循环可能让 KV cache 实测不一定更快；这里更重要的是 attention score 数量从“每步重算完整历史”降到“新增 token 看历史”，真实大模型/GPU decode 才会把这个理论收益转成稳定延迟收益。

## 4. Chunk 级生成

真实视频生成不一定每次只生成一帧，也可以每次生成一个 chunk：

```text
prompt + history -> next 4 frames
```

优点：

- 更好利用 GPU batch/GEMM。
- 降低 Python 和调度开销。
- 提升吞吐。

缺点：

- 首帧延迟可能变高。
- chunk 内部的交互响应不够细。
- causal mask 和 cache 更新更复杂。

一句话：实时交互看单帧延迟，吞吐优先看 chunk 生成。

## 5. Sliding Window KV Cache

KV cache 会随历史长度线性增长：

```text
cache_memory ∝ layers * heads * history_tokens * head_dim
```

长时间生成时必须限制历史：

- 只保留最近 N 帧。
- 保留关键帧 token。
- 对旧历史做压缩。
- 把静态条件和动态历史分开 cache。

否则显存会随着直播时长不断增长。

## 6. Prefix Cache

Prompt、角色设定、参考图、首帧等条件在生成过程中可能长期不变，可以做 prefix cache。

做法：

```text
prefill text/reference tokens once
后续每帧复用 prefix K/V
```

这对数字人、会议交互、固定场景生成尤其重要。

## 7. Attention 优化

Streaming VGM 的 attention 优化要同时满足训练和推理。

可选方向：

- Causal attention：保证不看未来。
- Sliding window attention：限制每个 token 看的历史范围。
- Sparse attention：只看局部、关键帧或全局摘要。
- Ring attention：长上下文多卡分摊 KV。
- FlashAttention decode kernel：优化单步或小 batch decode。

注意：某些 sparse attention 训练时很快，但如果不能流式更新 KV cache，推理端可能不可用。

## 8. VAE 解码优化

视频生成最终要把 latent decode 成像素帧。

优化方向：

- VAE decode 与下一步 DiT 推理 overlap。
- 按 chunk decode，避免一次 decode 太多帧。
- 使用低精度 VAE。
- 对固定背景或静态区域做缓存。
- 解码和视频编码放到独立 stream 或独立服务。

## 9. 视频编码与输出

实时系统里生成帧后还要编码成视频流。

需要关注：

- GPU 硬件编码器。
- encode latency。
- 帧队列长度。
- 网络发送延迟。
- 音视频同步。

如果只优化 DiT forward，而忽略 VAE decode 和 video encode，端到端延迟仍可能不达标。

## 10. 多请求 Serving

真实服务要同时处理多个用户请求。

优化方向：

- continuous batching。
- 按生成阶段分队列。
- prefix cache 复用。
- 相同分辨率/帧率请求合批。
- 长请求和短请求分队列。
- 把 prefill 和 decode 分离调度。

Streaming VGM 类似 LLM serving，也有 prefill 和 decode 两个阶段。

## 11. 低精度与量化

推理侧可以更激进地使用：

- BF16。
- FP8。
- FP4/NVFP4。
- weight-only quantization。
- KV cache quantization。

但需要验证：

- 长时间生成是否漂移。
- 运动一致性是否下降。
- prompt 跟随是否变差。
- VAE decode 是否放大量化误差。

## 12. CUDA Graph 与编译

Streaming decode 阶段 shape 通常比较稳定，适合：

- CUDA Graph。
- torch.compile。
- TensorRT / 自研 engine。
- fused kernel。

但 dynamic prompt、dynamic resolution、dynamic chunk size 会破坏 graph capture，因此 serving 系统通常需要 bucket shape。

## 13. 端到端延迟拆解

建议把 streaming inference latency 拆成：

```text
text encode / prompt update
DiT prefill
DiT decode step
VAE decode
video encode
network send
queue wait
```

报告指标：

```text
time_to_first_frame
per_frame_latency_p50/p90/p99
frames_per_second
gpu_memory_per_request
max_concurrent_requests
quality_metric
```

## 14. 本 mini 项目展示了什么

本项目展示两个版本：

- `generate_naive`：每一步重算完整历史。
- `generate_stream`：prefill prompt 后用 KV cache 逐帧生成。

对比脚本：

```text
benchmark_inference.py
```

它说明 streaming inference 的第一条原则：**历史必须缓存，新增 token 必须增量计算。**

## 15. 一句话总结

优化 streaming VGM inference 的核心是把“完整视频生成”改造成“prefill + 增量 decode”的系统，并围绕 KV cache、chunk、sliding window、VAE decode、video encode 和 continuous batching 控制端到端延迟。
