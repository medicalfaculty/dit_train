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

## 16. 追加问题：KV Cache 还能怎么精细管理

目前很多 streaming VGM 对 KV cache 的管理仍然比较粗糙，常见做法只是保留 prompt sink token、最近窗口和少量参考帧；但视频生成天然具有背景稳定、临近帧相似、主体运动连续等特点，因此 cache 可以比 LLM decode 做得更有结构。

可以考虑四类方向：

1. **重要 block 保留**：除了 sink token，还可以保留背景、主体、关键交互区域、参考帧对应的时空 block，而不是只按时间最近原则保留。
2. **KV pruning**：根据 attention score、token 运动幅度、语义重要性或重建误差，删掉对未来生成贡献较小的历史 token。
3. **KV compression**：把旧历史 token 合并成低分辨率摘要，例如时间池化、空间池化、聚类、low-rank projection 或 learned memory token。
4. **分层 memory**：把 cache 分成短期高精度窗口、中期压缩记忆、长期语义记忆，让不同 attention 层或不同 head 读取不同层级。

一个更适合视频的 KV cache 结构可以是：

```text
prompt / identity sink tokens：长期保留
最近 N 帧局部 token：完整保留
关键帧 token：稀疏保留
背景 memory：压缩保留
主体 memory：按检测或 attention 动态保留
旧历史：低频压缩或丢弃
```

更进一步，可以基于语义设计辅助记忆模块：例如把人物身份、场景布局、镜头运动、物体轨迹、用户指令状态分别维护为 memory token，并让视频 token 通过 cross-attention 读取这些结构化记忆。

关键评估指标不是 cache 命中率，而是：在相同显存预算下，长时间生成的一致性、可控性和延迟是否更好。

## 17. 追加问题：Chunk 内 full-attention 还能不能稀疏化

当前很多 streaming VGM 会把生成拆成 chunk，但 chunk 内仍然使用 full attention，这会让每个 chunk 的计算量仍然很高。

可以进一步做 chunk 内 sparsity：

- **Temporal local attention**：每个 token 只看相邻时间窗口内的 token。
- **Spatial local attention**：同一帧内只看局部空间邻域。
- **Frame-level global tokens**：每帧保留少量全局 token，局部 token 通过它们交换全局信息。
- **Reference-frame attention**：只对少数关键帧或参考帧做全局 attention。
- **Motion-aware sparse attention**：沿运动轨迹连接 token，而不是按固定窗口连接。
- **Head-wise sparsity**：部分 head 做局部，部分 head 做全局或语义 memory。
- **Layer-wise sparsity**：低层偏局部，高层偏全局，避免所有层都 dense。

一个可行设计是：

```text
chunk 内：局部时空 attention + 少量 global token
chunk 间：KV cache / memory token / key frame token
文本和参考图：作为 prefix 或 cross-attention memory
```

注意：稀疏 attention 只有在底层 kernel 真正跳过无效计算时才会加速；如果只是 dense kernel 上加 mask，通常只能改变注意力模式，不能显著降低计算量。

## 18. 追加问题：显卡容量约束下如何设计 Streaming VGM Serving

已有 streaming VGM serving 往往默认单卡显存足够，并且一张卡能同时跑多个任务；但在超长序列、多用户和高分辨率下，KV cache、VAE buffer、batch queue 和中间 activation 都会受显存容量限制。

更细的 serving 设计应当把显存预算显式纳入调度：

```text
每请求显存 = 模型常驻权重分摊
          + prompt/prefix cache
          + video KV cache
          + VAE decode buffer
          + output frame queue
          + runtime workspace
```

调度器应该维护每个请求的 cache footprint，并基于显存预算决定：

- 是否接受新请求。
- 每个请求允许保留多少历史 token。
- 是否降低 chunk size。
- 是否启用更强 cache compression。
- 是否把长请求迁移到更空闲 GPU。
- 是否把 prefill 和 decode 分配到不同 GPU。
- 是否对低优先级请求降级帧率或分辨率。

在超长序列场景下，SP/CP 也不能再默认“单卡放得下”。可以考虑：

- **按时间维 SP**：不同 GPU 持有不同时间段 token 或 cache。
- **按空间 block SP**：高分辨率时按空间 block 切分。
- **CP + Ring Attention**：本地 query 通过 ring 读取远端 KV，避免每卡保存完整历史。
- **分层 cache placement**：最近窗口放本地 GPU，旧历史压缩后放远端 GPU 或 CPU。
- **request-aware parallel group**：长请求分配多卡，短请求单卡批处理。

一个更工程化的 serving 架构可以是：

```text
router
  -> admission control：检查显存预算
  -> prefill workers：处理 prompt / reference / initial frames
  -> decode workers：流式生成后续 chunk
  -> cache manager：管理 KV/memory 压缩、迁移、淘汰
  -> VAE/video workers：异步解码和编码输出
```

核心原则：把 KV cache 当作一等公民管理，而不是让它隐式堆在模型进程里。

## 19. 追加问题：长序列可控性和一致性需要什么 Benchmark

长序列 streaming VGM 不能只看单帧质量或短视频 FVD，因为真正难点是长时间持续生成中的可控性和一致性。

需要设计更贴近 streaming 的 benchmark：

1. **身份一致性**：同一人物、角色、物体在长时间生成中是否保持外观一致。
2. **背景稳定性**：静态背景是否漂移、闪烁或逐渐变形。
3. **运动连续性**：物体轨迹是否平滑，是否突然跳变。
4. **指令保持**：prompt 中的约束在长时间后是否仍然有效。
5. **交互响应**：中途插入新指令后，模型是否及时改变生成，同时不破坏已有场景。
6. **长程因果一致性**：前面发生的事件是否影响后续生成，而不是被遗忘。
7. **循环漂移**：连续生成几百帧后，画面是否逐渐失真或偏离主题。
8. **实时指标**：time-to-first-frame、per-frame latency、p99 latency、cache memory per request。

可以把 benchmark 设计成任务集：

```text
固定角色 + 长时间说话
固定房间 + 镜头缓慢移动
物体从左到右移动并保持颜色
中途插入“转身/换表情/拿起物体”指令
多轮交互后回到初始主题
长时间背景不变但前景运动
```

每个任务同时记录质量和系统指标：

```text
视觉一致性分数
文本/指令跟随分数
identity similarity
background drift
motion smoothness
latency p50/p90/p99
KV cache memory
frames per second
```

理想 benchmark 应当同时约束模型质量和 serving 系统，因为 streaming VGM 的好坏不是单张图质量，而是在有限显存和低延迟下长期稳定可控地生成视频。

## 20. 追加总结

Streaming VGM 的下一阶段优化重点不是单纯“加 KV cache”，而是把 KV cache、chunk sparsity、显存预算、SP/CP 切分和长序列 benchmark 统一设计；只有这样，模型才能从 demo 级流式生成走向真实实时服务。
