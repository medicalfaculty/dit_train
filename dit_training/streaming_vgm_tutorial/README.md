# How to Train a Streaming VGM

这里的 VGM 指 Video Generative Model；streaming VGM 指模型不是一次性生成完整视频，而是根据已有上下文持续生成后续帧，适合实时交互、数字人、会议视频和在线世界模型等场景。

这个 mini 项目用最小代码说明 streaming VGM 的训练和采样：

- `streaming_vgm/model.py`：一个支持 causal attention 和 KV cache 的 mini 自回归视频生成模型。
- `streaming_vgm/data.py`：生成移动亮块 toy 视频，并把 latent 解码成 PGM 帧条带。
- `train.py`：teacher forcing 训练。
- `sample_stream.py`：流式采样，每次只生成下一帧 latent。

## 1. Streaming VGM 的核心问题

离线视频生成通常是：

```text
prompt -> 一次性生成 T 帧
```

Streaming VGM 更像：

```text
prompt + 历史帧 -> 生成下一帧 -> 更新历史 -> 继续生成
```

因此它必须满足：

- **因果性**：当前帧不能看未来帧。
- **低延迟**：每次生成只处理新增 token，而不是重算完整历史。
- **状态复用**：推理时复用 KV cache。
- **训练推理一致**：训练时的 mask 和推理时的可见上下文要一致。
- **长上下文管理**：历史越来越长时，需要窗口、压缩或记忆机制。

## 2. 训练目标

本项目把每一帧压成一个二维 latent 位置，用 prompt 控制移动方向。

训练数据形状：

```text
latents: [batch, frames, 2]
prompt_id: [batch]
```

训练方式是 teacher forcing：

```text
输入: prompt, BOS, latent_0, latent_1, ..., latent_{t-1}
目标: latent_0, latent_1, ..., latent_t
```

loss：

```text
MSE(pred_next_latent, target_latent)
```

真实视频模型里，`latent` 通常来自 VAE，`prompt_id` 通常换成 T5 text embeddings。

## 3. Causal Mask

Streaming VGM 训练时必须用 causal attention：

```text
token_i 只能看 token_0 ... token_i
```

否则模型会偷看未来帧，训练 loss 很低，但流式推理会崩。

代码位置：

```text
streaming_vgm/model.py -> CausalSelfAttention.forward
```

## 4. KV Cache

流式采样时，如果每生成一帧都重算完整历史，延迟会越来越高。

KV cache 的思想是：

```text
历史 token 的 K/V 缓存下来
新 token 只计算自己的 Q/K/V
attention 时 Q_new 看 K/V_cache + K/V_new
```

代码位置：

```text
StreamingVGM.prefill_prompt
StreamingVGM.stream_step
StreamingVGM.generate_stream
```

## 5. 跑通训练

```bash
cd /data/home/sheshuchen/dit_train/dit_training/streaming_vgm_tutorial
python train.py --steps 300 --frames 12
```

输出类似：

```text
step=0001 loss=0.39
step=0050 loss=0.02
...
saved=outputs/streaming_vgm.pt
```

## 6. 跑通流式采样

```bash
python sample_stream.py --ckpt outputs/streaming_vgm.pt --prompt "move right" --frames 12
```

输出：

```text
saved=outputs/stream_sample.pgm prompt='move right' frames=12
```

`outputs/stream_sample.pgm` 是横向排列的视频帧，可以用普通图片查看器打开。

## 7. 真实 Streaming VGM 框架应该怎么扩展

把这个 mini 项目扩展成真实框架时，对应替换：

- toy latent -> VAE video latent。
- prompt id -> T5 text embedding。
- 每帧一个 token -> 每帧多个 patch token。
- MSE next latent -> diffusion/flow/action-conditioned target。
- 全历史 KV cache -> sliding window / memory compression / reference frame cache。
- 单机 toy train -> FSDP/TP/SP/CP/Ring Attention 分布式训练。

## 8. 训练时最容易踩的坑

- **偷看未来**：mask 错误会让模型在训练时看到未来帧，导致推理不可用。
- **训练推理不一致**：训练用 full attention，推理用 streaming cache，会造成分布偏移。
- **延迟不可控**：不使用 KV cache 会导致生成越久越慢。
- **历史无限增长**：KV cache 不截断会让显存随生成时长线性增长。
- **数据链路污染训练**：真实训练中 VAE/T5/视频解码最好离线缓存，训练只读 latent。
- **稀疏注意力不可流式**：某些 sparse pattern 训练快，但不支持在线生成。

## 9. 一句话总结

训练 streaming VGM 的关键是：用 causal/streaming 约束训练模型预测下一个视频 latent，并在推理时用 KV cache 复用历史，让模型能够低延迟持续生成。
