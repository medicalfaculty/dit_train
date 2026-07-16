# Mini 自回归视频 DiT 训练框架

这个目录是一个教学版视频生成训练框架，参考 Open-Sora、FastVideo、Diffusers 这类项目的组织思想，但代码只保留最小闭环：离线 T5/VAE 缓存、自回归 DiT 训练、动态 packing、稀疏 attention、采样解码。

它不是为了生成高质量视频，而是为了让你学完后能自己写出完整框架。

## 目录结构

- `prepare_cache.py`：离线阶段。模拟视频解码、T5 编码、VAE 编码，把结果保存为缓存。
- `train.py`：纯训练阶段。只读取 text token 和 video latent，不再做视频解码/T5/VAE 推理。
- `sample.py`：采样阶段。用 prompt 生成 latent token，再用 VAE 解码成视频帧条带。
- `mini_video_ar_dit/toy_modules.py`：ToyT5Encoder、ToyVideoVAE、合成视频和图片保存。
- `mini_video_ar_dit/data.py`：缓存读取、动态 packing、batch collate。
- `mini_video_ar_dit/model.py`：自回归视频 DiT。
- `mini_video_ar_dit/attention.py`：dense attention 与 sparse attention 的教学实现。

## 安装

```bash
cd /data/home/sheshuchen/dit_train/dit_training/mini_video_ar_dit
python -m pip install -r requirements.txt
```

## 跑通全链路

先做离线缓存：

```bash
python prepare_cache.py --num-samples 32 --out outputs/cache.pt
```

再训练：

```bash
python train.py --cache outputs/cache.pt --steps 30 --attention dense
```

然后采样：

```bash
python sample.py --ckpt outputs/ar_video_dit.pt --prompt "moving square right" --frames 4
```

输出 `outputs/sample_strip.pgm`，它是一条横向排列的视频帧。

## 为什么训练链路要纯粹

真实视频生成训练里，正式训练前常常还要做视频解码、VAE encode、T5 encode。如果这些操作混在 `train_step` 里，会带来：

- 显存碎片。
- encoder/VAE 占用额外显存。
- 数据处理和 GPU 训练互相等待。
- 最后被迫 offload activation 或 optimizer，训练性能下降。

本教程把这些都放到 `prepare_cache.py`：

```text
raw video + prompt
  -> ToyT5Encoder
  -> ToyVideoVAE
  -> outputs/cache.pt
```

训练时只做：

```text
load cached text/latents -> ARVideoDiT -> loss -> optimizer
```

这就是“纯训练链路”的最小形态。

## 动态 Packing

视频长度不同，latent token 数也不同。固定 batch size 会导致 padding 浪费和 GPU 负载不均。

`data.py` 里的 `make_packs` 使用 token budget：

```text
pack 内所有视频 token 总数 <= token_budget
```

长视频少放几个，短视频多放几个。这样每个 step 的 token 数更接近，训练负载更均衡。

## 自回归视频 DiT

本教程的模型不是扩散去噪，而是自回归预测下一个 VAE latent token：

```text
text tokens + BOS + latent_0 + ... + latent_{n-1}
      -> predict latent_0 + ... + latent_n
```

训练 loss：

```text
MSE(predicted_next_latent, target_latent)
```

真实项目里可以把 toy latent 换成 VAE latent，把 ToyT5Encoder 换成 T5，把模型规模放大，并加入并行训练。

## Sparse Attention

DiT 里的 dense attention 计算量很大，复杂度约为 `O(N^2)`。如果视频 token 很长，attention 会成为主要瓶颈。

本教程支持：

```bash
python train.py --attention sparse
```

教学版 sparse attention 的策略是：

- 文本 prefix 对所有 latent token 可见。
- latent token 只看最近 `sparse_window` 个历史 latent。

这不等同于工业级 block sparse kernel，但它清楚展示了稀疏化的本质：减少每个 query 需要看的 key 数量，从而减少 attention 计算。

## 继续扩展方向

学完这个目录后，可以按下面顺序升级成真实框架：

1. 用真实视频 reader 替换 `make_toy_video`。
2. 用真实 VAE 替换 `ToyVideoVAE`，并离线保存 latent。
3. 用 T5 替换 `ToyT5Encoder`，并离线保存 text embedding。
4. 把 `make_packs` 扩展成多 worker / 多 GPU 的 token-balanced sampler。
5. 把 `attention.py` 替换成 FlashAttention、Ring Attention 或 block sparse kernel。
6. 加入 FSDP/TP/SP/CP、checkpoint、EMA、日志和恢复训练。
