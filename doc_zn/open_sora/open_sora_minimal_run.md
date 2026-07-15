# Open-Sora 最小运行实验记录

实验日期：2026-07-15

结论：Open-Sora 代码环境已最小跑起。官方 inference 入口可以完成分布式初始化、配置解析、dataset 构建并进入模型构建；完整视频生成没有完成，原因是仓库 `ckpts/` 为空，缺少 Open-Sora v2 11B、Hunyuan VAE、T5 和 CLIP 权重。为了验证模型代码本身不是环境阻塞，又额外构造了一个极小随机权重 Flux/MMDiT，在 GPU 上完成了一次 forward。

## 仓库与入口

仓库位置：

```bash
/data/home/sheshuchen/Open-Sora
```

官方 README 给出的 256px 推理入口：

```bash
torchrun --nproc_per_node 1 --standalone \
  scripts/diffusion/inference.py \
  configs/diffusion/inference/256px.py \
  --prompt "raining, sea"
```

实际入口文件：

- `scripts/diffusion/inference.py`
- `configs/diffusion/inference/256px.py`
- `opensora/utils/sampling.py`
- `opensora/models/mmdit/model.py`

## 独立环境

不能复用 DiT 的 `aiinfra-interview` 环境。该环境的 `torchvision 0.28.0+cu132` 在导入 `torchvision.io.video` 时会报：

```text
ModuleNotFoundError: No module named 'pytorch'
```

因此按 Open-Sora README 创建独立环境：

```bash
conda create -n opensora python=3.10 -y
```

安装 PyTorch / torchvision：

```bash
/data/home/sheshuchen/.conda/envs/opensora/bin/python \
  -m pip install torch==2.4.0 torchvision==0.19.0
```

安装项目依赖：

```bash
cd /data/home/sheshuchen/Open-Sora
/data/home/sheshuchen/.conda/envs/opensora/bin/python -m pip install -e .
```

安装 TensorNVMe：

```bash
/data/home/sheshuchen/.conda/envs/opensora/bin/python \
  -m pip install git+https://github.com/hpcaitech/TensorNVMe.git --no-build-isolation
```

安装 flash-attn 时，直接 `pip install flash-attn --no-build-isolation` 失败在跨文件系统 rename：

```text
Invalid cross-device link
```

解决方法是安装它自动推断出的 release wheel：

```bash
/data/home/sheshuchen/.conda/envs/opensora/bin/python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

TensorNVMe 的动态库路径需要显式加入：

```bash
export LD_LIBRARY_PATH=/data/home/sheshuchen/.tensornvme/lib:$LD_LIBRARY_PATH
```

## 环境验证

验证命令：

```bash
LD_LIBRARY_PATH=/data/home/sheshuchen/.tensornvme/lib:$LD_LIBRARY_PATH \
/data/home/sheshuchen/.conda/envs/opensora/bin/python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("torchvision", torchvision.__version__)
from torchvision.io.video import _check_av_available
print("torchvision video import ok")

mods = [
    "opensora.registry",
    "opensora.utils.config",
    "opensora.utils.cai",
    "opensora.utils.sampling",
    "opensora.models.mmdit.model",
    "scripts.diffusion.inference",
]
for m in mods:
    __import__(m, fromlist=["*"])
    print(m, "OK")
PY
```

关键结果：

```text
torch 2.4.0+cu121 cuda True 8
torchvision 0.19.0+cu121
torchvision video import ok
opensora.utils.cai OK
opensora.utils.sampling OK
opensora.models.mmdit.model OK
scripts.diffusion.inference OK
```

## 官方 inference 启动实验

为了减少运行成本，覆盖了 prompt、采样步数和帧数：

```bash
cd /data/home/sheshuchen/Open-Sora

LD_LIBRARY_PATH=/data/home/sheshuchen/.tensornvme/lib:$LD_LIBRARY_PATH \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
/data/home/sheshuchen/.conda/envs/opensora/bin/python \
  -m torch.distributed.run --nproc_per_node=1 --standalone \
  scripts/diffusion/inference.py \
  configs/diffusion/inference/256px.py \
  --save-dir /tmp/opensora_samples \
  --prompt "raining, sea" \
  --sampling-option.num-steps 1 \
  --sampling-option.num-frames 1 \
  --num-samples 1 \
  --num-workers 0
```

已经跑到的阶段：

```text
Distributed environment is initialized, world size: 1
Inference configuration: ...
Building dataset...
Dataset contains 1 samples.
Building models...
```

停止原因：

```text
Checkpoint not found at ./ckpts/Open_Sora_v2.safetensors
huggingface_hub.errors.HFValidationError: Repo id ... './ckpts'
```

这说明官方 inference 入口已经跑起，但当前仓库没有模型权重。`configs/diffusion/inference/256px.py` 需要：

```text
./ckpts/Open_Sora_v2.safetensors
./ckpts/hunyuan_vae.safetensors
./ckpts/google/t5-v1_1-xxl
./ckpts/openai/clip-vit-large-patch14
```

README 推荐下载：

```bash
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
```

该权重是 11B 级别，未在本次最小实验中下载。

## 极小 Flux/MMDiT forward

为了确认模型代码本身能在 GPU 上执行，构造了一个极小随机权重 Flux 模型，不加载 checkpoint，不走 T5/CLIP/VAE：

```bash
LD_LIBRARY_PATH=/data/home/sheshuchen/.tensornvme/lib:$LD_LIBRARY_PATH \
/data/home/sheshuchen/.conda/envs/opensora/bin/python - <<'PY'
import torch
from opensora.models.mmdit.model import Flux

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = Flux(
    from_pretrained="",
    device_map=device,
    torch_dtype=dtype,
    in_channels=4,
    vec_in_dim=6,
    context_in_dim=5,
    hidden_size=24,
    mlp_ratio=2.0,
    num_heads=3,
    depth=1,
    depth_single_blocks=1,
    axes_dim=[2, 2, 4],
    theta=10000,
    qkv_bias=True,
    guidance_embed=False,
    cond_embed=False,
    fused_qkv=True,
    use_liger_rope=False,
).eval()

img = torch.randn(1, 2, 4, device=device, dtype=dtype)
img_ids = torch.tensor([[[0, 0, 0], [0, 0, 1]]], device=device, dtype=dtype)
txt = torch.randn(1, 3, 5, device=device, dtype=dtype)
txt_ids = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 1, 1]]], device=device, dtype=dtype)
timesteps = torch.tensor([0.5], device=device, dtype=dtype)
y_vec = torch.randn(1, 6, device=device, dtype=dtype)

with torch.inference_mode():
    out = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=timesteps, y_vec=y_vec)

print("device", device)
print("dtype", dtype)
print("params", sum(p.numel() for p in model.parameters()))
print("out_shape", tuple(out.shape))
print("out_mean", float(out.float().mean().cpu()))
PY
```

实际输出：

```text
device cuda:0
dtype torch.bfloat16
params 32452
out_shape (1, 2, 4)
out_mean -0.03704833984375
```

## 结论

- Open-Sora 独立环境已搭好。
- 关键模块全部可导入。
- 官方 inference 脚本能启动到模型加载阶段。
- 当前不能完整生成视频的唯一直接阻塞是 `ckpts/` 为空。
- 极小 Flux/MMDiT 随机权重模型已在 GPU 上 forward 成功，证明模型执行链路可用。

要完整生成视频，下一步是下载 Open-Sora v2 权重：

```bash
cd /data/home/sheshuchen/Open-Sora
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
```

然后重新运行 256px inference 命令。
