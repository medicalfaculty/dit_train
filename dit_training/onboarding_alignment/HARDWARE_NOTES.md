# 硬件问题补课：H200、B200、GB300、NVLink、RDMA 与 Profiling

这份文档解释入职后会频繁听到的硬件问题，目标不是背规格，而是知道这些硬件特性如何影响视频 DiT 训练优化。

## 1. 为什么训练优化必须懂硬件

训练优化不是单纯改 PyTorch 代码，而是在模型计算、显存容量、HBM 带宽、GPU 间通信、跨机网络、CPU 数据链路和存储链路之间做系统平衡。

一个训练 step 变慢，可能不是模型 FLOPs 不够快，而是数据解码、显存碎片、activation 保存、NCCL 通信、HBM 读写或 kernel launch 中的任意一环拖慢。

## 2. Hopper：H100/H200 应该关注什么

H200 属于 Hopper 架构，最重要的变化是 HBM3e 显存容量和带宽相比 H100 明显提升。

NVIDIA 官方 H200 页面给出的核心信息是：H200 提供 141GB HBM3e，显存带宽 4.8TB/s。  
来源：https://www.nvidia.com/en-us/data-center/h200/

对训练优化来说，H200 的意义是：

- 更大显存让更长视频、更大 batch 或更少 offload 成为可能。
- 更高 HBM 带宽让 memory-bound 算子更容易受益。
- 计算架构仍是 Hopper，因此很多 H100 优化经验可以迁移，但显存容量约束不同。
- 如果原方案在 H100 上必须 activation offload，H200 上可能可以改成 checkpointing 或直接保留 activation。

一句话：H200 主要是 Hopper 体系下的“更大更快显存版本”，适合先把显存瓶颈和 HBM 带宽瓶颈分清楚。

## 3. Blackwell：B200 应该关注什么

B200 属于 Blackwell 架构，重点不只是显存增大，还包括新的 Tensor Core、FP4/NVFP4 能力和第五代 NVLink。

NVIDIA DGX B200 文档说明 DGX B200 使用 8 张 B200 GPU，总 GPU 显存为 1440GB，也就是每张约 180GB 级别；其他公开资料通常描述 B200 为 192GB HBM3e SKU，具体以机型为准。  
来源：https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html

NVIDIA Blackwell 架构页面强调 NVLink Switch 可在 72-GPU NVLink domain 中提供 130TB/s GPU 带宽，并支持更高效的聚合通信。  
来源：https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/

对训练优化来说，B200 的意义是：

- FP8/FP4/NVFP4 能力更强，但训练是否能用要看收敛稳定性。
- 单卡显存更大，长视频 token 和更大 micro-batch 更容易放下。
- NVLink 带宽更高，FSDP/TP/CP/Ring Attention 的通信代价模型会变化。
- Blackwell 的 kernel、Tensor Core pipeline 和内存层级可能改变 Hopper 上的最优实现。

一句话：B200 不是简单的“更快 H200”，而是会改变低精度、通信和 kernel 设计空间的新平台。

## 4. GB300 / NVL72 应该关注什么

GB300 NVL72 是 rack-scale 系统，核心是把大量 Blackwell Ultra GPU 通过 NVLink/NVSwitch 组成一个更大的 GPU domain。

NVIDIA NVL72 AI Factory 文档说明 Blackwell 第五代 NVLink 每 GPU 可达 1800GB/s，目标是让更大 NVLink domain 支撑大模型推理和训练。  
来源：https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html

HPE 的 GB300 NVL72 页面描述其包含 72 个 Blackwell Ultra GPU、36 个 Grace CPU、最多约 20TB HBM3e 和约 569TB/s 总显存带宽，并配有 ConnectX-8 SuperNIC。  
来源：https://buy.hpe.com/sa/en/compute/rack-scale-system/nvidia-nvl-system/nvidia-gb300-nvl72-by-hpe/p/1014890105

Supermicro GB300 NVL72 datasheet 描述每 GPU 通过 1.8TB/s NVLink 互连，72 GPU 可形成约 21TB HBM3e 池，并支持 800Gb/s 级别网络。  
来源：https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB300_NVL72.pdf

对训练优化来说，GB300 的意义是：

- 单 rack 内通信能力极强，模型并行可以更激进。
- 大显存池让更大模型或更长上下文可行，但软件是否能高效利用仍是关键。
- 通信拓扑变复杂，rank mapping、parallel group 划分和 NCCL 拓扑感知更重要。
- 液冷、功耗、CPU-GPU 协同、网络 fabric 都会影响长期稳定训练。

一句话：GB300 不是单卡问题，而是一个 rack-scale 训练/推理系统问题。

## 5. HBM、显存容量和显存带宽

HBM 容量决定能放下多少参数、activation、optimizer state 和 batch；HBM 带宽决定 memory-bound 算子每秒能搬多少数据。

常见误区是只看 TFLOPS：如果算子反复读写大 tensor、计算密度低，那么瓶颈往往是 HBM 而不是 Tensor Core。

视频 DiT 中容易受 HBM 影响的部分：

- LayerNorm、RMSNorm。
- Dropout、残差加法。
- 非融合的小算子链。
- Attention score/mask/softmax 中间结果。
- optimizer update。
- 大量 padding token 造成的无效读写。

一句话：显存容量解决“放不放得下”，HBM 带宽解决“搬得快不快”。

## 6. Tensor Core 与低精度

Tensor Core 是 GEMM/attention 这类密集矩阵计算的核心加速单元，BF16、FP8、FP4 等低精度能力会直接影响训练和推理吞吐。

训练中使用低精度要额外关注：

- loss scale 或数值范围是否稳定。
- attention logits、norm、optimizer state 是否需要更高精度。
- FP8/FP4 是只用于 forward，还是 backward 和 optimizer 也参与。
- 精度下降是否影响 Pretrain loss、SFT 效果和蒸馏质量。

一句话：低精度不是简单开关，而是“吞吐提升”和“收敛风险”的系统实验。

## 7. NVLink、NVSwitch 和单机多卡通信

NVLink 是 GPU 间高速互联，NVSwitch 负责把多张 GPU 连接成更大的高带宽通信域。

它主要影响：

- FSDP all-gather 参数速度。
- TP 中每层 all-reduce / reduce-scatter 速度。
- CP 中 all-gather K/V 或 ring 传递速度。
- Pipeline 或 expert parallel 的 P2P 传输速度。
- 多卡 checkpoint save/load 的数据搬运效率。

如果 NVLink 拓扑没有用好，可能出现某些 rank 通信绕路，导致单个并行组拖慢整体 step。

一句话：NVLink/NVSwitch 决定单机或单 rack 内模型并行到底能多激进。

## 8. RDMA、InfiniBand、RoCE 和跨机通信

RDMA 让网卡直接读写远端机器内存，减少 CPU 参与，是多机训练降低延迟和提高带宽的重要技术。

跨机训练中 RDMA 影响：

- 多机 data parallel 梯度同步。
- 跨机 FSDP 参数/梯度分片通信。
- 跨机 checkpoint、数据服务、远程 VAE/T5 推理结果传输。
- 多节点 pipeline 或 expert parallel 的 P2P 数据交换。

需要关注的指标：

- 网络带宽是否匹配 GPU 消费速度。
- P99 latency 是否稳定。
- RDMA 是否真正启用，而不是退化到 TCP。
- NCCL 是否选择了预期网卡和拓扑。
- 多租户或交换机拥塞是否造成 step time 抖动。

一句话：单机内看 NVLink，跨机扩展看 RDMA/IB/RoCE 和 NCCL 拓扑。

## 9. GPU 视频解码器

GPU 视频解码器是独立于 Tensor Core 的硬件单元，可用于把 H.264/H.265/AV1 等压缩视频解码成帧。

在视频训练中它的价值是：

- 减少 CPU 解码压力。
- 降低数据 loader 等待 GPU 的概率。
- 让视频解码与模型训练部分重叠。
- 避免把大量 raw frame 长时间堆在 CPU 内存里。

但它也会带来工程问题：

- 解码后的帧是否直接进入 GPU pipeline。
- 解码和 VAE encode 是否会争抢 GPU 显存。
- 解码进程是否与训练进程隔离。
- 是否需要远程解码或远程 VAE/T5 服务。

一句话：GPU 解码器能解决数据入口瓶颈，但必须和训练显存生命周期隔离好。

## 10. 显存分页、虚拟内存和碎片

现代 GPU 支持复杂的虚拟内存和分页机制，但深度学习训练仍然可能因为大 tensor 反复申请释放产生碎片和抖动。

容易产生碎片的场景：

- 训练 step 中混入 VAE/T5 推理。
- 动态 shape 变化很大。
- 不同长度视频导致 attention workspace 大小频繁变化。
- checkpoint、采样验证和训练共用同一进程。
- 频繁创建销毁临时 buffer。

优化方向：

- 预处理和训练进程拆分。
- 固定 bucket shape。
- 复用 buffer。
- 避免在训练 step 中创建大临时对象。
- 用 profiler 和 memory snapshot 确认峰值来源。

一句话：显存碎片不是玄学，通常来自动态 shape、大临时张量和不纯训练链路。

## 11. Kernel、算子融合和编译系统

Kernel 是 GPU 执行的基本程序，多个小算子如果分开执行，会产生多次 HBM 读写和 kernel launch overhead。

算子融合的价值：

- 减少中间 tensor 写回 HBM。
- 降低 kernel launch 数量。
- 提高数据局部性。
- 让低精度和特定布局更容易发挥效果。

视频 DiT 中值得关注的融合：

- bias + activation。
- dropout + residual。
- norm + projection。
- attention softmax 相关融合。
- optimizer update。

一句话：当模型已经很大时，kernel 级别的小浪费会被 token 数和层数放大成明显吞吐损失。

## 12. Profiling 应该看什么

硬件 profiling 不是先猜瓶颈，而是从 timeline 和指标中确定瓶颈类型。

建议先回答这些问题：

- GPU 是否长时间空闲。
- DataLoader 是否跟不上。
- VAE/T5 是否混入训练 step。
- NCCL 通信是否阻塞计算。
- HBM 带宽是否打满。
- Tensor Core 利用率是否低。
- 是否有大量小 kernel。
- step time 是否随视频长度剧烈波动。
- 显存峰值来自 activation、optimizer 还是临时 workspace。

常用工具：

- `torch.profiler`：从 PyTorch 视角看算子、显存和调用栈。
- Nsight Systems：看 CPU/GPU/NCCL/DataLoader 的端到端 timeline。
- Nsight Compute：看单个 kernel 的带宽、占用率和 Tensor Core 使用情况。
- NCCL logs：看通信算法、网卡、拓扑和异常退化。

一句话：先用 profiling 把瓶颈归类，再决定是改数据、改并行、改 kernel 还是改模型。

## 13. 面向入职的硬件学习清单

- 能说清 H200 和 B200/GB300 在显存、低精度和互联上的差别。
- 能解释为什么 HBM 带宽会限制 LayerNorm、optimizer 和非融合小算子。
- 能解释 NVLink 和 RDMA 分别解决哪一级通信问题。
- 能看懂 NCCL all-reduce、all-gather、reduce-scatter 和 P2P 在 timeline 中的位置。
- 能用 torch profiler 定位一次 step 的耗时和显存峰值。
- 能用 Nsight Systems 看 GPU 空洞、通信阻塞和数据链路等待。
- 能把硬件特性映射到训练策略，例如 FSDP、TP、SP、CP、offload、packing 和 sparse attention。

## 14. 资料来源

- NVIDIA H200 官方页面：https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA DGX B200 用户文档：https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html
- NVIDIA Blackwell 架构页面：https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- NVIDIA NVL72 AI Factory Components：https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html
- HPE GB300 NVL72 页面：https://buy.hpe.com/sa/en/compute/rack-scale-system/nvidia-nvl-system/nvidia-gb300-nvl72-by-hpe/p/1014890105
- Supermicro GB300 NVL72 Datasheet：https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB300_NVL72.pdf
