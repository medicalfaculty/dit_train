# 视频生成训练优化入职补课与认知对齐

这份文档用于把面试信息整理成入职前应补齐的技术地图；每个概念只用一句话说明，先建立共同语言，再进入代码和系统细节。

## 一、岗位与业务认知

- **岗位主线**：这个岗位主要做视频生成模型的训练优化，核心目标是让 DiT 类大模型在真实数据、真实硬件和真实训练链路上更快、更稳、更省资源地训练。
- **模型路线**：团队模型以 DiT 为基础，并关注自回归 Diffusion 或自回归 DiT 这类更适合实时交互的视频生成路线。
- **训练阶段**：Pretrain、Post-train、SFT 和蒸馏分别对应基础能力学习、后续增强、监督对齐和采样加速。
- **当前重点**：团队当前最关注从数据处理到 Pretrain 的完整训练链路，因为这是模型规模、数据质量和训练效率共同作用的主战场。
- **产品方向**：团队目标不是传统离线视频生成工具，而是实时多模态交互和超实时视频生成。
- **竞争理解**：可灵、TikTok 等视频生成产品在大方向上是竞争对手，但团队更强调实时交互、推理优化和自研模型路线。
- **模型与产品关系**：当前阶段模型和数据能力是产品能力的基础，产品形态通常是强模型能力的外化结果。
- **自研定位**：公司主要使用自己设计和训练的模型，不是简单调用外部模型提供服务。
- **世界模型认知**：很多视频世界模型本质上仍建立在强视频生成能力之上，因此视频生成能力是理解世界模型路线的重要入口。
- **base 选择**：北京和深圳都可行，深圳更利于和直接指导者线下交流，北京则可用于更快入职和减少搬家阻塞。

## 二、训练链路必须理解的模块

- **视频解码**：视频解码是把压缩视频转成帧张量的过程，若混在训练 step 中会造成 CPU/GPU 等待、显存碎片和吞吐下降。
- **GPU 视频解码器**：GPU 自带视频解码器可以把解码从 CPU 转移到硬件专用单元，从而减少数据准备瓶颈。
- **VAE**：VAE 把像素视频压缩到 latent 空间，让 DiT 在更短、更低维的 token 序列上训练。
- **T5**：T5 在视频生成中通常作为文本 encoder，把 prompt 编成条件 token 供 DiT 使用。
- **离线缓存**：离线缓存是提前完成视频解码、VAE encode 和 T5 encode，让正式训练只读 latent 与 text embedding。
- **纯训练链路**：纯训练链路指 train step 中只包含 forward、loss、backward、optimizer 和通信，不夹杂解码或 encoder 推理。
- **远程推理服务**：远程推理服务指在训练节点外完成 VAE/T5 等预处理，再把结果传回训练节点以降低训练进程资源占用。
- **数据清洗**：数据清洗决定模型上限，因为低质量视频、错误文本和重复样本会直接污染生成能力。
- **元信息处理**：视频长度、分辨率、fps、宽高比和文本质量等元信息是后续 bucketing、packing 和采样策略的基础。
- **端到端链路**：端到端链路不是只看模型 forward，而是从数据读取到 checkpoint 保存的完整时间和资源消耗。

## 三、DiT 与视频生成基础概念

- **DiT**：DiT 是用 Transformer 替代 U-Net 的 diffusion backbone，把图像或视频 latent patch 当作 token 处理。
- **Patch token**：Patch token 是把空间或时空 latent 切块后得到的 Transformer 输入单位。
- **时空 token**：视频模型中的 token 同时包含时间、空间和通道信息，因此 token 数会随帧数和分辨率快速增长。
- **Attention**：Attention 是让每个 token 读取其他 token 信息的机制，也是 DiT 中计算量和显存压力最大的模块之一。
- **Dense Attention**：Dense Attention 让每个 token 看所有 token，计算复杂度约为 token 数平方。
- **Sparse Attention**：Sparse Attention 只让 token 看部分邻域、关键帧或规则块，以牺牲少量表达换取大幅计算下降。
- **流式推理**：流式推理要求模型能边接收上下文边生成后续内容，因此模型结构和 attention mask 必须支持在线生成。
- **自回归 DiT**：自回归 DiT 按时间或 token 顺序逐步生成，天然更贴近实时交互，但训练和缓存设计更复杂。
- **Diffusion**：Diffusion 通过学习从噪声到数据的反向过程来生成样本，通常需要多步采样。
- **Flow Matching**：Flow Matching 学习从噪声分布流向数据分布的速度场，是近年生成模型训练的重要路线。
- **噪声预测**：噪声预测训练模型估计加到样本中的噪声，是 DDPM/DiT 训练中最常见的目标之一。
- **蒸馏**：蒸馏用多步或大模型 teacher 指导少步或小模型 student，从而降低推理成本。

## 四、训练优化核心问题

- **显存模型**：训练显存主要由参数、梯度、optimizer state、activation、通信 buffer 和临时 workspace 组成。
- **Activation**：Activation 是前向中保存给反向使用的中间结果，通常随 batch、序列长度和模型深度快速增长。
- **Optimizer State**：AdamW 等优化器会保存一阶和二阶动量，显存可能达到参数量的数倍。
- **显存碎片**：显存碎片来自不同大小临时张量反复申请释放，会导致明明总显存足够却无法分配大块连续空间。
- **Activation Offload**：Activation offload 把激活搬到 CPU 或 NVMe 省显存，但会用数据搬运时间换显存。
- **Optimizer Offload**：Optimizer offload 把优化器状态搬出 GPU，能省显存但通常降低 step 吞吐。
- **Gradient Checkpointing**：重计算用额外 forward 计算换 activation 显存，是大模型训练常用折中。
- **低精度训练**：BF16、FP8、FP4 等低精度能提升吞吐和降低显存，但必须验证稳定性和最终收敛。
- **动态 Packing**：动态 packing 按 token budget 组合不同长度样本，使每个 GPU 或 batch 的有效 token 数更均衡。
- **Bucketing**：Bucketing 按分辨率、帧数或宽高比分桶，减少 padding 浪费并稳定每步计算量。
- **负载均衡**：负载均衡要求不同 GPU 在同一步处理相近计算量，否则最慢 GPU 会拖慢整个 step。
- **Step Time**：Step time 是单次训练迭代耗时，是定位训练瓶颈最直接的指标之一。
- **吞吐量**：训练吞吐量表示单位时间处理多少样本、视频或 token，比单看 MFU 更贴近业务效率。
- **MFU**：MFU 是实际模型计算量与硬件理论峰值算力之比，用来衡量算力利用率但不能单独代表端到端效率。

## 五、并行与通信需要补齐的概念

- **Data Parallel**：数据并行让每张卡处理不同 batch，并在反向后同步梯度。
- **FSDP / ZeRO**：FSDP 或 ZeRO 把参数、梯度和优化器状态切到不同 GPU 上，以降低单卡显存。
- **Tensor Parallel**：Tensor Parallel 把单层矩阵或 attention head 切到多张卡上，以支撑更大层宽。
- **Sequence Parallel**：Sequence Parallel 按序列维切分 activation，让 LayerNorm/MLP 等逐 token 计算分摊到多卡。
- **Context Parallel**：Context Parallel 按序列维切分上下文，让本地 query 通过通信看到全局 key/value。
- **Pipeline Parallel**：Pipeline Parallel 把模型层切到不同 GPU 上，以流水线方式处理 micro-batch。
- **Expert Parallel**：Expert Parallel 用于 MoE，把不同 expert 分布到不同设备上并通过路由通信连接。
- **AllReduce**：AllReduce 把各卡张量求和或求平均后发回所有卡，常用于梯度同步。
- **AllGather**：AllGather 把各卡分片拼成完整张量发给所有卡，常用于 FSDP 参数聚合或 CP 获取全局 K/V。
- **ReduceScatter**：ReduceScatter 先归约再切片发回各卡，常用于 FSDP 梯度分片。
- **P2P 通信**：P2P 通信是 GPU 间点对点发送数据，Ring Attention 和 pipeline 都经常依赖它。
- **NVLink**：NVLink 是 GPU 间高速互联，决定单机多卡通信带宽和延迟上限。
- **RDMA**：RDMA 允许跨机器绕过 CPU 进行低延迟高带宽数据传输，是多机训练的重要通信基础。
- **NCCL**：NCCL 是 NVIDIA GPU 集体通信库，训练框架中的多卡通信通常最终落到 NCCL。

## 六、硬件与系统方向

- **Hopper**：Hopper 是 H100/H200 所属架构，需要重点理解 Tensor Core、HBM、NVLink 和 Transformer Engine。
- **Blackwell**：Blackwell 是 B200/GB200/GB300 相关新架构，需要关注更强低精度能力、内存层级和通信拓扑。
- **GB300**：GB300 资源意味着团队可以研究最新 GPU 的内存、分页、通信和算子执行特性来挖掘性能。
- **HBM**：HBM 是 GPU 高带宽显存，训练中很多算子并非算力瓶颈而是 HBM 带宽瓶颈。
- **显存分页**：显存分页和虚拟内存机制会影响大模型训练中的分配、迁移和碎片行为。
- **Kernel**：Kernel 是 GPU 上执行的基本计算程序，算子融合和自定义 kernel 是性能优化的重要手段。
- **算子融合**：算子融合把多个小操作合成一个 kernel，减少读写 HBM 和 launch overhead。
- **编译优化**：编译系统可以通过图优化、kernel 生成和调度搜索，把模型结构转化为更高效的执行计划。
- **Profiling**：Profiling 是用数据定位瓶颈，先量化时间、显存、通信和 kernel 占比，再决定优化方向。
- **Nsight Systems**：Nsight Systems 用于看端到端 timeline，适合定位 CPU/GPU 等待和通信重叠问题。
- **Nsight Compute**：Nsight Compute 用于看单个 kernel 的带宽、占用率和 Tensor Core 利用情况。
- **torch profiler**：torch profiler 适合从 PyTorch 层快速定位算子耗时、内存和调用栈。

## 七、算法与系统协同

- **训练推理协同**：工业模型结构必须同时考虑训练效率、推理效率、服务成本和产品形态。
- **稀疏注意力选择**：稀疏 attention 不只看训练加速，还要看是否支持流式推理和最终产品交互模式。
- **低精度协同**：推理可用的 FP8/FP4 方法未必能直接用于训练，因为训练还要保证梯度稳定和收敛质量。
- **模型结构约束**：模型结构的任何优化都要同时评估精度、吞吐、显存、通信和上线可部署性。
- **数据与系统协同**：数据长度分布、分辨率分布和采样策略会直接决定系统负载是否均衡。
- **算法与编译协同**：稀疏 attention、动态 shape 和 packing 只有与 kernel/编译系统结合，才能真正转化为吞吐收益。

## 八、入职后建议补课顺序

- **第一周目标**：先跑通完整训练链路，明确数据从视频文件到 latent、text embedding、DiT loss 和 checkpoint 的路径。
- **第二步目标**：学习显存构成和 profiling 工具，能解释一次 step 中时间和显存花在哪里。
- **第三步目标**：补齐 FSDP、TP、SP、CP、PP 和 NCCL collective 的基本原理。
- **第四步目标**：理解动态 packing、bucketing 和长短视频负载均衡问题。
- **第五步目标**：深入 DiT attention，比较 dense、sparse、ring 和 streaming attention 的训练/推理约束。
- **第六步目标**：学习 H200、B200、GB300 等硬件差异，建立硬件能力到训练方案的映射。
- **第七步目标**：读 Open-Sora、FastVideo、Diffusers 等项目，但要以本团队模型和硬件的 profiling 结果为准。
- **长期目标**：逐步从能改训练脚本，成长到能设计训练框架、并行策略、数据链路和模型结构约束。

## 九、面试与沟通认知

- **长期意愿**：团队不希望候选人只做短期暑期实习，因此需要明确表达长期投入和持续学习意愿。
- **学习速度**：训练优化高度依赖真实硬件和真实任务，入职后快速学习和快速实验比纸面知识更重要。
- **沟通方式**：团队时间安排灵活但要求及时响应，遇到性能问题要用数据和 profiling 结果沟通。
- **方案态度**：行业方案可以参考但不能照搬，因为模型结构、硬件平台、数据分布和产品目标都不同。
- **实习价值**：前沿训练优化很难靠个人小实验完全复现，进入有真实模型、数据和新硬件的团队本身就是最快学习路径。

## 十、入职前最该会说清楚的十句话

1. 我知道这个岗位不是单纯写模型，而是围绕视频 DiT 的端到端训练链路做系统化优化。
2. 我理解训练链路要尽量纯粹，视频解码、VAE 和 T5 最好前置或服务化，避免污染 train step。
3. 我理解视频样本长度变化会造成 token 数差异，因此需要 bucketing 和 dynamic packing。
4. 我知道 DiT 的 attention 是核心计算瓶颈，Sparse Attention 的价值在于减少每个 query 看的 key 数。
5. 我知道并行策略不是越多越好，而是要根据显存、通信、计算和模型结构做组合。
6. 我知道 MFU 只是指标之一，最终还要看 token 吞吐、step time 和端到端训练时间。
7. 我知道训练优化和推理优化需要协同，因为模型最终要服务实时交互产品。
8. 我知道低精度、offload 和重计算都是显存速度折中，不能只看单点收益。
9. 我知道硬件优化不是等 NVIDIA 改硬件，而是先充分理解和利用现有 GPU 的底层能力。
10. 我愿意长期投入这个方向，并能在真实任务中快速补齐 profiling、并行、通信和 kernel 优化能力。
