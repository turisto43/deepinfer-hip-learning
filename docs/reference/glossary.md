# 术语表

本表只收录当前阅读中最常见的词。定义以 DeepInfer 和 ROCm 语境为主，遇到其他项目时仍应回到对应文档确认。

## HIP

AMD 提供的 GPU 编程接口，源码风格接近 CUDA。HIP 代码可以通过 HIP runtime 在支持的 AMD 设备上运行，部分项目也使用 HIP 作为跨厂商抽象层。语法相似不代表所有 CUDA 特性、库和性能模型都能直接等价。

## ROCm

AMD 的开放 GPU 计算软件栈，包括编译器、运行时、数学库、通信库、调试工具和性能分析工具。HIP 是其中面向 GPU kernel 和运行时的主要编程接口之一。

## gfx

AMD GPU 架构标识。例如 gfx950 对应 MI350X/MI355X。wavefront 大小、LDS、矩阵指令、FP8/FP4 能力和编译器支持都可能随 gfx 架构变化。

## TopK

从每行输入中选出最大的 K 个元素。高性能实现通常需要处理：

- 是否返回 value；
- 是否按 value 排序；
- 是否按 index 排序；
- 可变行长度；
- 索引类型和输出 stride；
- NaN、重复值和不足 K 个有效元素。

## DSA

DeepSeek Sparse Attention。它不是普通 dense attention 的同义替换；需要结合 indexer、TopK 索引和 sparse kernel 一起理解。

## EP

Expert Parallelism，专家并行。MoE 模型把专家分布到不同设备，token 需要通过 dispatch/combine 在设备间交换。

## EPLB

Expert Parallel Load Balancing，专家并行负载均衡。根据各专家收到的 token 数调整专家放置或复制，以缓解不同 rank 负载不均。

## TileLang

一种面向高性能 kernel 的 DSL，用 Python 描述 tile 级计算并生成目标设备代码。DeepInfer 的 `tilelang_ops` 通过平台后端目录选择具体实现。

## Triton

用 Python 风格编写 GPU kernel 的编译器工具链。ROCm 也有 Triton 后端，但可用的指令、调优参数和编译行为取决于版本与目标架构。

## SFA / Scale Factor

量化矩阵使用的 scale factor。FP8/FP4 路径不仅要匹配数值格式，还要匹配 scale 的 block 形状、编码类型和内存布局。

## LDS

Local Data Share，AMD GPU 上可被同一 workgroup 共享的低延迟片上内存。可类比 CUDA shared memory，但容量、bank 和同步语义需要按具体架构核查。

## RCCL

ROCm Collective Communications Library，AMD 平台的集合通信库。DeepInfer 的 ROCm PR 使用 `nccl` 作为 PyTorch 通信后端的名字，但底层依赖 ROCm 侧的 RCCL 实现。

## CUDA Graph / HIP Graph

把一组设备操作捕获成图并重复提交，以减少 CPU 启动开销。迁移时需要关注动态形状、内存地址稳定性、跨流依赖和捕获期间禁止的 API。

## wavefront

AMD GPU 的执行基本单位，类似 NVIDIA warp，但宽度通常不同。依赖 `warpSize`、lane shuffle 或 warp 内归约的代码不能假设两边完全一致。

