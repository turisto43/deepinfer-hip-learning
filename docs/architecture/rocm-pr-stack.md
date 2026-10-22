# ROCm PR 栈 #202–#206

## 结论先行

DeepInfer 官方仓库已经有一组面向 AMD gfx950 的 ROCm 实现。它被拆成五层，从平台接缝逐级扩展到完整模型。个人学习仓库应把它作为阅读和验证起点，而不是再提出一条从零开始的泛化 RFC。

本文核对日期为 **2026-10-22**。后续状态可能变化，PR 内容和测试数字应以最新公开页面为准。

## 依赖关系

```text
#202 平台与 TileLang HIP
  └─ #203 稠密 GEMM
       └─ #204 稀疏 attention、indexer、TopK
            └─ #205 MoE、EP、Engram、CP
                 └─ #206 文档与端到端结果
```

这是一条 stacked PR 栈。后面的 PR 在早期 PR 合并前会包含前序提交，评审时应先找到该 PR 自己的顶层提交，而不是把整个 diff 都算作新增工作。

## 各层内容

### #202：平台与 TileLang HIP

目标是把“发现 ROCm 设备、选择平台对象、构建 HIP 扩展、运行基础 TileLang 算子”这一条链打通。

主要改动包括：

- 增加 `rocm` 加速器类型；
- 通过 KFD sysfs 探测设备顺序和 NUMA 信息；
- 增加 `PlatformROCm`；
- 在 `setup.py` 中 hipify C++ 扩展；
- 增加 ROCm 构建脚本和依赖；
- 为 TileLang 0.1.15 提供 HIP 代码生成补丁。

这层决定后续所有设备功能能否进入。阅读时应先弄清 HIP 设备编号、可见设备变量、编译目标和 RCCL 后端分别在哪里生效。

### #203：稠密 GEMM

目标是用 ROCm 生态实现 DeepInfer 使用的 `deep_gemm` API。

主要路径包括：

- MXFP8/MXFP4 相关 GEMM 使用 Triton；
- BF16 GEMM 使用 hipBLASLt 或针对小 M 的 split-K kernel；
- 处理 E8M0 scale layout；
- 为 ViT 提供 CK flash-attention 和 BF16 linear 路径。

这层不仅替换 GEMM，还改变了部分激活 scale 的数据布局。后续验证不能只比较矩阵乘结果，还要覆盖 scale 编码和模型层拼装。

### #204：稀疏 attention、indexer 与 TopK

这是和 DeepSelect 最直接相关的一层。

主要改动包括：

- 使用 AITER 或 Triton 实现 sparse prefill；
- 使用 Gluon/Triton 实现 paged sparse decode；
- 实现 indexer 使用的 MQA logits；
- 在 `vendor/rocm/deep_select` 中加入 Triton radix-select TopK；
- 在 ROCm 上关闭部分融合 norm/rope/attention/cast 路径。

根据公开 PR 描述，ROCm TopK 的主要差异包括：

- 部分 FP32 全行选择仍调用 `torch.topk`；
- 公开接口注释标明 ROCm 路径不检测 NaN；
- 当前测试重点集中在 DeepInfer 使用的 BF16 indexer 和 sampler 形状；
- 正确性和性能数据来自 gfx950 环境，不能直接外推到其他 gfx 架构。

这些差异适合用独立一致性测试固定下来。参考 [TopK 契约](../labs/topk/contract.md)。

### #205：MoE、EP、Engram 与 CP

这一层覆盖分布式和多卡场景：

- FlyDSL 融合 MoE；
- 基于 PyTorch symmetric memory 的专家交换；
- Engram 的共享主机内存或 HIP-IPC 路径；
- RCCL bucket collective；
- EPLB 的 peer-memory 拷贝。

PR 描述明确当前 ROCm expert exchange 和 Engram 主要验证单节点。不能根据单节点结果推断多节点拓扑、容错和带宽表现。

### #206：文档与端到端结果

这层增加 ROCm 文档、支持硬件说明和 MI350X/MI355X 上的端到端结果。它依赖前面所有层，适合用于了解目标环境、运行命令和已知限制，但不替代个人在目标机器上的复现。

## 未合并方案中值得核查的问题

以下是阅读时形成的验证问题，不代表已经发现 bug：

1. **设备覆盖：** gfx950 之外的 MI300X、MI250 或其他 ROCm 设备能否构建和运行？
2. **接口差异：** TopK 的 NaN 行为是否应与 CUDA 行为一致，还是长期作为 ROCm 差异保留？
3. **测试跳过：** 哪些测试实际执行，哪些因为算子缺失而跳过？
4. **数据布局：** FP8/FP4 scale layout 的差异是否有跨后端参考测试？
5. **多卡边界：** 单节点 EP 结果与多节点部署之间还缺哪些验证？
6. **数值稳定性：** 1-ulp 差异是否会累积到模型输出，允许误差如何定义？
7. **构建可复现性：** TileLang 补丁、ROCm 版本、PyTorch wheel 和 AITER 版本是否能够完全固定？

## 个人参与方式

优先级应从证据价值高、改动小的任务开始：

1. 在目标硬件复现单个 TopK 用例，记录环境和精确结果；
2. 补齐接口一致性测试，先暴露差异再讨论修复；
3. 对明确差异提出最小补丁；
4. 再考虑扩展到其他 gfx 架构或完整模型。

在新的 PR 或 Issue 中提问前，先检查公开讨论是否已有答案，并附上可复现命令和版本信息。

## 固定版本

| PR | Head commit |
| --- | --- |
| #202 | `6ce17ef03fcb1d49a9042cef1815f8fb3204dd00` |
| #203 | `36a1457e4ca51ad590273e20f6479babb7d79596` |
| #204 | `6a80688d7c74e737d986f87e1522a431565b153a` |
| #205 | `6a0c37ddad3f613b81cb4385d534ea3ccb43fa07` |
| #206 | `9a0e3531f241e29b789b83fb010f485eee0efced` |

完整快照和复核命令见 [上游版本快照](../reference/upstream-snapshot.md)。

