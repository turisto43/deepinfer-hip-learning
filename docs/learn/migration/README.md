# ROCm 迁移学习

这条路线回答“DeepInfer 的哪些边界需要替换，现有 ROCm 方案覆盖到哪里，怎样用最小实验验证”。

## 迁移框架

DeepInfer 主分支当前公开支持 NVIDIA 和 Ascend。ROCm 相关工作以 PR 栈形式存在，阅读顺序如下：

| 层级 | 内容 | 对应 PR | 学习重点 |
| --- | --- | --- | --- |
| 1 | 平台发现与 TileLang HIP | #202 | 设备发现、平台对象、构建、Graph、RCCL |
| 2 | 稠密 GEMM | #203 | MXFP8/MXFP4、hipBLASLt、scale layout |
| 3 | 稀疏 attention、indexer、TopK | #204 | FlashMLA 替代路径、BF16 indexer、物理 KV 索引 |
| 4 | MoE、EP、Engram、CP | #205 | 专家交换、symmetric memory、单节点边界 |
| 5 | 文档与端到端结果 | #206 | 目标环境、完整模型、已知限制 |

详细状态和提交固定见 [ROCm PR 栈笔记](../../architecture/rocm-pr-stack.md)。

## 学习顺序

```text
平台接缝
  → 一个基础 kernel
  → 一个真实算子契约
  → attention / GEMM
  → 多卡通信
  → 完整模型
```

当前从 TopK 开始，因为它有清晰输入输出，同时又能覆盖可变长度、输出索引、内存布局和 Graph 捕获等真实问题。

## 迁移专题

| 序号 | 专题 | 状态 | 入口 |
| --- | --- | --- | --- |
| 01 | PR #204 TopK：BF16 indexer prefill 正确性与调用契约 | 进行中，待继续补充 | [阅读笔记](01-topk-bf16-indexer-prefill.md) |
| 02 | PR #204 TopK：decode 与候选 block | 待开始 | 后续专题 |
| 03 | PR #204 attention 与 indexer | 待开始 | 后续专题 |
| 04 | PR #202 平台与构建 | 待开始 | 后续专题 |
| 05 | PR #203 稠密 GEMM | 待开始 | 后续专题 |
| 06 | PR #205 多卡与 MoE | 待开始 | 后续专题 |

## 当前专题的完成标准

第一篇 TopK 专题需要同时补齐：

- 固定 #204 的参考提交；
- 列出 DeepInfer 的真实调用契约；
- 对照 #204 的 Python 接口、Triton kernel 和调用方；
- 记录尚未测试的项；
- 在获得 HIP 环境后补充正确性结果；
- 确认测试是否实际执行，而不是被跳过；
- 最后再记录性能数据。

## 返回

- [学习路径首页](../README.md)
- [理论框架学习](../theory/README.md)

