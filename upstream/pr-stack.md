# DeepInfer ROCm PR 栈追踪

**最后核对：** 2026-10-22

## 状态

| PR | 范围 | 参考 commit | 个人状态 |
| --- | --- | --- | --- |
| [#201](https://github.com/deepseek-ai/DeepInfer/pull/201) | 一体化 ROCm 实现 | 已关闭 | 已阅读关闭原因 |
| [#202](https://github.com/deepseek-ai/DeepInfer/pull/202) | 平台与 TileLang HIP | `6ce17ef` | 待逐文件阅读 |
| [#203](https://github.com/deepseek-ai/DeepInfer/pull/203) | 稠密 GEMM | `36a1457` | 待重点阅读 scale layout |
| [#204](https://github.com/deepseek-ai/DeepInfer/pull/204) | attention、indexer、TopK | `6a80688` | 当前实验重点 |
| [#205](https://github.com/deepseek-ai/DeepInfer/pull/205) | MoE、EP、Engram、CP | `6a0c37d` | 待阅读 |
| [#206](https://github.com/deepseek-ai/DeepInfer/pull/206) | 文档与结果 | `9a0e353` | 待逐项复核 |

#201 的关闭说明表示完整实现已拆分成 #202–#206，并称拆分后的最终代码树除 vendor 包发现方式外保持一致。

## 待核查

- 每个 PR 的实际顶层提交和独立 diff；
- gfx950 以外的设备覆盖；
- TopK NaN 行为差异；
- 测试中执行、跳过和未覆盖的数量；
- 单节点 MoE/EP 与多节点部署的边界；
- PyTorch、ROCm、Triton、AITER 和 TileLang 的版本约束；
- 合并前后文件路径和接口是否变化。

## 阅读和参与顺序

1. 先读 #202 的平台、构建和算子分派；
2. 再读 #204 的 TopK 和 attention；
3. 用最小用例验证 TopK；
4. 根据结果选择补充测试或局部修复；
5. 在具备条件后再进入 #203、#205、#206 的完整复现。

## 相关笔记

- [ROCm PR 栈架构笔记](../docs/architecture/rocm-pr-stack.md)
- [TopK 契约](../labs/topk/contract.md)
- [上游版本快照](../docs/reference/upstream-snapshot.md)

