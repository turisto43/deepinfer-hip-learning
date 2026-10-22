# DeepSelect TopK 实验

## 目标

从 DeepSelect TopK 的接口契约出发，建立一套最小一致性测试，用于比较官方 CUDA/Ascend 实现和 DeepInfer ROCm PR 中的实现，并逐步判断能否在其他 HIP 设备上运行。

## 为什么选择 TopK

- 输入输出定义相对清楚；
- 不需要先加载完整大模型；
- 同时涉及数据类型、排序、索引、内存布局和边界条件；
- 可以在后续 attention/indexer 中复用测试思路；
- 现有 ROCm PR 已提供一个 Triton radix-select 实现，适合独立审查和验证。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 固定上游版本 | 已完成 |
| 整理接口契约 | 进行中 |
| PR #204 静态实现对照 | 已完成 |
| 设计一致性用例 | 已完成 |
| 准备可执行验证脚本 | 已完成 |
| 准备 ROCm 环境 | 未开始 |
| 运行正确性测试 | 未开始 |
| 运行性能测试 | 未开始 |
| 提交上游改进或测试 | 未开始 |

## 文件

- [接口契约](contract.md)：输入、输出、排序和边界行为。
- [首个验证计划](test-plan.md)：PR #204 BF16 indexer prefill 的固定用例和通过标准。
- [验证脚本](check_pr204_prefill.py)：在有 ROCm 环境时执行正确性和 Graph 检查。
- [环境记录](environment.md)：实际测试环境与当前限制。
- [结果记录](results.md)：正确性、性能和失败结果。

## 相关上游

- [DeepSelect](https://github.com/deepseek-ai/DeepSelect)
- [DeepSelect 固定版本](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18)
- [DeepInfer ROCm PR #204](https://github.com/deepseek-ai/DeepInfer/pull/204)

