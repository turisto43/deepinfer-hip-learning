# 2026-10-22 ROCm TopK 迁移学习

## 目标

把 TopK 学习笔记归入 ROCm 迁移板块，并锁定 PR #204 中 BF16 indexer prefill 的正确性与调用契约验证。

## 阅读与参考

- DeepInfer `main@9bd9cf2`；
- `model_executor/models/v4_modules/index_attn.py`；
- `model_executor/layers/attention_backend.py`；
- `model_executor/layers/sampler.py`；
- `model_executor/layers/sampler_utils.py`；
- `npu_ops/patch_imports.py`；
- [deepinfer-unpacked `main@fde5ed7`](https://github.com/ZiyuanDu/deepinfer-unpacked/tree/fde5ed766717794470dbccd4d0e2a8bf9b532844)。

## 已阅读

- DeepInfer 的 TopK 至少包含 indexer prefill、indexer decode 和 sampler 三类调用。
- Indexer 路径只需要索引，使用 BF16 logits、可变 `end`、`int32` 和 `-1` 填充。
- Decode 索引会在 TopK 后转换成 KV cache 物理槽位。
- Sampler 路径使用 FP32，可能需要 value，并可能按 value 排序。
- TopK 直接通过 `deep_select.interface` 导入，不经过 `tilelang_ops/ops.py` 的统一分派。
- Ascend 使用 import patch 替换 TopK；ROCm PR 使用单独的实现包。
- PR #204 的 BF16 非排序路径进入 Triton radix-select，supplied output buffer 会原样返回。
- `end`、offset、填充和升序索引的静态实现与 DeepInfer prefill 调用一致。
- `abort_when_nan_found` 被忽略，且正 NaN 与负 NaN 在 radix key 中的位置不同。

## 已本地验证

- 在固定上游源码中定位并交叉核对了上述调用点。
- 确认 `deepinfer-unpacked` 的模块表可以作为导航，但当前正文仍较少，事实判断以 DeepInfer 源码为准。
- 对照 PR #204 head `6a80688` 的 `interface.py`、`_radix.py` 和 TopK 测试，完成静态契约对照。
- 编写 BF16 indexer prefill 验证脚本，并通过 Python 语法检查。

## 已在 HIP 硬件验证

本次未进行。

## 结论

首个 HIP 实验聚焦 BF16 indexer prefill TopK，而不是先做一个通用排序 benchmark。这个路径直接进入 sparse attention，契约包含可变长度、输出复用、对齐和无效索引，覆盖面比单一 `torch.topk` 对比更接近 DeepInfer 的真实需求。

专题笔记当前仍为“进行中”，见 [PR #204 TopK：BF16 indexer prefill](../docs/learn/migration/01-topk-bf16-indexer-prefill.md)。

## 下一步

在目标 HIP 环境执行 [验证计划](../labs/topk/test-plan.md)，先跑 smoke 和代表性 prefill 形状，再检查 Graph 捕获。只有在实际执行后，才补充正确性与性能结果。

