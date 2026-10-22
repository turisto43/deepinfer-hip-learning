# 2026-10-22 TopK 调用路径

## 目标

弄清 DeepInfer 中 TopK 的真实调用位置，以及 HIP 实现首先需要兼容哪一条路径。

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

## 已本地验证

- 在固定上游源码中定位并交叉核对了上述调用点。
- 确认 `deepinfer-unpacked` 的模块表可以作为导航，但当前正文仍较少，事实判断以 DeepInfer 源码为准。

## 已在 HIP 硬件验证

本次未进行。

## 结论

首个 HIP 实验应聚焦 BF16 indexer prefill TopK，而不是先做一个通用排序 benchmark。这个路径直接进入 sparse attention，契约包含可变长度、输出复用、对齐和无效索引，覆盖面比单一 `torch.topk` 对比更接近 DeepInfer 的真实需求。

完整笔记见 [TopK 调用路径](../docs/architecture/topk-call-paths.md)。

## 下一步

把 indexer prefill、indexer decode、candidate block 和 sampler 四条调用形态加入测试矩阵，然后逐项对照 ROCm PR #204。

