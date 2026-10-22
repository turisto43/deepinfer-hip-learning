# 01 从整体框架开始学习 DeepInfer

学习一个大型推理框架，先追单个 kernel 容易失去上下文。更合适的方法是先建立整体地图，再按模块深入，最后把迁移工作放回对应位置。

本篇基于 DeepInfer `main@9bd9cf2`，只梳理阅读框架，不评价尚未实际测过的性能。

## 一个最小的心智模型

可以把 DeepInfer 分成三个平面：

| 平面 | 主要问题 | 代表目录 |
| --- | --- | --- |
| 控制平面 | 请求怎样进入、排队、调度、结束 | `entrypoints/`、`engine/`、`worker/`、`csrc/engine/` |
| 计算平面 | 模型怎样执行，层和算子落在哪里 | `model_executor/`、`layers/`、`tilelang_ops/`、`vendor/` |
| 平台平面 | 设备怎样被发现，代码怎样构建和分派 | `deep_cluster/`、`platform.py`、`setup.py`、`scripts/` |

迁移后端时，三个平面都会发生变化。只替换 kernel 不够，只改设备探测也不够。

## 一次推理的大致路径

```text
用户调用
  LLM.generate
      │
      ▼
请求与引擎
  add_request → EngineArgs → LLMEngine
      │
      ▼
调度
  C++ scheduler → SchedulerOutputs → SequenceMetadata
      │
      ▼
设备执行
  Worker.execute_model_step
      ├─ ModelRunner.execute_main_model
      ├─ 模型 forward
      ├─ 算子与 kernel
      └─ ModelRunner.execute_sample
      │
      ▼
输出整理与请求完成
```

阅读任何模块时，都可以先问：

1. 它位于这条路径的哪一段？
2. 它的输入来自哪里，输出交给谁？
3. 它依赖哪一种设备能力？
4. 它是否处于 Graph 捕获、异步流水或多卡通信范围内？

## 建议的模块拆分

### A. 请求与调度

从 `LLM.generate` 进入 `LLMEngine`，再观察 C++ scheduler 如何生成 prefill/decode 的执行计划。这个模块决定每一步处理哪些请求、哪些 token 以及多少缓存资源。

阅读入口：

- [DeepInfer 系统地图](../../docs/architecture/system-map.md)
- `deepinfer/entrypoints/llm.py`
- `deepinfer/engine/llm_engine.py`
- `csrc/engine/scheduler.hpp`

### B. Worker 与模型执行

Worker 把调度结果转成模型输入，调用主模型 forward，再执行 sampler。异步调度开启时，不同步骤的 forward、sample 和后处理会形成流水线。

阅读入口：

- `deepinfer/worker/worker.py`
- `deepinfer/worker/model_runner.py`
- `deepinfer/model_executor/models/`

### C. 层与算子

模型层并不都直接包含 kernel。部分算子通过 `tilelang_ops` 分派，部分直接调用 FlashMLA、DeepGEMM、DeepSelect、DeepEP 等 vendor 包。

阅读时必须区分：

- Python 接口；
- TileLang 工厂；
- vendor API；
- 编译后的设备扩展；
- 平台 import patch。

### D. 平台与构建

DeepInfer 目前用 `accelerator_kind()` 区分 CPU、NVIDIA 和 Ascend，用 `PLATFORM` 选择算子目录和通信后端，并用 `DEEPINFER_ACCELERATOR` 选择构建目标。

阅读入口：

- [后端接缝](../../docs/architecture/backend-seams.md)
- `deep_cluster/hardware/`
- `deepinfer/platform.py`
- `setup.py`
- `scripts/build_vendor.sh`

### E. 缓存、并行与高级机制

理解主执行链后，再进入 KV cache、上下文并行、专家并行、EPLB、MTP、结构化输出和 Graph 捕获。这些模块通常横跨控制平面、计算平面和平台平面，不适合作为第一站。

## 理论框架与迁移框架

后续学习分成两条轨道：

| 轨道 | 目标 | 当前入口 |
| --- | --- | --- |
| 理论框架学习 | 理解 DeepInfer 的设计和调用链 | [理论框架学习](theory/README.md) |
| 迁移学习 | 理解 ROCm PR 栈并逐项验证算子 | [ROCm 迁移学习](migration/README.md) |

ROCm 迁移不是单独的 kernel 列表，而是沿着以下层级推进：

```text
设备发现
  → 平台对象
  → Python/C++ 扩展构建
  → TileLang 与 vendor 算子
  → attention / TopK / GEMM / MoE
  → 多卡通信
  → 模型端到端
```

DeepInfer 已有一组面向 gfx950 的 ROCm PR 栈。当前从其中的 TopK 子问题开始，是为了先建立“调用契约 → 实现差异 → 测试证据”的完整方法，再扩展到其他模块。

## 推荐阅读方法

1. 固定一个上游 commit；
2. 从调用点向上找输入，向下找实现；
3. 把接口参数写成契约；
4. 标记“已阅读”“已本地验证”“已实机验证”；
5. 一次只验证一个调用形态；
6. 将结论和失败都写回对应专题笔记。

`deepinfer-unpacked` 的模块编号可以辅助建立目录感，本仓库采用其中 M03 调度执行、M06 算子、M11 外置组件和 M12 平台工程的划分作为参考。代码事实仍以 DeepInfer 固定 commit 为准。

## 继续学习

- [理论框架学习](theory/README.md)
- [ROCm 迁移学习](migration/README.md)

## 参考

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)
- [deepinfer-unpacked](https://github.com/ZiyuanDu/deepinfer-unpacked/tree/fde5ed766717794470dbccd4d0e2a8bf9b532844)

