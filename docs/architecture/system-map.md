# DeepInfer 系统地图

本文基于 DeepInfer `main@9bd9cf2` 的目录、README 和关键入口源码整理。目标不是复述每个文件，而是先建立一张能用于阅读和定位问题的地图。

## 一句话理解

DeepInfer 是一个面向 DeepSeek 模型的高性能离线推理框架。Python 层负责配置、模型结构、调度和算子入口；C++ 扩展负责调度器等紧耦合组件；设备算子由 TileLang 以及 DeepGEMM、DeepEP、DeepSelect、FlashMLA 等 vendor 包提供。

## 分层结构

```text
examples/
  offline_inference.py
        │
        ▼
deepinfer/entrypoints/llm.py
        │
        ▼
deepinfer/engine/llm_engine.py
  ├─ Python 调度与请求生命周期
  ├─ csrc/engine/ 中的 C++ 引擎扩展
  └─ deepinfer/worker/ 中的设备工作进程
        │
        ├───────────────┬──────────────────┐
        ▼               ▼                  ▼
deep_cluster/       model_executor/      distributed/
硬件发现与拓扑      模型、层、算子调度    通信与共享内存
        │               │
        │               ├─ layers/tilelang_ops/
        │               ├─ models/
        │               └─ vendor 算子 API
        │
        ▼
设备运行时：NVIDIA CUDA 或 Huawei Ascend
```

ROCm 是个人仓库正在研究的目标后端，不在上述固定版本的官方支持列表中。

## 主要目录

| 目录 | 职责 | 阅读时关注 |
| --- | --- | --- |
| `deepinfer/entrypoints/` | 面向用户的调用入口 | 参数如何进入引擎 |
| `deepinfer/engine/` | 请求、调度和输出管理 | 调度边界、批处理、生命周期 |
| `deepinfer/worker/` | 设备侧运行与模型执行 | 哪些状态依赖具体硬件 |
| `deepinfer/model_executor/` | 模型、层和算子入口 | 算子接口、平台分支、数据布局 |
| `deep_cluster/hardware/` | GPU/NPU 发现和 NUMA 信息 | 设备编号与可见设备变量 |
| `deepinfer/distributed/` | 多进程通信辅助 | 通信后端、共享内存 |
| `csrc/engine/` | C++ 调度与引擎扩展 | Python/C++ 边界、线程与锁 |
| `csrc/graph_timer/` | 图计时与设备同步 | CUDA 与 Ascend 的差异 |
| `vendor/` | 上游高性能算子 | 固定版本、构建参数、设备能力 |
| `tests/` | 测试和平台标记 | `ci`、`ci_npu`、`gpu` 覆盖范围 |

## 一次离线推理的阅读路径

建议沿着以下顺序阅读，而不是一开始进入 kernel 细节：

1. **入口和配置：** 从 `examples/offline_inference.py` 进入 `LLM`，确认模型路径、设备可见性和并行配置。
2. **引擎与调度：** 阅读 `llm_engine.py`，找到请求排队、prefill/decode 和异步调度的边界。
3. **模型执行：** 进入 `model_executor`，先看模型如何被加载和拆分成 layer。
4. **层与算子：** 在 `layers/` 中找到 attention、linear、sampler 等入口，再追到 TileLang 或 vendor API。
5. **设备边界：** 对每个算子标记它依赖的 CUDA、Ascend 或通用 Python 能力。
6. **构建边界：** 回到 `setup.py`、`build.sh` 和 `scripts/build_vendor.sh`，确认代码如何被编译为当前平台产物。

## 三个关键边界

### Python 与 C++ 的边界

调度器和部分引擎状态位于 `csrc/engine/`，通过 pybind 暴露给 Python。移植平台时，不能只替换 PyTorch 设备名；这段扩展也需要确认使用了哪些设备 API。

### 模型代码与 kernel 的边界

模型层通常调用稳定的 Python 函数或算子工厂。真正的平台特化可以出现在：

- `tilelang_ops` 的后端模块；
- `vendor/` 下的 DeepGEMM、DeepEP、DeepSelect、FlashMLA；
- 为某个平台关闭的 fused path；
- 数据布局和 scale 编码。

### 编译期与运行期的边界

`DEEPINFER_ACCELERATOR`、平台标签和可见设备变量影响构建目录与运行时选择。看到“某平台已支持”时，需要同时验证：

1. 构建能选择正确编译器；
2. Python 能找到对应扩展；
3. 运行时硬件发现返回正确设备；
4. 算子分派到存在的实现；
5. 测试确实覆盖该后端。

## 阅读时的标注方式

在个人笔记中，建议给每个发现标记来源等级：

| 标记 | 含义 |
| --- | --- |
| `[源码]` | 可直接在固定 commit 的源码中看到 |
| `[文档]` | 来自 README、PR 描述或项目文档 |
| `[推断]` | 根据结构做出的暂定解释，尚待运行验证 |
| `[实机]` | 已在具体硬件和环境中执行并记录结果 |

这张地图会在阅读更多源码后继续修订。上游项目变化较快，重要结论都应同时记录 commit。

## 参考

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)
- [DeepInfer 固定版本](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
- [后端接缝](backend-seams.md)
- [ROCm PR 栈](rocm-pr-stack.md)

