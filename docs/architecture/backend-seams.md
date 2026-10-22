# DeepInfer 后端接缝

理解一个推理框架能否迁移，关键不是找到所有 `cuda` 字符串，而是找出平台决策发生在哪里、由谁传递、在编译期还是运行期生效。

本文基于 DeepInfer `main@9bd9cf2`，只讨论官方已支持的 NVIDIA 和 Ascend 路径。ROCm 相关内容来自尚未合并的 PR，不作为当前主分支事实。

## 接缝总览

| 接缝 | 主要文件 | 回答的问题 |
| --- | --- | --- |
| 硬件发现 | `deep_cluster/hardware/` | 机器上有哪些设备，使用什么可见设备变量 |
| 平台对象 | `deepinfer/platform.py` | 当前平台使用哪种算子目录、通信后端和 FP8 scale 类型 |
| 构建入口 | `build.sh`、`scripts/build_vendor.sh` | 选择哪些 vendor 包和编译目标 |
| Python 扩展 | `setup.py` | C++ 源码使用 CUDA、NPU 还是其他编译器 |
| 算子分派 | `tilelang_ops/ops.py` | Python 层调用哪个平台的 `*_tl` 工厂 |

## 1. 硬件发现

`deep_cluster/hardware/device.py` 定义硬件无关接口，并通过 `accelerator_kind()` 返回 `cpu`、`nvidia` 或 `ascend`。

当前固定版本的判断顺序是：

1. 先检查 Ascend 设备；
2. 再检查 NVIDIA 设备；
3. 都不存在时返回 CPU。

可见设备变量也由硬件类型决定：

| 类型 | 可见设备变量 |
| --- | --- |
| NVIDIA | `CUDA_VISIBLE_DEVICES` |
| Ascend | `ASCEND_RT_VISIBLE_DEVICES` |
| CPU | 无 |

这里用一个轻量文件系统和运行时探测来避免过早初始化设备。对新增后端而言，这一层需要回答：设备从哪里发现、编号顺序是什么、NUMA 信息如何获取、设备可见性是否真的生效。

## 2. 平台对象

`deepinfer/__init__.py` 延迟初始化 `PLATFORM`，避免在导入阶段创建设备上下文。`deepinfer/platform.py` 中的平台对象目前有两种：

- `PlatformGPU`：`tilelang_ops_backend = "cuda"`，通信后端为 `nccl`；
- `PlatformNPU`：`tilelang_ops_backend = "asc"`，通信后端为 `hccl`。

平台对象至少影响：

- 使用哪个 TileLang 后端目录；
- 使用哪个 `torch.distributed` 后端；
- FP8 scale factor 使用什么 dtype；
- 主机内存 pin/unpin 使用什么设备 API；
- 分布式进程组参数如何构造。

当前 `get_platform()` 在非 Ascend 平台选择 `PlatformGPU`。因此新增 ROCm 后端不能只让硬件发现返回一个新字符串；还需要有对应的平台类和明确的默认行为，否则可能误落到 CUDA 路径。

## 3. 构建入口

`build.sh` 当前只接受：

```text
DEEPINFER_ACCELERATOR=nvidia
DEEPINFER_ACCELERATOR=ascend
```

`scripts/build_vendor.sh` 根据加速器选择 vendor 包：

| 加速器 | DeepSelect / FlashMLA 构建目标 | DeepEP / DeepGEMM 包 |
| --- | --- | --- |
| NVIDIA | `CUDA` | `DeepEP`、`DeepGEMM` |
| Ascend | `ASCEND` | `DeepEP-Ascend`、`DeepGEMM-Ascend` |

这说明平台支持是“主框架 + 多个算子包 + 构建脚本”的组合，不是单一开关。新增后端时，需要逐项列出每个依赖是否有对应实现。

## 4. Python 扩展

`setup.py` 当前有两个路径：

- Ascend：`NpuExtension`，加入 `asc_utils.cpp` 和 `graph_timer_asc.cpp`；
- 非 Ascend：`CUDAExtension`，加入 `cuda_utils.cpp` 和 `graph_timer_cuda.cpp`。

构建脚本使用 `DEEPINFER_ACCELERATOR` 和机器架构生成平台标签，产物放到 `deepinfer/_so/<accelerator>-<arch>/`。同名架构的多个平台产物可以共存。

迁移时要特别检查：

- C++ 扩展是否调用了 CUDA Runtime API；
- 主机内存 pin 和事件/流是否绑定到 CUDA；
- 图捕获与同步语义是否有 HIP 对应能力；
- 编译参数、库路径和 ABI 是否需要调整；
- Python 包初始化是否会因为找不到扩展而静默回退。

## 5. 算子分派

`tilelang_ops/ops.py` 根据 `PLATFORM.tilelang_ops_backend` 导入平台目录，再收集所有以 `_tl` 结尾的工厂函数。模型层只调用 `ops.xxx_tl`，后端选择集中在这个文件中。

这种设计把接口和实现分开了，但也意味着：

- 只存在接口名不等于该后端有实现；
- 缺少某个工厂时会产生 `KernelUnavailableError`，测试可能将其转换为跳过；
- “测试通过”必须区分真正执行和跳过；
- 同一个融合入口在不同平台可能需要拆成多个基础算子。

## 一个后端需要跨过的最小门槛

以新增 HIP/ROCm 后端为例，最小可验证门槛可以写成：

1. 硬件发现能返回正确的设备数量、编号和 NUMA 信息；
2. 构建脚本和 `setup.py` 能生成可导入的设备扩展；
3. `PLATFORM` 能明确选择 HIP 对应的算子目录和通信后端；
4. 至少一个简单算子能通过相同输入下的正确性测试；
5. 测试报告能证明该用例实际运行，而不是被跳过；
6. 再进入 attention、GEMM、MoE 和分布式通信。

现有 ROCm PR 栈已经按这个方向拆分，详见 [ROCm PR 栈](rocm-pr-stack.md)。

## 参考源码

- `deep_cluster/hardware/device.py`
- `deepinfer/__init__.py`
- `deepinfer/platform.py`
- `setup.py`
- `build.sh`
- `scripts/build_vendor.sh`
- `deepinfer/model_executor/layers/tilelang_ops/ops.py`

以上文件均可在 [DeepInfer 固定版本](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) 中查看。

