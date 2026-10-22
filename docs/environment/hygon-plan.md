# 海光 DCU 环境调研清单

海光 DCU 与 ROCm/HIP 生态相关，但具体产品、驱动、编译器、PyTorch wheel 和通信库版本需要以实际平台文档为准。本文件只列确认项，不预设 DeepInfer 的 AMD ROCm PR 可以直接运行。

## 首先确认的硬件信息

- DCU 产品型号；
- 加速卡数量；
- 单卡显存和互连方式；
- 固件和驱动版本；
- 设备节点名称；
- 运行时能够识别的 gfx 或其他架构标识；
- 是否支持目标 FP8/FP4 数据类型和矩阵指令。

这些信息决定能否直接使用 gfx950 代码，还是需要单独编译或降级到未融合路径。

## 软件栈确认

| 项目 | 需要记录 |
| --- | --- |
| 厂商工具链 | 名称、版本、安装路径 |
| HIP 编译器 | 命令、版本、目标架构 |
| ROCm 兼容层 | 版本和兼容范围 |
| PyTorch | 厂商 wheel 或社区 wheel、版本、`torch.version.hip` |
| Triton | 是否可用、版本、目标后端 |
| TileLang | 版本和补丁需求 |
| 通信库 | RCCL/厂商集合通信库及版本 |
| 性能工具 | profiler、监控工具和权限 |
| 容器 | 镜像、设备映射、挂载和动态库路径 |

如果厂商工具使用自己的命令名，不要用 AMD 工具名代替事实记录。例如在确认前，不把 `rocm-smi`、`hipcc` 记为一定可用。

## 兼容性验证

按从低到高的顺序测试：

1. 设备枚举；
2. 基础 HIP 程序；
3. PyTorch GPU tensor 运算；
4. Triton 最小 kernel；
5. `torch.topk` 参考结果；
6. DeepSelect TopK；
7. DeepInfer TileLang 算子；
8. attention/indexer；
9. 多卡通信；
10. 模型加载和完整推理。

每一层都记录实际命令和输出。上一层的兼容性不能证明下一层可用。

## 代码差异风险

需要重点检查：

- gfx 架构和目标编译参数；
- wavefront 宽度以及 warp 内原语；
- LDS 容量和 bank 行为；
- FP8/FP4 scale layout；
- `hipMemcpy`、peer memory 和 symmetric memory 支持；
- RCCL/厂商通信库对 CUDA Graph 的支持；
- PyTorch 对 `cuda` API 的兼容映射程度；
- TileLang、Triton 和 AITER 是否能生成目标设备代码。

## 数据记录模板

```text
环境标签：
DCU 型号：
卡数量：
驱动/固件：
工具链：
ROCm 兼容层：
PyTorch：
Triton：
TileLang：
通信库：
DeepInfer commit：
DeepSelect commit：
通过的层级：
第一个失败层级：
错误信息：
最小复现命令：
```

## 上游沟通前

先确认问题属于：

- 个人环境配置；
- AMD gfx950 实现的其他 gfx 兼容性；
- 海光厂商工具链差异；
- DeepInfer 通用后端接口；
- DeepSelect TopK 接口契约。

不同问题应附不同证据，也应由合适的项目或维护者处理。不要把厂商私有环境问题直接表述成 DeepInfer 的通用缺陷。

