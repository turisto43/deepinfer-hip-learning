# 2026-10-22 初始调研

## 目标

确认 DeepInfer 当前的硬件后端情况，判断 HIP/ROCm 学习仓库应当从哪个切入点开始。

## 阅读与参考

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
- [DeepInfer 固定版本](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
- [DeepSelect README](https://github.com/deepseek-ai/DeepSelect)
- [ROCm PR #202–#206](https://github.com/deepseek-ai/DeepInfer/pull/202)

## 关键发现

### 官方主分支

DeepInfer 主分支公开列出 NVIDIA SM100 和 Huawei Ascend 950 支持。固定版本的硬件发现接口只返回 `cpu`、`nvidia` 或 `ascend`，平台对象目前也只区分 GPU 与 NPU 两条路径。

这说明 ROCm 不能只通过修改一个环境变量接入；至少需要同时处理设备发现、平台对象、构建目标和算子实现。

### 已有 ROCm 工作

官方仓库存在按 #202–#206 拆分的 ROCm PR 栈，目标设备是 gfx950。覆盖范围从平台和 TileLang，一直延伸到 GEMM、attention、TopK、MoE、EP 和端到端结果。

因此，再次提交“HIP 是否可行”的泛化 RFC 会重复现有讨论。更有价值的起点是：

1. 阅读架构和 PR 分层；
2. 选择 TopK 这样边界清楚的最小算子；
3. 在可获得的环境里独立复现；
4. 用测试或小补丁补充现有工作。

## 环境与命令

本次只进行了阅读和公开信息核对，没有执行 HIP 实验，因此不记录环境结论。

上游版本通过只读 `git ls-remote` 固定，具体哈希见 [上游版本快照](../docs/reference/upstream-snapshot.md)。

## 已阅读

- DeepInfer 的 README、构建入口、硬件发现、平台对象和 TileLang 算子分派。
- DeepSelect 支持的输入类型、TopK 场景、索引类型和输出选项。
- ROCm PR 栈的拆分方式、目标硬件和公开验证范围。

## 已本地验证

- Git 能读取上游 `main` 和各 PR head 的提交哈希。
- 本地仓库已建立文档目录并保存版本快照。

## 已在 HIP 硬件验证

本次未进行。

## 未解决问题

- 哪个 AMD GPU 或海光 DCU 型号可以作为首个实验目标？
- 对应环境使用哪一版 ROCm、PyTorch、Triton 和通信库？
- gfx950 之外的设备存在哪些指令、LDS、矩阵指令和编译器差异？
- ROCm TopK 不检测 NaN 是否会成为稳定的接口差异？

## 下一步

完成 TopK 契约和测试矩阵，并为第一个可获得的环境准备可复制的最小命令。

