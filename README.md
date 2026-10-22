# DeepInfer HIP 中文学习笔记

这是一个独立维护的中文学习仓库，用来记录阅读 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 源码、理解 GPU 推理框架结构，以及探索 HIP/ROCm 迁移的过程。

> 本项目不是 DeepInfer 官方移植项目。笔记中的结论只代表当前阅读和实验所覆盖的范围；未在 HIP 硬件上验证的内容会明确标注。

## 当前进度

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 仓库结构与阅读入口 | 进行中 | 本 README、路线图和日志模板 |
| DeepInfer 后端架构阅读 | 进行中 | [系统地图](docs/architecture/system-map.md)、[后端接缝](docs/architecture/backend-seams.md) |
| ROCm PR #202–#206 追踪 | 进行中 | [PR 栈笔记](docs/architecture/rocm-pr-stack.md) |
| DeepSelect TopK 契约整理 | 进行中 | [TopK 实验档案](labs/topk/README.md) |
| AMD/HIP 实机验证 | 未开始 | 当前设备不满足验证条件 |

## 从哪里开始

1. [DeepInfer 系统地图](docs/architecture/system-map.md)：先看框架由哪些层组成，以及一次推理大致经过哪些模块。
2. [后端接缝](docs/architecture/backend-seams.md)：重点看平台发现、构建、算子分派和通信后端如何连接。
3. [ROCm PR 栈](docs/architecture/rocm-pr-stack.md)：了解现有 #202–#206 已经覆盖什么，以及哪些地方仍需要独立验证。
4. [环境准备](docs/environment/amd-rocm.md)：查看 ROCm 基础检查、版本记录和最小验证顺序。
5. [TopK 实验](labs/topk/README.md)：从 DeepSelect 的输入输出契约开始准备最小实验。
6. [学习日志](logs/README.md)：按日期查看实际阅读、命令、失败和修正。
7. [路线图](ROADMAP.md)：查看当前阶段和下一项可验证产出。

## 最近更新

**2026-10-22：** 固定 DeepInfer、DeepSelect 和 ROCm PR 栈的参考提交；确认官方已有 ROCm 方案，因此后续以架构阅读、复现和聚焦改进为主，不重复提出泛泛的可行性 RFC。

完整时间线见 [日志索引](logs/README.md)。

环境信息可用 [采集脚本](scripts/snapshot-environment.sh) 统一记录。上游 PR 和真实贡献追踪见 [上游追踪](upstream/README.md)。

## 仓库结构

```text
docs/       稳定的架构、环境、术语和上游版本说明
logs/       按日期记录的学习过程和阶段性结论
labs/       可执行实验的目标、契约、环境和结果
upstream/   上游 PR、Issue、评论和贡献记录
scripts/    环境信息采集等辅助工具
```

计划与设计记录保存在 `docs/superpowers/`，用于保留仓库本身的演进过程。

## 记录原则

- 把“读过源码”“本地运行过”“在 HIP 硬件上验证过”分开标注。
- 上游讨论和实现必须附公开链接，并尽量固定 commit。
- 实验记录包含环境、命令、输入规模、预期结果和实际结果。
- 失败和回退同样保留，不用未经验证的性能数字代替测量。
- 个人笔记先形成可复现证据，再参与上游讨论或提交补丁。

## 上游资料

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- [DeepSelect](https://github.com/deepseek-ai/DeepSelect)
- [DeepInfer ROCm PR #202](https://github.com/deepseek-ai/DeepInfer/pull/202)
- [DeepInfer ROCm PR #203](https://github.com/deepseek-ai/DeepInfer/pull/203)
- [DeepInfer ROCm PR #204](https://github.com/deepseek-ai/DeepInfer/pull/204)
- [DeepInfer ROCm PR #205](https://github.com/deepseek-ai/DeepInfer/pull/205)
- [DeepInfer ROCm PR #206](https://github.com/deepseek-ai/DeepInfer/pull/206)

## 许可证

本项目使用 [Apache License 2.0](LICENSE)。引用上游源码或补丁时继续遵循对应项目的许可证。

