# DeepInfer HIP 中文学习笔记

这是一个独立维护的中文学习仓库，用来记录阅读 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 源码、理解 GPU 推理框架结构，以及探索 HIP/ROCm 迁移的过程。

> 本项目不是 DeepInfer 官方移植项目。笔记中的结论只代表当前阅读和实验所覆盖的范围；未在 HIP 硬件上验证的内容会明确标注。

## 当前进度

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 仓库结构与阅读入口 | 进行中 | 本 README、路线图和日志模板 |
| 推理主流程学习笔记 | 已发布 | [从“你好”到“好呀”](docs/learn/theory/01-token-journey.md) |
| DeepInfer 后端架构阅读 | 进行中 | [系统地图](docs/architecture/system-map.md)、[后端接缝](docs/architecture/backend-seams.md) |
| ROCm PR #202–#206 追踪 | 进行中 | [PR 栈笔记](docs/architecture/rocm-pr-stack.md) |
| DeepSelect TopK 契约整理 | 进行中 | [TopK 实验档案](labs/topk/README.md) |
| AMD/HIP 实机验证 | 未开始 | 尚无实机结果 |

## 从哪里开始

1. [整体框架与学习路线](docs/learn/01-deepinfer-overview.md)：先建立 DeepInfer 的整体地图和学习顺序。
2. [从“你好”到“好呀”](docs/learn/theory/01-token-journey.md)：沿着一次请求看 tokenizer、调度、Prefill、Decode 和采样。
3. [理论框架学习](docs/learn/theory/README.md)：按请求、调度、模型、算子和平台分块阅读。
4. [ROCm 迁移学习](docs/learn/migration/README.md)：从 PR 栈进入具体迁移专题。
5. [TopK 迁移专题](docs/learn/migration/01-topk-bf16-indexer-prefill.md)：当前进行中的 BF16 indexer prefill 正确性与调用契约验证。
6. [环境准备](docs/environment/amd-rocm.md)：查看 ROCm 基础检查、版本记录和最小验证顺序。
7. [学习日志](logs/README.md)：按日期查看实际阅读、命令、失败和修正。
8. [路线图](ROADMAP.md)：查看当前阶段和下一项可验证产出。

## 最近更新

**2026-10-22：** 固定 DeepInfer、DeepSelect 和 ROCm PR 栈的参考提交；确认官方已有 ROCm 方案，因此后续以架构阅读、复现和聚焦改进为主，不重复提出泛泛的可行性 RFC。

**2026-10-22：** 完成第一篇源码主线笔记，区分 indexer prefill、indexer decode、candidate block 和 sampler 四类 TopK 调用，并确定优先验证 BF16 indexer 路径。

**2026-10-22：** 调整学习结构：先读整体框架，再分为理论框架和 ROCm 迁移两条路线；TopK 专题归入迁移学习并标记为进行中。

**2026-10-22：** 完成 PR #204 BF16 indexer prefill 的静态实现对照，并加入可直接在 ROCm 环境运行的正确性与 Graph 验证脚本。

**2026-10-24：** 发布第一篇面向初学者的理论笔记，区分 DeepInfer 离线路径与生产服务路径，并整理一次 token 从输入到输出的完整旅程。

完整时间线见 [日志索引](logs/README.md)。

环境信息可用 [采集脚本](scripts/snapshot-environment.sh) 统一记录。上游 PR 和真实贡献追踪见 [上游追踪](upstream/README.md)。

## 仓库结构

```text
docs/       稳定的架构、环境、术语和上游版本说明
docs/learn/ 整体框架、理论学习和迁移学习路线
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

