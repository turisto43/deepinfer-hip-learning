# DeepInfer HIP 中文学习仓库设计

**日期：** 2026-10-22

**目标：** 建立一个公开、持续更新的中文学习仓库，记录从阅读 DeepInfer 架构到验证 HIP/ROCm 行为的完整过程，并逐步形成可用于海光或 AMD 生态求职的作品集。

**仓库名：** `deepinfer-hip-learning`

**许可证：** Apache-2.0

## 项目定位

仓库是一个独立的学习与实验档案，不是 DeepInfer 官方移植项目，也不复制上游完整源码。内容围绕三条主线组织：

1. **中文学习笔记：** 解释 DeepInfer 的分层架构、NVIDIA/Ascend/ROCm 后端接缝、构建链路以及 DeepSelect TopK 契约。
2. **可复现实验：** 对每个实验记录环境、命令、输入规模、预期行为、实际结果和失败原因。
3. **上游协作档案：** 跟踪阅读、审查、Issue 评论、复现验证和小型补丁，公开提交与评审过程。

项目强调真实进展和可验证证据，不以“完整跑通大模型”作为第一阶段成功标准。

## 目标读者

- 正在学习 GPU 推理框架和 HIP/ROCm 的中文读者。
- 希望了解 DeepInfer 后端扩展方式的上游贡献者。
- 关注候选人在异构计算、性能工程和开源协作方面表现的招聘方。

## 第一阶段范围

第一阶段只覆盖仓库建设、上游结构阅读和 TopK 实验准备：

- 建立稳定、易扩展的文档和实验目录。
- 固定上游 DeepInfer、DeepSelect 及 ROCm PR 栈的版本。
- 写出一份基于源码的 DeepInfer 后端架构概览。
- 梳理 ROCm PR #202–#206 的拆分、依赖关系和验证范围。
- 定义 DeepSelect TopK 的接口契约、测试场景和实验记录模板。
- 提供 AMD ROCm 环境采集脚本和海光环境调研清单。
- 建立持续日志模板与首次研究日志。
- 完成本地 Git 初始化、分批提交和公开 GitHub 仓库创建。

第一阶段不包含完整模型推理、真实性能排名、未经硬件验证的移植结论，也不复制上游供应商源码。

## 信息架构

```text
deepinfer-hip-learning/
├─ README.md
├─ ROADMAP.md
├─ LICENSE
├─ .gitignore
├─ docs/
│  ├─ architecture/
│  │  ├─ system-map.md
│  │  ├─ backend-seams.md
│  │  └─ rocm-pr-stack.md
│  ├─ environment/
│  │  ├─ amd-rocm.md
│  │  └─ hygon-plan.md
│  ├─ reference/
│  │  ├─ upstream-snapshot.md
│  │  └─ glossary.md
│  └─ contribution-etiquette.md
├─ logs/
│  ├─ README.md
│  ├─ TEMPLATE.md
│  └─ 2026-10-22-initial-research.md
├─ labs/
│  ├─ README.md
│  └─ topk/
│     ├─ README.md
│     ├─ contract.md
│     ├─ environment.md
│     └─ results.md
├─ upstream/
│  ├─ README.md
│  ├─ pr-stack.md
│  └─ contribution-log.md
└─ scripts/
   └─ snapshot-environment.sh
```

每个目录有明确职责：

- `docs/architecture/` 保存稳定的概念说明，不记录单次实验流水。
- `docs/environment/` 保存可复用的环境搭建知识。
- `docs/reference/` 保存版本快照和术语，便于核查笔记时效性。
- `logs/` 按日期保存增量学习记录，是仓库的主要更新流。
- `labs/` 保存可执行实验的目标、契约和结果，不把实验结论混入长期文档。
- `upstream/` 保存上游 PR、Issue、评论和补丁的追踪信息。
- `scripts/` 只保存可直接运行的辅助脚本。

## 内容规范

### 学习日志

每篇日志使用统一模板，至少包含：

- 日期和目标。
- 阅读的源码或上游讨论链接。
- 关键概念及个人解释。
- 执行过的命令和可复现环境。
- 已验证结论、暂定假设和未解决问题。
- 下一步最小行动。

结论必须区分“已阅读”“已本地验证”“已在 HIP 硬件验证”。

### 实验记录

每个实验在 `labs/<topic>/` 下维护：

- `README.md`：实验目标、当前状态和目录导览。
- `contract.md`：输入、输出、边界条件和兼容性要求。
- `environment.md`：操作系统、驱动、ROCm、Python、GPU 和依赖版本。
- `results.md`：按日期追加正确性、性能和失败记录。

未运行的测试不得写成通过，未测得的性能数字不得估算后当作结果展示。

### 上游追踪

`upstream/pr-stack.md` 记录 PR 依赖链、发现时间和核查状态。`upstream/contribution-log.md` 只记录真实发生的 Issue、评论、PR、评审或合并事件，并附公开链接。

## 首批上游版本

首次研究固定以下提交，后续笔记若引用新版本必须再次记录：

| 对象 | Commit |
| --- | --- |
| DeepInfer `main` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| DeepInfer PR #202 | `6ce17ef03fcb1d49a9042cef1815f8fb3204dd00` |
| DeepInfer PR #203 | `36a1457e4ca51ad590273e20f6479babb7d79596` |
| DeepInfer PR #204 | `6a80688d7c74e737d986f87e1522a431565b153a` |
| DeepInfer PR #205 | `6a0c37ddad3f613b81cb4385d534ea3ccb43fa07` |
| DeepInfer PR #206 | `9a0e3531f241e29b789b83fb010f485eee0efced` |
| DeepSelect `main` | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` |

## 实施顺序

1. 初始化 Git 仓库、许可证和根目录说明。
2. 固定上游版本并编写架构学习笔记。
3. 建立日志、实验和上游追踪模板。
4. 编写 AMD/海光环境说明与环境采集脚本。
5. 检查文档链接、目录完整性和工作树状态。
6. 创建公开 GitHub 仓库并推送全部提交。

初始搭建拆成多个提交，以保留清晰的学习和建设轨迹。

## 成功标准

第一阶段完成后应满足：

- 新访客能在 README 中用两分钟理解项目目的、当前进度和下一步。
- 所有上游事实都能追溯到公开链接或固定 commit。
- 所有环境实验都有明确的可复现入口。
- 目录、模板和许可证可以直接支持后续日志与实验。
- 本地仓库通过文档链接和关键内容检查。
- GitHub 上的公开仓库与本地首个里程碑同步。

## 风险与约束

- 当前开发环境是 Windows，没有 `hipcc` 或 `rocminfo`，第一阶段不能声称完成 HIP 实机验证。
- GitHub CLI 当前令牌失效，远程创建和推送需要用户重新完成一次浏览器登录。
- 上游 ROCm PR 栈仍可能变化，所有引用必须注明 commit 和核查日期。
- 上游 ROCm 实现仍有设备、数值行为和多卡通信边界，个人仓库只陈述自己实际验证的部分。
- 学习笔记使用中文，代码标识符和命令保留英文，避免翻译破坏可执行性。
