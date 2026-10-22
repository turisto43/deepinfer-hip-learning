# DeepInfer HIP 中文学习仓库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建可公开维护的 DeepInfer HIP/ROCm 中文学习仓库，并完成首批架构笔记、实验模板和上游追踪资料。

**Architecture:** 以 `README.md` 作为入口，把稳定知识放在 `docs/`，把按日期变化的学习过程放在 `logs/`，把实验契约和结果放在 `labs/`，把上游 PR 状态和真实贡献记录放在 `upstream/`。所有上游事实固定到 commit 或公开链接。

**Tech Stack:** Markdown、Git、GitHub Actions 不使用外部运行时、Bash 环境采集脚本、Apache-2.0。

---

### Task 1: 建立根目录与项目入口

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `ROADMAP.md`

- [ ] **Step 1: 写入 Apache-2.0 许可证**

将 Apache License 2.0 完整英文文本写入 `LICENSE`，版权行使用 `Copyright 2026 turisto43`。

- [ ] **Step 2: 写入忽略规则**

` .gitignore` 至少包含 Python 缓存、构建目录、编辑器状态、环境文件、模型与本地实验输出：

```gitignore
__pycache__/
*.py[cod]
*.so
build/
dist/
.venv/
.env
.idea/
.vscode/
.DS_Store
Thumbs.db
*.log
models/
outputs/
```

- [ ] **Step 3: 写入 README**

README 包含项目定位、非官方声明、当前阶段、三条阅读入口、仓库结构、最近更新、已验证边界和下一步。不得出现求职、作品集或招聘描述。

- [ ] **Step 4: 写入路线图**

`ROADMAP.md` 分为当前阶段、架构阅读、环境准备、TopK 实验、HIP 迁移和上游协作六部分；每项标注 `未开始`、`进行中` 或 `已完成`。

- [ ] **Step 5: 检查入口文档并提交**

Run:

```powershell
rg -n "求职|作品集|招聘" README.md ROADMAP.md
```

Expected: 无匹配，命令返回 1。

```powershell
git add .gitignore LICENSE README.md ROADMAP.md
git commit -m "docs: initialize HIP learning repository"
```

Expected: 创建首个仓库骨架提交。

### Task 2: 编写架构阅读与版本快照

**Files:**
- Create: `docs/architecture/system-map.md`
- Create: `docs/architecture/backend-seams.md`
- Create: `docs/architecture/rocm-pr-stack.md`
- Create: `docs/reference/upstream-snapshot.md`
- Create: `docs/reference/glossary.md`

- [ ] **Step 1: 编写系统地图**

`system-map.md` 用目录树和调用链说明入口、调度器、硬件发现、模型执行器、TileLang 算子、供应商算子、构建脚本之间的关系，并链接 DeepInfer README。

- [ ] **Step 2: 编写后端接缝**

`backend-seams.md` 基于 `deep_cluster/hardware/device.py`、`deepinfer/platform.py`、`setup.py`、`scripts/build_vendor.sh` 和 `tilelang_ops/ops.py` 说明平台选择、构建分派、算子分派与测试标记。明确这五个接缝分别负责什么。

- [ ] **Step 3: 编写 ROCm PR 栈笔记**

`rocm-pr-stack.md` 记录 #202–#206 的依赖顺序、主要文件、验证范围、设备限制和仍需独立验证的部分。已有 PR 是阅读和协作起点，不另提重复可行性 RFC。

- [ ] **Step 4: 固定上游快照**

`upstream-snapshot.md` 记录 2026-10-22 获取的 DeepInfer、PR #202–#206 和 DeepSelect commit，给出 `git ls-remote` 复核命令。

- [ ] **Step 5: 写核心术语表**

`glossary.md` 解释 HIP、ROCm、gfx、TopK、DSA、EP、EPLB、TileLang、Triton、SFA、LDS、RCCL 和 CUDA Graph。

- [ ] **Step 6: 检查版本与提交**

Run:

```powershell
rg -n "9bd9cf2|6ce17ef|36a1457|6a80688|6a0c37d|9a0e353|bfa4507" docs/reference/upstream-snapshot.md
```

Expected: 七个 commit 前缀全部出现。

```powershell
git add docs/architecture docs/reference
git commit -m "docs: map DeepInfer backends and upstream ROCm work"
```

Expected: 架构和版本资料提交完成。

### Task 3: 建立日志、实验和上游追踪流程

**Files:**
- Create: `docs/contribution-etiquette.md`
- Create: `logs/README.md`
- Create: `logs/TEMPLATE.md`
- Create: `logs/2026-10-22-initial-research.md`
- Create: `labs/README.md`
- Create: `labs/topk/README.md`
- Create: `labs/topk/contract.md`
- Create: `labs/topk/environment.md`
- Create: `labs/topk/results.md`
- Create: `upstream/README.md`
- Create: `upstream/pr-stack.md`
- Create: `upstream/contribution-log.md`

- [ ] **Step 1: 建立日志模板与首篇日志**

模板必须包含目标、上游阅读、概念解释、环境与命令、结论与证据、未解决问题、下一步。首篇日志记录本日的 PR 发现、重复 RFC 风险、仓库定位和下一实验。

- [ ] **Step 2: 建立 TopK 实验档案**

`labs/topk/README.md` 定义目标与状态；`contract.md` 记录输入、输出、排序、填充、索引类型、NaN 和 stride 契约；`environment.md` 明确当前 Windows 环境未验证 HIP；`results.md` 使用表格追加未来结果。

- [ ] **Step 3: 建立上游追踪档案**

`upstream/pr-stack.md` 记录 #202–#206 的核查状态和待验证点；`contribution-log.md` 使用“日期、类型、链接、内容、结果”表格，首阶段只记录真实发生的事件。

- [ ] **Step 4: 写贡献礼仪**

`contribution-etiquette.md` 说明先复现再提问、提供版本和命令、尊重已有方案、小步提交、明确区分个人笔记与上游结论。

- [ ] **Step 5: 检查内容并提交**

Run:

```powershell
rg -n "TBD|TODO|待补充|fill in" logs labs upstream docs/contribution-etiquette.md
```

Expected: 无匹配，命令返回 1。

```powershell
git add docs/contribution-etiquette.md logs labs upstream
git commit -m "docs: add learning logs and TopK experiment workflow"
```

Expected: 持续记录模板提交完成。

### Task 4: 编写环境说明与采集脚本

**Files:**
- Create: `docs/environment/amd-rocm.md`
- Create: `docs/environment/hygon-plan.md`
- Create: `scripts/snapshot-environment.sh`

- [ ] **Step 1: 编写 AMD ROCm 环境说明**

涵盖 Linux 前提、ROCm 安装验证、`rocminfo`、`hipcc`、Python/PyTorch、GPU 架构识别、上游固定版本、构建入口和常见失败记录方式。不得宣称已在当前机器验证。

- [ ] **Step 2: 编写海光调研清单**

记录需要确认的 DCU 型号、驱动版本、ROCm/HIP 兼容层、可用 PyTorch、Triton、RCCL、编译器与集群环境；把未知项写成待确认问题，不预判兼容性。

- [ ] **Step 3: 编写环境采集脚本**

脚本使用标准 Bash，输出时间、OS、内核、CPU、内存、Python、pip、Git、ROCm 版本、`rocminfo` 摘要、`hipcc --version`、PyTorch/Triton 可用性。命令不存在时打印 `not found` 并继续。

- [ ] **Step 4: 验证脚本**

Run:

```powershell
bash -n scripts/snapshot-environment.sh
```

Expected: 无语法错误输出，退出码 0。

- [ ] **Step 5: 提交**

```powershell
git add docs/environment scripts/snapshot-environment.sh
git commit -m "docs: add ROCm environment guide and snapshot script"
```

Expected: 环境资料提交完成。

### Task 5: 全仓库验证与发布准备

**Files:**
- Modify: `README.md`
- Verify: 所有 Markdown 文件和 Git 历史

- [ ] **Step 1: 更新 README 的最新进展**

把最新提交主题、当前阶段、下一最小行动和主要入口补充到 README，保持两分钟可读。

- [ ] **Step 2: 验证文档链接**

使用 Python 标准库扫描仓库内 Markdown 的相对链接，检查目标路径存在并输出缺失链接。

Expected: `missing_links=0`。

- [ ] **Step 3: 验证目录和禁用措辞**

Run:

```powershell
rg -n "求职|作品集|招聘|TBD|TODO|待补充|fill in" . -g "!docs/superpowers/**"
```

Expected: 无匹配，命令返回 1。

- [ ] **Step 4: 验证 Git 状态并提交**

```powershell
git status --short
git add README.md
git commit -m "docs: finalize initial learning milestone"
git status --short
git log --oneline --decorate -8
```

Expected: 最终工作树干净，提交历史包含本计划产生的独立提交。

- [ ] **Step 5: 创建并推送 GitHub 仓库**

先运行 `gh auth status`。如果令牌有效，则执行：

```powershell
gh repo create deepinfer-hip-learning --public --source . --remote origin --push --description "中文记录 DeepInfer 架构与 HIP/ROCm 迁移学习过程"
```

Expected: 创建公开仓库并以 `main` 推送。若令牌仍无效，停止远程发布并报告本地仓库已经完成的提交。
