# 上游版本快照

**核对日期：** 2026-10-22

DeepInfer 和 DeepSelect 都处于快速演进阶段。本仓库中的架构判断、代码路径和 PR 状态都应当能对应到一个明确的提交。

## 固定提交

| 对象 | Commit | 用途 |
| --- | --- | --- |
| DeepInfer `main` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` | 官方主分支和现有 NVIDIA/Ascend 架构 |
| DeepInfer PR #202 | `6ce17ef03fcb1d49a9042cef1815f8fb3204dd00` | 平台、构建和 TileLang HIP |
| DeepInfer PR #203 | `36a1457e4ca51ad590273e20f6479babb7d79596` | 稠密 GEMM |
| DeepInfer PR #204 | `6a80688d7c74e737d986f87e1522a431565b153a` | sparse attention、indexer、TopK |
| DeepInfer PR #205 | `6a0c37ddad3f613b81cb4385d534ea3ccb43fa07` | MoE、EP、Engram、CP |
| DeepInfer PR #206 | `9a0e3531f241e29b789b83fb010f485eee0efced` | ROCm 文档和端到端结果 |
| DeepSelect `main` | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | CUDA/Ascend TopK 参考实现和契约 |

## 当时的公开状态

- DeepInfer 主分支公开支持 NVIDIA SM100 和 Ascend 950；
- #201 是一体化 ROCm PR，已关闭并拆分为 #202–#206；
- #202–#206 当时均处于未合并状态；
- 已有 ROCm 实现主要面向 MI350X/MI355X 的 gfx950；
- 个人阅读时尚未获得可执行 HIP 验证的硬件环境。

PR 状态变化很快，以上内容只描述快照时点。

## 复核命令

在任意有网络和 Git 的 Linux、macOS 或 Windows Bash 环境中执行：

```bash
git ls-remote https://github.com/deepseek-ai/DeepInfer.git \
  refs/heads/main \
  refs/pull/202/head \
  refs/pull/203/head \
  refs/pull/204/head \
  refs/pull/205/head \
  refs/pull/206/head

git ls-remote https://github.com/deepseek-ai/DeepSelect.git \
  refs/heads/main
```

PowerShell 单行版本：

```powershell
git ls-remote https://github.com/deepseek-ai/DeepInfer.git refs/heads/main refs/pull/202/head refs/pull/203/head refs/pull/204/head refs/pull/205/head refs/pull/206/head
git ls-remote https://github.com/deepseek-ai/DeepSelect.git refs/heads/main
```

如果不是上述哈希，说明上游已经变化。不要直接覆盖旧记录，应新增新的快照日期并说明哪些笔记需要重新核查。

## 临时检出上游源码

只在本地查看、不修改上游时，可以执行：

```bash
git clone --filter=blob:none https://github.com/deepseek-ai/DeepInfer.git
cd DeepInfer
git switch --detach 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
```

查看某个 PR 的 head：

```bash
git fetch origin pull/204/head:review/rocm-204
git show --stat review/rocm-204
```

PR 采用 stacked 结构时，diff 会包含前序提交。评审前应先确认该 PR 自己的顶层提交。

## 链接

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- [DeepSelect](https://github.com/deepseek-ai/DeepSelect)
- [PR #202](https://github.com/deepseek-ai/DeepInfer/pull/202)
- [PR #203](https://github.com/deepseek-ai/DeepInfer/pull/203)
- [PR #204](https://github.com/deepseek-ai/DeepInfer/pull/204)
- [PR #205](https://github.com/deepseek-ai/DeepInfer/pull/205)
- [PR #206](https://github.com/deepseek-ai/DeepInfer/pull/206)

