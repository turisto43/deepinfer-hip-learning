# AMD ROCm 环境准备

本文用于准备 DeepInfer HIP/ROCm 实验环境。具体安装命令随发行版和 ROCm 版本变化，应优先使用 AMD 官方安装文档。

## 前提

- Linux 环境；
- 受目标 ROCm 版本支持的 AMD GPU；
- 可用的内核驱动；
- Python 3.12；
- Git、编译器和足够磁盘空间；
- 能够安装 ROCm、PyTorch ROCm wheel、Triton 和项目依赖。

Windows 可以用于阅读代码和编写文档，但当前 DeepInfer 的构建脚本以 Linux Bash 和 Linux 动态库路径为前提。

## 1. 验证驱动和运行时

安装后先确认工具存在：

```bash
command -v rocminfo
command -v hipcc
command -v rocm-smi
```

查看 GPU 和架构：

```bash
rocminfo
rocm-smi
hipcc --version
```

需要记录：

- GPU 型号；
- 实际 gfx 架构，例如 `gfx950`；
- 驱动版本；
- ROCm 版本；
- `rocminfo` 中可见设备数量；
- 是否存在与目标架构相关的警告。

不要仅根据商品名推断 gfx 架构。`gfx942`、`gfx950` 等名称直接决定编译器目标、指令特性和部分库支持。

## 2. 验证 PyTorch

使用与 ROCm 版本匹配的 PyTorch wheel。安装来源和目标版本应以 PyTorch 官方说明为准。

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.ones(1024, device="cuda")
    print("sum:", x.sum().item())
PY
```

预期能够看到 ROCm 版本、至少一个设备，并且 GPU tensor 运算成功。

如果 `torch.cuda.is_available()` 为假，先检查 wheel 是否真的是 ROCm 版本、用户是否在 `video`/`render` 组、设备节点权限和可见设备变量。不要继续修改项目算子。

## 3. 创建独立 Python 环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

记录 Python 和 pip 版本：

```bash
python --version
python -m pip --version
```

不建议在系统 Python 中混合多个 PyTorch 来源。

## 4. 固定上游版本

```bash
git clone --filter=blob:none https://github.com/deepseek-ai/DeepInfer.git
cd DeepInfer
git switch --detach 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
```

如果研究尚未合并的 ROCm 分支，应创建独立工作区，并记录 PR head：

```bash
git fetch origin pull/202/head:review/rocm-202
git worktree add ../DeepInfer-rocm-202 review/rocm-202
```

不要在一个工作区里混用多个 PR head，否则很难确认当前文件和测试来自哪一层。

## 5. 环境快照

在个人学习仓库根目录执行：

```bash
bash scripts/snapshot-environment.sh > rocm-environment.txt
```

检查输出中是否包含 GPU 型号、gfx 架构和 ROCm 版本，再把关键内容摘要写入 `labs/topk/environment.md`。原始输出如果较大，可以只保留在本地，不提交含机器路径或内部主机名的内容。

## 6. 构建前检查

在 DeepInfer 工作区中先确认：

- 当前 commit；
- 环境变量 `DEEPINFER_ACCELERATOR`；
- 使用的 vendor 包；
- TileLang、Triton、AITER、RCCL 版本；
- CUDA 兼容层或 HIP 专用代码路径；
- Python 扩展是否已经有旧构建产物。

清理旧产物时只操作当前 DeepInfer 工作区，并先确认路径：

```bash
pwd
git rev-parse --show-toplevel
```

## 7. 最小验证顺序

建议严格按以下顺序推进：

1. PyTorch GPU 基础运算；
2. Triton vector add；
3. DeepSelect TopK 小用例；
4. DeepSelect 常用形状；
5. DeepInfer TileLang 基础测试；
6. attention/indexer；
7. 多卡通信；
8. 完整模型。

每一步失败时先保留环境和命令，不要跨层继续。

## 常见记录项

| 现象 | 优先检查 |
| --- | --- |
| `hipcc: not found` | ROCm 安装和 `PATH` |
| `rocminfo` 看不到设备 | 驱动、权限、容器设备映射 |
| PyTorch 显示 CPU-only 版本 | wheel 来源 |
| Triton kernel 编译失败 | gfx 目标、版本和 kernel 使用的指令 |
| 测试被跳过 | 后端算子缺失或测试标记 |
| 数值轻微不一致 | dtype、FMA/rounding、scale layout 和允许误差 |
| CUDA Graph 捕获失败 | 动态内存分配、同步 API、流依赖 |

## 安全边界

- 不通过强制覆盖 gfx 目标来宣称硬件已支持；
- 不把一套 ROCm 版本的结果外推到另一套版本；
- 不在性能测试中混用动态频率和未知功耗状态；
- 不提交包含内网主机名、私有模型路径或访问令牌的完整环境日志。

