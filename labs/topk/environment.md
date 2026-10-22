# TopK 实验环境

## 当前记录

截至 2026-10-22，尚无 AMD/HIP 实机验证结果。本地开发机信息不发布到仓库。

## 获得 Linux + ROCm 环境后

先运行：

```bash
bash scripts/snapshot-environment.sh
```

再把完整输出保存到本文件的新日期小节。至少记录：

- Linux 发行版和内核；
- CPU 和内存；
- GPU 型号与 gfx 架构；
- 驱动版本；
- ROCm 版本；
- `hipcc --version`；
- Python、PyTorch、Triton、AITER、TileLang 版本；
- DeepInfer 和 DeepSelect commit；
- `torch.cuda.is_available()`、设备数量和设备名称；
- RCCL 或 `torch.distributed` 的验证结果。

## 环境变化规则

不要覆盖旧环境记录。每次更换驱动、ROCm、PyTorch 或硬件后新增日期小节，并在结果表中引用对应环境标签。

建议环境标签格式：

```text
YYYY-MM-DD-<gpu>-<rocm>-<torch>
```

## 最小设备检查

```bash
rocminfo
hipcc --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
PY
```

如果这些命令本身无法运行，先解决环境问题，不要开始修改算子。

