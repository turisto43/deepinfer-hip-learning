# PR #204 BF16 Indexer Prefill 验证计划

## 目标

验证 DeepInfer ROCm PR #204 的 `deep_select.interface.topk` 在 BF16 indexer prefill 调用方式下：

- 能返回正确的每行 top-k 索引；
- 满足 `end`、offset、填充和升序索引契约；
- 正确复用调用方提供的输出缓冲区；
- 不返回 value；
- 能重复执行并进入 Graph 捕获；
- 把未覆盖项和已知差异留下来，而不是用“测试通过”概括全部行为。

## 固定版本

| 对象 | Commit |
| --- | --- |
| DeepInfer 调用方 | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| DeepInfer PR #204 | `6a80688d7c74e737d986f87e1522a431565b153a` |

## 调用形态

首个实验固定使用与 `MQA._topk_prefill` 相一致的参数：

```python
values, output = deep_select.topk(
    x,
    k,
    sorted=False,
    end=end,
    indices_type=torch.int32,
    sorted_index=True,
    output_idx=output,
    output_idx_offset=offset,
    idx_oob_fill_value=-1,
    return_value=False,
)
```

输入和输出：

- `x`：CUDA/HIP 设备上的 BF16 `(rows, width)`；
- `end`：`int32 (rows,)`，每行有效上界；
- `offset`：`int32 (rows,)`，加到有效输出索引；
- `output`：`int32 (rows, k)`；
- 返回的 `values` 必须为 `None`；
- 返回的索引张量必须就是传入的 `output`。

## 正确性检查

每个普通用例检查：

1. 输出 dtype、shape 和对象地址；
2. 有效位置的数量为 `min(end, k)`；
3. 无效位置全部为 `-1`，且不会被 offset 修改；
4. 减去 offset 后，有效索引都在 `[0, end)`；
5. 有效索引没有重复；
6. 输出索引按升序排列；
7. 选中元素的最小值不小于未选中有效元素的最大值；
8. 没有返回 value。

并列值只检查 top-k 条件和索引顺序，不要求与另一个实现选择完全相同的并列索引。

## 测试矩阵

### Smoke

- rows：8；
- width：4096；
- topk：512；
- `end`：循环包含 `0`、`1`、`k/3`、`k`、`k+1`、`width`；
- offset：有、无两种；
- Graph：执行一次捕获和两次 replay。

### Prefill

- rows：64；
- width：65536；
- topk：512；
- 可变 `end`；
- offset：有；
- 同一 Graph 内修改 `end` 后再次 replay。

## 运行方式

在具备 ROCm PyTorch 和 Triton 的环境中，从本仓库根目录执行：

```bash
python labs/topk/check_pr204_prefill.py \
  --vendor-path /path/to/DeepInfer/vendor/rocm \
  --profile smoke
```

代表性 prefill 形状：

```bash
python labs/topk/check_pr204_prefill.py \
  --vendor-path /path/to/DeepInfer/vendor/rocm \
  --profile prefill
```

只在排查 Graph 问题时使用：

```bash
python labs/topk/check_pr204_prefill.py \
  --vendor-path /path/to/DeepInfer/vendor/rocm \
  --profile smoke \
  --skip-graph
```

## 通过标准

普通 finite 输入全部通过后，可以记录：

```text
BF16 indexer prefill finite-case contract: PASS
```

只有 Graph 用例也通过，才能记录：

```text
BF16 indexer prefill finite-case + graph: PASS
```

NaN 和负 NaN 不包含在首批普通用例中，原因见 [静态实现对照](../../docs/learn/migration/01-topk-bf16-indexer-prefill.md#仍需实机确认的行为)。需要单独运行，因为 DeepSelect 的其他后端可能在发现 NaN 时主动终止进程。

## 记录要求

每次运行必须在 [结果记录](results.md) 中保存：

- 日期和环境标签；
- GPU 型号和 gfx；
- ROCm、PyTorch、Triton 版本；
- 实际 commit；
- 命令；
- 通过、失败或跳过；
- 失败时的最小复现和错误信息；
- 是否执行了 Graph 用例。

