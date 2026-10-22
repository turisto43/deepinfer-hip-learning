# TopK 接口契约

本文以 DeepSelect `main@bfa4507` 和 DeepInfer ROCm PR #204 的公开实现为参考。最终一致性测试应先读取当前接口，而不是依赖本文的静态描述。

## 基本调用

```python
values, indices = deep_select.topk(
    input,
    topk,
    sorted=False,
    sorted_index=True,
    indices_type=torch.int32,
    return_value=True,
)
```

## DeepInfer 中的实际调用形态

DeepInfer `main@9bd9cf2` 至少包含以下形态：

| 调用方 | dtype | 关键参数 | 后续处理 |
| --- | --- | --- | --- |
| Indexer prefill | BF16 | `return_value=False`、可变 `end`、`sorted_index=True`、输出缓冲区 | 用于 sparse attention |
| Indexer decode | BF16 | `return_value=False`、可变 `end`、`sorted_index=True` | 逻辑索引转换为物理 KV slot |
| Sampler force-top-k | FP32 | `sorted=False`、返回值 | 用 index 映射真实词表 ID |
| Sampler top-logprobs | FP32 | `sorted=True`、返回值 | 返回 top token 及 logprob |

Indexer 路径直接导入 `deep_select.interface.topk`，不经过 TileLang 算子分派。详细调用链见 [ROCm TopK 迁移专题](../../docs/learn/migration/01-topk-bf16-indexer-prefill.md)。

## 输入

| 项目 | 契约 |
| --- | --- |
| 形状 | 二维 `(batch_size, width)` |
| 数据类型 | `torch.bfloat16` 或 `torch.float32` |
| 连续性 | 最后一维连续 |
| 行 stride | 满足 `get_stride_requirement()[0]` 的字节对齐要求 |
| 行长度 | 默认整行参与；可用 `end` 指定每行有效上界 |

不同后端支持的 dtype 不完全相同。DeepSelect README 说明 Ascend 路径只支持 BF16，FP32 采样场景目前只在 CUDA 中实现。

## 输出

| 输出 | 形状 | 说明 |
| --- | --- | --- |
| `values` | `(batch_size, topk)` | 输入 dtype；`return_value=False` 时不返回 |
| `indices` | `(batch_size, topk)` | `torch.int32` 或 `torch.int64` |

输出缓冲区可能因字节对齐而不完全连续。测试应检查逻辑形状和值，也要检查 stride 是否满足接口要求。

## 排序选项

- `sorted=False`：结果不要求按 value 排序。
- `sorted=True`：结果按 value 从大到小排列。
- `sorted_index=True`：每行索引按升序输出。
- `sorted=True` 与 `sorted_index=True` 不能同时使用。
- `sorted=True` 时通常要求 `return_value=True`。

“不排序”只表示不承诺顺序，不表示允许遗漏、重复或选择错误元素。测试应先比较 top-k 集合，再在声明排序时比较顺序。

## 可变长度与填充

`end` 是形状为 `(batch_size,)` 的逐行上界。若有效元素少于 `topk`：

- 有效元素应排在前面；
- 剩余位置使用 `idx_oob_fill_value` 和 `value_oob_fill_value` 填充；
- 输出索引是否加上 `output_idx_offset` 取决于调用参数。

零长度、刚好等于 `topk`、略小于 `topk` 和远小于 `topk` 都应进入测试矩阵。

## 索引与输出偏移

- 默认索引从 0 开始；
- `output_idx_offset` 为有效索引增加固定偏移；
- 自定义 `output_idx` 必须匹配 dtype、形状和对齐要求；
- `indices_type` 支持 `torch.int32` 或 `torch.int64`，具体可用性以后端为准。

## NaN 行为

DeepSelect README 描述 CUDA 实现始终检查 NaN，默认发现 NaN 时终止进程。DeepInfer ROCm PR #204 的公开接口说明则标明 ROCm 路径不检测 NaN。

这是一个明确的接口差异候选。测试设计需要分别记录：

1. 输入包含 NaN 时后端实际行为；
2. 是否与后端公开说明一致；
3. 是否应该要求不同后端保持一致；
4. 测试进程被终止时如何隔离报告。

## 测试维度

| 维度 | 最小覆盖 |
| --- | --- |
| dtype | BF16、FP32 |
| shape | 小 batch、大 batch、宽行、窄行 |
| topk | 1、常用值 512、边界 4096 |
| row length | 0、1、小于 topk、等于 topk、大于 topk |
| 排序 | 默认、value sorted、index sorted |
| 输出 | value 返回/省略、int32/int64、自定义输出缓冲区 |
| 数值 | 普通分布、重复值、`+inf`、`-inf`、NaN |
| 布局 | 连续输入、满足对齐的非零行 stride |

## 正确性判定

对于不排序结果，至少检查：

- 每行选中元素数量正确；
- 选中索引在有效范围内；
- 没有非法重复；
- 选中集合与参考实现的 top-k 集合一致；
- 边界元素相等时允许合法的并列选择差异。

参考实现可以使用 `torch.topk`，但对重复值、NaN 和排序差异要单独定义预期。

## 参考

- [DeepSelect README](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/README.md)
- [DeepSelect interface](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/deep_select/interface.py)
- [DeepInfer ROCm PR #204](https://github.com/deepseek-ai/DeepInfer/pull/204)

