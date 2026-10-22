# PR #204 TopK：BF16 Indexer Prefill 正确性与调用契约（进行中）

本文属于 **ROCm 迁移学习 01**，针对 DeepInfer ROCm PR #204 中的 TopK 实现，验证 BF16 indexer prefill 路径是否正确满足 DeepInfer 的调用契约。

**当前状态：进行中。** 已完成 DeepInfer 调用链和接口参数整理；尚未完成 PR #204 的逐项实现对照、HIP 实机正确性测试和性能测试。

| 检查项 | 状态 | 证据 |
| --- | --- | --- |
| DeepInfer indexer prefill 调用点 | 已完成 | `index_attn.py` |
| BF16 TopK 接口契约 | 已完成 | 本文与 `labs/topk/contract.md` |
| PR #204 Python 接口逐项对照 | 进行中 | 待继续补充 |
| PR #204 Triton radix-select 行为核对 | 进行中 | 待继续补充 |
| HIP 实机正确性 | 未开始 | 尚无测试结果 |
| Graph 捕获与重复调用 | 未开始 | 尚无测试结果 |
| 性能 | 未开始 | 尚无测量结果 |

## 专题范围

本专题只负责 BF16 indexer prefill。其他 TopK 路径先作为调用上下文，不在本文提前下结论。

| 路径 | 是否属于本文 | 说明 |
| --- | --- | --- |
| BF16 indexer prefill | 是 | 当前正确性与调用契约验证对象 |
| BF16 indexer decode | 否 | 涉及物理 KV slot，后续单独验证 |
| Candidate block 两阶段 TopK | 否 | 涉及 block 选择和索引恢复，后续单独验证 |
| FP32 sampler force-top-k | 否 | dtype、value 输出和映射方式都不同 |
| FP32 top-logprobs | 否 | 需要按 value 排序 |

## 在整体框架中的位置

DeepInfer 至少有三类 TopK 调用：

| 场景 | 主要输入 | 输出用途 | 关键选项 |
| --- | --- | --- | --- |
| Sparse indexer prefill | BF16 indexer logits | 选出每行要参与稀疏 attention 的压缩 KV | `return_value=False`、可变长度、逻辑索引 |
| Sparse indexer decode | BF16 indexer logits | 选出 paged KV cache 中的物理槽位 | `return_value=False`、结果原地转换 |
| Sampler | FP32 logits/logprobs | 缩小采样空间或返回 top-logprobs | 可能需要 value、可能按 value 排序 |

这三类调用形式相近，但性能瓶颈和边界条件不同。本文先处理 sparse indexer prefill。

## 从请求到 TopK

### 引擎与 worker

高层调用链可以概括为：

```text
LLM.generate
  -> LLMEngine.add_request
  -> LLMEngine.step
  -> C++ scheduler.schedule
  -> Worker.execute_model_step
  -> ModelRunner.execute_main_model
  -> 模型 forward
  -> attention / indexer
  -> deep_select.topk
  -> attention / sampler
```

`Worker.execute_model_step` 的一个 step 分为四个阶段：

1. 执行当前 step 的主模型 forward；
2. 后处理上一步；
3. 对当前 step 的模型输出采样；
4. 后处理当前 step。

异步调度开启时，decode 的后处理可以推迟到下一步，与下一次 forward 重叠。因此 TopK 实现即使数值正确，也不能引入意外的同步，否则会破坏这条流水线。

### Sparse indexer

`deepinfer/model_executor/models/v4_modules/index_attn.py` 中的 `MQA` 负责稀疏注意力 indexer：

1. 计算 query 与压缩 KV 的 logits；
2. 对每行选择 `topk_tokens` 个候选；
3. 把索引传给 sparse attention；
4. decode 时再把逻辑索引转换到 paged KV cache 的物理槽位。

Prefill 路径的核心调用形态是：

```python
topk(
    p_sum,
    self.topk_tokens,
    sorted=False,
    end=topk_lens[chunk],
    indices_type=torch.int32,
    sorted_index=True,
    output_idx=topk_index[chunk],
    output_idx_offset=ks[chunk] if candidate_blocks is None else None,
    idx_oob_fill_value=-1,
    return_value=False,
)
```

这里有几个迁移时不能忽略的属性：

- `p_sum` 是 BF16；
- `return_value=False`，只需要索引；
- 每行有效长度不同，通过 `end` 传入；
- 输出索引按 index 升序；
- 调用方提供输出缓冲区；
- dense prefill 可能给索引增加序列偏移；
- 无效位置填入 `-1`。

Decode 路径同样返回 `int32` 索引，但之后会调用 `convert_kv_index`，把逻辑压缩位置转换为 KV cache 的物理 slot。TopK 本身返回“序列内位置”并不等于 attention 可以直接读取的最终地址。

### Sampler

`deepinfer/model_executor/layers/sampler_utils.py` 在 `force_topk > 0` 时先缩小 logits 空间：

```python
logits, token_index = topk(
    logits,
    force_topk,
    False,
    indices_type=torch.int32,
)
```

这条路径需要同时返回 value 和 index，输入是 FP32，之后还要用 `token_index` 把采样结果映射回真实词表 ID。

`deepinfer/model_executor/layers/sampler.py` 的 `_prepare_top_info` 还会在请求 top-logprobs 时调用：

```python
top_logprobs, top_token_ids = topk(
    logprobs,
    max_top_logprobs,
    True,
    indices_type=torch.int32,
)
```

这条路径要求按 value 排序，但通常不需要可变长度和逻辑地址转换。

## 后端选择不是发生在 TileLang 分派器

这是第一轮阅读里最重要的发现之一。

普通 TileLang 算子通过 `tilelang_ops/ops.py` 根据 `PLATFORM.tilelang_ops_backend` 分派，但 TopK 在模型代码中直接导入：

```python
from deep_select.interface import topk
```

因此 TopK 的后端接入机制与 TileLang 不同：

- Ascend 路径通过 `deepinfer/npu_ops/patch_imports.py` 拦截 `from deep_select.interface import topk`，替换为 NPU wrapper；
- Ascend indexer 使用真实 TopK kernel；
- Ascend sampler 暂未支持时回退到 `torch.topk`；
- ROCm PR 采用另一条路线：提供 `vendor/rocm/deep_select`，用 Triton radix-select 实现 indexer 路径，FP32 全行采样部分仍可使用 `torch.topk`。

所以在 HIP 迁移中，首先需要确定是复用 vendor 包接口、增加平台 wrapper，还是把 TopK 纳入统一分派。不能只搜索 CUDA 源码并逐行 hipify。

## PR #204 当前对照基线

根据 PR #204 的公开描述和固定 head，当前实现基线与待核查项如下：

| 项目 | 公开方案 | 本文处理状态 |
| --- | --- | --- |
| 实现位置 | `vendor/rocm/deep_select/` | 待逐文件对照 |
| Indexer 算法 | Triton radix-select | 待阅读 `_radix.py` |
| FP32 采样 | 部分调用 `torch.topk` | 不属于本文主范围 |
| 目标设备 | gfx950 环境 | 其他设备未验证 |
| NaN | 公开接口注明不检测 | 待确认是否影响 indexer 契约 |
| 正确性 | PR 内有 gfx950 测试 | 需核对实际执行和未覆盖场景 |

这部分仍是阅读状态，不能据此断言实现有缺陷。需要先把调用参数与实现细节逐项对齐，再给出结论。

## HIP 实现要跨过的四层边界

### 1. Python 接口边界

签名、默认参数、可变长度、输出缓冲区和索引类型必须兼容。仅实现一个 `topk(x, k)` 不够，因为 DeepInfer 会直接使用这些细节。

### 2. 数值与布局边界

Indexer 是 BF16，sampler 是 FP32。输入行 stride、输出 stride、输出对齐、并列值、`inf` 和 NaN 的行为都可能影响调用方。

### 3. 执行边界

TopK 位于模型 forward 和可能的 CUDA/HIP Graph 捕获范围内。实现应避免隐式同步和不可捕获的设备操作。decode 路径还会原地修改索引，因此输出别名和生命周期也要验证。

### 4. 语义边界

Prefill 索引是压缩 token 的序列位置，decode 输出会在后续转换成 KV cache 物理槽位。测试如果只比较“每行最大的 K 个数”，仍可能漏掉偏移和无效索引处理错误。

## 当前专题验证清单

本文只完成第一组，其他路径只保留为后续专题：

1. **Indexer prefill 最小用例**
   - BF16；
   - `return_value=False`；
   - `sorted_index=True`；
   - 可变 `end`；
   - `int32`；
   - 无效位置为 `-1`。

2. **Indexer decode 形状**
   - 小 batch 的 paged KV 场景；
   - 检查逻辑索引到物理槽位的转换；
   - 检查 Graph 捕获和重复调用。

3. **Candidate block 路径**
   - block size 为 8 或 16；
   - 先选候选 block，再在缩窄范围内选 token；
   - 检查 `output_idx_offset` 和索引恢复。

4. **Sampler FP32**
   - `force_topk` 路径需要 value 和 index；
   - top-logprobs 路径需要 value 排序；
   - 与前三条路径分开记录。

这套顺序与 DeepInfer 的真实调用比例和失败影响相匹配，也能直接补充到 [TopK 接口契约](../../../labs/topk/contract.md)。

需要逐项验证的契约包括：

- [x] 调用点已定位；
- [x] 输入 dtype 和输出 dtype 已确认；
- [x] `end`、`output_idx` 和 `output_idx_offset` 的调用方式已记录；
- [ ] PR #204 的 Python 接口逐参数一致；
- [ ] 输出对齐要求与 DeepInfer 调用方一致；
- [ ] 可变长度和不足 `topk` 的填充一致；
- [ ] `sorted_index=True` 的语义一致；
- [ ] 无 value 输出路径没有额外计算；
- [ ] 无隐式设备同步；
- [ ] 在目标 HIP 环境实际运行；
- [ ] 对照 `torch.topk` 验证正确性；
- [ ] 记录性能。

## deepinfer-unpacked 可以怎样使用

[deepinfer-unpacked](https://github.com/ZiyuanDu/deepinfer-unpacked) 提出了清晰的模块编号：

- M03：调度与执行；
- M06：算子与内核；
- M11：外部组件；
- M12：工程与平台。

它的模块表适合作为仓库导航。当前项目仍处于启动阶段，源码结论仍应回到 DeepInfer 固定 commit 核对。

## 下一步

继续补充本专题，先对照 DeepInfer ROCm PR #204 的实现：

1. 阅读 `vendor/rocm/deep_select/interface.py` 和 `_radix.py`；
2. 把每个参数映射到 DeepInfer 的 prefill 调用；
3. 找出未实现或被忽略的行为；
4. 把待测项同步到 `labs/topk/contract.md`；
5. 获得 HIP 环境后先跑最小用例，再跑候选 block 和大形状。

在完成逐项实现对照前，本文保持“进行中”，不把 PR 描述中的测试结果当作个人验证结果。

## 参考源码

- [index_attn.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/index_attn.py)
- [attention_backend.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/attention_backend.py)
- [sampler_utils.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py)
- [sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py)
- [npu_ops/deep_select.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/npu_ops/deep_select.py)
- [worker.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)
- [model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py)

## 继续学习

- [ROCm 迁移学习](README.md)
- [理论框架学习](../theory/README.md)
- [整体框架与学习路线](../01-deepinfer-overview.md)

