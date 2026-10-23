# DeepInfer 学习笔记（一）：从“你好”到“好呀”，推理引擎里发生了什么

> 本笔记是 `turisto43/deepinfer-hip-learning` 的**理论框架学习线**产出，与该仓库的 HIP/ROCm 迁移线互补：
> 迁移线关注“怎么把内核搬到 AMD 上”，本线关注“这个推理引擎到底在干什么”。
> 面向读者：**推理引擎零基础**，看得懂 Python、没读过 vLLM / CUDA 也能跟。
>
> **版本基准：** DeepInfer `main@9bd9cf2`（2026-10-22 快照）
> **核对日期：** 2026-10-24
>
> 姊妹篇“生产系统与加速机制”仍在整理中，暂未发布。

## 目录

- [一、DeepInfer 在开源系统中的位置](#一先搞清楚-deepinfer-在开源系统里的位置)
- [二、一个 token 的完整旅程](#二一个-token-的完整旅程)
- [三、四个最小心智模型](#三四个最小心智模型)
- [四、加速机制索引](#四加速机制索引本篇不展开)
- [五、源码地图](#五源码地图)
- [六、术语速查](#六术语速查)
- [七、本篇的未解问题](#七本篇的未解问题)
- [参考](#参考)

## 关于这篇笔记

我是从零开始的：知道大模型是什么、写过 Python，但没读过任何推理框架源码。文中所有“我一开始以为 X”的地方都是真实踩过的认知坑，不是修辞。

**可信度标记**（沿用仓库的记录原则）：

| 标记 | 含义 |
|---|---|
| 🟢 **官方** | 来自 DeepInfer 仓库源码 / README / `doc/inference_deployment_qa.md`、模型 config 与技术报告 |
| 🟡 **二手** | 第三方技术解读，逻辑自洽但未逐行对照源码 |
| ⚪ **推断** | 本人的理解或类比，尚未验证 |

**来源写法**：每节只保留最关键的 1～3 个来源，直接给可点击链接，不用编号脚注。二手文章只用来补充解释，不支撑核心事实。

**验证状态**：当前阶段尚未具备完整运行条件，所有结论都是静态阅读产物。等后续开始贴运行结果，会加上“本地运行过”的区分。

---

## 一、先搞清楚：DeepInfer 在开源系统里的位置

### 1.1 我一开始以为：这下能一键部署聊天服务了

**这是错的。** DeepInfer 的 README 自己就写明了：

> This open-source release provides **offline inference** with the core features needed for serving at scale.
> —— [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md) 🟢

它还单列了一节 *“Required for a complete inference service (not included in this release)”*，包含三项 🟢：

1. 一个 KV cache 系统
2. VE / Prefill / Decode 分离部署——注意后面紧跟一句 **“most required engine functionality is already implemented”**
3. Prefill 和 Decode 各自在 DP 副本间的负载均衡

第 2 条那半句很容易被漏掉，但它决定了我们该怎么理解这个仓库：**PD 分离不是“完全没有代码”，而是引擎侧能力大部分已在，缺的是完整的服务编排**。仓库里确实存在 `deepinfer/engine/engine_type.py`，引擎是有类型之分的 🟢。

> 💡 一句话概括（这是我自己的说法，不是 README 原话）：**给了发动机和大部分传动，没给车身、车轮和方向盘。**

### 1.2 必须分清的两条路径

这是本篇最重要的一张图。生产和离线是**两条不同的路径**，把它们画成同一条无条件链路，是理解 DeepInfer 时最容易犯的错。

```text
【仓库内的离线路径】—— 这篇笔记的主线
  用户输入（str）
    → Hugging Face tokenizer 编码
    → LLMEngine → Scheduler → Worker → 模型前向
    → 采样 → tokenizer 解码 → 输出文本

【生产服务路径】—— 不在这次开源范围内，后续笔记展开
  API 请求
    → deepseek-recipe 做协议与 token 编解码
    → 路由 / KV Cache 前缀命中
    → E / P / D 分离部署的引擎集群
    → 流式返回
```

**关键证据**：离线路径上，分词根本不经过 `deepseek-recipe`。`LLM._add_request()` 里直接就是：

```python
prompt_token_ids = self.llm_engine.tokenizer.encode(prompt)
```

—— [`deepinfer/entrypoints/llm.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/entrypoints/llm.py) 🟢

`deepseek-recipe` 是**另一个独立仓库**（MIT 协议），负责 Chat Completions / Responses / Messages 等 API 格式与模型 token 序列之间的双向转换 🟢。它是给“要搭线上服务的人”用的，不是离线主线的必经环节。

### 1.3 仓库里到底有什么

单次推理所需的**引擎内**组件是齐的 🟢：

| 能力 | README 中的表述 |
|---|---|
| 算子 | FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect + 仓库内 TileLang 算子 |
| 模型与采样器运行时 | Optimized kernels, model and sampler runtime |
| 并行 | Attention DP / balanced CP、MoE EP + EPLB |
| 投机解码 | DSpark + confidence-scheduled verification |
| 缓存优化 | SWA bounded replay |
| 调度 | Asynchronous scheduling（zero-overhead batch scheduler） |
| 确定性 | Batch-invariant inference、Deterministic sampling |
| 结构化输出 | regex / JSON Schema / EBNF 约束生成 |

也就是说，**单次推理需要的 paged KV Cache、`CacheEngine`、block 管理，是仓库内的东西**。

### 1.4 仓库外要自己补什么

未开源的是**跨请求、跨机器**的那一层：

- 跨请求的持久 KV Cache 存储与 3FS 集群
- 完整的 E/P/D 生产调度链路与服务编排
- 四套负载均衡（Prefill / Decode / Vision Encoder / Expert-Parallel）

这些是“从能跑一次，到扛住大规模在线流量”之间的差距，将在后续的生产系统笔记中展开。

### 1.5 并行规模怎么配

这里我之前写错过，更正如下 🟢：

**离线示例是显式命令行参数**，不是从环境推导：

```python
parser.add_argument("--ep", default=8, required=False, type=int)
parser.add_argument("--dp", default=None, required=False, type=int)
parser.add_argument("--cp", default=1, required=False, type=int)
...
dp_size = max(ep_size // cp_size, 1) if args.dp is None else args.dp
world_size = dp_size * cp_size
assert world_size % ep_size == 0
```

—— [`examples/offline_inference.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/offline_inference.py)

从 `WORLD_SIZE × visible devices` 反推 `ep_size` 的做法，只出现在批量跑实验的 benchmark 脚本里。**两者不能混为一谈。**

### 1.6 硬件与模型

- 硬件：NVIDIA SM100 GPU、华为昇腾 950 NPU 🟢
- 模型：目前只有 DeepSeek-V4.1-Flash（默认 checkpoint 为 `DeepSeek-V4.1-Flash-1017`）🟢
- 构建要求：Linux、GCC ≥ 13、Python 3.12、CUDA 13 / CANN 9.2 🟢

---

## 二、一个 token 的完整旅程

下面以“用户说『你好』，模型回『好呀』”为例，走一遍**离线路径**的完整链路。这是全文主线。

### 全景图

```text
用户："你好"
  │
  ├─ ① 文本 → token ID（HF tokenizer 编码）
  │
  ├─ ② 请求抵达 LLMEngine，调度器接手
  │     决定这一轮算什么、分配 KV Cache block
  │
  ├─ ③ Embedding：token ID → 高维向量
  │
  ├─ ④ Prefill：整个 prompt 一次性前向，建立 KV Cache
  │     （V4.1 的 CED 架构让大部分 token 只走前半层）
  │
  ├─ ⑤ 采样第 1 个输出 token："好"
  │
  ├─ ⑥ Decode 循环：把"好"追加进 KV，再算下一个 → "呀"
  │
  ├─ ⑦ 命中停止条件
  │
  └─ ⑧ tokenizer 解码 + 返回："好呀"
```

### ① 文本 → token ID

**我一开始以为**这一步就是“分词器”。对离线路径来说基本没错——它就是 `tokenizer.encode()`，把字符串查表换成整数 ID 序列。

但要记住：**在生产路径上，这一层叫 Recipe，分词器只是它的一个零件**。Recipe 还要管 chat template 拼接、对话历史、stop 定义、默认采样参数、工具调用格式。两条路径的复杂度差很多。

顺带两个容易搞混的点：

- **stop token**：是模型输出序列里的终止信号，训练阶段就埋进了模型约定里；调用方还可以**额外追加** stop 字符串。它不是模型自己判断“该说完了”。
- **temperature**：是作用在 logits 上的**外部**参数，不是模型内省出来的。

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

`T → 0` 趋近贪心，`T → ∞` 趋近均匀随机。它由调用方决定（用户传 / 服务端默认 / 上层 Agent 按场景预设）。所谓“严谨问题自动调低、创作自动调高”是**上层策略**，不是模型能力。

### ② 引擎接手：调度器在管什么

调度器在每个 step 要决定：哪些新请求能进 batch、每个请求这一轮算多少 token、KV Cache block 怎么分配回收、Prefill 和 Decode 怎么混批。

**我一开始以为**调度器只在“很多人同时问”时才有用——这个说法对了一半。**单请求也需要它**：chunk 边界、显存分配、什么时候从 prefill 切到 decode，仍然归调度器管。只是“把多个请求的 decode 步合并成一次大矩阵乘法”这件事，要在并发时才体现价值。

### ③ Embedding：整数怎么变成向量

token ID 进模型第一件事是**查表**：一个 `[vocab_size, hidden_dim]` 形状的嵌入矩阵，把每个整数换成一行几千维的浮点数向量。之后所有注意力、FFN 计算都在这个向量空间里进行。

反过来，模型末端是**最终位置的 hidden state 经过 LM head** 投影出 `vocab_size` 维的 logits——注意不是“最后一层直接输出 logits”，中间还有一次 LM head 变换。

### ④ Prefill：一次性把 prompt 读完

把整个 prompt 的所有 token 一起送进模型做一次前向，算出各层各位置的 K/V，写进 KV Cache。

**为什么能一起算**：prompt 全部已知，位置之间只受因果掩码约束（位置 i 只能看到 ≤ i 的位置），不需要等谁，可以打包成一个大矩阵乘法——这正是 GPU 最擅长的事。

**V4.1 的 CED 架构让这一步变轻** 🟢（模型 config 与技术报告）：

- 主干 40 层 = 前 20 层因果 Encoder + 后 20 层 Decoder
- **Decoder 的 global KV 不再由各层自身隐藏状态生成，而是由 Encoder 末层的 hidden state 投影得到**
- 因此 prefill 阶段大部分 token 只经过前 20 层

⚠️ **但这不等于 20 个 Decoder 层完全跳过**：Decoder 侧的 **SWA 仍需要为一个滑动窗口内的 token 逐层计算**。准确的说法是“主体计算量接近减半”，而不是“后一半层不跑”。

配套的还有 CSA2（跨层共享 KV）和 FP4 量化，把 global KV 压到约 **890 B/token**。这部分将在后续的生产系统与加速机制笔记中展开。

### ⑤ 第一个输出 token 怎么冒出来

最终位置的 hidden state 经 LM head 得到 logits，然后：

```text
logits → temperature 缩放 → top_k / top_p 截断 → softmax → 采样 → token ID
```

采样出 `ID("好")`，这就是第一个输出 token。

### ⑥ Decode 循环

进入自回归循环，每一步：

```text
输入：上一步刚生成的 1 个 token
  → 生成 q
  → 用 q 读【全部历史 KV Cache】做注意力
  → 过 FFN / MoE
  → 输出 logits → 采样 → 下一个 token
→ 把新 token 的 K/V 追加进 KV Cache，循环
```

**我一开始以为** decode 是“把上一步输出拼进输入重跑一遍全文”。**其实不是**——靠 KV Cache，历史位置不重算，只算新 token 的 q 然后去读缓存好的 K/V。这正是“首字慢、后续快”的根因。

两个阶段瓶颈完全不同：

| | Prefill | Decode |
|---|---|---|
| 特征 | 算力密集（大矩阵乘法） | **访存密集**（反复读 KV 与权重） |
| 决定什么 | TTFT（首 token 延迟） | ITL / TBT（字间延迟） |

因为“每生成一个词都要把整块历史 KV 读一遍”，decode 的优化方向是**降低单卡要读的权重和 KV 量**——这直接解释了 Wide-EP 等后续机制。

### ⑦ 停止与返回

采样到 stop token 或达到 `max_tokens` 就停止，然后 `tokenizer.decode()` 把 ID 序列翻回文字。

⚠️ 有个工程细节：tokenizer 不是逐 token 完美可逆的，流式输出时单个 token 可能拼不出完整汉字，服务层要做**累积缓冲**。生产路径上 recipe 用跨切片状态机处理这件事。

---

## 三、四个最小心智模型

读完上面，真正需要刻进脑子里的只有四个：

| 概念 | 一句话 | 为什么重要 |
|---|---|---|
| **Prefill** | 整个 prompt 一次性前向，建 KV Cache | 算力密集，决定 TTFT |
| **Decode** | 一次前向产一个 token，增量读写 KV | 访存密集，决定 ITL |
| **KV Cache** | 缓存的 K/V，让历史位置不用重算 | 两阶段的桥梁，也是显存大头 |
| **调度器** | 每个 step 决定“算什么、算多少、给谁算” | 单请求也要，不只是并发才用 |

其余所有优化，都是插在这四个东西上的。

---

## 四、加速机制索引（本篇不展开）

下面这些机制插在主线的某个位置上，**本篇只要求知道它们存在、解决什么问题**，将在后续笔记中逐项展开：

| 机制 | 解决什么 | 插在主线哪里 |
|---|---|---|
| Chunked Prefill | 长 prefill 独占 GPU 造成的队头阻塞 | ② 调度 |
| 异步批调度 | CPU 组批时 GPU 空转 | ② 调度 |
| CSA2 + FP4 | 长上下文 KV 显存压力 | ④ Prefill |
| 持久 KV Cache / 3FS | 跨请求重复 prefill | ④ Prefill 之前 |
| SWA Bounded Replay | SWA 状态缺失时的重算成本 | ④ Prefill |
| DSpark 投机解码 | 一次前向只产一个 token | ⑥ Decode |
| Wide-EP + EPLB | MoE 专家负载不均、单卡访存压力 | ⑥ Decode 的 MoE 部分 |
| 结构化输出 / 批不变 | 输出不合法、同样输入结果不一致 | ⑤ 采样 |

---

## 五、源码地图

后续笔记按这张表往下啃 🟢（仓库实际结构）：

| 路径 | 职责 |
|---|---|
| `examples/` | 离线推理入口，`offline_inference.py` 是最好起点 |
| `deepinfer/entrypoints/` | 用户侧 LLM API，含 `llm.py` |
| `deepinfer/engine/` | `llm_engine.py`（Step 编排）、`engine_type.py`、`arg_utils.py` |
| `csrc/engine/` | C++ 热路径：Scheduler、Sequence、Block、MTP Budget |
| `deepinfer/worker/` | ModelRunner、CacheEngine、Graph |
| `deepinfer/model_executor/models/` | 主模型与 Decoder 层定义 |
| `deepinfer/model_executor/models/v4_modules/` | Attention、Indexer、MoE、YOCO |
| `deepinfer/model_executor/layers/tilelang_ops/` | 仓库内自研 TileLang 算子（CUDA / Ascend） |
| `deepinfer/model_executor/eplb/` | 专家副本、放置与 shuffle 策略 |
| `deep_cluster/` | 集群分布式与并行通信 |
| `vendor/` | 子模块：DeepSelect、FlashMLA、DeepEP、DeepGEMM（+ 两个 Ascend 变体） |
| `doc/` | `inference_deployment_qa.md` 是生产数据的一手来源 |

⚠️ 两处容易记错：

1. **TileKernels 不在 `vendor/` 子模块里**。`.gitmodules` 只列了 DeepSelect、FlashMLA、DeepEP、DeepGEMM、DeepEP-Ascend、DeepGEMM-Ascend；TileKernels 是依赖包 🟢。
2. `vendor/` 是子模块，克隆后要先 `git submodule update --init`。

读代码建议从 `examples/offline_inference.py` **倒着往回追**，比从 `deepinfer/` 顶层顺着读舒服得多。

---

## 六、术语速查

| 术语 | 一句话 |
|---|---|
| **Prefill** | 整个 prompt 一次性前向，建 KV Cache；算力密集，决定 TTFT |
| **Decode** | 一次前向产一个 token，增量读写 KV；访存密集，决定 ITL |
| **KV Cache** | 缓存的 K/V，避免历史位置重算 |
| **TTFT / ITL (TBT)** | 首 token 延迟 / 字间延迟 |
| **Continuous Batching** | 迭代级调度，请求可在任意 step 进出 batch |
| **Chunked Prefill** | 长 prefill 切块，与**其他请求**的 decode 交错 |
| **PD 分离 / EPD** | Prefill 与 Decode（+Vision Encoder）放不同 GPU 池 |
| **CED** | Causal Encoder-Decoder，40 层 = 20 编码 + 20 解码 |
| **YOCO** | You Only Cache Once，Decoder 全局 KV 由 Encoder 末层投影 |
| **CSA2** | 压缩稀疏注意力 2，层间共享 KV |
| **SWA** | Sliding Window Attention，无条件读最近若干 token |
| **DSpark** | DeepSeek 的投机解码，带置信度调度 |
| **EPLB** | Expert Parallel Load Balancer，冗余专家副本 + 重排 |
| **Wide-EP** | Decode 侧把 EP 开到 64～128 路 |
| **Recipe** | API 协议 ↔ token 序列的编解码层（含分词器） |
| **DeepSelect** | 高性能**专用 TopK** 算子（用于 DSA 路由与采样） |

---

## 七、本篇的未解问题

留给后续笔记（也欢迎在 issue 里讨论）：

1. **Encoder 末层 → Decoder 全局 KV 的投影**具体发生在哪个函数？矩阵形状如何？
2. **仓库内 paged KV Cache 的 block 管理**具体怎么组织？和 vLLM 的 PagedAttention 差别在哪？
3. **Chunked Prefill 的默认 chunk 大小**在 DeepInfer 里是多少？环境变量还是硬编码？
4. **DeepSelect 的四类 TopK 调用**分别对应主线的哪个环节？这也是迁移线的当前靶点。
5. **`engine_type.py` 里的几种引擎类型**各自代表什么部署角色？
6. **vLLM 当壳、DeepInfer 当发动机**是否可行？

---

## 参考

**官方一手**（本篇核心事实主要来自这些位置）

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)——开源范围、缺失三项、特性清单、构建要求
- [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)——生产部署 Q&A
- [`examples/offline_inference.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/offline_inference.py)
- [`deepinfer/entrypoints/llm.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/entrypoints/llm.py)
- [`.gitmodules`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/.gitmodules)

**模型侧**

- [DeepSeek-V4.1-Flash-1017 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [模型 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)
- [V4.1-Flash 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

**配套仓库**

- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)——协议与 token 编解码

**本仓库**

- [`turisto43/deepinfer-hip-learning`](https://github.com/turisto43/deepinfer-hip-learning)——记录原则与上游追踪

---

*本篇为学习过程记录，非技术结论。凡标注 ⚪ 推断的内容，请勿直接引用。*
