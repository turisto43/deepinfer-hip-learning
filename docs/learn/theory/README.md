# 理论框架学习

这条路线回答“DeepInfer 为什么这样组织，以及一次推理怎样运行”。先读框架，再进入缓存、模型、算子和并行机制。

## 当前顺序

| 序号 | 主题 | 状态 | 入口 |
| --- | --- | --- | --- |
| 01 | 整体框架与学习路线 | 已完成初稿 | [整体框架](../01-deepinfer-overview.md) |
| 02 | 系统模块与目录职责 | 已有参考笔记 | [系统地图](../../architecture/system-map.md) |
| 03 | 后端接缝与构建分派 | 已有参考笔记 | [后端接缝](../../architecture/backend-seams.md) |
| 04 | 请求、调度与 Worker 执行链 | 阅读中 | 后续专题 |
| 05 | 模型加载与层结构 | 待开始 | `deepinfer/model_executor/` |
| 06 | KV cache 与资源管理 | 待开始 | `worker/cache_engine.py`、`csrc/engine/` |
| 07 | TileLang 与 vendor 算子 | 待开始 | `tilelang_ops/`、`vendor/` |
| 08 | 并行、通信与高级机制 | 待开始 | `parallel_utils/`、`eplb/`、`deep_cluster/` |

## 当前重点

下一步先把请求、调度和 Worker 执行链写成一篇完整源码笔记，弄清一次 `step()` 中 forward、sample 和后处理的关系。

## 写作要求

- 每个结论标明对应的源码文件；
- 区分设计意图和运行结果；
- 用调用链解释模块，而不是只列目录；
- 对尚未阅读的模块保留“待开始”，不提前拼凑结论。

## 返回

- [学习路径首页](../README.md)
- [ROCm 迁移学习](../migration/README.md)

