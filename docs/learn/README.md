# DeepInfer 学习路径

学习内容分为两条主线：

- **理论框架学习：** 先理解 DeepInfer 由哪些模块组成，以及一次推理如何经过调度、执行和算子。
- **ROCm 迁移学习：** 在理解框架之后，逐个检查现有 ROCm 方案、算子契约和验证结果。

建议先阅读 [01 从整体框架开始](01-deepinfer-overview.md)，再选择后面的学习方向。

## 学习顺序

1. [整体框架与学习路线](01-deepinfer-overview.md)
2. [从“你好”到“好呀”：推理引擎里发生了什么](theory/01-token-journey.md)
3. [理论框架学习](theory/README.md)
4. [ROCm 迁移学习](migration/README.md)

理论学习和迁移学习可以交替进行，但每次迁移实验都应能回到一个明确的框架位置。

