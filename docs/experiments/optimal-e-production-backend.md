# Optimal E 生产优化后端选择

## 当前判断

正式 114 组实验已经确认：PyTorch softmax Adam 平均降低 loss `20.62%`，
当前 CUDA mirror 为 `4.96%`；对齐 `lr=0.05` 的修复旧 CUDA Adam 已达到
`19.55%`。后续学习率 sweep 又确认，CUDA mirror 使用 `lr=1.0` 时可达到
`18.89%`，逐样本从五个 mirror 学习率中选优可达到 `20.06%`。详细数据见
[多场景优化器对照实验](optimizer-comparison.md)和
[Optimal E 执行流程](optimal-e-optimizer-workflow.md)。

当前工程先采用依赖最少的 production mirror `lr=1.0`；test-only legacy
normalized-sigmoid Adam 不进入 renderer。softmax-logits Adam 保留为后续
确有需要时的专用 CUDA 替换路线。

推荐的生产路线是：

```text
保留 Python/PyTorch 作为数学参考与调参工具
                    ↓
在现有 CUDA 目标函数/梯度上实现 softmax-logits Adam
                    ↓
继续用同一 .spcoe 做两向 objective 对拍和多场景回归
```

不建议把 Python 解释器、LibTorch 或导出的 PyTorch 运行时直接放进 renderer。
原因不是这些方案不能工作，而是当前问题还没有复杂到值得承担它们的依赖成本。
第一版替换还应保留调好步长的 mirror 候选：固定 `lr=1.0` 已在 64/114 组
低于 PyTorch，按同一 objective 选优可以吸收两种 optimizer 的互补结果。

## 这个问题并不是神经网络训练

当前待优化量是一个约 `300 × 300` 的逐行概率矩阵 `q`。真实路径样本给出
目标函数：

\[
L(q)=\sum_i
\frac{f_i^2}
{p_{0,i}+\sum_{k\in i} a_k E_{r_k,c_k}(q)}
\]

其中 `E` 是 `q` 与 conservative uniform distribution 的混合。对可行概率
矩阵而言，分母是 `q` 的仿射函数，`1/x` 在正数域上是凸函数，因此整个目标
在概率单纯形上是凸的。

这带来三个直接结论：

1. 不需要神经网络层，也不需要矩阵乘法框架；主要工作是稀疏路径累加、
   概率梯度、逐行归一化和逐元素 optimizer update。
2. 样本不充分时，最优 `q` 不一定唯一：如果不同的 `q` 在已有路径上产生相同
   的有效密度，它们可以有相同 loss。我们应比较 objective 和独立样本上的
   渲染效果，而不是要求两套 optimizer 返回逐元素相同的 `q`。
3. PyTorch 更低的 final loss 说明它当前的参数化、Adam 更新或超参在固定预算
   内更有效，不自动说明 autograd 或整套 PyTorch runtime 是必要条件。

## 三条部署路线

| 路线 | 能否做 | 当前适合度 | 判断 |
|---|---|---:|---|
| Python 导出可链接库 | 可以探索 `torch.export` / AOTInductor | 低 | 更适合冻结计算图的非 Python 部署；当前是带 optimizer state 和多轮更新的在线优化 |
| LibTorch C++ frontend | 可以，含 CUDA tensor、autograd、Adam | 中低 | 功能完整，但把大型 tensor/autograd/runtime 依赖带入 renderer，只为一个固定概率矩阵 |
| 现有 CUDA 中实现 PyTorch 同款数学 | 可以 | **高** | 只需 softmax、链式梯度、Adam moments 和更新；与现有数据布局、测试和构建最贴合 |

PyTorch 官方把 C++ frontend 定位为 CPU/GPU tensor、autograd 和
`torch::optim` 的完整 C++ 训练接口，同时说明它偏向灵活性和易用性，有时会
牺牲微观优化；C++ API 目前仍标为 beta 稳定性。
[C++ frontend](https://docs.pytorch.org/cppdocs/frontend.html)、
[C++ API](https://docs.pytorch.org/cppdocs/)

把 Python 模型变成 C++ 可加载产物时，不应再选择 TorchScript：PyTorch 已
明确标记其 deprecated，并推荐 `torch.export`。AOTInductor 可以为非 Python
环境生成共享库，但官方文档的主要场景是部署已导出的模型图，而不是替 renderer
承载一段动态的 optimizer 训练循环。
[TorchScript 状态](https://docs.pytorch.org/docs/2.9/jit.html)、
[AOT 编译与 AOTInductor 的边界](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_compile.html)

## 推荐实现的准确形态

不要把测试中的“修复后旧 normalized-sigmoid Adam”直接当最终生产算法。
那个 target 的目的，是还原旧算法家族、修掉已知链式梯度和边界错误，回答
旧实现失败究竟来自公式还是优化策略。

若正式实验确认 Adam 路线更好，生产实现应对齐当前 PyTorch reference：

1. 用 active mask 下的 `softmax(logits)` 得到每行 `q`。
2. 继续调用现有 CUDA objective 和 `dL/dq`。
3. 使用
   \[
   \frac{\partial L}{\partial z_j}
   =q_j\left(g_j-\sum_k q_k g_k\right)
   \]
   将概率梯度变成 logits 梯度。
4. 保存两个同尺寸数组 `m/v`，执行 Adam bias correction 和逐元素更新。
5. 每轮重新计算 softmax；记录 best loss，并在非有限值或 loss 明显恶化时回退。
6. 先严格对拍 20 步，再单独调步数、学习率和停止条件。

PyTorch 的 Adam 文档给出了相同的 `m/v`、bias correction 和更新公式；CUDA
版本还提供 foreach/fused 实现，但本项目只有一个固定大小 tensor，直接在已有
CUDA 数据上做少量专用 kernel 更容易控制内存、同步和构建依赖。
[PyTorch Adam](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)

这条路线不需要 cuBLAS：没有稠密 GEMM。若后续要优化逐行 reduction，可以在
profile 证明必要后使用 CUB block reduction；Adam 本身只是逐元素运算。

## 何时重新考虑 LibTorch 或编译产物

只有当 Optimal E 后端从“直接优化 90k 个概率”升级为以下形态时，再重新评估：

- 用真正的 MLP/编码网络从场景或路径特征预测 `q`；
- 模型结构需要频繁试验，手工同步 forward/backward 已成为主要维护成本；
- 训练不再位于交互式 renderer 热路径，能够接受独立训练进程或较大的 runtime；
- AOT 导出的固定推理图可以替代在线训练。

在这些条件出现以前，最稳妥的分工是：PyTorch 负责 oracle、数学检查和超参
探索；renderer 负责经对拍验证过的精简 CUDA optimizer。
