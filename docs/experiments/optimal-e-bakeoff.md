# 最优 E 十镜头先导实验报告

> 本文保留第一轮 Bedroom 单场景先导结果。后续 11 场景、114 组正式结论见
> [Optimal E 多场景优化器对照实验](optimizer-comparison.md)。

## 结论与范围

这是一轮**单场景先导实验**，不是正式的全场景统计实验：

- 场景：默认 `bedroom.scene`
- 镜头：10 个确定性的 yaw / pitch / dolly 视角
- 每个镜头：1,000,000 条生产路径
- 优化预算：CUDA 与 PyTorch 各 20 步
- 随机性：当前实验入口没有显式 seed offset，因此没有形成“3 个独立种子”
- 未执行：把两个 final q 装回 renderer 后比较画面方差或 MSE

这轮实验的价值是验证数据导出、目标函数、梯度和两个优化器的闭环，并观察
不同镜头下的优化趋势；它不能代替“多场景 × 多镜头 × 多种子”的正式结论。

## 实验设置

- 两个优化器读取同一份真实 `.spcoe` snapshot。
- 两边使用完全相同的初始 `base_q`。
- 生产端使用 CUDA simplex mirror descent + backtracking。
- PyTorch 端在 CUDA 上使用 logits + Adam。
- conservative mixture、active-light mask 和 loss 定义保持一致。

## A 层：实现正确性

10/10 组均通过两向 objective 对拍。

| 检查 | 最大绝对误差 |
|---|---:|
| PyTorch 复算 CUDA 初始 q 的 objective | 2.95e-5 |
| PyTorch 与 CUDA 初始概率梯度 | 6.13e-6 |
| PyTorch 复算 CUDA-final q 的 objective | 1.72e-5 |
| CUDA 复算 PyTorch-final q 的 objective | 1.83e-5 |

最后一项就是补全后的“CUDA 反向评价 PyTorch-final q”：PyTorch 将 final q
以 little-endian float32 写出，生产 CUDA objective evaluator 读取同一个 q
重新计算 loss。误差包含 PyTorch float64 到生产 float32 的转换误差。

这说明两个实现对目标函数和概率表示的理解已经对齐；它不证明任一优化器
在 20 步内到达全局最优点。

## B 层：优化效果

| 镜头 | CUDA final loss | CUDA 降幅 | PyTorch final loss | PyTorch 降幅 | CUDA 反算 PyTorch q 误差 |
|---|---:|---:|---:|---:|---:|
| base | 53.6213 | 1.98% | 48.2961 | 11.72% | 1.06e-6 |
| yaw_m12 | 979.1412 | 1.79% | 951.4295 | 4.57% | 1.83e-5 |
| yaw_p12 | 103.7825 | 2.14% | 96.9965 | 8.54% | 2.15e-6 |
| yaw_m24 | 464.0024 | 1.26% | 446.4751 | 4.99% | 1.47e-5 |
| yaw_p24 | 168.3596 | 1.14% | 159.8136 | 6.15% | 3.60e-6 |
| pitch_m10 | 9.1468 | 2.73% | 7.0593 | 24.93% | 1.97e-7 |
| pitch_p10 | 13.9734 | 1.76% | 11.1182 | 21.83% | 4.20e-7 |
| dolly_in | 15.9634 | 2.01% | 12.5283 | 23.10% | 1.28e-7 |
| dolly_out | 92.7617 | 1.24% | 86.5537 | 7.85% | 3.47e-6 |
| yaw_p18_pitch_p7 | 45.8581 | 1.32% | 41.2271 | 11.29% | 1.87e-6 |

- CUDA mirror 平均相对下降：`1.74%`
- PyTorch Adam 平均相对下降：`12.50%`
- PyTorch 取得更低 20-step final loss：`10/10` 镜头

现有证据更支持“生产 mirror 的步长/backtracking 策略偏保守”，而不是
“生产 CUDA 公式写错”。下一步若要改生产优化器，应先调整优化策略并继续
使用两向 objective 对拍，不需要把 LibTorch 链入 renderer。

## 时间与数据量

- 平均真实路径采集：`6.24s/组`
- 平均生产 CUDA optimizer/validator：`0.252s/组`
- 平均 PyTorch 对拍 + optimizer：`3.24s/组`
- 平均第二次 CUDA validator：`0.252s/组`
- 当前 10 组 artifact：`284,508,673` bytes，约 `271 MiB`

这些是完整进程墙钟时间，不是隔离后的 kernel benchmark。第二次 CUDA
validator 为了复用现有 test target，会重新读取 snapshot 并重复一次生产
optimizer，然后再评价 PyTorch-final q；`0.252s` 不是单独 objective kernel
的耗时。

## 正式实验矩阵

目标矩阵应改为：

```text
有效场景 manifest × 每场景 5 个代表镜头 × 3 个显式 seed
```

当前 `assets/` 下共有 68 个 `.scene` 文件。若不筛选，理论上是：

```text
68 × 5 × 3 = 1020 组
```

按先导实验估算，1020 组约需 `29 GB` artifact、顺序执行约 `2.8 小时`；
复杂场景可能明显超过该时间。

但 68 个文件包含 `old`、`test`、`副本` 和同场景的多个实验变体，因此不能
直接当作 68 个独立有效场景。正式运行前还缺两项：

1. 建立经过 headless smoke 的有效场景 manifest。
2. 给采样 seed 派生增加显式 `experiment_seed`，并把 seed 写入 snapshot、
   summary 和 manifest；简单重复启动当前程序不能算 3 个独立种子。

每个场景的 5 个镜头也应基于该场景自己的基准相机和边界生成，不能复用
bedroom 的绝对坐标。

## 结果位置

- 镜头清单：`build/optimal-e-bakeoff-bedroom/cameras.json`
- 汇总 JSON：`build/optimal-e-bakeoff-bedroom/results.json`
- 汇总 CSV：`build/optimal-e-bakeoff-bedroom/results.csv`
- 每组目录保存 snapshot、CUDA result、PyTorch q、float32 q、
  CUDA 反向 objective 和 summary。
