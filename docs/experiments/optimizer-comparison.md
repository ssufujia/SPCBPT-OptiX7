# Optimal E 多场景优化器对照实验

> 后续 CUDA mirror 学习率 sweep 与三类优化器的完整公式/执行流程见
> [Optimal E：样本、损失、梯度与三类优化器](optimal-e-optimizer-workflow.md)。
> 下表保留 sweep 前 `lr=0.01` 的正式四路实验基线；后续根据五档 sweep，
> renderer 已选择仍属同一算法的 mirror `lr=1.0` 作为 production 默认值。

## 结论

正式实验已完成 `11 scenes × 38 cameras × 3 seeds = 114` 组真实数据：

- 114/114 组通过 CUDA/PyTorch objective、初始梯度和反向 objective 对拍。
- PyTorch softmax-logits Adam 平均降低 loss `20.62%`，在 109/114 组严格最优。
- sweep 前 production mirror descent 平均降低 `4.96%`，但在 5 个特定镜头/seed 上
  仍优于 PyTorch，说明替换时应暂时保留 mirror fallback。
- 修复后的旧 CUDA normalized-sigmoid Adam 使用历史 `lr=0.01` 时平均降低
  `6.59%`；只把学习率对齐到 `0.05` 后达到 `19.55%`。
- 后续 sweep 将 production mirror 的学习率提高到 `1.0` 后，平均降幅达到
  `18.89%`，与旧 CUDA Adam `lr=0.05` 的 `19.55%` 已属同一水平。
- 同口径的 33 组墙钟样本中，mirror `lr=1.0` 平均耗时 `0.277850 s`，
  旧 CUDA Adam `lr=0.05` 为 `0.280028 s`；mirror 约快 `0.8%`。
- 对齐步长的旧 CUDA 与 PyTorch 已很接近，但 PyTorch 在 114/114 组的
  final loss 都更低；差距平均为初始 loss 的 `1.08` 个百分点。

因此当前生产端保留实现更简单、同口径稍快的 CUDA mirror，并固定使用
`learning_rate=1.0`。PyTorch 的主要价值仍是数学参考：它证明
**softmax 参数化 + Adam + 合适步长** 能在这批数据上取得略低 loss，但结果
不支持为了这一个固定概率矩阵把完整 LibTorch 放进 renderer。若未来画质或
更大数据集证明这点 loss 差异值得追求，再在现有 CUDA objective/gradient
上实现 softmax-logits Adam。后端选型的完整理由见
[Optimal E 生产优化后端选择](optimal-e-production-backend.md)。

## 数据范围

| 项目 | 数值 |
|---|---:|
| 场景 | 11 |
| 代表镜头 | 38 |
| 每镜头 seeds | 3（11 / 29 / 47） |
| 实验组数 | 114 |
| 总路径数 | 114,000,000 |
| 总 path nodes | 248,322,062 |
| `.spcoe` snapshots | 3,395,759,680 bytes（约 3.16 GiB） |
| 含 q、日志和对比结果的全部本地产物 | 3,746,910,692 bytes（约 3.49 GiB） |
| 平均真实路径采集时间 | 7.82 s/组 |

场景、相机参数、seed 门禁、Bathroom 资源笔误以及 Conference/Hallway fallback
记录在[正式实验场景与镜头](scene-selection.md)。原始数据只保存在 ignored
`build/experiments/optimal-e-multiscene-v3`，不会上传 GitHub。

## 公平比较设置

四条路线都读取同一份真实 snapshot、同一个初始 `q0`，并使用 20 个 optimizer
steps：

| 名称 | 参数化 / 更新 | 学习率 | 用途 |
|---|---|---:|---|
| Production CUDA mirror（sweep 前基线） | 概率单纯形 mirror descent + backtracking | 0.01 初始值 | 历史正式对照 |
| Fixed legacy CUDA original | normalized sigmoid + 修复后的 exact chain + Adam | 0.01 | 还原旧实现的原始步长 |
| Fixed legacy CUDA matched | 同上 | 0.05 | 只消除与 PyTorch 的步长差异 |
| PyTorch reference | active-masked softmax logits + autograd + Adam | 0.05 | 数学参考和优化上界候选 |

“Fixed legacy” 是受控恢复，不是逐指令复刻历史程序。它保留旧
normalized-sigmoid/Adam 思路，但修复：

- `q=0/1` 时 inverse-logit 产生无穷；
- normalized sigmoid Jacobian 的链式梯度错误；
- inactive light support 未严格屏蔽；
- conservative mixture 被错误带入参数化导数。

它也使用完整 snapshot 的 20-step full-batch 预算，以便和另外两条路线公平
比较；历史代码原有的 mini-batch 轨迹不作为基线。

为避免 float32 CUDA objective 与 float64 PyTorch objective 混在一起决定
胜负，最终排名统一执行：

```text
各 optimizer 的 float32 final q
              ↓
同一个 PyTorch float64 objective evaluator
              ↓
final loss / 相对降幅 / 严格最优计数
```

生产 CUDA objective 仍独立评价 PyTorch-final q；它用于证明公式对齐，不用于
混精度排名。

## A 层：实现正确性

| 对拍项 | 114 组最大绝对误差 |
|---|---:|
| PyTorch 复算 CUDA 初始 objective | 3.32e-4 |
| PyTorch autograd vs CUDA 初始 `dL/dq` | 2.42e-4 |
| PyTorch 复算 production CUDA-final q | 1.07e-4 |
| CUDA 反向复算 PyTorch-final q | 1.01e-4 |
| PyTorch 复算 fixed legacy original-final q | 3.74e-4 |
| PyTorch 复算 fixed legacy matched-final q | 1.15e-4 |

全部误差都通过既定的绝对 + 相对容差。正式场景的 loss 尺度差异很大，因此
最大绝对误差高于 Bedroom 先导实验；这不是失败或公式漂移。

这层证明目标函数、概率表示、active mask、conservative mixture 和梯度实现
对齐。它不证明某个 20-step optimizer 已经到达唯一全局解。

## B 层：总体优化效果

| 方法 | 平均降幅 | 中位降幅 | 最小—最大降幅 | 严格最优组数 |
|---|---:|---:|---:|---:|
| Production CUDA mirror | 4.96% | 2.08% | 0.002%—24.07% | 5 / 114 |
| Fixed legacy original | 6.59% | 5.08% | 0.44%—15.30% | 0 / 114 |
| Fixed legacy matched | 19.55% | 16.67% | 1.43%—43.88% | 0 / 114 |
| PyTorch softmax Adam | **20.62%** | **18.84%** | 1.81%—46.03% | **109 / 114** |

补充的成对结果：

- PyTorch final loss 低于 fixed legacy matched：114/114。
- PyTorch final loss 低于 production mirror：109/114。
- Fixed legacy matched 低于 production mirror：108/114。
- Fixed legacy original 低于 production mirror：88/114。
- PyTorch 相对 fixed legacy matched 的优势，按初始 loss 归一后平均
  `1.08` 个百分点，中位 `0.76`，范围 `0.10`—`7.60`。

## 分场景结果

表内是每个场景所有镜头和 seeds 的平均相对 loss 降幅。

| 场景 | 组数 | Mirror | Legacy 0.01 | Legacy 0.05 | PyTorch | 严格最优 |
|---|---:|---:|---:|---:|---:|---|
| Bathroom | 12 | 1.42% | 2.17% | 7.31% | **8.31%** | PyTorch 12 |
| Bedroom | 9 | 1.14% | 2.03% | 6.44% | **6.69%** | PyTorch 9 |
| Breakfast | 9 | 5.32% | 6.16% | 19.43% | **21.19%** | PyTorch 8 / Mirror 1 |
| Conference | 12 | 7.46% | 11.90% | 34.38% | **34.94%** | PyTorch 12 |
| Cornell box | 6 | 1.09% | 7.37% | 20.30% | **22.68%** | PyTorch 6 |
| Glassroom | 15 | 8.06% | 14.03% | 40.28% | **41.27%** | PyTorch 15 |
| Hallway | 12 | 12.84% | 12.45% | 35.10% | **36.30%** | PyTorch 12 |
| Kitchen | 15 | 0.72% | 1.90% | 6.95% | **7.15%** | PyTorch 15 |
| Projector | 9 | 0.82% | 1.07% | 3.37% | **4.20%** | PyTorch 9 |
| Showcase | 6 | 0.20% | 1.55% | 4.48% | **4.74%** | PyTorch 6 |
| White room | 9 | 11.12% | 6.36% | 20.73% | **24.12%** | PyTorch 5 / Mirror 4 |

PyTorch 并非每个样本都赢。Mirror 严格更低的 5 组是：

- White room：base seed 29；
- White room：yaw -6° seed 11；
- White room：yaw +12° seeds 29 / 47；
- Breakfast：authored alternative seed 29。

这 5 个反例意味着生产替换不应删除 mirror。第一版 softmax Adam 可与 mirror
并行产生候选，再用已有 CUDA objective 选择较低 loss；optimizer 耗时远小于
路径采集，先换稳定性比过早省掉一次候选更重要。

## 如何理解结果

### 1. 现有 CUDA 公式没有写错

A 层已在 114 份生产 snapshot 上通过。PyTorch 更低主要是 optimizer 行为，
不是 loss 或梯度定义不同。

### 2. 收益大部分来自 Adam 和学习率

旧 CUDA Adam 从 `lr=0.01` 改到 `0.05`，平均降幅从 `6.59%` 提升到
`19.55%`，已经接近 PyTorch 的 `20.62%`。因此没有证据表明必须依赖 autograd
runtime 才能取得好结果。

### 3. softmax 参数化仍然值得正式实现

即使学习率一致，PyTorch 仍在 114/114 组低于 normalized-sigmoid 版本。
差距通常不大但稳定存在，说明下一版 CUDA 应对齐 softmax logits，而不是把
测试中的旧 normalized sigmoid 原样搬回生产。

### 4. 当前不需要画面 MSE 层

本轮目标是判断 objective/gradient 是否正确，以及哪个 optimizer 在同一真实
样本上找到更低 loss。把两个 q 再装回 renderer、用独立路径样本比较画面
方差/MSE 会显著扩大实验成本；它适合作为生产替换完成后的最终渲染验收，
不是决定后端依赖的前置条件。

## 生产选型计时

旧记录中的 production CUDA `0.270 s/组` 是 validator 完整进程墙钟，而
fixed legacy CUDA `0.051 s/组` 只计内部 20-step optimizer loop，不能直接
用于选型。为此补做了同一计时边界的配对实验：

- 从 11 个场景各取一份真实 snapshot；
- 两个可执行程序各预热一次；
- 都运行 20 optimizer steps，每个场景重复 3 次；
- 统一统计独立 validator 进程完整墙钟，共 33 组配对样本。

| 方法 | 学习率 | 平均墙钟 | 中位墙钟 | 配对胜出 |
|---|---:|---:|---:|---:|
| Production CUDA mirror | 1.0 | **0.277850 s** | **0.276097 s** | **17 / 33** |
| Fixed legacy CUDA Adam | 0.05 | 0.280028 s | 0.280416 s | 16 / 33 |

mirror 平均耗时约为 legacy 的 `99.22%`，本轮均值低约 `0.8%`；但胜负仅
`17:16`，应理解为两者速度基本持平，不能据此声称稳定或显著加速。这不是纯
kernel profile，而是两条实际 validator 调用链的公平端到端计时。结合两者
最终 loss 降幅接近，当前没有为了速度切换算法的依据，因此保留实现更简单的
mirror `lr=1.0`。

PyTorch `3.49 s/组` 仍包含 snapshot 读取、两项 cross-check、20-step
optimizer 和结果写出，作用是离线参考，不参与生产 CUDA 后端的速度选择。

## 复现入口

```powershell
pwsh -NoProfile -File scripts\run_optimal_e_multiscene.ps1 -SmokeOnly
pwsh -NoProfile -File scripts\run_optimal_e_multiscene.ps1
python scripts\compare_optimal_e_optimizers.py --device cuda
```

新实验默认统一写入/读取 `optimal-e-multiscene-v4`。若要复算本文保留的历史
v3 数据，显式增加
`--experiment-root build/experiments/optimal-e-multiscene-v3`。

机器结果位于：

- `build/experiments/optimal-e-multiscene-v3/manifest.json`
- `build/experiments/optimal-e-multiscene-v3/results.json`
- `build/experiments/optimal-e-multiscene-v3/results.csv`
- `build/experiments/optimal-e-multiscene-v3/optimizer-comparison.json`
- `build/experiments/optimal-e-multiscene-v3/optimizer-comparison.csv`

比较器会用 snapshot、legacy executable 和 candidate q 的 SHA-256，以及
schema/steps/learning-rate provenance 判断缓存能否复用。生成数据脚本还会
绑定 renderer、validator、两份 OptiX-IR、Python reference、实际 scene
descriptor 和整个 `assets/` 树的 SHA-256，防止后续 resume 把不同代码、
shader 或资产生成的 summary 混在同一批次。

本次 114 组是在逐 summary provenance 门禁加入前开始生成的，因此没有事后
伪造 `provenance_id`；实验根目录的 `generation-provenance.json` 单独记录了
当时实际 executable/reference 哈希和时间边界。后续新实验使用 manifest
schema 3，并在每个 summary 中锁定 provenance、steps、CUDA learning rate
和实际 device。当前生成器与 validator 默认 CUDA mirror `lr=1.0`。
