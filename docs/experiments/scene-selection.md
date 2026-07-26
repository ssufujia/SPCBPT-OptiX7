# Optimal E 正式实验场景与镜头

## 最终范围

正式矩阵不再遍历整个 `assets/`，而是覆盖 11 个有代表性的场景家族。镜头数
按场景复杂度分配为 2--5 个，每个镜头固定使用 seeds `11 / 29 / 47`：

```text
11 scenes × 38 cameras × 3 seeds = 114 runs
```

每个镜头先通过 headless 画面有效性检查，再进入真实路径采集。优先使用 scene
内置或注释中保留的 authored camera；缺少备选时，只围绕基准 eye 做小幅、
确定性的 yaw/pitch，不随机平移相机。

| 场景 | 实际 descriptor | 镜头数 | 关键相机参数 |
|---|---|---:|---|
| Bedroom | `assets/bedroom.scene` | 3 | 3 个 authored 视角 |
| White room | `assets/white-room/white-room-obj.scene` | 3 | 基准 + yaw -6° / +12° |
| Showcase | `assets/showcase/showcase.scene` | 2 | authored，FOV 20° |
| Cornell box | `assets/cornell_box/cornell.scene` | 2 | 基准 + yaw 12° |
| Conference | `assets/conference/conference.scene` | 4 | `conference2` 相机 + 1 个 `conference3` authored + yaw ±12° |
| Glassroom | `assets/glassroom/glassroom_project_final.scene` | 5 | 4 authored + 1 yaw；剔除全黑 authored 候选 |
| Bathroom | `assets/bathroom_b/scene_v4.scene` | 4 | 2 authored + yaw ±12°，FOV 60° |
| Breakfast | `assets/breafast_2.0/breafast_final.scene` | 3 | Z-up，2 authored + yaw 12° |
| Projector | `assets/projector/projector1.scene` | 3 | 基准 + yaw ±12°，FOV 65° |
| Kitchen | `assets/kitchen/kitchen_final.scene` | 5 | 基准 + yaw ±12° + pitch ±8° |
| Hallway | `assets/hallway/hallway-teaser.scene` | 4 | 基准 + yaw ±12° + pitch 8°，FOV 65° |

完整 eye/lookat/up/fov、镜头来源和 114 组身份由
`scripts/run_optimal_e_multiscene.ps1` 生成到实验目录的 `manifest.json`。

## 三个场景例外

### Conference

最初选择的 `conference2.scene` 在完整路径采集时可重复触发 Windows heap
corruption `0xC0000374`，但同一相机传给 `conference.scene` 后能完成 capture
和两向 objective 验证。正式 manifest 保留 requested/actual 两个路径，数据
使用稳定的 `conference.scene`；descriptor 差异和 heap corruption 留待独立
诊断。

### Hallway

最初选择的 `hallway-teaser_final.scene` 可稳定复现 CUDA illegal memory
access；同家族 `hallway-teaser.scene` 通过全部镜头 smoke，因此正式数据使用
后者。`final` 独有的 `newLight`、`lens2.obj`、`patch.obj` 和环境图仍然保留，
等待单独定位设备错误，不把它们当成垃圾资源删除。

### Bathroom

依赖闭包检查发现 `scene_v3.scene`、`scene_v4.scene` 和
`scene_v4_normal_c.scene` 都把已有的 `lightrim_o.obj` 误写成
`lightrim_0.obj`。旧 loader 会静默略过缺失模型，导致 smoke 假通过。本轮已
修正三个 descriptor，并在正式采集前重新检查 `scene_v4` 的 4 个镜头。

## Seed 门禁

显式 `experiment_seed` 已进入 device 随机数派生，并写入 `.spcoe`、
summary 和 manifest。正式批次启动前，用 Bedroom 基准镜头验证：

- seeds 11/29/47 的 snapshot SHA-256 均不同；
- 三组 path node 数和统计摘要不同；
- seed 11 独立重跑的 snapshot SHA-256 与统计摘要完全一致。

因此三组 seed 既确实不同，又可在当前环境字节级复现。

## 已执行的安全清理

依赖审计确认以下 descriptor 没有被保留场景、源码、测试、脚本或 CMake
入口引用。本轮只删除小型、Git 可恢复的 scene descriptor，不删除 OBJ、MTL、
纹理或环境图：

| 删除项 | 原因 |
|---|---|
| `glassroom_project - 副本.scene` | 与 `glassroom_project.scene` SHA-256 完全相同 |
| `glassroom3.scene` | 与 `glassroom2.scene` SHA-256 完全相同 |
| `projector2.scene` | 与 `projector1.scene` SHA-256 完全相同 |
| `cornell_test.scene` | 未引用的历史 test descriptor |
| `glassroom_project_test.scene` | 未引用的历史 test descriptor |
| `hallway-teaser_su3wrong.scene` | 文件名明确标记 wrong，且无入口引用 |
| `kitchen_final_old.scene` | 未引用的历史 old descriptor |

这次清理不把“正式实验未选中”等同于“可以删除”。其余材质、焦散、法线和
灯光变体仍可能用于后续验证，全部保留。
