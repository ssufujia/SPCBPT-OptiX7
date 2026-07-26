# SPCBPT-OptiX7

基于 OptiX 9 / CUDA 的验证性双向路径追踪渲染器。当前默认配置为
`LVCBPT + Path Guiding + Proxy (Experimental)`，Optimal-E 使用
CUDA mirror descent，初始学习率为 `1.0`。

- [构建与源码架构](docs/architecture-implementation.md)
- [运行与配置](docs/operation.md)
- [114 组真实数据优化器对照](docs/experiments/optimizer-comparison.md)
- [Optimal-E 样本、loss、梯度与三类优化器](docs/experiments/optimal-e-optimizer-workflow.md)

实验脚本和回归测试随仓库提交；约 3.49 GiB 的原始 snapshot、候选矩阵和日志
仅保存在本地 ignored `build/experiments/`，不上传 GitHub。

本仓库是论文
[SPCBPT：基于子空间概率连接的双向路径追踪](https://ssufujia.github.io/SPCBPT/)
的 OptiX 实验实现。当前实现以验证渲染流程、数据结构和 Blender 接入为主，
不以完整复现论文全部算法为目标。

在 11 个场景真实 snapshot 上补做的 33 组同口径计时中，CUDA mirror
`lr=1.0` 平均耗时 `0.277850 s`，旧 CUDA Adam `lr=0.05` 为
`0.280028 s`。两者速度和 loss 降幅都接近，没有为了速度切换算法的依据，
因此生产端保留实现更简单的 mirror。

## 环境要求

已验证的基线环境为 OptiX 9.1、CUDA 12.2、MSVC x64、Ninja 和
CMake 3.27 以上版本；全新构建另使用 CMake 4.4.0 验证通过。仓库内第三方
依赖版本见 [`third_party/README.md`](third_party/README.md)。

## 构建

仓库根目录是唯一支持的 CMake 源码入口，项目会拒绝源码内构建。

1. 将 `CMakeUserPresets.json.example` 复制为 `CMakeUserPresets.json`。
2. 填写本机的 `OptiX_ROOT` 和 Ninja 路径。
3. 打开 x64 Visual Studio Developer PowerShell。
4. 执行：

```powershell
cmake --fresh --preset release-optix9-local
cmake --build --preset release-optix9-local
```

可执行文件位于 `build/release-optix9/bin/optixPathTracer.exe`；原生
OptiX-IR 文件会部署到同级的 `bin/optix-ir/`。

普通运行不需要传参数：先把 `renderer_config.json.example` 复制为
`renderer_config.json`，然后在仓库根目录直接启动：

```powershell
.\build\release-optix9\bin\optixPathTracer.exe
```

程序默认读取当前目录的 `renderer_config.json`，并在启动时打印实际配置
来源、场景、算法、尺寸、路径模式、path guiding 与 Optimal-E 训练参数。
`--config`、`--scene`、`--dim` 只用于临时覆盖；完整字段说明见
[`docs/operation.md`](docs/operation.md)。

## 构建目标

- `spcbpt_renderer`：OptiX 场景、pipeline、算法和原生 CUDA；不依赖
  GLFW、glad、ImGui 或 OpenGL。
- `spcbpt_viewer`：窗口、输入、显示与界面。
- `optixPathTracer`：应用入口和 CLI 组装。
- `spcbpt_optix_ir`：由 CMake 原生编译的两个 OptiX shader。

场景资源位于 `assets/`，第三方依赖位于 `third_party/`，运行与界面操作见
[`docs/operation.md`](docs/operation.md)。

## 与论文版本的差异

- 当前禁用 `t=1` 策略，即光源子路径直接连接相机的策略，因为它通常效率较低。
- 跨迭代复用光源子路径、环境贴图和透明材质尚未完整实现。
- 子空间分类暂不考虑方向；多数场景中位置和法线更重要。
- 子空间采样矩阵从对应子空间对路径完整贡献积分构造的初始矩阵开始训练，
  以加快收敛。
- 训练路径由带 NEE 的简单单向路径追踪器生成。
- 过亮 firefly 仍比论文版本略多，后续再处理。
