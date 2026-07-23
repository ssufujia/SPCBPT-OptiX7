[SPCBPT: Subspace-based Probabilistic Connections for Bidirectional Path Tracing](https://ssufujia.github.io/SPCBPT/) 的 OptiX 实现。

### 环境

当前验证基线为 OptiX 9.1、CUDA 12.2、MSVC x64、Ninja 和 CMake
3.27+；已用 CMake 4.4.0 完成 clean build。vendored 依赖版本见
[`third_party/README.md`](third_party/README.md)。

### 构建方法

仓库根目录是唯一受支持的 CMake source root；禁止 in-source build。

1. 把 `CMakeUserPresets.json.example` 复制为 `CMakeUserPresets.json`。
2. 填写本机 `OptiX_ROOT` 和 Ninja 路径。
3. 打开 x64 Visual Studio Developer Shell。
4. 执行：

```powershell
cmake --fresh --preset release-optix9-local
cmake --build --preset release-optix9-local
```

程序位于 `build/release-optix9/bin/optixPathTracer.exe`，原生 OptiX-IR
位于同目录下的 `optix-ir/`。程序支持从任意工作目录启动；
`--scene=<path>` 可覆盖默认 bedroom 场景。

源码按 `app → viewer + renderer` 分层；场景资源位于 `assets/`，第三方源码
位于 `third_party/`，交互说明见 [`docs/operation.md`](docs/operation.md)。

### 与论文版本的代码的差异

由于种种原因，目前该实现的实现细节并非完全与论文中的一致，在某些地方有细微的差别，包括：

* t=1，即光子路径直接连接摄像头的策略并非被实现，因为一般而言这一策略都过于低效。
* 光子路的跨帧重用、环境光照和透明材质目前仍未完成，将在未来补全。
* 子空间分类时并非考虑入射方向因素，为了提供更好的分类表现，在目前的场景中我们需要把分类决策树的精度都给到位置和法线上。
* 子空间采样矩阵并非从均匀矩阵中训练而成，而是从一个初始的根据子空间之间的路径的贡献值积分来构建的矩阵上再进一步迭代训练，这能够加快训练的速度.
* 简单起见，追踪训练集使用的算法为单向路径追踪+NEE，而非论文中的BDPT。
* 目前的算法表现在某些地方的亮噪点要比我的论文版本的代码稍微多一点点，我会在未来找到这一问题的解答并改正。