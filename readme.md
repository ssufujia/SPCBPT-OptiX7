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

This repository is an OptiX implementation of
[SPCBPT: Subspace-based Probabilistic Connections for Bidirectional Path Tracing](https://ssufujia.github.io/SPCBPT/).

## Requirements

Verified baseline: OptiX 9.1, CUDA 12.2, MSVC x64, Ninja, and CMake 3.27+
(clean-build verified with CMake 4.4.0). Vendored versions are listed in
[`third_party/README.md`](third_party/README.md).

## Build

The repository root is the only supported CMake source directory. In-source
builds are rejected.

1. Copy `CMakeUserPresets.json.example` to `CMakeUserPresets.json`.
2. Set local `OptiX_ROOT` and Ninja paths.
3. Open an x64 Visual Studio Developer shell.
4. Run:

```powershell
cmake --fresh --preset release-optix9-local
cmake --build --preset release-optix9-local
```

The executable is `build/release-optix9/bin/optixPathTracer.exe`; native
OptiX-IR files are deployed beside it under `bin/optix-ir/`.

The renderer can start from any working directory. Use `--scene=<path>` to
override the default bedroom scene and `--dim=<width>x<height>` to override the
image dimensions.

## Build architecture

- `spcbpt_renderer`: OptiX scene/pipeline, algorithms and native CUDA;
  no GLFW, glad, ImGui or OpenGL dependency.
- `spcbpt_viewer`: window, input, display and UI.
- `optixPathTracer`: application and CLI composition.
- `spcbpt_optix_ir`: CMake-native compilation of the two OptiX shaders.

Scenes live in `assets/`, dependencies in `third_party/`, and controls in
[`docs/operation.md`](docs/operation.md).

## Difference from the paper-version code

Due to various reasons, some details of this implementation are slightly
different from the paper-version code.

- This implementation disables the t = 1 strategy, i.e., the strategy of
  light sub-path connecting to the eye sub-path directly, because it is
  usually of low efficiency.
- The parts of cross-iteration reuse of light sub-path, environment map, and
  transparent material are not yet completed.
- Direction is ignored in the classification. Position and normal are more
  important in most cases.
- Subspace Sampling Matrix is trained from an initial matrix built from the
  full contribution integral of the paths in the corresponding subspace pair
  to speed up the training.
- Paths for training are traced by a simple unidirectional path tracer with
  NEE implementation.
- The over-bright fireflies are slightly more than in the paper-version code;
  this remains future work.
