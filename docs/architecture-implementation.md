# 构建与源码架构重构实现说明

## 结果概览

本次重构把工程从 NVIDIA OptiX Samples 风格的混合目录，整理为一个以仓库根为唯一 CMake 入口、以 target 表达依赖关系的工程。核心依赖方向为：

```text
optixPathTracer ──> spcbpt_viewer ──> spcbpt_renderer
        │                                  │
        └──────────────────────────────────┘

spcbpt_smoke ───────────────────────> spcbpt_renderer
```

`spcbpt_renderer` 负责场景解析与迁移、OptiX context/module/pipeline/SBT、native CUDA/Thrust 和渲染算法；`spcbpt_viewer` 负责 GLFW、glad、ImGui、OpenGL 显示与交互；`optixPathTracer` 负责程序参数和窗口循环。

## CMake 实现

- 根 `CMakeLists.txt` 是唯一 source root，拒绝 in-source build。
- `CMakePresets.json` 保存共享配置；机器相关的 OptiX/Ninja 路径放在 gitignored `CMakeUserPresets.json`。
- `spcbpt_project_options` 统一第一方编译选项。
- `spcbpt_device_api` 统一 OptiX-IR 的 include、宏和 CUDA 配置。
- `raygen.cu` 与 `hit_program.cu` 使用 CMake 原生 `CUDA_OPTIX_COMPILATION` 生成 `.optixir`，并部署到 `bin/optix-ir/`。
- `include(CTest)` 注册 `spcbpt.headless_smoke` GPU 测试。

## 运行时资源与配置

资源统一位于仓库根 `assets/`。生成的 `spcbptConfig.h` 只提供：

- `SPCBPT_ASSETS_DIR`
- `SPCBPT_OPTIX_IR_DIR`

旧的 PTX/NVRTC/SAMPLES 路径不再参与运行时。`Scene.cpp` 直接读取构建部署的 OptiX-IR，因此应用和 smoke 均可从任意工作目录启动。

## Renderer / Viewer 边界

图像编解码实现集中在 `src/renderer/ImageCodecs.cpp`。此前 stb 实现位于 viewer 的 `sutil.cpp`，导致 renderer 静态库无法独立链接；无窗口 smoke 在链接阶段发现并修复了该隐藏依赖。

旧 `device_include/` 中基于 OptiX 6 `optixu` 的 `helpers.h`、`random.h` 和无调用者的 intersection refinement 已删除。有效共享结构迁到 `src/cuda/commonStructs.h`，算法头显式 include `cuda/random.h`，不再依赖 include 目录顺序。未编译的旧 `cuda/cuProg`、`whitted`、`sphere`、`curve` 实现也已清理。

## 无窗口 Smoke

`spcbpt_smoke` 不链接 `spcbpt_viewer`、GLFW、glad、ImGui 或 OpenGL。测试执行：

1. 读取 bedroom 场景；
2. 将场景与光源迁移到 renderer 数据结构；
3. 创建 OptiX context、GAS/IAS、module、program groups、pipeline 与 SBT；
4. 切换到基础 path-tracing raygen；
5. 完成一次 64×64 `optixLaunch`；
6. 同步并读取首像素后正常退出。

运行方式：

```powershell
ctest --test-dir build/release-optix9 -R spcbpt.headless_smoke --output-on-failure
```

## OptiX 9 Launcher

旧代码曾设置 `OPTIX_FORCE_DEPRECATED_LAUNCHER=1`。在独立 smoke 和 viewer bedroom 回归均通过后，该兼容开关已移除，当前使用 OptiX 9 默认 launcher。

## 验证范围

- MSVC Developer Environment 下全量编译和链接；
- renderer-only smoke 从仓库外工作目录执行；
- CTest smoke 通过；
- viewer 使用 OptiX 9 默认 launcher 进入连续帧；
- CMake/C++ IDE diagnostics 无新增错误。

第三方依赖版本与兼容性例外见 `third_party/README.md`。
