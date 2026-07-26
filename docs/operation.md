# 运行与配置

默认 `RendererConfig` 为 `LVCBPT + Proxy (Experimental)`，同时启用 path
guiding 和 self-training。可复制 `renderer_config.json.example` 为
`renderer_config.json`；程序启动时会自动读取它，也可用 `--config=<path>`
显式指定其他配置。ImGui 中的 draft 只有点击 **Apply** 后才重建 renderer
并生效；配置中的 `scene` 选择场景，`--scene=<path>` 可临时覆盖。

普通启动无需参数：

```powershell
.\build\release-optix9\bin\optixPathTracer.exe
```

配置管理场景、分辨率、路径深度、连接数、算法、是否仅保留焦散路径、
path guiding 开关/自训练，以及 Optimal-E 的初始学习率和迭代次数。程序会
在训练前把这些关键值打印到终端。`--no-gl-interop` 属于显示后端开关，不是
渲染算法设置，因此仍只保留为可选 CLI 参数。

正常算法选项为 `PT`、`LVCBPT` 和 `LVCBPT + Proxy (Experimental)`。
窗口保持宽高比缩放时立即轻量 resize；拖动导致宽高比变化时，显示层暂时缩放
旧结果，并把新尺寸标成 pending，点击 **Apply** 后只做一次同步重建/训练。

- **C**：打印当前相机参数。
- **P**：保存/检查当前帧结果。
- **W**：相机前移。
