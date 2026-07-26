# 运行与配置

默认 `RendererConfig` 为 `LVCBPT + Proxy (Experimental)`，同时启用 path
guiding 和 self-training。可复制 `renderer_config.json.example` 为
`renderer_config.json`，再以 `--config=renderer_config.json` 启动。ImGui
中的 draft 只有点击 **Apply** 后才重建 renderer 并生效；未指定配置路径时，
界面保存到 ignored `renderer_config.json`。

正常算法选项为 `PT`、`LVCBPT` 和 `LVCBPT + Proxy (Experimental)`。
窗口保持宽高比缩放时立即轻量 resize；拖动导致宽高比变化时，显示层暂时缩放
旧结果，并把新尺寸标成 pending，点击 **Apply** 后只做一次同步重建/训练。

- **C**：打印当前相机参数。
- **P**：保存/检查当前帧结果。
- **W**：相机前移。
