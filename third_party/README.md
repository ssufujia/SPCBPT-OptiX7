# Vendored dependencies

System SDKs (OptiX, CUDA, OpenGL, MSVC/Windows SDK) are discovered externally
and are not vendored here. Each dependency below is pinned and build-tested.

| Dependency | Pinned version | Upgrade result |
|------------|----------------|----------------|
| GLFW | 3.4 | Latest stable; build and runtime passed |
| Dear ImGui | 1.92.8 | Latest stable tag; build and runtime passed |
| glad | 2.0.8 | Latest stable generator; OpenGL 3.3 compatibility loader, no extensions; build and runtime passed |
| stb_image | 2.30 | Latest official header; build and runtime passed |
| TinyGLTF | 2.9.7 | Latest stable v2 C++ API; v3 is still an experimental API rewrite |
| TinyEXR | 1.0.13 | Latest stable single-header C++ API; uses stb zlib backend |
| tinyobjloader | 0.9.16 | Compatibility hold: v1.0.6 changes mesh storage and `LoadObj`; requires a dedicated scene-loader migration |

Rules:

- Prefer the newest stable release.
- Pin a reviewed tag or commit; never track a floating branch in committed code.
- Upgrade and verify one dependency at a time.
- Record API incompatibilities instead of silently keeping an old version.
- Keep first-party warning and platform flags out of these targets.
