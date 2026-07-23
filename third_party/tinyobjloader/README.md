# tinyobjloader

- Upstream: https://github.com/tinyobjloader/tinyobjloader
- Vendored file: `tiny_obj_loader.h`
- Version: 0.9.16 (compatibility hold)
- License: MIT
- Local changes: implementation remains in the project wrapper source
- Upgrade status: v1.0.6 was build-tested and rejected for this phase because
  it moves positions, normals and texture coordinates out of `mesh_t` and
  changes the `LoadObj` API. Migrating it requires a dedicated scene-loader
  rewrite and render-result regression.
