#pragma once

#include <renderer/RendererConfig.h>
#include <renderer/Scene.h>
#include <renderer/core/launch_params.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

struct Scene;

namespace spcbpt
{

struct SceneCameraOverride
{
    float3 eye;
    float3 lookat;
    float3 up;
    float  fov_y;
};

struct SceneConfig
{
    std::string path;
    std::string resource_root;
    std::optional<SceneCameraOverride> camera_override;

    static SceneConfig defaultScene();
};

class RendererRuntime
{
  public:
    RendererRuntime();
    ~RendererRuntime();

    RendererRuntime( const RendererRuntime& ) = delete;
    RendererRuntime& operator=( const RendererRuntime& ) = delete;

    void loadScene( const SceneConfig& config );
    void reloadScene();
    void reloadScene( const SceneConfig& config );
    void unloadScene();
    void initialize( const RendererConfig& config );
    void resize( unsigned int width, unsigned int height );
    void resetAccumulation();
    void markImageDirty();
    void uploadParams();
    void synchronize();
    void reset();

    bool isSceneLoaded() const { return m_scene != nullptr; }
    bool isInitialized() const { return m_device_params != nullptr; }
    std::uint64_t sceneGeneration() const { return m_scene_generation; }

    MyParams& params() { return m_params; }
    const MyParams& params() const { return m_params; }
    MyParams* deviceParams() const { return m_device_params; }
    const RendererConfig& config() const { return m_config; }
    const SceneConfig& sceneConfig() const { return m_scene_config; }

    sutil::Scene& scene();
    const sutil::Scene& scene() const;

  private:
    friend void renderConfiguredFrame( RendererRuntime&, uchar4* );

    void renderFrame( uchar4* output );
    void releaseLaunchBuffers();

    MyParams                       m_params        = {};
    MyParams*                      m_device_params = nullptr;
    RendererConfig                 m_config        = {};
    bool                           m_scene_finalized = false;
    std::unique_ptr<::Scene>       m_source_scene;
    std::unique_ptr<sutil::Scene>  m_scene;
    SceneConfig                    m_scene_config = {};
    std::uint64_t                  m_scene_generation = 0;
};

} // namespace spcbpt
