#pragma once

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

struct RendererConfig
{
    unsigned int width                   = 64;
    unsigned int height                  = 64;
    int          active_path_depth       = 12;
    int          connection_count        = 1;
    bool         spcbpt_pure              = true;
    bool         rmis_enabled             = true;
    bool         path_guiding_enabled     = false;
    bool         path_guiding_self_train  = true;
    bool         path_guiding_more_training = false;
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
    void markImageDirty();
    void renderFrame( uchar4* output, const std::string& raygen = "pt" );
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
