#pragma once

#include <renderer/Scene.h>
#include <renderer/core/launch_params.h>

#include <memory>
#include <string>

struct Scene;

namespace spcbpt
{

struct SceneConfig
{
    std::string path;
};

struct RendererConfig
{
    unsigned int width  = 64;
    unsigned int height = 64;
};

class RendererRuntime
{
  public:
    RendererRuntime();
    ~RendererRuntime();

    RendererRuntime( const RendererRuntime& ) = delete;
    RendererRuntime& operator=( const RendererRuntime& ) = delete;

    void loadScene( const SceneConfig& config );
    void initialize( const RendererConfig& config );
    void renderFrame( uchar4* output, const std::string& raygen = "pt" );
    void uploadParams();
    void synchronize();
    void reset();

    bool isSceneLoaded() const { return m_scene != nullptr; }
    bool isInitialized() const { return m_device_params != nullptr; }

    MyParams& params() { return m_params; }
    const MyParams& params() const { return m_params; }
    MyParams* deviceParams() const { return m_device_params; }

    sutil::Scene& scene();
    const sutil::Scene& scene() const;

  private:
    void releaseLaunchBuffers();

    MyParams                       m_params        = {};
    MyParams*                      m_device_params = nullptr;
    std::unique_ptr<::Scene>       m_source_scene;
    std::unique_ptr<sutil::Scene>  m_scene;
};

} // namespace spcbpt
