#pragma once

#include <renderer/RendererConfig.h>

#include <memory>
#include <string>

struct uchar4;

namespace spcbpt
{

class RendererRuntime;

class RendererWorkflow
{
  public:
    explicit RendererWorkflow( RendererRuntime& runtime );
    ~RendererWorkflow();

    RendererWorkflow( const RendererWorkflow& ) = delete;
    RendererWorkflow& operator=( const RendererWorkflow& ) = delete;

    void initializeAlgorithmState();
    void runPreprocessing();
    void captureOptimalEProblem( const std::string& output_path );
    RendererConfigChange applyConfig( const RendererConfig& config );
    void renderFrame( uchar4* output );

  private:
    void synchronizeSceneGeneration();

    class Impl;
    RendererRuntime& m_runtime;
    std::unique_ptr<Impl> m_impl;
};

} // namespace spcbpt
