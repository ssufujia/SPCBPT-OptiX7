#pragma once

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
    void renderFrame( uchar4* output, const std::string& raygen );

  private:
    void synchronizeSceneGeneration();

    class Impl;
    RendererRuntime& m_runtime;
    std::unique_ptr<Impl> m_impl;
};

} // namespace spcbpt
