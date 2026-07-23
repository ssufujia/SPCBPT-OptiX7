#include <renderer/Exception.h>
#include <renderer/RendererRuntime.h>
#include <renderer/RendererWorkflow.h>
#include <spcbptConfig.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

constexpr unsigned int SMOKE_WIDTH  = 64;
constexpr unsigned int SMOKE_HEIGHT = 64;

struct SmokeOptions
{
    std::string scene_path;
    bool        reload_check = false;
};

SmokeOptions parseOptions( int argc, char* argv[] )
{
    SmokeOptions options;
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--reload-check" )
        {
            options.reload_check = true;
        }
        else if( arg.rfind( "--scene=", 0 ) == 0 && arg.size() > 8 )
        {
            options.scene_path = arg.substr( 8 );
        }
        else
        {
            throw std::invalid_argument(
                "Usage: spcbpt_smoke [--scene=<path>] [--reload-check]"
            );
        }
    }
    return options;
}

class FrameBuffer
{
  public:
    explicit FrameBuffer( size_t pixel_count )
    {
        CUDA_CHECK( cudaMalloc( &m_pixels, pixel_count * sizeof( uchar4 ) ) );
        CUDA_CHECK( cudaMemset( m_pixels, 0, pixel_count * sizeof( uchar4 ) ) );
    }

    ~FrameBuffer()
    {
        if( m_pixels )
            CUDA_CHECK_NOTHROW( cudaFree( m_pixels ) );
    }

    FrameBuffer( const FrameBuffer& ) = delete;
    FrameBuffer& operator=( const FrameBuffer& ) = delete;

    uchar4* get() const { return static_cast<uchar4*>( m_pixels ); }

  private:
    void* m_pixels = nullptr;
};

} // namespace

int main( int argc, char* argv[] )
{
    try
    {
        const SmokeOptions options = parseOptions( argc, argv );
        spcbpt::SceneConfig scene_config = spcbpt::SceneConfig::defaultScene();
        if( !options.scene_path.empty() )
            scene_config.path = options.scene_path;

        spcbpt::RendererRuntime runtime;
        runtime.loadScene( scene_config );
        runtime.initialize( { SMOKE_WIDTH, SMOKE_HEIGHT } );

        FrameBuffer frame( SMOKE_WIDTH * SMOKE_HEIGHT );
        runtime.renderFrame( frame.get() );

        if( options.reload_check )
        {
            spcbpt::RendererWorkflow workflow( runtime );
            workflow.initializeAlgorithmState();
            workflow.runPreprocessing();
            workflow.renderFrame( frame.get(), "SPCBPT_eye" );

            runtime.reloadScene();
            workflow.initializeAlgorithmState();
            runtime.renderFrame( frame.get() );

            spcbpt::SceneConfig second_scene = {
                "projector/projector.scene",
                SPCBPT_ASSETS_DIR,
                spcbpt::SceneCameraOverride{
                    make_float3( 0.0f, 1.0f, 4.0f ),
                    make_float3( 0.0f, 1.0f, 0.0f ),
                    make_float3( 0.0f, 1.0f, 0.0f ),
                    40.0f
                }
            };
            runtime.reloadScene( second_scene );
            if( runtime.params().eye.x != 0.0f
                || runtime.params().eye.y != 1.0f
                || runtime.params().eye.z != 4.0f )
            {
                throw std::runtime_error( "Scene camera override was not applied" );
            }
            workflow.initializeAlgorithmState();
            workflow.runPreprocessing();
            workflow.renderFrame( frame.get(), "SPCBPT_eye" );

            runtime.unloadScene();
            if( runtime.isSceneLoaded() || runtime.isInitialized() )
                throw std::runtime_error( "RendererRuntime unload left live state" );
        }

        uchar4 first_pixel = {};
        CUDA_CHECK( cudaMemcpy(
            &first_pixel,
            frame.get(),
            sizeof( first_pixel ),
            cudaMemcpyDeviceToHost
        ) );
        std::cout << "SPCBPT headless smoke passed: "
                  << SMOKE_WIDTH << 'x' << SMOKE_HEIGHT
                  << ", first pixel rgba=("
                  << static_cast<int>( first_pixel.x ) << ','
                  << static_cast<int>( first_pixel.y ) << ','
                  << static_cast<int>( first_pixel.z ) << ','
                  << static_cast<int>( first_pixel.w ) << ")\n";
        return EXIT_SUCCESS;
    }
    catch( const std::exception& error )
    {
        std::cerr << "SPCBPT headless smoke failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
