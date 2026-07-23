#include <renderer/Exception.h>
#include <renderer/RendererRuntime.h>
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

std::string parseScenePath( int argc, char* argv[] )
{
    std::string path = std::string( SPCBPT_ASSETS_DIR ) + "/bedroom.scene";
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg.rfind( "--scene=", 0 ) != 0 || arg.size() == 8 )
            throw std::invalid_argument( "Usage: spcbpt_smoke [--scene=<path>]" );
        path = arg.substr( 8 );
    }
    return path;
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
        const std::string scene_path = parseScenePath( argc, argv );
        spcbpt::RendererRuntime runtime;
        runtime.loadScene( { scene_path } );
        runtime.initialize( { SMOKE_WIDTH, SMOKE_HEIGHT } );

        FrameBuffer frame( SMOKE_WIDTH * SMOKE_HEIGHT );
        runtime.renderFrame( frame.get() );

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
