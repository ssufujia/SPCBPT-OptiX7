#include <optix_stubs.h>
#include <spcbptConfig.h>
#include <renderer/Exception.h>

#include <renderer/core/launch_params.h>
#include <renderer/Scene.h>
#include <renderer/core/sceneLoader.h>
#include <renderer/core/scene_shift.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr unsigned int SMOKE_WIDTH  = 64;
constexpr unsigned int SMOKE_HEIGHT = 64;

class CudaAllocation
{
  public:
    CudaAllocation() = default;
    CudaAllocation( const CudaAllocation& ) = delete;
    CudaAllocation& operator=( const CudaAllocation& ) = delete;

    ~CudaAllocation()
    {
        if( m_ptr )
            cudaFree( m_ptr );
    }

    void allocate( size_t bytes )
    {
        CUDA_CHECK( cudaMalloc( &m_ptr, bytes ) );
    }

    void adopt( CUdeviceptr ptr )
    {
        m_ptr = reinterpret_cast<void*>( ptr );
    }

    void* get() const { return m_ptr; }

  private:
    void* m_ptr = nullptr;
};

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

} // namespace

int main( int argc, char* argv[] )
{
    try
    {
        const std::string scene_path = parseScenePath( argc, argv );
        if( !std::filesystem::is_regular_file( scene_path ) )
            throw std::runtime_error( "Scene file is missing or unreadable: " + scene_path );

        std::unique_ptr<Scene> source_scene( LoadScene( scene_path.c_str() ) );
        if( !source_scene )
            throw std::runtime_error( "Scene loader returned no scene" );

        MyParams params = {};
        params.width     = SMOKE_WIDTH;
        params.height    = SMOKE_HEIGHT;
        params.max_depth = MAX_PATH_LENGTH_FOR_MIS;
        params.miss_color = make_float3( 0.1f );
        params.spcbpt_pure = true;

        sutil::Scene scene;
        Scene_shift( *source_scene, scene );
        LightSource_shift( *source_scene, params, scene );
        CudaAllocation lights;
        lights.adopt( params.lights.data );

        scene.finalize();
        scene.switchRaygen( "pt" );

        const sutil::Camera camera = scene.camera();
        sutil::Camera smoke_camera = camera;
        smoke_camera.setAspectRatio(
            static_cast<float>( SMOKE_WIDTH ) / static_cast<float>( SMOKE_HEIGHT )
        );
        params.eye = smoke_camera.eye();
        smoke_camera.UVWFrame( params.U, params.V, params.W );
        params.handle = scene.traversableHandle();

        std::vector<MaterialData::Pbr> host_materials;
        host_materials.reserve( scene.materials().size() );
        for( const MaterialData& material : scene.materials() )
            host_materials.push_back( material.pbr );
        if( host_materials.empty() )
            throw std::runtime_error( "Scene contains no materials" );

        CudaAllocation materials;
        materials.allocate( host_materials.size() * sizeof( MaterialData::Pbr ) );
        CUDA_CHECK( cudaMemcpy(
            materials.get(),
            host_materials.data(),
            host_materials.size() * sizeof( MaterialData::Pbr ),
            cudaMemcpyHostToDevice
        ) );
        params.materials.data = reinterpret_cast<CUdeviceptr>( materials.get() );
        params.materials.count = static_cast<uint32_t>( host_materials.size() );
        params.materials.byte_stride = sizeof( MaterialData::Pbr );
        params.materials.elmt_byte_size = sizeof( MaterialData::Pbr );

        const size_t pixel_count = SMOKE_WIDTH * SMOKE_HEIGHT;
        CudaAllocation accumulation;
        CudaAllocation frame;
        CudaAllocation device_params;
        accumulation.allocate( pixel_count * sizeof( float4 ) );
        frame.allocate( pixel_count * sizeof( uchar4 ) );
        device_params.allocate( sizeof( MyParams ) );
        CUDA_CHECK( cudaMemset( accumulation.get(), 0, pixel_count * sizeof( float4 ) ) );
        CUDA_CHECK( cudaMemset( frame.get(), 0, pixel_count * sizeof( uchar4 ) ) );

        params.accum_buffer = static_cast<float4*>( accumulation.get() );
        params.frame_buffer = static_cast<uchar4*>( frame.get() );
        CUDA_CHECK( cudaMemcpy(
            device_params.get(),
            &params,
            sizeof( params ),
            cudaMemcpyHostToDevice
        ) );

        OPTIX_CHECK( optixLaunch(
            scene.pipeline(),
            nullptr,
            reinterpret_cast<CUdeviceptr>( device_params.get() ),
            sizeof( params ),
            scene.sbt(),
            SMOKE_WIDTH,
            SMOKE_HEIGHT,
            1
        ) );
        CUDA_SYNC_CHECK();

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
