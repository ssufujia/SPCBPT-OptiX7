#include <renderer/RendererRuntime.h>

#include <renderer/Camera.h>
#include <renderer/Exception.h>
#include <renderer/core/sceneLoader.h>
#include <renderer/core/scene_shift.h>
#include <spcbptConfig.h>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace spcbpt
{

RendererRuntime::RendererRuntime() = default;

RendererRuntime::~RendererRuntime()
{
    reset();
}

SceneConfig SceneConfig::defaultScene()
{
    return { std::string( SPCBPT_ASSETS_DIR ) + "/bedroom.scene" };
}

void RendererRuntime::loadScene( const SceneConfig& config )
{
    if( config.path.empty() )
        throw std::invalid_argument( "Scene path is empty" );
    if( !std::filesystem::is_regular_file( config.path ) )
        throw std::runtime_error( "Scene file is missing or unreadable: " + config.path );

    reset();
    m_source_scene.reset( LoadScene( config.path.c_str() ) );
    if( !m_source_scene )
        throw std::runtime_error( "Scene loader returned no scene" );

    m_scene = std::make_unique<sutil::Scene>();
    Scene_shift( *m_source_scene, *m_scene );
    LightSource_shift( *m_source_scene, m_params, *m_scene );
}

void RendererRuntime::initialize( const RendererConfig& config )
{
    if( !m_scene )
        throw std::logic_error( "RendererRuntime::loadScene must be called first" );
    if( config.width == 0 || config.height == 0 )
        throw std::invalid_argument( "Renderer dimensions must be non-zero" );
    if( config.active_path_depth <= 0
        || config.active_path_depth > SPCBPT_DEVICE_MAX_PATH_DEPTH )
    {
        throw std::invalid_argument(
            "Active path depth must be in [1, "
            + std::to_string( SPCBPT_DEVICE_MAX_PATH_DEPTH ) + "]"
        );
    }
    if( config.connection_count <= 0 )
        throw std::invalid_argument( "Connection count must be positive" );
    if( m_scene_finalized && config.spcbpt_pure != m_config.spcbpt_pure )
    {
        throw std::logic_error(
            "SPCBPT algorithm mode cannot change after scene finalization"
        );
    }

    releaseLaunchBuffers();

    if( !m_scene_finalized )
    {
        m_scene->setSpcbptPure( config.spcbpt_pure );
        m_scene->finalize();
        m_scene_finalized = true;
    }

    m_config                   = config;
    m_params.width             = config.width;
    m_params.height            = config.height;
    m_params.active_path_depth = config.active_path_depth;
    m_params.connection_count  = config.connection_count;
    m_params.subframe_index    = 0;
    m_params.frame_buffer      = nullptr;
    m_params.miss_color        = make_float3( 0.1f );
    m_params.handle            = m_scene->traversableHandle();
    m_params.spcbpt_pure       = config.spcbpt_pure;
    m_params.rmis_enabled      = config.rmis_enabled;

    std::vector<MaterialData::Pbr> materials;
    materials.reserve( m_scene->materials().size() );
    for( const MaterialData& material : m_scene->materials() )
        materials.push_back( material.pbr );
    if( materials.empty() )
        throw std::runtime_error( "Scene contains no materials" );

    m_params.materials = HostToDeviceBuffer( materials.data(), static_cast<int>( materials.size() ) );

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &m_params.accum_buffer ),
        static_cast<size_t>( config.width ) * config.height * sizeof( float4 )
    ) );
    CUDA_CHECK( cudaMemset(
        m_params.accum_buffer,
        0,
        static_cast<size_t>( config.width ) * config.height * sizeof( float4 )
    ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &m_device_params ),
        sizeof( MyParams )
    ) );

    m_params.subspace_info.eye_tree = nullptr;
    m_params.subspace_info.light_tree = nullptr;
    m_params.subspace_info.Q = nullptr;
    m_params.subspace_info.CMFGamma = nullptr;
    m_params.estimate_pr.ready = false;
    m_params.estimate_pr.ref_buffer = nullptr;

    sutil::Camera camera = m_scene->camera();
    camera.setAspectRatio( static_cast<float>( config.width ) / config.height );
    m_params.eye = camera.eye();
    camera.UVWFrame( m_params.U, m_params.V, m_params.W );
}

void RendererRuntime::resize( unsigned int width, unsigned int height )
{
    if( !isInitialized() )
        throw std::logic_error( "RendererRuntime::initialize must be called first" );
    if( width == 0 || height == 0 )
        throw std::invalid_argument( "Renderer dimensions must be non-zero" );

    CUDA_CHECK( cudaFree( m_params.accum_buffer ) );
    m_params.accum_buffer = nullptr;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &m_params.accum_buffer ),
        static_cast<size_t>( width ) * height * sizeof( float4 )
    ) );
    m_config.width  = width;
    m_config.height = height;
    m_params.width  = width;
    m_params.height = height;
    m_params.dot_params.pixel_dirty = true;
}

void RendererRuntime::markImageDirty()
{
    m_params.dot_params.pixel_dirty = true;
}

void RendererRuntime::renderFrame( uchar4* output, const std::string& raygen )
{
    if( !isInitialized() )
        throw std::logic_error( "RendererRuntime::initialize must be called first" );
    if( !output )
        throw std::invalid_argument( "Frame output pointer is null" );

    m_scene->switchRaygen( raygen );
    m_params.frame_buffer = output;
    uploadParams();

    OPTIX_CHECK( optixLaunch(
        m_scene->pipeline(),
        nullptr,
        reinterpret_cast<CUdeviceptr>( m_device_params ),
        sizeof( MyParams ),
        m_scene->sbt(),
        m_params.width,
        m_params.height,
        1
    ) );
    synchronize();
}

void RendererRuntime::uploadParams()
{
    if( !m_device_params )
        throw std::logic_error( "RendererRuntime device params are not initialized" );

    CUDA_CHECK( cudaMemcpyAsync(
        m_device_params,
        &m_params,
        sizeof( MyParams ),
        cudaMemcpyHostToDevice,
        nullptr
    ) );
}

void RendererRuntime::synchronize()
{
    CUDA_SYNC_CHECK();
}

sutil::Scene& RendererRuntime::scene()
{
    if( !m_scene )
        throw std::logic_error( "Renderer scene is not loaded" );
    return *m_scene;
}

const sutil::Scene& RendererRuntime::scene() const
{
    if( !m_scene )
        throw std::logic_error( "Renderer scene is not loaded" );
    return *m_scene;
}

void RendererRuntime::releaseLaunchBuffers()
{
    if( m_params.accum_buffer )
    {
        CUDA_CHECK_NOTHROW( cudaFree( m_params.accum_buffer ) );
        m_params.accum_buffer = nullptr;
    }
    if( m_params.materials.data )
    {
        CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( m_params.materials.data ) ) );
        m_params.materials = {};
    }
    if( m_device_params )
    {
        CUDA_CHECK_NOTHROW( cudaFree( m_device_params ) );
        m_device_params = nullptr;
    }
}

void RendererRuntime::reset()
{
    releaseLaunchBuffers();
    if( m_params.lights.data )
    {
        CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( m_params.lights.data ) ) );
        m_params.lights = {};
    }
    m_scene.reset();
    m_source_scene.reset();
    m_params          = {};
    m_config          = {};
    m_scene_finalized = false;
}

} // namespace spcbpt
