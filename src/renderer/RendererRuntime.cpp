#include <renderer/RendererRuntime.h>

#include <renderer/Camera.h>
#include <renderer/Exception.h>
#include <renderer/core/cuda_thrust/device_thrust.h>
#include <renderer/core/sceneLoader.h>
#include <renderer/core/scene_shift.h>
#include <spcbptConfig.h>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace spcbpt
{

namespace
{

std::filesystem::path normalizedAbsolutePath( const std::filesystem::path& path )
{
    return std::filesystem::absolute( path ).lexically_normal();
}

SceneConfig resolveSceneConfig( const SceneConfig& config )
{
    if( config.path.empty() )
        throw std::invalid_argument( "Scene path is empty" );

    SceneConfig resolved = config;
    const std::filesystem::path resource_root = normalizedAbsolutePath(
        config.resource_root.empty()
            ? std::filesystem::path( SPCBPT_ASSETS_DIR )
            : std::filesystem::path( config.resource_root )
    );
    std::filesystem::path scene_path = config.path;
    if( scene_path.is_relative() )
    {
        const std::filesystem::path cwd_path = normalizedAbsolutePath( scene_path );
        const std::filesystem::path root_path =
            ( resource_root / scene_path ).lexically_normal();
        scene_path = std::filesystem::is_regular_file( cwd_path ) ? cwd_path : root_path;
    }
    else
    {
        scene_path = scene_path.lexically_normal();
    }

    resolved.path = scene_path.string();
    resolved.resource_root = resource_root.string();
    return resolved;
}

bool isGltfScene( const std::filesystem::path& path )
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        []( unsigned char character )
        {
            return static_cast<char>( std::tolower( character ) );
        }
    );
    return extension == ".glb" || extension == ".gltf";
}

template <typename T>
void freeDevicePointer( T*& pointer )
{
    if( pointer )
    {
        CUDA_CHECK_NOTHROW( cudaFree( pointer ) );
        pointer = nullptr;
    }
}

void releaseAlgorithmBuffers( MyParams& params )
{
    freeDevicePointer( params.lt.ans );
    freeDevicePointer( params.lt.validState );
    freeDevicePointer( params.lt.lightImage );
    freeDevicePointer( params.lt.lightIndex );
    freeDevicePointer( params.lt.lightBuffer );
    freeDevicePointer( params.lt.rand_state );
    freeDevicePointer( params.pre_tracer.paths );
    freeDevicePointer( params.pre_tracer.conns );
    params.lt = {};
    params.pre_tracer = {};
    params.sampler = {};
    params.subspace_info = {};
    params.pg_params = {};
    params.dot_params = {};
    params.sky = {};
}

const char* mainRaygenName( RendererAlgorithm algorithm )
{
    switch( algorithm )
    {
        case RendererAlgorithm::PathTracing:
            return "pt";
        case RendererAlgorithm::Lvcbpt:
            return "SPCBPT_eye_ForcePure";
        case RendererAlgorithm::LvcbptProxyExperimental:
            return "SPCBPT_eye";
    }
    throw std::logic_error( "Unknown renderer algorithm" );
}

} // namespace

RendererRuntime::RendererRuntime() = default;

RendererRuntime::~RendererRuntime()
{
    reset();
}

SceneConfig SceneConfig::defaultScene()
{
    return { "bedroom.scene", SPCBPT_ASSETS_DIR, std::nullopt };
}

void RendererRuntime::loadScene( const SceneConfig& config )
{
    const SceneConfig resolved_config = resolveSceneConfig( config );
    if( !std::filesystem::is_regular_file( resolved_config.path ) )
    {
        throw std::runtime_error(
            "Scene file is missing or unreadable: " + resolved_config.path
        );
    }

    unloadScene();
    try
    {
        m_scene = std::make_unique<sutil::Scene>();
        if( isGltfScene( resolved_config.path ) )
        {
            sutil::loadScene( resolved_config.path, *m_scene );
            m_scene->setResourceRoot( resolved_config.resource_root );
            if( resolved_config.camera_override )
            {
                const SceneCameraOverride& source = *resolved_config.camera_override;
                sutil::Camera camera;
                camera.setEye( source.eye );
                camera.setLookat( source.lookat );
                camera.setUp( source.up );
                camera.setFovY( source.fov_y );
                m_scene->setCamera( camera );
            }
        }
        else
        {
            m_source_scene.reset( LoadScene(
                resolved_config.path.c_str(),
                resolved_config.resource_root.c_str()
            ) );
            if( !m_source_scene )
                throw std::runtime_error( "Scene loader returned no scene" );

            if( resolved_config.camera_override )
            {
                const SceneCameraOverride& camera = *resolved_config.camera_override;
                m_source_scene->eye = camera.eye;
                m_source_scene->lookat = camera.lookat;
                m_source_scene->up = camera.up;
                m_source_scene->fov = camera.fov_y;
                m_source_scene->use_camera = true;
            }

            Scene_shift( *m_source_scene, *m_scene );
            LightSource_shift( *m_source_scene, m_params, *m_scene );
        }

        m_scene_config = resolved_config;
    }
    catch( ... )
    {
        unloadScene();
        throw;
    }
}

void RendererRuntime::reloadScene()
{
    if( m_scene_config.path.empty() )
        throw std::logic_error( "RendererRuntime has no scene to reload" );
    reloadScene( m_scene_config );
}

void RendererRuntime::reloadScene( const SceneConfig& config )
{
    const bool was_initialized = isInitialized();
    const RendererConfig renderer_config = m_config;
    loadScene( config );
    if( was_initialized )
        initialize( renderer_config );
}

void RendererRuntime::initialize( const RendererConfig& config )
{
    if( !m_scene )
        throw std::logic_error( "RendererRuntime::loadScene must be called first" );
    validateRendererConfig( config );

    const bool spcbpt_pure = !usesProxyRendererAlgorithm( config.algorithm );
    if( m_scene_finalized
        && spcbpt_pure
            != !usesProxyRendererAlgorithm( m_config.algorithm ) )
    {
        throw std::logic_error(
            "Proxy renderer mode cannot change after scene finalization"
        );
    }

    releaseLaunchBuffers();

    if( !m_scene_finalized )
    {
        m_scene->setSpcbptPure( spcbpt_pure );
        m_scene->finalize();
        m_scene_finalized = true;
    }

    m_config                   = config;
    m_params.width             = config.width;
    m_params.height            = config.height;
    m_params.active_path_depth = config.active_path_depth;
    m_params.connection_count  = config.connection_count;
    m_params.subframe_index    = 0;
    m_params.experiment_seed   = 0;
    m_params.frame_buffer      = nullptr;
    m_params.miss_color        = make_float3( 0.1f );
    m_params.handle            = m_scene->traversableHandle();
    m_params.spcbpt_pure       = spcbpt_pure;
    m_params.rmis_enabled      = 1;
    m_params.caustic_path_only = config.caustic_path_only;

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

void RendererRuntime::resetAccumulation()
{
    if( !isInitialized() )
        throw std::logic_error( "RendererRuntime::initialize must be called first" );
    CUDA_CHECK( cudaMemset(
        m_params.accum_buffer,
        0,
        static_cast<size_t>( m_params.width )
            * m_params.height * sizeof( float4 )
    ) );
    m_params.subframe_index = 0;
    markImageDirty();
}

void RendererRuntime::resize( unsigned int width, unsigned int height )
{
    if( !isInitialized() )
        throw std::logic_error( "RendererRuntime::initialize must be called first" );

    RendererConfig resized_config = m_config;
    resized_config.width = width;
    resized_config.height = height;
    validateRendererConfig( resized_config );
    if( width == m_config.width && height == m_config.height )
        return;

    float4* resized_accumulation = nullptr;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &resized_accumulation ),
        static_cast<size_t>( width ) * height * sizeof( float4 )
    ) );
    try
    {
        CUDA_CHECK( cudaMemset(
            resized_accumulation,
            0,
            static_cast<size_t>( width ) * height * sizeof( float4 )
        ) );
    }
    catch( ... )
    {
        CUDA_CHECK_NOTHROW( cudaFree( resized_accumulation ) );
        throw;
    }

    CUDA_CHECK_NOTHROW( cudaFree( m_params.accum_buffer ) );
    m_params.accum_buffer = resized_accumulation;
    m_params.width = width;
    m_params.height = height;
    m_params.subframe_index = 0;
    m_config = resized_config;
    markImageDirty();
}

void RendererRuntime::markImageDirty()
{
    m_params.dot_params.pixel_dirty = true;
}

void RendererRuntime::renderFrame( uchar4* output )
{
    if( !isInitialized() )
        throw std::logic_error( "RendererRuntime::initialize must be called first" );
    if( !output )
        throw std::invalid_argument( "Frame output pointer is null" );

    m_scene->switchRaygen( mainRaygenName( m_config.algorithm ) );
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
    unloadScene();
}

void RendererRuntime::unloadScene()
{
    const bool had_scene_state =
        m_scene != nullptr
        || m_source_scene != nullptr
        || m_device_params != nullptr
        || m_params.lt.ans != nullptr
        || m_params.pre_tracer.paths != nullptr;
    releaseAlgorithmBuffers( m_params );
    if( had_scene_state )
        MyThrustOp::invalidate_scene_caches();
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
    ++m_scene_generation;
}

} // namespace spcbpt
