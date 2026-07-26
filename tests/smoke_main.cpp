#include <renderer/Exception.h>
#include <renderer/RendererRuntime.h>
#include <renderer/RendererWorkflow.h>
#include <spcbptConfig.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr unsigned int SMOKE_WIDTH  = 64;
constexpr unsigned int SMOKE_HEIGHT = 64;

struct SmokeOptions
{
    std::string scene_path;
    std::string optimal_e_export_path;
    std::optional<float3> eye;
    std::optional<float3> lookat;
    std::optional<float3> up;
    std::optional<float>  fov_y;
    unsigned int experiment_seed = 0;
    bool        validate_frame = false;
    bool        reload_check = false;
    bool        default_mode_init = false;
};

float3 parseFloat3( const std::string& value, const char* option_name )
{
    std::istringstream input( value );
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    char first_separator = '\0';
    char second_separator = '\0';
    if( !( input >> x >> first_separator >> y >> second_separator >> z )
        || first_separator != ','
        || second_separator != ','
        || !std::isfinite( x )
        || !std::isfinite( y )
        || !std::isfinite( z ) )
    {
        throw std::invalid_argument(
            std::string( option_name ) + " must be three finite comma-separated numbers"
        );
    }
    input >> std::ws;
    if( !input.eof() )
    {
        throw std::invalid_argument(
            std::string( option_name ) + " contains trailing data"
        );
    }
    return make_float3( x, y, z );
}

float parseFloat( const std::string& value, const char* option_name )
{
    std::istringstream input( value );
    float result = 0.0f;
    if( !( input >> result ) || !std::isfinite( result ) )
        throw std::invalid_argument( std::string( option_name ) + " must be finite" );
    input >> std::ws;
    if( !input.eof() )
        throw std::invalid_argument( std::string( option_name ) + " contains trailing data" );
    return result;
}

unsigned int parseUnsigned( const std::string& value, const char* option_name )
{
    if( value.empty() || value.front() == '-' )
        throw std::invalid_argument( std::string( option_name ) + " must be an unsigned integer" );
    std::size_t parsed = 0;
    unsigned long long result = 0;
    try
    {
        result = std::stoull( value, &parsed );
    }
    catch( const std::exception& )
    {
        throw std::invalid_argument( std::string( option_name ) + " must be an unsigned integer" );
    }
    if( parsed != value.size()
        || result > std::numeric_limits<unsigned int>::max() )
    {
        throw std::invalid_argument( std::string( option_name ) + " is out of range" );
    }
    return static_cast<unsigned int>( result );
}

void validateCameraOverride(
    const float3& eye,
    const float3& lookat,
    const float3& up,
    float fov_y
)
{
    const float x = lookat.x - eye.x;
    const float y = lookat.y - eye.y;
    const float z = lookat.z - eye.z;
    const float up_length_squared = up.x * up.x + up.y * up.y + up.z * up.z;
    constexpr float MIN_DIRECTION_LENGTH_SQUARED = 1e-12f;
    if( x * x + y * y + z * z <= MIN_DIRECTION_LENGTH_SQUARED )
    {
        throw std::invalid_argument(
            "--eye and --lookat must describe a non-zero view direction"
        );
    }
    if( up_length_squared <= MIN_DIRECTION_LENGTH_SQUARED )
        throw std::invalid_argument( "--up must be non-zero" );
    const float cross_x = y * up.z - z * up.y;
    const float cross_y = z * up.x - x * up.z;
    const float cross_z = x * up.y - y * up.x;
    if( cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
        <= MIN_DIRECTION_LENGTH_SQUARED )
    {
        throw std::invalid_argument(
            "Camera view direction must not be parallel to --up"
        );
    }
    if( fov_y <= 0.0f || fov_y >= 180.0f )
        throw std::invalid_argument( "--fov must be between 0 and 180 degrees" );
}

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
        else if( arg == "--default-mode-init" )
        {
            options.default_mode_init = true;
        }
        else if( arg == "--validate-frame" )
        {
            options.validate_frame = true;
        }
        else if( arg.rfind( "--scene=", 0 ) == 0 && arg.size() > 8 )
        {
            options.scene_path = arg.substr( 8 );
        }
        else if( arg.rfind( "--export-optimal-e=", 0 ) == 0
                 && arg.size() > 19 )
        {
            options.optimal_e_export_path = arg.substr( 19 );
        }
        else if( arg.rfind( "--eye=", 0 ) == 0 && arg.size() > 6 )
        {
            options.eye = parseFloat3( arg.substr( 6 ), "--eye" );
        }
        else if( arg.rfind( "--lookat=", 0 ) == 0 && arg.size() > 9 )
        {
            options.lookat = parseFloat3( arg.substr( 9 ), "--lookat" );
        }
        else if( arg.rfind( "--up=", 0 ) == 0 && arg.size() > 5 )
        {
            options.up = parseFloat3( arg.substr( 5 ), "--up" );
        }
        else if( arg.rfind( "--fov=", 0 ) == 0 && arg.size() > 6 )
        {
            options.fov_y = parseFloat( arg.substr( 6 ), "--fov" );
        }
        else if( arg.rfind( "--experiment-seed=", 0 ) == 0 && arg.size() > 18 )
        {
            options.experiment_seed =
                parseUnsigned( arg.substr( 18 ), "--experiment-seed" );
        }
        else
        {
            throw std::invalid_argument(
                "Usage: spcbpt_smoke [--scene=<path>] [--reload-check] "
                "[--default-mode-init] "
                "[--export-optimal-e=<path>] "
                "[--eye=<x,y,z> --lookat=<x,y,z> --up=<x,y,z> --fov=<degrees>] "
                "[--experiment-seed=<uint32>] [--validate-frame]"
            );
        }
    }
    if( options.reload_check && !options.optimal_e_export_path.empty() )
    {
        throw std::invalid_argument(
            "--reload-check and --export-optimal-e cannot be combined"
        );
    }
    if( options.default_mode_init
        && ( options.reload_check || !options.optimal_e_export_path.empty() ) )
    {
        throw std::invalid_argument(
            "--default-mode-init cannot be combined with reload or export"
        );
    }
    const int camera_option_count =
        static_cast<int>( options.eye.has_value() )
        + static_cast<int>( options.lookat.has_value() )
        + static_cast<int>( options.up.has_value() )
        + static_cast<int>( options.fov_y.has_value() );
    if( camera_option_count != 0 && camera_option_count != 4 )
    {
        throw std::invalid_argument(
            "--eye, --lookat, --up and --fov must be provided together"
        );
    }
    if( options.eye )
        validateCameraOverride(
            *options.eye, *options.lookat, *options.up, *options.fov_y
        );
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
        if( options.eye )
        {
            scene_config.camera_override = spcbpt::SceneCameraOverride{
                *options.eye,
                *options.lookat,
                *options.up,
                *options.fov_y
            };
        }

        spcbpt::RendererRuntime runtime;
        runtime.loadScene( scene_config );
        spcbpt::RendererConfig renderer_config;
        renderer_config.width = SMOKE_WIDTH;
        renderer_config.height = SMOKE_HEIGHT;
        if( !options.default_mode_init )
        {
            renderer_config.algorithm =
                options.reload_check || !options.optimal_e_export_path.empty()
                ? spcbpt::RendererAlgorithm::Lvcbpt
                : spcbpt::RendererAlgorithm::PathTracing;
            renderer_config.path_guiding_enabled = false;
        }
        runtime.initialize( renderer_config );
        runtime.params().experiment_seed = options.experiment_seed;

        spcbpt::RendererWorkflow workflow( runtime );
        if( options.default_mode_init )
        {
            if( renderer_config.algorithm
                    != spcbpt::RendererAlgorithm::LvcbptProxyExperimental
                || !renderer_config.path_guiding_enabled
                || !renderer_config.path_guiding_self_train
                || runtime.params().spcbpt_pure )
            {
                throw std::runtime_error(
                    "Production default mode was not initialized"
                );
            }
            workflow.initializeAlgorithmState();
            try
            {
                workflow.renderFrame( nullptr );
                throw std::runtime_error(
                    "Default advanced mode rendered without preprocessing"
                );
            }
            catch( const std::logic_error& )
            {
            }
            std::cout << "SPCBPT default-mode initialization passed\n";
            return EXIT_SUCCESS;
        }
        if( !options.optimal_e_export_path.empty() )
        {
            workflow.initializeAlgorithmState();
            workflow.captureOptimalEProblem( options.optimal_e_export_path );
            std::cout << "SPCBPT optimal-E capture passed: "
                      << options.optimal_e_export_path << '\n';
            return EXIT_SUCCESS;
        }

        FrameBuffer frame( SMOKE_WIDTH * SMOKE_HEIGHT );
        if( spcbpt::requiresRendererPreprocessing( renderer_config ) )
        {
            workflow.initializeAlgorithmState();
            workflow.runPreprocessing();
        }
        workflow.renderFrame( frame.get() );

        if( options.reload_check )
        {
            runtime.reloadScene();
            workflow.initializeAlgorithmState();
            workflow.runPreprocessing();
            workflow.renderFrame( frame.get() );

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
            workflow.renderFrame( frame.get() );

            runtime.unloadScene();
            if( runtime.isSceneLoaded() || runtime.isInitialized() )
                throw std::runtime_error( "RendererRuntime unload left live state" );
        }

        std::vector<uchar4> host_pixels( SMOKE_WIDTH * SMOKE_HEIGHT );
        CUDA_CHECK( cudaMemcpy(
            host_pixels.data(),
            frame.get(),
            host_pixels.size() * sizeof( uchar4 ),
            cudaMemcpyDeviceToHost
        ) );
        const uchar4 first_pixel = host_pixels.front();
        const size_t non_black_pixels = static_cast<size_t>( std::count_if(
            host_pixels.begin(),
            host_pixels.end(),
            []( const uchar4& pixel )
            {
                return pixel.x != 0 || pixel.y != 0 || pixel.z != 0;
            }
        ) );
        if( options.validate_frame && non_black_pixels == 0 )
            throw std::runtime_error( "Rendered frame contains no non-black pixels" );
        std::cout << "SPCBPT headless smoke passed: "
                  << SMOKE_WIDTH << 'x' << SMOKE_HEIGHT
                  << ", non-black pixels=" << non_black_pixels
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
