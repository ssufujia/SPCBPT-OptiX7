#include <renderer/RendererConfig.h>
#include <renderer/SamplingProgress.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{

void require( bool condition, const char* message )
{
    if( !condition )
        throw std::runtime_error( message );
}

void writeText( const std::filesystem::path& path, const std::string& text )
{
    std::ofstream output( path, std::ios::binary | std::ios::trunc );
    output << text;
    if( !output )
        throw std::runtime_error( "Test fixture write failed" );
}

std::string readText( const std::filesystem::path& path )
{
    std::ifstream input( path, std::ios::binary );
    return {
        std::istreambuf_iterator<char>( input ),
        std::istreambuf_iterator<char>()
    };
}

void requireInvalidJson(
    const std::filesystem::path& path,
    const std::string& text
)
{
    writeText( path, text );
    try
    {
        (void)spcbpt::loadRendererConfig( path );
    }
    catch( const std::invalid_argument& )
    {
        return;
    }
    catch( const std::exception& )
    {
        throw std::runtime_error(
            "Invalid JSON must report std::invalid_argument"
        );
    }
    throw std::runtime_error( "Invalid renderer JSON was accepted" );
}

bool hasTemporaryConfigFile(
    const std::filesystem::path& directory,
    const std::filesystem::path& target
)
{
    const std::string prefix = target.filename().string() + ".tmp.";
    for( const auto& entry : std::filesystem::directory_iterator( directory ) )
    {
        if( entry.path().filename().string().rfind( prefix, 0 ) == 0 )
            return true;
    }
    return false;
}

void checkStrictJson( const std::filesystem::path& path )
{
    requireInvalidJson( path, R"({"width":)" );
    requireInvalidJson( path, R"({"future_option": true})" );
    requireInvalidJson( path, R"({"width": 64.5})" );
    requireInvalidJson( path, R"({"width": -1})" );
    requireInvalidJson( path, R"({"width": 4294967360})" );
    requireInvalidJson( path, R"({"active_path_depth": "12"})" );
    requireInvalidJson( path, R"({"connection_count": false})" );
    requireInvalidJson( path, R"({"algorithm": 1})" );
    requireInvalidJson( path, R"({"path_guiding": {"enabled": 1}})" );
    requireInvalidJson( path, R"({"path_guiding": {"self_trian": true}})" );
    requireInvalidJson( path, R"({"width": 64} trailing)" );
}

void checkConfigBehavior()
{
    using namespace spcbpt;

    RendererConfig config;
    require(
        config.algorithm == RendererAlgorithm::LvcbptProxyExperimental,
        "Default renderer algorithm changed"
    );
    require(
        config.path_guiding_enabled,
        "Default renderer config must enable path guiding"
    );
    require(
        config.path_guiding_self_train,
        "Default renderer config must enable path-guiding self-training"
    );
    require(
        requiresRendererPreprocessing( config ),
        "Default LVCBPT + PG + Proxy mode must require preprocessing"
    );
    require(
        std::string( rendererAlgorithmDisplayName(
            RendererAlgorithm::PathTracing
        ) ) == "PT"
            && std::string( rendererAlgorithmDisplayName(
                RendererAlgorithm::Lvcbpt
            ) ) == "LVCBPT"
            && std::string( rendererAlgorithmDisplayName(
                RendererAlgorithm::LvcbptProxyExperimental
            ) ) == "LVCBPT + Proxy (Experimental)",
        "Public renderer algorithm names changed"
    );

    RendererConfig path_tracing = config;
    path_tracing.algorithm = RendererAlgorithm::PathTracing;
    path_tracing.path_guiding_enabled = false;
    require(
        !requiresRendererPreprocessing( path_tracing ),
        "Plain PT should not require preprocessing"
    );
    path_tracing.path_guiding_enabled = true;
    require(
        requiresRendererPreprocessing( path_tracing ),
        "Path-guided PT must require preprocessing"
    );

    RendererConfig changed = config;
    changed.connection_count = 2;
    require(
        classifyRendererConfigChange( config, changed )
            == RendererConfigChange::Rebuild,
        "Persistent changes must rebuild renderer state"
    );

    changed = config;
    changed.connection_count = MAX_CONNECTION_COUNT + 1;
    try
    {
        validateRendererConfig( changed );
        throw std::runtime_error(
            "Excessive connection count was accepted"
        );
    }
    catch( const std::invalid_argument& )
    {
    }

    changed = config;
    changed.width *= 2;
    changed.height *= 2;
    require(
        classifyRendererConfigChange( config, changed )
            == RendererConfigChange::Resize,
        "Same-aspect dimension changes must use the lightweight resize path"
    );

    changed = config;
    changed.width += 1;
    require(
        classifyRendererConfigChange( config, changed )
            == RendererConfigChange::Rebuild,
        "Aspect-ratio changes must rebuild view-dependent preprocessing"
    );
}

void checkSamplingProgressGuard()
{
    spcbpt::detail::SamplingProgressGuard guard( 3 );
    guard.record( 0, "test sampling" );
    guard.record( 1, "test sampling" );
    guard.record( 0, "test sampling" );
    guard.record( 0, "test sampling" );

    bool rejected = false;
    try
    {
        guard.record( 0, "test sampling" );
    }
    catch( const std::runtime_error& error )
    {
        rejected =
            std::string( error.what() ).find( "test sampling" )
            != std::string::npos;
    }
    require(
        rejected,
        "Sampling must fail after the configured consecutive no-progress limit"
    );
}

void checkDimensionLimits()
{
    spcbpt::RendererConfig config;
    config.width = spcbpt::MAX_RENDER_WIDTH;
    config.height = spcbpt::MAX_RENDER_HEIGHT;
    spcbpt::validateRendererConfig( config );

    config.width = spcbpt::MAX_RENDER_WIDTH + 1;
    bool rejected = false;
    try
    {
        spcbpt::validateRendererConfig( config );
    }
    catch( const std::invalid_argument& )
    {
        rejected = true;
    }
    require( rejected, "Renderer width limit was not enforced" );

    config.width = spcbpt::MAX_RENDER_WIDTH;
    config.height = spcbpt::MAX_RENDER_HEIGHT + 1;
    rejected = false;
    try
    {
        spcbpt::validateRendererConfig( config );
    }
    catch( const std::invalid_argument& )
    {
        rejected = true;
    }
    require( rejected, "Renderer height limit was not enforced" );
}

#ifdef _WIN32
HANDLE openConfigForRead(
    const std::filesystem::path& path,
    DWORD sharing
)
{
    HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        sharing,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if( handle == INVALID_HANDLE_VALUE )
        throw std::runtime_error( "Could not lock config test file" );
    return handle;
}

void checkAtomicWindowsSave(
    const std::filesystem::path& directory,
    const std::filesystem::path& path
)
{
    spcbpt::RendererConfig original;
    spcbpt::saveRendererConfig( path, original );

    spcbpt::RendererConfig replacement = original;
    replacement.connection_count = 7;
    HANDLE readable = openConfigForRead(
        path,
        FILE_SHARE_READ | FILE_SHARE_DELETE
    );
    spcbpt::saveRendererConfig( path, replacement );
    CloseHandle( readable );
    require(
        spcbpt::loadRendererConfig( path ) == replacement,
        "Atomic save must replace a destination open for reading"
    );

    const std::string preserved = readText( path );
    HANDLE blocked = openConfigForRead( path, FILE_SHARE_READ );
    bool save_failed = false;
    try
    {
        spcbpt::saveRendererConfig( path, original );
    }
    catch( const std::exception& )
    {
        save_failed = true;
    }
    CloseHandle( blocked );
    require( save_failed, "Blocked atomic replacement should fail" );
    require(
        readText( path ) == preserved,
        "Failed save damaged the previous config"
    );
    require(
        !hasTemporaryConfigFile( directory, path ),
        "Failed save left a temporary config file"
    );
}
#endif

} // namespace

int main()
{
    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / ( "spcbpt-renderer-config-test-" + std::to_string( unique_id ) );
    std::filesystem::create_directory( directory );
    const std::filesystem::path path = directory / "renderer.json";

    try
    {
        checkConfigBehavior();
        checkSamplingProgressGuard();
        checkDimensionLimits();
        checkStrictJson( path );

        spcbpt::RendererConfig config;
        config.path_guiding_enabled = true;
        spcbpt::saveRendererConfig( path, config );
        require(
            spcbpt::loadRendererConfig( path ) == config,
            "Renderer config JSON round trip changed values"
        );
#ifdef _WIN32
        checkAtomicWindowsSave( directory, path );
#endif
        std::filesystem::remove( path );
        std::filesystem::remove( directory );
    }
    catch( ... )
    {
        std::error_code ignored;
        std::filesystem::remove_all( directory, ignored );
        throw;
    }
    return 0;
}
