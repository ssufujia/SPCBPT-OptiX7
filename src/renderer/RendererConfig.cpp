#include <renderer/RendererConfig.h>

#include <renderer/core/device_compile_config.h>
#include <tinygltf/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace spcbpt
{

namespace
{

RendererAlgorithm parseAlgorithm( const std::string& name )
{
    if( name == "pt" )
        return RendererAlgorithm::PathTracing;
    if( name == "lvcbpt" )
        return RendererAlgorithm::Lvcbpt;
    if( name == "lvcbpt_proxy_experimental" )
        return RendererAlgorithm::LvcbptProxyExperimental;
    throw std::invalid_argument( "Unknown renderer algorithm: " + name );
}

bool isAllowedTopLevelKey( const std::string& key )
{
    return key == "width"
        || key == "height"
        || key == "active_path_depth"
        || key == "connection_count"
        || key == "algorithm"
        || key == "path_guiding";
}

bool isAllowedPathGuidingKey( const std::string& key )
{
    return key == "enabled"
        || key == "self_train"
        || key == "more_training";
}

void validateTopLevelKeys( const nlohmann::json& json )
{
    if( !json.is_object() )
        throw std::invalid_argument( "Renderer config root must be an object" );
    for( auto field = json.begin(); field != json.end(); ++field )
    {
        if( !isAllowedTopLevelKey( field.key() ) )
        {
            throw std::invalid_argument(
                "Unknown renderer config key: " + field.key()
            );
        }
    }
}

unsigned int readUnsignedInteger(
    const nlohmann::json& json,
    const char* key,
    unsigned int fallback
)
{
    const auto field = json.find( key );
    if( field == json.end() )
        return fallback;
    if( !field->is_number_integer() && !field->is_number_unsigned() )
        throw std::invalid_argument( std::string( key ) + " must be an integer" );

    if( field->is_number_unsigned() )
    {
        const auto value = field->get<std::uint64_t>();
        if( value > std::numeric_limits<unsigned int>::max() )
            throw std::invalid_argument( std::string( key ) + " is too large" );
        return static_cast<unsigned int>( value );
    }

    const auto value = field->get<std::int64_t>();
    if( value < 0 )
        throw std::invalid_argument( std::string( key ) + " must not be negative" );
    if( static_cast<std::uint64_t>( value )
        > std::numeric_limits<unsigned int>::max() )
    {
        throw std::invalid_argument( std::string( key ) + " is too large" );
    }
    return static_cast<unsigned int>( value );
}

int readInteger( const nlohmann::json& json, const char* key, int fallback )
{
    const auto field = json.find( key );
    if( field == json.end() )
        return fallback;
    if( !field->is_number_integer() && !field->is_number_unsigned() )
        throw std::invalid_argument( std::string( key ) + " must be an integer" );

    if( field->is_number_unsigned() )
    {
        const auto value = field->get<std::uint64_t>();
        if( value > static_cast<std::uint64_t>( std::numeric_limits<int>::max() ) )
            throw std::invalid_argument( std::string( key ) + " is too large" );
        return static_cast<int>( value );
    }

    const auto value = field->get<std::int64_t>();
    if( value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max() )
    {
        throw std::invalid_argument( std::string( key ) + " is out of range" );
    }
    return static_cast<int>( value );
}

bool readBoolean( const nlohmann::json& json, const char* key, bool fallback )
{
    const auto field = json.find( key );
    if( field == json.end() )
        return fallback;
    if( !field->is_boolean() )
        throw std::invalid_argument( std::string( key ) + " must be a boolean" );
    return field->get<bool>();
}

std::string readString(
    const nlohmann::json& json,
    const char* key,
    const std::string& fallback
)
{
    const auto field = json.find( key );
    if( field == json.end() )
        return fallback;
    if( !field->is_string() )
        throw std::invalid_argument( std::string( key ) + " must be a string" );
    return field->get<std::string>();
}

RendererConfig parseRendererConfig( const nlohmann::json& json )
{
    validateTopLevelKeys( json );

    RendererConfig config;
    config.width = readUnsignedInteger( json, "width", config.width );
    config.height = readUnsignedInteger( json, "height", config.height );
    config.active_path_depth =
        readInteger( json, "active_path_depth", config.active_path_depth );
    config.connection_count =
        readInteger( json, "connection_count", config.connection_count );
    config.algorithm = parseAlgorithm( readString(
        json,
        "algorithm",
        rendererAlgorithmName( config.algorithm )
    ) );

    const auto path_guiding = json.find( "path_guiding" );
    if( path_guiding != json.end() )
    {
        if( !path_guiding->is_object() )
            throw std::invalid_argument( "path_guiding must be an object" );
        for( auto field = path_guiding->begin(); field != path_guiding->end(); ++field )
        {
            if( !isAllowedPathGuidingKey( field.key() ) )
            {
                throw std::invalid_argument(
                    "Unknown path_guiding key: " + field.key()
                );
            }
        }
        config.path_guiding_enabled =
            readBoolean( *path_guiding, "enabled", config.path_guiding_enabled );
        config.path_guiding_self_train =
            readBoolean(
                *path_guiding,
                "self_train",
                config.path_guiding_self_train
            );
        config.path_guiding_more_training =
            readBoolean(
                *path_guiding,
                "more_training",
                config.path_guiding_more_training
            );
    }

    validateRendererConfig( config );
    return config;
}

std::filesystem::path temporaryConfigPath(
    const std::filesystem::path& path
)
{
    static std::atomic<unsigned long long> sequence = 0;
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = path;
    std::string suffix = ".tmp.";
#ifdef _WIN32
    suffix += std::to_string( GetCurrentProcessId() ) + ".";
#endif
    temporary += suffix + std::to_string( timestamp )
        + "." + std::to_string( sequence.fetch_add( 1 ) );
    return temporary;
}

void replaceFileAtomically(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination
)
{
#ifdef _WIN32
    if( ReplaceFileW(
            destination.c_str(),
            temporary.c_str(),
            nullptr,
            REPLACEFILE_IGNORE_MERGE_ERRORS,
            nullptr,
            nullptr
        ) )
    {
        return;
    }

    DWORD error = GetLastError();
    if( error == ERROR_FILE_NOT_FOUND )
    {
        if( MoveFileExW(
                temporary.c_str(),
                destination.c_str(),
                MOVEFILE_WRITE_THROUGH
            ) )
        {
            return;
        }
        error = GetLastError();
    }
    throw std::system_error(
        static_cast<int>( error ),
        std::system_category(),
        "Could not replace renderer config"
    );
#else
    std::filesystem::rename( temporary, destination );
#endif
}

} // namespace

bool operator==( const RendererConfig& lhs, const RendererConfig& rhs )
{
    return lhs.width == rhs.width
        && lhs.height == rhs.height
        && lhs.active_path_depth == rhs.active_path_depth
        && lhs.connection_count == rhs.connection_count
        && lhs.algorithm == rhs.algorithm
        && lhs.path_guiding_enabled == rhs.path_guiding_enabled
        && lhs.path_guiding_self_train == rhs.path_guiding_self_train
        && lhs.path_guiding_more_training == rhs.path_guiding_more_training;
}

bool operator!=( const RendererConfig& lhs, const RendererConfig& rhs )
{
    return !( lhs == rhs );
}

const char* rendererAlgorithmName( RendererAlgorithm algorithm )
{
    switch( algorithm )
    {
        case RendererAlgorithm::PathTracing:
            return "pt";
        case RendererAlgorithm::Lvcbpt:
            return "lvcbpt";
        case RendererAlgorithm::LvcbptProxyExperimental:
            return "lvcbpt_proxy_experimental";
    }
    throw std::invalid_argument( "Invalid renderer algorithm" );
}

const char* rendererAlgorithmDisplayName( RendererAlgorithm algorithm )
{
    switch( algorithm )
    {
        case RendererAlgorithm::PathTracing:
            return "PT";
        case RendererAlgorithm::Lvcbpt:
            return "LVCBPT";
        case RendererAlgorithm::LvcbptProxyExperimental:
            return "LVCBPT + Proxy (Experimental)";
    }
    throw std::invalid_argument( "Invalid renderer algorithm" );
}

bool isAdvancedRendererAlgorithm( RendererAlgorithm algorithm )
{
    return algorithm != RendererAlgorithm::PathTracing;
}

bool usesProxyRendererAlgorithm( RendererAlgorithm algorithm )
{
    return algorithm == RendererAlgorithm::LvcbptProxyExperimental;
}

bool requiresRendererPreprocessing( const RendererConfig& config )
{
    return isAdvancedRendererAlgorithm( config.algorithm )
        || config.path_guiding_enabled;
}

void validateRendererConfig( const RendererConfig& config )
{
    if( config.width == 0 || config.height == 0 )
        throw std::invalid_argument( "Renderer dimensions must be non-zero" );
    if( config.width > MAX_RENDER_WIDTH || config.height > MAX_RENDER_HEIGHT )
    {
        throw std::invalid_argument(
            "Renderer dimensions must not exceed "
            + std::to_string( MAX_RENDER_WIDTH ) + "x"
            + std::to_string( MAX_RENDER_HEIGHT )
        );
    }
    if( config.active_path_depth <= 0
        || config.active_path_depth > SPCBPT_DEVICE_MAX_PATH_DEPTH )
    {
        throw std::invalid_argument(
            "Active path depth must be in [1, "
            + std::to_string( SPCBPT_DEVICE_MAX_PATH_DEPTH ) + "]"
        );
    }
    if( config.connection_count <= 0
        || config.connection_count > MAX_CONNECTION_COUNT )
    {
        throw std::invalid_argument(
            "Connection count must be in [1, "
            + std::to_string( MAX_CONNECTION_COUNT ) + "]"
        );
    }
    rendererAlgorithmName( config.algorithm );
}

RendererConfigChange classifyRendererConfigChange(
    const RendererConfig& current,
    const RendererConfig& next
)
{
    if( current == next )
        return RendererConfigChange::None;

    RendererConfig current_at_next_size = current;
    current_at_next_size.width = next.width;
    current_at_next_size.height = next.height;
    if( current_at_next_size != next )
        return RendererConfigChange::Rebuild;

    const std::uint64_t current_aspect =
        static_cast<std::uint64_t>( current.width ) * next.height;
    const std::uint64_t next_aspect =
        static_cast<std::uint64_t>( next.width ) * current.height;
    return current_aspect == next_aspect
        ? RendererConfigChange::Resize
        : RendererConfigChange::Rebuild;
}

RendererConfig loadRendererConfig( const std::filesystem::path& path )
{
    std::ifstream input( path );
    if( !input )
        throw std::runtime_error( "Could not open renderer config: " + path.string() );

    nlohmann::json json;
    try
    {
        json = nlohmann::json::parse( input );
        return parseRendererConfig( json );
    }
    catch( const std::invalid_argument& )
    {
        throw;
    }
    catch( const nlohmann::json::exception& error )
    {
        throw std::invalid_argument(
            "Invalid renderer config " + path.string() + ": " + error.what()
        );
    }
}

void saveRendererConfig(
    const std::filesystem::path& path,
    const RendererConfig& config
)
{
    validateRendererConfig( config );
    const nlohmann::json json = {
        { "width", config.width },
        { "height", config.height },
        { "active_path_depth", config.active_path_depth },
        { "connection_count", config.connection_count },
        { "algorithm", rendererAlgorithmName( config.algorithm ) },
        {
            "path_guiding",
            {
                { "enabled", config.path_guiding_enabled },
                { "self_train", config.path_guiding_self_train },
                { "more_training", config.path_guiding_more_training }
            }
        }
    };

    if( path.empty() || path.filename().empty() )
        throw std::invalid_argument( "Renderer config path must name a file" );

    const std::filesystem::path temporary = temporaryConfigPath( path );
    try
    {
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc
        );
        if( !output )
        {
            throw std::runtime_error(
                "Could not write renderer config: " + temporary.string()
            );
        }
        output << json.dump( 2 ) << '\n';
        output.flush();
        if( !output )
            throw std::runtime_error( "Could not flush renderer config" );
        output.close();
        if( !output )
            throw std::runtime_error( "Could not close renderer config" );
        replaceFileAtomically( temporary, path );
    }
    catch( ... )
    {
        std::error_code ignored;
        std::filesystem::remove( temporary, ignored );
        throw;
    }
}

} // namespace spcbpt
