#pragma once

#include <filesystem>
#include <string>

namespace spcbpt
{

enum class RendererAlgorithm
{
    PathTracing = 0,
    Lvcbpt = 1,
    LvcbptProxyExperimental = 2
};

inline constexpr int RENDERER_ALGORITHM_COUNT = 3;
inline constexpr unsigned int MAX_RENDER_WIDTH = 16384;
inline constexpr unsigned int MAX_RENDER_HEIGHT = 16384;
inline constexpr int MAX_CONNECTION_COUNT = 64;
inline constexpr float DEFAULT_OPTIMAL_E_LEARNING_RATE = 1.0f;
inline constexpr int DEFAULT_OPTIMAL_E_ITERATIONS = 20;
inline constexpr int MAX_OPTIMAL_E_ITERATIONS = 10000;

enum class RendererConfigChange
{
    None,
    Resize,
    Rebuild
};

struct RendererConfig
{
    std::string       scene_path                  = "bedroom.scene";
    unsigned int      width                       = 64;
    unsigned int      height                      = 64;
    int               active_path_depth           = 12;
    int               connection_count            = 1;
    RendererAlgorithm algorithm                   =
        RendererAlgorithm::LvcbptProxyExperimental;
    bool              path_guiding_enabled        = true;
    bool              path_guiding_self_train     = true;
    bool              path_guiding_more_training  = false;
    bool              caustic_path_only           = false;
    float             optimal_e_learning_rate     =
        DEFAULT_OPTIMAL_E_LEARNING_RATE;
    int               optimal_e_iterations        = DEFAULT_OPTIMAL_E_ITERATIONS;
};

bool operator==( const RendererConfig& lhs, const RendererConfig& rhs );
bool operator!=( const RendererConfig& lhs, const RendererConfig& rhs );

const char* rendererAlgorithmName( RendererAlgorithm algorithm );
const char* rendererAlgorithmDisplayName( RendererAlgorithm algorithm );
bool isAdvancedRendererAlgorithm( RendererAlgorithm algorithm );
bool usesProxyRendererAlgorithm( RendererAlgorithm algorithm );
bool requiresRendererPreprocessing( const RendererConfig& config );

void validateRendererConfig( const RendererConfig& config );
RendererConfigChange classifyRendererConfigChange(
    const RendererConfig& current,
    const RendererConfig& next
);

RendererConfig loadRendererConfig( const std::filesystem::path& path );
void saveRendererConfig(
    const std::filesystem::path& path,
    const RendererConfig& config
);

} // namespace spcbpt
