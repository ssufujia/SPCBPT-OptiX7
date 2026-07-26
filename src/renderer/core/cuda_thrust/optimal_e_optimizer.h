#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spcbpt
{

struct OptimalEProblem
{
    int num_paths;
    int num_nodes;
    int num_eye;
    int num_light;

    const float* f_squared;
    const float* fixed_pdf;
    const float* peak_pdf;
    const int*   path_offsets;
    const int*   matrix_indices;
    const int*   active_light = nullptr;
    int          active_light_count = 0;
};

struct OptimalEOptimizerOptions
{
    float conservative_rate = 0.1f;
    float learning_rate = 1.0f;
    int   iterations = 20;
    int   max_backtracking_steps = 12;
    float epsilon = 1e-8f;
};

struct OptimalEOptimizerResult
{
    float initial_objective;
    float final_objective;
    int   accepted_steps;
};

struct OptimalEHostSnapshot
{
    int num_paths;
    int num_nodes;
    int num_eye;
    int num_light;
    float conservative_rate;
    std::uint32_t experiment_seed;

    std::vector<float> f_squared;
    std::vector<float> fixed_pdf;
    std::vector<float> peak_pdf;
    std::vector<int>   path_offsets;
    std::vector<int>   matrix_indices;
    std::vector<int>   active_light;
    std::vector<float> base_distribution;
};

void saveOptimalESnapshot(
    const std::string& path,
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate,
    std::uint32_t experiment_seed = 0
);

OptimalEHostSnapshot loadOptimalESnapshot( const std::string& path );

void normalizeOptimalERows(
    float* base_distribution,
    int num_eye,
    int num_light,
    const int* active_light = nullptr,
    int active_light_count = 0
);

float evaluateOptimalEObjective(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate
);

void computeOptimalEGradient(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float* gradient,
    float conservative_rate
);

void takeOptimalEMirrorStep(
    float* base_distribution,
    const float* gradient,
    int num_eye,
    int num_light,
    float learning_rate,
    float epsilon,
    const int* active_light = nullptr,
    int active_light_count = 0
);

OptimalEOptimizerResult optimizeOptimalE(
    const OptimalEProblem& problem,
    float* base_distribution,
    const OptimalEOptimizerOptions& options = {}
);

} // namespace spcbpt
