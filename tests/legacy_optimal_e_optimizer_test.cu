#include "optimal_e_optimizer.h"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int CUDA_BLOCK_SIZE = 256;

struct LegacyOptions
{
    int   steps = 20;
    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    float probability_epsilon = 1e-7f;
};

struct LegacyResult
{
    float initial_objective = 0.0f;
    float parameterized_initial_objective = 0.0f;
    float final_objective = 0.0f;
    double elapsed_seconds = 0.0;
};

struct CommandLine
{
    std::string snapshot_path;
    std::string output_q_path;
    std::string metrics_path;
    LegacyOptions optimizer;
};

void checkCuda( cudaError_t error, const char* operation )
{
    if( error != cudaSuccess )
    {
        throw std::runtime_error(
            std::string( operation ) + ": " + cudaGetErrorString( error )
        );
    }
}

void require( bool condition, const std::string& message )
{
    if( !condition )
        throw std::runtime_error( message );
}

void requireNear(
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance,
    const std::string& message
)
{
    const double tolerance = absolute_tolerance
        + relative_tolerance * std::abs( expected );
    if( !std::isfinite( actual ) || std::abs( actual - expected ) > tolerance )
    {
        throw std::runtime_error(
            message + ": expected " + std::to_string( expected )
            + ", got " + std::to_string( actual )
        );
    }
}

__host__ __device__ double stableSigmoid( double value )
{
    if( value >= 0.0 )
        return 1.0 / ( 1.0 + exp( -value ) );
    const double exponential = exp( value );
    return exponential / ( 1.0 + exponential );
}

__global__ void initializeLogitsKernel(
    const float* base_q,
    float* logits,
    int matrix_size,
    int num_light,
    const int* active_light,
    float epsilon
)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index >= matrix_size )
        return;
    const int light = index % num_light;
    if( active_light && !active_light[light] )
    {
        logits[index] = 0.0f;
        return;
    }
    const double lower_bound = static_cast<double>( epsilon );
    const double probability = fmin(
        1.0 - lower_bound,
        fmax( lower_bound, static_cast<double>( base_q[index] ) )
    );
    logits[index] = static_cast<float>(
        log( probability / ( 1.0 - probability ) )
    );
}

__global__ void normalizedSigmoidKernel(
    const float* logits,
    float* base_q,
    int num_light,
    const int* active_light
)
{
    extern __shared__ double shared[];
    const int eye = blockIdx.x;
    double thread_sum = 0.0;
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        if( !active_light || active_light[light] )
            thread_sum += stableSigmoid( logits[eye * num_light + light] );
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for( int stride = blockDim.x / 2; stride > 0; stride /= 2 )
    {
        if( threadIdx.x < stride )
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    const double row_sum = shared[0];
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        const int index = eye * num_light + light;
        base_q[index] = active_light && !active_light[light]
            ? 0.0f
            : static_cast<float>( stableSigmoid( logits[index] ) / row_sum );
    }
}

__global__ void normalizedSigmoidGradientKernel(
    const float* logits,
    const float* base_q,
    const float* base_gradient,
    float* theta_gradient,
    int num_light,
    const int* active_light
)
{
    extern __shared__ double shared[];
    double* sigmoid_sums = shared;
    double* gradient_dots = shared + blockDim.x;
    const int eye = blockIdx.x;
    double thread_sigmoid_sum = 0.0;
    double thread_gradient_dot = 0.0;

    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        if( active_light && !active_light[light] )
            continue;
        const int index = eye * num_light + light;
        thread_sigmoid_sum += stableSigmoid( logits[index] );
        thread_gradient_dot +=
            static_cast<double>( base_gradient[index] ) * base_q[index];
    }
    sigmoid_sums[threadIdx.x] = thread_sigmoid_sum;
    gradient_dots[threadIdx.x] = thread_gradient_dot;
    __syncthreads();

    for( int stride = blockDim.x / 2; stride > 0; stride /= 2 )
    {
        if( threadIdx.x < stride )
        {
            sigmoid_sums[threadIdx.x] += sigmoid_sums[threadIdx.x + stride];
            gradient_dots[threadIdx.x] += gradient_dots[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const double sigmoid_sum = sigmoid_sums[0];
    const double gradient_dot = gradient_dots[0];
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        const int index = eye * num_light + light;
        if( active_light && !active_light[light] )
        {
            theta_gradient[index] = 0.0f;
            continue;
        }
        const double sigmoid_value = stableSigmoid( logits[index] );
        theta_gradient[index] = static_cast<float>(
            sigmoid_value * ( 1.0 - sigmoid_value ) / sigmoid_sum
            * ( static_cast<double>( base_gradient[index] ) - gradient_dot )
        );
    }
}

__global__ void adamStepKernel(
    float* logits,
    const float* gradient,
    float* first_moment,
    float* second_moment,
    int matrix_size,
    int num_light,
    const int* active_light,
    float learning_rate,
    float beta1,
    float beta2,
    float one_minus_beta1_power,
    float one_minus_beta2_power,
    float epsilon
)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index >= matrix_size )
        return;
    const int light = index % num_light;
    if( active_light && !active_light[light] )
        return;

    const float value = gradient[index];
    const float moment1 =
        beta1 * first_moment[index] + ( 1.0f - beta1 ) * value;
    const float moment2 =
        beta2 * second_moment[index] + ( 1.0f - beta2 ) * value * value;
    first_moment[index] = moment1;
    second_moment[index] = moment2;
    const float corrected1 = moment1 / one_minus_beta1_power;
    const float corrected2 = moment2 / one_minus_beta2_power;
    logits[index] -= learning_rate * corrected1
        / ( sqrtf( corrected2 ) + epsilon );
}

void updateBaseQ(
    const thrust::device_vector<float>& logits,
    thrust::device_vector<float>& base_q,
    int num_eye,
    int num_light,
    const thrust::device_vector<int>& active_light
)
{
    normalizedSigmoidKernel<<<num_eye, CUDA_BLOCK_SIZE,
        CUDA_BLOCK_SIZE * sizeof( double )>>>(
        thrust::raw_pointer_cast( logits.data() ),
        thrust::raw_pointer_cast( base_q.data() ),
        num_light,
        active_light.empty()
            ? nullptr
            : thrust::raw_pointer_cast( active_light.data() )
    );
    checkCuda( cudaGetLastError(), "Compute legacy normalized sigmoid" );
}

LegacyResult optimizeLegacySigmoidAdam(
    const spcbpt::OptimalEProblem& problem,
    thrust::device_vector<float>& base_q,
    const thrust::device_vector<int>& active_light,
    float conservative_rate,
    const LegacyOptions& options
)
{
    require( options.steps > 0, "Legacy optimizer steps must be positive" );
    require(
        options.learning_rate > 0.0f
            && std::isfinite( options.learning_rate ),
        "Legacy optimizer learning rate must be finite and positive"
    );

    const int matrix_size = problem.num_eye * problem.num_light;
    thrust::device_vector<float> logits( matrix_size );
    thrust::device_vector<float> base_gradient( matrix_size );
    thrust::device_vector<float> theta_gradient( matrix_size );
    thrust::device_vector<float> first_moment( matrix_size, 0.0f );
    thrust::device_vector<float> second_moment( matrix_size, 0.0f );
    const int blocks =
        ( matrix_size + CUDA_BLOCK_SIZE - 1 ) / CUDA_BLOCK_SIZE;
    const int* active_ptr = active_light.empty()
        ? nullptr
        : thrust::raw_pointer_cast( active_light.data() );
    LegacyResult result;
    result.initial_objective = spcbpt::evaluateOptimalEObjective(
        problem,
        thrust::raw_pointer_cast( base_q.data() ),
        conservative_rate
    );

    initializeLogitsKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        thrust::raw_pointer_cast( base_q.data() ),
        thrust::raw_pointer_cast( logits.data() ),
        matrix_size,
        problem.num_light,
        active_ptr,
        options.probability_epsilon
    );
    checkCuda( cudaGetLastError(), "Initialize legacy logits" );
    updateBaseQ(
        logits,
        base_q,
        problem.num_eye,
        problem.num_light,
        active_light
    );
    checkCuda( cudaDeviceSynchronize(), "Synchronize legacy initialization" );

    result.parameterized_initial_objective =
        spcbpt::evaluateOptimalEObjective(
        problem,
        thrust::raw_pointer_cast( base_q.data() ),
        conservative_rate
    );
    const auto start = std::chrono::steady_clock::now();
    for( int step = 1; step <= options.steps; ++step )
    {
        spcbpt::computeOptimalEGradient(
            problem,
            thrust::raw_pointer_cast( base_q.data() ),
            thrust::raw_pointer_cast( base_gradient.data() ),
            conservative_rate
        );
        normalizedSigmoidGradientKernel<<<problem.num_eye, CUDA_BLOCK_SIZE,
            2 * CUDA_BLOCK_SIZE * sizeof( double )>>>(
            thrust::raw_pointer_cast( logits.data() ),
            thrust::raw_pointer_cast( base_q.data() ),
            thrust::raw_pointer_cast( base_gradient.data() ),
            thrust::raw_pointer_cast( theta_gradient.data() ),
            problem.num_light,
            active_ptr
        );
        checkCuda(
            cudaGetLastError(),
            "Compute fixed legacy normalized-sigmoid gradient"
        );
        adamStepKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast( logits.data() ),
            thrust::raw_pointer_cast( theta_gradient.data() ),
            thrust::raw_pointer_cast( first_moment.data() ),
            thrust::raw_pointer_cast( second_moment.data() ),
            matrix_size,
            problem.num_light,
            active_ptr,
            options.learning_rate,
            options.beta1,
            options.beta2,
            1.0f - std::pow( options.beta1, static_cast<float>( step ) ),
            1.0f - std::pow( options.beta2, static_cast<float>( step ) ),
            options.adam_epsilon
        );
        checkCuda( cudaGetLastError(), "Take fixed legacy Adam step" );
        updateBaseQ(
            logits,
            base_q,
            problem.num_eye,
            problem.num_light,
            active_light
        );
    }
    checkCuda( cudaDeviceSynchronize(), "Synchronize legacy optimizer" );
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    result.final_objective = spcbpt::evaluateOptimalEObjective(
        problem,
        thrust::raw_pointer_cast( base_q.data() ),
        conservative_rate
    );
    return result;
}

spcbpt::OptimalEProblem makeProblem(
    const spcbpt::OptimalEHostSnapshot& snapshot,
    const thrust::device_vector<float>& f_squared,
    const thrust::device_vector<float>& fixed_pdf,
    const thrust::device_vector<float>& peak_pdf,
    const thrust::device_vector<int>& path_offsets,
    const thrust::device_vector<int>& matrix_indices,
    const thrust::device_vector<int>& active_light
)
{
    spcbpt::OptimalEProblem problem{};
    problem.num_paths = snapshot.num_paths;
    problem.num_nodes = snapshot.num_nodes;
    problem.num_eye = snapshot.num_eye;
    problem.num_light = snapshot.num_light;
    problem.f_squared = thrust::raw_pointer_cast( f_squared.data() );
    problem.fixed_pdf = thrust::raw_pointer_cast( fixed_pdf.data() );
    problem.peak_pdf = thrust::raw_pointer_cast( peak_pdf.data() );
    problem.path_offsets = thrust::raw_pointer_cast( path_offsets.data() );
    problem.matrix_indices =
        thrust::raw_pointer_cast( matrix_indices.data() );
    problem.active_light = active_light.empty()
        ? nullptr
        : thrust::raw_pointer_cast( active_light.data() );
    problem.active_light_count = active_light.empty()
        ? 0
        : std::accumulate(
            snapshot.active_light.begin(),
            snapshot.active_light.end(),
            0
        );
    return problem;
}

void writeRawQ(
    const std::filesystem::path& path,
    const thrust::host_vector<float>& base_q
)
{
    if( !path.parent_path().empty() )
        std::filesystem::create_directories( path.parent_path() );
    std::ofstream output( path, std::ios::binary | std::ios::trunc );
    output.write(
        reinterpret_cast<const char*>( base_q.data() ),
        static_cast<std::streamsize>( base_q.size() * sizeof( float ) )
    );
    if( !output )
        throw std::runtime_error( "Cannot write legacy optimizer q" );
}

void writeMetrics(
    const std::filesystem::path& path,
    const LegacyResult& result,
    const LegacyOptions& options
)
{
    if( !path.parent_path().empty() )
        std::filesystem::create_directories( path.parent_path() );
    std::ofstream output( path, std::ios::trunc );
    const double relative_improvement =
        ( result.initial_objective - result.final_objective )
        / result.initial_objective;
    output << std::setprecision( 17 )
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"optimizer\": \"fixed_legacy_sigmoid_adam_cuda\",\n"
        << "  \"steps\": " << options.steps << ",\n"
        << "  \"learning_rate\": " << options.learning_rate << ",\n"
        << "  \"initial_loss\": " << result.initial_objective << ",\n"
        << "  \"parameterized_initial_loss\": "
        << result.parameterized_initial_objective << ",\n"
        << "  \"parameterization_abs_error\": "
        << std::abs(
            result.parameterized_initial_objective
            - result.initial_objective
        ) << ",\n"
        << "  \"final_loss\": " << result.final_objective << ",\n"
        << "  \"relative_improvement\": " << relative_improvement << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << "\n"
        << "}\n";
    if( !output )
        throw std::runtime_error( "Cannot write legacy optimizer metrics" );
}

CommandLine parseCommandLine( int argc, char* argv[] )
{
    CommandLine command;
    for( int index = 1; index < argc; ++index )
    {
        const std::string argument( argv[index] );
        if( argument.rfind( "--snapshot=", 0 ) == 0 )
            command.snapshot_path = argument.substr( 11 );
        else if( argument.rfind( "--output-q=", 0 ) == 0 )
            command.output_q_path = argument.substr( 11 );
        else if( argument.rfind( "--metrics-output=", 0 ) == 0 )
            command.metrics_path = argument.substr( 17 );
        else if( argument.rfind( "--steps=", 0 ) == 0 )
            command.optimizer.steps = std::stoi( argument.substr( 8 ) );
        else if( argument.rfind( "--learning-rate=", 0 ) == 0 )
        {
            command.optimizer.learning_rate =
                std::stof( argument.substr( 16 ) );
        }
        else
            throw std::invalid_argument( "Unknown argument: " + argument );
    }
    require( !command.snapshot_path.empty(), "--snapshot is required" );
    require( !command.output_q_path.empty(), "--output-q is required" );
    require( !command.metrics_path.empty(), "--metrics-output is required" );
    return command;
}

int runSyntheticTest()
{
    constexpr int NUM_EYE = 2;
    constexpr int NUM_LIGHT = 5;
    constexpr float CONSERVATIVE_RATE = 0.2f;
    const std::vector<float> host_f_squared{ 4.0f, 1.5f, 2.0f };
    const std::vector<float> host_fixed_pdf{ 0.8f, 0.6f, 1.1f };
    const std::vector<float> host_peak_pdf{ 0.25f, 3.0f, 1.2f, 0.8f };
    const std::vector<int> host_path_offsets{ 0, 2, 3, 4 };
    const std::vector<int> host_matrix_indices{ 0, 1, 2, 3 };
    const std::vector<int> host_active_light{ 1, 1, 1, 1, 0 };
    const std::vector<float> host_initial_q{
        0.25f, 0.25f, 0.25f, 0.25f, 0.0f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.0f
    };

    spcbpt::OptimalEHostSnapshot snapshot{};
    snapshot.num_paths = static_cast<int>( host_f_squared.size() );
    snapshot.num_nodes = static_cast<int>( host_peak_pdf.size() );
    snapshot.num_eye = NUM_EYE;
    snapshot.num_light = NUM_LIGHT;
    snapshot.conservative_rate = CONSERVATIVE_RATE;
    snapshot.active_light = host_active_light;

    thrust::device_vector<float> f_squared(
        host_f_squared.begin(), host_f_squared.end()
    );
    thrust::device_vector<float> fixed_pdf(
        host_fixed_pdf.begin(), host_fixed_pdf.end()
    );
    thrust::device_vector<float> peak_pdf(
        host_peak_pdf.begin(), host_peak_pdf.end()
    );
    thrust::device_vector<int> path_offsets(
        host_path_offsets.begin(), host_path_offsets.end()
    );
    thrust::device_vector<int> matrix_indices(
        host_matrix_indices.begin(), host_matrix_indices.end()
    );
    thrust::device_vector<int> active_light(
        host_active_light.begin(), host_active_light.end()
    );
    thrust::device_vector<float> base_q(
        host_initial_q.begin(), host_initial_q.end()
    );
    const spcbpt::OptimalEProblem problem = makeProblem(
        snapshot,
        f_squared,
        fixed_pdf,
        peak_pdf,
        path_offsets,
        matrix_indices,
        active_light
    );

    thrust::device_vector<float> logits( NUM_EYE * NUM_LIGHT );
    thrust::device_vector<float> base_gradient( NUM_EYE * NUM_LIGHT );
    thrust::device_vector<float> theta_gradient( NUM_EYE * NUM_LIGHT );
    initializeLogitsKernel<<<1, CUDA_BLOCK_SIZE>>>(
        thrust::raw_pointer_cast( base_q.data() ),
        thrust::raw_pointer_cast( logits.data() ),
        NUM_EYE * NUM_LIGHT,
        NUM_LIGHT,
        thrust::raw_pointer_cast( active_light.data() ),
        LegacyOptions{}.probability_epsilon
    );
    updateBaseQ( logits, base_q, NUM_EYE, NUM_LIGHT, active_light );
    spcbpt::computeOptimalEGradient(
        problem,
        thrust::raw_pointer_cast( base_q.data() ),
        thrust::raw_pointer_cast( base_gradient.data() ),
        CONSERVATIVE_RATE
    );
    normalizedSigmoidGradientKernel<<<NUM_EYE, CUDA_BLOCK_SIZE,
        2 * CUDA_BLOCK_SIZE * sizeof( double )>>>(
        thrust::raw_pointer_cast( logits.data() ),
        thrust::raw_pointer_cast( base_q.data() ),
        thrust::raw_pointer_cast( base_gradient.data() ),
        thrust::raw_pointer_cast( theta_gradient.data() ),
        NUM_LIGHT,
        thrust::raw_pointer_cast( active_light.data() )
    );
    checkCuda( cudaDeviceSynchronize(), "Check fixed legacy gradient" );
    const thrust::host_vector<float> host_logits = logits;
    const thrust::host_vector<float> host_base_q = base_q;
    const thrust::host_vector<float> host_base_gradient = base_gradient;
    const thrust::host_vector<float> host_theta_gradient = theta_gradient;
    for( int eye = 0; eye < NUM_EYE; ++eye )
    {
        double sigmoid_sum = 0.0;
        double gradient_dot = 0.0;
        for( int light = 0; light < NUM_LIGHT; ++light )
        {
            if( !host_active_light[light] )
                continue;
            const int index = eye * NUM_LIGHT + light;
            sigmoid_sum += stableSigmoid( host_logits[index] );
            gradient_dot +=
                host_base_gradient[index] * host_base_q[index];
        }
        for( int light = 0; light < NUM_LIGHT; ++light )
        {
            const int index = eye * NUM_LIGHT + light;
            const double sigmoid_value =
                stableSigmoid( host_logits[index] );
            const double expected = host_active_light[light]
                ? sigmoid_value * ( 1.0 - sigmoid_value ) / sigmoid_sum
                    * ( host_base_gradient[index] - gradient_dot )
                : 0.0;
            requireNear(
                host_theta_gradient[index],
                expected,
                2e-6,
                2e-5,
                "Fixed legacy chain gradient is incorrect"
            );
        }
    }

    const LegacyResult result = optimizeLegacySigmoidAdam(
        problem,
        base_q,
        active_light,
        CONSERVATIVE_RATE,
        {}
    );
    require(
        result.final_objective < result.initial_objective,
        "Fixed legacy optimizer must reduce the synthetic objective"
    );
    const thrust::host_vector<float> optimized = base_q;
    for( int eye = 0; eye < NUM_EYE; ++eye )
    {
        double row_sum = 0.0;
        for( int light = 0; light < NUM_LIGHT; ++light )
        {
            const float value = optimized[eye * NUM_LIGHT + light];
            require(
                std::isfinite( value ) && value >= 0.0f,
                "Fixed legacy q must be finite and non-negative"
            );
            if( host_active_light[light] )
                row_sum += value;
            else
                requireNear(
                    value,
                    0.0,
                    0.0,
                    0.0,
                    "Fixed legacy q must keep inactive columns zero"
                );
        }
        requireNear(
            row_sum,
            1.0,
            1e-6,
            1e-6,
            "Fixed legacy q row must sum to one"
        );
    }
    std::cout << "Fixed legacy CUDA optimizer test passed: "
              << result.initial_objective << " -> "
              << result.final_objective << '\n';
    return EXIT_SUCCESS;
}

int runSnapshot( const CommandLine& command )
{
    const spcbpt::OptimalEHostSnapshot snapshot =
        spcbpt::loadOptimalESnapshot( command.snapshot_path );
    thrust::device_vector<float> f_squared(
        snapshot.f_squared.begin(), snapshot.f_squared.end()
    );
    thrust::device_vector<float> fixed_pdf(
        snapshot.fixed_pdf.begin(), snapshot.fixed_pdf.end()
    );
    thrust::device_vector<float> peak_pdf(
        snapshot.peak_pdf.begin(), snapshot.peak_pdf.end()
    );
    thrust::device_vector<int> path_offsets(
        snapshot.path_offsets.begin(), snapshot.path_offsets.end()
    );
    thrust::device_vector<int> matrix_indices(
        snapshot.matrix_indices.begin(), snapshot.matrix_indices.end()
    );
    thrust::device_vector<int> active_light(
        snapshot.active_light.begin(), snapshot.active_light.end()
    );
    thrust::device_vector<float> base_q(
        snapshot.base_distribution.begin(),
        snapshot.base_distribution.end()
    );
    const spcbpt::OptimalEProblem problem = makeProblem(
        snapshot,
        f_squared,
        fixed_pdf,
        peak_pdf,
        path_offsets,
        matrix_indices,
        active_light
    );
    const LegacyResult result = optimizeLegacySigmoidAdam(
        problem,
        base_q,
        active_light,
        snapshot.conservative_rate,
        command.optimizer
    );
    const thrust::host_vector<float> host_q = base_q;
    writeRawQ( command.output_q_path, host_q );
    writeMetrics( command.metrics_path, result, command.optimizer );
    std::cout << "Fixed legacy CUDA optimizer: "
              << result.initial_objective << " -> "
              << result.final_objective << " in "
              << result.elapsed_seconds << " s\n";
    return EXIT_SUCCESS;
}

} // namespace

int main( int argc, char* argv[] )
{
    try
    {
        if( argc == 1 )
            return runSyntheticTest();
        return runSnapshot( parseCommandLine( argc, argv ) );
    }
    catch( const std::exception& error )
    {
        std::cerr << "Fixed legacy CUDA optimizer failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
