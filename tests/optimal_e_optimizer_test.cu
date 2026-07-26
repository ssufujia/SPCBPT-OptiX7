#include <optimal_e_optimizer.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int NUM_EYE = 2;
constexpr int NUM_LIGHT = 5;
constexpr int ACTIVE_LIGHT_COUNT = 4;
constexpr float CONSERVATIVE_RATE = 0.1f;

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
    const double tolerance =
        absolute_tolerance
        + relative_tolerance * std::max( std::abs( actual ), std::abs( expected ) );
    if( std::abs( actual - expected ) > tolerance )
    {
        throw std::runtime_error(
            message + ": expected " + std::to_string( expected )
            + ", got " + std::to_string( actual )
        );
    }
}

struct HostProblem
{
    std::vector<double> f_squared;
    std::vector<double> fixed_pdf;
    std::vector<double> peak_pdf;
    std::vector<int> path_offsets;
    std::vector<int> matrix_indices;
};

double objective(
    const HostProblem& problem,
    const std::vector<double>& base_distribution
)
{
    double result = 0.0;
    for( int path = 0; path < static_cast<int>( problem.f_squared.size() ); ++path )
    {
        double density = problem.fixed_pdf[path];
        for( int node = problem.path_offsets[path];
             node < problem.path_offsets[path + 1];
             ++node )
        {
            const double mixed_probability =
                ( 1.0 - CONSERVATIVE_RATE )
                    * base_distribution[problem.matrix_indices[node]]
                + CONSERVATIVE_RATE / ACTIVE_LIGHT_COUNT;
            density += mixed_probability * problem.peak_pdf[node];
        }
        result += problem.f_squared[path] / density;
    }
    return result;
}

std::vector<double> gradient(
    const HostProblem& problem,
    const std::vector<double>& base_distribution
)
{
    std::vector<double> result( NUM_EYE * NUM_LIGHT, 0.0 );
    for( int path = 0; path < static_cast<int>( problem.f_squared.size() ); ++path )
    {
        double density = problem.fixed_pdf[path];
        for( int node = problem.path_offsets[path];
             node < problem.path_offsets[path + 1];
             ++node )
        {
            density += (
                ( 1.0 - CONSERVATIVE_RATE )
                    * base_distribution[problem.matrix_indices[node]]
                + CONSERVATIVE_RATE / ACTIVE_LIGHT_COUNT
            ) * problem.peak_pdf[node];
        }
        const double path_gradient =
            -( 1.0 - CONSERVATIVE_RATE )
            * problem.f_squared[path] / ( density * density );
        for( int node = problem.path_offsets[path];
             node < problem.path_offsets[path + 1];
             ++node )
        {
            result[problem.matrix_indices[node]] +=
                path_gradient * problem.peak_pdf[node];
        }
    }
    return result;
}

std::vector<double> mirrorStep(
    const std::vector<double>& base_distribution,
    const std::vector<double>& loss_gradient,
    double learning_rate,
    double epsilon
)
{
    std::vector<double> result( base_distribution.size() );
    for( int eye = 0; eye < NUM_EYE; ++eye )
    {
        std::vector<double> scores( NUM_LIGHT );
        for( int light = 0; light < NUM_LIGHT; ++light )
        {
            const int index = eye * NUM_LIGHT + light;
            if( light >= ACTIVE_LIGHT_COUNT )
            {
                scores[light] = -std::numeric_limits<double>::infinity();
                continue;
            }
            scores[light] =
                std::log( std::max( base_distribution[index], epsilon ) )
                - learning_rate * loss_gradient[index];
        }
        const double row_max = *std::max_element( scores.begin(), scores.end() );
        double row_sum = 0.0;
        for( int light = 0; light < NUM_LIGHT; ++light )
        {
            double& score = scores[light];
            if( light >= ACTIVE_LIGHT_COUNT )
            {
                score = 0.0;
                continue;
            }
            score = std::exp( score - row_max );
            row_sum += score;
        }
        for( int light = 0; light < NUM_LIGHT; ++light )
            result[eye * NUM_LIGHT + light] = scores[light] / row_sum;
    }
    return result;
}

template <typename T>
void writeResultValue( std::ofstream& output, const T& value )
{
    output.write(
        reinterpret_cast<const char*>( &value ),
        static_cast<std::streamsize>( sizeof( T ) )
    );
}

void runSnapshotValidation(
    const std::string& snapshot_path,
    const std::string& result_path,
    const std::string& candidate_q_path,
    const std::string& candidate_objective_path,
    float learning_rate
)
{
    const spcbpt::OptimalEHostSnapshot snapshot =
        spcbpt::loadOptimalESnapshot( snapshot_path );
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
    thrust::device_vector<float> base_distribution(
        snapshot.base_distribution.begin(),
        snapshot.base_distribution.end()
    );
    const int active_light_count = std::accumulate(
        snapshot.active_light.begin(),
        snapshot.active_light.end(),
        0
    );
    const spcbpt::OptimalEProblem problem{
        snapshot.num_paths,
        snapshot.num_nodes,
        snapshot.num_eye,
        snapshot.num_light,
        thrust::raw_pointer_cast( f_squared.data() ),
        thrust::raw_pointer_cast( fixed_pdf.data() ),
        thrust::raw_pointer_cast( peak_pdf.data() ),
        thrust::raw_pointer_cast( path_offsets.data() ),
        thrust::raw_pointer_cast( matrix_indices.data() ),
        thrust::raw_pointer_cast( active_light.data() ),
        active_light_count
    };

    const float initial_objective = spcbpt::evaluateOptimalEObjective(
        problem,
        thrust::raw_pointer_cast( base_distribution.data() ),
        snapshot.conservative_rate
    );
    thrust::device_vector<float> initial_gradient(
        snapshot.num_eye * snapshot.num_light
    );
    spcbpt::computeOptimalEGradient(
        problem,
        thrust::raw_pointer_cast( base_distribution.data() ),
        thrust::raw_pointer_cast( initial_gradient.data() ),
        snapshot.conservative_rate
    );
    const spcbpt::OptimalEOptimizerResult result = spcbpt::optimizeOptimalE(
        problem,
        thrust::raw_pointer_cast( base_distribution.data() ),
        {
            snapshot.conservative_rate,
            learning_rate,
            20,
            12,
            1e-8f
        }
    );
    const thrust::host_vector<float> host_gradient = initial_gradient;
    const thrust::host_vector<float> final_q = base_distribution;

    const std::filesystem::path output_path( result_path );
    if( !output_path.parent_path().empty() )
        std::filesystem::create_directories( output_path.parent_path() );
    std::ofstream output( output_path, std::ios::binary | std::ios::trunc );
    if( !output )
        throw std::runtime_error( "Cannot create CUDA validation result" );
    constexpr std::array<char, 8> magic{
        'S', 'P', 'C', 'B', 'R', '0', '0', '1'
    };
    constexpr std::uint32_t schema_version = 1;
    constexpr std::uint32_t endian_marker = 0x01020304;
    output.write( magic.data(), magic.size() );
    writeResultValue( output, schema_version );
    writeResultValue( output, endian_marker );
    writeResultValue(
        output,
        static_cast<std::uint32_t>( snapshot.num_eye )
    );
    writeResultValue(
        output,
        static_cast<std::uint32_t>( snapshot.num_light )
    );
    writeResultValue(
        output,
        static_cast<std::uint32_t>( result.accepted_steps )
    );
    writeResultValue( output, initial_objective );
    writeResultValue( output, result.final_objective );
    output.write(
        reinterpret_cast<const char*>(
            thrust::raw_pointer_cast( host_gradient.data() )
        ),
        static_cast<std::streamsize>(
            host_gradient.size() * sizeof( float )
        )
    );
    output.write(
        reinterpret_cast<const char*>(
            thrust::raw_pointer_cast( final_q.data() )
        ),
        static_cast<std::streamsize>( final_q.size() * sizeof( float ) )
    );
    if( !output )
        throw std::runtime_error( "Cannot finish CUDA validation result" );

    std::cout
        << "Optimal E real snapshot CUDA validation passed: "
        << initial_objective << " -> " << result.final_objective
        << " in " << result.accepted_steps << " accepted steps\n";

    if( !candidate_q_path.empty() )
    {
        const size_t matrix_size =
            static_cast<size_t>( snapshot.num_eye ) * snapshot.num_light;
        const std::filesystem::path input_path( candidate_q_path );
        if( std::filesystem::file_size( input_path )
            != matrix_size * sizeof( float ) )
        {
            throw std::runtime_error(
                "Candidate q size does not match snapshot dimensions"
            );
        }
        std::ifstream candidate_input( input_path, std::ios::binary );
        std::vector<float> candidate_q( matrix_size );
        candidate_input.read(
            reinterpret_cast<char*>( candidate_q.data() ),
            static_cast<std::streamsize>( matrix_size * sizeof( float ) )
        );
        if( !candidate_input )
            throw std::runtime_error( "Cannot read candidate q" );

        for( int eye = 0; eye < snapshot.num_eye; ++eye )
        {
            double row_sum = 0.0;
            for( int light = 0; light < snapshot.num_light; ++light )
            {
                const float value =
                    candidate_q[eye * snapshot.num_light + light];
                if( !std::isfinite( value ) || value < 0.0f )
                    throw std::runtime_error( "Candidate q must be finite and non-negative" );
                if( snapshot.active_light[light] )
                    row_sum += value;
                else if( value != 0.0f )
                    throw std::runtime_error( "Candidate q activates an inactive column" );
            }
            requireNear(
                row_sum,
                1.0,
                1e-5,
                1e-5,
                "Candidate q row must sum to one"
            );
        }

        thrust::device_vector<float> device_candidate_q(
            candidate_q.begin(), candidate_q.end()
        );
        const float candidate_objective = spcbpt::evaluateOptimalEObjective(
            problem,
            thrust::raw_pointer_cast( device_candidate_q.data() ),
            snapshot.conservative_rate
        );
        const std::filesystem::path objective_path( candidate_objective_path );
        if( !objective_path.parent_path().empty() )
            std::filesystem::create_directories( objective_path.parent_path() );
        std::ofstream objective_output( objective_path, std::ios::trunc );
        objective_output
            << std::setprecision( std::numeric_limits<float>::max_digits10 )
            << candidate_objective << '\n';
        if( !objective_output )
            throw std::runtime_error( "Cannot write candidate objective" );
        std::cout << "CUDA candidate-q objective: "
                  << candidate_objective << '\n';
    }
}

} // namespace

int runSyntheticTest()
{
    try
    {
        const HostProblem host_problem{
            { 4.0, 1.5, 2.0 },
            { 0.8, 0.6, 1.1 },
            { 0.25, 3.0, 1.2, 0.8 },
            { 0, 2, 3, 4 },
            { 0, 1, 2, 3 }
        };
        std::vector<double> host_base{
            1.0, 0.0, 0.0, 0.0, 9.0,
            0.0, 0.0, 0.0, 0.0, 7.0
        };
        const std::vector<int> host_active_light{ 1, 1, 1, 1, 0 };

        thrust::device_vector<float> f_squared(
            host_problem.f_squared.begin(),
            host_problem.f_squared.end()
        );
        thrust::device_vector<float> fixed_pdf(
            host_problem.fixed_pdf.begin(),
            host_problem.fixed_pdf.end()
        );
        thrust::device_vector<float> peak_pdf(
            host_problem.peak_pdf.begin(),
            host_problem.peak_pdf.end()
        );
        thrust::device_vector<int> path_offsets(
            host_problem.path_offsets.begin(),
            host_problem.path_offsets.end()
        );
        thrust::device_vector<int> matrix_indices(
            host_problem.matrix_indices.begin(),
            host_problem.matrix_indices.end()
        );
        thrust::device_vector<float> base_distribution(
            host_base.begin(),
            host_base.end()
        );
        thrust::device_vector<int> active_light(
            host_active_light.begin(),
            host_active_light.end()
        );

        const spcbpt::OptimalEProblem problem{
            static_cast<int>( host_problem.f_squared.size() ),
            static_cast<int>( host_problem.peak_pdf.size() ),
            NUM_EYE,
            NUM_LIGHT,
            thrust::raw_pointer_cast( f_squared.data() ),
            thrust::raw_pointer_cast( fixed_pdf.data() ),
            thrust::raw_pointer_cast( peak_pdf.data() ),
            thrust::raw_pointer_cast( path_offsets.data() ),
            thrust::raw_pointer_cast( matrix_indices.data() ),
            thrust::raw_pointer_cast( active_light.data() ),
            ACTIVE_LIGHT_COUNT
        };

        spcbpt::normalizeOptimalERows(
            thrust::raw_pointer_cast( base_distribution.data() ),
            NUM_EYE,
            NUM_LIGHT,
            thrust::raw_pointer_cast( active_light.data() ),
            ACTIVE_LIGHT_COUNT
        );
        const thrust::host_vector<float> normalized = base_distribution;
        for( int light = 0; light < ACTIVE_LIGHT_COUNT; ++light )
        {
            host_base[NUM_LIGHT + light] = 1.0 / ACTIVE_LIGHT_COUNT;
            requireNear(
                normalized[NUM_LIGHT + light],
                1.0 / ACTIVE_LIGHT_COUNT,
                1e-6,
                1e-6,
                "zero row must become uniform"
            );
        }
        host_base[NUM_LIGHT - 1] = 0.0;
        host_base[2 * NUM_LIGHT - 1] = 0.0;
        requireNear(
            normalized[NUM_LIGHT - 1],
            0.0,
            0.0,
            0.0,
            "inactive column must be zero"
        );
        requireNear(
            normalized[2 * NUM_LIGHT - 1],
            0.0,
            0.0,
            0.0,
            "inactive zero-row column must be zero"
        );

        const double host_objective = objective( host_problem, host_base );
        const float device_objective = spcbpt::evaluateOptimalEObjective(
            problem,
            thrust::raw_pointer_cast( base_distribution.data() ),
            CONSERVATIVE_RATE
        );
        requireNear(
            device_objective,
            host_objective,
            1e-6,
            1e-5,
            "GPU objective disagrees with CPU oracle"
        );

        thrust::device_vector<float> device_gradient( NUM_EYE * NUM_LIGHT );
        spcbpt::computeOptimalEGradient(
            problem,
            thrust::raw_pointer_cast( base_distribution.data() ),
            thrust::raw_pointer_cast( device_gradient.data() ),
            CONSERVATIVE_RATE
        );
        const thrust::host_vector<float> host_device_gradient = device_gradient;
        const std::vector<double> host_gradient =
            gradient( host_problem, host_base );
        for( int index = 0; index < NUM_EYE * NUM_LIGHT; ++index )
        {
            requireNear(
                host_device_gradient[index],
                host_gradient[index],
                2e-5,
                2e-4,
                "GPU gradient disagrees with CPU oracle"
            );
        }

        constexpr double LEARNING_RATE = 0.5;
        constexpr double EPSILON = 1e-6;
        const std::vector<double> expected_step =
            mirrorStep( host_base, host_gradient, LEARNING_RATE, EPSILON );
        spcbpt::takeOptimalEMirrorStep(
            thrust::raw_pointer_cast( base_distribution.data() ),
            thrust::raw_pointer_cast( device_gradient.data() ),
            NUM_EYE,
            NUM_LIGHT,
            static_cast<float>( LEARNING_RATE ),
            static_cast<float>( EPSILON ),
            thrust::raw_pointer_cast( active_light.data() ),
            ACTIVE_LIGHT_COUNT
        );
        const thrust::host_vector<float> actual_step = base_distribution;
        for( int index = 0; index < NUM_EYE * NUM_LIGHT; ++index )
        {
            requireNear(
                actual_step[index],
                expected_step[index],
                2e-6,
                2e-5,
                "GPU mirror step disagrees with CPU oracle"
            );
        }
        require(
            actual_step[1] > 0.0f,
            "mirror step must be able to revive a zero-probability column"
        );

        thrust::copy( normalized.begin(), normalized.end(), base_distribution.begin() );
        const float initial_loss = spcbpt::evaluateOptimalEObjective(
            problem,
            thrust::raw_pointer_cast( base_distribution.data() ),
            CONSERVATIVE_RATE
        );
        const spcbpt::OptimalEOptimizerResult result =
            spcbpt::optimizeOptimalE(
                problem,
                thrust::raw_pointer_cast( base_distribution.data() ),
                {
                    CONSERVATIVE_RATE,
                    0.5f,
                    20,
                    12,
                    static_cast<float>( EPSILON )
                }
            );
        require(
            result.final_objective < initial_loss,
            "optimizer must reduce the synthetic objective"
        );
        require(
            result.accepted_steps > 0,
            "optimizer must accept at least one descent step"
        );

        const thrust::host_vector<float> optimized = base_distribution;
        for( int eye = 0; eye < NUM_EYE; ++eye )
        {
            double row_sum = 0.0;
            for( int light = 0; light < NUM_LIGHT; ++light )
            {
                const float probability = optimized[eye * NUM_LIGHT + light];
                require(
                    probability >= 0.0f && std::isfinite( probability ),
                    "optimized probability must be finite and non-negative"
                );
                row_sum += probability;
            }
            requireNear(
                row_sum,
                1.0,
                1e-5,
                1e-5,
                "optimized row must sum to one"
            );
        }
        for( int light = 0; light < ACTIVE_LIGHT_COUNT; ++light )
        {
            requireNear(
                optimized[NUM_LIGHT + light],
                1.0 / ACTIVE_LIGHT_COUNT,
                1e-5,
                1e-5,
                "unobserved row must stay uniform"
            );
        }
        requireNear(
            optimized[NUM_LIGHT - 1],
            0.0,
            0.0,
            0.0,
            "inactive optimized column must remain zero"
        );

        std::cout
            << "Optimal E optimizer test passed: "
            << initial_loss << " -> " << result.final_objective
            << " in " << result.accepted_steps << " accepted steps\n";
        return EXIT_SUCCESS;
    }
    catch( const std::exception& error )
    {
        std::cerr << "Optimal E optimizer test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

int main( int argc, char* argv[] )
{
    if( argc == 1 )
        return runSyntheticTest();

    try
    {
        std::string snapshot_path;
        std::string result_path;
        std::string candidate_q_path;
        std::string candidate_objective_path;
        float learning_rate = 1.0f;
        for( int index = 1; index < argc; ++index )
        {
            const std::string argument = argv[index];
            if( argument.rfind( "--snapshot=", 0 ) == 0
                && argument.size() > 11 )
            {
                snapshot_path = argument.substr( 11 );
            }
            else if( argument.rfind( "--result=", 0 ) == 0
                     && argument.size() > 9 )
            {
                result_path = argument.substr( 9 );
            }
            else if( argument.rfind( "--candidate-q=", 0 ) == 0
                     && argument.size() > 14 )
            {
                candidate_q_path = argument.substr( 14 );
            }
            else if( argument.rfind( "--candidate-objective=", 0 ) == 0
                     && argument.size() > 22 )
            {
                candidate_objective_path = argument.substr( 22 );
            }
            else if( argument.rfind( "--learning-rate=", 0 ) == 0
                     && argument.size() > 16 )
            {
                const std::string value = argument.substr( 16 );
                std::size_t parsed = 0;
                learning_rate = std::stof( value, &parsed );
                if( parsed != value.size()
                    || !std::isfinite( learning_rate )
                    || learning_rate <= 0.0f )
                {
                    throw std::invalid_argument(
                        "--learning-rate must be a finite positive number"
                    );
                }
            }
            else
            {
                throw std::invalid_argument(
                    "Usage: spcbpt_optimal_e_optimizer_test "
                    "[--snapshot=<path> --result=<path> "
                    "[--learning-rate=<positive-float>] "
                    "[--candidate-q=<float32-path> "
                    "--candidate-objective=<path>]]"
                );
            }
        }
        if( snapshot_path.empty() || result_path.empty() )
            throw std::invalid_argument( "Both --snapshot and --result are required" );
        if( candidate_q_path.empty() != candidate_objective_path.empty() )
        {
            throw std::invalid_argument(
                "--candidate-q and --candidate-objective must be provided together"
            );
        }
        runSnapshotValidation(
            snapshot_path,
            result_path,
            candidate_q_path,
            candidate_objective_path,
            learning_rate
        );
        return EXIT_SUCCESS;
    }
    catch( const std::exception& error )
    {
        std::cerr << "Optimal E real snapshot validation failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
