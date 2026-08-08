#include "optimal_e_optimizer.h"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <array>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace spcbpt
{

namespace
{

constexpr int CUDA_BLOCK_SIZE = 256;
constexpr std::array<char, 8> SNAPSHOT_MAGIC{
    'S', 'P', 'C', 'B', 'E', '0', '0', '1'
};
constexpr std::uint32_t SNAPSHOT_SCHEMA = 2;
constexpr std::uint32_t ENDIAN_MARKER = 0x01020304;
constexpr std::uint64_t SNAPSHOT_V1_HEADER_SIZE =
    8 + sizeof( std::uint32_t ) * 5 + sizeof( std::uint64_t ) * 2
    + sizeof( float );
constexpr std::uint64_t SNAPSHOT_V2_HEADER_SIZE =
    SNAPSHOT_V1_HEADER_SIZE + sizeof( std::uint32_t );

static_assert( sizeof( int ) == sizeof( std::int32_t ) );
static_assert( sizeof( float ) == 4 );

void checkCuda( cudaError_t error, const char* operation );

template <typename T>
void writeValue( std::ofstream& output, const T& value )
{
    output.write(
        reinterpret_cast<const char*>( &value ),
        static_cast<std::streamsize>( sizeof( T ) )
    );
}

template <typename T>
void writeValues( std::ofstream& output, const std::vector<T>& values )
{
    if( !values.empty() )
    {
        output.write(
            reinterpret_cast<const char*>( values.data() ),
            static_cast<std::streamsize>( values.size() * sizeof( T ) )
        );
    }
}

template <typename T>
T readValue( std::ifstream& input )
{
    T value{};
    input.read(
        reinterpret_cast<char*>( &value ),
        static_cast<std::streamsize>( sizeof( T ) )
    );
    if( !input )
        throw std::runtime_error( "Optimal E snapshot header is truncated" );
    return value;
}

template <typename T>
std::vector<T> readValues( std::ifstream& input, std::size_t count )
{
    std::vector<T> values( count );
    if( count != 0 )
    {
        input.read(
            reinterpret_cast<char*>( values.data() ),
            static_cast<std::streamsize>( count * sizeof( T ) )
        );
        if( !input )
            throw std::runtime_error( "Optimal E snapshot array is truncated" );
    }
    return values;
}

template <typename T>
std::vector<T> copyDeviceValues( const T* values, std::size_t count )
{
    std::vector<T> host_values( count );
    if( count != 0 )
    {
        checkCuda(
            cudaMemcpy(
                host_values.data(),
                values,
                count * sizeof( T ),
                cudaMemcpyDeviceToHost
            ),
            "Copy Optimal E snapshot data"
        );
    }
    return host_values;
}

void checkCuda( cudaError_t error, const char* operation )
{
    if( error != cudaSuccess )
    {
        throw std::runtime_error(
            std::string( operation ) + ": " + cudaGetErrorString( error )
        );
    }
}

struct NegativeValue
{
    __host__ __device__ bool operator()( float value ) const
    {
        return value < 0.0f;
    }
};

struct InvalidFiniteValue
{
    __host__ __device__ bool operator()( float value ) const
    {
        return !isfinite( value );
    }
};

struct InvalidMatrixIndex
{
    int size;

    __host__ __device__ bool operator()( int index ) const
    {
        return index < 0 || index >= size;
    }
};

struct InvalidActiveFlag
{
    __host__ __device__ bool operator()( int value ) const
    {
        return value != 0 && value != 1;
    }
};

struct InvalidInactiveNode
{
    const int* active_light;
    int num_light;
    const int* matrix_indices;
    const float* peak_pdf;

    __host__ __device__ bool operator()( int node ) const
    {
        const int light = matrix_indices[node] % num_light;
        return active_light[light] == 0 && peak_pdf[node] != 0.0f;
    }
};

void validateDimensions( int num_eye, int num_light )
{
    if( num_eye <= 0 || num_light <= 0 )
        throw std::invalid_argument( "Optimal E dimensions must be positive" );
}

void validateConservativeRate( float conservative_rate )
{
    if( !std::isfinite( conservative_rate )
        || conservative_rate < 0.0f
        || conservative_rate >= 1.0f )
    {
        throw std::invalid_argument(
            "Optimal E conservative rate must be in [0, 1)"
        );
    }
}

void validateDeviceValues(
    const float* values,
    int count,
    const char* name
)
{
    if( count <= 0 )
        return;
    if( !values )
        throw std::invalid_argument( std::string( name ) + " is null" );

    const auto begin = thrust::device_pointer_cast( values );
    const auto non_finite_count = thrust::count_if(
            begin,
            begin + count,
            InvalidFiniteValue{}
        );
    const auto negative_count = thrust::count_if(
        begin,
        begin + count,
        NegativeValue{}
    );
    if( non_finite_count != 0 || negative_count != 0 )
    {
        throw std::invalid_argument(
            std::string( name )
            + " contains " + std::to_string( negative_count )
            + " negative and " + std::to_string( non_finite_count )
            + " non-finite values"
        );
    }
}

void validateProblem( const OptimalEProblem& problem )
{
    validateDimensions( problem.num_eye, problem.num_light );
    if( problem.num_paths <= 0 )
        throw std::invalid_argument( "Optimal E requires at least one path" );
    if( problem.num_nodes < 0 )
        throw std::invalid_argument( "Optimal E node count cannot be negative" );
    if( !problem.path_offsets )
        throw std::invalid_argument( "Optimal E path offsets are null" );
    if( problem.active_light )
    {
        const auto active_begin =
            thrust::device_pointer_cast( problem.active_light );
        if( problem.active_light_count <= 0
            || problem.active_light_count > problem.num_light
            || thrust::count_if(
                active_begin,
                active_begin + problem.num_light,
                InvalidActiveFlag{}
            ) != 0
            || thrust::count(
                active_begin,
                active_begin + problem.num_light,
                1
            ) != problem.active_light_count )
        {
            throw std::invalid_argument(
                "Optimal E active-light mask/count is inconsistent"
            );
        }
    }
    else if( problem.active_light_count != 0 )
    {
        throw std::invalid_argument(
            "Optimal E active-light count requires a mask"
        );
    }

    validateDeviceValues(
        problem.f_squared,
        problem.num_paths,
        "Optimal E f_squared"
    );
    validateDeviceValues(
        problem.fixed_pdf,
        problem.num_paths,
        "Optimal E fixed_pdf"
    );
    validateDeviceValues(
        problem.peak_pdf,
        problem.num_nodes,
        "Optimal E peak_pdf"
    );

    const auto offset_begin =
        thrust::device_pointer_cast( problem.path_offsets );
    const thrust::host_vector<int> offsets(
        offset_begin,
        offset_begin + problem.num_paths + 1
    );
    if( offsets.front() != 0 || offsets.back() != problem.num_nodes )
    {
        throw std::invalid_argument(
            "Optimal E path offsets must span every node"
        );
    }
    for( int path = 0; path < problem.num_paths; ++path )
    {
        if( offsets[path] > offsets[path + 1] )
        {
            throw std::invalid_argument(
                "Optimal E path offsets must be monotonic"
            );
        }
    }

    if( problem.num_nodes > 0 )
    {
        if( !problem.matrix_indices )
        {
            throw std::invalid_argument(
                "Optimal E matrix indices are null"
            );
        }
        const auto index_begin =
            thrust::device_pointer_cast( problem.matrix_indices );
        if( thrust::count_if(
                index_begin,
                index_begin + problem.num_nodes,
                InvalidMatrixIndex{ problem.num_eye * problem.num_light }
            ) != 0 )
        {
            throw std::invalid_argument(
                "Optimal E matrix index is outside the matrix"
            );
        }
        if( problem.active_light
            && thrust::count_if(
                thrust::make_counting_iterator( 0 ),
                thrust::make_counting_iterator( problem.num_nodes ),
                InvalidInactiveNode{
                    problem.active_light,
                    problem.num_light,
                    problem.matrix_indices,
                    problem.peak_pdf
                }
            ) != 0 )
        {
            throw std::invalid_argument(
                "Optimal E matrix index refers to an inactive light column"
            );
        }
    }
}

__global__ void normalizeRowsKernel(
    float* values,
    int num_light,
    const int* active_light,
    int active_light_count
)
{
    extern __shared__ double shared[];
    const int eye = blockIdx.x;
    double thread_sum = 0.0;
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        if( !active_light || active_light[light] )
            thread_sum += values[eye * num_light + light];
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
        const bool is_active = !active_light || active_light[light] != 0;
        values[eye * num_light + light] =
            !is_active
            ? 0.0f
            : row_sum > 0.0
            ? static_cast<float>( values[eye * num_light + light] / row_sum )
            : 1.0f / active_light_count;
    }
}

__global__ void computePathDensitiesKernel(
    int num_paths,
    int num_light,
    const float* fixed_pdf,
    const float* peak_pdf,
    const int* path_offsets,
    const int* matrix_indices,
    const float* base_distribution,
    float conservative_rate,
    int active_light_count,
    double* path_densities
)
{
    const int path = blockIdx.x * blockDim.x + threadIdx.x;
    if( path >= num_paths )
        return;

    double density = fixed_pdf[path];
    const double uniform_probability =
        static_cast<double>( conservative_rate ) / active_light_count;
    const double learned_rate = 1.0 - conservative_rate;
    for( int node = path_offsets[path];
         node < path_offsets[path + 1];
         ++node )
    {
        const double probability =
            learned_rate * base_distribution[matrix_indices[node]]
            + uniform_probability;
        density += probability * peak_pdf[node];
    }
    path_densities[path] = density;
}

struct ObjectiveTerm
{
    const float* f_squared;
    const double* path_densities;

    __host__ __device__ double operator()( int path ) const
    {
        return f_squared[path] / path_densities[path];
    }
};

struct InvalidDensity
{
    __host__ __device__ bool operator()( double density ) const
    {
        return !isfinite( density ) || density <= 0.0;
    }
};

__global__ void computeGradientKernel(
    int num_paths,
    const float* f_squared,
    const float* peak_pdf,
    const int* path_offsets,
    const int* matrix_indices,
    const double* path_densities,
    float learned_rate,
    float* gradient
)
{
    const int path = blockIdx.x * blockDim.x + threadIdx.x;
    if( path >= num_paths )
        return;

    const double density = path_densities[path];
    const double path_gradient =
        -static_cast<double>( learned_rate )
        * f_squared[path] / ( density * density );
    for( int node = path_offsets[path];
         node < path_offsets[path + 1];
         ++node )
    {
        atomicAdd(
            gradient + matrix_indices[node],
            static_cast<float>( path_gradient * peak_pdf[node] )
        );
    }
}

__global__ void mirrorStepKernel(
    float* base_distribution,
    const float* gradient,
    int num_light,
    float learning_rate,
    float epsilon,
    const int* active_light
)
{
    extern __shared__ double shared[];
    const int eye = blockIdx.x;
    double thread_max = -1.0e300;
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        if( active_light && !active_light[light] )
            continue;
        const int index = eye * num_light + light;
        const double score =
            log( fmax(
                static_cast<double>( base_distribution[index] ),
                static_cast<double>( epsilon )
            ) )
            - static_cast<double>( learning_rate ) * gradient[index];
        thread_max = fmax( thread_max, score );
    }
    shared[threadIdx.x] = thread_max;
    __syncthreads();

    for( int stride = blockDim.x / 2; stride > 0; stride /= 2 )
    {
        if( threadIdx.x < stride )
            shared[threadIdx.x] =
                fmax( shared[threadIdx.x], shared[threadIdx.x + stride] );
        __syncthreads();
    }

    const double row_max = shared[0];
    double thread_sum = 0.0;
    for( int light = threadIdx.x; light < num_light; light += blockDim.x )
    {
        if( active_light && !active_light[light] )
            continue;
        const int index = eye * num_light + light;
        const double score =
            log( fmax(
                static_cast<double>( base_distribution[index] ),
                static_cast<double>( epsilon )
            ) )
            - static_cast<double>( learning_rate ) * gradient[index];
        thread_sum += exp( score - row_max );
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
        if( active_light && !active_light[light] )
        {
            base_distribution[index] = 0.0f;
            continue;
        }
        const double score =
            log( fmax(
                static_cast<double>( base_distribution[index] ),
                static_cast<double>( epsilon )
            ) )
            - static_cast<double>( learning_rate ) * gradient[index];
        base_distribution[index] =
            static_cast<float>( exp( score - row_max ) / row_sum );
    }
}

thrust::device_vector<double> computePathDensities(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate
)
{
    thrust::device_vector<double> path_densities( problem.num_paths );
    const int blocks =
        ( problem.num_paths + CUDA_BLOCK_SIZE - 1 ) / CUDA_BLOCK_SIZE;
    computePathDensitiesKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        problem.num_paths,
        problem.num_light,
        problem.fixed_pdf,
        problem.peak_pdf,
        problem.path_offsets,
        problem.matrix_indices,
        base_distribution,
        conservative_rate,
        problem.active_light
            ? problem.active_light_count
            : problem.num_light,
        thrust::raw_pointer_cast( path_densities.data() )
    );
    checkCuda( cudaGetLastError(), "Compute Optimal E path densities" );
    if( thrust::count_if(
            path_densities.begin(),
            path_densities.end(),
            InvalidDensity{}
        ) != 0 )
    {
        throw std::runtime_error(
            "Optimal E objective contains a non-positive or non-finite density"
        );
    }
    return path_densities;
}

float evaluateObjective(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate
)
{
    const thrust::device_vector<double> path_densities =
        computePathDensities( problem, base_distribution, conservative_rate );
    thrust::device_vector<double> terms( problem.num_paths );
    thrust::transform(
        thrust::make_counting_iterator( 0 ),
        thrust::make_counting_iterator( problem.num_paths ),
        terms.begin(),
        ObjectiveTerm{
            problem.f_squared,
            thrust::raw_pointer_cast( path_densities.data() )
        }
    );
    return static_cast<float>(
        thrust::reduce( terms.begin(), terms.end(), 0.0 )
    );
}

void computeGradient(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float* gradient,
    float conservative_rate
)
{
    const thrust::device_vector<double> path_densities =
        computePathDensities( problem, base_distribution, conservative_rate );
    const int matrix_size = problem.num_eye * problem.num_light;
    thrust::fill(
        thrust::device_pointer_cast( gradient ),
        thrust::device_pointer_cast( gradient ) + matrix_size,
        0.0f
    );
    const int blocks =
        ( problem.num_paths + CUDA_BLOCK_SIZE - 1 ) / CUDA_BLOCK_SIZE;
    computeGradientKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        problem.num_paths,
        problem.f_squared,
        problem.peak_pdf,
        problem.path_offsets,
        problem.matrix_indices,
        thrust::raw_pointer_cast( path_densities.data() ),
        1.0f - conservative_rate,
        gradient
    );
    checkCuda( cudaGetLastError(), "Compute Optimal E gradient" );
}

void takeMirrorStep(
    float* base_distribution,
    const float* gradient,
    int num_eye,
    int num_light,
    float learning_rate,
    float epsilon,
    const int* active_light
)
{
    mirrorStepKernel<<<
        num_eye,
        CUDA_BLOCK_SIZE,
        CUDA_BLOCK_SIZE * sizeof( double )
    >>>(
        base_distribution,
        gradient,
        num_light,
        learning_rate,
        epsilon,
        active_light
    );
    checkCuda( cudaGetLastError(), "Take Optimal E mirror step" );
}

} // namespace

void saveOptimalESnapshot(
    const std::string& path,
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate,
    std::uint32_t experiment_seed
)
{
    if( path.empty() )
        throw std::invalid_argument( "Optimal E snapshot path is empty" );
    validateProblem( problem );
    validateConservativeRate( conservative_rate );

    const std::size_t matrix_size =
        static_cast<std::size_t>( problem.num_eye ) * problem.num_light;
    validateDeviceValues(
        base_distribution,
        static_cast<int>( matrix_size ),
        "Optimal E base distribution"
    );
    const auto base_begin = thrust::device_pointer_cast( base_distribution );
    thrust::device_vector<float> normalized(
        base_begin,
        base_begin + matrix_size
    );
    normalizeOptimalERows(
        thrust::raw_pointer_cast( normalized.data() ),
        problem.num_eye,
        problem.num_light,
        problem.active_light,
        problem.active_light
            ? problem.active_light_count
            : problem.num_light
    );

    OptimalEHostSnapshot snapshot{
        problem.num_paths,
        problem.num_nodes,
        problem.num_eye,
        problem.num_light,
        conservative_rate,
        experiment_seed,
        copyDeviceValues( problem.f_squared, problem.num_paths ),
        copyDeviceValues( problem.fixed_pdf, problem.num_paths ),
        copyDeviceValues( problem.peak_pdf, problem.num_nodes ),
        copyDeviceValues( problem.path_offsets, problem.num_paths + 1 ),
        copyDeviceValues( problem.matrix_indices, problem.num_nodes ),
        {},
        copyDeviceValues(
            thrust::raw_pointer_cast( normalized.data() ),
            matrix_size
        )
    };
    if( problem.active_light )
    {
        snapshot.active_light =
            copyDeviceValues( problem.active_light, problem.num_light );
    }
    else
    {
        snapshot.active_light.assign( problem.num_light, 1 );
    }

    const std::filesystem::path output_path( path );
    if( !output_path.parent_path().empty() )
        std::filesystem::create_directories( output_path.parent_path() );
    std::ofstream output( output_path, std::ios::binary | std::ios::trunc );
    if( !output )
        throw std::runtime_error( "Cannot create Optimal E snapshot: " + path );

    output.write( SNAPSHOT_MAGIC.data(), SNAPSHOT_MAGIC.size() );
    writeValue( output, SNAPSHOT_SCHEMA );
    writeValue( output, ENDIAN_MARKER );
    writeValue( output, static_cast<std::uint64_t>( problem.num_paths ) );
    writeValue( output, static_cast<std::uint64_t>( problem.num_nodes ) );
    writeValue( output, static_cast<std::uint32_t>( problem.num_eye ) );
    writeValue( output, static_cast<std::uint32_t>( problem.num_light ) );
    const std::uint32_t active_light_count = static_cast<std::uint32_t>(
        std::accumulate(
            snapshot.active_light.begin(),
            snapshot.active_light.end(),
            0
        )
    );
    writeValue( output, active_light_count );
    writeValue( output, experiment_seed );
    writeValue( output, conservative_rate );
    writeValues( output, snapshot.f_squared );
    writeValues( output, snapshot.fixed_pdf );
    writeValues( output, snapshot.peak_pdf );
    writeValues( output, snapshot.path_offsets );
    writeValues( output, snapshot.matrix_indices );
    writeValues( output, snapshot.active_light );
    writeValues( output, snapshot.base_distribution );
    if( !output )
        throw std::runtime_error( "Cannot finish Optimal E snapshot: " + path );
}

OptimalEHostSnapshot loadOptimalESnapshot( const std::string& path )
{
    if( path.empty() )
        throw std::invalid_argument( "Optimal E snapshot path is empty" );
    const std::filesystem::path input_path( path );
    std::ifstream input( input_path, std::ios::binary );
    if( !input )
        throw std::runtime_error( "Cannot open Optimal E snapshot: " + path );

    std::array<char, 8> magic{};
    input.read( magic.data(), magic.size() );
    if( magic != SNAPSHOT_MAGIC )
        throw std::runtime_error( "Unsupported Optimal E snapshot magic" );
    const std::uint32_t schema_version = readValue<std::uint32_t>( input );
    const std::uint32_t endian_marker = readValue<std::uint32_t>( input );
    const std::uint64_t num_paths_64 = readValue<std::uint64_t>( input );
    const std::uint64_t num_nodes_64 = readValue<std::uint64_t>( input );
    const std::uint32_t num_eye_32 = readValue<std::uint32_t>( input );
    const std::uint32_t num_light_32 = readValue<std::uint32_t>( input );
    const std::uint32_t active_light_count =
        readValue<std::uint32_t>( input );
    const std::uint32_t experiment_seed =
        schema_version >= 2 ? readValue<std::uint32_t>( input ) : 0;
    const float conservative_rate = readValue<float>( input );

    if( ( schema_version != 1 && schema_version != SNAPSHOT_SCHEMA )
        || endian_marker != ENDIAN_MARKER )
        throw std::runtime_error( "Unsupported Optimal E snapshot schema" );
    if( num_paths_64 == 0 || num_eye_32 == 0 || num_light_32 == 0 )
        throw std::runtime_error( "Optimal E snapshot dimensions are invalid" );
    if( num_paths_64 > static_cast<std::uint64_t>( std::numeric_limits<int>::max() )
        || num_nodes_64 > static_cast<std::uint64_t>( std::numeric_limits<int>::max() )
        || num_eye_32 > static_cast<std::uint32_t>( std::numeric_limits<int>::max() )
        || num_light_32 > static_cast<std::uint32_t>( std::numeric_limits<int>::max() ) )
    {
        throw std::runtime_error( "Optimal E snapshot exceeds production index limits" );
    }
    const std::uint64_t matrix_size_64 =
        static_cast<std::uint64_t>( num_eye_32 ) * num_light_32;
    if( matrix_size_64 > static_cast<std::uint64_t>( std::numeric_limits<int>::max() )
        || active_light_count == 0 || active_light_count > num_light_32 )
    {
        throw std::runtime_error( "Optimal E snapshot matrix dimensions are invalid" );
    }
    validateConservativeRate( conservative_rate );

    const std::uint64_t array_element_count =
        2 * num_paths_64 + 2 * num_nodes_64 + num_paths_64 + 1
        + num_light_32 + matrix_size_64;
    const std::uint64_t expected_size =
        ( schema_version == 1
            ? SNAPSHOT_V1_HEADER_SIZE
            : SNAPSHOT_V2_HEADER_SIZE )
        + array_element_count * sizeof( std::uint32_t );
    if( std::filesystem::file_size( input_path ) != expected_size )
        throw std::runtime_error( "Optimal E snapshot size disagrees with its header" );

    OptimalEHostSnapshot snapshot{
        static_cast<int>( num_paths_64 ),
        static_cast<int>( num_nodes_64 ),
        static_cast<int>( num_eye_32 ),
        static_cast<int>( num_light_32 ),
        conservative_rate,
        experiment_seed,
        readValues<float>( input, static_cast<std::size_t>( num_paths_64 ) ),
        readValues<float>( input, static_cast<std::size_t>( num_paths_64 ) ),
        readValues<float>( input, static_cast<std::size_t>( num_nodes_64 ) ),
        readValues<int>( input, static_cast<std::size_t>( num_paths_64 + 1 ) ),
        readValues<int>( input, static_cast<std::size_t>( num_nodes_64 ) ),
        readValues<int>( input, num_light_32 ),
        readValues<float>( input, static_cast<std::size_t>( matrix_size_64 ) )
    };

    if( std::accumulate(
            snapshot.active_light.begin(),
            snapshot.active_light.end(),
            0
        ) != static_cast<int>( active_light_count ) )
    {
        throw std::runtime_error( "Optimal E snapshot active-light count is inconsistent" );
    }
    if( snapshot.path_offsets.front() != 0
        || snapshot.path_offsets.back() != snapshot.num_nodes )
    {
        throw std::runtime_error( "Optimal E snapshot path offsets are invalid" );
    }
    for( int path_index = 0; path_index < snapshot.num_paths; ++path_index )
    {
        if( snapshot.path_offsets[path_index]
            > snapshot.path_offsets[path_index + 1] )
        {
            throw std::runtime_error( "Optimal E snapshot path offsets are not monotonic" );
        }
    }
    for( const int index : snapshot.matrix_indices )
    {
        if( index < 0 || index >= static_cast<int>( matrix_size_64 ) )
            throw std::runtime_error( "Optimal E snapshot matrix index is invalid" );
    }
    return snapshot;
}

void normalizeOptimalERows(
    float* base_distribution,
    int num_eye,
    int num_light,
    const int* active_light,
    int active_light_count
)
{
    validateDimensions( num_eye, num_light );
    validateDeviceValues(
        base_distribution,
        num_eye * num_light,
        "Optimal E base distribution"
    );
    if( active_light )
    {
        if( active_light_count <= 0 || active_light_count > num_light )
            throw std::invalid_argument( "Optimal E active-light count is invalid" );
    }
    else
    {
        active_light_count = num_light;
    }
    normalizeRowsKernel<<<
        num_eye,
        CUDA_BLOCK_SIZE,
        CUDA_BLOCK_SIZE * sizeof( double )
    >>>(
        base_distribution,
        num_light,
        active_light,
        active_light_count
    );
    checkCuda( cudaGetLastError(), "Normalize Optimal E rows" );
}

float evaluateOptimalEObjective(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float conservative_rate
)
{
    validateProblem( problem );
    validateConservativeRate( conservative_rate );
    validateDeviceValues(
        base_distribution,
        problem.num_eye * problem.num_light,
        "Optimal E base distribution"
    );
    return evaluateObjective( problem, base_distribution, conservative_rate );
}

void computeOptimalEGradient(
    const OptimalEProblem& problem,
    const float* base_distribution,
    float* gradient,
    float conservative_rate
)
{
    validateProblem( problem );
    validateConservativeRate( conservative_rate );
    validateDeviceValues(
        base_distribution,
        problem.num_eye * problem.num_light,
        "Optimal E base distribution"
    );
    if( !gradient )
        throw std::invalid_argument( "Optimal E gradient output is null" );
    computeGradient(
        problem,
        base_distribution,
        gradient,
        conservative_rate
    );
}

void takeOptimalEMirrorStep(
    float* base_distribution,
    const float* gradient,
    int num_eye,
    int num_light,
    float learning_rate,
    float epsilon,
    const int* active_light,
    int active_light_count
)
{
    validateDimensions( num_eye, num_light );
    if( active_light )
    {
        if( active_light_count <= 0 || active_light_count > num_light )
            throw std::invalid_argument( "Optimal E active-light count is invalid" );
    }
    else
    {
        active_light_count = num_light;
    }
    if( !std::isfinite( learning_rate ) || learning_rate <= 0.0f )
        throw std::invalid_argument( "Optimal E learning rate must be positive" );
    if( !std::isfinite( epsilon ) || epsilon <= 0.0f )
        throw std::invalid_argument( "Optimal E epsilon must be positive" );
    validateDeviceValues(
        base_distribution,
        num_eye * num_light,
        "Optimal E base distribution"
    );
    if( !gradient )
        throw std::invalid_argument( "Optimal E gradient is null" );
    const auto gradient_begin = thrust::device_pointer_cast( gradient );
    if( thrust::count_if(
            gradient_begin,
            gradient_begin + num_eye * num_light,
            InvalidFiniteValue{}
        ) != 0 )
    {
        throw std::invalid_argument(
            "Optimal E gradient contains NaN or infinity"
        );
    }
    takeMirrorStep(
        base_distribution,
        gradient,
        num_eye,
        num_light,
        learning_rate,
        epsilon,
        active_light
    );
}

OptimalEOptimizerResult optimizeOptimalE(
    const OptimalEProblem& problem,
    float* base_distribution,
    const OptimalEOptimizerOptions& options
)
{
    validateProblem( problem );
    validateConservativeRate( options.conservative_rate );
    if( !std::isfinite( options.learning_rate )
        || options.learning_rate <= 0.0f )
    {
        throw std::invalid_argument(
            "Optimal E learning rate must be positive"
        );
    }
    if( options.iterations <= 0 || options.max_backtracking_steps <= 0 )
    {
        throw std::invalid_argument(
            "Optimal E iteration counts must be positive"
        );
    }
    if( !std::isfinite( options.epsilon ) || options.epsilon <= 0.0f )
        throw std::invalid_argument( "Optimal E epsilon must be positive" );

    const auto throwIfCancelled = [&options]
    {
        if( options.should_cancel && options.should_cancel() )
            throw std::runtime_error( "Optimal E optimization was cancelled" );
    };

    throwIfCancelled();
    normalizeOptimalERows(
        base_distribution,
        problem.num_eye,
        problem.num_light,
        problem.active_light,
        problem.active_light
            ? problem.active_light_count
            : problem.num_light
    );
    throwIfCancelled();

    const int matrix_size = problem.num_eye * problem.num_light;
    thrust::device_vector<float> gradient( matrix_size );
    thrust::device_vector<float> previous( matrix_size );
    float current_objective = evaluateObjective(
        problem,
        base_distribution,
        options.conservative_rate
    );
    throwIfCancelled();
    const float initial_objective = current_objective;
    float learning_rate = options.learning_rate;
    int accepted_steps = 0;

    for( int iteration = 0; iteration < options.iterations; ++iteration )
    {
        throwIfCancelled();
        computeGradient(
            problem,
            base_distribution,
            thrust::raw_pointer_cast( gradient.data() ),
            options.conservative_rate
        );
        throwIfCancelled();
        const auto gradient_begin = gradient.begin();
        if( thrust::count_if(
                gradient_begin,
                gradient_begin + matrix_size,
                InvalidFiniteValue{}
            ) != 0 )
        {
            throw std::runtime_error(
                "Optimal E gradient became NaN or infinite"
            );
        }
        thrust::copy(
            thrust::device_pointer_cast( base_distribution ),
            thrust::device_pointer_cast( base_distribution ) + matrix_size,
            previous.begin()
        );

        bool accepted = false;
        float trial_rate = learning_rate;
        for( int attempt = 0;
             attempt < options.max_backtracking_steps;
             ++attempt )
        {
            throwIfCancelled();
            thrust::copy(
                previous.begin(),
                previous.end(),
                thrust::device_pointer_cast( base_distribution )
            );
            takeMirrorStep(
                base_distribution,
                thrust::raw_pointer_cast( gradient.data() ),
                problem.num_eye,
                problem.num_light,
                trial_rate,
                options.epsilon,
                problem.active_light
            );
            const float candidate_objective = evaluateObjective(
                problem,
                base_distribution,
                options.conservative_rate
            );
            throwIfCancelled();
            if( std::isfinite( candidate_objective )
                && candidate_objective <= current_objective )
            {
                current_objective = candidate_objective;
                learning_rate = trial_rate;
                ++accepted_steps;
                accepted = true;
                break;
            }
            trial_rate *= 0.5f;
        }
        if( !accepted )
        {
            thrust::copy(
                previous.begin(),
                previous.end(),
                thrust::device_pointer_cast( base_distribution )
            );
            break;
        }
    }

    return {
        initial_objective,
        current_objective,
        accepted_steps
    };
}

} // namespace spcbpt
