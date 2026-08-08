#include <renderer/RendererWorkflow.h>

#include <renderer/RendererRuntime.h>
#include <renderer/SamplingProgress.h>
#include <spcbptConfig.h>

#include <renderer/core/PG_host.h>
#include <renderer/core/cuda_thrust/device_thrust.h>
#include <renderer/core/decisionTree/classTree_host.h>
#include <renderer/core/scene_shift.h>

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spcbpt
{

namespace
{

struct InteractiveRendererState
{
    float3 eye;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;
    float camera_aspect_ratio;
    bool eye_subspace_visualize;
    bool light_subspace_visualize;
    bool caustic_path_only;
    bool specular_subspace_visualize;
    bool caustic_prob_visualize;
    bool pg_grid_visualize;
    bool error_heat_visual;
    EstimationParams estimation;
};

InteractiveRendererState captureInteractiveState(
    const MyParams& params
)
{
    return {
        params.eye,
        params.U,
        params.V,
        params.W,
        static_cast<float>( params.width ) / params.height,
        params.eye_subspace_visualize,
        params.light_subspace_visualize,
        params.caustic_path_only,
        params.specular_subspace_visualize,
        params.caustic_prob_visualize,
        params.PG_grid_visualize,
        params.error_heat_visual,
        params.estimate_pr
    };
}

void restoreInteractiveState(
    MyParams& params,
    const InteractiveRendererState& state,
    const RendererConfig& config
)
{
    const float next_aspect_ratio =
        static_cast<float>( config.width ) / config.height;
    const float aspect_ratio_scale =
        next_aspect_ratio / state.camera_aspect_ratio;
    params.eye = state.eye;
    params.U = state.camera_u * aspect_ratio_scale;
    params.V = state.camera_v;
    params.W = state.camera_w;
    params.eye_subspace_visualize = state.eye_subspace_visualize;
    params.light_subspace_visualize = state.light_subspace_visualize;
    params.caustic_path_only = config.caustic_path_only;
    params.specular_subspace_visualize =
        state.specular_subspace_visualize;
    params.caustic_prob_visualize = state.caustic_prob_visualize;
    params.PG_grid_visualize = state.pg_grid_visualize;
    params.error_heat_visual = state.error_heat_visual;
    params.estimate_pr = state.estimation;
}

int lightTraceElementCount( const LightTraceParams& params )
{
    return params.num_core * params.core_padding;
}

int preTraceElementCount( const PreTraceParams& params )
{
    return params.num_core * params.padding;
}

template <typename T>
class ScopedCudaAllocation
{
  public:
    explicit ScopedCudaAllocation( size_t count )
    {
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &m_pointer ),
            count * sizeof( T )
        ) );
    }

    ~ScopedCudaAllocation()
    {
        if( m_pointer )
            CUDA_CHECK_NOTHROW( cudaFree( m_pointer ) );
    }

    ScopedCudaAllocation( const ScopedCudaAllocation& ) = delete;
    ScopedCudaAllocation& operator=( const ScopedCudaAllocation& ) = delete;

    T* release()
    {
        T* pointer = m_pointer;
        m_pointer = nullptr;
        return pointer;
    }

  private:
    T* m_pointer = nullptr;
};

class ScopedTexture
{
  public:
    explicit ScopedTexture( spcbpt::Texture texture )
        : m_texture( texture )
    {
    }

    ~ScopedTexture()
    {
        if( m_texture.texture )
            CUDA_CHECK_NOTHROW( cudaDestroyTextureObject( m_texture.texture ) );
        if( m_texture.array )
            CUDA_CHECK_NOTHROW( cudaFreeArray( m_texture.array ) );
    }

    ScopedTexture( const ScopedTexture& ) = delete;
    ScopedTexture& operator=( const ScopedTexture& ) = delete;

    const spcbpt::Texture& get() const { return m_texture; }

    spcbpt::Texture release()
    {
        const spcbpt::Texture texture = m_texture;
        m_texture = {};
        return texture;
    }

  private:
    spcbpt::Texture m_texture;
};

template <typename T>
void freeCudaAllocation( T*& pointer )
{
    if( pointer )
    {
        CUDA_CHECK_NOTHROW( cudaFree( pointer ) );
        pointer = nullptr;
    }
}

} // namespace

void renderConfiguredFrame( RendererRuntime& runtime, uchar4* output )
{
    runtime.renderFrame( output );
}

class RendererWorkflow::Impl
{
  public:
    explicit Impl( RendererRuntime& renderer )
        : runtime( renderer )
        , params( renderer.params() )
        , lt_params( params.lt )
        , pr_params( params.pre_tracer )
        , subspace_info( params.subspace_info )
        , dot_params( params.dot_params )
        , train_vector(
              dropOut_tracing::max_u,
              std::vector<std::vector<std::vector<float2>>>(
                  dropOut_tracing::default_specularSubSpaceNumber,
                  std::vector<std::vector<float2>>(
                      dropOut_tracing::default_surfaceSubSpaceNumber,
                      std::vector<float2>()
                  )
              )
          )
        , train_finish(
              dropOut_tracing::max_u,
              std::vector<std::vector<bool>>(
                  dropOut_tracing::default_specularSubSpaceNumber,
                  std::vector<bool>( dropOut_tracing::default_surfaceSubSpaceNumber, false )
              )
          )
        , scene_generation( renderer.sceneGeneration() )
    {
    }

    void initializeAlgorithmState()
    {
        if( !runtime.isInitialized() )
            throw std::logic_error( "RendererWorkflow requires an initialized RendererRuntime" );
        if( algorithm_state_initialized )
            throw std::logic_error( "RendererWorkflow algorithm state is already initialized" );

        try
        {
            dropOutTracingParamsInit();
            ltParamsSetup();
            preTracerParamsSetup();
            envParamsSetup();
            algorithm_state_initialized = true;
        }
        catch( ... )
        {
            rollbackAlgorithmState();
            throw;
        }
    }

    void runPreprocessing( std::function<bool()> should_cancel )
    {
        if( !algorithm_state_initialized )
            throw std::logic_error( "RendererWorkflow::initializeAlgorithmState must be called first" );
        if( preprocessing_complete )
            throw std::logic_error( "RendererWorkflow preprocessing has already completed" );

        cancellation_check = std::move( should_cancel );
        try
        {
            throwIfCancelled();
            pathGuidingParamsSetup();
            throwIfCancelled();
            dropOutTracingParamsSetup();
            throwIfCancelled();
            preprocessing();
            throwIfCancelled();
            preprocessing_complete = true;
            cancellation_check = {};
        }
        catch( ... )
        {
            cancellation_check = {};
            throw;
        }
    }

    void captureOptimalEProblem( const std::string& output_path )
    {
        if( !algorithm_state_initialized )
            throw std::logic_error( "RendererWorkflow::initializeAlgorithmState must be called first" );
        if( output_path.empty() )
            throw std::invalid_argument( "Optimal E capture path is empty" );

        pathGuidingParamsSetup();
        dropOutTracingParamsSetup();
        preprocessing( &output_path );
    }

    void renderFrame( uchar4* output )
    {
        const RendererAlgorithm algorithm = runtime.config().algorithm;
        if( isAdvancedRendererAlgorithm( algorithm ) )
        {
            if( !preprocessing_complete )
                throw std::logic_error( "Advanced LVCBPT rendering requires preprocessing" );
            launchLVCTrace();
            updateDropOutTracingParams();
            updateDropOutTracingCombineWeight();
        }
        renderConfiguredFrame( runtime, output );
    }

    std::uint64_t sceneGeneration() const { return scene_generation; }

  private:
    static constexpr int TRAIN_CAPACITY = 30000;

    void throwIfCancelled() const
    {
        if( cancellation_check && cancellation_check() )
            throw std::runtime_error( "Renderer preprocessing was cancelled" );
    }

    std::vector<int> surroundsIndex( int index, const envInfo& infos ) const
    {
        const int2 coord = infos.index2coord( index );
        std::vector<int2> surrounding_coords;
        for( int dx = -2; dx <= 2; ++dx )
        {
            for( int dy = -2; dy <= 2; ++dy )
            {
                if( std::abs( dx ) + std::abs( dy ) <= 2 )
                    surrounding_coords.push_back( coord + make_int2( dx, dy ) );
            }
        }

        std::vector<int> indices;
        for( const int2 surrounding_coord : surrounding_coords )
        {
            if( surrounding_coord.x >= 0
                && surrounding_coord.y >= 0
                && surrounding_coord.x < infos.width
                && surrounding_coord.y < infos.height )
            {
                indices.push_back( infos.coord2index( surrounding_coord ) );
            }
        }
        return indices;
    }

    thrust::device_ptr<float> envMapCMFBuild( float4* luminance, int size, const envInfo& infos )
    {
        std::vector<float> cmf( size );
        constexpr float uniform_rate = 0.25f;
        const float uniform_pdf = 1.0f / size;
        for( int i = 0; i < size; ++i )
        {
            const std::vector<int> surrounding_indices = surroundsIndex( i, infos );
            cmf[i] = float3weight( make_float3( luminance[i] ) );
            for( const int surrounding_index : surrounding_indices )
            {
                cmf[i] += float3weight( make_float3( luminance[surrounding_index] ) )
                          / surrounding_indices.size();
            }
            if( i >= 1 )
                cmf[i] += cmf[i - 1];
        }

        const float sum = cmf[size - 1];
        for( int i = 0; i < size; ++i )
        {
            cmf[i] /= sum;
            cmf[i] = cmf[i] * ( 1.0f - uniform_rate )
                     + uniform_pdf * ( i + 1 ) * uniform_rate;
        }
        return MyThrustOp::envMapCMFBuild( cmf.data(), size );
    }

    void envParamsSetup()
    {
        sutil::Scene& scene = runtime.scene();
        if( scene.getEnvFilePath().empty() )
        {
            params.sky.valid = false;
            return;
        }

        std::printf(
            "load and build sampling cmf from file %s\n",
            scene.getEnvFilePath().c_str()
        );
        const std::filesystem::path env_path =
            std::filesystem::path( scene.getResourceRoot() ) / scene.getEnvFilePath();
        HDRLoader hdr_env( env_path.lexically_normal().string() );

        const float3 default_color = make_float3( 1.0f );
        envInfo next_sky = {};
        next_sky.height   = hdr_env.height();
        next_sky.width    = hdr_env.width();
        next_sky.divLevel = std::sqrt( 0.5f * NUM_SUBSPACE_LIGHTSOURCE );
        next_sky.ssBase   = 0;
        next_sky.size     = hdr_env.height() * hdr_env.width();

        float4* hdr_raster = reinterpret_cast<float4*>( hdr_env.raster() );
        for( int i = 0; i < scene.dir_lights.size(); ++i )
        {
            const auto& dir_light = scene.dir_lights[i];
            float3 dir = dir_light.first;
            dir.y = -dir.y;
            const float2 uv = dir2uv( -dir );
            const auto coord = next_sky.uv2coord( uv );
            const int index = next_sky.coord2index( coord );
            hdr_raster[index] += make_float4(
                dir_light.second * next_sky.size / ( 4 * M_PI ),
                0.0f
            );
            std::printf(
                "Add directional light %f %f %f in index %d\n",
                dir_light.first.x,
                dir_light.first.y,
                dir_light.first.z,
                index
            );
        }

        ScopedTexture env_texture(
            hdr_env.loadTexture( default_color, nullptr )
        );
        next_sky.tex = env_texture.get().texture;
        next_sky.cmf = thrust::raw_pointer_cast(
            envMapCMFBuild( hdr_raster, hdr_env.height() * hdr_env.width(), next_sky )
        );
        next_sky.center   = scene.aabb().center();
        next_sky.r        = length( scene.aabb().m_min - scene.aabb().m_max );
        next_sky.valid    = true;
        next_sky.light_id = params.lights.count - 1;

        scene.adoptTexture( env_texture.get() );
        env_texture.release();
        params.sky = next_sky;
    }

    void rollbackAlgorithmState()
    {
        freeCudaAllocation( lt_params.ans );
        freeCudaAllocation( lt_params.validState );
        freeCudaAllocation( lt_params.lightImage );
        freeCudaAllocation( lt_params.lightIndex );
        freeCudaAllocation( lt_params.lightBuffer );
        freeCudaAllocation( lt_params.rand_state );
        freeCudaAllocation( pr_params.paths );
        freeCudaAllocation( pr_params.conns );
        lt_params = {};
        pr_params = {};
        params.sky = {};
        MyThrustOp::invalidate_scene_caches();
        algorithm_state_initialized = false;
        preprocessing_complete = false;
    }

    void ltParamsSetup()
    {
        lt_params.M_per_core   = 10;
        lt_params.core_padding = 800;
        lt_params.num_core     = 1000;
        lt_params.M            = lt_params.M_per_core * lt_params.num_core;
        lt_params.launch_frame = 0;

        const size_t light_trace_count =
            static_cast<size_t>( lightTraceElementCount( lt_params ) );
        ScopedCudaAllocation<BDPTVertex> lvc( light_trace_count );
        ScopedCudaAllocation<bool> valid( light_trace_count );
        ScopedCudaAllocation<float3> image(
            static_cast<size_t>( params.width ) * params.height
        );
        ScopedCudaAllocation<int> indices( light_trace_count );
        ScopedCudaAllocation<float3> buffer( light_trace_count );
        ScopedCudaAllocation<curandState> random_states( light_trace_count );

        lt_params.ans         = lvc.release();
        lt_params.validState  = valid.release();
        lt_params.lightImage  = image.release();
        lt_params.lightBuffer = buffer.release();
        lt_params.lightIndex  = indices.release();
        lt_params.rand_state  = random_states.release();
    }

    void setLightImage()
    {
        std::vector<float3> light_buffer( lightTraceElementCount( params.lt ) );
        std::vector<int> indices( lightTraceElementCount( params.lt ) );
        std::vector<float3> light_image( params.width * params.height );
        std::unique_ptr<bool[]> valid(
            new bool[lightTraceElementCount( params.lt )]
        );

        CUDA_CHECK( cudaMemcpy(
            light_buffer.data(),
            params.lt.lightBuffer,
            lightTraceElementCount( params.lt ) * sizeof( float3 ),
            cudaMemcpyDeviceToHost
        ) );
        CUDA_CHECK( cudaMemcpy(
            indices.data(),
            params.lt.lightIndex,
            lightTraceElementCount( params.lt ) * sizeof( int ),
            cudaMemcpyDeviceToHost
        ) );
        CUDA_CHECK( cudaMemcpy(
            valid.get(),
            params.lt.validState,
            lightTraceElementCount( params.lt ) * sizeof( bool ),
            cudaMemcpyDeviceToHost
        ) );

        for( int i = 0; i < lightTraceElementCount( params.lt ); ++i )
        {
            if( !valid[i] )
                continue;
            const int id = indices[i];
            if( id >= 0 )
            {
                const float intensity =
                    light_buffer[i].x + light_buffer[i].y + light_buffer[i].z;
                if( !std::isnan( intensity ) && std::isfinite( intensity ) )
                    light_image[id] += light_buffer[i];
            }
        }
        for( float3& pixel : light_image )
            pixel /= params.lt.M;

        CUDA_CHECK( cudaMemcpy(
            params.lt.lightImage,
            light_image.data(),
            params.width * params.height * sizeof( float3 ),
            cudaMemcpyHostToDevice
        ) );
    }

    void preTracerParamsSetup()
    {
        pr_params.num_core  = 10000;
        pr_params.padding   = 10;
        pr_params.iteration = 0;

        ScopedCudaAllocation<preTracePath> paths(
            static_cast<size_t>( pr_params.num_core )
        );
        ScopedCudaAllocation<preTraceConnection> connections(
            static_cast<size_t>( preTraceElementCount( pr_params ) )
        );
        pr_params.paths   = paths.release();
        pr_params.conns   = connections.release();
        pr_params.PG_mode = false;
    }

    void launchLightTrace()
    {
        throwIfCancelled();
        ++lt_params.launch_frame;
        runtime.uploadParams();

        sutil::Scene& scene = runtime.scene();
        scene.switchRaygen( "light trace" );
        OPTIX_CHECK( optixLaunch(
            scene.pipeline(),
            nullptr,
            reinterpret_cast<CUdeviceptr>( runtime.deviceParams() ),
            sizeof( MyParams ),
            scene.sbt(),
            lt_params.num_core,
            1,
            1
        ) );
        runtime.synchronize();
        throwIfCancelled();
    }

    void launchLVCTrace()
    {
        throwIfCancelled();
        if( !params.spcbpt_pure )
            dot_params.discard_ratio = dot_params.discard_ratio_next;

        launchLightTrace();
        const auto vertices = thrust::device_pointer_cast( params.lt.ans );
        const auto valid = thrust::device_pointer_cast( params.lt.validState );
        SubspaceSampler sampler =
            MyThrustOp::LVC_Process(
                vertices,
                valid,
                lightTraceElementCount( params.lt )
            );
        setLightImage();
        params.sampler = sampler;
        if( optimal_gamma )
        {
            subspace_info.CMFGamma = thrust::raw_pointer_cast(
                MyThrustOp::Gamma2CMFGamma(
                    optimal_gamma,
                    params.sampler.subspace
                )
            );
        }

        if( !params.spcbpt_pure )
        {
            sampler = MyThrustOp::LVC_Process_glossyOnly(
                vertices,
                valid,
                lightTraceElementCount( params.lt ),
                params.materials
            );
            params.sampler.glossy_count         = sampler.glossy_count;
            params.sampler.glossy_index         = sampler.glossy_index;
            params.sampler.glossy_subspace_bias = sampler.glossy_subspace_bias;
            params.sampler.glossy_subspace_num  = sampler.glossy_subspace_num;
        }
        throwIfCancelled();
    }

    int launchPretrace()
    {
        throwIfCancelled();
        ++pr_params.iteration;
        runtime.uploadParams();

        sutil::Scene& scene = runtime.scene();
        scene.switchRaygen( "pretrace" );
        OPTIX_CHECK( optixLaunch(
            scene.pipeline(),
            nullptr,
            reinterpret_cast<CUdeviceptr>( runtime.deviceParams() ),
            sizeof( MyParams ),
            scene.sbt(),
            pr_params.num_core,
            1,
            1
        ) );
        runtime.synchronize();
        throwIfCancelled();

        const int valid_samples = MyThrustOp::valid_sample_gather(
            thrust::device_pointer_cast( pr_params.paths ),
            pr_params.num_core,
            thrust::device_pointer_cast( pr_params.conns ),
            preTraceElementCount( pr_params )
        );
        throwIfCancelled();
        return valid_samples;
    }

    void pathGuidingParamsSetup()
    {
        throwIfCancelled();
        constexpr int pg_training_data_batch = 10;
        constexpr int pg_training_data_online_batch = 0;
        constexpr int batch_sample_count = 1000000;

        if( !runtime.config().path_guiding_enabled )
        {
            params.pg_params.pg_enable = 0;
            return;
        }

        std::vector<path_guiding::PG_training_mat> training_materials;
        int build_iteration_max = 12;
        if( runtime.config().path_guiding_self_train )
        {
            constexpr int initial_path = 1000;
            int split_limit = initial_path;
            int target_path = initial_path;
            if( runtime.config().path_guiding_more_training )
            {
                build_iteration_max += 4;
                split_limit *= 2;
            }

            pg_trainer.init( runtime.scene().aabb() );
            for( int i = 0; i < build_iteration_max; ++i )
            {
                MyThrustOp::clear_training_set();
                int current_sample_count = 0;
                int accumulated_sample_count = 0;
                int accumulated_iterations = 0;
                detail::SamplingProgressGuard progress_guard;
                while( current_sample_count + accumulated_sample_count < target_path )
                {
                    const int added_samples = launchPretrace();
                    progress_guard.record(
                        added_samples,
                        "Path-guiding self-training"
                    );
                    current_sample_count += added_samples;
                    ++accumulated_iterations;
                    if( current_sample_count > batch_sample_count )
                    {
                        accumulated_sample_count += current_sample_count;
                        current_sample_count = 0;
                        std::vector<path_guiding::PG_training_mat> new_materials =
                            MyThrustOp::get_data_for_path_guiding( -1, pr_params.PG_mode );
                        training_materials.insert(
                            training_materials.end(),
                            new_materials.begin(),
                            new_materials.end()
                        );
                        MyThrustOp::clear_training_set();
                    }
                }

                std::vector<path_guiding::PG_training_mat> new_materials =
                    MyThrustOp::get_data_for_path_guiding( -1, pr_params.PG_mode );
                training_materials.insert(
                    training_materials.end(),
                    new_materials.begin(),
                    new_materials.end()
                );
                MyThrustOp::clear_training_set();

                std::printf(
                    "get %zu samples for pg building iteration %d; target path%d; "
                    "split-limit %d; average nodes %f \n",
                    training_materials.size(),
                    i,
                    target_path,
                    split_limit,
                    static_cast<float>( training_materials.size() ) / accumulated_iterations
                );
                pg_trainer.set_training_set( training_materials );
                pg_trainer.build_tree(
                    split_limit,
                    static_cast<int>( training_materials.size() )
                );
                throwIfCancelled();
                params.pg_params.spatio_trees = MyThrustOp::spatio_tree_to_device(
                    pg_trainer.s_tree.nodes.data(),
                    static_cast<int>( pg_trainer.s_tree.nodes.size() )
                );
                params.pg_params.quad_trees = MyThrustOp::quad_tree_to_device(
                    pg_trainer.q_tree_group.nodes.data(),
                    static_cast<int>( pg_trainer.q_tree_group.nodes.size() )
                );
                params.pg_params.pg_enable   = 1;
                params.pg_params.epsilon_lum = 0.001f;
                params.pg_params.guide_ratio = 0.5f;

                target_path *= 2;
                split_limit = static_cast<int>( split_limit * std::sqrt( 2.0f ) );
                training_materials.clear();
            }
        }
        else
        {
            for( int i = 0; i < pg_training_data_batch; ++i )
            {
                MyThrustOp::clear_training_set();
                int current_sample_count = 0;
                detail::SamplingProgressGuard progress_guard;
                while( current_sample_count < batch_sample_count )
                {
                    const int added_samples = launchPretrace();
                    progress_guard.record(
                        added_samples,
                        "Path-guiding training"
                    );
                    current_sample_count += added_samples;
                    std::printf(
                        "regenerate data for pg %d %zu\n",
                        current_sample_count,
                        training_materials.size()
                    );
                }
                std::vector<path_guiding::PG_training_mat> new_materials =
                    MyThrustOp::get_data_for_path_guiding( -1, pr_params.PG_mode );
                training_materials.insert(
                    training_materials.end(),
                    new_materials.begin(),
                    new_materials.end()
                );
                MyThrustOp::clear_training_set();
            }

            std::printf( "get mats size %zu\n", training_materials.size() );
            pg_trainer.set_training_set( training_materials );
            pg_trainer.init( runtime.scene().aabb() );
            for( int i = 0; i < build_iteration_max; ++i )
            {
                pg_trainer.build_tree();
                throwIfCancelled();
            }

            for( int i = 0; i < pg_training_data_online_batch; ++i )
            {
                training_materials.clear();
                MyThrustOp::clear_training_set();
                int current_sample_count = 0;
                detail::SamplingProgressGuard progress_guard;
                while( current_sample_count < batch_sample_count )
                {
                    const int added_samples = launchPretrace();
                    progress_guard.record(
                        added_samples,
                        "Path-guiding online training"
                    );
                    current_sample_count += added_samples;
                }
                std::printf( "online training for pg batch %d \n", i );
                pg_trainer.set_training_set( MyThrustOp::get_data_for_path_guiding() );
                pg_trainer.online_training();
            }
        }

        pg_trainer.mats_cache.clear();
        pg_trainer.mats.clear();
        training_materials.clear();
        params.pg_params.spatio_trees = MyThrustOp::spatio_tree_to_device(
            pg_trainer.s_tree.nodes.data(),
            static_cast<int>( pg_trainer.s_tree.nodes.size() )
        );
        params.pg_params.quad_trees = MyThrustOp::quad_tree_to_device(
            pg_trainer.q_tree_group.nodes.data(),
            static_cast<int>( pg_trainer.q_tree_group.nodes.size() )
        );
        params.pg_params.pg_enable   = 1;
        params.pg_params.epsilon_lum = 0.001f;
        params.pg_params.guide_ratio = 0.5f;
        pr_params.PG_mode = false;
        throwIfCancelled();
    }

    void dropOutTracingParamsInit()
    {
        dot_params.is_init          = false;
        dot_params.specularSubSpace = nullptr;
        dot_params.surfaceSubSpace  = nullptr;
        dot_params.record_buffer    = nullptr;
    }

    void dropOutTracingParamsSetup()
    {
        throwIfCancelled();
        if( params.spcbpt_pure )
            return;

        dot_params.pixel_dirty = true;
        dot_params.discard_ratio =
            dropOut_tracing::light_subpath_caustic_discard_ratio;
        dot_params.discard_ratio_next =
            dropOut_tracing::light_subpath_caustic_discard_ratio;
        dot_params.specularSubSpaceNumber =
            dropOut_tracing::default_specularSubSpaceNumber;
        dot_params.surfaceSubSpaceNumber =
            dropOut_tracing::default_surfaceSubSpaceNumber;
        dot_params.data.on_GPU = false;

        MyThrustOp::clear_training_set();
        constexpr int target_sample_count = 100000;
        int current_sample_count = 0;
        detail::SamplingProgressGuard progress_guard;
        while( current_sample_count < target_sample_count )
        {
            const int added_samples = launchPretrace();
            progress_guard.record(
                added_samples,
                "Proxy preprocessing"
            );
            current_sample_count += added_samples;
        }

        std::vector<classTree::divide_weight> unlabeled_samples =
            MyThrustOp::getCausticCentroidCandidate( false, 100000 );
        classTree::tree specular_subspace = classTree::buildTreeBaseOnExistSample()(
            unlabeled_samples,
            dot_params.specularSubSpaceNumber - 1,
            1
        );
        throwIfCancelled();
        dot_params.specularSubSpace = MyThrustOp::DOT_specular_tree_to_device(
            specular_subspace.v,
            specular_subspace.size
        );

        unlabeled_samples =
            MyThrustOp::get_weighted_point_for_tree_building( false, 10000 );
        classTree::tree normal_surface_subspace = classTree::buildTreeBaseOnExistSample()(
            unlabeled_samples,
            dot_params.surfaceSubSpaceNumber - 1,
            1
        );
        throwIfCancelled();
        dot_params.surfaceSubSpace = MyThrustOp::DOT_surface_tree_to_device(
            normal_surface_subspace.v,
            normal_surface_subspace.size
        );

        dot_params.data.size =
            dot_params.specularSubSpaceNumber
            * dot_params.surfaceSubSpaceNumber
            * dropOut_tracing::slot_number
            * static_cast<int>( dropOut_tracing::DropOutType::DropOutTypeNumber );
        thrust::host_vector<dropOut_tracing::statistics_data_struct> statistics(
            dot_params.data.size
        );
        thrust::fill(
            statistics.begin(),
            statistics.end(),
            dropOut_tracing::statistics_data_struct()
        );
        dot_params.data.host_data = statistics.data();
        dot_params.data.device_data =
            MyThrustOp::DOT_statistics_data_to_device(
                dot_params.data.host_data,
                dot_params.data.size
            );

        thrust::host_vector<dropOut_tracing::PGParams> pg_data(
            dot_params.specularSubSpaceNumber
            * dot_params.surfaceSubSpaceNumber
            * dropOut_tracing::max_u
        );
        dot_params.data.device_PGParams = MyThrustOp::DOT_PG_data_to_device( pg_data );
        dot_params.data.on_GPU = true;

        dot_params.record_buffer_core = lt_params.num_core;
        dot_params.record_buffer_padding =
            lt_params.core_padding * dropOut_tracing::record_buffer_width;
        dot_params.record_buffer = MyThrustOp::DOT_get_statistic_record_buffer(
            dot_params.record_buffer_core * dot_params.record_buffer_padding
        );
        dot_params.statistics_iteration_count = 0;

        if( dot_params.pixel_dirty )
        {
            thrust::host_vector<float> fractions(
                params.width * params.height,
                0.5f
            );
            dot_params.pixel_caustic_refract =
                MyThrustOp::DOT_causticFrac_to_device( fractions );
            dot_params.pixel_record =
                MyThrustOp::DOT_set_pixelRecords_size( params.width * params.height );
            dot_params.pixel_dirty = false;
        }
        dot_params.is_init = true;
        dot_params.selection_const = 0.0f;
        throwIfCancelled();
    }

    void initializeCombineWeightState()
    {
        const size_t pixel_count =
            static_cast<size_t>( params.width ) * params.height;
        combine_fractions.resize( pixel_count );
        thrust::fill( combine_fractions.begin(), combine_fractions.end(), 0.5f );
        normal_weight.assign( pixel_count, 0.0f );
        normal_count.assign( pixel_count, 0 );
        caustic_weight.assign( pixel_count, 0.0f );
        caustic_count.assign( pixel_count, 0 );

        const size_t gamma_count_size =
            dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE;
        caustic_gamma.resize( gamma_count_size );
        thrust::fill(
            caustic_gamma.begin(),
            caustic_gamma.end(),
            1.0f / dropOut_tracing::default_specularSubSpaceNumber
        );
        gamma_non_normalized.assign( gamma_count_size, 0.000001f );
        gamma_non_normalized_single.assign( gamma_count_size, 0.000001f );
        gamma_sample_count.assign( gamma_count_size, 0 );
        combine_state_initialized = true;
    }

    void updateDropOutTracingCombineWeight()
    {
        if( params.spcbpt_pure )
            return;
        if( !combine_state_initialized )
            initializeCombineWeightState();

        if( dot_params.pixel_dirty )
        {
            const size_t pixel_count =
                static_cast<size_t>( params.width ) * params.height;
            combine_fractions.resize( pixel_count );
            thrust::fill(
                combine_fractions.begin(),
                combine_fractions.end(),
                0.5f
            );
            normal_weight.assign( pixel_count, 0.0f );
            normal_count.assign( pixel_count, 0 );
            caustic_weight.assign( pixel_count, 0.0f );
            caustic_count.assign( pixel_count, 0 );

            dot_params.pixel_caustic_refract =
                MyThrustOp::DOT_causticFrac_to_device( combine_fractions );
            dot_params.pixel_record =
                MyThrustOp::DOT_set_pixelRecords_size( params.width * params.height );
            dot_params.pixel_dirty = false;
            return;
        }
        if( combine_train_iteration > 0
            && combine_train_iteration > dropOut_tracing::iteration_stop_learning )
        {
            return;
        }
        ++combine_train_iteration;

        const thrust::host_vector<dropOut_tracing::pixelRecord> records =
            MyThrustOp::DOT_get_pixelRecords();
        for( const dropOut_tracing::pixelRecord& record : records )
        {
            if( !record.valid() || !record.is_caustic() )
                continue;
            if( std::isnan( record.record ) || std::isinf( record.record ) )
                continue;

            float weight = std::abs( record.record )
                           * caustic_gamma[
                               record.eyeId
                                   * dropOut_tracing::default_specularSubSpaceNumber
                               + record.specularId
                           ];
            if( weight > 1000000.0f )
                weight = 1000000.0f;
            const unsigned int id =
                record.eyeId * dropOut_tracing::default_specularSubSpaceNumber
                + record.specularId;
            ++gamma_sample_count[id];
            gamma_non_normalized[id] += weight * weight;
            gamma_non_normalized_single[id] = lerp(
                gamma_non_normalized_single[id],
                weight * weight,
                1.0f / gamma_sample_count[id]
            );
        }

        std::vector<float> gamma_sum( NUM_SUBSPACE, 0.0f );
        for( int i = 0; i < NUM_SUBSPACE; ++i )
        {
            for( int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; ++j )
            {
                const unsigned int id =
                    j + i * dropOut_tracing::default_specularSubSpaceNumber;
                gamma_sum[i] += std::sqrt( gamma_non_normalized_single[id] );
            }
        }

        for( int i = 0; i < NUM_SUBSPACE; ++i )
        {
            for( int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; ++j )
            {
                const unsigned int id =
                    j + i * dropOut_tracing::default_specularSubSpaceNumber;
                caustic_gamma[id] =
                    std::sqrt( gamma_non_normalized_single[id] ) / gamma_sum[i]
                        * ( 1.0f - CONSERVATIVE_RATE )
                    + 1.0f / dropOut_tracing::default_specularSubSpaceNumber
                        * CONSERVATIVE_RATE;
            }
        }

        for( int i = 0; i < records.size(); ++i )
        {
            const dropOut_tracing::pixelRecord& record = records[i];
            if( !record.valid() || record.record < 0 )
                continue;
            const uint2 pixel_label =
                dot_params.Id2pixel( i, make_uint2( params.width, params.height ) );
            const int final_label = dot_params.pixel2unitId(
                pixel_label,
                make_uint2( params.width, params.height )
            );
            if( !record.is_caustic() )
            {
                ++normal_count[final_label];
                normal_weight[final_label] += record.record;
            }
            else
            {
                ++caustic_count[final_label];
                caustic_weight[final_label] += record.record;
            }
        }

        for( int i = 0; i < combine_fractions.size(); ++i )
        {
            if( caustic_count[i] + normal_count[i] == 0 )
            {
                combine_fractions[i] = 0.5f;
            }
            else
            {
                const float recommendation =
                    caustic_weight[i] / ( normal_weight[i] + caustic_weight[i] );
                combine_fractions[i] =
                    recommendation < 0.05f ? CONSERVATIVE_RATE : 1.0f;
            }
        }

        subspace_info.CMFCausticGamma =
            MyThrustOp::DOT_causticCMFGamma_to_device( caustic_gamma );
        dot_params.CMF_Gamma = subspace_info.CMFCausticGamma;
        dot_params.pixel_caustic_refract =
            MyThrustOp::DOT_causticFrac_to_device( combine_fractions );
    }

    void initializeDropoutUpdateState()
    {
        dropout_average_count.assign(
            dropOut_tracing::max_u,
            std::vector<std::vector<int>>(
                dot_params.specularSubSpaceNumber,
                std::vector<int>( dot_params.surfaceSubSpaceNumber, 0 )
            )
        );
        dropout_average.assign(
            dropOut_tracing::max_u,
            std::vector<std::vector<float>>(
                dot_params.specularSubSpaceNumber,
                std::vector<float>( dot_params.surfaceSubSpaceNumber, 0.0f )
            )
        );
        dropout_variance.assign(
            dropOut_tracing::max_u,
            std::vector<std::vector<float>>(
                dot_params.specularSubSpaceNumber,
                std::vector<float>( dot_params.surfaceSubSpaceNumber, 0.0f )
            )
        );
        dropout_update_state_initialized = true;
    }

    void updateDropOutTracingParams()
    {
        if( params.spcbpt_pure )
            return;
        if( dropout_train_iteration > 0
            && dropout_train_iteration > dropOut_tracing::iteration_stop_learning )
        {
            if( dropout_train_iteration
                == dropOut_tracing::iteration_stop_learning + 1 )
            {
                ++dropout_train_iteration;
                std::printf( "iteration more than stop point, stop params learning\n" );
            }
            return;
        }
        ++dropout_train_iteration;
        if( !dropout_update_state_initialized )
            initializeDropoutUpdateState();

        const bool disable_print = !DOT_DEBUG_INFO_ENABLE;
        thrust::host_vector<dropOut_tracing::statistics_data_struct> statistics =
            MyThrustOp::DOT_statistics_data_to_host();
        thrust::host_vector<dropOut_tracing::PGParams> pg_data =
            MyThrustOp::DOT_PG_data_to_host();
        dot_params.data.host_data = statistics.data();
        dot_params.data.host_PGParams = pg_data.data();
        dot_params.data.on_GPU = false;
        const thrust::host_vector<dropOut_tracing::statistic_record> records =
            MyThrustOp::DOT_get_host_statistic_record_buffer();

        for( const dropOut_tracing::statistic_record& record : records )
        {
            if( record.data_slot != DOT_usage::Average )
                continue;
            float& average =
                dropout_average[static_cast<int>( record.type )]
                               [record.specular_subspaceId]
                               [record.surface_subspaceId];
            float& variance =
                dropout_variance[static_cast<int>( record.type )]
                                [record.specular_subspaceId]
                                [record.surface_subspaceId];
            int& count =
                dropout_average_count[static_cast<int>( record.type )]
                                     [record.specular_subspaceId]
                                     [record.surface_subspaceId];
            average = lerp( average, static_cast<float>( record ), 1.0f / ( count + 1 ) );
            variance = lerp(
                variance,
                static_cast<float>( record ) * static_cast<float>( record ),
                1.0f / ( count + 1 )
            );
            ++count;
        }

        for( int i = 0; i < dropOut_tracing::max_u; ++i )
        {
            for( int j = 0; j < dot_params.specularSubSpaceNumber; ++j )
            {
                for( int k = 0; k < dot_params.surfaceSubSpaceNumber; ++k )
                {
                    dropOut_tracing::statistics_data_struct& statistic =
                        dot_params.get_statistic_data(
                            static_cast<dropOut_tracing::DropOutType>( i ),
                            j,
                            k
                        );
                    if( statistic.valid )
                    {
                        statistic.average = dropout_average[i][j][k];
                        statistic.variance = dropout_variance[i][j][k];
                        if( !disable_print )
                        {
                            std::printf(
                                "Average reciprocal PDF for ID S:%d C:%d U:%d is %f , "
                                "sqrt variance %f\n",
                                j,
                                k,
                                i,
                                statistic.average,
                                std::sqrt( statistic.variance )
                            );
                        }
                    }
                    if( dropout_average_count[i][j][k] != 0 )
                        statistic.valid = true;
                }
            }
        }

        std::vector<std::vector<std::vector<float>>> bounds(
            dropOut_tracing::max_u,
            std::vector<std::vector<float>>(
                dot_params.specularSubSpaceNumber,
                std::vector<float>( dot_params.surfaceSubSpaceNumber, 0.0f )
            )
        );
        for( const dropOut_tracing::statistic_record& record : records )
        {
            if( record.data_slot != DOT_usage::Bound || std::isinf( record ) )
                continue;
            float& bound =
                bounds[static_cast<int>( record.type )]
                      [record.specular_subspaceId]
                      [record.surface_subspaceId];
            bound = std::max( static_cast<float>( record ), bound );
        }
        for( int i = 0; i < dropOut_tracing::max_u; ++i )
        {
            for( int j = 0; j < dot_params.specularSubSpaceNumber; ++j )
            {
                for( int k = 0; k < dot_params.surfaceSubSpaceNumber; ++k )
                {
                    dropOut_tracing::statistics_data_struct& statistic =
                        dot_params.get_statistic_data(
                            static_cast<dropOut_tracing::DropOutType>( i ),
                            j,
                            k
                        );
                    statistic.bound = std::max( statistic.bound, bounds[i][j][k] );
                    if( statistic.valid && !disable_print )
                    {
                        std::printf(
                            "Bound Setting for ID S:%d C:%d U:%d is %f\n",
                            j,
                            k,
                            i,
                            statistic.bound
                        );
                    }
                }
            }
        }

        if( dropOut_tracing::PG_reciprocal_estimation_enable )
        {
            int count = 0;
            for( const dropOut_tracing::statistic_record& record : records )
            {
                if( record.data_slot != DOT_usage::Dirction )
                    continue;
                ++count;
                std::vector<float2>& training_samples =
                    train_vector[static_cast<int>( record.type )]
                                [record.specular_subspaceId]
                                [record.surface_subspaceId];
                if( training_samples.size() != TRAIN_CAPACITY )
                    training_samples.push_back( make_float2( record.data, record.data2 ) );
            }

            for( int i = 0; i < dropOut_tracing::max_u; ++i )
            {
                for( int j = 0; j < dot_params.specularSubSpaceNumber; ++j )
                {
                    for( int k = 0; k < dot_params.surfaceSubSpaceNumber; ++k )
                    {
                        const int sample_count =
                            static_cast<int>( train_vector[i][j][k].size() );
                        if( sample_count == 0 )
                            continue;
                        if( sample_count == TRAIN_CAPACITY && train_finish[i][j][k] )
                            continue;
                        if( sample_count == TRAIN_CAPACITY )
                        {
                            train_finish[i][j][k] = true;
                            std::printf( "S:%d C:%d U:%d train end!!!\n", j, k, i );
                            dot_params.get_PGParams_pointer(
                                static_cast<dropOut_tracing::DropOutType>( i ),
                                j,
                                k
                            )->trainEnd = 1;
                        }
                        dot_params.get_PGParams_pointer(
                            static_cast<dropOut_tracing::DropOutType>( i ),
                            j,
                            k
                        )->loadIn( train_vector[i][j][k] );
                        if( !disable_print )
                        {
                            std::printf(
                                "PG traning for ID S:%d C:%d U:%d with size %d\n",
                                j,
                                k,
                                i,
                                sample_count
                            );
                        }
                    }
                }
            }
            std::printf( "we get %d record success\n", count );
        }

        ++dot_params.statistics_iteration_count;
        std::printf(
            "received %zu valid records in the Light Tracing\n",
            records.size()
        );

        const float total_blocks =
            static_cast<float>(
                dropOut_tracing::max_u
                * dot_params.specularSubSpaceNumber
                * dot_params.surfaceSubSpaceNumber
            );
        int bad_count = 0;
        for( int i = 0; i < dropOut_tracing::max_u; ++i )
        {
            for( int j = 0; j < dot_params.specularSubSpaceNumber; ++j )
            {
                for( int k = 0; k < dot_params.surfaceSubSpaceNumber; ++k )
                {
                    const dropOut_tracing::statistics_data_struct& statistic =
                        dot_params.get_statistic_data(
                            static_cast<dropOut_tracing::DropOutType>( i ),
                            j,
                            k
                        );
                    if( std::isnan( statistic.average )
                        || std::isinf( statistic.average )
                        || std::isnan( statistic.bound )
                        || std::isinf( statistic.bound ) )
                    {
                        ++bad_count;
                    }
                }
            }
        }
        std::printf( "Bad Block: %.2f%%\n", 100.0f * bad_count / total_blocks );

        dot_params.data.device_data =
            MyThrustOp::DOT_statistics_data_to_device( statistics );
        dot_params.data.device_PGParams =
            MyThrustOp::DOT_PG_data_to_device( pg_data );
        dot_params.data.on_GPU = true;

        dot_params.selection_const =
            dropOut_tracing::connection_uniform_sample
                ? lt_params.M_per_core * lt_params.num_core
                      / static_cast<float>( params.sampler.glossy_count )
                : static_cast<float>( lt_params.M_per_core * lt_params.num_core );
        dot_params.selection_const *= 1.0f - dot_params.discard_ratio;
        dot_params.specular_Q = MyThrustOp::DOT_get_Q();

        std::printf(
            "get %d specular subpath, discard ratio %f\n",
            params.sampler.glossy_count,
            dot_params.discard_ratio
        );
        const float incomplete_ratio =
            static_cast<float>( params.sampler.glossy_count + 1 )
            / ( 1.0f - dot_params.discard_ratio )
            / ( lt_params.M_per_core * lt_params.num_core );
        const float train_t = 1.0f / dropout_train_iteration;
        average_incomplete_ratio =
            average_incomplete_ratio * ( 1.0f - train_t )
            + incomplete_ratio * train_t;
        dot_params.discard_ratio_next = clamp(
            1.0f
                - dropOut_tracing::target_num_incomplete_subpath
                      / ( average_incomplete_ratio
                          * lt_params.M_per_core
                          * lt_params.num_core ),
            0.0f,
            0.99f
        );
    }

    void preprocessing( const std::string* capture_path = nullptr )
    {
        throwIfCancelled();
        MyThrustOp::clear_training_set();
        constexpr int target_sample_count = 1000000;
        int current_sample_count = 0;
        detail::SamplingProgressGuard camera_progress_guard;
        while( current_sample_count < target_sample_count )
        {
            const int added_samples = launchPretrace();
            camera_progress_guard.record(
                added_samples,
                "LVCBPT camera-path preprocessing"
            );
            current_sample_count += added_samples;
        }

        MyThrustOp::sample_reweight(
            static_cast<int>( params.width ),
            static_cast<int>( params.height )
        );
        throwIfCancelled();
        std::vector<classTree::divide_weight> unlabeled_samples =
            MyThrustOp::get_weighted_point_for_tree_building( true, 10000 );
        classTree::tree eye_tree = classTree::buildTreeBaseOnExistSample()(
            unlabeled_samples,
            NUM_SUBSPACE,
            0
        );
        throwIfCancelled();

        unlabeled_samples =
            MyThrustOp::get_weighted_point_for_tree_building( false, 10000 );
        classTree::tree light_tree = classTree::buildTreeBaseOnExistSample()(
            unlabeled_samples,
            NUM_SUBSPACE - NUM_SUBSPACE_LIGHTSOURCE,
            0
        );
        throwIfCancelled();

        subspace_info.eye_tree =
            MyThrustOp::eye_tree_to_device( eye_tree.v, eye_tree.size );
        subspace_info.light_tree =
            MyThrustOp::light_tree_to_device( light_tree.v, light_tree.size );
        throwIfCancelled();

        constexpr int target_q_samples = 2000000;
        int current_q_samples = 0;
        detail::SamplingProgressGuard light_progress_guard;
        thrust::device_ptr<float> q_star = nullptr;
        while( current_q_samples < target_q_samples )
        {
            launchLVCTrace();
            const int added_samples = MyThrustOp::preprocess_getQ(
                thrust::device_pointer_cast( params.lt.ans ),
                thrust::device_pointer_cast( params.lt.validState ),
                lightTraceElementCount( params.lt ),
                q_star
            );
            light_progress_guard.record(
                added_samples,
                "LVCBPT light-path preprocessing"
            );
            current_q_samples += added_samples;
            updateDropOutTracingParams();
            throwIfCancelled();
        }
        MyThrustOp::Q_zero_handle( q_star );
        MyThrustOp::node_label( subspace_info.eye_tree, subspace_info.light_tree );

        thrust::device_ptr<float> gamma;
        MyThrustOp::build_optimal_E_train_data( target_sample_count );
        MyThrustOp::preprocess_getGamma( gamma );
        throwIfCancelled();
        if( capture_path )
        {
            MyThrustOp::save_optimal_E_snapshot(
                *capture_path,
                gamma,
                params.experiment_seed
            );
            std::printf(
                "Optimal E problem snapshot saved to %s\n",
                capture_path->c_str()
            );
            return;
        }
        MyThrustOp::train_optimal_E(
            gamma,
            runtime.config().optimal_e_learning_rate,
            runtime.config().optimal_e_iterations,
            cancellation_check
        );
        throwIfCancelled();
        optimal_gamma = gamma;

        subspace_info.Q = thrust::raw_pointer_cast( q_star );
        subspace_info.CMFGamma = thrust::raw_pointer_cast(
            MyThrustOp::Gamma2CMFGamma( gamma, params.sampler.subspace )
        );

        if( !params.spcbpt_pure )
        {
            thrust::host_vector<float> caustic_gamma_values(
                dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE
            );
            thrust::fill(
                caustic_gamma_values.begin(),
                caustic_gamma_values.end(),
                1.0f / dropOut_tracing::default_specularSubSpaceNumber
            );
            subspace_info.CMFCausticGamma =
                MyThrustOp::DOT_causticCMFGamma_to_device( caustic_gamma_values );
            dot_params.CMF_Gamma = subspace_info.CMFCausticGamma;

            thrust::device_ptr<float> caustic_ratio;
            MyThrustOp::get_caustic_frac( caustic_ratio );
            subspace_info.caustic_ratio =
                thrust::raw_pointer_cast( caustic_ratio );
        }
        throwIfCancelled();
    }

    RendererRuntime& runtime;
    MyParams& params;
    LightTraceParams& lt_params;
    PreTraceParams& pr_params;
    subspaceMacroInfo& subspace_info;
    DropOutTracing_params& dot_params;
    path_guiding::SD_PGTrainer pg_trainer;

    bool algorithm_state_initialized = false;
    bool preprocessing_complete = false;
    std::uint64_t scene_generation = 0;
    thrust::device_ptr<float> optimal_gamma = nullptr;

    int combine_train_iteration = 0;
    bool combine_state_initialized = false;
    thrust::host_vector<float> combine_fractions;
    thrust::host_vector<float> caustic_gamma;
    std::vector<float> normal_weight;
    std::vector<int> normal_count;
    std::vector<float> caustic_weight;
    std::vector<int> caustic_count;
    std::vector<float> gamma_non_normalized;
    std::vector<float> gamma_non_normalized_single;
    std::vector<int> gamma_sample_count;

    int dropout_train_iteration = 0;
    float average_incomplete_ratio = 0.0f;
    bool dropout_update_state_initialized = false;
    std::vector<std::vector<std::vector<int>>> dropout_average_count;
    std::vector<std::vector<std::vector<float>>> dropout_average;
    std::vector<std::vector<std::vector<float>>> dropout_variance;

    std::vector<std::vector<std::vector<std::vector<float2>>>> train_vector;
    std::vector<std::vector<std::vector<bool>>> train_finish;
    std::function<bool()> cancellation_check;
};

RendererWorkflow::RendererWorkflow( RendererRuntime& runtime )
    : m_runtime( runtime )
    , m_impl( std::make_unique<Impl>( runtime ) )
{
}

RendererWorkflow::~RendererWorkflow() = default;

void RendererWorkflow::synchronizeSceneGeneration()
{
    if( m_impl->sceneGeneration() != m_runtime.sceneGeneration() )
        m_impl = std::make_unique<Impl>( m_runtime );
}

void RendererWorkflow::initializeAlgorithmState()
{
    synchronizeSceneGeneration();
    m_impl->initializeAlgorithmState();
}

void RendererWorkflow::runPreprocessing( std::function<bool()> should_cancel )
{
    synchronizeSceneGeneration();
    m_impl->runPreprocessing( std::move( should_cancel ) );
}

void RendererWorkflow::captureOptimalEProblem( const std::string& output_path )
{
    synchronizeSceneGeneration();
    m_impl->captureOptimalEProblem( output_path );
}

RendererConfigChange RendererWorkflow::applyConfig( const RendererConfig& config )
{
    validateRendererConfig( config );
    const RendererConfig previous_config = m_runtime.config();
    const RendererConfigChange change =
        classifyRendererConfigChange( previous_config, config );
    if( change == RendererConfigChange::None )
        return change;
    if( change == RendererConfigChange::Resize )
    {
        m_runtime.resize( config.width, config.height );
        return change;
    }

    const InteractiveRendererState interactive_state =
        captureInteractiveState( m_runtime.params() );
    const SceneConfig scene_config = m_runtime.sceneConfig();
    const auto rebuild = [&]( const RendererConfig& target_config )
    {
        m_runtime.loadScene( scene_config );
        m_runtime.initialize( target_config );
        restoreInteractiveState(
            m_runtime.params(),
            interactive_state,
            target_config
        );
        synchronizeSceneGeneration();
        if( requiresRendererPreprocessing( target_config ) )
        {
            m_impl->initializeAlgorithmState();
            m_impl->runPreprocessing( {} );
        }
        m_runtime.resetAccumulation();
    };

    try
    {
        rebuild( config );
    }
    catch( ... )
    {
        const std::exception_ptr apply_error = std::current_exception();
        try
        {
            rebuild( previous_config );
        }
        catch( ... )
        {
            m_runtime.reset();
            throw std::runtime_error(
                "Renderer config apply failed and the previous renderer state "
                "could not be restored"
            );
        }
        std::rethrow_exception( apply_error );
    }
    return change;
}

void RendererWorkflow::renderFrame( uchar4* output )
{
    synchronizeSceneGeneration();
    m_impl->renderFrame( output );
}

} // namespace spcbpt
