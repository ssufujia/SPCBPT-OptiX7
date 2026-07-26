#include "device_thrust.h"
#include "optimal_e_optimizer.h"

#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h> 
#include <thrust/device_vector.h> 
#include<thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h> 
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <cuda_runtime.h> 
#include <fstream>
#include <thrust/extrema.h>
#include<thrust/count.h>
#include <thrust/execution_policy.h>
#include <vector>

timerecord_stage time_records;

typedef thrust::host_vector<float> thrust_host_float;
typedef thrust::device_vector<float> thrust_dev_float;
typedef thrust::device_vector<int> thrust_dev_int;
typedef thrust::host_vector<int> thrust_host_int;
typedef thrust::device_vector<bool> thrust_dev_bool;
typedef thrust::host_vector<bool> thrust_host_bool;


namespace MyThrustOp
{
    template<typename T>
    struct debug_print
    {
        debug_print(thrust::device_vector<T>& aa, char* hint)
        {
            thrust::host_vector<T> a = aa;
            printf("%s:\n", hint);
            for (int i = 0; i < a.size(); i++)
            {
                std::cout << a[i] << " ";
            }
            printf("\n");
        }
    };

    typedef debug_print<float> debug_print_float;
    struct LVCSubspaceInfoCopy
    {
        BDPTVertex* v;
        int* subspaceId;
        float* weight;
        bool glossy_subspace;
        __device__ __host__ LVCSubspaceInfoCopy(BDPTVertex* v, int* subspaceId, float* weight, bool glossy_subspace) :
            v(v), subspaceId(subspaceId), weight(weight), glossy_subspace(glossy_subspace)
        {

        }
        __device__ __host__ void operator()(int i)
        { 
            if (glossy_subspace)
            {
                subspaceId[i] = abs(v[i].specular_record);
                weight[i] = 1;
            }
            else
            {

                subspaceId[i] = v[i].subspaceId;
                float res = float3weight(v[i].flux) / v[i].pdf;
                res = isinf(res) ? 0 : res;
                //weight[i] = 1;
                weight[i] = isnan(res) ? 0 : res;

                //if (weight[i] > 10000)
                //    weight[i] = 10000;
              //  printf("vinfo %d %d %f\n", i, v[i].depth, weight[i] / v[i].pdf);
            }
        } 
    };
    struct is_path_begin
    {
        BDPTVertex* v;
        bool* validState;
        __device__ __host__ is_path_begin(BDPTVertex* v, bool* validState) :v(v), validState(validState)
        {

        }
        __device__ __host__ bool operator()(int i)
        {
            if (validState[i] == true && v[i].depth == 0)
            {
                return true;
            }
            else
            {
                return false;
            } 
        }
    };
    template<typename T>
    struct identical_transform
    {
        __device__ __host__ T operator()(const T& a)
        {
            return a;
        }
    };

    struct glossy_index_check
    {
        BDPTVertex* v;
        bool* validState;
        BufferView<MaterialData::Pbr> mats;
        glossy_index_check(BDPTVertex* v, bool* validState, BufferView<MaterialData::Pbr> mats) :
            v(v), validState(validState), mats(mats)
        {
        }
        __device__ __host__ bool operator()(int i)
        {
            if (validState[i] && v[i].depth >= 1)
            {
                //return v[i].specular_record.id < 0;
                //if (v[i].specular_record.id < 0)return true;
                //else return false;
                for (int k = 0; k < v[i].depth; k++)
                {
                    const MaterialData::Pbr& mat = mats[v[i - k].materialId];
                    if (max(mat.metallic, mat.trans) < 0.9 || mat.roughness > 0.4)
                    {
                        return false;
                    }
                    break;
                }
                //printf("specular check %d\n", v[i].specular_record.id);
                return true;
            }
            return false;
        }
    };

    struct glossy_filter_check
    {
        BDPTVertex* v;
        bool* validState; 
        bool glossy_valid;
        glossy_filter_check(BDPTVertex* v, bool* validState, bool glossy_valid) :
            v(v), validState(validState), glossy_valid(glossy_valid)
        {
        }
        __device__ __host__ bool operator()(int i)
        {  
            if (validState[i] == false)return false; 
            return (glossy_valid && v[i].specular_record < 0) || (!glossy_valid && v[i].specular_record >= 0);
        }
    };


    SubspaceSampler LVC_Process(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange)
    {
        SubspaceSampler sampler;
        thrust_dev_bool d_validState(validState, validState + countRange);
        thrust::host_vector<BDPTVertex> h_vertices(vertices, vertices + countRange);
        static thrust_dev_int d_Vsubspace_info(countRange);
        static thrust_host_int h_Vsubspace_info(countRange);
        static thrust_dev_float d_weight(countRange);
        static thrust_host_float h_weight(countRange);
         
        //thrust_host_bool h_validState = d_validState;
        //thrust_dev_bool d_validState_glossy(countRange);
        //thrust::transform(thrust::make_counting_iterator(0),
        //    thrust::make_counting_iterator(0) + countRange,
        //    d_validState_glossy.begin(),
        //    glossy_filter_check(thrust::raw_pointer_cast(vertices), thrust::raw_pointer_cast(validState), false));

        thrust_host_bool h_validState = d_validState; 
        int valid_count = thrust::count_if(d_validState.begin(), d_validState.begin() + countRange, identical_transform<bool>());
        //printf("non-glossy valid count %d-%d\n", valid_count,
        //    thrust::count_if(d_validState_glossy.begin(), d_validState_glossy.begin() + countRange, identical_transform<bool>()));
        //copy necessary info
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data()),false
            ));
        h_Vsubspace_info = d_Vsubspace_info;
        h_weight = d_weight; 
         
        thrust_host_int num_subspace_vertex(NUM_SUBSPACE);
        thrust_host_float Q_subspace_vertex(NUM_SUBSPACE);
        std::vector<std::vector<int>> sparse_jump_vector(NUM_SUBSPACE);
        std::vector<std::vector<float>> sparse_pmf_vector(NUM_SUBSPACE);
         

        for (int i = 0; i < countRange; i++)
        {
            if (!h_validState[i])
                continue; 

            //valid_count++;
            int subspace = h_Vsubspace_info[i];
            if (subspace < 0 || subspace >= NUM_SUBSPACE)
                continue;
            const float vertex_weight =
                isfinite(h_weight[i]) && h_weight[i] > 0.0f
                ? h_weight[i]
                : 0.0f;
            num_subspace_vertex[subspace] += 1;
            Q_subspace_vertex[subspace] += vertex_weight;
            sparse_jump_vector[subspace].push_back(i);
            sparse_pmf_vector[subspace].push_back(vertex_weight);
            if (sparse_pmf_vector[subspace].size() > 1)
            {
                sparse_pmf_vector[subspace][sparse_pmf_vector[subspace].size() - 1] += sparse_pmf_vector[subspace][sparse_pmf_vector[subspace].size() - 2];
            }
        }
         
        static thrust_dev_float ans_cmf;
        static thrust_dev_int ans_jump;
        static thrust::device_vector<Subspace> ans_subspace(NUM_SUBSPACE);
        
        static thrust_host_float h_cmf;
        static thrust_host_int h_jump;
        static thrust::host_vector<Subspace> h_subspace(NUM_SUBSPACE);
        h_cmf.reserve(valid_count);
        h_jump.reserve(valid_count);  
        h_cmf.resize(valid_count);
        h_jump.resize(valid_count);

        int acc_pointer = 0;
        int jump_bias = 0;
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            auto& subspace = h_subspace[i];
            subspace.id = i;
            subspace.jump_bias = jump_bias;
            subspace.size = sparse_jump_vector[i].size();
            subspace.sum_pmf = Q_subspace_vertex[i];
            jump_bias += subspace.size;
            const bool use_uniform_cmf =
                subspace.size > 0
                && (!isfinite(subspace.sum_pmf) || subspace.sum_pmf <= 0.0f);
            if (use_uniform_cmf)
                subspace.sum_pmf = 1.0f;

            for (int j = 0; j < subspace.size; j++)
            {
                h_jump[acc_pointer] = sparse_jump_vector[i][j];
                h_cmf[acc_pointer] = use_uniform_cmf
                    ? static_cast<float>(j + 1) / subspace.size
                    : sparse_pmf_vector[i][j] / subspace.sum_pmf;
                acc_pointer++;
            }    

        }
        h_cmf.resize(acc_pointer);
        h_jump.resize(acc_pointer);
        ans_cmf = h_cmf;
        //printf("sampler count %d %d\n", ans_cmf.size(), h_cmf.size());
        ans_jump = h_jump;
        ans_subspace = h_subspace;
         
        sampler.vertex_count = acc_pointer;
        sampler.path_count = thrust::count_if(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + countRange,
            is_path_begin(thrust::raw_pointer_cast(vertices), thrust::raw_pointer_cast(validState)));
        //sampler.path_count = 100000;
        // printf("path count %d\n", sampler.path_count);
        sampler.jump_buffer = thrust::raw_pointer_cast(ans_jump.data());
        sampler.cmfs = thrust::raw_pointer_cast(ans_cmf.data());
        sampler.subspace = thrust::raw_pointer_cast(ans_subspace.data());
        sampler.LVC = thrust::raw_pointer_cast(vertices); 
        return sampler;
    }

    thrust_dev_float glossy_subspace_Q(
        dropOut_tracing::default_specularSubSpaceNumber,
        0
    );
    int glossy_launch_count = 0;
    SubspaceSampler LVC_Process_glossyOnly(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange, BufferView<MaterialData::Pbr> mats)
    {
        SubspaceSampler sampler;
        thrust_dev_bool d_validState(validState, validState + countRange);
        thrust_host_bool h_validState = d_validState;
        //thrust::host_vector<BDPTVertex> h_vertices(vertices, vertices + countRange);   

        thrust_dev_bool d_valid_glossy(countRange);
        thrust::transform(thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            d_valid_glossy.begin(),
            glossy_index_check(thrust::raw_pointer_cast(vertices), thrust::raw_pointer_cast(validState), mats));
        thrust_host_bool h_valid_glossy = d_valid_glossy;


        static thrust_dev_int d_Vsubspace_info(countRange);
        static thrust_host_int h_Vsubspace_info(countRange);
        static thrust_dev_float d_weight(countRange); 
        //int valid_count = thrust::count_if(validState, validState + countRange, identical_transform<bool>());
        //copy necessary info------subspace info 
        {
            thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data()),true
            ));
            h_Vsubspace_info = d_Vsubspace_info;
        }

        thrust_host_int h_indexes;
        for (int i = 0; i < countRange; i++)
        {
            if (!h_valid_glossy[i])
                continue;
            if (h_Vsubspace_info[i] == DOT_INVALID_SPECULARID)continue;
            
            h_indexes.push_back(i);
        }
        thrust_host_int h_subspace_vertex_count(dropOut_tracing::default_specularSubSpaceNumber);
        thrust::fill(h_subspace_vertex_count.begin(), h_subspace_vertex_count.end(), 0);
        for (int i = 0; i < h_indexes.size(); i++)
        {
            int index = h_indexes[i];
            h_subspace_vertex_count[h_Vsubspace_info[index]] += 1;
        }
        thrust_host_int h_indexes_rearrange(h_indexes.size());
        thrust_host_int h_vertex_bias;
        {
            thrust_host_int h_subspace_vertex_count_bias(dropOut_tracing::default_specularSubSpaceNumber);
            thrust::exclusive_scan(h_subspace_vertex_count.begin(), h_subspace_vertex_count.end(), h_subspace_vertex_count_bias.begin());
            h_vertex_bias = h_subspace_vertex_count_bias;
            for (int i = 0; i < h_indexes.size(); i++)
            {
                int index_o = h_indexes[i];
                int subspace =  h_Vsubspace_info[index_o];
                int index_n = h_subspace_vertex_count_bias[subspace];
                h_indexes_rearrange[index_n] = index_o;
                h_subspace_vertex_count_bias[subspace]++;
            }
        }
        for (int i = 0; i < dropOut_tracing::default_specularSubSpaceNumber; i++)
        {
            if(DOT_DEBUG_INFO_ENABLE)
                printf("get %d vertex at specular subspace %d\n", h_subspace_vertex_count[i], i);
        }
        static thrust_dev_int d_indexes;
        d_indexes = h_indexes_rearrange;
        // printf("glossy vertices number %d\n", h_indexes_rearrange.size());
        sampler.glossy_count = h_indexes_rearrange.size();
        sampler.glossy_index = thrust::raw_pointer_cast(d_indexes.data());

        static thrust_dev_int d_glossy_subspace_bias;
        static thrust_dev_int d_glossy_subsapce_number_vertex;
        d_glossy_subspace_bias = h_vertex_bias;
        d_glossy_subsapce_number_vertex = h_subspace_vertex_count;
        sampler.glossy_subspace_num = thrust::raw_pointer_cast(d_glossy_subsapce_number_vertex.data());
        sampler.glossy_subspace_bias = thrust::raw_pointer_cast(d_glossy_subspace_bias.data());

        thrust_host_float h_glossy_Q = glossy_subspace_Q;
        float t = 1.0f / (glossy_launch_count + 1);
        for (int i = 0; i < dropOut_tracing::default_specularSubSpaceNumber; i++)
        {            
            h_glossy_Q[i] = h_glossy_Q[i] * (1 - t) + t * float(h_subspace_vertex_count[i]);
        }
        glossy_subspace_Q = h_glossy_Q;
        glossy_launch_count++;

        return sampler;
    }

    float* DOT_get_Q()
    {
        return thrust::raw_pointer_cast(glossy_subspace_Q.data());
    }


    static thrust_host_float h_Q_vec(NUM_SUBSPACE);
    static thrust_dev_float Q_vec;
    static thrust_dev_int optimal_E_active_light(NUM_SUBSPACE, 1);
    static int optimal_E_active_light_count = NUM_SUBSPACE;
    void Q_zero_handle(thrust::device_ptr<float>& Q)
    {
        thrust_host_int h_active_light(NUM_SUBSPACE, 0);
        optimal_E_active_light_count = 0;
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            if (isfinite(h_Q_vec[i]) && h_Q_vec[i] > 0.0f)
            {
                h_active_light[i] = 1;
                optimal_E_active_light_count++;
            }
            else
            {
                h_Q_vec[i] = FLT_MAX;
            }
        }
        optimal_E_active_light = h_active_light;
        Q_vec = h_Q_vec;
        Q = Q_vec.data();
    }
    int preprocess_getQ(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange, thrust::device_ptr<float> &Q)
    {
        static int acc_valid_path = 0;
        thrust_host_float tmp_Q_vec(NUM_SUBSPACE);
        if (!Q)
        {
            acc_valid_path = 0;
            h_Q_vec.resize(NUM_SUBSPACE);
            for (int i = 0; i < NUM_SUBSPACE; i++)
            {
                h_Q_vec[i] = 0;
            }          
        } 
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            tmp_Q_vec[i] = 0;
        }

        thrust_dev_bool d_validState(validState, validState + countRange);
        thrust_host_bool h_validState = d_validState;
        
        static thrust_dev_int d_Vsubspace_info(countRange);
        static thrust_host_int h_Vsubspace_info(countRange);
        static thrust_dev_float d_weight(countRange);
        static thrust_host_float h_weight(countRange);

        int valid_count = thrust::count_if(validState, validState + countRange, identical_transform<bool>());  
        int path_count = thrust::count_if(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + countRange,
            is_path_begin(thrust::raw_pointer_cast(vertices), thrust::raw_pointer_cast(validState)));
        if (path_count == 0)
        {
            Q = Q_vec.data();
            return 0;
        }
        acc_valid_path += path_count;
        float t = path_count / (float)(acc_valid_path);

        //copy necessary info
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data()), false
            ));
        h_Vsubspace_info = d_Vsubspace_info;
        h_weight = d_weight; 

        for (int i = 0; i < countRange; i++)
        {
            if (!h_validState[i])
                continue;
            int subspace = h_Vsubspace_info[i];  
            tmp_Q_vec[subspace] += h_weight[i];
        }  

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            tmp_Q_vec[i] /= path_count;
            h_Q_vec[i] = h_Q_vec[i] * (1 - t) + tmp_Q_vec[i] * t;  
        }

        Q_vec = h_Q_vec;
        Q = Q_vec.data(); 
        return path_count;
    }
    template<typename T>
    struct valid_op :thrust::unary_function<T, bool>
    {
        __device__ __host__
            bool operator()(const T& s)
        {
            return s.valid;
        }
    };
    template<typename T>
    struct valid_op_count :thrust::unary_function<T, int>
    {
        __device__ __host__
            int operator()(const T& s)
        {
            return s.valid?1:0;
        }
    };
    thrust::device_vector<preTracePath> neat_paths;
    thrust::device_vector<preTraceConnection> neat_conns;

    struct bias_arrange_op
    {
        preTracePath* sample;
        preTraceConnection* node;
        int* bias_flag;
        int sample_bias;
        int node_bias;
        __host__ __device__
            bias_arrange_op(preTracePath* sample, preTraceConnection* node, int* bias_flag, int sample_bias, int node_bias) :
            sample(sample), node(node), bias_flag(bias_flag), sample_bias(sample_bias), node_bias(node_bias) {}
        __host__ __device__
            bool operator()(int id)
        {
            int bias = sample[id].begin_ind - bias_flag[sample[id].begin_ind];
            sample[id].begin_ind += node_bias - bias;
            sample[id].end_ind += node_bias - bias;
            for (int i = sample[id].begin_ind - node_bias; i < sample[id].end_ind - node_bias; i++)
            {
                node[i].path_id = id + sample_bias;
            }
        }

    };
    static int acc_num_nodes = 0;
    static int acc_num_samples = 0;
    thrust::device_vector<int> sample_bias_flag;
    void clear_training_set()
    {
        acc_num_nodes = 0;
        acc_num_samples = 0;
        neat_conns.resize(0);
        neat_paths.resize(0);
    }
    thrust::device_vector<float4> d_reference_img_buffer;
    float4* reference_h2d(thrust::host_vector<float4> h_ref)
    {  
        d_reference_img_buffer = h_ref;
        return thrust::raw_pointer_cast(d_reference_img_buffer.data());
    }

    int valid_sample_gather(thrust::device_ptr<preTracePath> raw_paths, int maxPathSize, 
        thrust::device_ptr<preTraceConnection> raw_conns,int maxConns)
    {
        int sample_count = thrust::count_if(raw_paths, raw_paths + maxPathSize, valid_op<preTracePath>());
        int node_count = thrust::count_if(raw_conns, raw_conns + maxConns, valid_op<preTraceConnection>());
        
        static thrust::device_vector<int> sample_bias_flag(maxConns);
        sample_bias_flag.reserve(maxConns);
        sample_bias_flag.resize(maxConns);

        if (acc_num_nodes + node_count > neat_conns.size())
        {
            neat_conns.resize(acc_num_nodes + node_count);
        }
        if (acc_num_samples + sample_count > neat_paths.size())
        {
            neat_paths.resize(acc_num_samples + sample_count);
        }
        thrust::exclusive_scan(
            thrust::make_transform_iterator(raw_conns, valid_op_count<preTraceConnection>()),
            thrust::make_transform_iterator(raw_conns, valid_op_count<preTraceConnection>()) + maxConns,
            sample_bias_flag.begin());
        thrust::copy_if(raw_paths, raw_paths + maxPathSize, neat_paths.begin() + acc_num_samples, valid_op<preTracePath>());
        thrust::copy_if(raw_conns, raw_conns + maxConns, neat_conns.begin() + acc_num_nodes, valid_op<preTraceConnection>());
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + sample_count
            , bias_arrange_op(
                thrust::raw_pointer_cast(neat_paths.data()) + acc_num_samples,
                thrust::raw_pointer_cast(neat_conns.data() + acc_num_nodes),
                thrust::raw_pointer_cast(sample_bias_flag.data()), acc_num_samples, acc_num_nodes));

        acc_num_nodes += node_count;
        acc_num_samples += sample_count;
        //printf("pretrace get %d/%d paths and %d conns\n", sample_count, acc_num_samples,acc_num_nodes);

         
        return sample_count; 
    } 

    std::vector<classTree::divide_weight> getCausticCentroidCandidate(bool eye_side, int max_size)
    {
        thrust::host_vector<preTracePath> h_neat_paths = neat_paths;
        thrust::host_vector<preTraceConnection> h_neat_conns = neat_conns;
        std::vector<classTree::divide_weight> ans;
        float weights = 0;

        int sizeLimit = max_size == 0 ? h_neat_paths.size() : (h_neat_paths.size() > max_size ? max_size : h_neat_paths.size());
        for (int i = 0; i < sizeLimit; i++)
        {
            if (h_neat_paths[i].is_caustic == false)continue;
            if (!isfinite(h_neat_paths[i].sample_pdf)
                || h_neat_paths[i].sample_pdf <= 0.0f)
            {
                continue;
            }
            const float sample_weight =
                float3weight(h_neat_paths[i].contri)
                / h_neat_paths[i].sample_pdf;
            if (!isfinite(sample_weight) || sample_weight < 0.0f)
                continue;
            int j = h_neat_paths[i].begin_ind + h_neat_paths[i].caustic_id;

            classTree::divide_weight t;
            if (eye_side)
            {
                t.dir = h_neat_conns[j].A_dir();
                t.normal = h_neat_conns[j].A_normal();
                t.position = h_neat_conns[j].A_position;
            }
            else
            {
                if (h_neat_conns[j].light_source)
                    continue;
                t.dir = h_neat_conns[j].B_dir();
                t.normal = h_neat_conns[j].B_normal();
                t.position = h_neat_conns[j].B_position;
            }
            t.weight = sample_weight;
            ans.push_back(t);
        }
        printf("get %zu caustic subpaths\n",ans.size());
        return ans;
    }

    std::vector<classTree::divide_weight> get_weighted_point_for_tree_building(bool eye_side, int max_size)
    {
        thrust::host_vector<preTracePath> h_neat_paths = neat_paths;
        thrust::host_vector<preTraceConnection> h_neat_conns = neat_conns;
        std::vector<classTree::divide_weight> ans;
        float weights = 0;
        
        int sizeLimit = max_size == 0? h_neat_paths.size():( h_neat_paths.size() > max_size ? max_size : h_neat_paths.size());
        for (int i = 0; i < sizeLimit; i++)
        {
            if (!isfinite(h_neat_paths[i].sample_pdf)
                || h_neat_paths[i].sample_pdf <= 0.0f)
            {
                continue;
            }
            const float sample_weight =
                float3weight(h_neat_paths[i].contri)
                / h_neat_paths[i].sample_pdf;
            if (!isfinite(sample_weight) || sample_weight < 0.0f)
                continue;
            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            {
                classTree::divide_weight t;
                if (eye_side)
                {
                    t.dir = h_neat_conns[j].A_dir();
                    t.normal = h_neat_conns[j].A_normal();
                    t.position = h_neat_conns[j].A_position;
                }
                else
                {
                    if (h_neat_conns[j].light_source)
                        continue;
                    t.dir = h_neat_conns[j].B_dir();
                    t.normal = h_neat_conns[j].B_normal();
                    t.position = h_neat_conns[j].B_position;
                }
                t.weight = sample_weight;
                ans.push_back(t);
            }
        }
        return ans;
    }

    struct Tree_save
    {
        thrust::device_vector<classTree::tree_node> light_tree;
        thrust::device_vector<classTree::tree_node> eye_tree;
        void clear()
        {
            light_tree.clear();
            eye_tree.clear();
        }
    } tree_save;
    classTree::tree_node* light_tree_to_device(classTree::tree_node* a, int size)
    {
        thrust::host_vector<classTree::tree_node> h_v(a, a + size);
        thrust::device_vector<classTree::tree_node>& d_v = tree_save.light_tree;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }

    classTree::tree_node* eye_tree_to_device(classTree::tree_node* a, int size)
    {
        thrust::host_vector<classTree::tree_node> h_v(a, a + size);
        thrust::device_vector<classTree::tree_node>& d_v = tree_save.eye_tree;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }


    thrust::device_vector<classTree::tree_node> dropout_tracing_specular_tree;
    classTree::tree_node* DOT_specular_tree_to_device(classTree::tree_node* a, int size)
    {
        thrust::host_vector<classTree::tree_node> h_v(a, a + size);
        thrust::device_vector<classTree::tree_node>& d_v = dropout_tracing_specular_tree;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }
     
    thrust::device_vector<classTree::tree_node> dropout_tracing_surface_tree;
    classTree::tree_node* DOT_surface_tree_to_device(classTree::tree_node* a, int size)
    {
        thrust::host_vector<classTree::tree_node> h_v(a, a + size);
        thrust::device_vector<classTree::tree_node>& d_v = dropout_tracing_surface_tree;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }

    thrust::device_vector<dropOut_tracing::PGParams> DOT_PG_data;
    dropOut_tracing::PGParams* DOT_PG_data_to_device(thrust::host_vector<dropOut_tracing::PGParams> h_v)
    { 
        thrust::device_vector<dropOut_tracing::PGParams>& d_v = DOT_PG_data;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }
    thrust::host_vector<dropOut_tracing::PGParams> DOT_PG_data_to_host()
    {
        thrust::host_vector<dropOut_tracing::PGParams> h_v;
        h_v = DOT_PG_data;
        return h_v;
    }


    thrust::device_vector<dropOut_tracing::statistics_data_struct> DOT_statistics_data;
    dropOut_tracing::statistics_data_struct* DOT_statistics_data_to_device(dropOut_tracing::statistics_data_struct* a, int size)
    {
        thrust::host_vector<dropOut_tracing::statistics_data_struct> h_v(a, a + size);
        thrust::device_vector<dropOut_tracing::statistics_data_struct>& d_v = DOT_statistics_data;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }

    dropOut_tracing::statistics_data_struct* DOT_statistics_data_to_device(thrust::host_vector<dropOut_tracing::statistics_data_struct> h_v)
    { 
        thrust::device_vector<dropOut_tracing::statistics_data_struct>& d_v = DOT_statistics_data;
        d_v = h_v;
        return thrust::raw_pointer_cast(d_v.data());
    }

    thrust::host_vector<dropOut_tracing::statistics_data_struct> DOT_statistics_data_to_host()
    {
        thrust::device_vector<dropOut_tracing::statistics_data_struct>& d_v = DOT_statistics_data;
        thrust::host_vector<dropOut_tracing::statistics_data_struct> h_v = d_v;
        return h_v;
    }
     
    thrust::device_vector<dropOut_tracing::statistic_record> DOT_statistics_record_buffer;
    dropOut_tracing::statistic_record* DOT_get_statistic_record_buffer(int size)
    {
        if (size > 0)
        {
            dropOut_tracing::statistic_record temp(DOT_type::LS, 0, 0, DOT_usage::Average);
            temp = 0;
            temp.valid = false;
            DOT_statistics_record_buffer.resize(size);
            thrust::fill(DOT_statistics_record_buffer.begin(), DOT_statistics_record_buffer.end(), temp);
        }
        return thrust::raw_pointer_cast(DOT_statistics_record_buffer.data());
    }

    thrust::host_vector<dropOut_tracing::statistic_record> DOT_get_host_statistic_record_buffer(bool valid_only)
    {
        thrust::host_vector<dropOut_tracing::statistic_record> h_v;
        thrust::host_vector<dropOut_tracing::statistic_record> h_v_filter;
        h_v = DOT_statistics_record_buffer; 

        dropOut_tracing::statistic_record temp(DOT_type::LS, 0, 0, DOT_usage::Average);
        temp = 0;
        temp.valid = false; 
        thrust::fill(DOT_statistics_record_buffer.begin(), DOT_statistics_record_buffer.end(), temp);

        if (valid_only)
        {
            for (int i = 0; i < h_v.size(); i++)
            {
                if (valid_op<dropOut_tracing::statistic_record>()(h_v[i]))
                {
                    h_v_filter.push_back(h_v[i]);
                }
            } 
        } 
        else
        {
            h_v_filter = h_v;
        }
        return h_v_filter;
    }



    thrust::host_vector<uchar4> copy_to_host(uchar4* data, int size)
    {
        thrust::device_vector<uchar4> d_vec(data, data + size);
        thrust::host_vector<uchar4> h_vec = d_vec;
        return h_vec;
    }

    thrust::host_vector<float4> copy_to_host(float4* data, int size)
    {
        thrust::device_vector<float4> d_vec(data, data + size);
        thrust::host_vector<float4> h_vec = d_vec;
        return h_vec;
    }
    


    struct tree_label_op
    {
        classTree::tree_node* eye_tree;
        classTree::tree_node* light_tree;
        __device__ __host__
            tree_label_op(classTree::tree_node* eye_tree, classTree::tree_node* light_tree) :eye_tree(eye_tree), light_tree(light_tree) {}
        __device__ __host__
            bool operator()(preTraceConnection& s)
        {
            s.label_A = classTree::tree_index(eye_tree, s.A_position, s.A_normal(), s.A_dir());
            if (!s.light_source)
                s.label_B = classTree::tree_index(light_tree, s.B_position, s.B_normal(), s.B_dir());
            //printf("label %d %d\n", s.label_A, s.label_B);
        }
    };
    void node_label(classTree::tree_node* eye_tree, classTree::tree_node* light_tree)
    {
        thrust::for_each(neat_conns.begin(), neat_conns.begin() + acc_num_nodes, tree_label_op(eye_tree, light_tree));
        printf("\n\nnode label complete\n\n");
    }
    void sample_reweight(int width, int height)
    {
        if (width <= 0 || height <= 0)
            throw std::invalid_argument("Sample reweight dimensions must be positive");
        thrust::host_vector<preTracePath> h_samples = neat_paths;
        constexpr int cell_size = 10;
        const int grid_width = (width + cell_size - 1) / cell_size;
        const int grid_height = (height + cell_size - 1) / cell_size;
        thrust::host_vector<float> weight(grid_width * grid_height, 0.0f);
        for (int i = 0; i < h_samples.size(); i++)
        {
            const int pixel_x = h_samples[i].pixel_id.x;
            const int pixel_y = h_samples[i].pixel_id.y;
            if (pixel_x < 0 || pixel_x >= width
                || pixel_y < 0 || pixel_y >= height)
            {
                continue;
            }
            const int n_id =
                pixel_x / cell_size
                + (pixel_y / cell_size) * grid_width;
            if (!isfinite(h_samples[i].sample_pdf)
                || h_samples[i].sample_pdf <= 0.0f)
            {
                continue;
            }
            float ww = float3weight(h_samples[i].contri) / h_samples[i].sample_pdf;
            if (!isfinite(ww) || ww < 0.0f)continue;
            weight[n_id] += ww;
        }
        for (int i = 0; i < h_samples.size(); i++)
        {
            const int pixel_x = h_samples[i].pixel_id.x;
            const int pixel_y = h_samples[i].pixel_id.y;
            if (pixel_x < 0 || pixel_x >= width
                || pixel_y < 0 || pixel_y >= height)
            {
                continue;
            }
            if (!isfinite(h_samples[i].sample_pdf)
                || h_samples[i].sample_pdf <= 0.0f)
            {
                h_samples[i].contri = make_float3(0.0f);
                continue;
            }
            const float sample_weight =
                float3weight(h_samples[i].contri) / h_samples[i].sample_pdf;
            if (!isfinite(sample_weight) || sample_weight < 0.0f)
            {
                h_samples[i].contri = make_float3(0.0f);
                continue;
            }
            const int cell_x = pixel_x / cell_size;
            const int cell_y = pixel_y / cell_size;
            const int n_id = cell_x + cell_y * grid_width;
            const int cell_pixel_count =
                min(cell_size, width - cell_x * cell_size)
                * min(cell_size, height - cell_y * cell_size);
            float w = weight[n_id] / cell_pixel_count + 0.1f;
            h_samples[i].contri = h_samples[i].contri / w;
        }
        neat_paths = h_samples;
    }
    static thrust_dev_float Gamma_vec;
    static thrust_dev_float Gamma_vec_caustic;
    static thrust_host_float h_Gamma(NUM_SUBSPACE* NUM_SUBSPACE);

    void get_caustic_frac(thrust::device_ptr<float>& frac)
    {
        thrust_host_float h_frac(NUM_SUBSPACE);
        static thrust_dev_float d_frac;
        thrust::fill(h_frac.begin(), h_frac.end(), 0);

        thrust::host_vector<preTracePath> h_neat_paths = neat_paths;
        thrust::host_vector<preTraceConnection> h_neat_conns = neat_conns;

        thrust_host_float non_normalized_sum(NUM_SUBSPACE);
        thrust_host_float non_normalized_caustic(NUM_SUBSPACE);
        thrust::fill(non_normalized_caustic.begin(), non_normalized_caustic.end(), 0);
        thrust::fill(non_normalized_sum.begin(), non_normalized_sum.end(), 0);

        for (int i = 0; i < h_neat_paths.size(); i++)
        {
            if (!isfinite(h_neat_paths[i].sample_pdf)
                || h_neat_paths[i].sample_pdf <= 0.0f)
            {
                continue;
            }
            float weight = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;
            if (!isfinite(weight) || weight < 0.0f)
                continue;

            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            {
                int eye_id = h_neat_conns[j].label_A;
                if (eye_id < 0 || eye_id >= NUM_SUBSPACE)
                    continue;

                float weight2 = min(weight, 10.0);
                if (h_neat_paths[i].is_caustic && j - h_neat_paths[i].begin_ind == h_neat_paths[i].caustic_id)
                {
                    non_normalized_caustic[eye_id] += weight2;
                }
                non_normalized_sum[eye_id] += weight2;
            }
        }
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            h_frac[i] = non_normalized_sum[i] > 0.0f
                ? non_normalized_caustic[i] / non_normalized_sum[i]
                : 0.0f;
        }
        d_frac = h_frac;
        frac = d_frac.data();
    }

    path_guiding::quad_tree_node* quad_tree_to_device(path_guiding::quad_tree_node* a, int size)
    {
        thrust::host_vector<path_guiding::quad_tree_node> h_vec(a, a + size);
        static thrust::device_vector<path_guiding::quad_tree_node> d_vec;
        d_vec = h_vec;
        return thrust::raw_pointer_cast(d_vec.data());
    }

    path_guiding::Spatio_tree_node* spatio_tree_to_device(path_guiding::Spatio_tree_node* a, int size)
    {
        thrust::host_vector<path_guiding::Spatio_tree_node> h_vec(a, a + size);
        static thrust::device_vector<path_guiding::Spatio_tree_node> d_vec;
        d_vec = h_vec;
        return thrust::raw_pointer_cast(d_vec.data());
    }

    std::vector<path_guiding::PG_training_mat> get_data_for_path_guiding(int num_datas, bool UPT_ONLY)
    {
        thrust::host_vector<preTracePath> h_neat_paths = neat_paths;
        thrust::host_vector<preTraceConnection> h_neat_conns = neat_conns;
        std::vector<path_guiding::PG_training_mat> ans;
        int slice_range = num_datas == -1 ? h_neat_conns.size() : (num_datas < h_neat_conns.size() ? num_datas : h_neat_conns.size());
        for (int i = 0; i < acc_num_samples; i++)
        { 
            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            { 
                path_guiding::PG_training_mat mat;
                if(!UPT_ONLY)
                    mat.lum = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;
                else
                   mat.lum = h_neat_conns[j].get_PG_weight();
                if (mat.lum > 100000)mat.lum = 100000;
                if (!isfinite(mat.lum) || mat.lum < 0.0f)continue;
                mat.position = h_neat_conns[j].A_position;
                mat.uv = dir2uv(normalize(h_neat_conns[j].B_position - h_neat_conns[j].A_position));
                mat.valid = true;
                ans.push_back(mat);
                if (ans.size() >= slice_range)return ans;
                //break;
            }
        }
        return ans;
    } 

    thrust::device_vector<dropOut_tracing::pixelRecord> DOT_pixelRecords;
    dropOut_tracing::pixelRecord* DOT_set_pixelRecords_size(int size)
    {
        DOT_pixelRecords.resize(size);
        dropOut_tracing::pixelRecord t;
        t.record = 0; 
        t.is_valid = false;
        thrust::fill(DOT_pixelRecords.begin(), DOT_pixelRecords.end(), t);
        return thrust::raw_pointer_cast(DOT_pixelRecords.data());
    }

    thrust::host_vector<dropOut_tracing::pixelRecord> DOT_get_pixelRecords()
    {
        thrust::host_vector<dropOut_tracing::pixelRecord> host_records(DOT_pixelRecords);

        // Set all records in DOT_pixelRecords to 0
        dropOut_tracing::pixelRecord t;
        t.record = 0;
        t.is_valid = false;
        thrust::fill(DOT_pixelRecords.begin(), DOT_pixelRecords.end(), t);
        return host_records;


        ////filter code
        //// Use thrust::copy_if to copy only elements with record > 0 to a new device vector
        //thrust::device_vector<dropOut_tracing::pixelRecord> filtered_records(DOT_pixelRecords.size());
        //auto new_end = thrust::copy_if(DOT_pixelRecords.begin(), DOT_pixelRecords.end(), 
        //    filtered_records.begin(), [](const dropOut_tracing::pixelRecord& pr) { return abs(pr.record) > 0; });

        //// Resize the new vector to the number of elements copied
        //filtered_records.resize(thrust::distance(filtered_records.begin(), new_end));
        //thrust::host_vector<dropOut_tracing::pixelRecord> host_filtered_records(filtered_records);
        //return host_filtered_records;
    }

    float* DOT_causticFrac_to_device(thrust::host_vector<float> DOT_h_frac)
    {
        static thrust_dev_float d_frac;
        d_frac = DOT_h_frac;
        return thrust::raw_pointer_cast(d_frac.data());
    }

    float* DOT_causticCMFGamma_to_device(thrust::host_vector<float> DOT_h_GAMMA)
    { 
        thrust_host_float p = DOT_h_GAMMA;
        static thrust_dev_float d_cmf_gamma(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE);
        d_cmf_gamma = DOT_h_GAMMA;
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; j++)
            { 
                p[i * dropOut_tracing::default_specularSubSpaceNumber + j] = p[i * dropOut_tracing::default_specularSubSpaceNumber + j];
            }
        }
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; j++)
            {
                int index = i * dropOut_tracing::default_specularSubSpaceNumber + j;
                if (j != 0)
                {
                    p[index] += p[index - 1];
                }
            }
            p[(i + 1) * dropOut_tracing::default_specularSubSpaceNumber - 1] = 1;
        }

        d_cmf_gamma = p;
        return thrust::raw_pointer_cast(d_cmf_gamma.data());
    }

    void preprocess_getGamma(thrust::device_ptr<float>& Gamma, bool caustic_case)
    {
        thrust_dev_float& d_gamma = caustic_case ? Gamma_vec_caustic : Gamma_vec;
        thrust::host_vector<preTracePath> h_neat_paths = neat_paths;
        thrust::host_vector<preTraceConnection> h_neat_conns = neat_conns;
        int caustic_filter_count = 0;
        for (int i = 0; i < NUM_SUBSPACE * NUM_SUBSPACE; i++) h_Gamma[i] = 0;
        for (int i = 0; i < h_neat_paths.size(); i++)
        {
            //in the case of caustic Gamma, ignore the ordinary path
            //but in the case of normal Gamma, most of the connections of caustic path are valid still.
            if (caustic_case != h_neat_paths[i].is_caustic)
            {
                caustic_filter_count++;
                if(caustic_case) continue;
                
            } 

            if (!isfinite(h_neat_paths[i].sample_pdf)
                || h_neat_paths[i].sample_pdf <= 0.0f)
            {
                continue;
            }
            float weight =
                float3weight(h_neat_paths[i].contri)
                / h_neat_paths[i].sample_pdf;
            if (!isfinite(weight) || weight < 0.0f)
                continue;
            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            {
                if (caustic_case)
                {
                    if (j - h_neat_paths[i].begin_ind != h_neat_paths[i].caustic_id)
                    {
                        continue;
                    }
                }


                int eye_id = h_neat_conns[j].label_A;
                int light_id = h_neat_conns[j].label_B;
                if (eye_id < 0 || eye_id >= NUM_SUBSPACE
                    || light_id < 0 || light_id >= NUM_SUBSPACE)
                {
                    continue;
                }
                int GammaId = eye_id * NUM_SUBSPACE + light_id;
//                float peak_pdf = h_neat_conns[j].peak_pdf / h_Q_vec[light_id];
  //              float weight2 = weight * float3weight(h_neat_paths[i].contri) / peak_pdf;

                float weight2 = min(weight, 10.0);
                h_Gamma[GammaId] += weight2;
            }
        }
        
        if (caustic_case == false)
        {
            printf("%d / %zu paths are caustic paths and deleted from the training of ordinaryGamma.\n", caustic_filter_count, h_neat_paths.size());
        }
        else
        {
            printf("%d / %zu paths are ordinary paths and deleted from the training of causticGamma.\n", caustic_filter_count, h_neat_paths.size());
        }

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            float weightS = 0;
            for (int j = 0; j < NUM_SUBSPACE; j++)
                weightS += h_Gamma[i * NUM_SUBSPACE + j];
            if (!isfinite(weightS) || weightS <= 1e-10f)
            {
                for (int j = 0; j < NUM_SUBSPACE; j++)
                    h_Gamma[i * NUM_SUBSPACE + j] = 1.0f / NUM_SUBSPACE;
            }
            else
            {
                for (int j = 0; j < NUM_SUBSPACE; j++)
                    h_Gamma[i * NUM_SUBSPACE + j] /= weightS;
            }
        }
        d_gamma = h_Gamma;
        Gamma = d_gamma.data();
    }

#define optimal_E_loss_threshold (1000000.0)
    thrust::device_vector<float> b_f_square;
    thrust::device_vector<float> b_pdf0;
    thrust::device_vector<float> b_pdf_peak;
    thrust::device_vector<int> b_label_E;
    thrust::device_vector<int> b_P2N_ind_d;
    spcbpt::OptimalEProblem optimal_E_problem = {};
    thrust::device_vector<float> d_E;

    struct get_sample_light_id :thrust::unary_function<preTracePath, int>
    {
        preTraceConnection* nodes;
        get_sample_light_id(preTraceConnection* nodes) :nodes(nodes) {}
        __host__ __device__ int operator()(preTracePath& s)
        {
            if (nodes[s.begin_ind].light_source)return 0;
            return s.choice_id;
        }
    };

    struct construct_optimal_E_data_sample :thrust::unary_function<int, int>
    {
        float* f_square;
        float* pdf0;
        int* P2N_ind;
        preTracePath* samples;
        construct_optimal_E_data_sample(float* f_square, float* pdf0, int* P2N_ind, preTracePath* samples) :
            f_square(f_square), pdf0(pdf0), P2N_ind(P2N_ind), samples(samples) {}
        __host__ __device__
            int operator()(int id)
        {
            preTracePath& s = samples[id];
            const float weight = float3weight(s.contri);
            const float loss = weight * weight / s.sample_pdf;
            const bool invalid =
                !isfinite(weight)
                || !isfinite(s.sample_pdf)
                || s.sample_pdf <= 0.0f
                || !isfinite(s.fix_pdf)
                || s.fix_pdf < 0.0f
                || !isfinite(loss)
                || loss < 0.0f;
            if (invalid || loss == 0.0f)
            {
                // A zero objective weight is equivalent to filtering this path
                // while preserving the existing CSR path/node layout.
                f_square[id] = 0.0f;
                pdf0[id] = 1.0f;
            }
            else
            {
                f_square[id] = min(loss, float(optimal_E_loss_threshold));
                pdf0[id] = s.fix_pdf;
            }
            P2N_ind[id] = s.begin_ind;
            return invalid ? 1 : 0;
        }
    };

    struct construct_optimal_E_data_node :thrust::unary_function<int, bool>
    {
        float* peak_pdf;
        float* Q;
        const int* active_light;
        int* label_E;
        preTraceConnection* nodes;
        int dim_light;
        construct_optimal_E_data_node(
            float* peak_pdf,
            float* Q,
            const int* active_light,
            int* label_E,
            preTraceConnection* nodes,
            int dim_light
        ) :
            peak_pdf(peak_pdf),
            Q(Q),
            active_light(active_light),
            label_E(label_E),
            nodes(nodes),
            dim_light(dim_light)
        {
        }
        __host__ __device__
            bool operator()(int id)
        {
            preTraceConnection& s = nodes[id];
            int eye_id = s.label_A;
            int light_id = s.label_B;
            if (eye_id < 0 || eye_id >= dim_light
                || light_id < 0 || light_id >= dim_light)
            {
                label_E[id] = -1;
                peak_pdf[id] = 0.0f;
                return true;
            }
            label_E[id] = eye_id * dim_light + light_id;
            peak_pdf[id] =
                active_light[light_id] && Q[light_id] > 0.0
                ? s.peak_pdf / Q[light_id]
                : 0.0;
            if (isnan(peak_pdf[id]) || isinf(peak_pdf[id]))
            {
                peak_pdf[id] = 0;
            }
            return false;
        }
    };
    __host__ __device__ float optimal_e_outlier_value(
        const preTracePath& sample,
        const preTraceConnection* nodes,
        const float* Q
    )
    {
        if (!isfinite(sample.fix_pdf) || sample.fix_pdf < 0.0f
            || !isfinite(sample.sample_pdf) || sample.sample_pdf <= 0.0f)
        {
            return 0.0f;
        }
        float denominator = sample.fix_pdf;
        const float weight = float3weight(sample.contri);
        float loss = weight * weight / sample.sample_pdf;
        if (!isfinite(loss) || loss < 0.0f)
            return 0.0f;
        loss = min(loss, float(optimal_E_loss_threshold));
        for (int i = sample.begin_ind; i < sample.end_ind; i++)
        {
            const float q = Q[nodes[i].label_B];
            if (q > 0.0f && isfinite(q) && isfinite(nodes[i].peak_pdf))
                denominator += nodes[i].peak_pdf / q / 1000.0f;
        }
        if (!isfinite(denominator) || denominator <= 0.0f)
            return 0.0f;
        const float result = loss / denominator;
        return isfinite(result) ? result : 0.0f;
    }

    struct get_outler_value :thrust::unary_function<int, float>
    {
        preTraceConnection* nodes;
        preTracePath* samples;
        float* Q;
        get_outler_value(preTracePath* samples, preTraceConnection* nodes, float* Q) :nodes(nodes), samples(samples), Q(Q) {}
        __host__ __device__
            float operator()(int id)
        {
            return optimal_e_outlier_value(samples[id], nodes, Q);
        }
    };

    struct clean_outler_value :thrust::unary_function<preTracePath, bool>
    {
        preTraceConnection* nodes;
        float* Q;
        float threshold;
        clean_outler_value(preTraceConnection* nodes, float* Q, float threshold) :
            nodes(nodes), Q(Q), threshold(threshold) {}
        __host__ __device__
            float operator()(preTracePath& s)
        {
            const float outler_value =
                optimal_e_outlier_value(s, nodes, Q);
            if (outler_value > threshold)
            {
                s.contri *= 0;
                return true;
            }
            return false;
        }
    };

    struct count_outler_value :thrust::unary_function<preTracePath, bool>
    {
        preTraceConnection* nodes;
        float* Q;
        float threshold;
        count_outler_value(preTraceConnection* nodes, float* Q, float threshold) :
            nodes(nodes), Q(Q), threshold(threshold) {}
        __host__ __device__
            float operator()(const preTracePath& s)
        {
            const float outler_value =
                optimal_e_outlier_value(s, nodes, Q);
            if (outler_value > threshold)
            {
                return true;
            }
            return false;
        }
    };
    void build_optimal_E_train_data(int N_samples)
    {
        thrust::device_vector<int> ids(acc_num_samples);
        thrust::transform(neat_paths.begin(), neat_paths.end(), ids.begin(), get_sample_light_id(thrust::raw_pointer_cast(neat_conns.data())));
        thrust::sort(ids.begin(), ids.end(), thrust::greater<int>());
        thrust::device_vector<int> sum_ids(acc_num_samples);
        auto new_end = thrust::reduce_by_key(ids.begin(), ids.end(), thrust::make_constant_iterator(1), thrust::make_discard_iterator(), sum_ids.begin());
        thrust::sort(sum_ids.begin(), new_end.second, thrust::greater<int>());


        thrust::host_vector<preTracePath> h_tem(neat_paths.begin() + N_samples - 1, neat_paths.begin() + N_samples);
        int M_nodes = h_tem[0].end_ind;
        b_f_square.resize(N_samples);
        b_pdf0.resize(N_samples);
        b_pdf_peak.resize(M_nodes);
        b_label_E.resize(M_nodes);
        b_P2N_ind_d.resize(N_samples + 1);

        {
            thrust::device_vector<float> t_outler(1000);
            thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 1000, t_outler.begin(),
                get_outler_value(
                    thrust::raw_pointer_cast(neat_paths.data()),
                    thrust::raw_pointer_cast(neat_conns.data()),
                    thrust::raw_pointer_cast(Q_vec.data())));
            thrust::sort(t_outler.begin(), t_outler.end());
            thrust::host_vector<float> h_outler = t_outler;
            float outler_value = h_outler[999];
            thrust::for_each(neat_paths.begin(), neat_paths.end(),
                clean_outler_value(thrust::raw_pointer_cast(neat_conns.data()), thrust::raw_pointer_cast(Q_vec.data()), outler_value));
            int irr_count = thrust::count_if(neat_paths.begin(), neat_paths.end(),
                count_outler_value(thrust::raw_pointer_cast(neat_conns.data()), thrust::raw_pointer_cast(Q_vec.data()), outler_value));
            printf("\n\nsample should be clean:%d / %zu %f\n\n", irr_count, neat_paths.size(), outler_value);
        }

        thrust::device_vector<int> invalid_samples(N_samples);
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + N_samples,
            invalid_samples.begin(),
            construct_optimal_E_data_sample(
                thrust::raw_pointer_cast(b_f_square.data()),
                thrust::raw_pointer_cast(b_pdf0.data()),
                thrust::raw_pointer_cast(b_P2N_ind_d.data()),
                thrust::raw_pointer_cast(neat_paths.data())));
        const int invalid_sample_count =
            thrust::reduce(invalid_samples.begin(), invalid_samples.end(), 0);
        printf(
            "Optimal E excluded %d / %d invalid paths\n",
            invalid_sample_count,
            N_samples
        );

        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + M_nodes,
            construct_optimal_E_data_node(
                thrust::raw_pointer_cast(b_pdf_peak.data()),
                thrust::raw_pointer_cast(Q_vec.data()),
                thrust::raw_pointer_cast(optimal_E_active_light.data()),
                thrust::raw_pointer_cast(b_label_E.data()),
                thrust::raw_pointer_cast(neat_conns.data()),
                Q_vec.size()));
        if (optimal_E_active_light_count == 0)
            throw std::runtime_error("Optimal E has no active light subspace");
        thrust::fill(
            b_P2N_ind_d.begin() + N_samples,
            b_P2N_ind_d.end(),
            M_nodes
        );

        optimal_E_problem = {
            N_samples,
            M_nodes,
            NUM_SUBSPACE,
            NUM_SUBSPACE,
            thrust::raw_pointer_cast(b_f_square.data()),
            thrust::raw_pointer_cast(b_pdf0.data()),
            thrust::raw_pointer_cast(b_pdf_peak.data()),
            thrust::raw_pointer_cast(b_P2N_ind_d.data()),
            thrust::raw_pointer_cast(b_label_E.data()),
            thrust::raw_pointer_cast(optimal_E_active_light.data()),
            optimal_E_active_light_count
        };
        printf("Q_vec size %zu\n\n", Q_vec.size());


    }

    void train_optimal_E(thrust::device_ptr<float>& E_ptr)
    {
        d_E.assign(E_ptr, E_ptr + NUM_SUBSPACE * NUM_SUBSPACE);
        const spcbpt::OptimalEOptimizerResult result =
            spcbpt::optimizeOptimalE(
                optimal_E_problem,
                thrust::raw_pointer_cast(d_E.data()),
                {
                    CONSERVATIVE_RATE,
                    1.0f,
                    20,
                    12,
                    1e-8f
                }
            );
        printf(
            "Optimal E objective: %.9g -> %.9g (%d accepted steps)\n",
            result.initial_objective,
            result.final_objective,
            result.accepted_steps
        );
        Gamma_vec = d_E;
        E_ptr = Gamma_vec.data();

    }

    void save_optimal_E_snapshot(
        const std::string& path,
        thrust::device_ptr<float> base_distribution,
        unsigned int experiment_seed
    )
    {
        spcbpt::saveOptimalESnapshot(
            path,
            optimal_E_problem,
            thrust::raw_pointer_cast( base_distribution ),
            CONSERVATIVE_RATE,
            experiment_seed
        );
    }
#include<fstream>

    void load_Gamma_file(thrust::device_ptr<float>& Gamma)
    {
        thrust::host_vector<float> h_E_current(Gamma, Gamma + NUM_SUBSPACE * NUM_SUBSPACE);

        std::ifstream inFile;
        inFile.open("E.txt");
        float value;
        thrust::host_vector<float> h_E;
        h_E.clear();
        int id_eye = 0;
        int id_light = 0;
        while (inFile >> value)
        {
            if (id_light < NUM_SUBSPACE - NUM_SUBSPACE_LIGHTSOURCE)
            {
                h_E.push_back(value);
            }
            else
            {
                h_E.push_back(h_E_current[id_eye * NUM_SUBSPACE + id_light]);
            }
            id_light++;
            if (id_light == NUM_SUBSPACE)
            {
                id_light = 0;
                id_eye++;
            }
        }
        static thrust::device_vector<float> E_dev = h_E;
        printf("load E size %zu\n", E_dev.size());
        Gamma = E_dev.data();

    }

    thrust::device_vector<float> env_map_cmf;

    template <typename Vector>
    void release_vector(Vector& values)
    {
        Vector empty;
        values.swap(empty);
    }

    void invalidate_scene_caches()
    {
        clear_training_set();
        release_vector(neat_conns);
        release_vector(neat_paths);
        release_vector(sample_bias_flag);
        // The reference image belongs to the estimator, not to a scene.
        // Keeping it alive makes the captured EstimationParams pointer stable
        // across config-driven reloads of the same scene.
        release_vector(tree_save.light_tree);
        release_vector(tree_save.eye_tree);
        release_vector(dropout_tracing_specular_tree);
        release_vector(dropout_tracing_surface_tree);
        release_vector(DOT_PG_data);
        release_vector(DOT_statistics_data);
        release_vector(DOT_statistics_record_buffer);
        release_vector(DOT_pixelRecords);
        release_vector(Gamma_vec);
        release_vector(Gamma_vec_caustic);
        release_vector(b_f_square);
        release_vector(b_pdf0);
        release_vector(b_pdf_peak);
        release_vector(b_label_E);
        release_vector(b_P2N_ind_d);
        release_vector(d_E);
        optimal_E_problem = {};
        release_vector(env_map_cmf);
        thrust::fill(
            glossy_subspace_Q.begin(),
            glossy_subspace_Q.end(),
            0.0f
        );
        glossy_launch_count = 0;
    }

    thrust::device_ptr<float> envMapCMFBuild(float* pmf, int size)
    {
        thrust::host_vector<float> p2(pmf, pmf + size);
        env_map_cmf = p2;
        return env_map_cmf.data();
    }
    void load_Q_file(thrust::device_ptr<float>& Q)
    { 
        std::ifstream inFile;
        inFile.open("Q.txt");
        float value;
        thrust::host_vector<float> h_Q;
        h_Q.clear();
        while (inFile>>value)
        {
            h_Q.push_back(value);
            printf("Q2 %zu %f\n",h_Q.size() - 1, value);
        }
        static thrust::device_vector<float> Q_dev = h_Q;
        printf("load Q size %zu\n",Q_dev.size());
        Q = Q_dev.data();
    }

    thrust::device_ptr<float> Gamma2CMFGamma(thrust::device_ptr<float> Gamma,Subspace* subspace)
    {
        thrust_host_float p(Gamma, Gamma + NUM_SUBSPACE * NUM_SUBSPACE);
        static thrust_dev_float d_CMFGamma;
        static thrust_dev_float d_CMFGamma_caustic;
        thrust_dev_float& d_cmf_gamma = d_CMFGamma;

        thrust::device_vector<Subspace> d_a(subspace, subspace + NUM_SUBSPACE);
        thrust::host_vector<Subspace> h_a = d_a;

        int active_subspace_count = 0;
        for (int j = 0; j < NUM_SUBSPACE; j++)
        {
            active_subspace_count +=
                h_a[j].size > 0
                && isfinite(h_a[j].sum_pmf)
                && h_a[j].sum_pmf > 0.0f
                ? 1
                : 0;
        }
        if (active_subspace_count == 0)
        {
            throw std::runtime_error(
                "Cannot build the LVCBPT sampling distribution: "
                "all light-vertex subspaces are empty"
            );
        }

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            float active_probability_sum = 0.0f;
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
                const int index = i * NUM_SUBSPACE + j;
                const bool is_active =
                    h_a[j].size > 0
                    && isfinite(h_a[j].sum_pmf)
                    && h_a[j].sum_pmf > 0.0f;
                if (!is_active || !isfinite(p[index]) || p[index] < 0.0f)
                    p[index] = 0.0f;
                active_probability_sum += p[index];
            }

            const bool use_uniform_base =
                !isfinite(active_probability_sum) || active_probability_sum <= 1e-10f;
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
                const int index = i * NUM_SUBSPACE + j;
                if (h_a[j].size == 0
                    || !isfinite(h_a[j].sum_pmf)
                    || h_a[j].sum_pmf <= 0.0f)
                {
                    p[index] = 0.0f;
                    continue;
                }
                float t = CONSERVATIVE_RATE;
                const float base_probability = use_uniform_base
                    ? 1.0f / active_subspace_count
                    : p[index] / active_probability_sum;
                p[index] =
                    base_probability * (1.0f - t)
                    + t / active_subspace_count;
            }
        }
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
                int index = i * NUM_SUBSPACE + j;
                if (j != 0)
                {
                    p[index] += p[index - 1];
                }
            }
            p[(i + 1) * NUM_SUBSPACE - 1] = 1;
        }
        
        d_cmf_gamma = p;
        return d_cmf_gamma.data();
    }
}
