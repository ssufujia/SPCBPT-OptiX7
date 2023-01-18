#include "device_thrust.h"

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
#include <random>
#include<thrust/count.h>
#include <thrust/execution_policy.h>
#include <cublas_v2.h>   
#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

using std::default_random_engine;

timerecord_stage time_records;
default_random_engine random_generator_;




float rnd_float()
{
    return float(random_generator_()) / random_generator_.max();
}
int rnd_int(int mod = 1000)
{
    return random_generator_() % mod;
}
void initialFloat(float* p, int size)
{
    for (int i = 0; i < size; i++)
    {
        p[i] = rnd_float();
    }
}

template<typename T>
T*& createDevBuffer_withHost(T* h_ptr, int w, int h = 1, int d = 1)
{
    int n_bytes = sizeof(T) * w * h * d;
    T* d_ptr;

    cudaMalloc((void**)&d_ptr, n_bytes);
    cudaMemcpy(d_ptr, h_ptr, n_bytes, cudaMemcpyHostToDevice);

    return d_ptr;
}
template<typename T>
T*& createHostBuffer(int w, int h = 1, int d = 1)
{
    int n_bytes = sizeof(T) * w * h * d;
    T* h_ptr = (T*)malloc(n_bytes);
    return h_ptr;
}
float*& createDevFloatBuffer_withInit(int w, int h = 1, int d = 1)
{
    float* h_ptr = createHostBuffer<float>(w, h, d);
    initialFloat(h_ptr, w * h * d);
    float* d_ptr = createDevBuffer_withHost<float>(h_ptr, w, h, d);
    return d_ptr;
}

__global__ void foo()
{
    printf("CUDA!\n");
}

void batch_gemm_test()
{
    cublasHandle_t cnpHandle;
    cublasStatus_t status = cublasCreate(&cnpHandle);
    int sample_sum = 2000000;
    int batch_size = 10000;
    int num_batches = sample_sum / batch_size;
    int epoches = 20;

    int ssNum = 1000;

    int input_dim = 60;
    int output_dim = 16;

    int m = 16;
    int n = 1;
    int k = 60;
    int rowsA = m;
    int colsA = k;
    int rowsB = k;
    int colsB = n;
    int rowsC = m;
    int colsC = n;

    int matrixSizeA = rowsA * colsA;
    int matrixSizeB = rowsB * colsB;
    int matrixSizeC = rowsC * colsC;

    int N = batch_size;
    float** devPtrA = createHostBuffer<float*>(N);
    float** devPtrB = createHostBuffer<float*>(N);
    float** devPtrC = createHostBuffer<float*>(N);
    for (int i = 0; i < N; i++)
    {//矩阵初始化，并把矩阵的地址放到host端
        devPtrA[i] = createDevFloatBuffer_withInit(rowsA, colsA);
        devPtrB[i] = createDevFloatBuffer_withInit(rowsB, colsB);
        devPtrC[i] = createDevFloatBuffer_withInit(rowsC, colsC);
    } //地址传到device端
    float** devPtrA_dev = createDevBuffer_withHost<float*>(devPtrA, N);
    float** devPtrB_dev = createDevBuffer_withHost<float*>(devPtrB, N);
    float** devPtrC_dev = createDevBuffer_withHost<float*>(devPtrC, N);

    float* alpha_array = createHostBuffer<float>(N);
    initialFloat(alpha_array, N);
    for (int i = 0; i < epoches * num_batches; i++)
    {
        cublasSgemmBatched(cnpHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            alpha_array,
            devPtrA_dev, rowsA,
            devPtrB_dev, rowsB,
            alpha_array,
            devPtrC_dev, rowsC,
            N);
    }
    cublasDestroy(cnpHandle);
    thrust::constant_iterator<int> fitt(10);
    //thrust_dev_float gradients(batch_size );
}
#include<vector>
void useCUDA()
{
    thrust::device_vector<int> d_data;
    thrust::host_vector<int> h_data;
    for (int i = 0; i < 5; i++)
    {
        h_data.push_back(i);
    }
    d_data = h_data;
    thrust::device_ptr<int> label_P = d_data.data();

    int num_nodes = h_data.size();
    int summ = thrust::reduce(label_P, label_P + num_nodes);//pdfs = sum(pdf_strategy)
    printf("summary:%d\n", summ);
    batch_gemm_test();
    foo << <1, 5 >> > ();
    CHECK(cudaDeviceSynchronize());

}

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
        __device__ __host__ LVCSubspaceInfoCopy(BDPTVertex* v, int* subspaceId, float* weight) :v(v), subspaceId(subspaceId), weight(weight)
        {

        }
        __device__ __host__ void operator()(int i)
        { 
            subspaceId[i] = v[i].subspaceId;
            float res = float3weight(v[i].flux) / v[i].pdf;
            res = isinf(res) ? 0 : res;
            //weight[i] = 1;
            weight[i] =  isnan(res) ? 0 : res;

            //if (weight[i] > 10000)
            //    weight[i] = 10000;
          //  printf("vinfo %d %d %f\n", i, v[i].depth, weight[i] / v[i].pdf);
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
                for (int k = 0; k < v[i].depth; k++)
                {
                    const MaterialData::Pbr& mat = mats[v[i - k].materialId];
                    if (max(mat.metallic, mat.trans) < 0.9 || mat.roughness > 0.4)
                    {
                        return false;
                    }
                    break;
                }
                return true;
            }
            return false;
        }
    };
    SubspaceSampler LVC_Process(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange)
    {
        SubspaceSampler sampler;
        thrust_dev_bool d_validState(validState, validState + countRange);
        thrust_host_bool h_validState = d_validState; 
        //thrust::host_vector<BDPTVertex> h_vertices(vertices, vertices + countRange);
        static thrust_dev_int d_Vsubspace_info(countRange);
        static thrust_host_int h_Vsubspace_info(countRange);
        static thrust_dev_float d_weight(countRange);
        static thrust_host_float h_weight(countRange);
        
        int valid_count = thrust::count_if(validState, validState + countRange, identical_transform<bool>());
          
        //copy necessary info
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data())
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
            num_subspace_vertex[subspace] += 1;
            Q_subspace_vertex[subspace] += h_weight[i]; 
            sparse_jump_vector[subspace].push_back(i);
            sparse_pmf_vector[subspace].push_back(h_weight[i]);
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

            for (int j = 0; j < subspace.size; j++)
            {
                h_jump[acc_pointer] = sparse_jump_vector[i][j];
                h_cmf[acc_pointer] = sparse_pmf_vector[i][j] / subspace.sum_pmf;
                acc_pointer++;
            }    

        }
        ans_cmf = h_cmf;
        //printf("sampler count %d %d\n", ans_cmf.size(), h_cmf.size());
        ans_jump = h_jump;
        ans_subspace = h_subspace;

        sampler.vertex_count = valid_count;
        sampler.path_count = thrust::count_if(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + countRange,
            is_path_begin(thrust::raw_pointer_cast(vertices), thrust::raw_pointer_cast(validState)));
        //sampler.path_count = 100000;
        printf("path count %d\n", sampler.path_count);
        sampler.jump_buffer = thrust::raw_pointer_cast(ans_jump.data());
        sampler.cmfs = thrust::raw_pointer_cast(ans_cmf.data());
        sampler.subspace = thrust::raw_pointer_cast(ans_subspace.data());
        sampler.LVC = thrust::raw_pointer_cast(vertices); 
        return sampler;
    }

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
        int valid_count = thrust::count_if(validState, validState + countRange, identical_transform<bool>());
        //copy necessary info------subspace info 
        {
            thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data())
            ));
            h_Vsubspace_info = d_Vsubspace_info;
        }

        thrust_host_int h_indexes;
        for (int i = 0; i < countRange; i++)
        {
            if (!h_valid_glossy[i])
                continue; 
            
            h_indexes.push_back(i);
        }
        thrust_host_int h_subspace_vertex_count(NUM_SUBSPACE);
        thrust::fill(h_subspace_vertex_count.begin(), h_subspace_vertex_count.end(), 0);
        for (int i = 0; i < h_indexes.size(); i++)
        {
            int index = h_indexes[i];
            h_subspace_vertex_count[h_Vsubspace_info[index]] += 1;
        }
        thrust_host_int h_indexes_rearrange(h_indexes.size());
        thrust_host_int h_vertex_bias;
        {
            thrust_host_int h_subspace_vertex_count_bias(NUM_SUBSPACE);
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

        static thrust_dev_int d_indexes;
        d_indexes = h_indexes_rearrange;
        printf("glossy vertices number %d\n", h_indexes_rearrange.size());
        sampler.glossy_count = h_indexes_rearrange.size();
        sampler.glossy_index = thrust::raw_pointer_cast(d_indexes.data());

        static thrust_dev_int d_glossy_subspace_bias;
        static thrust_dev_int d_glossy_subsapce_number_vertex;
        d_glossy_subspace_bias = h_vertex_bias;
        d_glossy_subsapce_number_vertex = h_subspace_vertex_count;
        sampler.glossy_subspace_num = thrust::raw_pointer_cast(d_glossy_subsapce_number_vertex.data());
        sampler.glossy_subspace_bias = thrust::raw_pointer_cast(d_glossy_subspace_bias.data());
        return sampler;
    }

    static thrust_host_float h_Q_vec(NUM_SUBSPACE);
    static thrust_dev_float Q_vec;
    void Q_zero_handle(thrust::device_ptr<float>& Q)
    {
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            if (h_Q_vec[i] == 0)
            {
                h_Q_vec[i] = FLT_MAX;
            } 
        }
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
        acc_valid_path += path_count;
        float t = path_count / (float)(acc_valid_path);

        //copy necessary info
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + countRange,
            LVCSubspaceInfoCopy(
                thrust::raw_pointer_cast(vertices),
                thrust::raw_pointer_cast(d_Vsubspace_info.data()),
                thrust::raw_pointer_cast(d_weight.data())
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
        return acc_valid_path;
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
        printf("pretrace get %d/%d paths and %d conns\n", sample_count, acc_num_samples,acc_num_nodes);

         
        return sample_count; 
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
            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            {
                classTree::divide_weight t;
                if (eye_side)
                {
                    t.dir = h_neat_conns[j].A_dir();
                    t.normal = h_neat_conns[j].A_normal();
                    t.position = h_neat_conns[j].A_position;
                    t.weight = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;
                }
                else if(h_neat_conns[j].light_source == false)
                {
                    t.dir = h_neat_conns[j].B_dir();
                    t.normal = h_neat_conns[j].B_normal();
                    t.position = h_neat_conns[j].B_position;
                    t.weight = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;
                }
                 
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
    void sample_reweight()
    {
        thrust::host_vector<preTracePath> h_samples = neat_paths;
        thrust::host_vector<preTraceConnection> h_conns = neat_conns;

        //for(int i=0;i<1000;i++)
        //{
        //    printf("\ninfo print of traced sub-paths %d\n", i);
        //    printf("contri %f %f %f, sample pdf %f, fix pdf %f\n", h_samples[i].contri.x, h_samples[i].contri.y, h_samples[i].contri.z,h_samples[i].sample_pdf,h_samples[i].fix_pdf);
        //    for (int j = h_samples[i].begin_ind; j < h_samples[i].end_ind; j++)
        //    {
        //        printf("label %d - %d\n", h_conns[j].label_A, h_conns[j].label_B);
        //        printf("peak pdf %f\n", h_conns[j].peak_pdf); 
        //    }
        //}


        thrust::host_vector<float> weight(1920 * 1000 / 100 * 1.1);
        thrust::host_vector<int> count(1920 * 1000 / 100 * 1.1);
        for (int i = 0; i < weight.size(); i++)
        {
            weight[i] = 0;
            count[i] = 0;
        }
        for (int i = 0; i < h_samples.size(); i++)
        {
            //printf("id %d %d\n", h_samples[i].pixiv_id.x, h_samples[i].pixiv_id.y);
            int id_x = h_samples[i].pixel_id.x / 10;
            int id_y = h_samples[i].pixel_id.y / 10;
            int n_id = id_x + id_y * 192;
            float ww = float3weight(h_samples[i].contri) / h_samples[i].sample_pdf;
            if (isnan(ww) || isinf(ww))continue;
            weight[n_id] += ww;
            count[n_id]++;
        }
        for (int i = 0; i < h_samples.size(); i++)
        {
            int id_x = h_samples[i].pixel_id.x / 10;
            int id_y = h_samples[i].pixel_id.y / 10;
            int n_id = id_x + id_y * 192;
            float w = weight[n_id] / 100 + 0.1;
            h_samples[i].contri = h_samples[i].contri / w;

            if (float3weight(h_samples[i].contri/ h_samples[i].sample_pdf) > 1000)
            {
             //   h_samples[i].contri *= 1000 / float3weight(h_samples[i].contri/ h_samples[i].sample_pdf);
            }
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
            float weight = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;

            for (int j = h_neat_paths[i].begin_ind; j < h_neat_paths[i].end_ind; j++)
            { 
                int eye_id = h_neat_conns[j].label_A;
                int light_id = h_neat_conns[j].label_B; 

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
            h_frac[i] = non_normalized_caustic[i] / non_normalized_sum[i];
        }
        d_frac = h_frac;
        frac = d_frac.data();
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

            float weight = float3weight(h_neat_paths[i].contri) / h_neat_paths[i].sample_pdf;  
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
                int GammaId = eye_id * NUM_SUBSPACE + light_id; 
//                float peak_pdf = h_neat_conns[j].peak_pdf / h_Q_vec[light_id];
  //              float weight2 = weight * float3weight(h_neat_paths[i].contri) / peak_pdf;

                float weight2 = min(weight, 10.0);
                h_Gamma[GammaId] += weight2;
            }
        }
        
        if (caustic_case == false)
        {
            printf("%d / %d paths are caustic paths and deleted from the training of ordinaryGamma.\n", caustic_filter_count, h_neat_paths.size()); 
        }
        else
        {
            printf("%d / %d paths are ordinary paths and deleted from the training of causticGamma.\n", caustic_filter_count, h_neat_paths.size());
        }

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            float weightS = 0;
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
               // h_Gamma[i * NUM_SUBSPACE + j] = sqrt(h_Gamma[i * NUM_SUBSPACE + j]);

                weightS += h_Gamma[i * NUM_SUBSPACE + j];
            }
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
                h_Gamma[i * NUM_SUBSPACE + j] /= weightS;
                if (weightS <= 1e-10f)
                { 
                    h_Gamma[i * NUM_SUBSPACE + j] = 1.0 / NUM_SUBSPACE;
                }
            } 
        }
        d_gamma = h_Gamma;
        Gamma = d_gamma.data();
    }

    //optimization
    namespace MyMISAware_optimization
    {

        template <typename T>
        struct linear_index_to_row_index : public thrust::unary_function<T, T>
        {
            T C; // number of columns

            __host__ __device__
                linear_index_to_row_index(T C) : C(C) {}

            __host__ __device__
                T operator()(T i)
            {
                return i / C;
            }
        };

        template <typename T>
        struct linear_index_to_input_index : public thrust::unary_function<T, T>
        {
            T I;//input_dim
            T O;//output_dim

            __host__ __device__
                linear_index_to_input_index(T _I, T _O) : I(_I), O(_O) {}

            __host__ __device__
                T operator()(T i)
            {
                return ((i / I / O) * I) + (i % I);
            }
        };

        template <typename T>
        struct sigmoid : public thrust::unary_function<T, T>
        {
            __host__ __device__
                T operator()(T a)
            {
                return 1.0 / (1.0 + exp(-a));
            }
        };

        template <typename T>
        struct inver_sigmoid : public thrust::unary_function<T, T>
        {
            __host__ __device__
                T operator()(T a)
            {
                return -log(1.0 / a - 1);
            }
        };
        template <typename T>
        struct sigmoid_gradient_theta : public thrust::unary_function<T, T>
        {
            __host__ __device__
                T operator()(T theta)
            {
                T sig = 1.0 / (1.0 + exp(-theta));
                return sig * (1 - sig);
            }
        };



        void gradient_compute_by_vector(
            const thrust_dev_float& input_vector,
            const thrust_dev_float& error_vector,
            thrust_dev_float& gradient,
            const int batch_size,
            const int input_dim,
            const int output_dim)
        {
            thrust::transform(
                thrust::make_permutation_iterator(input_vector.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_input_index<int>(input_dim, output_dim))),
                thrust::make_permutation_iterator(input_vector.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(input_dim * output_dim * batch_size), linear_index_to_input_index<int>(input_dim, output_dim))),
                thrust::make_permutation_iterator(error_vector.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_row_index<int>(input_dim))),
                gradient.begin(),
                thrust::multiplies<float>());


        }

        void thrust_bp_test_run()
        {
            cublasHandle_t cnpHandle;
            cublasStatus_t status = cublasCreate(&cnpHandle);
            int sample_sum = 2000000;
            int batch_size = 10000;
            int num_batches = sample_sum / batch_size;
            int epoches = 20;

            int ssNum = 1000;

            int input_dim = 60;
            int output_dim = 16;




            thrust_dev_float input_vector(batch_size * input_dim);
            thrust_dev_float output_vector(batch_size * output_dim);
            thrust_dev_float gradients(batch_size * input_dim * output_dim);
            thrust::fill(input_vector.begin(), input_vector.end(), 0.115);
            thrust::fill(output_vector.begin(), output_vector.end(), 0.115);
            thrust::fill(gradients.begin(), gradients.end(), 0.115);
            for (int i = 0; i < epoches; i++)
                for (int j = 0; j < num_batches; j++)
                {
                    //break;
                    gradient_compute_by_vector(input_vector, output_vector, gradients, batch_size, input_dim, output_dim);
                }
        }
        struct optimal_E_sample_info
        {
            float weight;
            float pdf0;
            int index_begin;
            int index_end;
        };
        __host__ void calculate_row_sums(
            const int R,
            const int C,
            const thrust_dev_float& array,
            thrust_dev_float& row_sums,
            thrust_dev_int& row_indices) {
            // compute row sums by summing values with equal row indices
            thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R * C),
                array.begin(),
                row_indices.begin(),
                row_sums.begin(),
                thrust::equal_to<int>(),
                thrust::plus<float>());
        }

        void sigmoid_normalize(thrust_dev_float& theta, thrust_dev_float& E, thrust_dev_float& E_sum)
        {
            int dim_eye = E_sum.size();
            int dim_light = theta.size() / dim_eye;
            thrust::transform(theta.begin(), theta.end(), E.begin(), sigmoid<float>()); //sigmoid
            thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)),
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)) + (dim_light * dim_eye),
                E.begin(),
                thrust::make_discard_iterator(),
                E_sum.begin(),
                thrust::equal_to<int>(),
                thrust::plus<float>());//accumulate alpha

            thrust::transform(E.begin(), E.end(),
                thrust::make_permutation_iterator(
                    E_sum.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                E.begin(),
                thrust::divides<float>());

        }

        template <typename T>
        struct inver_gradient : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
                T operator()(T value, T denominator)
            {
                return -value / denominator / denominator;
            }
        };
        template <typename T>
        struct inver_gradient_res : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
                T operator()(T res, T denominator)
            {
                T value = res * denominator;
                return -value / denominator / denominator;
            }
        };


        //all in one gradient compute for E2optimal
        template <typename T>
        struct theta_gradient : public thrust::binary_function<T, T, T>
        {
            __host__ __device__
                T operator()(T normalized_sigmoid, T sigmoid_sum)
            {
                T sigmoid = normalized_sigmoid * sigmoid_sum;
                return sigmoid * (1 - sigmoid) / (sigmoid_sum);
            }
        };


        __global__ void kernel_accum(float* dst_array, int* ind_array, float* data_array)
        {
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

            dst_array[ind_array[ix]] = data_array[ix];
        }



        template<typename T>
        struct debug_print_max
        {
            debug_print_max(thrust::device_vector<T>& aa, char* hint)
            {
                thrust::host_vector<T> a = aa;
                int max_ind = 0;
                T max_value = abs(a[max_ind]);
                printf("max %s:\n", hint);
                for (int i = 0; i < a.size(); i++)
                {
                    //std::cout << a[i] << " ";
                    if (abs(a[i]) > max_value)
                    {
                        max_value = abs(a[i]);
                        max_ind = i;
                    }
                }
                std::cout << max_ind << " " << max_value;
                printf("\n");
            }
        };

        void debug_print_mean(thrust_dev_float& vv, char* hint)
        {
            thrust_host_float v = vv;
            int s = v.size();
            float sum = 0;
            for (int i = 0; i < v.size(); i++)
            {
                sum += v[i];
            }
            printf("mean of %s : %e\n", hint, sum / s);
        }
        void debug_print_float_old(thrust_dev_float& aa, char* hint)
        {
            thrust_host_float a = aa;
            printf("%s:\n", hint);
            for (int i = 0; i < a.size(); i++)
            {
                printf("%f ", a[i]);
            }
            printf("\n");
        }


        struct matrix_optimal_operator
        {
            int num_samples;
            int num_nodes;
            int num_paras;
            int dim_light;
            int dim_eye;

            //para_list
            thrust_dev_float E;
            thrust_dev_float pdfs_strategy;
            thrust_dev_float pdfs_p;
            thrust_dev_float d_pdfs;// for each path 
            thrust_dev_float d_E_each_strategy;
            thrust_dev_float dloss_de;
            thrust_dev_int key_dloss_de;
            thrust_dev_float dE;
            thrust_dev_float dE_sum;
            thrust_dev_float dEdSum;
            thrust_dev_float dloss_dtheta;
            thrust_dev_float de_dtheta_0;

            matrix_optimal_operator(int N, int n_nodes, int d_a, int d_b) :num_samples(N), num_nodes(n_nodes), dim_light(d_a), dim_eye(d_b) {
                num_paras = d_a * d_b;

                E = thrust_dev_float(num_paras);
                pdfs_strategy = thrust_dev_float(num_nodes);

                pdfs_p = thrust_dev_float(num_samples);
                d_pdfs = thrust_dev_float(num_samples);// for each path
                d_E_each_strategy = thrust_dev_float(num_nodes);

                dloss_de = thrust_dev_float(num_nodes);
                key_dloss_de = thrust_dev_int(num_nodes);
                dE = thrust_dev_float(num_paras);

                dE_sum = thrust_dev_float(dim_eye);
                dEdSum = thrust_dev_float(dim_eye * dim_light);
                dloss_dtheta = thrust_dev_float(num_paras);

                de_dtheta_0 = thrust_dev_float(dim_light * dim_eye);
                E = thrust_dev_float(num_paras);
            }
            void reset_num_nodes(int nodes)
            {
                num_nodes = nodes;
                if (num_nodes > pdfs_strategy.size())
                {
                    int n_size = num_nodes * 1.1;
                    pdfs_strategy = thrust_dev_float(n_size);
                    d_E_each_strategy = thrust_dev_float(n_size);
                    dloss_de = thrust_dev_float(n_size);
                    key_dloss_de = thrust_dev_float(n_size);

                }

            }

            thrust_dev_float& get_forward_pdfs(
                thrust_dev_float& E,
                thrust::device_ptr<float> pdf_peak,
                thrust::device_ptr<int> label_E
            )

            {
                //thrust_dev_float pdfs_strategy(pdf_peak,pdf_peak + num_nodes); 
                //static thrust_dev_float E_uniform(E.size());
                //thrust::copy(E.begin(), E.end(), E_uniform.begin());
                //thrust::transform(E_uniform.begin(), E_uniform.end(), thrust::make_constant_iterator(float(0.9)),E_uniform.begin(), thrust::multiplies<float>());
                //thrust::transform(E_uniform.begin(), E_uniform.end(), thrust::make_constant_iterator(float(0.1 / MAX_ZONE)), E_uniform.begin(), thrust::plus<float>());

                thrust::transform(
                    pdf_peak, pdf_peak + num_nodes,
                    thrust::make_permutation_iterator(E.begin(), label_E),
                    pdfs_strategy.begin(),
                    thrust::multiplies<float>());//pdfs = e * weight   
                return pdfs_strategy;
            }


            thrust_dev_float& get_loss_gradient(
                thrust_dev_float& pdfs,
                thrust::device_ptr<int> label_P,
                thrust::device_ptr<float> peak_pdf,
                thrust::device_ptr<float>loss_weight,
                thrust::device_ptr<float>pdf0)
            {
                //thrust_dev_float pdfs_p(num_samples);
                thrust::reduce_by_key(label_P, label_P + num_nodes,
                    pdfs.begin(), thrust::make_discard_iterator(), pdfs_p.begin());//pdfs = sum(pdf_strategy)

                thrust::transform(pdfs_p.begin(), pdfs_p.end(), pdf0, pdfs_p.begin(), thrust::plus<float>());//pdfs = pdf0 + pdfs_sum

                //thrust_host_float a = pdfs_p;

                //thrust_host_float b(loss_weight,loss_weight + 10000);
                //printf("error pathPdf:%e %e\n\n", a[5370],b[5370]);
                //forward finish  
                //thrust_dev_float d_pdfs(num_samples);// for each path
                thrust::transform(
                    loss_weight, loss_weight + num_samples, pdfs_p.begin(), d_pdfs.begin(), inver_gradient<float>());//dloss / dpdf = d(loss / pdf)
                //thrust_host_float c = d_pdfs;
                //printf("error path gradient:%e \n\n", c[5370]);

                return d_pdfs;

            }
            template<typename T>
            struct mod_func : public thrust::unary_function<T, T>
            {
                T C; // number of columns

                __host__ __device__
                    mod_func(T C) : C(C) {}

                __host__ __device__
                    T operator()(T i)
                {
                    return i % C;
                }
            };

            thrust_dev_float& get_dE(
                thrust_dev_float& dpdfs,
                thrust::device_ptr<int> label_E,
                thrust::device_ptr<int> label_P,
                thrust::device_ptr<float> peak_pdfs
            )
            {
                //thrust_dev_float d_E_each_strategy(num_nodes);
                thrust::transform(
                    peak_pdfs, peak_pdfs + num_nodes,
                    thrust::make_permutation_iterator(dpdfs.begin(),
                        thrust::make_transform_iterator(label_P, mod_func<int>(num_samples))
                    ),
                    d_E_each_strategy.begin(),
                    thrust::multiplies<float>());//dpdf_each_strategy = dpdf......dE_strategy = dpdf_each_strategy * full_weight

                //debug_print_mean(d_E_each_strategy, "max_E_gradient");

                //thrust_dev_float dloss_de(num_nodes);
                //thrust_dev_int key_dloss_de(num_nodes);
                //thrust_dev_int sort_EA(label_E, label_E + num_nodes);


                static thrust_dev_int sort_E(0);
                if (sort_E.size() < num_nodes)
                    sort_E.resize(num_nodes * 1.1);
                thrust::copy(label_E, label_E + num_nodes, sort_E.begin());


                thrust::sort_by_key(sort_E.begin(), sort_E.begin() + num_nodes, d_E_each_strategy.begin());
                auto new_end = thrust::reduce_by_key(sort_E.begin(), sort_E.begin() + num_nodes, d_E_each_strategy.begin(), key_dloss_de.begin(), dloss_de.begin());
                //thrust_dev_float dE(num_paras);
                thrust::fill(dE.begin(), dE.end(), 0.0);

                int valid_key_size = new_end.first - key_dloss_de.begin();

                dim3 grid_accum(valid_key_size);
                dim3 block_accum(1);
                kernel_accum << <grid_accum, block_accum >> > (
                    thrust::raw_pointer_cast(dE.data()),
                    thrust::raw_pointer_cast(key_dloss_de.data()),
                    thrust::raw_pointer_cast(dloss_de.data()));//mybe it can be replaced by cusparse or csr2dense
                //now dE saves the dloss/de for e in range(1000,1000)
                return dE;
            }
            thrust_dev_float& gradient_E2theta(
                thrust_dev_float& dE,
                thrust_dev_float& E_sum,
                thrust_dev_float& E,
                thrust_dev_float& theta
            )
            {
                //thrust_dev_float dE_sum(dim_eye);
                //thrust_dev_float dEdSum(dim_eye * dim_light);
                thrust::transform(E.begin(), E.end(),
                    thrust::make_permutation_iterator(
                        E_sum.begin(),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light))),
                    dEdSum.begin(),
                    inver_gradient_res<float>());

                thrust::transform(dEdSum.begin(), dEdSum.end(), dE.begin(), dEdSum.begin(), thrust::multiplies<float>());
                thrust::reduce_by_key(
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)) + (dim_light * dim_eye),
                    dEdSum.begin(),
                    thrust::make_discard_iterator(),
                    dE_sum.begin());//accumulate dE_sum  
                //thrust::transform(dE_sum.begin(), dE_sum.end(), E_sum.begin(), dE_sum.begin(), inver_gradient<float>()); 
                //dE_sum now save dloss / dE_sum  ==dloss / sigmoid_sum


                //thrust_dev_float dloss_dtheta(num_paras);
                //begin
                thrust::transform(theta.begin(), theta.end(), dloss_dtheta.begin(), sigmoid_gradient_theta<float>());//d sigmoid / d theta
                thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(),
                    thrust::make_permutation_iterator(
                        dE_sum.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                    dloss_dtheta.begin(), thrust::multiplies<float>());
                //thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(), dE.begin(), dloss_dtheta.end(), thrust::multiplies<float>());
                //end



                //thrust_dev_float de_dtheta_0(dim_light * dim_eye);
                thrust::transform(
                    E.begin(), E.end(),
                    thrust::make_permutation_iterator(
                        E_sum.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                    de_dtheta_0.begin(),
                    theta_gradient<float>());//compute de/dtheta  step0, fuse the normalization op.      de/dtheta = de / dsigmoid * dsigmoid / dtheta



                thrust_dev_float& dloss_dtheta_0 = de_dtheta_0;
                thrust::transform(de_dtheta_0.begin(), de_dtheta_0.end(), dE.begin(), dloss_dtheta_0.begin(), thrust::multiplies<float>());//save_1 


                thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(), dloss_dtheta_0.begin(), dloss_dtheta.begin(), thrust::plus<float>());

                return dloss_dtheta;
            }
            thrust_dev_float& get_E(thrust_dev_float& theta, thrust_dev_float& E_sum)
            {
                //thrust_dev_float E(num_paras);
                thrust::transform(theta.begin(), theta.end(), E.begin(), sigmoid<float>()); //sigmoid
                thrust::reduce_by_key(
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)) + (dim_light * dim_eye),
                    E.begin(),
                    thrust::make_discard_iterator(),
                    E_sum.begin(),
                    thrust::equal_to<int>(),
                    thrust::plus<float>());//accumulate alpha

                thrust::transform(E.begin(), E.end(),
                    thrust::make_permutation_iterator(
                        E_sum.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                    E.begin(),
                    thrust::divides<float>());

                //conservative sampling
                thrust::transform(E.begin(), E.end(), thrust::make_constant_iterator(1 - CONSERVATIVE_RATE), E.begin(), thrust::multiplies<float>());
                thrust::transform(E.begin(), E.end(), thrust::make_constant_iterator(CONSERVATIVE_RATE / float(NUM_SUBSPACE)), E.begin(), thrust::plus<float>());
                return E;
            }

            thrust_dev_float& operator()(
                thrust_dev_float& theta,
                thrust::device_ptr<float>& pdf0,
                thrust::device_ptr<float>& loss_weight,
                thrust::device_ptr<float>& peak_pdf,
                thrust::device_ptr<int>& label_E,
                thrust::device_ptr<int>& label_P)
            {
                static thrust_dev_float E_sum(dim_eye);
                thrust_dev_float& E = get_E(theta, E_sum);
                thrust_dev_float& pdfs = get_forward_pdfs(E, peak_pdf, label_E);
                thrust_dev_float& dloss_gradient = get_loss_gradient(pdfs, label_P, peak_pdf, loss_weight, pdf0);
                thrust_dev_float& dE = get_dE(dloss_gradient, label_E, label_P, peak_pdf);
                thrust_dev_float& g = gradient_E2theta(dE, E_sum, E, theta);
                return g;
            }

            void get_loss(
                thrust_dev_float& theta,
                thrust::device_ptr<float>& pdf0,
                thrust::device_ptr<float>& loss_weight,
                thrust::device_ptr<float>& peak_pdf,
                thrust::device_ptr<int>& label_E,
                thrust::device_ptr<int>& label_P)
            {
                thrust_dev_float E_sum(dim_eye);
                thrust_dev_float& E = get_E(theta, E_sum);
                thrust_dev_float& pdfs = get_forward_pdfs(E, peak_pdf, label_E);

                //thrust_dev_float pdfs_p(num_samples);
                thrust::reduce_by_key(label_P, label_P + num_nodes,
                    pdfs.begin(), thrust::make_discard_iterator(), pdfs_p.begin());//pdfs = sum(pdf_strategy)
                //thrust::fill(pdfs_p.begin(), pdfs_p.end(), 0.0);
                thrust::transform(pdfs_p.begin(), pdfs_p.end(), pdf0, pdfs_p.begin(), thrust::plus<float>());//pdfs = pdf0 + pdfs_sum

                //forward finish  
                thrust_dev_float loss_cache(num_samples);
                thrust::transform(loss_weight, loss_weight + num_samples, pdfs_p.begin(), loss_cache.begin(), thrust::divides<float>());//pdfs = pdf0 + pdfs_sum
                debug_print_mean(loss_cache, "loss");

                if (true)
                {
                    thrust_host_float loss_h = loss_cache;
                    if (isnan(loss_h[0]) || isinf(loss_h[0]))
                    {
                        debug_print<float>(thrust::device_vector<float>(peak_pdf, peak_pdf + num_nodes), "var_pdf_node");
                        debug_print<float>(thrust::device_vector<float>(pdf0, pdf0 + 1), "pdf0");
                        debug_print<float>(thrust::device_vector<float>(loss_weight, loss_weight + 1), "lossweight");
                        debug_print<float>(thrust::device_vector<float>(pdfs_p.begin(), pdfs_p.begin() + 1), "pdf_sum");
                        
                    }
                }
            }
        };
        void optimal_E_train(int num_samples, int num_nodes, int dim_light, int dim_eye)//, optimal_E_sample_info* infos, float* pdfs_ptr, int* ids_ptr)
        {
            float initial_theta[] = { 1,3,5,7,9,44 ,15,74,98 };
            float initial_weight[] = { 113,0.15,48.2,11.2,11 };
            float initial_pdf0[] = { 13,15,4.82 };
            float initial_loss_weight[] = { 0.1,1.4,482 };
            int   initial_weight_id[] = { 2,5,3,2,7 };
            int   initial_path_ids[] = { 0,0,1,2,2 };

            thrust_host_float theta_host(initial_theta, initial_theta + 9);
            //int num_nodes = infos[num_samples - 1].index_end;
            thrust_dev_float theta(dim_light * dim_eye);
            thrust_dev_float E(dim_light * dim_eye);
            thrust_dev_float E_sum(dim_eye);
            thrust::fill(theta.begin(), theta.end(), 0.0);

            theta = theta_host;
            //batch_begin



            sigmoid_normalize(theta, E, E_sum);//E = softmax(theta) 

            //thrust_dev_float weight(num_nodes);//full pdf for each strategy     input
            thrust_dev_float weight(initial_weight, initial_weight + 5);//full pdf for each strategy     input
            //thrust_dev_float pdf0(num_samples);//pdf0 for each path             input
            thrust_dev_float pdf0(initial_pdf0, initial_pdf0 + 3);//pdf0 for each path             input
            //thrust_dev_float loss_weight(num_samples);//f_square                input
            thrust_dev_float loss_weight(initial_loss_weight, initial_loss_weight + 3);//f_square                input
            //thrust_dev_int   weight_ids(num_nodes);//label for each strategy    input
            thrust_dev_int   weight_ids(initial_weight_id, initial_weight_id + 5);//label for each strategy    input
            //thrust_dev_int   path_ids(num_nodes);//path id for each strategy    input
            thrust_dev_int   path_ids(initial_path_ids, initial_path_ids + 5);//path id for each strategy    input

            matrix_optimal_operator op(num_samples, num_nodes, dim_light, dim_eye);
            thrust_dev_float g0 = op(theta,
                thrust::device_ptr<float>(thrust::raw_pointer_cast(pdf0.data())),
                thrust::device_ptr<float>(thrust::raw_pointer_cast(loss_weight.data())),
                thrust::device_ptr<float>(thrust::raw_pointer_cast(weight.data())),
                thrust::device_ptr<int>(thrust::raw_pointer_cast(weight_ids.data())),
                thrust::device_ptr<int>(thrust::raw_pointer_cast(path_ids.data()))
            );
            debug_print_float(g0, "g0");


            thrust_dev_float pdfs_strategy(num_nodes);
            thrust_dev_float pdfs(num_samples);

            thrust::transform(
                weight.begin(), weight.end(),
                thrust::make_permutation_iterator(E.begin(), weight_ids.begin()),
                pdfs_strategy.begin(),
                thrust::multiplies<float>());//pdfs = e * weight

            thrust::reduce_by_key(path_ids.begin(), path_ids.end(), pdfs_strategy.begin(), thrust::make_discard_iterator(), pdfs.begin());//pdfs = sum(pdf_strategy)
            thrust::transform(pdfs.begin(), pdfs.end(), pdf0.begin(), pdfs.begin(), thrust::plus<float>());//pdfs = pdf0 + pdfs_sum
            //forward finish  

            thrust_dev_float& d_pdfs = pdfs;//share the same memory
            thrust::transform(
                loss_weight.begin(), loss_weight.end(), pdfs.begin(), d_pdfs.begin(), inver_gradient<float>());//dloss / dpdf = d(loss / pdf)
            thrust_dev_float& d_E_each_strategy = pdfs_strategy;
            thrust::transform(
                weight.begin(), weight.end(),
                thrust::make_permutation_iterator(d_pdfs.begin(), path_ids.begin()),
                d_E_each_strategy.begin(),
                thrust::multiplies<float>());//dpdf_each_strategy = dpdf......dE_strategy = dpdf_each_strategy * full_weight


            thrust_dev_float dloss_de(num_samples);
            thrust_dev_int key_dloss_de(num_samples);
            thrust::sort_by_key(weight_ids.begin(), weight_ids.end(), d_E_each_strategy.begin());
            auto new_end = thrust::reduce_by_key(weight_ids.begin(), weight_ids.end(), d_E_each_strategy.begin(), key_dloss_de.begin(), dloss_de.begin());
            thrust_dev_float dE(dim_light * dim_eye);
            thrust::fill(dE.begin(), dE.end(), 0.0);

            int valid_key_size = new_end.first - key_dloss_de.begin();

            dim3 grid_accum(valid_key_size);
            dim3 block_accum(1);
            kernel_accum << <grid_accum, block_accum >> > (
                thrust::raw_pointer_cast(dE.data()),
                thrust::raw_pointer_cast(key_dloss_de.data()),
                thrust::raw_pointer_cast(dloss_de.data()));//mybe it can be replaced by cusparse or csr2dense
            //now dE saves the dloss/de for e in range(1000,1000)





            thrust_dev_float dE_sum(dim_eye);
            thrust_dev_float dEdSum(dim_eye * dim_light);
            thrust::transform(E.begin(), E.end(),
                thrust::make_permutation_iterator(
                    E_sum.begin(),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light))),
                dEdSum.begin(),
                inver_gradient_res<float>());
            thrust::transform(dEdSum.begin(), dEdSum.end(), dE.begin(), dEdSum.begin(), thrust::multiplies<float>());
            thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)),
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)) + (dim_light * dim_eye),
                dEdSum.begin(),
                thrust::make_discard_iterator(),
                dE_sum.begin());//accumulate dE_sum  
            //thrust::transform(dE_sum.begin(), dE_sum.end(), E_sum.begin(), dE_sum.begin(), inver_gradient<float>()); 
            //dE_sum now save dloss / dE_sum  ==dloss / sigmoid_sum



            thrust_dev_float dloss_dtheta(dim_light * dim_eye);
            //begin
            thrust::transform(theta.begin(), theta.end(), dloss_dtheta.begin(), sigmoid_gradient_theta<float>());//d sigmoid / d theta
            thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(),
                thrust::make_permutation_iterator(
                    dE_sum.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                dloss_dtheta.begin(), thrust::multiplies<float>());
            //thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(), dE.begin(), dloss_dtheta.end(), thrust::multiplies<float>());
            //end



            thrust_dev_float de_dtheta_0(dim_light * dim_eye);
            thrust::transform(
                E.begin(), E.end(),
                thrust::make_permutation_iterator(
                    E_sum.begin(),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                de_dtheta_0.begin(),
                theta_gradient<float>());//compute de/dtheta  step0, fuse the normalization op.      de/dtheta = de / dsigmoid * dsigmoid / dtheta



            thrust_dev_float& dloss_dtheta_0 = de_dtheta_0;
            thrust::transform(de_dtheta_0.begin(), de_dtheta_0.end(), dE.begin(), dloss_dtheta_0.begin(), thrust::multiplies<float>());//save_1 


            thrust::transform(dloss_dtheta.begin(), dloss_dtheta.end(), dloss_dtheta_0.begin(), dloss_dtheta.begin(), thrust::plus<float>());

            debug_print_float(dloss_dtheta, "dLoss_dtheta");
            exit(0);
            return;
        }

        __global__ void sumMatrixOnGPU2D(int* MatA, int* MatB, int* MatC, int nx, int ny) {
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int idx = iy * nx + ix;

            if (ix < nx && iy < ny)
                MatC[idx] = MatA[idx] + MatB[idx];
        }
        //pad_sample: input_dim
        //pad_batch:  encoding_dim * 2 * input_dim
        __global__ void position_encoding(float* sample_ptr, float* batch_ptr, int pad_sample, int pad_batch, int batch_bias)
        {
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;//0~30
            unsigned int id = iy * 2 + ix * pad_batch;

            int r = iy / 3;//encoding position range from 1 to encodingL
            int d = iy % 3;//x or y or z

            unsigned int id2 = d + ix * pad_sample + batch_bias;
            batch_ptr[id] = sinf(sample_ptr[id2] * (1 << r) * M_PI);
            batch_ptr[id + 1] = cosf(sample_ptr[id2] * (1 << r) * M_PI);


        }

        __global__ void network_forward(float* input_ptr, float* output_ptr, float* Matrix, int* label_ptr, int input_dim, int output_dim)
        {
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//sample_id
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;//0~output_dim

            unsigned int output_id = ix * output_dim + iy;
            unsigned int input_id0 = ix * input_dim;
            unsigned int label_id = label_ptr[ix];
            unsigned int weight_id0 = label_id * input_dim * output_dim + iy * input_dim;

            output_ptr[output_id] = 0;
            for (int i = 0; i < input_dim; i++)
            {
                int input_id = input_id0 + i;
                int weight_id = weight_id0 + i;
                output_ptr[output_id] += Matrix[weight_id] * input_ptr[input_id];
            }

        }
        __global__ void adam_step(int t, float lr, float beta1, float beta2, float epsilon, float* theta_ptr, float* g_ptr, float* m_ptr, float* v_ptr) {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            float& theta = theta_ptr[id];
            float& g = g_ptr[id];
            float& m = m_ptr[id];
            float& v = v_ptr[id];
            float m_hat;
            float v_hat;
            m = beta1 * m + (1 - beta1) * g;
            v = beta2 * v + (1 - beta2) * (g * g);
            m_hat = m / (1 - powf(beta1, t));
            v_hat = v / (1 - powf(beta2, t));
            if (isnan(m_hat / (sqrtf(v_hat) + epsilon)))
                return;

            theta -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        }

        template <typename T>
        struct adam_step_func : public thrust::unary_function<int, T>
        {
            int t;
            float lr;
            float beta1;
            float beta2;
            float epsilon;
            float* theta_ptr;
            float* g_ptr;
            float* m_ptr;
            float* v_ptr;
            __host__ __device__
                adam_step_func(int t, float lr, float beta1, float beta2, float epsilon, float* theta, float* g, float* m, float* v) :t(t), lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon),
                theta_ptr(theta), g_ptr(g), m_ptr(m), v_ptr(v) {}


            __host__ __device__
                T operator()(int id)
            {
                float theta = theta_ptr[id];
                float& g = g_ptr[id];
                float& m = m_ptr[id];
                float& v = v_ptr[id];
                float m_hat;
                float v_hat;
                m = beta1 * m + (1 - beta1) * g;
                v = beta2 * v + (1 - beta2) * (g * g);
                m_hat = m / (1 - powf(beta1, t));
                v_hat = v / (1 - powf(beta2, t));
                if (!(isnan(m_hat / (sqrtf(v_hat) + epsilon))))
                {
                    theta -= lr * m_hat / (sqrtf(v_hat) + epsilon);
                }
                //if (id == 0)
                //{
                   // printf("%f %f\n", theta_ptr[id], theta);
                //}
                return theta;
            }
        };

        float runtime_acc = 0;
        //cudaEvent_t start_scale, stop_scale;
        //create_events_and_start(start_scale, stop_scale);
        //float matrix_scale_time = 0.0;
        //measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");
        //runtime_acc += matrix_scale_time;
        template<typename T = float>
        struct kaiming_init :public thrust::unary_function<int, T>
        {
            int m;
            kaiming_init(int m) :m(m) {}
            __device__ __host__
                T operator()(int n)
            {

                thrust::default_random_engine rng;
                thrust::random::normal_distribution<T> dist(0.0, 1.0);
                rng.discard(n);

                T a = dist(rng) * sqrt(2.0 / m);
                //printf("%f\n", a);
                return a;
            }
        };

        struct learning_parameter
        {
            thrust_dev_float data;

            thrust_dev_float m;
            thrust_dev_float v;

            int t;
            float lr;
            float beta1;
            float beta2;
            float epsilon;
            learning_parameter(int size, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8) :t(0), lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon)
            {
                data = thrust_dev_float(size);
                m = thrust_dev_float(size);
                v = thrust_dev_float(size);
                thrust::fill(m.begin(), m.end(), 0.0);
                thrust::fill(v.begin(), v.end(), 0.0);
                thrust::fill(data.begin(), data.end(), 0.0);

            }
            void resize(int s)
            {
                data.resize(s);
                m.resize(s);
                v.resize(s);
                thrust::fill(m.begin(), m.end(), 0.0);
                thrust::fill(v.begin(), v.end(), 0.0);
                thrust::fill(data.begin(), data.end(), 0.0);
            }
            void minimize(thrust_dev_float& g)
            {

                t += 1;
                thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + data.size(),
                    data.begin(),
                    adam_step_func<float>(t, lr, beta1, beta2, epsilon,
                        thrust::raw_pointer_cast(data.data()),
                        thrust::raw_pointer_cast(g.data()),
                        thrust::raw_pointer_cast(m.data()),
                        thrust::raw_pointer_cast(v.data())));

                //dim3 grid(data.size());
                //dim3 block(1);
                //adam_step << <grid, block >> > (t, lr, beta1, beta2, epsilon,
                //    thrust::raw_pointer_cast(data.data()),
                //    thrust::raw_pointer_cast(g.data()),
                //    thrust::raw_pointer_cast(m.data()),
                //    thrust::raw_pointer_cast(v.data()));
            }


        };

        struct matrix_parameter :public learning_parameter
        {
            int dim_eye;
            int dim_light;
            matrix_parameter(int dim_eye, int dim_light, float lr = 0.01, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8) :
                dim_eye(dim_eye), dim_light(dim_light),
                learning_parameter(dim_eye* dim_light, lr, beta1, beta2, epsilon) {}
            void initial_with_inver_sigmoid(thrust::device_ptr<float> p, int size = 0)
            {
                if (size == 0)
                {
                    size = dim_eye * dim_light;
                }

                data.resize(size);
                thrust::transform(p, p + size, data.begin(), inver_sigmoid<float>());
                //data = thrust_dev_float(p, p + size);
            }
            void toE(thrust_dev_float& E)
            {
                E.resize(data.size());
                thrust_dev_float E_sum(dim_eye);
                thrust::transform(data.begin(), data.end(), E.begin(), sigmoid<float>()); //sigmoid
                thrust::reduce_by_key(
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(dim_light)) + (dim_light * dim_eye),
                    E.begin(),
                    thrust::make_discard_iterator(),
                    E_sum.begin(),
                    thrust::equal_to<int>(),
                    thrust::plus<float>());//accumulate alpha

                thrust::transform(E.begin(), E.end(),
                    thrust::make_permutation_iterator(
                        E_sum.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(dim_light))),
                    E.begin(),
                    thrust::divides<float>());
            }

            struct train_data
            {
                int N_path;
                int M_node;
                thrust::device_ptr<float> f_square; // N
                thrust::device_ptr<float> pdf_0;    // N
                thrust::device_ptr<float> pdf_peak;    // M
                thrust::device_ptr<int> label_E;    // M
                thrust::device_ptr<int> label_P;    // M
                int* P2N_ind;    //size N P2N_ind[i] record the begin index of path i

            };

            //记得用一个派生学习类来重写一次这个函数
            void fit(//int num_samples, int num_nodes,//N,M
                int batch_size, int epoches,
                train_data td
            )
            {
                int num_samples = td.N_path;
                int num_nodes = td.M_node;
                int num_batches = num_samples / batch_size;


                matrix_optimal_operator op(batch_size, 0, dim_eye, dim_light);

                for (int epoch = 0; epoch < epoches; epoch++)
                {

                    for (int batch = 0; batch < num_batches; batch++)
                    {
                        int bias_sample = batch * batch_size;

                        //bias_sample = 280000;
                        //int n_batch_size = 100;
                        int bias_node = td.P2N_ind[bias_sample];
                        int seg_nodes = (batch == num_batches - 1) ? num_nodes - bias_node : td.P2N_ind[bias_sample + batch_size] - bias_node;


                        op.reset_num_nodes(seg_nodes);
                        //continue;
                        thrust::device_ptr<float> pdf0_ptr = td.pdf_0 + bias_sample;
                        thrust::device_ptr<float> loss_weight_ptr = td.f_square + bias_sample;
                        thrust::device_ptr<float> pdf_peak_ptr = td.pdf_peak + bias_node;
                        thrust::device_ptr<int> label_E_ptr = td.label_E + bias_node;
                        thrust::device_ptr<int> label_P_ptr = td.label_P + bias_node;

                        thrust_dev_float& g = op(data, pdf0_ptr, loss_weight_ptr, pdf_peak_ptr, label_E_ptr, label_P_ptr); 
                        minimize(g);
                        //printf("sample_id %d-%d\n", bias_sample, bias_sample + batch_size);
                        op.get_loss(data, pdf0_ptr, loss_weight_ptr, pdf_peak_ptr, label_E_ptr, label_P_ptr);
                    }
                    printf("=============================================================================\n");
                }
            }
            //记得用一个派生学习类来重写一次这个函数
            void fit(//int num_samples, int num_nodes,//N,M
                int batch_size, int epoches,
                std::vector<float>& f_square, //size N
                std::vector<float>& pdf_0,    //size N

                std::vector<float>& pdf_peak, //size M
                std::vector<int>& label_E,  //size M
                std::vector<int>& label_P,  //size M
                std::vector<int>& P2N_ind  //size N P2N_ind[i] record the begin index of path i
            )
            {
                int num_samples = f_square.size();
                int num_nodes = pdf_peak.size();
                int num_batches = num_samples / batch_size;
                thrust_dev_float f_square_dev(f_square.begin(), f_square.end());
                thrust_dev_float pdf_0_dev(pdf_0.begin(), pdf_0.end());
                thrust_dev_float pdf_peak_dev(pdf_peak.begin(), pdf_peak.end());
                thrust_dev_int label_E_dev(label_E.begin(), label_E.end());
                thrust_dev_int label_P_dev(label_P.begin(), label_P.end());


                matrix_optimal_operator op(batch_size, 0, 1000, 1000);

                for (int epoch = 0; epoch < epoches; epoch++)
                {

                    for (int batch = 0; batch < num_batches; batch++)
                    {
                        int bias_sample = batch * batch_size;

                        int bias_node = P2N_ind[bias_sample];
                        int seg_nodes = (batch == num_batches - 1) ? num_nodes - bias_node : P2N_ind[batch * batch_size + batch_size] - bias_node;


                        op.reset_num_nodes(seg_nodes);
                        //continue;
                        thrust::device_ptr<float> pdf0_ptr = pdf_0_dev.data() + bias_sample;
                        thrust::device_ptr<float> loss_weight_ptr = f_square_dev.data() + bias_sample;
                        thrust::device_ptr<float> pdf_peak_ptr = pdf_peak_dev.data() + bias_node;
                        thrust::device_ptr<int> label_E_ptr = label_E_dev.data() + bias_node;
                        thrust::device_ptr<int> label_P_ptr = label_P_dev.data() + bias_node;

                        thrust_dev_float& g = op(data, pdf0_ptr, loss_weight_ptr, pdf_peak_ptr, label_E_ptr, label_P_ptr);

                        minimize(g);

                        op.get_loss(data, pdf0_ptr, loss_weight_ptr, pdf_peak_ptr, label_E_ptr, label_P_ptr);
                    }
                }
            }
        };

        template<typename T1, typename T2 = int>
        struct ptr_plus :public thrust::unary_function<T2, T1>
        {
            T1 ptr;
            __host__ __device__
                ptr_plus(T1 p) :ptr(p) {}

            __host__ __device__
                T1 operator()(T2 a)
            {
                return ptr + a;
            }
        };

        template<typename T>
        struct addr_index :public thrust::unary_function<T, T>
        {

            __host__ __device__
                T operator()(T* a)
            {
                return *a;
            }
        };

        template<typename T>
        struct mul_func : public thrust::unary_function<T, T>
        {
            T C; // number of stride

            __host__ __device__
                mul_func(T C) :C(C) {}

            __host__ __device__
                T operator()(T i)
            {
                return i * C;
            }
        };
        thrust_dev_float& gather_reduce(
            thrust_dev_int& key,//random_key
            thrust_dev_float& value,
            thrust_dev_float& res
        )
        {
            thrust_dev_int n_key(key.size());
            thrust_dev_float n_value(key.size());


            thrust::sort_by_key(key.begin(), key.end(), value.begin());
            auto new_end = thrust::reduce_by_key(key.begin(), key.end(), value.begin(), n_key.begin(), n_value.begin());



            //thrust_dev_float dE(num_paras);
            thrust::fill(res.begin(), res.end(), 0.0);

            int valid_key_size = new_end.first - n_key.begin();
            //printf("%d %d %d %d\n", valid_key_size,res.size(),n_key.size(),n_value.size());


            dim3 grid_accum(valid_key_size);
            dim3 block_accum(1);
            kernel_accum << <grid_accum, block_accum >> > (
                thrust::raw_pointer_cast(res.data()),
                thrust::raw_pointer_cast(n_key.data()),
                thrust::raw_pointer_cast(n_value.data()));//mybe it can be replaced by cusparse or csr2dense
            //now dE saves the dloss/de for e in range(1000,1000)
            //printf("%d %d %d %d\n", valid_key_size, res.size(), n_key.size(), n_value.size());
            return res;
        }

        template<typename T>
        struct dev_vector_slim
        {
            thrust::device_vector<T> v;
            int begin_ind;
            int end_ind;
            dev_vector_slim()
            {
                begin_ind = 0;
                end_ind = 0;
            }
            dev_vector_slim(int size) :v(size), begin_ind(0), end_ind(size)
            {
            }
            int size()
            {
                return end_ind - begin_ind;
            }
            void check_size(int size)
            {
                if (size > v.size())
                {
                    v.resize(size);
                }
                end_ind = size;

            }
            thrust::device_ptr<T> begin()
            {
                return v.data() + begin_ind;
            }

            thrust::device_ptr<T> end()
            {
                return v.data() + end_ind;
            }
            void set_bias(int i)
            {
                begin_ind = i;
            }
            void set_slice(int j)
            {
                end_ind = begin_ind + j;
            }
            void set_slice(int i, int j)
            {
                set_bias(i);
                set_slice(j);
            }
            thrust::device_ptr<T> data()
            {
                return v.data() + begin_ind;
            }
        };

        struct network_operator
        {
            int input_dim;
            int output_dim;
            int encode_dim;
            std::vector<int> layer_size;
            thrust_dev_int matrix_size;
            std::vector<dev_vector_slim<float>> layers;
            std::vector<dev_vector_slim<float>> g_layers;

            int encoding_L;
            int num_classes;
            //int batch_size;
            cublasHandle_t cuhnd;

            dev_vector_slim<float> g_matrix;
            dev_vector_slim<float> g_matrix_batch;
            int num_nodes;
            int num_samples;
            network_operator(int num_classes, int input_dim, int output_dim, std::vector<int> layers_data, int encoding_L = 10) :
                num_classes(num_classes), input_dim(input_dim), output_dim(output_dim), encoding_L(encoding_L), encode_dim(encoding_L* input_dim * 2)
            {
                cublasCreate(&cuhnd);

                layer_size = std::vector<int>(layers_data.begin(), layers_data.end());
                std::vector<int> t_matrix_size(layer_size.size() - 1);
                for (int i = 0; i < layer_size.size() - 1; i++)
                {
                    t_matrix_size[i] = layer_size[i] * layer_size[i + 1];
                }
                matrix_size = thrust_dev_int(t_matrix_size.begin(), t_matrix_size.end());
                //layer_size.push_back(encode_dim);
                //layer_size.push_back(16);
                //layer_size.push_back(16);
                //layer_size.push_back(output_dim);

                layers.resize(layer_size.size());
                g_layers.resize(layer_size.size());
            }
            int matrix_bias(int layer_id)
            {
                int sum = 0;
                for (int i = 1; i < layer_id; i++)
                {
                    sum += num_classes * layer_size[i - 1] * layer_size[i];
                }
                return sum;
            }
            int batch_matrix_bias(int layer_id)
            {
                int sum = 0;
                for (int i = 1; i < layer_id; i++)
                {
                    sum += num_nodes * layer_size[i - 1] * layer_size[i];
                }
                return sum;
            }
            void set_nums(int samples, int nodes)
            {
                num_nodes = nodes;
                num_samples = samples;
                for (int i = 0; i < layers.size(); i++)
                {
                    int t_temple_size = max(num_nodes, 15000);
                    layers[i].check_size(layer_size[i] * num_nodes);

                    if (i > 0)
                        g_layers[i].check_size(layer_size[i] * num_nodes);
                }
                g_matrix.check_size(matrix_bias(layer_size.size()));
                g_matrix_batch.check_size(batch_matrix_bias(layer_size.size()));
            }
            thrust::device_ptr<float> batch_position_encoding(
                int batch_size,
                thrust::device_ptr<float> input_ptr
            )
            {
                dev_vector_slim<float>& ans = layers[0];
                ans.check_size(batch_size * encode_dim);
                dim3 grid(batch_size, 1);
                dim3 block(1, encoding_L * input_dim);
                position_encoding << <grid, block >> > (
                    thrust::raw_pointer_cast(input_ptr),
                    thrust::raw_pointer_cast(ans.data()),
                    input_dim, encode_dim, 0
                    );
                //cudaDeviceSynchronize();
                return ans.data();
            }
            thrust_dev_float& gemm_backward_weight(
                int batch_size,
                thrust_dev_float& input,       //original input
                int input_dim,
                thrust_dev_float& output,      //error aware
                int output_dim,
                thrust_dev_float& gradient //item we need to compute//  **cols major**
            )
            {

                thrust::transform(
                    thrust::make_permutation_iterator(input.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_input_index<int>(input_dim, output_dim))),
                    thrust::make_permutation_iterator(input.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_input_index<int>(input_dim, output_dim))) + input_dim * output_dim * batch_size,
                    thrust::make_permutation_iterator(output.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_row_index<int>(input_dim))),
                    gradient.begin(),
                    thrust::multiplies<float>());

                return gradient;
            }

            void gemm_backward_matrix(
                int batch_size,
                thrust::device_ptr<float> input,       //original input
                int input_dim,
                thrust::device_ptr<float> output,      //error aware
                int output_dim,
                thrust::device_ptr<float> gradient //item we need to compute//  **cols major**
            )
            {

                thrust::transform(
                    thrust::make_permutation_iterator(input,
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_input_index<int>(input_dim, output_dim))),
                    thrust::make_permutation_iterator(input,
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_input_index<int>(input_dim, output_dim))) + input_dim * output_dim * batch_size,
                    thrust::make_permutation_iterator(output,
                        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_row_index<int>(input_dim))),
                    gradient,
                    thrust::multiplies<float>());

            }
            template<typename T>
            struct bp_index_reset : public thrust::binary_function<T, T, T>
            {
                T step;
                __host__ __device__
                    bp_index_reset(T step) : step(step) {}

                __host__ __device__
                    T operator()(T original_step, T reset_step)
                {
                    return (original_step % step) + reset_step * step;
                }
            };

            thrust_dev_float& gemm_backward_gather(
                thrust_dev_float& gradient,
                thrust_dev_int& ind,
                int single_matrix_size,
                thrust_dev_float& gather_gradient
            )
            {
                //build keys for gather
                //from ind info
                thrust_dev_int keys(gradient.size());

                thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + keys.size(),
                    thrust::make_permutation_iterator(
                        ind.begin(),
                        thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            linear_index_to_row_index<int>(single_matrix_size))),
                    keys.begin(),
                    bp_index_reset<int>(single_matrix_size));

                gather_reduce(keys, gradient, gather_gradient);
                return gather_gradient;
            }

            thrust_dev_float& batch_gemm_forward(
                int batch_size, int m, int n, int k,//n = 1 in most cases. k is dim_input and m is dim_output
                thrust_dev_float& matrix, //ssNum * m * k, 
                thrust_dev_int& ind,      //batch_size
                thrust_dev_float& input   //batch_size * k * n 
            )
            {
                static thrust_dev_float output;
                output.resize(batch_size * m * n);
                thrust_host_float alpha_array(batch_size);
                thrust::fill(alpha_array.begin(), alpha_array.end(), 1.0);


                thrust::device_vector<float*> matrix_ind(batch_size);
                thrust::device_vector<float*> input_ind(batch_size);
                thrust::device_vector<float*> output_ind(batch_size);

                thrust::transform(
                    thrust::make_transform_iterator(ind.begin(), mul_func<int>(m * k)),
                    thrust::make_transform_iterator(ind.begin(), mul_func<int>(m * k)) + batch_size,
                    matrix_ind.begin(),
                    ptr_plus<float*>(thrust::raw_pointer_cast(matrix.data())));
                thrust::transform(
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(k * n)),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(k * n)) + batch_size,
                    input_ind.begin(),
                    ptr_plus<float*>(thrust::raw_pointer_cast(input.data())));
                thrust::transform(
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(m * n)),
                    thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(m * n)) + batch_size,
                    output_ind.begin(),
                    ptr_plus<float*>(thrust::raw_pointer_cast(output.data())));

                //printf("%d %d\n", matrix.size(),1000 * m * k);
                //printf("%d %d %d\n", m, n, k);
                //printf("%d %d %d\n", matrix.size(),ind.size(),input.size());
                //debug_print<int>(ind,"index");



                cublasSgemmBatched(cuhnd, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    thrust::raw_pointer_cast(alpha_array.data()),

                    thrust::raw_pointer_cast(matrix_ind.data()), m, //A
                    thrust::raw_pointer_cast(input_ind.data()), k,  //B

                    thrust::raw_pointer_cast(alpha_array.data()),

                    thrust::raw_pointer_cast(output_ind.data()), m, //C
                    batch_size);
                cudaDeviceSynchronize();

                //debug_print<int>(ind, "index");

                return output;
            }


            struct gemm_index_setting :thrust::binary_function<int, int, float*>
            {
                float* matrix;
                float* input;
                float* output;
                float** input_ind;
                float** output_ind;
                int matrix_stride;
                int input_stride;
                int output_stride;

                __host__ __device__ __forceinline__ gemm_index_setting(
                    float* matrix, float* input, float* output, float** input_ind, float** output_ind, int matrix_stride, int input_stride, int output_stride) :
                    matrix(matrix), input(input), output(output), input_ind(input_ind), output_ind(output_ind), matrix_stride(matrix_stride), input_stride(input_stride), output_stride(output_stride)
                {

                }
                __host__ __device__ __forceinline__
                    float* operator()(int id_matrix, int id)
                {
                    input_ind[id] = input + id * input_stride;
                    output_ind[id] = output + id * output_stride;
                    return matrix + id_matrix * matrix_stride;
                }
            };
            thrust::device_ptr<float> batch_gemm_forward(
                int batch_size, int m, int n, int k,//n = 1 in most cases. k is dim_input and m is dim_output
                thrust::device_ptr<float> matrix, //ssNum * m * k, 
                thrust::device_ptr<int> ind,      //batch_size
                thrust::device_ptr<float> input,   //batch_size * k * n 
                thrust::device_ptr<float> output,   //batch_size * m * n 
                bool transpose = false
            )
            {

                //static dev_vector_slim<float> output;
                //output.check_size(batch_size * m * n);
                static thrust_host_float alpha_array;
                static thrust_host_float beta_array;
                if (alpha_array.size() < batch_size);
                {
                    alpha_array.resize(batch_size);
                    beta_array.resize(batch_size);
                    thrust::fill(alpha_array.begin(), alpha_array.end(), 1.0);
                    thrust::fill(beta_array.begin(), beta_array.end(), 0.0);
                }

                static dev_vector_slim<float*> matrix_ind;
                matrix_ind.check_size(batch_size);
                static dev_vector_slim<float*> input_ind;
                input_ind.check_size(batch_size);
                static dev_vector_slim<float*> output_ind;
                output_ind.check_size(batch_size);


                thrust::transform(ind, ind + batch_size, thrust::make_counting_iterator(0), matrix_ind.begin(),
                    gemm_index_setting(
                        thrust::raw_pointer_cast(matrix),
                        thrust::raw_pointer_cast(input),
                        thrust::raw_pointer_cast(output),
                        thrust::raw_pointer_cast(input_ind.data()),
                        thrust::raw_pointer_cast(output_ind.data()),
                        m * k, k, m));

                //printf("%d %d\n", matrix.size(),1000 * m * k);
                //printf("%d %d %d\n", m, n, k);
                //printf("%d %d %d\n", matrix.size(),ind.size(),input.size());
                //debug_print<int>(ind,"index");



                cublasSgemmBatched(cuhnd,
                    transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                    CUBLAS_OP_N, m, n, k,
                    thrust::raw_pointer_cast(alpha_array.data()),

                    thrust::raw_pointer_cast(matrix_ind.data()), transpose ? k : m,  //A
                    thrust::raw_pointer_cast(input_ind.data()), k,  //B

                    thrust::raw_pointer_cast(beta_array.data()),

                    thrust::raw_pointer_cast(output_ind.data()), m, //C
                    batch_size);
                //cudaDeviceSynchronize();

                //debug_print<int>(ind, "index");

                return output;
            }

            template<typename T>
            struct relu_kernel :thrust::unary_function<T, T>
            {
                //T* flag;

                __host__ __device__ __forceinline__ relu_kernel()
                {
                }
                __host__ __device__ __forceinline__
                    T operator()(T x)
                {
                    //flag[id] = x > 0 ? 1 : 0;
                    return x > 0 ? x : 0;
                }
            };
            void relu(thrust::device_ptr<float> array, int size)
            {
                thrust::transform(array, array + size, array, relu_kernel<float>());
            }

            template<typename T>
            struct g_relu_kernel :thrust::binary_function<T, T, T>
            {
                //T* flag;

                __host__ __device__ __forceinline__ g_relu_kernel()
                {
                }
                __host__ __device__ __forceinline__
                    T operator()(T x, T y)
                {
                    //flag[id] = x > 0 ? 1 : 0;
                    return y > 0 ? x : 0;
                }
            };
            void g_relu(thrust::device_ptr<float> g_vector, thrust::device_ptr<float> o_vector, int size)
            {
                thrust::transform(g_vector, g_vector + size, o_vector, g_vector, g_relu_kernel<float>());

            }
            thrust::device_ptr<float> forward(thrust_dev_float& matrix,
                thrust::device_ptr<int> ind,
                thrust::device_ptr<float> input)
            {
                //thrust_dev_int index_vec(ind, ind + num_nodes); 
                thrust::device_ptr<float> encoding_vec = batch_position_encoding(num_nodes, input);

                //debug_print_float(encoding_vec, "encode_layer");
                //debug_print<int>(index_vec, "index");
                //exit(0); 
                //thrust_dev_float& ans = batch_gemm_forward(num_nodes, output_dim,  1, encode_dim, matrix, index_vec, encoding_vec);
                for (int i = 1; i < layers.size(); i++)
                {
                    batch_gemm_forward(num_nodes, layer_size[i], 1, layer_size[i - 1], matrix.data() + matrix_bias(i), ind, layers[i - 1].data(), layers[i].data());

                    if (i < layers.size() - 1)
                    {
                        relu(layers[i].data(), layers[i].size());
                    }
                }
                //printf("%d\n", matrix_bias(1)); 
                //debug_print<float>(thrust_dev_float(layers[0].data(), layers[0].data() + 60), "layer_0"); 
                //debug_print<float>(thrust_dev_float(layers[1].data(), layers[1].data() + 16), "layer_1"); 
                //debug_print<float>(thrust_dev_float(layers[2].data(), layers[2].data() + 16), "layer_2"); 
                //debug_print<float>(thrust_dev_float(layers[3].data(), layers[3].data() + 32), "layer_3"); 
                //thrust::device_ptr<float> ans = batch_gemm_forward(num_nodes, output_dim,  1, encode_dim, matrix.data(), ind, encoding_vec);

                //thrust::fill(ans.begin(), ans.end(), 1);
                //thrust_dev_float gradient(num_nodes * output_dim * encode_dim);
                //gemm_backward_weight(num_nodes, encoding_vec, encode_dim, ans, output_dim, gradient);

                //debug_print<float>(gradient, "seg_gradient");
                //static thrust_dev_float n_g(matrix.size());
                //thrust_dev_float& g = gemm_backward_gather(gradient, index_vec, output_dim * encode_dim, n_g);
                //debug_print<float>(n_g, "final_gradient");

                return layers[3].data();
            }

            template<typename T>
            struct inch_max :thrust::unary_function<int, int>
            {
                int inch;
                T* data;
                __host__ __device__
                    inch_max(int inch, T* data) :inch(inch), data(data) {}
                __host__ __device__
                    int operator()(int id)
                {
                    int begin = id * inch;
                    int max_i = begin;
                    T max_v = FLT_MIN;
                    for (int i = begin; i < begin + inch; i++)
                    {
                        if (data[i] > max_v)
                        {
                            max_i = i;
                            max_v = data[i];
                        }
                    }
                    return max_i - begin;
                }
            };
            struct close_set_index :thrust::binary_function<int, int, int>
            {
                int* close_set;
                int close_num;
                __host__ __device__
                    close_set_index(int* close_set, int close_num) :close_set(close_set), close_num(close_num) { }
                __host__ __device__
                    int operator()(int id, int c_id)
                {
                    return close_set[c_id * close_num + id];
                }
            };
            void predict(thrust_dev_float& matrix, thrust::device_ptr<int >ind, thrust::device_ptr<float> input, thrust::device_ptr<int> ans, thrust::device_ptr<int> close_set)
            {
                //thrust_dev_int index_vec(ind, ind + num_nodes); 
                thrust::device_ptr<float> encoding_vec = batch_position_encoding(num_nodes, input);

                //debug_print_float(encoding_vec, "encode_layer");
                //debug_print<int>(index_vec, "index");
                //exit(0); 
                //thrust_dev_float& ans = batch_gemm_forward(num_nodes, output_dim,  1, encode_dim, matrix, index_vec, encoding_vec);
                for (int i = 1; i < layers.size(); i++)
                {
                    batch_gemm_forward(num_nodes, layer_size[i], 1, layer_size[i - 1], matrix.data() + matrix_bias(i), ind, layers[i - 1].data(), layers[i].data());

                    if (i < layers.size() - 1)
                    {
                        relu(layers[i].data(), layers[i].size());
                    }
                }
                thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + num_nodes,
                    ans, inch_max<float>(output_dim, thrust::raw_pointer_cast(layers[3].data())));
                thrust::transform(ans, ans + num_nodes, ind, ans, close_set_index(thrust::raw_pointer_cast(close_set), output_dim));
                return;
            }

            struct inver_index_next :public thrust::unary_function<int, int>
            {
                int* inver_index;
                int* inver_begin_index;
                int limit;
                __host__ __device__
                    inver_index_next(int* inver_index, int* inver_begin_index, int limit) :inver_index(inver_index), inver_begin_index(inver_begin_index), limit(limit)
                {

                }
                __host__ __device__
                    int operator()(int label)
                {
                    int i = inver_begin_index[label];
                    while (limit > inver_index[i])
                    {
                        i++;
                    }
                    //printf("%d %d %d %d\n",label,i,inver_index[i],inver_index[i-1]);
                    return i;
                }
            };

            struct gradient_gather_op :public thrust::unary_function<int, float>
            {
                float* g_matrix_batch;
                int* inver_index;
                int* inver_begin_index;
                int node_id_base;

                int num_classes;
                int num_nodes;
                int* matrix_size;
                __host__ __device__ gradient_gather_op(float* g_matrix_batch, int* inver_index, int* inver_begin_index,
                    int node_id_base, int num_classes, int num_nodes, int* matrix_size) :
                    g_matrix_batch(g_matrix_batch), inver_index(inver_index), inver_begin_index(inver_begin_index),
                    node_id_base(node_id_base), num_classes(num_classes), num_nodes(num_nodes), matrix_size(matrix_size)
                {

                }
                __host__ __device__ __forceinline__
                    float operator()(int id)
                {
                    int layer_id = 0;
                    int t = matrix_size[layer_id] * num_classes;
                    int search_base = 0;
                    while (id >= t)
                    {
                        id -= t;
                        search_base += matrix_size[layer_id] * num_nodes;
                        layer_id += 1;
                        t = matrix_size[layer_id] * num_classes;
                    }

                    int label_id = id / matrix_size[layer_id];
                    int inver_index_c = inver_begin_index[label_id];

                    int n_id = id % matrix_size[layer_id];

                    float sum = 0;
                    for (; ; inver_index_c++)
                    {
                        int node_id = inver_index[inver_index_c] - node_id_base;
                        if (node_id >= num_nodes)break;

                        sum += g_matrix_batch[search_base + node_id * matrix_size[layer_id] + n_id];

                    }
                    return sum;
                }
            };

            void backward(thrust_dev_float& matrix,
                thrust::device_ptr<int> ind,
                thrust::device_ptr<int> inver_index,
                thrust::device_ptr<int> inver_begin_index,
                int base_node_id)
            {
                for (int i = g_layers.size() - 1; i > 1; i--)
                {
                    int forward_layer_id = i;
                    int backward_layer_id = i - 1;
                    batch_gemm_forward(num_nodes, layer_size[backward_layer_id], 1, layer_size[forward_layer_id],
                        matrix.data() + matrix_bias(i), ind, g_layers[forward_layer_id].data(), g_layers[backward_layer_id].data(), true);
                    g_relu(g_layers[backward_layer_id].data(), layers[backward_layer_id].data(), layers[backward_layer_id].size());

                }

                for (int i = g_layers.size() - 1; i > 0; i--)
                {
                    int forward_layer_id = i;
                    int backward_layer_id = i - 1;

                    gemm_backward_matrix(num_nodes,
                        g_layers[forward_layer_id].data(), layer_size[forward_layer_id],
                        layers[backward_layer_id].data(), layer_size[backward_layer_id],
                        g_matrix_batch.data() + batch_matrix_bias(backward_layer_id));
                    //debug_print<float>(thrust_dev_float(
                    //    g_matrix_batch.data() + batch_matrix_bias(backward_layer_id),
                    //    g_matrix_batch.data() + batch_matrix_bias(backward_layer_id) + layer_size[backward_layer_id] * layer_size[forward_layer_id]
                    //),"g_batch");
                }
                //debug_print<int>(thrust_dev_int(inver_begin_index, inver_begin_index + 805), "A");
                thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + num_classes,
                    inver_begin_index, inver_index_next(
                        thrust::raw_pointer_cast(inver_index),
                        thrust::raw_pointer_cast(inver_begin_index),
                        base_node_id));
                //debug_print<int>(thrust_dev_int(inver_begin_index, inver_begin_index + 805),"B");
                //exit(0);
                //printf("%d\n", base_node_id);
                thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + g_matrix.size(),
                    g_matrix.begin(),
                    gradient_gather_op(
                        thrust::raw_pointer_cast(g_matrix_batch.data()),
                        thrust::raw_pointer_cast(inver_index),
                        thrust::raw_pointer_cast(inver_begin_index),
                        base_node_id, num_classes, num_nodes, thrust::raw_pointer_cast(matrix_size.data())));
                //exit(0); 
            }


            template<typename T>
            struct backward_loss_0_kernel : thrust::unary_function<int, T>
            {
                int nearby_size;      //32
                float* pdf0_ptr;      //N
                float* loss_value_ptr;//N
                int* label_eye_ptr;   //M
                int* label_light_ptr; //M
                float* input_ptr;     //M * nearby_size = M * 32
                float* gradient_ptr;  //M * nearby_size = M * 32
                float* E;             // ssNum * ssNum = 1000 * 1000
                float* peak_pdf;      //M
                int* close_set;       // ssNum * nearby_size = 1000 * 32
                int* P2N_ind;       // ssNum * nearby_size = 1000 * 32
                //id-size


                float* sigmoid_buffer;    //M * nearby_size
                float* sigmoid_sum_buffer;//M
                float* sigmoid_sum_g_buffer;//M
                int dim_light;
                int P2N_bias;

                __host__ __device__
                    backward_loss_0_kernel(
                        float* pdf0_ptr,      //N
                        float* loss_value_ptr,//N
                        int* label_eye_ptr,   //M
                        int* label_light_ptr, //M
                        float* input_ptr,     //M * nearby_size = M * 32
                        float* gradient_ptr,  //M * nearby_size = M * 32
                        float* E,             // ssNum * ssNum = 1000 * 1000
                        float* peak_pdf,      //M
                        int* close_set,       // ssNum * nearby_size = 1000 * 32
                        int* P2N_ind,       // ssNum * nearby_size = 1000 * 32
                        float* sigmoid_buffer,    //M * nearby_size
                        float* sigmoid_sum_buffer,//M
                        float* sigmoid_sum_g_buffer,//M
                        int nearby_size,
                        int dim_light, int P2N_bias) :nearby_size(nearby_size), pdf0_ptr(pdf0_ptr), loss_value_ptr(loss_value_ptr), label_eye_ptr(label_eye_ptr), label_light_ptr(label_light_ptr),
                    input_ptr(input_ptr), gradient_ptr(gradient_ptr), E(E), peak_pdf(peak_pdf),
                    close_set(close_set), P2N_ind(P2N_ind),
                    sigmoid_buffer(sigmoid_buffer), sigmoid_sum_buffer(sigmoid_sum_buffer), sigmoid_sum_g_buffer(sigmoid_sum_g_buffer),
                    dim_light(dim_light), P2N_bias(P2N_bias)
                {}


                __host__ __device__ __forceinline__
                    float sigmoid(float x)
                {
                    return 1.0 / (1.0 + exp(-x));
                }

                __host__ __device__ __forceinline__
                    float sigmoid_gradient(float sig)
                {
                    return sig * (1 - sig);
                }
                __host__ __device__
                    T operator()(int id)//id: 0~N
                {
                    float pdf_sum = pdf0_ptr[id];
                    float* E_light_fix = E + label_light_ptr[id];

                    for (int n_id = P2N_ind[id] - P2N_bias; n_id < P2N_ind[id + 1] - P2N_bias; n_id++)
                    {
                        //int n_id = id;
                        int base = n_id * nearby_size;
                        float sum = 0;
                        int* E_current = close_set + label_eye_ptr[n_id] * nearby_size;
                        float m_peak_pdf = peak_pdf[n_id];
                        float pdf_sum_n = 0.0;
                        for (int i = 0; i < nearby_size; i++)
                        {
                            float t = sigmoid(input_ptr[base + i]);

                            sum += t;
                            sigmoid_buffer[base + i] = t;
                            gradient_ptr[base + i] = m_peak_pdf * E_light_fix[dim_light * E_current[i]];
                            pdf_sum_n += t * gradient_ptr[base + i];
                        }
                        sigmoid_sum_buffer[n_id] = sum;
                        sigmoid_sum_g_buffer[n_id] = pdf_sum_n;
                        pdf_sum += pdf_sum_n / sum;
                    }

                    float gradient_0 = -loss_value_ptr[id] / (pdf_sum * pdf_sum);//dloss / dpdf
                    for (int n_id = P2N_ind[id] - P2N_bias; n_id < P2N_ind[id + 1] - P2N_bias; n_id++)
                    {
                        float sum = sigmoid_sum_buffer[n_id];
                        float g_sum = gradient_0 * (-sigmoid_sum_g_buffer[n_id] / sum / sum); // dloss/dsum = dloss/dpdf * dpdf/dsum

                        int base = n_id * nearby_size;
                        int* E_current = close_set + label_eye_ptr[n_id] * nearby_size;
                        float m_peak_pdf = peak_pdf[n_id];

                        for (int i = 0; i < nearby_size; i++)
                        {
                            float sig = sigmoid_buffer[base + i];
                            float dsig_dvalue = sigmoid_gradient(sig);
                            //dpdf_dsig = peak / sum
                            //dsum_dsig = 1
                            //dloss_dvalue = dloss/dpdf * dpdf/dsig * dsig/dvalue + dloss/dsum * dsum/dsig * dsig/dvalue
                            gradient_ptr[base + i] = dsig_dvalue * (gradient_ptr[base + i] / sum + g_sum);
                        }
                    }
                    return 0.0;
                }

            };
            thrust::device_ptr<float> rows_sum(
                thrust::device_ptr<float> matrix, thrust::device_ptr<float> C,
                int row, int col)//matrix[0 ~ row - 1],matrix[row ~ 2 * row - 1] .... matrix[(col - 1) * row ~ col * row - 1]
                //A: 1   * row
                //B: row * col - matrix
                //C: 1   * col
            {
                static thrust_dev_float A;
                if (row > A.size())
                {
                    A.resize(row);
                    thrust::fill(A.begin(), A.end(), 1.0);
                }
                //thrust::fill(C.begin(), C.begin() + col, 0.0);
                float alpha = 1;
                float beta = 0;
                cublasSgemm(cuhnd, CUBLAS_OP_N, CUBLAS_OP_N,
                    1, col, row,
                    &alpha,
                    thrust::raw_pointer_cast(A.data()), 1,
                    thrust::raw_pointer_cast(matrix), row,
                    &beta,
                    thrust::raw_pointer_cast(C), 1);
                //cudaDeviceSynchronize(); 
                //debug_print_mean(thrust_dev_float(C, C + col), "test");
                return C;
            }
#define SOFTMAX_K0 4
#define SOFTMAX_K1 0.1
            template<typename T>
            struct sigmoid_peak_op : thrust::binary_function<T, int, T>
            {
                float* input;
                float* E;
                float* peak_pdf;
                int* label_eye;
                int* label_light;
                int* close_set;
                int dim_light;
                int dim_output;
                sigmoid_peak_op(float* input, float* E, float* peak_pdf, int* label_eye, int* label_light, int* close_set, int dim_light, int dim_output) :
                    input(input), E(E), peak_pdf(peak_pdf), label_eye(label_eye), label_light(label_light), close_set(close_set), dim_light(dim_light), dim_output(dim_output) {}
                __host__ __device__ __forceinline__
                    T operator()(T x, int id)
                {
                    int bias = id % 32 == 0 ? 1 : 0;
                    float sigmoid = exp(SOFTMAX_K0 * (SOFTMAX_K1 * x + bias));//1.0 / (1.0 + exp(-x));
                    int node_id = id / dim_output;
                    int y = id % dim_output;
                    int light_id = label_light[node_id];
                    int eye_id = label_eye[node_id];
                    int n_eye_id = close_set[eye_id * dim_output + y];
                    input[id] = sigmoid;
                    return sigmoid * peak_pdf[node_id] * E[n_eye_id * dim_light + light_id];
                }
            };


            template<typename T>
            struct backward_loss0_compute_op : thrust::binary_function<T, int, T>
            {
                int* label_P;
                float* sigmoid_sum;//num_nodes
                float* peak_values;//32 * num_nodes
                float* strategy_pdfs;//num_nodes, peak_values sum / sigmoid_sum
                float* variable_pdfs;//num_samples
                float* pdf0;
                float* loss;
                int dim_output;
                int label_P_bias;

                __host__ __device__ __forceinline__
                    backward_loss0_compute_op(int* label_P, float* sigmoid_sum, float* peak_values, float* strategy_pdfs, float* variable_pdfs, float* pdf0, float* loss, int dim_output, int label_P_bias) :
                    label_P(label_P), sigmoid_sum(sigmoid_sum), peak_values(peak_values), strategy_pdfs(strategy_pdfs), variable_pdfs(variable_pdfs), pdf0(pdf0), loss(loss), dim_output(dim_output), label_P_bias(label_P_bias) {}
                __host__ __device__ __forceinline__
                    T operator()(T sigmoid, int id)
                {
                    int node_id = id / dim_output;
                    //int y = id % dim_output;
                    int path_id = label_P[node_id] - label_P_bias;

                    float pdf = pdf0[path_id] + variable_pdfs[path_id];
                    //if (path_id == 0)
                    //    printf("%f\n", pdf);
                    float dloss_dpdf = -(loss[path_id] / pdf / pdf);
                    float dpdf_dsigmoid = peak_values[id] / sigmoid;// / sigmoid_sum[node_id];
                    float dpdf_dsum = -strategy_pdfs[node_id];// / sigmoid_sum[node_id];
                    float dloss_dsigmoid = dloss_dpdf * (dpdf_dsigmoid + dpdf_dsum) / sigmoid_sum[node_id];
                    //return  peak_values[id] / sigmoid * dloss_dpdf;
                    return sigmoid * SOFTMAX_K0 * SOFTMAX_K1 * dloss_dsigmoid;


                }
            };

            template<typename T>
            struct P2N_accum :thrust::unary_function<int, T>
            {
                int* P2N_ind;
                int P2N_bias;
                float* node_pdf;

                __host__ __device__ __forceinline__ P2N_accum(int* P2N_ind, float* node_pdf, int P2N_bias) :P2N_ind(P2N_ind), node_pdf(node_pdf), P2N_bias(P2N_bias)
                {

                }
                __host__ __device__ __forceinline__
                    T operator()(int id)
                {
                    T sum = 0;
                    for (int i = P2N_ind[id] - P2N_bias; i < P2N_ind[id + 1] - P2N_bias; i++)
                    {
                        sum += node_pdf[i];
                    }
                    return sum;
                }
            };


            template<typename T>
            struct loss_eval_func :thrust::unary_function<int, T>
            {
                float* pdf0;
                float* pdf_var;
                float* loss;

                __host__ __device__ __forceinline__
                    loss_eval_func(float* pdf0, float* pdf_var, float* loss) :pdf0(pdf0), pdf_var(pdf_var), loss(loss)
                {

                }
                __host__ __device__ __forceinline__
                    T operator()(int id)
                {
                    //return pdf_var[id];
                    return loss[id] / (pdf0[id] + pdf_var[id]);
                }
            };
            thrust::device_ptr<float> backward_loss_0_thrust_ver(
                thrust::device_ptr<float> pdf0_ptr,
                thrust::device_ptr<float> loss_value_ptr,
                thrust::device_ptr<int> label_eye_ptr,   //M
                thrust::device_ptr<int> label_light_ptr, //M
                thrust::device_ptr<int> label_P_ptr, //M
                thrust::device_ptr<float> input_ptr,     //M * nearby_size = M * 32
                //float* gradient_ptr,  //M * nearby_size = M * 32
                thrust::device_ptr<float> E,             // ssNum * ssNum = 1000 * 1000
                thrust::device_ptr<float> peak_pdf,      //M
                thrust::device_ptr<int> close_set,       // ssNum * nearby_size = 1000 * 32 
                thrust::device_ptr<int> P2N_ind,       // ssNum * nearby_size = 1000 * 32 
                int nearby_size,
                int dim_light, int label_P_bias, int P2N_bias
            )
            {
                //debug_print<float>(thrust_dev_float(input_ptr, input_ptr + 32), "input");
                static dev_vector_slim<float> sigmoid_sum;
                sigmoid_sum.check_size(num_nodes);
                static dev_vector_slim<float> sigmoid_peak_values;
                sigmoid_peak_values.check_size(num_nodes * output_dim);
                static dev_vector_slim<float> sigmoid_peak_values_sum;
                sigmoid_peak_values_sum.check_size(num_nodes);
                static dev_vector_slim<float> variable_path_pdfs;
                variable_path_pdfs.check_size(num_samples);

                auto output_layer_ptr = g_layers.back().data();

                //op 1 sigmoid // 136ms
                //thrust::transform(input_ptr, input_ptr + num_nodes * output_dim, input_ptr, sigmoid<float>());
                //thrust::fill(input_ptr, input_ptr + num_nodes * output_dim, 1.0);//记得改这个


                //op 3 sigmoid * peak value 129ms  renew = 163ms when sigmoid is fused
                thrust::transform(
                    input_ptr, input_ptr + num_nodes * output_dim,
                    thrust::make_counting_iterator(0),
                    sigmoid_peak_values.begin(),
                    sigmoid_peak_op<float>(
                        thrust::raw_pointer_cast(output_layer_ptr),
                        thrust::raw_pointer_cast(E),
                        thrust::raw_pointer_cast(peak_pdf),
                        thrust::raw_pointer_cast(label_eye_ptr),
                        thrust::raw_pointer_cast(label_light_ptr),
                        thrust::raw_pointer_cast(close_set),
                        dim_light, nearby_size));

                //debug_print_mean(thrust_dev_float(input_ptr, input_ptr + num_nodes), "test");

                //op 2 sigmoid_sum 96ms
                rows_sum(output_layer_ptr, sigmoid_sum.begin(), output_dim, num_nodes);

                //debug_print<float>(thrust_dev_float(output_layer_ptr, output_layer_ptr + 32), "exp");
                //debug_print<float>(thrust_dev_float(sigmoid_sum.data(), sigmoid_sum.data() + 1), "exp_sum");

                //op 4 row sum the peak value 80ms
                rows_sum(sigmoid_peak_values.begin(), sigmoid_peak_values_sum.begin(), output_dim, num_nodes);

                //op 4.5 devide the peak pdf by sigmoid_sum 62ms

                thrust::transform(sigmoid_peak_values_sum.begin(), sigmoid_peak_values_sum.end(), sigmoid_sum.begin(), sigmoid_peak_values_sum.begin(), thrust::divides<float>());

                //op 5 reduce the variable path pdfs 185ms
                //printf("samples %d\n", num_samples);

                //thrust::reduce_by_key(label_P_ptr, label_P_ptr + num_nodes, sigmoid_peak_values_sum.begin(),thrust::make_discard_iterator(),variable_path_pdfs.begin());
                thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_samples,
                    variable_path_pdfs.begin(),
                    P2N_accum<float>(
                        thrust::raw_pointer_cast(P2N_ind),
                        thrust::raw_pointer_cast(sigmoid_peak_values_sum.data()), P2N_bias));
                //thrust::fill(variable_path_pdfs.begin(), variable_path_pdfs.end(), 10); 
                //debug_print<float>(thrust_dev_float(variable_path_pdfs.data(), variable_path_pdfs.data() + 1), "pdf_sum");
                //printf("samples %d\n", variable_path_pdfs.v.size());
                //printf("P2N_bias %d\n", P2N_bias);

                {
                    thrust_dev_float loss_compute(num_samples);
                    thrust::transform(
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(num_samples),
                        loss_compute.begin(),
                        loss_eval_func<float>(
                            thrust::raw_pointer_cast(pdf0_ptr),
                            thrust::raw_pointer_cast(variable_path_pdfs.data()),
                            thrust::raw_pointer_cast(loss_value_ptr)));

                    debug_print<float>(thrust::device_vector<float>(pdf0_ptr, pdf0_ptr + 1), "pdf0_ptr");
                    debug_print<float>(thrust::device_vector<float>(loss_value_ptr, loss_value_ptr + 1), "loss");
                    debug_print_mean(loss_compute, "stage");
                }
                //finally compute the gradient for the layer 0 174ms/110ms
                thrust::transform(output_layer_ptr, output_layer_ptr + num_nodes * output_dim,
                    thrust::make_counting_iterator(0),
                    output_layer_ptr,
                    backward_loss0_compute_op<float>(
                        thrust::raw_pointer_cast(label_P_ptr),
                        thrust::raw_pointer_cast(sigmoid_sum.data()),
                        thrust::raw_pointer_cast(sigmoid_peak_values.data()),
                        thrust::raw_pointer_cast(sigmoid_peak_values_sum.data()),
                        thrust::raw_pointer_cast(variable_path_pdfs.data()),
                        thrust::raw_pointer_cast(pdf0_ptr),
                        thrust::raw_pointer_cast(loss_value_ptr),
                        nearby_size, label_P_bias));
                //debug_print_max(thrust_dev_int(label_P_ptr, label_P_ptr + num_nodes), "max_P");
                //printf("samples %d\n", num_samples);  
                //debug_print<float>(thrust_dev_float(output_layer_ptr, output_layer_ptr + 32), "dlossdsoftmax");
                return output_layer_ptr;
            }
            thrust_dev_float& backward_loss_0(
                float* pdf0_ptr,      //N
                float* loss_value_ptr,//N
                int* label_eye_ptr,   //M
                int* label_light_ptr, //M
                float* input_ptr,     //M * nearby_size = M * 32
                //float* gradient_ptr,  //M * nearby_size = M * 32
                float* E,             // ssNum * ssNum = 1000 * 1000
                float* peak_pdf,      //M
                int* close_set,       // ssNum * nearby_size = 1000 * 32
                int* P2N_ind,
                int nearby_size,
                int dim_light, int P2N_bias)
            {
                thrust_dev_float sigmoid_buffer(num_nodes * output_dim);
                thrust_dev_float sigmoid_sum_buffer(num_nodes);
                thrust_dev_float sigmoid_sum_g_buffer(num_nodes);
                thrust_dev_float discard_buffer(num_samples);

                static thrust_dev_float gradient_buffer;
                gradient_buffer.resize(num_nodes * output_dim);


                thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_samples,
                    discard_buffer.begin(),
                    backward_loss_0_kernel<float>(
                        pdf0_ptr,      //N
                        loss_value_ptr,//N
                        label_eye_ptr,   //M
                        label_light_ptr, //M
                        input_ptr,     //M * nearby_size = M * 32
                        thrust::raw_pointer_cast(gradient_buffer.data()),  //M * nearby_size = M * 32
                        E,             // ssNum * ssNum = 1000 * 1000
                        peak_pdf,      //M
                        close_set,       // ssNum * nearby_size = 1000 * 32
                        P2N_ind,       // ssNum * nearby_size = 1000 * 32
                        thrust::raw_pointer_cast(sigmoid_buffer.data()),    //M * nearby_size
                        thrust::raw_pointer_cast(sigmoid_sum_buffer.data()),//M
                        thrust::raw_pointer_cast(sigmoid_sum_g_buffer.data()),//M
                        nearby_size,
                        dim_light, P2N_bias)
                );

                return gradient_buffer;
            }
        };
        thrust_dev_float& batch_gemm_forward_0(
            cublasHandle_t& hnd,
            int batch_size, int m, int n, int k,//n = 1 in most cases. k is dim_input and m is dim_output
            thrust_dev_float& matrix, //ssNum * m * k, 
            thrust_dev_int& ind,      //batch_size
            thrust_dev_float& input   //batch_size * k * n 
        )
        {
            static thrust_dev_float output;
            output.resize(batch_size * m * n);
            thrust_dev_float alpha_array(batch_size);
            thrust::fill(alpha_array.begin(), alpha_array.end(), 1.0);

            thrust::device_vector<float*> matrix_ind(batch_size);
            thrust::device_vector<float*> input_ind(batch_size);
            thrust::device_vector<float*> output_ind(batch_size);
            thrust::transform(
                thrust::make_transform_iterator(ind.begin(), mul_func<int>(m * k)),
                thrust::make_transform_iterator(ind.begin(), mul_func<int>(m * k)) + batch_size,
                matrix_ind.begin(),
                ptr_plus<float*>(thrust::raw_pointer_cast(matrix.data())));
            thrust::transform(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(k * n)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(k * n)) + batch_size,
                input_ind.begin(),
                ptr_plus<float*>(thrust::raw_pointer_cast(input.data())));
            thrust::transform(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(m * n)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), mul_func<int>(m * n)) + batch_size,
                output_ind.begin(),
                ptr_plus<float*>(thrust::raw_pointer_cast(output.data())));

            cublasSgemmBatched(hnd, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                thrust::raw_pointer_cast(alpha_array.data()),

                thrust::raw_pointer_cast(matrix_ind.data()), m, //A
                thrust::raw_pointer_cast(input_ind.data()), k,  //B

                thrust::raw_pointer_cast(alpha_array.data()),

                thrust::raw_pointer_cast(output_ind.data()), m, //C
                batch_size);
            return output;
        }

        struct network_parameter :public learning_parameter
        {
            int dim_light;
            int dim_eye;
            std::vector<int> layers_para;
            network_parameter(int dim_light, int dim_eye, std::vector<int> para,
                float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8) :
                dim_eye(dim_eye), dim_light(dim_light),
                layers_para(para.begin(), para.end()),
                learning_parameter(1, lr, beta1, beta2, epsilon)
            {
                int sum = 0;
                for (int i = 1; i < layers_para.size(); i++)
                {
                    sum += layers_para[i] * layers_para[i - 1];
                }
                resize(sum * dim_eye);


                int init_sum = 0;
                for (int i = 1; i < layers_para.size(); i++)
                {
                    int t = layers_para[i] * layers_para[i - 1];
                    thrust::transform(
                        thrust::make_counting_iterator(init_sum),
                        thrust::make_counting_iterator(init_sum) + t * dim_eye,
                        data.begin() + init_sum,
                        kaiming_init<float>(layers_para[i - 1]));
                    init_sum += t * dim_eye;
                }
            }

            struct train_data
            {
                int N_path;
                int M_node;
                thrust::device_ptr<float> E_;
                thrust::device_ptr<float> f_square; // N
                thrust::device_ptr<float> pdf_0;    // N
                thrust::device_ptr<float> pdf_peak;    // M
                thrust::device_ptr<float> positions;    // M
                thrust::device_ptr<int> label_eye;    // M
                thrust::device_ptr<int> label_light;    // M
                thrust::device_ptr<int> label_P;    // M
                thrust::device_ptr<int> P2N_ind_d;    // N

                thrust::device_ptr<int> close_set;    // N
                thrust::device_ptr<int> predict;
                int* P2N_ind;    //size N P2N_ind[i] record the begin index of path i
                int close_num;
                int position_dim;
            };
            void fit(//int num_samples, int num_nodes,//N,M
                int batch_size, int epoches,
                train_data td
            )
            {
                int position_dim = td.position_dim;
                int num_samples = td.N_path;
                int num_nodes = td.M_node;
                int num_batches = num_samples / batch_size;


                thrust_dev_float E(td.E_, td.E_ + dim_eye * dim_light);

                //debug_print<float>(thrust_dev_float(matrix.begin(), matrix.begin() + 60 * 16), "matrix0");

                //theta.toE(E);
                network_operator op(dim_eye, position_dim, td.close_num, layers_para);

                thrust_dev_float& matrix = data;// (data.begin(), data.end());

                inver_index_table label_eye_inver_index(1000);
                thrust::host_vector<int> label_eye_h(td.label_eye, td.label_eye + td.M_node);
                std::vector<int> label_eye_v(label_eye_h.data(), label_eye_h.data() + label_eye_h.size());
                label_eye_inver_index.build(label_eye_v);

                thrust_dev_int inver_index(label_eye_inver_index.v.begin(), label_eye_inver_index.v.end());
                thrust_dev_int begin_index(label_eye_inver_index.begin_index.begin(), label_eye_inver_index.begin_index.end());

                for (int epoch = 0; epoch < epoches; epoch++)
                {
                    thrust_dev_int var_begin_index = begin_index;
                    //printf("AAAAAAAAAA%d:\n", var_begin_index.begin() - begin_index.begin());
                    for (int batch = 0; batch < num_batches; batch++)
                    {
                        if (batch == num_batches - 1)
                        {
                            if (epoch != epoches - 1)
                            {
                                //  continue;
                            }
                        }
                        int bias_sample = batch * batch_size;

                        int bias_node = td.P2N_ind[bias_sample];
                        int seg_nodes = (batch == num_batches - 1) ? num_nodes - bias_node : td.P2N_ind[batch * batch_size + batch_size] - bias_node;

                        printf("%d-%dA %d %d\n", epoch, batch, batch_size, seg_nodes);
                        if (epoch == 0 && batch == 0)
                            op.set_nums(batch_size, seg_nodes * 1.1);
                        op.set_nums(batch_size, seg_nodes);

                        //continue;
                        thrust::device_ptr<float> pdf0_ptr = td.pdf_0 + bias_sample;
                        thrust::device_ptr<float> loss_weight_ptr = td.f_square + bias_sample;
                        thrust::device_ptr<float> pdf_peak_ptr = td.pdf_peak + bias_node;
                        thrust::device_ptr<float> positions_ptr = td.positions + bias_node * position_dim;
                        thrust::device_ptr<int> label_light_ptr = td.label_light + bias_node;
                        thrust::device_ptr<int> label_eye_ptr = td.label_eye + bias_node;
                        thrust::device_ptr<int> label_P_ptr = td.label_P + bias_node;
                        thrust::device_ptr<int> P2N_ind_ptr = td.P2N_ind_d + bias_sample;


                        thrust::device_ptr<float> E_ptr = E.data();

                        //debug_print<float>(thrust_dev_float(E_ptr + 623 * 1000, E_ptr + 624 * 1000), "optimalE");

                        cudaEvent_t start_scale, stop_scale;
                        create_events_and_start(start_scale, stop_scale);
                        float matrix_scale_time = 0.0;
                        thrust::device_ptr<float> forward_tensor = op.forward(matrix, label_eye_ptr, positions_ptr);
                        measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");
                        time_records.record(0, matrix_scale_time);
                        //printf("B \n");

                        //op.backward_loss_0(
                        //    thrust::raw_pointer_cast(pdf0_ptr),      //N
                        //    thrust::raw_pointer_cast(loss_weight_ptr),//N
                        //    thrust::raw_pointer_cast(label_eye_ptr),   //M
                        //    thrust::raw_pointer_cast(label_light_ptr), //M
                        //    thrust::raw_pointer_cast(g.data()),     //M * nearby_size = M * 32
                        //    //float* gradient_ptr,  //M * nearby_size = M * 32
                        //    thrust::raw_pointer_cast(E_ptr),             // ssNum * ssNum = 1000 * 1000
                        //    thrust::raw_pointer_cast(pdf_peak_ptr),      //M
                        //    thrust::raw_pointer_cast(close_set_dev.data()),       // ssNum * nearby_size = 1000 * 32
                        //    thrust::raw_pointer_cast(P2N_ind_ptr),       // ssNum * nearby_size = 1000 * 32 
                        //    32,
                        //    dim_light, bias_node);
                        create_events_and_start(start_scale, stop_scale);
                        matrix_scale_time = 0.0;

                        op.backward_loss_0_thrust_ver(pdf0_ptr, loss_weight_ptr, label_eye_ptr, label_light_ptr, label_P_ptr, //581ms
                            forward_tensor, E_ptr, pdf_peak_ptr, td.close_set, P2N_ind_ptr, td.close_num, dim_light, bias_sample, bias_node);

                        measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");
                        time_records.record(1, matrix_scale_time);

                        //continue;


                        create_events_and_start(start_scale, stop_scale);
                        matrix_scale_time = 0.0;

                        op.backward(matrix, label_eye_ptr, inver_index.data(), var_begin_index.data(), bias_node);

                        measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");
                        time_records.record(2, matrix_scale_time);
                        //op.get_loss(data, pdf0_ptr, loss_weight_ptr, pdf_peak_ptr, label_E_ptr, label_P_ptr);
                        //printf("t: %f\n",matrix_scale_time);

                        create_events_and_start(start_scale, stop_scale);
                        matrix_scale_time = 0.0;
                        //debug_print<float>(thrust_dev_float(
                        //    op.g_matrix.v.data() + 623 * 60 * 16,
                        //    op.g_matrix.v.data() + 624 * 60 * 16
                        //), "g");
                        //exit(0);
                        minimize(op.g_matrix.v);
                        //printf("A");

                        measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");
                        time_records.record(3, matrix_scale_time);
                    }
                    printf("\n");
                }

                //predict
                for (int batch = 0; batch < num_batches; batch++)
                {
                    int bias_sample = batch * batch_size;

                    int bias_node = td.P2N_ind[bias_sample];
                    int seg_nodes = (batch == num_batches - 1) ? num_nodes - bias_node : td.P2N_ind[batch * batch_size + batch_size] - bias_node;

                    op.set_nums(batch_size, seg_nodes);

                    //continue;
                    thrust::device_ptr<float> pdf0_ptr = td.pdf_0 + bias_sample;
                    thrust::device_ptr<float> loss_weight_ptr = td.f_square + bias_sample;
                    thrust::device_ptr<float> pdf_peak_ptr = td.pdf_peak + bias_node;
                    thrust::device_ptr<float> positions_ptr = td.positions + bias_node * position_dim;
                    thrust::device_ptr<int> label_light_ptr = td.label_light + bias_node;
                    thrust::device_ptr<int> label_eye_ptr = td.label_eye + bias_node;
                    thrust::device_ptr<int> label_P_ptr = td.label_P + bias_node;
                    thrust::device_ptr<int> P2N_ind_ptr = td.P2N_ind_d + bias_sample;


                    thrust::device_ptr<float> E_ptr = E.data();

                    //debug_print<float>(thrust_dev_float(E_ptr + 623 * 1000, E_ptr + 624 * 1000), "optimalE");

                    op.predict(matrix, label_eye_ptr, positions_ptr, td.predict + bias_node, td.close_set);

                }
                time_records.print();
            }


        };




        classTree::tree_node* device_to(classTree::tree_node* a, int size)
        {
            thrust::host_vector<classTree::tree_node> h_v(a, a + size);
            static thrust::device_vector<classTree::tree_node> d_v;
            d_v = h_v;
            return thrust::raw_pointer_cast(d_v.data());
        }
    }



     

#define optimal_E_loss_threshold (1000000.0)
    thrust::device_vector<float> b_f_square;
    thrust::device_vector<float> b_pdf0;
    thrust::device_vector<float> b_pdf_peak;
    thrust::device_vector<float> b_positions;
    thrust::device_vector<int> b_close_set;
    thrust::device_vector<int> b_label_E;
    thrust::device_vector<int> b_label_eye;
    thrust::device_vector<int> b_label_light;
    thrust::device_vector<int> b_label_P;
    thrust::device_vector<int> b_P2N_ind_d;
    thrust::host_vector<int> b_P2N_ind;
    MyMISAware_optimization::matrix_parameter::train_data E_td;
    MyMISAware_optimization::network_parameter::train_data N_td;
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

    struct construct_optimal_E_data_sample :thrust::unary_function<int, bool>
    {
        float* f_square;
        float* pdf0;
        int* P2N_ind;
        preTracePath* samples;
        construct_optimal_E_data_sample(float* f_square, float* pdf0, int* P2N_ind, preTracePath* samples) :
            f_square(f_square), pdf0(pdf0), P2N_ind(P2N_ind), samples(samples) {}
        __host__ __device__
            bool operator()(int id)
        {
            preTracePath& s = samples[id];
            float weight = float3weight(s.contri);
            f_square[id] = weight * weight / s.sample_pdf;
            if (f_square[id] > optimal_E_loss_threshold || isnan(f_square[id]))
            {
                f_square[id] = optimal_E_loss_threshold;
            }
            pdf0[id] = s.fix_pdf;
            P2N_ind[id] = s.begin_ind;
        }
    };

    struct construct_optimal_E_data_node :thrust::unary_function<int, bool>
    {
        float* peak_pdf;
        float* Q;
        int* label_E;
        int* label_P;
        preTraceConnection* nodes;
        int dim_light;
        construct_optimal_E_data_node(float* peak_pdf, float* Q, int* label_E, int* label_P, preTraceConnection* nodes, int dim_light) :
            peak_pdf(peak_pdf), Q(Q), label_E(label_E), label_P(label_P), nodes(nodes), dim_light(dim_light) {}
        __host__ __device__
            bool operator()(int id)
        {
            preTraceConnection& s = nodes[id];
            int eye_id = s.label_A;
            int light_id = s.label_B;
            label_E[id] = eye_id * dim_light + light_id;
            label_P[id] = s.path_id;
            peak_pdf[id] = Q[light_id] > 0.0 ? s.peak_pdf / Q[light_id] : 0.0; 
            if (isnan(peak_pdf[id]) || isinf(peak_pdf[id]))
            {
                peak_pdf[id] = 0;
            }
        }
    };
    struct get_outler_value :thrust::unary_function<int, float>
    {
        preTraceConnection* nodes;
        preTracePath* samples;
        float* Q;
        get_outler_value(preTracePath* samples, preTraceConnection* nodes, float* Q) :nodes(nodes), samples(samples), Q(Q) {}
        __host__ __device__
            float operator()(int id)
        {
            preTracePath& s = samples[id];
            float outler_value = s.fix_pdf;
            float loss;
            float weight = float3weight(s.contri);
            loss = weight * weight / s.sample_pdf;
            if (loss > optimal_E_loss_threshold || isnan(loss))
            {
                loss = optimal_E_loss_threshold;
            }
            for (int i = s.begin_ind; i < s.end_ind; i++)
            {
                outler_value += nodes[i].peak_pdf / Q[nodes[i].label_B] / 1000.0;
            }
            outler_value = loss / outler_value;
            return outler_value;
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
            float outler_value = s.fix_pdf;
            float loss;
            float weight = float3weight(s.contri);
            loss = weight * weight / s.sample_pdf;
            if (loss > optimal_E_loss_threshold || isnan(loss))
            {
                loss = optimal_E_loss_threshold;
            }
            for (int i = s.begin_ind; i < s.end_ind; i++)
            {
                outler_value += nodes[i].peak_pdf / Q[nodes[i].label_B] / 1000.0;
            }
            outler_value = loss / outler_value;
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
            float outler_value = s.fix_pdf;
            float loss;
            float weight = float3weight(s.contri);
            loss = weight * weight / s.sample_pdf;
            if (loss > optimal_E_loss_threshold || isnan(loss))
            {
                loss = optimal_E_loss_threshold;
            }
            for (int i = s.begin_ind; i < s.end_ind; i++)
            {
                outler_value += nodes[i].peak_pdf / Q[nodes[i].label_B] / 1000.0;
            }
            outler_value = loss / outler_value;
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
        b_label_P.resize(M_nodes);
        b_P2N_ind_d.resize(N_samples);

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
            printf("\n\nsample should be clean:%d / %d %f\n\n", irr_count, neat_paths.size(), outler_value);
        }

        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + N_samples,
            construct_optimal_E_data_sample(
                thrust::raw_pointer_cast(b_f_square.data()),
                thrust::raw_pointer_cast(b_pdf0.data()),
                thrust::raw_pointer_cast(b_P2N_ind_d.data()),
                thrust::raw_pointer_cast(neat_paths.data())));

        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + M_nodes,
            construct_optimal_E_data_node(
                thrust::raw_pointer_cast(b_pdf_peak.data()),
                thrust::raw_pointer_cast(Q_vec.data()),
                thrust::raw_pointer_cast(b_label_E.data()),
                thrust::raw_pointer_cast(b_label_P.data()),
                thrust::raw_pointer_cast(neat_conns.data()),
                Q_vec.size()));
        b_P2N_ind = b_P2N_ind_d;

        E_td.f_square = b_f_square.data();
        E_td.pdf_0 = b_pdf0.data();
        E_td.pdf_peak = b_pdf_peak.data();
        E_td.label_E = b_label_E.data();
        E_td.label_P = b_label_P.data();
        E_td.P2N_ind = b_P2N_ind.data();
        E_td.N_path = N_samples;
        E_td.M_node = M_nodes;
        printf("Q_vec size %d\n\n", Q_vec.size());


    }

    void train_optimal_E(thrust::device_ptr<float>& E_ptr)
    {
        float lr = .01;
        int epoches = 1;
         
        MyMISAware_optimization::matrix_parameter theta(NUM_SUBSPACE, NUM_SUBSPACE, lr);//lr = 0.2 batch_size = 20k may be a better choice
        thrust::device_vector<float> tmp_E_buffer(E_ptr, E_ptr + NUM_SUBSPACE * NUM_SUBSPACE); 
        theta.initial_with_inver_sigmoid(tmp_E_buffer.data());
        theta.fit(20000, epoches, E_td);
        //(contri**2)(pdf_sum ** 2) * (pdf_sum / pdf_var) = contri**2 / pdf_var / pdf_sum  ?
        //contri**2  / pdf_sum / (pdf_sum * mis_weight)
        //contri / pdf_sum
         
        theta.toE(d_E);
        Gamma_vec = d_E;
        E_ptr = Gamma_vec.data(); 

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
        printf("load E size %d\n", E_dev.size());
        Gamma = E_dev.data();

    }

    thrust::device_ptr<float> envMapCMFBuild(float* pmf, int size)
    {
        thrust::host_vector<float> p2(pmf, pmf + size);
        static thrust::device_vector<float> ans = p2;


        return ans.data();
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
            printf("Q2 %d %f\n",h_Q.size() - 1, value);
        }
        static thrust::device_vector<float> Q_dev = h_Q;
        printf("load Q size %d\n",Q_dev.size());
        Q = Q_dev.data();
    }

    thrust::device_ptr<float> Gamma2CMFGamma(thrust::device_ptr<float> Gamma, bool caustic_case)
    {
        thrust_host_float p(Gamma, Gamma + NUM_SUBSPACE * NUM_SUBSPACE);
        static thrust_dev_float d_CMFGamma;
        static thrust_dev_float d_CMFGamma_caustic;
        thrust_dev_float& d_cmf_gamma = caustic_case ? d_CMFGamma_caustic : d_CMFGamma;

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < NUM_SUBSPACE; j++)
            {
                float t = caustic_case ? 0.01 : CONSERVATIVE_RATE;
                p[i * NUM_SUBSPACE + j] = p[i * NUM_SUBSPACE + j] * (1 - t) + (1.0 / NUM_SUBSPACE) * t;
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