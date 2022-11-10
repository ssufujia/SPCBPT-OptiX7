#include <stdio.h> 
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h> 
#include <thrust/device_vector.h> 
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h> 
#include <thrust/iterator/constant_iterator.h>
#include<thrust/iterator/discard_iterator.h>
#include <thrust/random.h>
#include<thrust/sort.h>
#include "hello_cuda.h"
//#include"sutil.h"
#include<random>
#include <cuda_runtime.h>
#include <cublas_v2.h> 
#include <fstream>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using std::default_random_engine;
timerecord_stage time_records;

default_random_engine random_generator_;
float rnd_float()
{
    return float(random_generator_()) / random_generator_.max();
}
int rnd_int(int mod=1000)
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

void initialInt(int* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = rnd_int();
    }
}
__global__ void helloFromGpu()
{
    printf("hello world from GPU\n");
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
template<typename T>
T*& createHostBuffer(int w, int h = 1, int d = 1)
{
    int n_bytes = sizeof(T) * w * h * d;
    T* h_ptr = (T*)malloc(n_bytes);
    return h_ptr;
}


template<typename T>
T*& createDevBuffer(int w, int h = 1, int d = 1)
{
    int n_bytes = sizeof(T) * w * h * d;
    T* d_ptr;

    cudaMalloc((void**)&d_ptr, n_bytes);
    return d_ptr;
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

float*& createDevFloatBuffer_withInit(int w, int h = 1, int d = 1)
{
    float* h_ptr = createHostBuffer<float>(w, h, d);
    initialFloat(h_ptr, w * h * d);
    float* d_ptr = createDevBuffer_withHost<float>(h_ptr, w, h, d);
    return d_ptr;
}
std::vector<float> optimalE_refer;
void load_theta()
{

    std::ifstream inFile;
    inFile.open("./exp_record/OPTP_data/optimalE_test.txt");

    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            float t;
            inFile >> t;
            optimalE_refer.push_back(t);
        }
    }
    inFile.close();
}
void hello_print()
{
    int sample_sum = 2000000;
    int batch_size = 10000;
    int num_batches = sample_sum / batch_size;
    int epoches = 20;

    int ssNum = 1000;

    int input_dim = 60;
    int output_dim = 16;




    float* h_samples = createHostBuffer<float>(sample_sum, 3);
    float* h_weights = createHostBuffer<float>(ssNum, input_dim, output_dim);
    int* h_label = createHostBuffer<int>(batch_size);

    initialFloat(h_samples, 3 * sample_sum);
    initialFloat(h_weights, input_dim * output_dim * ssNum);
    initialInt(h_label, batch_size);

    float* d_samples = createDevBuffer_withHost(h_samples, sample_sum, 3);
    float* d_weights = createDevBuffer_withHost(h_weights, ssNum, input_dim, output_dim);
    float* d_batch = createDevBuffer<float>(batch_size, input_dim);
    float* d_res = createDevBuffer<float>(batch_size, output_dim);
    int* d_label = createDevBuffer_withHost<int>(h_label, batch_size);

    dim3 block(50, 30);
    dim3 grid(200, 1);



    dim3 block2(25, 4);
    dim3 grid2(400, 4);

    dim3 block3(50, 32);
    dim3 grid3(200, 1);

    //t = sutil::currentTime();
    //sumMatrixOnGPU2D <<< grid, block >>> (d_MatA, d_MatB, d_MatC, nx, ny);
    for (int epoch = 0; epoch < epoches; epoch++)
    {
        for (int i = 0; i < num_batches; i++)
        {
            position_encoding << <grid, block >> > (d_samples, d_batch, 3, input_dim, i * batch_size * 3);
            network_forward << < grid2, block2 >> > (d_batch, d_res, d_weights, d_label, input_dim, output_dim);
            //network_forward <<< grid2, block2 >>> (d_res, d_res_1, d_weights_1, d_label, 16, 16);
            //network_forward <<< grid3, block3 >>> (d_res_1, d_res_2, d_weights_2, d_label, 16, 32);
        }
    }
    cudaDeviceSynchronize();
    //printf("cuda code%f\n",sutil::currentTime - t);

    free(h_samples); free(h_weights); free(h_label);
    cudaFree(d_res); cudaFree(d_weights); cudaFree(d_batch); cudaFree(d_label); cudaFree(d_res);

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


typedef thrust::host_vector<float> thrust_host_float;
typedef thrust::device_vector<float> thrust_dev_float;
typedef thrust::device_vector<int> thrust_dev_int;
typedef thrust::host_vector<int> thrust_host_int;
// convert a linear index to a row index
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
        //thrust::transform(E_uniform.begin(), E_uniform.end(), thrust::make_constant_iterator(float(0.1 / SUBSPACE_NUM)), E_uniform.begin(), thrust::plus<float>());

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

        if(false)
        {
            thrust_host_float loss_h = loss_cache;
            if (isnan(loss_h[0])||isinf(loss_h[0]))
            {
                debug_print<float>(thrust::device_vector<float>(peak_pdf, peak_pdf + num_nodes), "var_pdf_node");
                debug_print<float>(thrust::device_vector<float>(pdf0, pdf0 + 1), "pdf0");
                debug_print<float>(thrust::device_vector<float>(loss_weight, loss_weight + 1), "loss");
                
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
    void initial_with_inver_sigmoid(float* p,int size = 0)
    {
        if (size == 0)
        {
            size = dim_eye * dim_light;
        }
        for (int i = 0; i < size; i++)
        {
            //printf("compare %f %f\n", p[i], sigmoid<float>()(inver_sigmoid<float>()(p[i])));
            p[i] = inver_sigmoid<float>()(p[i]);
        }
        data = thrust_dev_float(p, p + size);
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

    //记得用一个派生学习类来重写一次这个函数
    void fit(//int num_samples, int num_nodes,//N,M
        int batch_size, int epoches,
        matrix_parameter& theta,
        std::vector<float>& f_square, //size N
        std::vector<float>& pdf_0,    //size N

        std::vector<float>& pdf_peak, //size M
        std::vector<float>& positions, //size M
        std::vector<int>& label_light,  //size M
        std::vector<int>& label_eye,  //size M
        std::vector<int>& label_P,  //size M
        std::vector<int>& P2N_ind,  //size N P2N_ind[i] record the begin index of path i
        std::vector<int>& close_set //nearby_set for each class
    )
    {
        int position_dim = 3;
        int num_samples = f_square.size();
        int num_nodes = pdf_peak.size();
        int num_batches = num_samples / batch_size;
        thrust_dev_float f_square_dev(f_square.begin(), f_square.end());
        thrust_dev_float pdf_0_dev(pdf_0.begin(), pdf_0.end());
        thrust_dev_float pdf_peak_dev(pdf_peak.begin(), pdf_peak.end());
        thrust_dev_float positions_dev(positions.begin(), positions.end());
        thrust_dev_int label_light_dev(label_light.begin(), label_light.end());
        thrust_dev_int label_eye_dev(label_eye.begin(), label_eye.end());
        thrust_dev_int label_P_dev(label_P.begin(), label_P.end());
        thrust_dev_int P2N_ind_dev(P2N_ind.begin(), P2N_ind.end());
        thrust_dev_int close_set_dev(close_set.begin(), close_set.end());


        thrust_dev_float& matrix = data;// (data.begin(), data.end());
        thrust_dev_float E(optimalE_refer.begin(), optimalE_refer.end());

        //debug_print<float>(thrust_dev_float(matrix.begin(), matrix.begin() + 60 * 16), "matrix0");

        //theta.toE(E);
        network_operator op(dim_eye, 3, 32, layers_para);


        inver_index_table label_eye_inver_index(1000);
        label_eye_inver_index.build(label_eye);

        thrust_dev_int inver_index(label_eye_inver_index.v.begin(), label_eye_inver_index.v.end());
        thrust_dev_int begin_index(label_eye_inver_index.begin_index.begin(), label_eye_inver_index.begin_index.end());

        for (int epoch = 0; epoch < epoches; epoch++)
        {
            thrust_dev_int var_begin_index = begin_index;
            //printf("AAAAAAAAAA%d:\n", var_begin_index.begin() - begin_index.begin());
            for (int batch = 0; batch < num_batches; batch++)
            {
                int bias_sample = batch * batch_size;

                int bias_node = P2N_ind[bias_sample];
                int seg_nodes = (batch == num_batches - 1) ? num_nodes - bias_node : P2N_ind[batch * batch_size + batch_size] - bias_node;

                op.set_nums(batch_size, seg_nodes);
                printf("%d-%d A %d %d\n", epoch, batch, batch_size, seg_nodes);

                //continue;
                thrust::device_ptr<float> pdf0_ptr = pdf_0_dev.data() + bias_sample;
                thrust::device_ptr<float> loss_weight_ptr = f_square_dev.data() + bias_sample;
                thrust::device_ptr<float> pdf_peak_ptr = pdf_peak_dev.data() + bias_node;
                thrust::device_ptr<float> positions_ptr = positions_dev.data() + bias_node * position_dim;
                thrust::device_ptr<int> label_light_ptr = label_light_dev.data() + bias_node;
                thrust::device_ptr<int> label_eye_ptr = label_eye_dev.data() + bias_node;
                thrust::device_ptr<int> label_P_ptr = label_P_dev.data() + bias_node;
                thrust::device_ptr<int> P2N_ind_ptr = P2N_ind_dev.data() + bias_sample;


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
                    forward_tensor, E_ptr, pdf_peak_ptr, close_set_dev.data(), P2N_ind_ptr, 32, dim_light, bias_sample, bias_node);

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
        time_records.print();
    }
};
void learn_by_data(
    std::vector<float>& ans,
    std::vector<float>& f_square, //size N
    std::vector<float>& pdf_0,    //size N

    std::vector<float>& pdf_peak, //size M
    std::vector<int>& label_E,   //size M
    std::vector<int>& label_P,  //size M
    std::vector<int>& P2N_ind  //size N P2N_ind[i] record the begin index of path i
)
{
    float matrix_a[] = {
        1.0,2.0,3.0,
        4.0,5.0,6.0,
        7.0,8.0,9.0,
        1.0,2.0,3.0,
        4.0,5.0,6.0,
        7.0,8.0,9.0,
        -1,-2,-3,
        -4,-5,-6,
        -7,-8,-9,
        -1,-2,-3,
        -4,-5,-6,
        -7,-8,-9
    };
    int ind_a[] = { 0,1,1,1,0 };
    float positions_a[] = {
        -4.25876, 4.72601, -3.30085,
        -1.57423, -0.339082, -0.381991,
        -0.157321, 2.72972, 1.4245,
        0.481087, 2.24358, 1.89483,
        -2.45704, -1.155, 5.9258
    };

    //batch_gemm_forward();
    matrix_parameter theta(1000, 1000, 0.1);//lr = 0.2 batch_size = 20k may be a better choice
    theta.fit(5000, 2, f_square, pdf_0, pdf_peak, label_E, label_P, P2N_ind);
    //(contri**2)(pdf_sum ** 2) * (pdf_sum / pdf_var) = contri**2 / pdf_var / pdf_sum  ?
    //contri**2  / pdf_sum / (pdf_sum * mis_weight)
    //contri / pdf_sum

    thrust_dev_float aannss;
    theta.toE(aannss);
    thrust_host_float h_aannss = aannss;
    ans = std::vector<float>(h_aannss.begin(), h_aannss.end());
}

struct test_weight_init :thrust::unary_function<int, float>
{
    __device__ __host__
        float operator()(int id)
    {
        int layer_0 = 1000 * 16 * 60;
        int layer_1 = 1000 * 16 * 16;
        int a = 16;
        int b = 60;

        if (id >= layer_0)
        {
            id -= layer_0;
            b = 16;

            if (id >= layer_1)
            {
                id -= layer_1;
                a = 32;
            }
        }
        //id = id % (a 6* b);
        return (((25 * id) % 37) * 0.05) - 0.8;
    }
};

struct test_E_init :thrust::unary_function<int, float>
{
    __device__ __host__
        float operator()(int id)
    {
        //return (17 * id) % 101 / 100;
        //return float(id % 2) + float(id) / 1000000;
        if (id == 699254)
            return 102;
        if (id == 249494)
            return 100;
        return float(id % 5) - 5;
    }
};

void learn_by_position(
    std::vector<float>& ans,
    std::vector<float>& f_square, //size N
    std::vector<float>& pdf_0,    //size N

    std::vector<float>& pdf_peak, //size M
    std::vector<float>& positions, //size M * 3 or5
    std::vector<int>& label_light,   //size M
    std::vector<int>& label_eye,   //size M
    std::vector<int>& label_P,  //size M
    std::vector<int>& P2N_ind,  //size N P2N_ind[i] record the begin index of path i
    std::vector<int>& close_set

)
{
    load_theta();
    matrix_parameter theta(1000, 1000, 0.1);//lr = 0.2 batch_size = 20k may be a better choice
    network_parameter network(1000, 1000, std::vector<int>{60, 16, 16, 32}, 0.005);


    //thrust::fill(network.data.begin(), network.data.end(), 1.0);
    //thrust::transform(thrust::make_counting_iterator(0),
    //    thrust::make_counting_iterator(0) + network.data.size(), network.data.begin(), test_weight_init());
    thrust::transform(thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + theta.data.size(), theta.data.begin(), test_E_init());
    network.fit(10000, 7, theta, f_square, pdf_0, pdf_peak, positions, label_light, label_eye, label_P, P2N_ind, close_set);

    printf("runtime for test code :%f\n", runtime_acc);
}


classTree::tree_node* device_to(classTree::tree_node* a, int size)
{
    thrust::host_vector<classTree::tree_node> h_v(a, a + size);
    static thrust::device_vector<classTree::tree_node> d_v;
    d_v = h_v;
    return thrust::raw_pointer_cast(d_v.data());
}


MLP::nVertex* to_device(MLP::nVertex* host_ptr, int size)
{
    printf("\nAA\n");
    static thrust::device_vector<MLP::nVertex> d_v(host_ptr, host_ptr + size);
    return thrust::raw_pointer_cast(d_v.data());

}

void to_device(float* host_ptr, int size, thrust::device_vector<float>& v)
{
    v = thrust::device_vector<float>(host_ptr, host_ptr + size);
    return;
}


void to_device(MLP::nVertex* host_ptr, int size, thrust::device_vector<MLP::nVertex>& v)
{
    v = thrust::device_vector<MLP::nVertex>(host_ptr, host_ptr + size);
    return;
}



void* to_device(void* host_ptr, int size)
{
    void* device_p;
    cudaMalloc((void**)&device_p, size);
    cudaMemcpy(device_p, host_ptr, size, cudaMemcpyHostToDevice);
    return device_p;
}
void* create_temple_buffer(int byte_size)
{
    void* device_p;
    cudaMalloc((void**)&device_p, byte_size);
    return device_p;

}



namespace MLP
{
    namespace data_obtain_cudaApi
    {

        struct nVertex_valid_op
        {
            __device__ __host__ bool operator()(const nVertex& v)
            {
                return v.valid;
            }
        };

        struct nVertex_valid_LVC_op
        {
            __device__ __host__ bool operator()(const nVertex& v)
            {
                return v.valid && v.isBrdf == false;
            }
        };
        template<typename T>
        struct general_valid_op
        {
            __device__ __host__ bool operator()(const T& v)
            {
                return v.valid;
            }
        };

        struct rawLVC_valid_op
        {
            __device__ __host__ bool operator()(const RAWVertex& v)
            {
                float w = ENERGY_WEIGHT((v.v.flux / v.v.pdf));
                if (isinf(w) || isnan(w))return false; 
                return v.valid && v.v.isBrdf == false;
            }
        };
        struct raw_LVC_transform_op :thrust::unary_function<RAWVertex, nVertex>
        {

            __device__ __host__ nVertex operator()(const RAWVertex& v)
            {
                nVertex nv(v.v, false);
                nv.valid = v.valid;
                return nv;
            }
        };

        struct vertex_arrange_op :thrust::binary_function<int, nVertex, nVertex>
        {
            __device__ __host__ nVertex& operator()(int id, nVertex& v)
            {
                if (v.isLightSource())
                    v.last_id = id;
                else
                {
                    v.last_id = id - 1;
                }
                return v;
            }
        };

        struct nVertex_sample_probility_op :thrust::unary_function<nVertex, float>
        {
            __device__ __host__ float operator()(nVertex& v)
            {
                return 1.0;
            }
        };

        struct M_counting_op
        {
            __device__ __host__ bool operator()(const nVertex& v)
            {
                return v.isLightSource();
            }
        };
        thrust::device_vector<nVertex> LVC;
        thrust::device_vector<nVertex> samples;
        thrust::device_vector<float> pmf;
        thrust::device_vector<pathInfo_node> p_nodes;
        thrust::device_vector<pathInfo_sample> p_samples;
        thrust::device_vector<int> sample_bias_flag;
        int valid_samples = 0;
        data_buffer* b_p;
        void set_data_buffer(data_buffer& b)
        {
            b_p = &b;
            b.launch_seed = 2874411;// +CONTINUE_RENDERING_BEGIN_FRAME;
            b.construct_frame = 0;
            b.construct_size = b.launch_size.x * b.launch_size.y;
        }
        void buffer_pointer_validation()
        {
            data_buffer& b = *b_p;
            b.launch_seed += 1;
            b.construct_frame += 1;
            b.p_nodes.v = thrust::raw_pointer_cast(p_nodes.data());
            b.p_nodes.size = p_nodes.size();

            b.p_samples.v = thrust::raw_pointer_cast(p_samples.data());
            b.p_samples.size = p_samples.size();


            b.LVC.v = thrust::raw_pointer_cast(LVC.data());
            b.LVC.size = LVC.size();

            b.samples.v = thrust::raw_pointer_cast(samples.data());
            b.samples.size = valid_samples;

            b.cmfs.v = thrust::raw_pointer_cast(pmf.data());
            b.cmfs.size = valid_samples;
            //b.sample_M = thrust::count_if(samples.begin(), samples.begin() + valid_samples, M_counting_op());
        }
        data_buffer result_buffer_setting()
        {
            data_buffer& b = *b_p;
            int core = b.launch_size.x * b.launch_size.y;
            int pad_sample = b.res_padding.x;
            int pad_node = b.res_padding.y;
            p_nodes.resize(core * pad_node);
            p_samples.resize(core * pad_sample);
            sample_bias_flag.resize(p_nodes.size());
            b.p_nodes.v = thrust::raw_pointer_cast(p_nodes.data());
            b.p_nodes.size = p_nodes.size();

            b.p_samples.v = thrust::raw_pointer_cast(p_samples.data());
            b.p_samples.size = p_samples.size();
            return b;
        }
        struct raw_vertex2BDPTVertex :thrust::unary_function<RAWVertex, BDPTVertex>
        {
            __host__ __device__
                BDPTVertex operator()(RAWVertex& c)
            {
                return c.v;
            }
        };

        template<typename T>
        struct depth_equal_zero_op :thrust::unary_function<T, bool>
        {
            __host__ __device__
                bool operator()(const T& c)
            {
                return c.depth == 0;
            }
        };
        void LVC_process_simple(thrust::device_ptr<RAWVertex> raw_p, thrust::device_ptr<BDPTVertex> lvc_p, 
            int search_size, int& path_count, int & vertex_count, int bias)
        {              
            //static thrust::device_vector<RAWVertex> temp_vertex(200000);
            static thrust::device_vector<RAWVertex> temp_vertex(search_size * 1.1);
            if (search_size > temp_vertex.size())
            {
                temp_vertex.resize(search_size * 1.1);
            }

            auto samples_end = thrust::copy_if(raw_p, raw_p + search_size, temp_vertex.begin(), rawLVC_valid_op());
            thrust::transform(temp_vertex.begin(), samples_end, lvc_p + bias, raw_vertex2BDPTVertex());
            vertex_count += samples_end - temp_vertex.begin();

            path_count += thrust::count_if(lvc_p + bias, lvc_p + bias + vertex_count, depth_equal_zero_op<BDPTVertex>());
   
            printf("get %d path and %d vertex\n", path_count, vertex_count);
            //context["b"]->getBuffer()->getDevicePointer()
            //context["raw_LVC"]->getBuffer()->unmap();
            //cudaFree(d_pdf); cudaFree(d_samples); cudaFree(d_LVC);
        }
        data_buffer LVC_process(thrust::device_ptr<RAWVertex> p, int acc, int size)
        {
            int old_valid_samples = valid_samples;
            LVC.resize(size + acc);
            samples.resize(size + acc);
            pmf.resize(size + acc);
             
            //auto it = thrust::copy_if(thrust::host, p.p, p.p + size, raw.begin(), raw_LVC_valid_op());
            //build the sample nVertex from rawLVC
            thrust::transform(p, p + size, LVC.begin() + acc, raw_LVC_transform_op()); 
            thrust::transform(
                thrust::make_counting_iterator(acc), thrust::make_counting_iterator(size + acc),
                LVC.begin() + acc,
                LVC.begin() + acc,
                vertex_arrange_op()); 
            auto samples_end = thrust::copy_if(LVC.begin() + acc, LVC.begin() + size + acc, samples.begin() + old_valid_samples, nVertex_valid_LVC_op());
            valid_samples = samples_end - samples.begin();

            //compute pmf and pmf sum
            thrust::transform(samples.begin() , samples.begin() + valid_samples, pmf.begin(), nVertex_sample_probility_op());
            float pmf_sum = thrust::reduce(pmf.begin(), pmf.end());

            //build cmf 
            thrust::inclusive_scan(pmf.begin(), pmf.begin() + valid_samples, pmf.begin());
            thrust::transform(pmf.begin(), pmf.begin() + valid_samples,
                thrust::make_constant_iterator(pmf_sum),
                pmf.begin(), thrust::divides<float>());


            //int valid_size = it - h_raw.begin();
            //auto it = thrust::transform_if(thrust::host, p.p, p.p + size, h_LVC.begin(), raw_LVC_transform_op(), raw_LVC_valid_op()); 

            printf("size____%d %d %d %d\n\n\n\n\n", pmf.size(), valid_samples, acc, size);

            data_buffer& b = *b_p;
            b.LVC.v = thrust::raw_pointer_cast(LVC.data());
            b.LVC.size = LVC.size();

            b.samples.v = thrust::raw_pointer_cast(samples.data());
            b.samples.size = valid_samples;

            b.cmfs.v = thrust::raw_pointer_cast(pmf.data());
            b.cmfs.size = valid_samples;
            b.sample_M = thrust::count_if(samples.begin(), samples.begin() + valid_samples, M_counting_op());
            b.use_resample = false;
            return b;
            //context["b"]->getBuffer()->getDevicePointer()
            //context["raw_LVC"]->getBuffer()->unmap();
            //cudaFree(d_pdf); cudaFree(d_samples); cudaFree(d_LVC);
        }
        struct sample_valid_op :thrust::unary_function<pathInfo_sample, bool>
        {
            __device__ __host__
                bool operator()(const pathInfo_sample& s)
            {
                return s.valid;
            }
        };

        struct node_valid_op :thrust::unary_function<pathInfo_sample, bool>
        {
            __device__ __host__
                bool operator()(const pathInfo_node& s)
            {
                return s.valid;
            }
        };
        template<typename T>
        struct valid_op :thrust::unary_function<T, bool>
        {
            __device__ __host__
                bool operator()(const T& s)
            {
                return s.valid;
            }
        };

        thrust::device_vector<pathInfo_node> res_nodes;
        int acc_num_nodes = 0;
        thrust::device_vector<pathInfo_sample> res_samples;
        int acc_num_samples = 0;

        struct bias_arrange_op
        {
            pathInfo_sample* sample;
            pathInfo_node* node;
            int* bias_flag;
            int sample_bias;
            int node_bias;
            __host__ __device__
                bias_arrange_op(pathInfo_sample* sample, pathInfo_node* node, int* bias_flag, int sample_bias, int node_bias) :
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
        int valid_sample_gather()
        {
            int sample_count = thrust::count_if(p_samples.begin(), p_samples.end(), valid_op<pathInfo_sample>());
            int node_count = thrust::count_if(p_nodes.begin(), p_nodes.end(), valid_op<pathInfo_node>());
            if (acc_num_nodes + node_count > res_nodes.size())
            {
                res_nodes.resize(acc_num_nodes + node_count);
            }
            if (acc_num_samples + sample_count > res_samples.size())
            {
                res_samples.resize(acc_num_samples + sample_count);
            }
            thrust::exclusive_scan(
                thrust::make_transform_iterator(p_nodes.begin(), valid_op<pathInfo_node>()),
                thrust::make_transform_iterator(p_nodes.begin(), valid_op<pathInfo_node>()) + p_nodes.size(),
                sample_bias_flag.begin());
            thrust::copy_if(p_samples.begin(), p_samples.end(), res_samples.begin() + acc_num_samples, valid_op<pathInfo_sample>());
            thrust::copy_if(p_nodes.begin(), p_nodes.end(), res_nodes.begin() + acc_num_nodes, valid_op<pathInfo_node>());
            thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + sample_count
                , bias_arrange_op(
                    thrust::raw_pointer_cast(res_samples.data()) + acc_num_samples,
                    thrust::raw_pointer_cast(res_nodes.data() + acc_num_nodes),
                    thrust::raw_pointer_cast(sample_bias_flag.data()), acc_num_samples, acc_num_nodes));

            acc_num_nodes += node_count;
            acc_num_samples += sample_count;
            printf("get %d samples and %d nodes \n%d-%d accmulate\n\n", sample_count, node_count, acc_num_samples, acc_num_nodes);

            printf("%d\n\n", valid_samples);
            //printf("memory use %d \n\n ", res_samples.size() * sizeof(pathInfo_sample) + res_nodes.size() * sizeof(pathInfo_node));
            return acc_num_samples;
        }

        struct tree_label_op
        {
            classTree::tree_node* eye_tree;
            classTree::tree_node* light_tree;
            __device__ __host__
                tree_label_op(classTree::tree_node* eye_tree, classTree::tree_node* light_tree) :eye_tree(eye_tree), light_tree(light_tree) {}
            __device__ __host__
                bool operator()(pathInfo_node& s)
            {
                s.label_A = classTree::tree_index(eye_tree, s.A_position, s.A_normal(), s.A_dir());
                if (!s.light_source)
                    s.label_B = classTree::tree_index(light_tree, s.B_position, s.B_normal(), s.B_dir());

            }
        };
        void node_label(classTree::tree_node* eye_tree, classTree::tree_node* light_tree)
        {
            thrust::for_each(res_nodes.begin(), res_nodes.begin() + acc_num_nodes, tree_label_op(eye_tree, light_tree));
            printf("\n\nnode label complete\n\n");
        }

        struct Q_tree_label_op :thrust::unary_function<nVertex, int>
        {
            classTree::tree_node* light_tree;
            __device__ __host__
                Q_tree_label_op(classTree::tree_node* light_tree) : light_tree(light_tree) {}
            __device__ __host__
                int operator()(nVertex& s)
            {
                if (!s.isLightSource())
                {
                    s.label_id = classTree::tree_index(light_tree, s.position,s.normal,s.dir);
                }
                return s.label_id;
            }
        };
        thrust::device_vector<float> Q;
        thrust::host_vector<float> h_Q(SUBSPACE_NUM);
        std::vector<float> get_Q(classTree::tree_node* light_tree)
        {
            thrust::for_each(samples.begin(), samples.begin() + valid_samples, Q_tree_label_op(light_tree));
            thrust::host_vector<nVertex> h_v = samples;

            thrust::fill(h_Q.begin(), h_Q.end(), 0.0);
            for (auto p = h_v.begin(); p != h_v.begin() + valid_samples; p++)
            {
                float3 r = (*p).weight / (*p).pdf;
                float weight = ENERGY_WEIGHT(r);
                if (isnan(weight) || isinf(weight))continue;
                h_Q[(*p).label_id] += weight ;
                //printf("%d %e %d\n", (*p).label_id, ENERGY_WEIGHT(r), b_p->sample_M);
            }
            thrust::transform(h_Q.begin(), h_Q.end(), thrust::make_constant_iterator((float)b_p->sample_M), h_Q.begin(), thrust::divides<float>());

            Q = h_Q;
            for (int i = 0; i < h_Q.size(); i++)
            {
                //printf("Q:%d %f\n", i, h_Q[i]);
                if (h_Q[i] == 0)
                {
                    h_Q[i] = FLT_MAX;
                }
            }
            return std::vector<float>(h_Q.data(), h_Q.data() + SUBSPACE_NUM);
        }


        struct get_nVertex_label_id :thrust::unary_function<nVertex, int>
        {
            __host__ __device__ int operator()(const nVertex& s)
            {
                return s.label_id;
            }
        };
        struct compute_pmf_value_by_label:thrust::unary_function<int, float>
        {
            int* count;
            int* valid_count;
            int zone_num;
            float normalize_item;
            int valid_sum;
            int samples_sum;
            compute_pmf_value_by_label(int * count, int* valid_count,int zone_num, int valid_sum,int samples_sum, float normalize_item = 1.0):
                count(count), zone_num(zone_num), normalize_item(normalize_item),valid_count(valid_count), valid_sum(valid_sum), samples_sum(samples_sum){}
            __host__ __device__ float operator()(int s)
            {
                //return (s >= 800 ? 1.0: 0.3) / normalize_item;
                //if (s < 800)return 0;
                if (true)
                {
                    int current = valid_sum;
                    float acc_limit = 1000000.0 / 20; 
                    float scale = float(acc_limit) / current;
                    int valid = valid_count[s];
                    float valids_final = valid * scale;
                    float aver_valid = valids_final / count[s] ;

                    float optimal_item = float(valid_count[s]) / (count[s]) / current; 

                    aver_valid *= optimal_item * samples_sum;
                    float refine_scale = aver_valid ;
                    if (refine_scale > 1.0)
                    {
                        optimal_item /= refine_scale;
                    }
                    float res = optimal_item;
                    if (isnan(res)|| res == 0)
                    {
                        return 1.0 / samples_sum / 10;
                    }
                    return optimal_item / normalize_item;
                }

                bool isLightSource = s >= SUBSPACE_NUM - MAX_LIGHT;
                float t = 1.0 / 200000.0;
                float optimal_item = float(valid_count[s]) / (count[s] + 400) *(isLightSource ? 1.0 : 1.0);
                
                return optimal_item / normalize_item;
            }
        };
        struct nodes_light_label_index :thrust::unary_function<pathInfo_node, int>
        {
            classTree::tree_node* light_tree;
            nodes_light_label_index(classTree::tree_node* light_tree) :light_tree(light_tree) {}
            __host__ __device__
                int operator()(const pathInfo_node& a)
            { 
                if (!a.light_source)
                {
                    return classTree::tree_index(light_tree, a.B_position,a.B_normal(),a.B_dir());
                }
                else
                {
                    return a.label_B;
                }
            }
        };

        struct get_sample_label_id :thrust::unary_function<pathInfo_sample, int>
        {
            pathInfo_node* nodes;
            classTree::tree_node* light_tree;
            get_sample_label_id(pathInfo_node* nodes, classTree::tree_node * light_tree) :nodes(nodes), light_tree(light_tree) {}
            __host__ __device__ int operator()(pathInfo_sample& s)
            {
                if (nodes[s.begin_ind].light_source)return nodes[s.begin_ind].label_B;
                return classTree::tree_index(light_tree, nodes[s.begin_ind].B_position, nodes[s.begin_ind].B_normal(), nodes[s.begin_ind].B_dir());
            }
        };
        void pmf_reset_by_tree(classTree::tree_node* light_tree)
        {
            thrust::device_vector<int> ids(acc_num_samples);
            thrust::transform(res_samples.begin(), res_samples.end(), ids.begin(), get_sample_label_id(thrust::raw_pointer_cast(res_nodes.data()),light_tree));
            
            //thrust::sort(ids.begin(), ids.end(), thrust::greater<int>());
            //thrust::device_vector<int> sum_ids(acc_num_samples);
            //auto new_end = thrust::reduce_by_key(ids.begin(), ids.end(), thrust::make_constant_iterator(1), thrust::make_discard_iterator(), sum_ids.begin());
            //thrust::sort(sum_ids.begin(), new_end.second, thrust::greater<int>());
            //debug_print<int>(thrust::device_vector<int>(sum_ids.begin(), sum_ids.begin() + 100), "points most refer");

            //thrust::device_vector<int> d_nodes_label(acc_num_nodes);
            //thrust::transform(res_nodes.begin(), res_nodes.end(), d_nodes_label.begin(), nodes_light_label_index(light_tree));
            thrust::host_vector<int> samples_label = ids;
            thrust::host_vector<int> valid_count(SUBSPACE_NUM, 0);
            for (int i = 0; i < samples_label.size(); i++)
            {
                valid_count[samples_label[i]]++;
            }
            int reduce_sum = 0;
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                printf("valid %d %d\n", i, reduce_sum);
                reduce_sum += valid_count[i];
            }
            thrust::device_vector<int> d_valid_count = valid_count;


            //thrust::transform(samples.begin(), samples.end() + valid_samples, Q_tree_label_op(light_tree));
            thrust::for_each(samples.begin(), samples.begin() + valid_samples, Q_tree_label_op(light_tree));

            thrust::host_vector<nVertex> h_samples = samples;
            thrust::host_vector<int> counts(SUBSPACE_NUM, 0);
            for (int i = 0; i < h_samples.size(); i++)
            {
                counts[h_samples[i].label_id]++;
            }
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                printf("%d %d %d %f\n", i, counts[i],valid_count[i],float(counts[i]) / valid_count[i]);
            }
            thrust::device_vector<int> d_counts = counts;
            static thrust::device_vector<float>  d_Q_weights(d_counts.size());

            thrust::transform(
                thrust::make_transform_iterator( samples.begin(),get_nVertex_label_id()),
                thrust::make_transform_iterator( samples.begin(),get_nVertex_label_id()) + valid_samples,
                pmf.begin(),
                compute_pmf_value_by_label(thrust::raw_pointer_cast(d_counts.data()), thrust::raw_pointer_cast(d_valid_count.data()), SUBSPACE_NUM, reduce_sum, valid_samples));


            float pmf_sum = thrust::reduce(pmf.begin(), pmf.end());
            thrust::inclusive_scan(pmf.begin(), pmf.begin() + valid_samples, pmf.begin());
            thrust::transform(pmf.begin(), pmf.begin() + valid_samples,
                thrust::make_constant_iterator(pmf_sum),
                pmf.begin(), thrust::divides<float>());

            thrust::transform(
                thrust::make_counting_iterator(0), 
                thrust::make_counting_iterator(0) + d_Q_weights.size(), d_Q_weights.begin(),
                compute_pmf_value_by_label(thrust::raw_pointer_cast(d_counts.data()), thrust::raw_pointer_cast(d_valid_count.data()), SUBSPACE_NUM, reduce_sum, valid_samples, pmf_sum));

            debug_print<float>(d_Q_weights, "pdf for each label");
            data_buffer& b = *b_p;
            b.Q_weight.v = thrust::raw_pointer_cast(d_Q_weights.data());
            b.Q_weight.size = SUBSPACE_NUM;
            b.use_resample = true;
            acc_num_samples = 0;
            acc_num_nodes = 0;
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
        matrix_parameter::train_data E_td;
        network_parameter::train_data N_td;
        thrust::device_vector<float> d_E;

        struct construct_optimal_E_data_sample :thrust::unary_function<int, bool>
        {
            float* f_square;
            float* pdf0;
            int* P2N_ind;
            pathInfo_sample* samples;
            construct_optimal_E_data_sample(float* f_square, float* pdf0, int* P2N_ind, pathInfo_sample* samples) :
                f_square(f_square), pdf0(pdf0), P2N_ind(P2N_ind), samples(samples) {}
            __host__ __device__
                bool operator()(int id)
            {
                pathInfo_sample& s = samples[id];
                float weight = ENERGY_WEIGHT(s.contri);
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
            pathInfo_node* nodes;
            int dim_light;
            construct_optimal_E_data_node(float* peak_pdf, float* Q, int* label_E, int* label_P, pathInfo_node* nodes, int dim_light) :
                peak_pdf(peak_pdf), Q(Q), label_E(label_E), label_P(label_P), nodes(nodes), dim_light(dim_light) {}
            __host__ __device__
                bool operator()(int id)
            {
                pathInfo_node& s = nodes[id];
                int eye_id = s.label_A;
                int light_id = s.label_B;
                label_E[id] = eye_id * dim_light + light_id;
                label_P[id] = s.path_id;
                peak_pdf[id] = Q[light_id] > 0.0 ? s.peak_pdf / Q[light_id] : 0.0;
            }
        };
        void print_node_info(pathInfo_node& node)
        {
            printf("label   :%d-%d\n", node.label_A, node.label_B);
            printf("peak_pdf:%e\n", node.peak_pdf / h_Q[node.label_B]);
            printf("path-id :%d\n", node.path_id);
            printf("position:%f %f %f\n", node.A_position.x, node.A_position.y, node.A_position.z);
        }
        void print_sample_info(pathInfo_sample& sample)
        {
            printf("loss       :%e\n", ENERGY_WEIGHT(sample.contri));
            printf("sample_pdf :%e\n", sample.sample_pdf);
            printf("fix_pdf    :%e\n", sample.fix_pdf);
            printf("loss_weight:%e\n", ENERGY_WEIGHT(sample.contri) * ENERGY_WEIGHT(sample.contri) / sample.sample_pdf);
        }
        void view_res_data()
        {
            debug_print<float>(Q, "Q");
            thrust::host_vector<pathInfo_node> h_node = res_nodes;
            thrust::host_vector<pathInfo_sample> h_sample = res_samples;
            for (int i = 0; i < 1; i++)
            {
                printf("sample_id: %d\n", i);
                print_sample_info(h_sample[i]);
                printf("----------------------------------------\n");
                for (int j = h_sample[i].begin_ind; j < h_sample[i].end_ind; j++)
                {
                    printf("node_id: %d\n", j);
                    print_node_info(h_node[j]);
                    printf("+++++++++++++++++++++++++\n");
                }
                printf("******************************************************************\n******************************************************************\n");
            }
        }
        struct get_outler_value :thrust::unary_function<int, float>
        {
            pathInfo_node* nodes;
            pathInfo_sample* samples;
            float* Q;
            get_outler_value(pathInfo_sample* samples, pathInfo_node* nodes, float* Q) :nodes(nodes), samples(samples), Q(Q) {}
            __host__ __device__
                float operator()(int id)
            {
                pathInfo_sample& s = samples[id];
                float outler_value = s.fix_pdf;
                float loss;
                float weight = ENERGY_WEIGHT(s.contri);
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

        struct clean_outler_value :thrust::unary_function<pathInfo_sample, bool>
        {
            pathInfo_node* nodes;
            float* Q;
            float threshold;
            clean_outler_value(pathInfo_node* nodes, float* Q, float threshold) :
                nodes(nodes), Q(Q), threshold(threshold) {}
            __host__ __device__
                float operator()(pathInfo_sample& s)
            {
                float outler_value = s.fix_pdf;
                float loss;
                float weight = ENERGY_WEIGHT(s.contri);
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

        struct count_outler_value :thrust::unary_function<pathInfo_sample, bool>
        {
            pathInfo_node* nodes;
            float* Q;
            float threshold;
            count_outler_value(pathInfo_node* nodes, float* Q, float threshold) :
                nodes(nodes), Q(Q), threshold(threshold) {}
            __host__ __device__
                float operator()(const pathInfo_sample& s)
            {
                float outler_value = s.fix_pdf;
                float loss;
                float weight = ENERGY_WEIGHT(s.contri);
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

        struct get_sample_light_id:thrust::unary_function<pathInfo_sample,int>
        {
            pathInfo_node *nodes;
            get_sample_light_id(pathInfo_node* nodes) :nodes(nodes) {}
            __host__ __device__ int operator()(pathInfo_sample& s)
            {
                if (nodes[s.begin_ind].light_source)return 0;
                return s.choice_id;
            }
        };


        std::vector<classTree::VPL> get_light_cut_sample(thrust::device_ptr<BDPTVertex> lvc_p,int size)
        {
            std::vector<classTree::VPL> ans;
            thrust::host_vector<BDPTVertex> h_samples(lvc_p,lvc_p + size);  
            for (int i = 0; i < size; i++)
            {

                classTree::VPL t;
                t.position = h_samples[i].position;
                t.dir = normalize(h_samples[i].lastPosition - h_samples[i].position);
                t.normal = h_samples[i].normal;
                t.weight = 1;
                t.color = h_samples[i].flux / h_samples[i].pdf;
                //t.color = make_float3(1);
                ans.push_back(t);
            } 

            return ans;
        }
        std::vector<classTree::VPL> get_light_cut_sample(int count)
        {
            std::vector<classTree::VPL> ans;
            thrust::device_vector<int> sampled_light_id(acc_num_samples);
            thrust::transform(res_samples.begin(), res_samples.end(), sampled_light_id.begin(), get_sample_light_id(thrust::raw_pointer_cast(res_nodes.data())));
            thrust::sort(sampled_light_id.begin(), sampled_light_id.end());
            thrust::device_vector<int> sum_ids(acc_num_samples);
            thrust::device_vector<int> ids(acc_num_samples);

            auto new_end = thrust::reduce_by_key(ids.begin(), ids.end(), thrust::make_constant_iterator(1), ids.begin(), sum_ids.begin());
            //thrust::sort(sum_ids.begin(), new_end.second, thrust::greater<int>());
            thrust::device_vector<int> ids_for_sample(valid_samples);
            thrust::fill(ids_for_sample.begin(), ids_for_sample.end(), 0);
            thrust::copy(sum_ids.begin(), new_end.second, thrust::make_permutation_iterator(ids_for_sample.begin(), ids.begin()));

            thrust::host_vector<nVertex> h_samples = samples;
            thrust::host_vector<int>  h_ids = ids_for_sample;
            int vpl_id = 0;
            while (ans.size() != count)
            {
                if (h_samples[vpl_id].isLightSource() == false)
                {
                    classTree::VPL t;
                    t.position = h_samples[vpl_id].position;
                    t.dir = h_samples[vpl_id].dir;
                    t.normal = h_samples[vpl_id].normal;
                    t.weight = h_ids[vpl_id] + 0.01;
                    t.color = h_samples[vpl_id].weight / h_samples[vpl_id].pdf;
                    //t.color = make_float3(1);
                    ans.push_back(t);
                }
                vpl_id++;
            }

            return ans;
        }

        thrust::device_vector<classTree::lightTreeNode> light_tree_dev;
        classTree::light_tree_api light_tree_to_device(classTree::lightTreeNode* p, int size)
        {
            classTree::light_tree_api ans;
            light_tree_dev = thrust::device_vector<classTree::lightTreeNode>(p, p + size);
            ans.p = thrust::raw_pointer_cast(light_tree_dev.data());
            ans.size = size;

            return ans;
        }
        void build_optimal_E_train_data(int N_samples)
        {
            thrust::device_vector<int> ids(acc_num_samples);
            thrust::transform(res_samples.begin(), res_samples.end(), ids.begin(),get_sample_light_id(thrust::raw_pointer_cast(res_nodes.data())));
            thrust::sort(ids.begin(),ids.end(), thrust::greater<int>());
            thrust::device_vector<int> sum_ids(acc_num_samples);  
            auto new_end = thrust::reduce_by_key(ids.begin(), ids.end(), thrust::make_constant_iterator(1), thrust::make_discard_iterator(), sum_ids.begin()); 
            thrust::sort(sum_ids.begin(), new_end.second, thrust::greater<int>());
            debug_print<int>(thrust::device_vector<int>(sum_ids.begin(),sum_ids.begin() + 100),"points most refer");


            thrust::host_vector<pathInfo_sample> h_tem(res_samples.begin() + N_samples - 1, res_samples.begin() + N_samples);
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
                        thrust::raw_pointer_cast(res_samples.data()),
                        thrust::raw_pointer_cast(res_nodes.data()),
                        thrust::raw_pointer_cast(Q.data())));
                thrust::sort(t_outler.begin(), t_outler.end());
                thrust::host_vector<float> h_outler = t_outler;
                float outler_value = h_outler[999];
                thrust::for_each(res_samples.begin(), res_samples.end(),
                    clean_outler_value(thrust::raw_pointer_cast(res_nodes.data()), thrust::raw_pointer_cast(Q.data()), outler_value));
                int irr_count = thrust::count_if(res_samples.begin(), res_samples.end(),
                    count_outler_value(thrust::raw_pointer_cast(res_nodes.data()), thrust::raw_pointer_cast(Q.data()), outler_value));
                printf("\n\nsample should be clean:%d / %d %f\n\n", irr_count, res_samples.size(), outler_value);
            }

            thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + N_samples,
                construct_optimal_E_data_sample(
                    thrust::raw_pointer_cast(b_f_square.data()),
                    thrust::raw_pointer_cast(b_pdf0.data()),
                    thrust::raw_pointer_cast(b_P2N_ind_d.data()),
                    thrust::raw_pointer_cast(res_samples.data())));

            thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + M_nodes,
                construct_optimal_E_data_node(
                    thrust::raw_pointer_cast(b_pdf_peak.data()),
                    thrust::raw_pointer_cast(Q.data()),
                    thrust::raw_pointer_cast(b_label_E.data()),
                    thrust::raw_pointer_cast(b_label_P.data()),
                    thrust::raw_pointer_cast(res_nodes.data()),
                    Q.size()));
            b_P2N_ind = b_P2N_ind_d;

            E_td.f_square = b_f_square.data();
            E_td.pdf_0 = b_pdf0.data();
            E_td.pdf_peak = b_pdf_peak.data();
            E_td.label_E = b_label_E.data();
            E_td.label_P = b_label_P.data();
            E_td.P2N_ind = b_P2N_ind.data();
            E_td.N_path = N_samples;
            E_td.M_node = M_nodes;
            printf("M_nodes: %d\n\n", M_nodes);

        }

        struct build_positions_for_NN :thrust::unary_function<int, bool>
        {
            pathInfo_node* nodes;
            float* positions;
            float3 bbox_min;
            float3 bbox_max;
            bool apply_direction;
            bool eye_side;
            __host__ __device__
                build_positions_for_NN(pathInfo_node* nodes, float* positions, bool apply_direction, bool eye_side, float3 bbox_min, float3 bbox_max) :
                nodes(nodes), positions(positions), apply_direction(apply_direction), eye_side(eye_side), bbox_min(bbox_min), bbox_max(bbox_max) {}
            __host__ __device__
                bool operator()(int id)
            {
                const pathInfo_node& s = nodes[id];
                int inch = apply_direction ? 6 : 3;
                float3 position = eye_side ? s.A_position : s.B_position;
                //position = (position - bbox_min) / (bbox_max - bbox_min);
                float3 dir = eye_side ? s.A_dir() : s.B_dir();

                positions[id * inch + 0] = position.x;
                positions[id * inch + 1] = position.y;
                positions[id * inch + 2] = position.z;
                if (apply_direction)
                {
                    positions[id * inch + 3] = dir.x;
                    positions[id * inch + 4] = dir.y;
                    positions[id * inch + 5] = dir.z;
                }
                return 1;
            }
        };
        void build_NN_train_data(classTree::tree t, int N_samples, int* close_set, int dim_eye, int dim_light,
            int position_dim, int close_num)
        {
            float3 bbox_min = t.bbox_min;
            float3 bbox_max = t.bbox_max;
            printf("bbox_min %f %f %f - %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);
            bool use_dir = position_dim == 3 ? false : true;
            N_td.f_square = E_td.f_square;
            N_td.pdf_0 = E_td.pdf_0;
            N_td.pdf_peak = E_td.pdf_peak;
            N_td.label_P = E_td.label_P;
            N_td.P2N_ind = E_td.P2N_ind;
            N_td.P2N_ind_d = b_P2N_ind_d.data();
            N_td.N_path = E_td.N_path;
            N_td.M_node = E_td.M_node;

            b_label_eye.resize(b_label_E.size());
            b_label_light.resize(b_label_E.size());
            b_positions.resize(N_td.M_node * position_dim);

            thrust::transform(b_label_E.begin(), b_label_E.begin() + N_td.M_node, thrust::make_constant_iterator(dim_light), b_label_eye.begin(), thrust::divides<int>());
            thrust::transform(b_label_E.begin(), b_label_E.begin() + N_td.M_node, thrust::make_constant_iterator(dim_light), b_label_light.begin(), thrust::modulus<int>());
            thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + N_td.N_path,
                build_positions_for_NN(
                    thrust::raw_pointer_cast(res_nodes.data()),
                    thrust::raw_pointer_cast(b_positions.data()), use_dir, true, bbox_min, bbox_max));

            b_close_set = thrust::device_vector<float>(close_set, close_set + close_num * dim_eye);

            N_td.close_num = close_num;
            N_td.label_eye = b_label_eye.data();
            N_td.label_light = b_label_light.data();
            N_td.E_ = d_E.data();
            N_td.positions = b_positions.data();
            N_td.position_dim = position_dim;
            N_td.close_set = b_close_set.data();

            debug_print<float>(thrust::device_vector<float>(N_td.positions, N_td.positions + 18), "position_preview");
        }
        struct node_get_predict_id :thrust::binary_function<int, pathInfo_node, int>
        {
            __device__ __host__
                int operator()(int id, pathInfo_node& n)
            {
                n.label_A = id;
                return id;
            }
        };
        void NN_train()
        {
            thrust::device_vector<int> predict_id(N_td.M_node);
            N_td.predict = predict_id.data();

            network_parameter network(1000, 1000, std::vector<int>{60, 16, 16, 32}, 0.005);
            network.fit(10000, 7, N_td);

            printf("begin get id");
            thrust::transform(predict_id.begin(), predict_id.end(), res_nodes.begin(), predict_id.begin(), node_get_predict_id());
            printf("end get id");

        }
        void destroy_E_train_data()
        {
            //b_f_square.resize(0);
            //b_pdf0.resize(0);
            //b_label_E.resize(0);
            //b_label_P.resize(0);
        }
        std::vector<float> get_fake_E()
        {

            std::vector<float> fake_E(SUBSPACE_NUM * SUBSPACE_NUM, 0.0);
            std::vector<float> fake_E_sum(SUBSPACE_NUM, 0.0);
            thrust::host_vector<pathInfo_sample> h_samples = res_samples;
            thrust::host_vector<pathInfo_node> h_nodes = res_nodes;
            for (int i = 0; i < h_samples.size(); i++)
            {
                auto& sample = h_samples[i];
                for (int j = sample.begin_ind; j < sample.end_ind; j++)
                { 
                    float w =  ENERGY_WEIGHT(sample.contri) / sample.sample_pdf;

                    if (isinf(w) || isnan(w))continue;
                    
                    auto& node = h_nodes[j];

                    //float peak_pdf = node.peak_pdf / h_Q[node.label_B];
                    //w *= ENERGY_WEIGHT(sample.contri) / peak_pdf;
                    //if (h_Q[node.label_B] <= 0.0)continue;
                    w = min(w, 10.0);
                    //w = w * w;
                    fake_E[h_nodes[j].label_A * SUBSPACE_NUM + h_nodes[j].label_B] += w;
                    //fake_E_sum[h_nodes[j].label_A] += w;
                }
            }
            if (true)
            {
                for (int i = 0; i < SUBSPACE_NUM; i++)
                {
                    for (int j = 0; j < SUBSPACE_NUM; j++)
                    {
                        //fake_E[i * SUBSPACE_NUM + j] = sqrt(fake_E[i * SUBSPACE_NUM + j]);
                        fake_E_sum[i] += fake_E[i * SUBSPACE_NUM + j];
                    }
                }

            }
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                for (int j = 0; j < SUBSPACE_NUM; j++)
                {
                    if (fake_E_sum[i] > 0.0)
                        fake_E[i * SUBSPACE_NUM + j] /= fake_E_sum[i];
                    else
                    {
                        fake_E[i * SUBSPACE_NUM + j] = 1.0 / SUBSPACE_NUM;
                    }
                }
            }
            return fake_E;
        }

        std::vector<float> train_optimal_E(std::vector<float> fake_weight)
        {
            float lr = .02;
            int epoches = 2;
#ifdef GLASSROOM
            lr = 0.0001;
            epoches = 2;
#endif

            //lr = 0.0001;
            //epoches = 2;
            std::vector<float> ans;
            matrix_parameter theta(SUBSPACE_NUM, SUBSPACE_NUM, lr);//lr = 0.2 batch_size = 20k may be a better choice
            theta.initial_with_inver_sigmoid(fake_weight.data());
            theta.fit(10000, epoches, E_td);
            //(contri**2)(pdf_sum ** 2) * (pdf_sum / pdf_var) = contri**2 / pdf_var / pdf_sum  ?
            //contri**2  / pdf_sum / (pdf_sum * mis_weight)
            //contri / pdf_sum

            //thrust_dev_float EE;
            theta.toE(d_E);
            thrust_host_float h_aannss = d_E;
            ans = std::vector<float>(h_aannss.begin(), h_aannss.end());
            return ans;
        }

        std::vector<float> train_optimal_E()
        {
            float lr = .2;
            int epoches = 2;
#ifdef GLASSROOM
            lr = 0.1;
            epoches = 4;
#endif

            std::vector<float> ans;
            matrix_parameter theta(SUBSPACE_NUM, SUBSPACE_NUM, lr);//lr = 0.2 batch_size = 20k may be a better choice
            theta.fit(10000, epoches, E_td);
            //(contri**2)(pdf_sum ** 2) * (pdf_sum / pdf_var) = contri**2 / pdf_var / pdf_sum  ?
            //contri**2  / pdf_sum / (pdf_sum * mis_weight)
            //contri / pdf_sum

            //thrust_dev_float EE;
            theta.toE(d_E);
            thrust_host_float h_aannss = d_E;
            ans = std::vector<float>(h_aannss.begin(), h_aannss.end());
            return ans;
        }

        void obtain_valid_path()
        {
            int count = thrust::count_if(p_samples.begin(), p_samples.end(), sample_valid_op());
            printf("%d out of 10000 success!!\n\n", count);
        }

        struct node2divide_weight :thrust::unary_function<pathInfo_node, classTree::divide_weight>
        {
            float mean_contri;
            float lerp_rate;
            pathInfo_sample* samples;
            bool eye_side;
            node2divide_weight(pathInfo_sample* samples, float mean_contri, float lerp_rate, bool eye_side) :
                samples(samples), mean_contri(mean_contri), lerp_rate(lerp_rate), eye_side(eye_side) {}
            __host__ __device__
                classTree::divide_weight operator()(pathInfo_node& n)
            {
                classTree::divide_weight ans;
                ans.position = eye_side ? n.A_position : n.B_position;
                float3 flux = (samples[n.path_id].contri / samples[n.path_id].sample_pdf);
                float contri_weight = ENERGY_WEIGHT(flux) / mean_contri;
                if (isnan(contri_weight))
                {
                    contri_weight = 0;
                }
                ans.weight = lerp(1.0, contri_weight, lerp_rate);
                ans.normal = eye_side ? n.A_normal() : n.B_normal();
                ans.dir = eye_side ? n.A_dir() : n.B_dir();

                if (!eye_side && n.light_source == true)
                    ans.weight = 0;
                return ans;
            }
        }; 

        struct node2divide_weight_weighted :thrust::unary_function<int, classTree::divide_weight>
        { 
            pathInfo_sample* samples;
            pathInfo_node* nodes;
            bool eye_side;
            float* E;
            float* Q;
            int dim_light;
            node2divide_weight_weighted(pathInfo_sample* samples,pathInfo_node * nodes, bool eye_side, float* E, float* Q, int dim_light) :
                samples(samples), eye_side(eye_side), dim_light(dim_light), E(E), Q(Q),nodes(nodes) {}
            __host__ __device__
                classTree::divide_weight operator()(int id)
            {
                pathInfo_node& n = nodes[id];
                classTree::divide_weight ans;
                ans.position = eye_side ? n.A_position : n.B_position;
                float3 flux = (samples[n.path_id].contri / samples[n.path_id].sample_pdf * samples[n.path_id].contri);
                float contri_weight = ENERGY_WEIGHT(flux);

                float weighted_function = 0;
                for (int i = samples[n.path_id].begin_ind; i != samples[n.path_id].end_ind; i++)
                {
                    int light_label = nodes[i].label_B;
                    int eye_label = nodes[i].label_A;
                    weighted_function += nodes[i].peak_pdf / Q[light_label] * E[eye_label * dim_light + light_label];
                }
                contri_weight /= weighted_function;
                weighted_function = n.peak_pdf / Q[n.label_B] * E[n.label_A * dim_light + n.label_B] / weighted_function;
                    
                contri_weight *=  weighted_function * weighted_function;
                if (isnan(contri_weight))
                {
                    contri_weight = 0;
                }
                ans.weight = sqrt(contri_weight);

                ans.normal = eye_side ? n.A_normal() : n.B_normal();
                ans.dir = eye_side ? n.A_dir() : n.B_dir();

                if (!eye_side && n.light_source == true)
                    ans.weight = 0;
                return ans;
            }
        };

        struct node2divide_weight_with_label :thrust::unary_function<pathInfo_node, classTree::divide_weight_with_label>
        {
            float mean_contri;
            float lerp_rate;
            pathInfo_sample* samples;
            bool eye_side;
            node2divide_weight_with_label(pathInfo_sample* samples, float mean_contri, float lerp_rate, bool eye_side) :
                samples(samples), mean_contri(mean_contri), lerp_rate(lerp_rate), eye_side(eye_side) {}
            __host__ __device__
                classTree::divide_weight_with_label operator()(pathInfo_node& n)
            {
                classTree::divide_weight_with_label ans;
                ans.position = eye_side ? n.A_position : n.B_position;
                float3 flux = (samples[n.path_id].contri / samples[n.path_id].sample_pdf);
                float contri_weight = ENERGY_WEIGHT(flux) / mean_contri;
                ans.weight = lerp(1.0, contri_weight, lerp_rate);
                if (!eye_side && n.light_source == true)
                    ans.weight = 0;
                ans.label = eye_side ? n.label_A : n.label_B;
                return ans;
            }
        };
        struct eval_flux : thrust::unary_function<pathInfo_sample, float>
        {
            __host__ __device__
                float operator()(const pathInfo_sample& s)
            {
                float ans = ENERGY_WEIGHT((s.contri / s.sample_pdf));
                if (isnan(ans)||isinf(ans))
                {
                    return 0;
                }
                return ans;
            }
        };

        void classification_data_get_flat(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set, float lerp_rate)
        {
            thrust::device_vector<classTree::divide_weight> weight_node_eye(acc_num_nodes);
            thrust::device_vector<classTree::divide_weight> weight_node_light(acc_num_nodes);
            float mean_contri = thrust::reduce(
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()),
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()) + res_samples.size()) / acc_num_samples;

            thrust::transform(res_nodes.begin(), res_nodes.end(), weight_node_eye.begin(), node2divide_weight(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, true));
            thrust::transform(res_nodes.begin(), res_nodes.end(), weight_node_light.begin(), node2divide_weight(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, false));

            printf("mean_contri %f\n", mean_contri);
            thrust::host_vector<classTree::divide_weight>h_e = weight_node_eye;
            thrust::host_vector<classTree::divide_weight>h_l = weight_node_light;
            eye_set = std::vector<classTree::divide_weight>(h_e.data(), h_e.data() + h_e.size());
            light_set = std::vector<classTree::divide_weight>(h_l.data(), h_l.data() + h_l.size());
            return;
        }

        struct divide_weight_valid_op
        {
            __host__ __device__
            bool operator()(const classTree::divide_weight& a)const
            {
                return a.weight > 0 && !isnan(a.weight) && !isinf(a.weight);
            }
        };
        void classification_data_get_flat(std::vector<classTree::divide_weight>& eye_set, float lerp_rate, bool eye_side)
        {
            int slic_index = acc_num_nodes * 0.1;
            thrust::device_vector<classTree::divide_weight> weight_node_eye(slic_index); 
            float mean_contri = thrust::reduce(
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()),
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()) + res_samples.size()) / acc_num_samples;

            thrust::transform(res_nodes.begin(), res_nodes.begin() + slic_index, weight_node_eye.begin(), node2divide_weight(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, eye_side)); 

            printf("mean_contri %f\n", mean_contri);

            thrust::device_vector<classTree::divide_weight> valid_e(slic_index);
            auto n_end = thrust::copy_if(weight_node_eye.begin(), weight_node_eye.end(), valid_e.begin(), divide_weight_valid_op());
            thrust::host_vector<classTree::divide_weight>h_e(valid_e.begin(), n_end);
            eye_set = std::vector<classTree::divide_weight>(h_e.data(), h_e.data() + h_e.size()); 
            printf("valid_clear %d\n", n_end - valid_e.begin());
            return;
        }

        void classification_weighted_function(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set,
            float* Gamma, int dim_eye, int dim_light, float* Q)
        {
            thrust::device_vector<classTree::divide_weight> weight_node_eye(acc_num_nodes);
            thrust::device_vector<classTree::divide_weight> weight_node_light(acc_num_nodes); 
            thrust::device_vector<float> d_Gamma(Gamma, Gamma + dim_eye * dim_light);
            thrust::device_vector<float> d_Q(Q, Q + dim_light); 
             
            thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(acc_num_nodes), weight_node_eye.begin(), node2divide_weight_weighted(
                thrust::raw_pointer_cast(res_samples.data()),
                thrust::raw_pointer_cast(res_nodes.data()),
                true, thrust::raw_pointer_cast(d_Gamma.data()),
                thrust::raw_pointer_cast(d_Q.data()), dim_light
            ));

            thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(acc_num_nodes), weight_node_light.begin(), node2divide_weight_weighted(
                thrust::raw_pointer_cast(res_samples.data()),
                thrust::raw_pointer_cast(res_nodes.data()),
                false, thrust::raw_pointer_cast(d_Gamma.data()),
                thrust::raw_pointer_cast(d_Q.data()), dim_light
            ));
            thrust::host_vector<classTree::divide_weight>h_e = weight_node_eye;
            thrust::host_vector<classTree::divide_weight>h_l = weight_node_light;
            eye_set = std::vector<classTree::divide_weight>(h_e.data(), h_e.data() + h_e.size());
            light_set = std::vector<classTree::divide_weight>(h_l.data(), h_l.data() + h_l.size());
            return;
        }

        void classification_weighted_function(std::vector<classTree::divide_weight>& eye_set,
            float* Gamma, int dim_eye, int dim_light, float* Q, bool eye_side)
        {
            int slic_index = acc_num_nodes * 1.0;
            thrust::device_vector<classTree::divide_weight> weight_node_eye(slic_index); 
            thrust::device_vector<float> d_Gamma(Gamma, Gamma + dim_eye * dim_light);
            thrust::device_vector<float> d_Q(Q, Q + dim_light);

            thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(slic_index), weight_node_eye.begin(), node2divide_weight_weighted(
                thrust::raw_pointer_cast(res_samples.data()),
                thrust::raw_pointer_cast(res_nodes.data()),
                eye_side, thrust::raw_pointer_cast(d_Gamma.data()),
                thrust::raw_pointer_cast(d_Q.data()), dim_light
            ));

            thrust::device_vector<classTree::divide_weight> valid_e(slic_index);
            auto n_end = thrust::copy_if(weight_node_eye.begin(), weight_node_eye.end(), valid_e.begin(), divide_weight_valid_op());
            thrust::host_vector<classTree::divide_weight>h_e(valid_e.begin(),n_end); 
            eye_set = std::vector<classTree::divide_weight>(h_e.data(), h_e.data() + h_e.size()); 
            return;
        }


        void get_tree_weight(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set, float lerp_rate)
        {
            thrust::device_vector<classTree::divide_weight> weight_node_eye(acc_num_nodes);
            thrust::device_vector<classTree::divide_weight> weight_node_light(acc_num_nodes);
            float mean_contri = thrust::reduce(
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()),
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()) + res_samples.size()) / acc_num_samples;

            thrust::transform(res_nodes.begin(), res_nodes.end(), weight_node_eye.begin(), node2divide_weight(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, true));
            thrust::transform(res_nodes.begin(), res_nodes.end(), weight_node_light.begin(), node2divide_weight(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, false));

            printf("mean_contri %f\n", mean_contri);
            thrust::host_vector<classTree::divide_weight>h_e = weight_node_eye;
            thrust::host_vector<classTree::divide_weight>h_l = weight_node_light;
            eye_set = std::vector<classTree::divide_weight>(h_e.data(), h_e.data() + h_e.size());
            light_set = std::vector<classTree::divide_weight>(h_l.data(), h_l.data() + h_l.size());
            return;
        }

        void get_tree_weight(std::vector<classTree::divide_weight_with_label>& light_set, float lerp_rate)
        {
            thrust::device_vector<classTree::divide_weight_with_label> weight_node_light(acc_num_nodes);
            float mean_contri = thrust::reduce(
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()),
                thrust::make_transform_iterator(res_samples.begin(), eval_flux()) + res_samples.size()) / acc_num_samples;

            thrust::transform(res_nodes.begin(), res_nodes.end(), weight_node_light.begin(), node2divide_weight_with_label(
                thrust::raw_pointer_cast(res_samples.data()),
                mean_contri, lerp_rate, false));

            thrust::host_vector<classTree::divide_weight_with_label>h_l = weight_node_light;
            light_set = std::vector<classTree::divide_weight_with_label>(h_l.data(), h_l.data() + h_l.size());
            return;
        }

        struct distance_square :thrust::binary_function<float3, float3, float>
        {
            __host__ __device__ float operator()(float3 a, float3 b)
            {
                float3 d = a - b;
                return d.x * d.x + d.y * d.y + d.z * d.z;
                //return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
            }
        };
        std::vector<int> get_close_set(void* centers_v, int num_classes, int close_num)
        {
            float3* centers = reinterpret_cast<float3*>(centers_v);
            thrust::device_vector<int> ans_array(num_classes * close_num);
            thrust::device_vector<int> temple_index_array(num_classes);
            thrust::device_vector<float> temple_array(num_classes);
            thrust::device_vector<float3> d_centers(centers, centers + num_classes);

            for (int i = 0; i < num_classes; i++)
            {
                thrust::transform(d_centers.begin(), d_centers.end(),
                    thrust::make_permutation_iterator(d_centers.begin(), thrust::make_constant_iterator(i)),
                    temple_array.begin(), distance_square());

                thrust::sequence(temple_index_array.begin(), temple_index_array.end());
                thrust::sort_by_key(temple_array.begin(), temple_array.end(), temple_index_array.begin());
                thrust::copy(
                    thrust::make_permutation_iterator(temple_index_array.begin(), thrust::make_counting_iterator(0)),
                    thrust::make_permutation_iterator(temple_index_array.begin(), thrust::make_counting_iterator(0)) + close_num,
                    ans_array.data() + i * close_num);
            }
            thrust::host_vector<int> ans = ans_array;
            return std::vector<int>(ans.data(), ans.data() + ans.size());
        }
        void export_data(classTree::tree t)
        {
            N_td.positions;
            N_td.E_;

            std::ofstream onFile;
            onFile.open("./exp_record/OPTP_data/re_optimal_test.txt");
            onFile << N_td.N_path << " " << N_td.M_node << std::endl;

            thrust::host_vector<float> h_E(N_td.E_, N_td.E_ + 1000 * 1000);
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    float t = h_E[i * 1000 + j];
                    onFile << t << " ";
                }
            }

            thrust::host_vector<int> close_set(N_td.close_set, N_td.close_set + 1000 * 32);
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    onFile << close_set[i * 32 + j] << " ";
                }
            }

            //classTree
            int num_tree = t.size;
            onFile << num_tree << " ";
            for (int i = 0; i < num_tree; i++)
            {
                int is_leaf = t.v[i].leaf ? 1 : 0;
                float3 position = t.v[i].mid;
                if (is_leaf == 0)
                {
                    onFile << is_leaf << " " << position.x << " " << position.y << " " << position.z << " ";
                    for (int j = 0; j < 8; j++)
                    {
                        onFile << t.v[i].child[j] << " ";
                    }
                }
                if (is_leaf)
                {
                    onFile << is_leaf << " " << t.v[i].label << " ";
                }

            }

            thrust::host_vector<float> loss(N_td.f_square, N_td.f_square + N_td.N_path);
            thrust::host_vector<float> pdf0(N_td.pdf_0, N_td.pdf_0 + N_td.N_path);
            thrust::host_vector<float> position(N_td.positions, N_td.positions + N_td.M_node * 3);
            thrust::host_vector<int> label_eye(N_td.label_eye, N_td.label_eye + N_td.M_node);
            thrust::host_vector<int> label_light(N_td.label_light, N_td.label_light + N_td.M_node);
            thrust::host_vector<float> peak_pdf(N_td.pdf_peak, N_td.pdf_peak + N_td.M_node);

            thrust::host_vector<pathInfo_node> nodes = res_nodes;
            for (int i = 0; i < N_td.N_path; i++)
            {
                int begin_id = N_td.P2N_ind[i];
                int end_id = i == N_td.N_path - 1 ? N_td.M_node : N_td.P2N_ind[i + 1];
                onFile << end_id - begin_id << " " << loss[i] << " " << pdf0[i] << " ";
                for (int j = begin_id; j < end_id; j++)
                {
                    onFile << position[j * 3 + 0] << " " << position[j * 3 + 1] << " " << position[j * 3 + 2] << " ";
                    onFile << nodes[j].A_dir().x << " " << nodes[j].A_dir().y << " " << nodes[j].A_dir().z << " ";
                    onFile << label_eye[j] << " " << label_light[j] << " ";
                    onFile << peak_pdf[j] << " ";
                }
            }

            onFile.close();
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
        void sample_reweight()
        {
            thrust::host_vector<pathInfo_sample> h_samples = res_samples;
            thrust::host_vector<float> weight(1920 * 1000 / 100 * 1.1);
            thrust::host_vector<int> count(1920 * 1000 / 100 * 1.1);
            for (int i = 0; i < h_samples.size(); i++)
            {
                //printf("id %d %d\n", h_samples[i].pixiv_id.x, h_samples[i].pixiv_id.y);
                int id_x = h_samples[i].pixiv_id.x / 10;
                int id_y = h_samples[i].pixiv_id.y / 10;
                int n_id = id_x + id_y * 192;
                float ww = ENERGY_WEIGHT( h_samples[i].contri) / h_samples[i].sample_pdf;
                if (isnan(ww) || isinf(ww))continue;
                weight[n_id] += ww;
                count[n_id]++;
            } 
            for (int i = 0; i < h_samples.size(); i++)
            {
                int id_x = h_samples[i].pixiv_id.x / 10;
                int id_y = h_samples[i].pixiv_id.y / 10;
                int n_id = id_x + id_y * 192;
                h_samples[i].contri /= (weight[n_id] / 100 + 0.1);
            }
            res_samples = h_samples;
        }
        void clear_thrust_vector()
        {
            p_nodes.clear();
            p_samples.clear();
            sample_bias_flag.clear();
            LVC.clear();
            samples.clear();
            pmf.clear();


            Q.clear();
            res_nodes.clear();
            res_samples.clear();
            LVC.clear();
            samples.clear();
            pmf.clear();
            p_nodes.clear();
            p_samples.clear();
            sample_bias_flag.clear();
            b_f_square.clear();
            b_pdf0.clear();
            b_pdf_peak.clear();
            b_positions.clear();
            b_close_set.clear();
            b_label_E.clear();
            b_label_eye.clear();
            b_label_light.clear();
            b_label_P.clear();
            b_P2N_ind_d.clear();
            b_P2N_ind.clear();
            d_E.clear();
            light_tree_dev.clear();
             
            tree_save.clear();
        }
    };

}

namespace HS_algorithm
{ 
#include<vector>
    using namespace std; 
    struct close_record  
    {
        int begin;
        int end;
        float metric;
        __host__ __device__ bool operator<(const close_record& b)const
        {
            return metric < b.metric;
        }
    };
    struct initial_record
    {
        close_record* a;
        initial_record(close_record* a):a(a){}
        __host__ __device__ void operator()(int id)
        {
            a[id].begin = id;
            a[id].end = 0;
            a[id].metric = FLT_MAX;
        }
    };

    template<typename T>
    struct update_record
    {
        int begin;
        int end;
        int* indices;
        T* Y;
        update_record(int begin, int end, int* indices, T* Y) :begin(begin), end(end), indices(indices), Y(Y) {}
        __host__ __device__ void operator()(close_record &a)
        { 
            for (int i = begin; i < end; i++)
            {
                int trans_id = indices[i];
                float metric = Y[a.begin].d(Y[trans_id]);
                if (metric < a.metric)
                {
                    a.end = trans_id;
                    a.metric = metric;
                }
            } 
        }
    };
    template<typename T>
    struct update_record_index_version
    {
        int begin;
        int end;
        int* indices;
        T* Y;
        update_record_index_version(int begin, int end, int* indices, T* Y) :begin(begin), end(end), indices(indices), Y(Y) {}
        __host__ __device__ void operator()(close_record& a)
        {
            for (int i = begin; i < end; i++)
            {
                int trans_id = indices[i];
                float metric = Y[a.begin].d(Y[trans_id]);
                if (metric < a.metric)
                {
                    a.end = i;
                    a.metric = metric;
                }
            }
        }
    };

    struct push_back_index
    {
        int* indices;
        int size;
        push_back_index(int* indices,int size):size(size),indices(indices){}
        __device__ __host__ void operator()(int n_id)
        {
            indices[size] = n_id;
        }
    };
    vector<int> label_with_center(gamma_point* Y_p, int* X_p, int size_Y, int size_center)
    {
        thrust::device_vector<gamma_point> d_Y(Y_p, Y_p + size_Y);
        thrust::device_vector<int> d_X(X_p, X_p + size_center);
        thrust::device_vector<close_record> d_Z(size_Y);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + size_Y, initial_record(thrust::raw_pointer_cast(d_Z.data())));

        //printf("d_X size %d %d\n", d_X.size(), d_Y.size());
        //debug_print<int>(d_X,"center id");
        thrust::for_each(d_Z.begin(), d_Z.end(), update_record_index_version<gamma_point>(0, size_center,
            thrust::raw_pointer_cast(d_X.data()), thrust::raw_pointer_cast(d_Y.data())));

        //debug_print<int>(d_X, "center id");
        thrust::host_vector<close_record> h_Z = d_Z;
        vector<int> ans(size_Y);
        for (int i = 0; i < size_Y; i++)
        {
            ans[i] = h_Z[i].end;
        }
        return ans;
    }
    //return the indices of cluster centers
    vector<int> Hochbaum_Shmonys(gamma_point* Y_p, int* X_p, int size_y, int size_x, int target_size)
    {
        thrust::device_vector<gamma_point> d_Y(Y_p,Y_p + size_y);
        thrust::device_vector<int> d_X(X_p, X_p + size_x);
        thrust::device_vector<close_record> d_Z(size_y);
        d_X.resize(target_size);

        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + size_y, initial_record(thrust::raw_pointer_cast(d_Z.data())));

        int n_begin = 0;
        while (size_x < target_size)
        {
            if (size_x == 0)
            {
                thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 1,
                    push_back_index(thrust::raw_pointer_cast(d_X.data()), size_x));
                size_x += 1;
                continue;
            }
            else
            {
                thrust::for_each(d_Z.begin(), d_Z.end(), update_record<gamma_point>(n_begin, size_x,
                    thrust::raw_pointer_cast(d_X.data()), thrust::raw_pointer_cast(d_Y.data())));

                auto p = thrust::max_element(d_Z.begin(), d_Z.end());
                thrust::host_vector<close_record> t_v(p, p + 1);
                //printf("hs algorithm id %d min distance %f\n", t_v[0].begin, t_v[0].metric);
                thrust::for_each(thrust::make_counting_iterator(t_v[0].begin), thrust::make_counting_iterator(t_v[0].begin) + 1,
                    push_back_index(thrust::raw_pointer_cast(d_X.data()), size_x));
                n_begin = size_x;
                size_x += 1;
            }
        }


        thrust::host_vector<int> h_X = d_X;
        return vector<int>(h_X.data(),h_X.data() + h_X.size());
    }

}
