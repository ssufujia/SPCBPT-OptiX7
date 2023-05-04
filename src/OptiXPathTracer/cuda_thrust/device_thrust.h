#ifndef DEVICE_THRUST
#define DEVICE_THRUST

#include<vector>


#include"../BDPTVertex.h"
#include"../optixPathTracer.h"
#include <stdio.h>
#include<thrust/device_vector.h> 
#include"../decisionTree/classTree_common.h"
#include"../../cuda/MaterialData.h"
#include"../../cuda/BufferView.h"
#include<thrust/host_vector.h>
#include"../PG_common.h"
#include"../dropOutTracing_common.h"
#include"../DOT_PG_trainingParams.h"
void useCUDA();



#define measure_event(start_event, stop_event, sum, name) { \
	cudaEventRecord(stop_event); \
	cudaEventSynchronize(stop_event); \
	float miliseconds_reference = 0; \
	cudaEventElapsedTime(&miliseconds_reference, start_event, stop_event); \
	sum += miliseconds_reference; \
	cudaEventDestroy(start_event); \
	cudaEventDestroy(stop_event); \
	} // 	printf("%s time = %f ms\n", name, miliseconds_reference); \

#define create_events_and_start(start_event, stop_event) { \
	cudaEventCreate(&start_event); \
	cudaEventCreate(&stop_event); \
	cudaEventRecord(start_event); \
}

struct inver_index_table
{
    std::vector<int> v;
    std::vector<int> begin_index;
    int num_classes;
    inver_index_table(int num_classes) :num_classes(num_classes)
    {
        begin_index.resize(num_classes);
    }
    void build(std::vector<int> labels)
    {

        for (int i = 0; i < num_classes; i++)
        {
            begin_index[i] = 1;//init
        }
        for (int i = 0; i < labels.size(); i++)
        {
            begin_index[labels[i]] += 1;//counting
        }
        std::vector<int> t_save(num_classes);
        for (int i = 0; i < num_classes; i++)
        {
            //printf("bb:%d \n", begin_index[i]);
            t_save[i] = 0;
        }

        int last = begin_index[0];
        begin_index[0] = 0;
        for (int i = 1; i < num_classes; i++)
        {
            int tt = last;
            last = begin_index[i];
            begin_index[i] = begin_index[i - 1] + tt;
        }


        v.resize(labels.size() + num_classes);
        for (int i = 0; i < labels.size(); i++)
        {
            int l = labels[i];
            int count = t_save[l];
            v[begin_index[l] + count] = i;
            t_save[l] ++;
        }
        for (int i = 0; i < num_classes; i++)
        {
            v[begin_index[i] + t_save[i]] = INT32_MAX;
        }
    }
};
struct timerecord_stage
{
    std::vector<float> v;
    timerecord_stage(int size = 10)
    {
        v.resize(10);
        for (int i = 0; i < size; i++)
        {
            v[i] = 0;
        }
    }
    void print()
    {
        float sum = 0;
        for (int i = 0; i < v.size(); i++)
        {
            printf("stage %d runtime :%f\n", i, v[i]);
            sum += v[i];
        }
        printf("runtim_sum:%f\n", sum);
    }
    void record(int i, float time)
    {
        v[i] += time;
    }
};
namespace MyThrustOp
{
    SubspaceSampler LVC_Process(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange);
    SubspaceSampler LVC_Process_glossyOnly(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange, BufferView<MaterialData::Pbr> mats);

    int valid_sample_gather(thrust::device_ptr<preTracePath> raw_paths, int maxPathSize,
        thrust::device_ptr<preTraceConnection> raw_conns, int maxConns);

    std::vector<classTree::divide_weight> get_weighted_point_for_tree_building(bool eye_side = false, int max_size = 0);


    int preprocess_getQ(thrust::device_ptr<BDPTVertex> vertices, thrust::device_ptr<bool> validState, int countRange, thrust::device_ptr<float>& Q);

    void preprocess_getGamma(thrust::device_ptr<float>& Gamma, bool caustic_case = false);
    void node_label(classTree::tree_node* eye_tree, classTree::tree_node* light_tree);
    void sample_reweight();
    void build_optimal_E_train_data(int N_samples);
    void Q_zero_handle(thrust::device_ptr<float>& Q);
    void train_optimal_E(thrust::device_ptr<float>& E_ptr);
    thrust::device_ptr<float> Gamma2CMFGamma(thrust::device_ptr<float> Gamma);


    void load_Q_file(thrust::device_ptr<float>& Q);
    void load_Gamma_file(thrust::device_ptr<float>& Gamma);
    void get_caustic_frac(thrust::device_ptr<float>& frac);

    thrust::device_ptr<float> envMapCMFBuild(float* pmf, int size);
    std::vector<path_guiding::PG_training_mat> get_data_for_path_guiding(int num_datas = -1);

    void clear_training_set();
    std::vector<classTree::divide_weight> getCausticCentroidCandidate(bool eye_side, int max_size);



    thrust::host_vector<uchar4> copy_to_host(uchar4* data, int size);
    thrust::host_vector<float4> copy_to_host(float4* data, int size);
    classTree::tree_node* light_tree_to_device(classTree::tree_node* a, int size);
    classTree::tree_node* eye_tree_to_device(classTree::tree_node* a, int size);
    classTree::tree_node* DOT_specular_tree_to_device(classTree::tree_node* a, int size);
    classTree::tree_node* DOT_surface_tree_to_device(classTree::tree_node* a, int size);
    dropOut_tracing::statistics_data_struct* DOT_statistics_data_to_device(dropOut_tracing::statistics_data_struct* a, int size);
    thrust::host_vector<dropOut_tracing::statistics_data_struct> DOT_statistics_data_to_host();
    dropOut_tracing::statistics_data_struct* DOT_statistics_data_to_device(thrust::host_vector<dropOut_tracing::statistics_data_struct> h_v);
    dropOut_tracing::statistic_record* DOT_get_statistic_record_buffer(int size = -1);
    thrust::host_vector<dropOut_tracing::statistic_record> DOT_get_host_statistic_record_buffer(bool valid_only = true);

    dropOut_tracing::PGParams* DOT_PG_data_to_device(thrust::host_vector<dropOut_tracing::PGParams> h_v);
    thrust::host_vector<dropOut_tracing::PGParams> DOT_PG_data_to_host();
    float* DOT_causticCMFGamma_to_device(thrust::host_vector<float> DOT_h_GAMMA);
    float* DOT_causticFrac_to_device(thrust::host_vector<float> DOT_h_frac);
    thrust::host_vector<dropOut_tracing::pixelRecord> DOT_get_pixelRecords();
    dropOut_tracing::pixelRecord* DOT_set_pixelRecords_size(int size);
    float* DOT_get_Q();


    path_guiding::quad_tree_node* quad_tree_to_device(path_guiding::quad_tree_node* a, int size);
    path_guiding::Spatio_tree_node* spatio_tree_to_device(path_guiding::Spatio_tree_node* a, int size);
}



#ifdef HISTORY_CODE

#endif

#endif