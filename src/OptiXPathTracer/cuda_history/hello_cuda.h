#ifndef __HELLO_H__
#define __HELLO_H__
#include<vector>
#include"classTree_common.h" 
#include"MLP_common.h"
#include<thrust/device_vector.h>
#include <thrust/count.h>
void hello_print();

void batch_gemm_test();
void thrust_bp_test_run();
void optimal_E_train(int num_samples, int num_nodes, int dim_light, int dim_eye); 

MLP::nVertex* to_device(MLP::nVertex* host_ptr, int size);
void to_device(float* host_ptr, int size,thrust::device_vector<float> &v); 

//void to_device(MLP::nVertex* host_ptr, int size, thrust::device_vector<MLP::nVertex>& v);
void* to_device(void* p, int size);
classTree::tree_node* device_to(classTree::tree_node* a, int size);
 
struct inver_index_table
{
    std::vector<int> v;
    std::vector<int> begin_index; 
    int num_classes;
    inver_index_table(int num_classes):num_classes(num_classes)
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
            begin_index[i] = begin_index[i-1] + tt;
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
    void record(int i,float time)
    {
        v[i] += time;
    }
};
void learn_by_data(
    std::vector<float>& ans, //size N
    std::vector<float>& f_square, //size N
    std::vector<float>& pdf_0,    //size N

    std::vector<float>& pdf_peak, //size M
    std::vector<int>& label_E,   //size M
    std::vector<int>& label_P,  //size M
    std::vector<int>& P2N_ind  //size N P2N_ind[i] record the begin index of path i
    );
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
);

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
 
namespace MLP
{
    namespace data_obtain_cudaApi
    {   

        void buffer_pointer_validation();
        data_buffer result_buffer_setting(); 
        data_buffer LVC_process(thrust::device_ptr<RAWVertex> p, int acc, int size);
        void obtain_valid_path();
        void set_data_buffer(data_buffer& b);

        void node_label(classTree::tree_node* eye_tree, classTree::tree_node* light_tree);
        std::vector<float> get_Q(classTree::tree_node* light_tree);
        void pmf_reset_by_tree(classTree::tree_node* light_tree);

        void build_optimal_E_train_data(int N_samples);
        int valid_sample_gather();

        std::vector<float> train_optimal_E();
        std::vector<float> train_optimal_E(std::vector<float>);

        void view_res_data();
        std::vector<float> get_fake_E();

        void get_tree_weight(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set, float lerp_rate = 0.0);
        void get_tree_weight(std::vector<classTree::divide_weight_with_label>& light_set, float lerp_rate = 0.0);

        std::vector<int> get_close_set(void* centers, int num_classes, int close_num);

        void build_NN_train_data(classTree::tree t, int N_samples, int* close_set, int dim_eye = SUBSPACE_NUM, int dim_light = SUBSPACE_NUM,
            int position_dim = 3, int close_num = 32);

        void NN_train();

        void export_data(classTree::tree t);
        void LVC_process_simple(thrust::device_ptr<RAWVertex> raw_p, thrust::device_ptr<BDPTVertex> lvc_p,
            int search_size, int& path_count, int& vertex_count, int bias = 0);

        classTree::tree_node* eye_tree_to_device(classTree::tree_node* a, int size);
        classTree::tree_node* light_tree_to_device(classTree::tree_node* a, int size);
        std::vector<classTree::VPL> get_light_cut_sample(int count);

        classTree::light_tree_api light_tree_to_device(classTree::lightTreeNode* p, int size);
        std::vector<classTree::VPL> get_light_cut_sample(thrust::device_ptr<BDPTVertex> lvc_p, int size);
        void classification_data_get_flat(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set, float lerp_rate);
        void classification_weighted_function(std::vector<classTree::divide_weight>& eye_set, std::vector<classTree::divide_weight>& light_set,
            float* Gamma, int dim_eye, int dim_light, float* Q);

        void classification_data_get_flat(std::vector<classTree::divide_weight>& eye_set, float lerp_rate, bool eye_side);

        void classification_weighted_function(std::vector<classTree::divide_weight>& eye_set,
            float* Gamma, int dim_eye, int dim_light, float* Q, bool eye_side);
        void clear_thrust_vector();
        void sample_reweight();
    };

}
namespace HS_algorithm
{

    std::vector<int> Hochbaum_Shmonys(gamma_point* Y_p, int* X_p, int size_y, int size_x, int target_size);
    std::vector<int> label_with_center(gamma_point* Y_p, int* X_p, int size_Y, int size_center);
}
#endif