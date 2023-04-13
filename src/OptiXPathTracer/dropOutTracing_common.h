#ifndef DROPOUTTRACING_COMMON
#define DROPOUTTRACING_COMMON 
#include"rt_function.h"
#include <optix.h>
#include"random.h"
#include"decisionTree/classTree_common.h"
//#include"sutil/vec_math.h"
//using namespace optix;
namespace dropOut_tracing
{
    const int slot_number = 5;
    const int default_specularSubSpaceNumber = 20;
    const int default_surfaceSubSpaceNumber = 100; 
    const int record_buffer_width = 1;
#define DOT_EMPTY_SURFACEID 0
    enum class DropOutType
    {
        LS,
        LDS=LS,
        LSS,
        LSSS,
        DropOutTypeNumber
    };

    /*
     * Enum to declare the usage of data slots.
     * There are 5 slots by default.
     * Slot 0 is used to calculate the mean of the inverse pdf of the path in the subspace.
     */
    enum class SlotUsage
    {
        Average, SlotUsageNumber
    };

    struct statistics_data
    {
        float* host_data;
        float* device_data;
        int size;
        bool on_GPU;
         
        RT_FUNCTION __host__ float& operator[](int i)
        {
            return on_GPU ? device_data[i] : host_data[i];
        }
    };

    struct statistic_record
    {
        float data;
        short specular_subspaceId;
        short surface_subspaceId;
        SlotUsage data_slot;
        DropOutType type;
        bool valid;

        RT_FUNCTION __host__ statistic_record() {}
        RT_FUNCTION __host__ statistic_record(DropOutType type, int specular_subspaceId, int surface_subspaceId, SlotUsage data_slot) :
            type(type),
            specular_subspaceId(specular_subspaceId),
            surface_subspaceId(surface_subspaceId),
            data_slot(data_slot), valid(true)
        {}
        RT_FUNCTION __host__ statistic_record(DropOutType type, int specular_subspaceId, int surface_subspaceId, int data_slot) :
            type(type),
            specular_subspaceId(specular_subspaceId),
            surface_subspaceId(surface_subspaceId),
            data_slot(SlotUsage(data_slot)), valid(true)
        {}
        RT_FUNCTION __host__ statistic_record& operator=(float value)
        {
            data = value;
            return *this;
        }
        RT_FUNCTION __host__ operator float() const
        {
            return data;
        }
    };

    struct dropOutTracing_params
    {
        bool is_init;
        int specularSubSpaceNumber;
        int surfaceSubSpaceNumber;
        classTree::tree_node* specularSubSpace;
        classTree::tree_node* surfaceSubSpace;
        statistic_record* record_buffer;
        int record_buffer_core;
        int record_buffer_padding;


        //// Initialize statistics_iteration_count to 0, indicating that no statistics data has been collected yet.
        // It is important not to use statistics data for other operations when the count is 0, as all statistics data will be set to 0 at this time.
        // 当计数为0时，说明尚未有任何的统计数据被统计，注意不要在此时使用统计数据来做别的操作，所有的统计数据在此时都会被设为0
        int statistics_iteration_count;


        statistics_data data; 

        RT_FUNCTION int get_specular_label(float3 position, float3 normal) { return specularSubSpace ? classTree::tree_index(specularSubSpace, position, normal) : 0; }
        RT_FUNCTION int get_surface_label(float3 position, float3 normal,float3 dir = make_float3(0.0))
        { return surfaceSubSpace? classTree::tree_index(surfaceSubSpace, position, normal, dir) : 0; }
        RT_FUNCTION __host__ int spaceId2DataId(int specular_id, int surface_id) { return surface_id * specularSubSpaceNumber + specular_id; }
        RT_FUNCTION __host__ int2 dataId2SpaceId(int data_id) { return make_int2(data_id % specularSubSpaceNumber, int(data_id / specularSubSpaceNumber)); }
        RT_FUNCTION __host__ float& get_statistic_data(int type, int specular_id, int surface_id, int data_slot)
        {
            if (record_buffer == nullptr)
            {
                printf("dot params is Not Initialized Yet\n");
            }
             
            if (statistics_iteration_count == 0 && data.on_GPU == true)
            {
                printf("warn: you are using the statistic data in DEVICE WITHOUT ANY data collected\n");
            }

            int slot_bias = 0;
            //if (type == DropOutType::LS)
            //{
            //    return data[slot_bias + specular_id * slot_number + data_slot];
            //} 
            //slot_bias += specularSubSpaceNumber * slot_number;
            slot_bias += type * specularSubSpaceNumber * surfaceSubSpaceNumber * slot_number;
             
            int one_dim_id = spaceId2DataId(specular_id, surface_id);
            return data[one_dim_id * slot_number + data_slot + slot_bias];
        } 
        RT_FUNCTION __host__ float& get_statistic_data(DropOutType type, int specular_id, int surface_id, SlotUsage data_slot)
        {
            return get_statistic_data(type, specular_id, surface_id, int(data_slot));
        }

        RT_FUNCTION __host__ float& get_statistic_data(DropOutType type, int specular_id, int surface_id, int data_slot)
        {
            return get_statistic_data(int(type), specular_id, surface_id, data_slot);
        }

        RT_FUNCTION __host__ float& get_statistic_data(int type, int specular_id, int surface_id, SlotUsage data_slot)
        {
            return get_statistic_data(type, specular_id, surface_id, int(data_slot));
        }
    };

};

typedef dropOut_tracing::dropOutTracing_params DropOutTracing_params;
typedef dropOut_tracing::DropOutType DOT_type;
typedef dropOut_tracing::SlotUsage DOT_usage;
typedef dropOut_tracing::statistic_record DOT_record;

#endif