#ifndef DROPOUTTRACING_COMMON
#define DROPOUTTRACING_COMMON 
#include"rt_function.h"
#include <optix.h>
#include"random.h"
#include"decisionTree/classTree_common.h"
#include"DOT_PG_trainingParams.h"
//#include"sutil/vec_math.h"
//using namespace optix;
namespace dropOut_tracing
{
    const int slot_number = 5;
    const int default_specularSubSpaceNumber =21;
    const int default_surfaceSubSpaceNumber = 10; 
    const int record_buffer_width = 1;
    const int max_u = 5;
    const unsigned pixel_unit_size = 10;
    const bool MIS_COMBINATION = true;
    const bool debug_PT_ONLY = false;

    const bool multi_bounce_disable = true; // if true, u mush be 1
    const bool CP_disable = true; // if true, only no control point is valid
    const bool CP_lightsource_only = false; // if true, CP must be on light source
    const bool CP_lightsource_disable = false; // if true, CP can't be on light source
    const bool CP_require = false; // if true, control point is required
    //true  true  true  false false = LSDE  enable
    //true  false true  false true  = LDSDE enable
    //true  false false true  true  = L(A)*DSDE enable 
    const int max_bound = 100;
    const int max_loop = 1000;
    const float light_subpath_caustic_discard_ratio = 0.1;

#define DOT_EMPTY_SURFACEID 0
#define DOT_INVALID_SPECULARID 0
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
        Average, Bound,Dirction, SlotUsageNumber
    };

    RT_FUNCTION __host__ DropOutType pathLengthToDropOutType(int num) {
        if (num < 1)printf("warn: pathLength for drop out type must be larger than 1\n");
        return static_cast<DropOutType>(num - 1);

    }


    struct statistics_data_struct
    {
        float average;
        float variance;
        float bound;
        bool valid;
        RT_FUNCTION __host__ statistics_data_struct():average(0),variance(0),bound(0),valid(false){}
    };

    struct statistics_data
    {
        statistics_data_struct* host_data;
        statistics_data_struct* device_data;
        PGParams* host_PGParams;
        PGParams* device_PGParams;

        int size;
        bool on_GPU;
         
        RT_FUNCTION __host__ statistics_data_struct& operator[](int i)
        {
            return on_GPU ? device_data[i] : host_data[i];
        }
        RT_FUNCTION __host__ PGParams& getPGParams(int i)
        {
            return on_GPU ? device_PGParams[i] : host_PGParams[i];
        }
    };


    struct statistic_record
    {
        float data;
        float data2;
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
    struct pixelRecord
    {
        float record;

        short specularId;
        short eyeId;
        bool is_caustic_record;
        bool is_valid;
        __host__ bool is_caustic()const
        {
            return is_caustic_record;
        }
        __host__ bool valid()const
        {
            return is_valid;
        }
        RT_FUNCTION __host__ operator float() const
        {
            return abs(record);
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

        pixelRecord* pixel_record;
        float* pixel_caustic_refract;
        
        //average specular path number for Q
        float* specular_Q;

        float* CMF_Gamma;

        int record_buffer_core;
        int record_buffer_padding;


        //// Initialize statistics_iteration_count to 0, indicating that no statistics data has been collected yet.
        // It is important not to use statistics data for other operations when the count is 0, as all statistics data will be set to 0 at this time.
        // 当计数为0时，说明尚未有任何的统计数据被统计，注意不要在此时使用统计数据来做别的操作，所有的统计数据在此时都会被设为0
        int statistics_iteration_count;
        float selection_const; 
        float discard_ratio;
        bool pixel_dirty; 

        statistics_data data; 

        RT_FUNCTION int get_specular_label(float3 position, float3 normal) { return specularSubSpace ? classTree::tree_index(specularSubSpace, position, normal) : 0; }
        RT_FUNCTION int get_surface_label(float3 position, float3 normal,float3 dir = make_float3(0.0))
        { return surfaceSubSpace? classTree::tree_index(surfaceSubSpace, position, normal, dir) : 0; }
        RT_FUNCTION __host__ int spaceId2DataId(int specular_id, int surface_id) { return surface_id * specularSubSpaceNumber + specular_id; }
        RT_FUNCTION __host__ int2 dataId2SpaceId(int data_id) { return make_int2(data_id % specularSubSpaceNumber, int(data_id / specularSubSpaceNumber)); }
        RT_FUNCTION __host__ PGParams* get_PGParams_pointer(DropOutType type, int specular_id, int surface_id)
        {
            if (false&&!statistic_available() && data.on_GPU == true)
            {
                printf("warn: you are using the PG data in DEVICE WITHOUT ANY data collected\n");
                return nullptr;
                //printf("warn: you are using the PG data in DEVICE WITHOUT ANY data collected\n");
            }

            int slot_bias = 0;
            slot_bias += int(type) * specularSubSpaceNumber * surfaceSubSpaceNumber;

            int one_dim_id = spaceId2DataId(specular_id, surface_id);
            return &data.getPGParams(one_dim_id);
        }
        RT_FUNCTION __host__ statistics_data_struct& get_statistic_data(DropOutType type, int specular_id, int surface_id)
        {
            if (record_buffer == nullptr)
            {
                printf("dot params is Not Initialized Yet\n");
            }
             
            if (!statistic_available() && data.on_GPU == true)
            {
                printf("warn: you are using the statistic data in DEVICE WITHOUT ANY data collected\n");
            }

            int slot_bias = 0; 
            slot_bias += int(type) * specularSubSpaceNumber * surfaceSubSpaceNumber ;
             
            int one_dim_id = spaceId2DataId(specular_id, surface_id);
            return data[one_dim_id  +  slot_bias];
        } 
        RT_FUNCTION __host__ float& get_statistic_data(DropOutType type, int specular_id, int surface_id, SlotUsage data_slot)
        {
            if(data_slot == SlotUsage::Average)
                return get_statistic_data(type, specular_id, surface_id).average;
            printf("dataslot undefined\n");
            return get_statistic_data(type, specular_id, surface_id).average; 
        }
         
        RT_FUNCTION __host__ bool statistic_available()
        {
            return statistics_iteration_count != 0;
        }
        RT_FUNCTION __host__ float selection_ratio(int eye_id, int specular_id)
        {
            //return selection_const;
            float weight = CMF_Gamma[eye_id * dropOut_tracing::default_specularSubSpaceNumber + specular_id];
            if (specular_id >= 1)
                weight -= CMF_Gamma[eye_id * dropOut_tracing::default_specularSubSpaceNumber + specular_id - 1];
            return weight / specular_Q[specular_id] * selection_const * (1 - discard_ratio);
        }
        __host__ void image_resize()
        {
            pixel_dirty = true;
        }

         

        RT_FUNCTION __host__ unsigned pixel2unitId(uint2 pixel, uint2 size)const
        {
            uint2 pixel_unit = make_uint2(pixel.x / dropOut_tracing::pixel_unit_size, pixel.y / dropOut_tracing::pixel_unit_size);
            size.x /= dropOut_tracing::pixel_unit_size;
            size.y /= dropOut_tracing::pixel_unit_size;
            return pixel_unit.x * size.y + pixel_unit.y; 
        }
         
        RT_FUNCTION __host__ uint2 Id2pixel(unsigned id, uint2 size)const
        {
            return make_uint2(id / size.y, id % size.y);
        }

        RT_FUNCTION __host__ unsigned pixel2Id(uint2 xy, uint2 size)const
        {
            return xy.x * size.y + xy.y;
        }

        RT_FUNCTION float get_caustic_prob(uint2 pixel, uint2 wh)
        {
            if (pixel_dirty)return 0.5;
            else 
            {
                int id = pixel2unitId(pixel, wh);
                return pixel_caustic_refract[id];
            }
        }
    };

};

typedef dropOut_tracing::dropOutTracing_params DropOutTracing_params;
typedef dropOut_tracing::DropOutType DOT_type;
typedef dropOut_tracing::SlotUsage DOT_usage;
typedef dropOut_tracing::statistic_record DOT_record;

#endif