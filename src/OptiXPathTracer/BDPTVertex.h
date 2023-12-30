#ifndef __BDPTVERTEX_H
#define __BDPTVERTEX_H


#include <optix.h>
#include"rt_function.h" 
#include"light_parameters.h" 
#include"MaterialData.h"

//struct specularIdRecord
//{
//    short id;
//    RT_FUNCTION __host__ short get()const
//    {
//        return abs(id);
//    }
//    RT_FUNCTION __host__ short set(unsigned a)
//    {
//        return id = static_cast<short> (-a);
//    }
//    RT_FUNCTION __host__ bool is_specular()const
//    {
//        return id < 0;
//    }
//    RT_FUNCTION __host__ specularIdRecord() :id(0) {}
//};
struct BDPTVertex
{
    float3 position;
    //should be normalized
    float3 normal;

    //flux is actually the local contribution term
    float3 flux;
    float3 color;

    //lastPosition: the position of the previous vertex, used to compute RMIS weight
    float3 lastPosition;
    
    //pg_lightPdf is only used for path guiding, too. 
    //float2 pg_lightPdf;

    //cache the RMIS weight for eye sub-path tracing
    float3 RMIS_pointer_3;
    float3 shade_normal;
    //to save the uv coordinate of the light source
    float2 uv;

    //cache the RMIS weight for light sub-path tracing
    float RMIS_pointer;
    float last_lum;

    //used for RMIS computing, dot(incident_direction, normal of the previous vertex)
    float lastNormalProjection;

    //pdf for the sub-path corresponding to THIS vertex to be sampled.
    float pdf;
    //used for RMIS computing, pdf for (previous vertex) -> (this vertex)
    float singlePdf;
    //used for RMIS computing, the single pdf of the previous vertex.
    float lastSinglePdf;

    //float d;     //can be replaced by RIS_pointer, consider to remove

    // --- Latest Update ---
    float inverPdfEst;

    long long path_record;//consider to remove
    short specular_record;
    short materialId;

    short subspaceId;//subspace ID, consider to rename
    short depth;


    //used for RMIS computing
    short lastZoneId;
    enum Type
    {
        SPHERE, QUAD, DIRECTION, ENV, HIT_LIGHT_SOURCE, ENV_MISS, NORMALHIT, DROPOUT_NOVERTEX, VertexTypeNum
    };
    short type = Type::QUAD;


    bool isOrigin;
    bool inBrdf = false;
    bool lastBrdf = false;
    bool isBrdf = false;

    bool isLastVertex_direction; //if this vertex comes from the directional light 

    __host__ __device__ BDPTVertex() :isBrdf(false), lastBrdf(false), path_record(0), specular_record() {}
    __host__ __device__ bool is_LL_DIRECTION()const { return isLastVertex_direction; }
    __host__ __device__ bool is_DIRECTION()const { return type == BDPTVertex::Type::DIRECTION||type == BDPTVertex::Type::ENV; }
    __host__ __device__ bool hit_lightSource()const { return type == BDPTVertex::Type::ENV_MISS||type == BDPTVertex::Type::HIT_LIGHT_SOURCE; }
    __host__ __device__ float contri_float() { return flux.x + flux.y + flux.z; } 
    __host__ __device__ const float3& get_shade_normal()const
    {
        return shade_normal;
    }
    __host__ __device__ void set_shade_normal(float3 sn)
    {
        shade_normal = sn;
    }
    template<typename B, typename T = MaterialData::Pbr>
    __host__ __device__ T  getMat(B mats)const
    {
        if (type != BDPTVertex::Type::NORMALHIT)
        {
            printf("call get Mat in undesigned case!!! %d\n", type);
            return mats[0];
        }
        T mat = mats[materialId];
        mat.base_color = make_float4(color, 1);
        mat.shade_normal = get_shade_normal();
        return mat;
    }

    RT_FUNCTION __host__ short get_specular_id()const
    {
        return abs(specular_record);
    }
    RT_FUNCTION __host__ void set_specular_id(unsigned a)
    {
        specular_record = -a;
    }
    RT_FUNCTION __host__ bool is_specluar()const
    {
        return specular_record < 0;
    }
};

struct BDPTPath
{
#define MAX_VERTEX_NUMBER 3
    BDPTVertex v[MAX_VERTEX_NUMBER];
    int size;
    RT_FUNCTION BDPTVertex& operator()(int i=0)
    {

        return v[(size - 1 - i) % MAX_VERTEX_NUMBER];
    }

    RT_FUNCTION BDPTVertex& currentVertex()
    {
        return this->operator()(0);
    }
    RT_FUNCTION BDPTVertex& nextVertex()
    {
        return this->operator()(-1);
    }
    RT_FUNCTION BDPTVertex& lastVertex()
    {
        return this->operator()(1);
    } 
    RT_FUNCTION void push(BDPTVertex & vertex)
    {
        this->operator()(-1) = vertex;
        size++;

    }
    RT_FUNCTION void clear()
    {
        size = 0;
    }
    RT_FUNCTION void pop()
    {
        size--;
    }
    RT_FUNCTION void push( )
    { 
        size++; 
    }
    RT_FUNCTION bool hit_lightSource()
    {
        return currentVertex().type == BDPTVertex::Type::HIT_LIGHT_SOURCE || currentVertex().type == BDPTVertex::Type::ENV_MISS;
    }
};
struct RAWVertex
{
    BDPTVertex v;
    int lastZoneId;
    int valid = 0;
};
#define float3weight(a) ((a).x + (a).y + (a).z) 
#endif