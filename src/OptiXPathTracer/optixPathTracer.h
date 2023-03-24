//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#ifndef OPTIXPATHTRACER_H
#define OPTIXPATHTRACER_H

#define NUM_SUBSPACE 1000
#define NUM_SUBSPACE_LIGHTSOURCE (int(0.2 * NUM_SUBSPACE))

#define RR_MIN_LIMIT
#define MIN_RR_RATE 0.3
#define CONSERVATIVE_RATE 0.2
#define CONNECTION_N 1

#define DIR_JUDGE 0
#define TRAIN_CAUSTIC_WEIGHT 10.0f
#define RMIS_FLAG true

/* 如果想要控制BDPT渲染哪些光路，需要打开这个开关 */
static const bool BDPT_CONTROL = 1;

/* 在PT中，打开这个只会渲染含有glossy顶点的路径 */
/* 在BDPT中，打开这个会在光子路选择是只选择焦散光子路，
    即末端是glossy顶点的光子路 */
static const bool S_ONLY = 1;

/* A 代表 Any*/

/* L - * - E*/
static const bool LAE_ENABLE = 0;

static const bool LE_ENABLE = 1;
static const bool LDE_ENABLE = 0;
static const bool LDSE_ENABLE = 0;

/* LDSAE */
static const bool LDSAE_ENABLE = 0;
/* LDSDE */
static const bool LDSDE_ENABLE = 0;
/* LSAE */
static const bool LSAE_ENABLE = 0;
/* LSE */
static const bool LSE_ENABLE = 0;
/* LSDE ����Ҫ�Ľ�ɢ�Դ */
static const bool LSDE_ENABLE = 1;

/* Path Guiding 开关 */
static const bool PG_ENABLE = 0;

#include"whitted.h"
#include"BDPTVertex.h"
#include"decisionTree/classTree_common.h"
#include"PG_common.h"
struct Subspace
{   
    int jump_bias;
    int id;
    int size;
    float sum_pmf;
    float Q;
};
struct LightTraceParams
{    
    int                      num_core;
    int                      core_padding;
    int                      M;
    int                      M_per_core; 
    BDPTVertex*              ans;
    bool*                    validState;
    int                      launch_frame;

    __host__ int get_element_count()
    {
        return num_core * core_padding;
    }
};
namespace TrainData
{
    struct pathInfo_sample;
    struct pathInfo_node;
}
typedef TrainData::pathInfo_sample preTracePath;
typedef TrainData::pathInfo_node preTraceConnection;

#define PRETRACE_CONN_PADDING 10
struct PreTraceParams
{
    int num_core;
    int padding;
    int iteration;
    preTracePath* paths;
    preTraceConnection* conns;
    __host__ int get_element_count()
    {
        return padding * num_core;
    }

};
struct SubspaceSampler
{
    const BDPTVertex* LVC;
    Subspace* subspace;
    float* cmfs;
    int* jump_buffer;
    
    int* glossy_index;
    int* glossy_subspace_num;
    int* glossy_subspace_bias;

    int vertex_count;
    int path_count;
    int glossy_count;
};
struct envInfo
{
    cudaTextureObject_t tex;
    float* cmf;
    float r;
    float3 center;
    int size;
    int width;
    int height;
    int divLevel;
    int ssBase;
    int light_id;
    bool valid;
    __device__ __host__ float projectPdf() { return 1; } 

    RT_FUNCTION __host__ int coord2index(int2 coord) const
    {
        return coord.x + coord.y * width;
    }
    RT_FUNCTION __host__ int2 index2coord(int index)const
    {
        int w = index % width;
        int h = index / width;
        return make_int2(w, h);
    }
    RT_FUNCTION __host__ float2 coord2uv(int2 coord)const
    {
        float u, v;
        u = coord.x / float(width);
        v = coord.y / float(height);
        return make_float2(u, v);
    }
    RT_FUNCTION __host__ int2 uv2coord(float2 uv)const
    {
        int x = uv.x * width;
        int y = uv.y * height;
        x = min(x, width - 1);
        y = min(y, height - 1);
        return make_int2(x, y);
    }
};

RT_FUNCTION __host__ float3 uv2dir(float2 uv)
{
    float3 dir;

    float theta, phi, x, y, z;
    float u = uv.x;
    float v = uv.y;

    phi = asinf(2 * v - 1.0);
    theta = u / (0.5 * M_1_PIf) - M_PIf;

    dir.x = cos(phi) * sin(theta);
    dir.y = cos(M_PIf * 0.5f - phi);
    dir.z = cos(phi) * cos(theta);
    return dir;
}
RT_FUNCTION __host__ float2 dir2uv(float3 dir)
{
    float2 uv;
    float theta = atan2f(dir.x, dir.z);
    float phi = M_PIf * 0.5f - acosf(dir.y);
    float u = (theta + M_PIf) * (0.5f * M_1_PIf);
    float v = 0.5f * (1.0f + sin(phi));
    uv = make_float2(u, v);

    return uv;
}

struct subspaceMacroInfo
{
    int subspaceNum;
    classTree::tree_node* eye_tree;
    classTree::tree_node* light_tree;
    float* Q;
    float* CMFGamma;
    float* CMFCausticGamma;
    float* caustic_ratio;
    RT_FUNCTION float Gamma(int eye_id, int light_id)
    {
        if (CMFGamma && Q)
        {
            return light_id == 0 ? CMFGamma[eye_id * NUM_SUBSPACE + light_id] :
                CMFGamma[eye_id * NUM_SUBSPACE + light_id] - CMFGamma[eye_id * NUM_SUBSPACE + light_id - 1]; 
        }
        return 1;
    }
    RT_FUNCTION float gamma_ss(int eye_id, int light_id)
    {
        if (CMFGamma && Q)
        {
            return Gamma(eye_id,light_id) / Q[light_id];
        }// Gamma[eye_id * NUM_SUBSPACE + light_id] / Q[light_id];
        return 1;
    }
}; 
struct EstimationParams
{
    float4* ref_buffer;
    int pre_trace_frame;
};
struct PTParams :whitted::LaunchParams
{
    LightTraceParams lt;
    SubspaceSampler sampler;
    PreTraceParams pre_tracer;
    subspaceMacroInfo subspace_info;
    envInfo sky;
    EstimationParams estimate_pr;

    PG_params pg_params;
};
typedef PTParams MyParams;


enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_LIGHTSUBPATH = 0,
    RAY_TYPE_EYESUBPATH = 0,
    RAY_TYPE_EYESUBPATH_SIMPLE = 2,
    RAY_TYPE_COUNT = 3
}; 

enum RayHitType
{
    RAYHIT_TYPE_LIGHTSOURCE = 0,
    RAYHIT_TYPE_NORMAL = 1, 
    RAYHIT_TYPE_COUNT = 2
};

struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};


struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3  emission_color;
    float3  diffuse_color;
    float4* vertices;
};

namespace TrainData
{
    struct nVertex
    {
        float3 position;
        float3 dir;
        float3 normal;
        float3 weight;//pdf for eye vertex, light contri for light vertex
        float3 color;
        float pdf;//light vertex only, for fast compute the overall sampled path pdf 
        float save_t;
        int last_id;//for cached light vertex
        int materialId;
        int label_id;
        int depth;

        bool isBrdf;
        //bool isDirection;
        bool valid;
        __host__ RT_FUNCTION nVertex() :valid(false) {}
        __host__ __device__
            void setLightSourceFlag(bool dir_light = false)
        {
            materialId = dir_light ? -2 : -1;
        }
        __host__ __device__
            bool isLightSource()const
        {
            return materialId < 0;
        }
        __host__ __device__
            bool isDirLight()const
        {
            return materialId == -2;
        }

        __host__ __device__
            bool brdf_weight()const
        {
            return !isBrdf;
        }

        __host__ __device__
            bool isAreaLight()const
        {
            return materialId == -1;
        }
        __host__ RT_FUNCTION nVertex(const BDPTVertex& a, bool eye_side) :
            position(a.position), normal(a.normal), color(a.color), materialId(a.materialId), pdf(a.pdf), valid(true), label_id(a.subspaceId), isBrdf(a.isBrdf), save_t(0), depth(a.depth)
            //isDirection(a.is_DIRECTION())
        {
            dir = a.depth == 0 ? make_float3(0.0) : normalize(a.lastPosition - a.position);
            weight = eye_side ? make_float3(pdf) : a.flux;
            if (eye_side == false && a.depth == 0)
            {
                if (a.type == BDPTVertex::Type::QUAD) setLightSourceFlag(false);
                if (a.is_DIRECTION()) setLightSourceFlag(true);
            }
        }

    };
    struct pathInfo_node
    {
        float3 A_position;
        float3 B_position;
        float3 A_dir_d;
        float3 B_dir_d;
        float3 A_normal_d;
        float3 B_normal_d;
        float peak_pdf;
        //about peak_pdf:
        // = a.pdf * b.contri when generated from tracing
        // = a.pdf * b.contri / Q_b after transform
        int path_id;
        int label_A;//empty until the inital tree is constructed
        int label_B;//empty until the inital tree is constructed
        bool valid;
        bool light_source;
        RT_FUNCTION pathInfo_node() :valid(false) {}
        RT_FUNCTION pathInfo_node(nVertex& a, nVertex& b) :
            A_position(a.position), B_position(b.position), A_dir_d(a.dir), B_dir_d(b.dir), valid(true), light_source(b.isLightSource()), label_B(b.label_id)
        {
            A_normal_d = a.normal;
            B_normal_d = b.normal;
            A_dir_d = a.dir;
            B_dir_d = b.dir;
            //if (dot(A_normal_d, A_dir_d) < 0)A_normal_d = - A_normal_d;
            //if (dot(B_normal_d, B_dir_d) < 0)B_normal_d = - B_normal_d;

            peak_pdf = a.weight.x * float3weight(b.weight) * b.brdf_weight() * a.brdf_weight();
            set_eye_depth(a.depth);
        }
        RT_FUNCTION int get_eye_depth() { return label_A; }
        RT_FUNCTION void set_eye_depth(int depth) { label_A = depth; }
        __host__ RT_FUNCTION float3 B_normal()const
        {
            return B_normal_d;
        }
        __host__ RT_FUNCTION float3 A_normal()const
        {
            return A_normal_d;
        }

        __host__ RT_FUNCTION float3 B_dir()const
        {
            return B_dir_d;
        }
        __host__ RT_FUNCTION float3 A_dir()const
        {
            return A_dir_d;
        }
    };
    struct pathInfo_sample
    {
        float3 contri;
        float sample_pdf;
        float fix_pdf;
        int begin_ind;
        int end_ind;
        int choice_id;
        int caustic_id;
        int2 pixel_id;
        bool valid;
        bool is_caustic; 
    };

}

#endif // !OPTIXPATHTRACER_H