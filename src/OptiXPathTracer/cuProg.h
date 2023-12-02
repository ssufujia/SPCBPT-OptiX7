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

#pragma once
#ifndef CUPROG_H
#define CUPROG_H

#include <sutil/vec_math.h>
#include <cmath>

#include "whitted.h"
#include "optixPathTracer.h"
#include "BDPTVertex.h" 
#include "decisionTree/classTree_device.h"
#include "pathControl.h"
#define SCENE_EPSILON 1e-3f


#define ISINVALIDVALUE(ans) (ans.x>100000.0f|| isnan(ans.x)||ans.y>100000.0f|| isnan(ans.y)||ans.z>100000.0f|| isnan(ans.z))
#define VERTEX_MAT(v) (v.getMat(Tracer::params.materials))
struct labelUnit
{
    float3 position;
    float3 normal;
    float3 dir;

    float2 uv;
    float2 tri_uv;
    int objectId;
    int tri_id;
    int type;
    bool light_side;
    RT_FUNCTION labelUnit(float3 position, float3 normal, float3 dir, float2 uv, float2 tri_uv, int objectId, int tri_id, bool light_side, int type = 0) :
        position(position), normal(normal), dir(dir), uv(uv), tri_uv(tri_uv), objectId(objectId), tri_id(tri_id), light_side(light_side), type(type) {}
    RT_FUNCTION labelUnit(float3 position, float3 normal, float3 dir, bool light_side) : position(position), normal(normal), dir(dir), light_side(light_side) {}

    RT_FUNCTION int getLabel();

};


//RT_FUNCTION float max(float a, float b)
//{
//	return a > b ? a : b;
//}
 
RT_FUNCTION bool refract(float3& r, float3 i, float3 n, float ior)
{ 
    float eta = ior;


    float3 nn = n;
    float3 ii = -i;
    float negNdotV = dot(ii, nn);

    if (negNdotV > 0.0f)
    {
        eta = ior;
        nn = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f) {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3(0.f);
        return false;
    }
    else {
        r = normalize(eta * ii - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }


    //if (dot(n, i) > 0) eta = 1 / ior;
    //float cosThetaI = dot(n, i);
    //float sin2ThetaI = 1 - cosThetaI * cosThetaI;
    //float sin2ThetaT = eta * eta * sin2ThetaI;
    //if (sin2ThetaT >= 1)
    //{
    //    return false;
    //}

    //float cosThetaT = sqrt(1 - sin2ThetaT);
    //
    //r = eta * -i + (eta * cosThetaI - cosThetaT) * n;

    ////printf("length test %f\n", length(r));
    //return true;
}
RT_FUNCTION bool isRefract(float3 normal, float3 in_dir, float3 out_dir)
{
    return dot(normal, in_dir) * dot(normal, out_dir) < 0;
}

namespace Tracer {
    using namespace whitted;

    extern "C" {
        __constant__ MyParams params;
    }
}

RT_FUNCTION float connectRate_SOL(int eye_label, int light_label, float lum_sum)
{
    return Tracer::params.subspace_info.gamma_ss(eye_label, light_label) * lum_sum * CONNECTION_N;
}

RT_FUNCTION float3 connectRate_SOL(int eye_label, int light_label, float3 lum_sum)
{
    return Tracer::params.subspace_info.gamma_ss(eye_label, light_label) * lum_sum * CONNECTION_N;
}


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    __forceinline__ __device__ void transform(float3& p) const
    {
        p = make_float3(dot(p, m_tangent), dot(p, m_binormal), dot(p, m_normal));
    }
    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};
//return true in probability rr_rate, rr_rate is assume to be within (0, 1)
RT_FUNCTION bool RR_TEST(unsigned& seed, float rr_rate)
{
    if (rnd(seed) < rr_rate)return true;
    return false;
}
RT_FUNCTION void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

RT_FUNCTION void sample_small_lobe_1(const float u1, const float u2, float alpha, float3& p)
{
    // Sample a small lobe around axis y.
    const float r = sqrtf(1-powf((1-u1), 2/(1+alpha)));
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

RT_FUNCTION void sample_small_lobe_2(const float u1, const float u2, float alpha, float3& p)
{
    // Sample a small lobe around axis y.
    const float r = powf(u1, alpha);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

RT_FUNCTION void sample_small_lobe_3(const float u1, const float u2, float roughness, float3& p)
{
    // Sample a small lobe around axis y.
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (roughness * roughness - 1.0f) * u1));
    const float r = sqrtf(1-cosTheta*cosTheta);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

RT_FUNCTION float2 sample_reverse_cosine(float3 N, float3 dir)
{
    Onb onb(N);
    onb.transform(dir);

    float r1 = dir.x * dir.x + dir.y * dir.y;
    float r = sqrtf(r1);
    float cosPhi = dir.x / r;
    float sinPhi = dir.y / r;
    float phiA = acos(cosPhi);
    float phiB = asin(sinPhi);
    float phi = phiB > 0 ? phiA : 2 * M_PI - phiA;

    float r2 = phi / 2.0f / M_PIf; 
    return make_float2(r1, r2);
}

RT_FUNCTION float pdf_small_lobe_1(float3 N, float3 dir, float alpha)
{
    Onb onb(N);
    onb.transform(dir);

    float r1 = dir.x * dir.x + dir.y * dir.y;
    float theta = asin(r1);
    float pdf = (alpha + 1) / (2 * M_PI) * powf(cos(theta), alpha);
    return pdf;
}

RT_FUNCTION float pdf_small_lobe_2(float3 N, float3 dir, float alpha)
{
    Onb onb(N);
    onb.transform(dir);

    float r1 = dir.x * dir.x + dir.y * dir.y;
    float theta = asin(r1);
    float pdf = 1/(2*M_PI*alpha) * powf(sin(theta), 1/alpha-2)*cos(theta);
    return pdf;
}

RT_FUNCTION float pdf_small_lobe_3(float3 N, float3 dir, float roughness)
{
    Onb onb(N);
    onb.transform(dir);

    float r1 = dir.x * dir.x + dir.y * dir.y;
    float theta = asin(r1);
    float cosTheta = cosf(theta);
    float a_2 = roughness * roughness;
    float tmp = (1 + (a_2 - 1) * cosTheta);
    float pdf = (cosTheta*a_2) / (M_PI*tmp*tmp);
    return pdf;
}

struct envInfo_device :envInfo
{

    RT_FUNCTION int coord2index(int2 coord)const
    {
        return coord.x + coord.y * width;
    }
    RT_FUNCTION int2 index2coord(int index)const
    {
        int w = index % width;
        int h = index / width;
        return make_int2(w, h);
    }
    RT_FUNCTION float2 coord2uv(int2 coord)const
    {
        float u, v;
        u = coord.x / float(width);
        v = coord.y / float(height);
        return make_float2(u, v);
    }
    RT_FUNCTION int2 uv2coord(float2 uv)const
    {
        int x = uv.x * width;
        int y = uv.y * height;
        x = min(x, width - 1);
        y = min(y, height - 1);
        return make_int2(x, y);
    }
    RT_FUNCTION float2 coord2uv(int2 coord, unsigned int& seed)const
    {
        float r1 = rnd(seed), r2 = rnd(seed);
        float u, v;
        u = float(coord.x + r1) / float(width);
        v = float(coord.y + r2) / float(height);
        return make_float2(u, v);
    }
    RT_FUNCTION float3 reverseSample(float2 uv)const
    {
        return uv2dir(uv);
    }
    RT_FUNCTION float3 sample(unsigned int& seed)const
    {
        float index = rnd(seed);
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < cmf[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
        int2 coord = index2coord(l);        
        float2 uv = coord2uv(coord, seed);
        uv.y = 1-uv.y;
        return uv2dir(uv);
    }
    RT_FUNCTION float3 sample_projectPos(float3 dir, unsigned int& seed)const
    {
        const float r1 = rnd(seed);
        const float r2 = rnd(seed);
        float3 pos;
        Onb onb(dir);
        cosine_sample_hemisphere(r1, r2, pos);

        return 10 * r * (dir) + pos.x * r * onb.m_tangent + pos.y * r * onb.m_binormal + center;
    }
    RT_FUNCTION float2 trace_reverse_uv(float3 position, float3 dir)const
    {
        float3 vec2center = center - position;
        float3 cosThetaProject = dot(vec2center, dir) * dir;
        float3 projectPos_scale = (position + cosThetaProject  - center) / r;

        return sample_reverse_cosine(dir, projectPos_scale);
//        return make_float2(0.5, 0.5);
    }
    RT_FUNCTION float projectPdf()const
    {
        return 1 / (M_PI * r * r);
    }

    RT_FUNCTION int getLabel(float3 dir)const
    {
        float2 uv = dir2uv(dir);
        int2 coord = uv2coord(uv);
        int index = coord2index(coord);


        int2 uv_div = make_int2(
            clamp(static_cast<int>(floorf(uv.x * divLevel)),int(0), int(divLevel - 1)),
            clamp(static_cast<int>(floorf(uv.y * divLevel)),int(0), int(divLevel - 1))           
            );

        int res_id = uv_div.x * divLevel + uv_div.y; 

        return NUM_SUBSPACE - 1 - res_id;
    }
    RT_FUNCTION float3 getColor(float3 dir)const
    {
        float2 uv = dir2uv(dir); 
       // return make_float3(pdf(dir)) * 5;
        return make_float3(tex2D<float4>(tex, uv.x, uv.y)); 
    }

    RT_FUNCTION float3 color(float3 dir)const
    {
//        printf("dir sample %f %f %f, %f %f %f\n", dir.x, dir.y, dir.z, getColor(dir).x, getColor(dir).y, getColor(dir).z);
        return getColor(dir);
    } 
    RT_FUNCTION float pdf(float3 dir)const
    {
        float2 uv = dir2uv(dir);
        uv.y = 1-uv.y;
        int2 coord = uv2coord(uv);
        int index = coord2index(coord);

        float pdf1 = index == 0 ? cmf[index] : cmf[index] - cmf[index - 1];

        //if (luminance(color(dir)) / (pdf1 * size / (4 * M_PI)) > 1000)
        //{
            //rtPrintf("%d %d\n", coord.x,coord.y); 
            //return 1000;
        //}
        return pdf1 * size / (4 * M_PI);
    } 
};

#define SKY (*(reinterpret_cast<const envInfo_device*>(&Tracer::params.sky)))


namespace Tracer {
    RT_FUNCTION int binary_sample(float* cmf, int size, unsigned int& seed, float& pmf, float cmf_range = 1.0)
    {

        float index = rnd(seed) * cmf_range;
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < cmf[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
        pmf = l == 0 ? cmf[l] : cmf[l] - cmf[l - 1];
        return l;
    }

    struct SubspaceSampler_device :public SubspaceSampler
    {
        RT_FUNCTION const BDPTVertex& sampleSecondStage(int subspaceId, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = subspace[subspaceId].jump_bias;
            int end_index = begin_index + subspace[subspaceId].size;
            //sample_pmf = 1.0 / subspace[subspaceId].size;
            //int index = rnd(seed) * subspace[subspaceId].size + begin_index;
            //printf("error %f %f %f\n", *(cmfs + begin_index), 1.0 / subspace[subspaceId].size, *(cmfs + end_index - 1));
            int index = binary_sample(cmfs + begin_index, subspace[subspaceId].size, seed, sample_pmf) + begin_index;
            //        if (params.lt.validState[index] == false)
            //            printf("error found");
                    //printf("index get %d %d\n",index, jump_buffer[index]);
            return LVC[jump_buffer[index]];
        }


        RT_FUNCTION int uniformSampleGlossy(unsigned int& seed, float& sample_pmf)
        {
            sample_pmf = 1.0 / glossy_count;
            int index = rnd(seed) * glossy_count;

            return glossy_index[index];
        }

        RT_FUNCTION const BDPTVertex& SampleGlossySecondStage(int subspaceId, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = glossy_subspace_bias[subspaceId];
            int end_index = begin_index + glossy_subspace_num[subspaceId];

            sample_pmf = 1.0 / (end_index - begin_index);
            int index = rnd(seed) * (end_index - begin_index) + begin_index;

            return LVC[glossy_index[index]];
        }

        RT_FUNCTION int SampleGlossySecondStageIndex(int subspaceId, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = glossy_subspace_bias[subspaceId];
            int end_index = begin_index + glossy_subspace_num[subspaceId];

            sample_pmf = 1.0 / (end_index - begin_index);
            int index = rnd(seed) * (end_index - begin_index) + begin_index;
            return glossy_index[index];
        }

        RT_FUNCTION const BDPTVertex& getVertexByIndex(int index) {
            return LVC[index];
        }

        RT_FUNCTION const BDPTVertex& uniformSample(unsigned int& seed, float& sample_pmf)
        {
            sample_pmf = 1.0 / vertex_count;
            int index = rnd(seed) * vertex_count;

            return LVC[jump_buffer[index]];
        }

        RT_FUNCTION const int uniformSampleByPath(unsigned int& seed, const int* pathIndex)
        {
            int index = rnd(seed) * path_count;

            return pathIndex[index];
        }


        RT_FUNCTION int sampleFirstStage(int eye_subsapce, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = eye_subsapce * NUM_SUBSPACE;
            int end_index = begin_index + NUM_SUBSPACE;
            int index = binary_sample(Tracer::params.subspace_info.CMFGamma + begin_index, NUM_SUBSPACE, seed, sample_pmf);
            //        if (params.lt.validState[index] == false)
            //            printf("error found");
                    //printf("index get %d %d\n",index, jump_buffer[index]); 
            //int index = int(rnd(seed) * NUM_SUBSPACE);
            //sample_pmf = 1.0 / NUM_SUBSPACE;
            return index;
        }

        RT_FUNCTION int SampleGlossyFirstStage(int eye_subsapce, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = eye_subsapce * dropOut_tracing::default_specularSubSpaceNumber;
            int end_index = begin_index + dropOut_tracing::default_specularSubSpaceNumber;
            int index = binary_sample(Tracer::params.subspace_info.CMFCausticGamma + begin_index, dropOut_tracing::default_specularSubSpaceNumber, seed, sample_pmf);
            return index;
        }
    };
    struct PayloadBDPTVertex
    {//记得初始化
        BDPTPath path;
        float3 origin;
        float3 ray_direction;
        float3 throughput;
        float3 result;
        float pdf;
        unsigned int seed;
        int depth;
        bool done;
        long long path_record;
        RT_FUNCTION void clear()
        {
            path.clear();
            depth = 0;
            done = false;
            throughput = make_float3(1);
            result = make_float3(0.0);
            path_record = 0;
        }
    };

    //------------------------------------------------------------------------------
    //
    // GGX/smith shading helpers
    // TODO: move into header so can be shared by path tracer and bespoke renderers
    //
    //------------------------------------------------------------------------------

    __device__ __forceinline__ float3 schlick(const float3 spec_color, const float V_dot_H)
    {
        return spec_color + (make_float3(1.0f) - spec_color) * powf(1.0f - V_dot_H, 5.0f);
    }

    __device__ __forceinline__ float vis(const float N_dot_L, const float N_dot_V, const float alpha)
    {
        const float alpha_sq = alpha * alpha;

        const float ggx0 = N_dot_L * sqrtf(N_dot_V * N_dot_V * (1.0f - alpha_sq) + alpha_sq);
        const float ggx1 = N_dot_V * sqrtf(N_dot_L * N_dot_L * (1.0f - alpha_sq) + alpha_sq);

        return 2.0f * N_dot_L * N_dot_V / (ggx0 + ggx1);
    }


    __device__ __forceinline__ float ggxNormal(const float N_dot_H, const float alpha)
    {
        const float alpha_sq = alpha * alpha;
        const float N_dot_H_sq = N_dot_H * N_dot_H;
        const float x = N_dot_H_sq * (alpha_sq - 1.0f) + 1.0f;
        return alpha_sq / (M_PIf * x * x);
    }

    __device__ __forceinline__ float float3sum(float3 c)
    {
        return c.x + c.y + c.z;
    }

    __device__ __forceinline__ float3 linearize(float3 c)
    {
        return make_float3(
            powf(c.x, 2.2f),
            powf(c.y, 2.2f),
            powf(c.z, 2.2f)
        );
    }


    //------------------------------------------------------------------------------
    //
    //
    //
    //------------------------------------------------------------------------------


    static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
    {
        const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }
    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_RADIANCE,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1);

    }
    static __forceinline__ __device__ void traceLightSubPath(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_LIGHTSUBPATH,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_LIGHTSUBPATH,        // missSBTIndex
            u0, u1);

    }
    static __forceinline__ __device__ void traceEyeSubPath(
        OptixTraversableHandle      handle, 
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            //RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            //RayType::RAY_TYPE_COUNT,           // SBT stride
            //RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            u0, u1
        );

    }

    static __forceinline__ __device__ void traceEyeSubPath_simple(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_EYESUBPATH_SIMPLE,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            //RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            //RayType::RAY_TYPE_COUNT,           // SBT stride
            //RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            u0, u1);

    }

    RT_FUNCTION bool visibilityTest(
        OptixTraversableHandle handle, float3 pos_A, float3 pos_B)
    {
        float3 bias_pos = pos_B - pos_A;
        float len = length(bias_pos);
        float3 dir = bias_pos / len;
        unsigned int u0 = __float_as_uint(1.f);
        optixTrace(
            handle,
            pos_A,
            dir,
            SCENE_EPSILON,
            len - SCENE_EPSILON,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RayType::RAY_TYPE_OCCLUSION,      // SBT offset
            RayType::RAY_TYPE_COUNT,          // SBT stride
            RayType::RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
        float res = __uint_as_float(u0);
        if (res > 0.5)
            return true;
        return false;
    }

    RT_FUNCTION bool visibilityTest(
        OptixTraversableHandle handle, const BDPTVertex& eyeVertex, const BDPTVertex& lightVertex)
    {
        if (lightVertex.is_DIRECTION())
        {
            float3 n_pos = -10 * SKY.r * lightVertex.normal + eyeVertex.position;
            return visibilityTest(handle, eyeVertex.position, n_pos);
        }
        else
        {
            return visibilityTest(handle, eyeVertex.position, lightVertex.position);
        }

    }

    __forceinline__ __device__ unsigned int getPayloadDepth()
    {
        return optixGetPayload_3();
    }

    static __forceinline__ __device__ float traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int u0 = __float_as_uint(1.f);
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RayType::RAY_TYPE_OCCLUSION,      // SBT offset
            RayType::RAY_TYPE_COUNT,          // SBT stride
            RayType::RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
        return __uint_as_float(u0);
    }


    __forceinline__ __device__ void setPayloadResult(float3 p)
    {
        optixSetPayload_0(__float_as_uint(p.x));
        optixSetPayload_1(__float_as_uint(p.y));
        optixSetPayload_2(__float_as_uint(p.z));
    }

    __forceinline__ __device__ float getPayloadOcclusion()
    {
        return __uint_as_float(optixGetPayload_0());
    }

    __forceinline__ __device__ void setPayloadOcclusion(float attenuation)
    {
        optixSetPayload_0(__float_as_uint(attenuation));
    }


#define RT_FUNCTION __device__ __forceinline__  
    struct lightSample
    {
        float3 position;
        float3 emission;
        float3 direction;
        float2 uv;
        const Light* bindLight;
        float pdf;
        union
        {
            float dir_pdf;
            float dir_pos_pdf;
        };
        int subspaceId;


        Light::Type type;
        RT_FUNCTION void ReverseSample(const Light& light, float2 uv_)
        {
            bindLight = &light;
            if (light.type == Light::Type::QUAD)
            {
                float r1 = uv_.x;
                float r2 = uv_.y;
                float r3 = 1 - r1 - r2;
                //printf("random %f %f\n", r1, r2);
                // 对四边形进行采样，已经经过验证
                position = light.quad.u * r1 + light.quad.v * r2 + light.quad.corner * r3;
                emission = light.quad.emission;
                //printf("albedo tex %d\n",light.albedoID);
                if(light.albedoID!=0)
                    emission *= make_float3(tex2D<float4>(light.albedoID, uv_.x, uv_.y));
                pdf = 1.0 / light.quad.area;
                pdf /= params.lights.count;
                uv = make_float2(r1, r2);
                {
                    int x_block = clamp(static_cast<int>(floorf(uv.x * light.divLevel)), int(0), int(light.divLevel - 1));
                    int y_block = clamp(static_cast<int>(floorf(uv.y * light.divLevel)), int(0), int(light.divLevel - 1));
                    int lightSpaceId = light.ssBase + x_block * light.divLevel + y_block;
                    subspaceId = NUM_SUBSPACE - lightSpaceId - 1;
                }
            }
            else if (light.type == Light::Type::ENV)
            {
                direction = SKY.reverseSample(uv_);
                emission = SKY.color(direction);
                subspaceId = SKY.getLabel(direction);
                uv = uv;
                pdf = SKY.pdf(direction);
                pdf /= params.lights.count;
            }
        }
        RT_FUNCTION void operator()(const Light& light, unsigned int& seed)
        {
            if (light.type == Light::Type::QUAD)
            {
                float r1 = rnd(seed);
                float r2 = rnd(seed);
                float r3 = 1 - r1 - r2;
                ReverseSample(light, make_float2(r1, r2));
            }
            else if (light.type == Light::Type::ENV)
            {
                direction = SKY.sample(seed);
                emission = SKY.color(direction);
                subspaceId = SKY.getLabel(direction);
                uv = dir2uv(direction);
                pdf = SKY.pdf(direction);
                pdf /= params.lights.count;
            }
            bindLight = &light;
        }
        RT_FUNCTION void operator()(unsigned int& seed)
        {
            int light_id = clamp(static_cast<int>(floorf(rnd(seed) * params.lights.count)), int(0), int(Tracer::params.lights.count - 1));
            this->operator()(params.lights[light_id], seed);

        }
        RT_FUNCTION float3 normal()
        {
            if (bindLight)
            {
                if (bindLight->type == Light::Type::QUAD)
                {
                    return bindLight->quad.normal;
                }
                else
                {
                    return -direction;
                }
            }

            return make_float3(0);
        }
        RT_FUNCTION float3 trace_direction()const
        {
            return (bindLight->type == Light::Type::ENV) ? -direction : direction;
        }
        RT_FUNCTION void traceMode(unsigned int& seed)
        {
            if (bindLight->type == Light::Type::QUAD)
            {
                float r1 = rnd(seed);
                float r2 = rnd(seed);

                Onb onb(bindLight->quad.normal);
                cosine_sample_hemisphere(r1, r2, direction);
                onb.inverse_transform(direction);
                dir_pdf = abs(dot(direction, bindLight->quad.normal)) / M_PIf;
            }
            else if (bindLight->type == Light::Type::ENV)
            {
                position = SKY.sample_projectPos(direction, seed);
                dir_pos_pdf = SKY.projectPdf();
            }
        }
    };

    static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
    {
        const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    template<typename T = PayloadRadiance>
    static __forceinline__ __device__ T* getPRD()
    {
        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

#include<cuda.h>
    RT_FUNCTION float sqr(float x) { return x * x; }

    RT_FUNCTION float SchlickFresnel(float u)
    {
        float m = clamp(1.0f - u, 0.0f, 1.0f);
        float m2 = m * m;
        return m2 * m2 * m; // pow(m,5)
    }

    RT_FUNCTION float GTR1(float NDotH, float a)
    {
        if (a >= 1.0f) return (1.0f / M_PIf);
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return (a2 - 1.0f) / (M_PIf * logf(a2) * t);
    }

    RT_FUNCTION float GTR2(float NDotH, float a)
    {
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return a2 / (M_PIf * t * t);
    }

    RT_FUNCTION float smithG_GGX(float NDotv, float alphaG)
    {
        float a = alphaG * alphaG;
        float b = NDotv * NDotv;
        return 1.0f / (NDotv + sqrtf(a + b - a * b));
    }

    RT_FUNCTION float D(const float3& wh, const float3& n)
    {
        //BeckmannDistribution
        const float alphax = 0.5f;
        const float alphay = 0.5f;// what does this mean?
        float wDotn = dot(wh, n);
        float TanTheta = length(cross(wh, n)) / wDotn;
        float tan2Theta = TanTheta * TanTheta;
        if (isinf(tan2Theta)) return 0.;
        float cos4Theta = wDotn * wDotn * wDotn * wDotn;
        return exp(-tan2Theta * (1.0f / (alphax * alphax))) /
            (M_PIf * alphax * alphay * cos4Theta);
    }

    RT_FUNCTION float fresnel(float cos_theta_i, float cos_theta_t, float eta)
    {
        const float rs = (cos_theta_i - cos_theta_t * eta) /
            (cos_theta_i + eta * cos_theta_t);
        const float rp = (cos_theta_i * eta - cos_theta_t) /
            (cos_theta_i * eta + cos_theta_t);

        return 0.5f * (rs * rs + rp * rp);
    }

    RT_FUNCTION float3 logf(float3 v)
    {
        return make_float3(log(v.x), log(v.y), log(v.z));
    }
    RT_FUNCTION float lerp(const float& a, const float& b, const float t)
    {
        return a + t * (b - a);
    }

    RT_FUNCTION float Lambda(const float3& w, const float3& n)
    {
        //BeckmannDistribution
        const float alphax = 0.5f;
        const float alphay = 0.5f;// what does this mean?
        float wDotn = dot(w, n);
        float absTanTheta = abs(length(cross(w, n)) / wDotn);
        if (isinf(absTanTheta)) return 0.;
        // Compute _alpha_ for direction _w_
        float alpha = alphax;
        //std::sqrt(wDotn * wDotn * alphax * alphax + (1 - wDotn * wDotn) * alphay * alphay);
        float a = 1 / (alpha * absTanTheta);
        if (a >= 1.6f) return 0;
        return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
    }

}
namespace Shift
{

    RT_FUNCTION bool glossy(const MaterialData::Pbr& mat)
    {
        return mat.roughness < 0.2 && max(mat.metallic, mat.trans) >= 0.99;
    }
    RT_FUNCTION bool glossy(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && glossy(Tracer::params.materials[v.materialId]);
    }
#define ROUGHNESS_A_LIMIT 0.001f
    RT_FUNCTION float2 sample_reverse_half(float a, float3 half)
    {
        a = max(ROUGHNESS_A_LIMIT, a);
        float cosTheta = half.z;
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = half.y / sinTheta;
        float cosPhi = half.x / sinTheta;
        float phiA = acos(cosPhi);
        float phiB = asin(sinPhi);
        float phi = phiB > 0 ? phiA : 2 * M_PI - phiA;

        float r1 = phi / 2.0f / M_PIf;
        float A = cosTheta * cosTheta;

        float r2 = (1 - A) / (A * a * a - A + 1);
        return make_float2(r1, r2);
    }

    RT_FUNCTION float2 sample_reverse_half(float a, const float3& N, float3 half)
    {
        Onb onb(N);
        onb.transform(half);
        return sample_reverse_half(a, half);
    }
    RT_FUNCTION float3 sample_half(float a, const float2 uv)
    {
        a = max(ROUGHNESS_A_LIMIT, a);

        float r1 = uv.x;
        float r2 = uv.y;
        float phi = r1 * 2.0 * M_PIf;
        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        return half;
    }

    RT_FUNCTION float3 sample_half(float a, const float3& N, const float2 uv)
    {
        Onb onb(N);
        float3 half = sample_half(a, uv);
        onb.inverse_transform(half);
        return half;
    }

    //eta 无论是否取倒数，结果都一样
    //wh 无论是朝着V还是L的方向，结果也都一样
    RT_FUNCTION float dwh_dwi_refract(float3 wh, float3 V, float3 L, float eta)
    {
        float sqrtDenom = dot(L, wh) + eta * dot(V, wh);
        float dwh_dwi =
            std::abs((eta * eta * dot(V, wh)) / (sqrtDenom * sqrtDenom));
        return dwh_dwi;
    }

    RT_FUNCTION float3 refract_half_fine(float3 in_dir, float3 out_dir, float3 normal, float eta)
    {
        if (dot(normal, in_dir) < 0) eta = 1 / eta;
        float3 ans;
        ans = eta * out_dir + in_dir;
        float3 half = normalize(ans);
        return dot(half, normal) > 0 ? half : -half;
    }
    RT_FUNCTION float3 refract_half(float3 in_dir, float3 out_dir, float3 normal, float eta)
    {
        eta = max(eta, 1 / eta);
        float3 ans;
        if (abs(dot(in_dir, normal)) > abs(dot(out_dir, normal)))
        {
            ans = eta * in_dir + out_dir;
        }
        else
        {
            ans = in_dir + eta * out_dir;
        }
        ans = normalize(ans);
        return dot(ans, normal) > 0 ? ans : -ans;
    }

}
namespace Tracer
{
    RT_FUNCTION float3 Eval_Transmit(const MaterialData::Pbr& mat, const float3& normal, const float3& V_vec, const float3& L_vec)
    {
        //float3 gN = normal;
        float3 N = mat.shade_normal;
        float3 V = V_vec;
        float3 L = L_vec;
        float NDotL = dot(N, L);
        float NDotV = dot(N, V);
        //float gNDotL = dot(gN, L);

        float mateta = mat.eta;
        float eta = 1 / mateta;
        if (NDotL > 0)
        {
            eta = 1 / eta;
            N = -N;
        }

        if (NDotL == 0 || NDotV == 0) return make_float3(0);
        float refract;
        if ((1 - NDotV * NDotV) * eta * eta >= 1)// ȫ    
            refract = 1;
        else
            refract = 0;
        float3 Cdlin = make_float3(mat.base_color);
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.
        float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
        float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        float3 wh = normalize(L + V * eta);
        if (dot(wh, N) < 0) wh = -wh;

        // Same side?
        if (dot(L, wh) * dot(V, wh) > 0) return make_float3(0);

        float sqrtDenom = dot(L, wh) + eta * dot(V, wh);
        float factor = 1 / eta;
        float3 T = mat.trans * make_float3(sqrt(mat.base_color.x), sqrt(mat.base_color.y), sqrt(mat.base_color.z));
        float roughg = sqr(mat.roughness * 0.5f + 0.5f);
        float Gs = 1 / (1 + Lambda(V, N) + Lambda(L, N));
        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(dot(N, wh), a);
        float FH = SchlickFresnel(dot(V, wh));
        float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
        float F = fresnel(abs(dot(V, wh)), abs(dot(L, wh)), eta);
        float3 out = (1 - refract) * (1.f - F) * T *
            std::abs(Ds * Gs * eta * eta *
                abs(dot(L, wh)) * abs(dot(V, wh)) * factor * factor /
                (NDotL * NDotL * sqrtDenom * sqrtDenom));
        return out;

    }
    RT_FUNCTION float3 Eval(const MaterialData::Pbr& mat, const float3& normal, const float3& V, const float3& L)
    {
        //printf("s normal %f %f %f gn %f %f %f\n", mat.shade_normal.x, mat.shade_normal.y, mat.shade_normal.z,normal.x,normal.y,normal.z);
        //float3 N = normal; 
        float3 N = mat.shade_normal;
        float3 gN = normal; 
        float mateta = mat.eta;
        float NDotL = dot(N, L);
        float gNDotL = dot(gN, L);
        float NDotV = dot(N, V);
        float gNDotV = dot(gN, V);
        float eta = 1 / mateta;        
        if (mat.trans<0.1&& (gNDotL * gNDotV <= 0.0f || NDotL * NDotV <= 0))
            return make_float3(0);// return Eval_Transmit(mat, normal, V, L); 
        if (mat.trans > 0.9 && (gNDotL * gNDotV * NDotL * NDotV <= 0))return make_float3(0.0);
        if (NDotL * NDotV <= 0)return Eval_Transmit(mat, normal, V, L);
        if (mat.metallic + mat.base_color.x + mat.base_color.y + mat.base_color.z <= 0)return make_float3(0);
        

        if (NDotL < 0.0f && NDotV < 0.0f)
        {
            N = -N;
            eta = 1 / eta;
            NDotL *= -1;
            NDotV *= -1;
        }
        float3 H = normalize(L + V);
        float NDotH = dot(N, H);
        float LDotH = dot(L, H);
        float VDotH = dot(V, H);

        float refract;
        if ((1 - NDotV * NDotV) * eta * eta >= 1)// ȫ    
            refract = 1;
        else
            refract = 0;

        float3 Cdlin = make_float3(mat.base_color);
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

        float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
        float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
        float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and mix in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
        float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
        float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

        // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
        // 1.25 scale is used to (roughly) preserve albedo
        // Fss90 used to "flatten" retroreflection based on roughness
        float Fss90 = LDotH * LDotH * mat.roughness;
        float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
        float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

        // specular
        //float aspect = sqrt(1-mat.anisotrokPic*.9);
        //float ax = Max(.001f, sqr(mat.roughness)/aspect);
        //float ay = Max(.001f, sqr(mat.roughness)*aspect);
        //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(NDotH, a);
        float FH = SchlickFresnel(LDotH);
        float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
        float roughg = sqr(mat.roughness * 0.5f + 0.5f);
        float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

        // sheen
        float3 Fsheen = FH * mat.sheen * Csheen;

        // clearcoat (ior = 1.5 -> F0 = 0.04)
        float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
        float Fr = lerp(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

        float trans = mat.trans;

        float cosThetaI = abs(dot(N, V));
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = eta * eta * sin2ThetaI;
        float cosThetaT = 1;
        if (sin2ThetaT <= 1)
        {
            cosThetaT = sqrt(1 - sin2ThetaT);
        }
        float F = fresnel(cosThetaI, cosThetaT, eta);

        float3 out = (((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
            * (1.0f - mat.metallic))
            * (1 - trans * (1 - F) * (1 - refract))
            ;
        if (trans > 0)
            out = out + Gs * Ds * (1 - trans * (1 - refract) * (1 - F));
        else
            out = out + Gs * Ds * Fs;
        return out;
    } 

    RT_FUNCTION float3 Sample_force
        (const MaterialData::Pbr& mat, const float3& N, const float3& V, unsigned int& seed, float refract_force = 0, float3 position = make_float3(0.0), bool use_pg = false)
    {
        if (use_pg && Tracer::params.pg_params.pg_enable)
        {
            //printf("A %f\n", Tracer::params.pg_params.guide_ratio);
            if (rnd(seed) < Tracer::params.pg_params.guide_ratio)
                return Tracer::params.pg_params.sample(seed, position);
        }

        //float3 N = normal;
        //float3 V = in_dir;
        //prd.origin = state.fhp;
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float r3 = rnd(seed);
        float r4 = rnd(seed);
        float3 dir;
        //float3 normal = N;
        float3 normal = mat.shade_normal;
        {
            Onb onb(normal);
            cosine_sample_hemisphere(r1, r2, dir);
            float r3 = rnd(seed);
            if (r3 < 0.5f)
                dir = -dir;
            onb.inverse_transform(dir);
            //return dir;
        }


        float mateta = mat.eta;
        float eta = 1 / mateta;
        if (dot(normal, V) < 0)
        {
            eta = 1 / eta;
            normal = -normal;
        } 

        float NdotV = abs(dot(normal, V));
        float transRatio = mat.trans;
        float transprob = rnd(seed);
        float refractRatio;
        float refractprob = rnd(seed);
        float probability = rnd(seed);
        float diffuseRatio = 0.5f * (1.0f - mat.metallic);// *(1 - transRatio);
        if (transprob < transRatio) // sample transmit
        {
            float refractRatio = 0;
            float temp = (1 - NdotV * NdotV) * (eta * eta);
            if (temp < 1)
                refractRatio = 1 - fresnel(NdotV, sqrt(1 - temp), eta);
            if (RR_TEST(seed, refract_force))
            {
                Onb onb(normal); // basis
                float a = mat.roughness;

                float phi = r3 * 2.0f * M_PIf;

                float cosTheta = sqrtf((1.0f - r4) / (1.0f + (a * a - 1.0f) * r4));
                float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
                float sinPhi = sinf(phi);
                float cosPhi = cosf(phi);
                float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                onb.inverse_transform(half);

                if (dot(V, normal) == 0) return -V;

                float cosThetaI = dot(half, V);
                float sin2ThetaI = 1 - cosThetaI * cosThetaI;
                float sin2ThetaT = eta * eta * sin2ThetaI;

                if (sin2ThetaT <= 1)
                {
                    float cosThetaT = sqrt(1 - sin2ThetaT);
                    //float y = -sqrt(1 - cosThetaI * cosThetaI * eta * eta)/sqrt(sin2ThetaI);
                    //float x = -(y + eta) * cosThetaI;
                    float3 L = normalize(eta * -V + (eta * cosThetaI - cosThetaT) * half);

                    if (cosThetaI < 0)//V与half在两侧。要求VdotH*LdotH<0
                    {
                        float LdotH = dot(L, half);
                        L = L - 2 * LdotH * half;
                    }
                    return L;
                }
                else
                {
                    return half * 2 * dot(half, V) - V; // ȫ����
                }
            }
        }
        Onb onb(normal); // basis

        if (probability < diffuseRatio) // sample diffuse
        {
            cosine_sample_hemisphere(r1, r2, dir);
            onb.inverse_transform(dir);
        }
        else
        {
            float a = mat.roughness;

            float phi = r1 * 2.0f * M_PIf;

            float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            onb.inverse_transform(half);

            dir = 2.0f * dot(V, half) * half - V; //reflection vector
        }
        return dir;
    }


    RT_FUNCTION float Pdf_force(MaterialData::Pbr& mat, float3 normal, float3 V, float3 L, float refract_force = 0, float3 position = make_float3(0.0), bool use_pg = false, bool force_refract = false)
    {
#ifdef BRDF
        if (mat.brdf)
            return 1.0f;// return abs(dot(L, normal));
#endif

        float transRatio = mat.trans;
        //float3 n = normal;
        float3 n = mat.shade_normal;
        float mateta = mat.eta;
        //        float eta = dot(L, n) > 0 ? (mateta) : (1/mateta);     
        float eta = mateta;
        if (dot(n, V) < 0)
        {
            eta = 1 / eta;
            n = -n;
        }
        float pdf = 0;
        float NdotV = abs(dot(V, n));
        float NdotL = abs(dot(L, n));
        float3 wh = -normalize(V + L * eta);

        float HdotV = abs(dot(V, wh));
        float HdotL = abs(dot(L, wh));

        float specularAlpha = mat.roughness;
        float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

        float diffuseRatio = 0.5f * (1.f - mat.metallic);// *(1 - transRatio);
        float specularRatio = 1.f - diffuseRatio;

        float3 half;
        half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        float ratio = 1.0f / (1.0f + mat.clearcoat);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PIf);

        float cosThetaI = abs(dot(wh, V));
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        float refractRatio = 0;
        float temp = (1 - NdotV * NdotV) / (eta * eta);
        if (temp < 1)
            refractRatio = 1 - fresnel(NdotV, sqrt(1 - temp), eta);
        //float refractRatio = 0.5;
        refractRatio = refract_force;


        pdf = (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * (1 - transRatio * refractRatio);// normal reflect

        if (sin2ThetaT <= 1 && dot(L, wh) * dot(V, wh) < 0)//refract
        {
            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            float sqrtDenom = eta * dot(L, wh) + dot(V, wh);
            float dwh_dwi =
                std::abs((eta * eta * dot(L, wh)) / (sqrtDenom * sqrtDenom));
            float a = max(0.001f, mat.roughness);
            float Ds = GTR2(abs(dot(wh, n)), a);
            float pdfTrans = Ds * abs(dot(n, wh)) * dwh_dwi;
            //printf("r pdf: %f\n", transRatio * pdfTrans * refractRatio);
            pdf += transRatio * pdfTrans * refractRatio;
        }

        cosThetaI = abs(dot(
            half, V));
        sin2ThetaI = 1 - cosThetaI * cosThetaI;
        sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        if (sin2ThetaT > 1)// full reflect
        { 
            pdf += (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * transRatio * refractRatio;
        }
         

        if (use_pg && Tracer::params.pg_params.pg_enable)
        {
            pdf *= 1 - Tracer::params.pg_params.guide_ratio;
            pdf += Tracer::params.pg_params.guide_ratio * Tracer::params.pg_params.pdf(position, L);
        }
        return pdf;
    }


    RT_FUNCTION float3 Sample(const MaterialData::Pbr& mat, const float3& N, const float3& V, unsigned int& seed, float3 position = make_float3(0.0), bool use_pg = false)
    {
        if (use_pg && Tracer::params.pg_params.pg_enable && Shift::glossy(mat) == false)
        { 
            if (rnd(seed) < Tracer::params.pg_params.guide_ratio)
            { 
                return Tracer::params.pg_params.sample(seed, position);
            }
        }

        //float3 N = normal;
        //float3 V = in_dir;
        //prd.origin = state.fhp;
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float r3 = rnd(seed);
        float r4 = rnd(seed);
        float3 dir;
        float3 normal = mat.shade_normal;
        {
            Onb onb(normal);
            cosine_sample_hemisphere(r1, r2, dir);
            float r3 = rnd(seed);
            if (r3 < 0.5f)
                dir = -dir;
            onb.inverse_transform(dir);
            //return dir;
        }


        float mateta = mat.eta;
        float eta = 1 / mateta;
        if (dot(normal, V) < 0)
        {
            eta = 1 / eta;
            normal = -normal;
        }
         

        float NdotV = abs(dot(normal, V));
        float transRatio = mat.trans;
        float transprob = rnd(seed);
        float refractRatio;
        float refractprob = rnd(seed);
        float probability = rnd(seed);
        float diffuseRatio = 0.5f * (1.0f - mat.metallic);// *(1 - transRatio);
        if (transprob < transRatio) // sample transmit
        {
            float refractRatio = 0;
            float temp = (1 - NdotV * NdotV) * (eta * eta);
            if (temp < 1)
                refractRatio = 1 - fresnel(NdotV, sqrt(1 - temp), eta);
            if (refractprob < refractRatio)
            {
                Onb onb(normal); // basis
                float a = mat.roughness;

                float phi = r3 * 2.0f * M_PIf;

                float cosTheta = sqrtf((1.0f - r4) / (1.0f + (a * a - 1.0f) * r4));
                float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
                float sinPhi = sinf(phi);
                float cosPhi = cosf(phi);
                float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                onb.inverse_transform(half);

                if (dot(V, normal) == 0) return -V;

                float cosThetaI = dot(half, V);
                float sin2ThetaI = 1 - cosThetaI * cosThetaI;
                float sin2ThetaT = eta * eta * sin2ThetaI;

                if (sin2ThetaT <= 1)
                {
                    float cosThetaT = sqrt(1 - sin2ThetaT);
                    //float y = -sqrt(1 - cosThetaI * cosThetaI * eta * eta)/sqrt(sin2ThetaI);
                    //float x = -(y + eta) * cosThetaI;
                    float3 L = normalize(eta * -V + (eta * cosThetaI - cosThetaT) * half);

                    if (cosThetaI < 0)//V与half在两侧。要求VdotH*LdotH<0
                    {
                        float LdotH = dot(L, half);
                        L = L - 2 * LdotH * half;
                    }
                    return L;
                }
                else
                {
                    return half * 2 * dot(half, V) - V; // ȫ����
                }
            }
        }
        Onb onb(normal); // basis

        if (probability < diffuseRatio) // sample diffuse
        {
            cosine_sample_hemisphere(r1, r2, dir);
            onb.inverse_transform(dir);
        }
        else
        {
            float a = mat.roughness;

            float phi = r1 * 2.0f * M_PIf;

            float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            onb.inverse_transform(half);

            dir = 2.0f * dot(V, half) * half - V; //reflection vector
        }
        return dir;
    }

    RT_FUNCTION float Pdf(MaterialData::Pbr& mat, float3 normal, float3 V, float3 L, float3 position = make_float3(0.0), bool use_pg = false)
    { 
#ifdef BRDF
        if (mat.brdf)
            return 1.0f;// return abs(dot(L, normal));
#endif

        float transRatio = mat.trans;
        float3 n = mat.shade_normal;
        float mateta = mat.eta;
        //        float eta = dot(L, n) > 0 ? (mateta) : (1/mateta);     
        float eta = mateta;
        if (dot(n, V) < 0)
        {
            eta = 1 / eta;
            n = -n;
        }
        float pdf = 0;
        float NdotV = abs(dot(V, n));
        float NdotL = abs(dot(L, n));
        float3 wh = -normalize(V + L * eta);

        float HdotV = abs(dot(V, wh));
        float HdotL = abs(dot(L, wh));

        float specularAlpha = mat.roughness;
        float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

        float diffuseRatio = 0.5f * (1.f - mat.metallic);// *(1 - transRatio);
        float specularRatio = 1.f - diffuseRatio;

        float3 half;
        half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        float ratio = 1.0f / (1.0f + mat.clearcoat);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PIf);

        float cosThetaI = abs(dot(wh, V));
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        float refractRatio = 0;
        float temp = (1 - NdotV * NdotV) / (eta * eta);
        if (temp < 1)
            refractRatio = 1 - fresnel(NdotV, sqrt(1 - temp), eta);
        //float refractRatio = 0.5;

        pdf = (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * (1 - transRatio * refractRatio);// normal reflect

        if (sin2ThetaT <= 1 && dot(L, wh) * dot(V, wh) < 0)//refract
        {
            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            float sqrtDenom = eta * dot(L, wh) + dot(V, wh);
            float dwh_dwi =
                std::abs((eta * eta * dot(L, wh)) / (sqrtDenom * sqrtDenom));
            float a = max(0.001f, mat.roughness);
            float Ds = GTR2(abs(dot(wh, n)), a);
            float pdfTrans = Ds * abs(dot(n, wh)) * dwh_dwi;
            //printf("r pdf: %f\n", transRatio * pdfTrans * refractRatio);
            pdf += transRatio * pdfTrans * refractRatio;
        }

        cosThetaI = abs(dot(
            half, V));
        sin2ThetaI = 1 - cosThetaI * cosThetaI;
        sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        if (sin2ThetaT > 1)// full reflect
        {
            //printf("l pdf: %f\n", (diffuseRatio * pdfDiff + specularRatio * pdfSpec));
            pdf += (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * transRatio * refractRatio;
        }
         

        if (use_pg && Tracer::params.pg_params.pg_enable && Shift::glossy(mat) == false)
        {
            pdf *= 1 - Tracer::params.pg_params.guide_ratio;
            pdf += Tracer::params.pg_params.guide_ratio * Tracer::params.pg_params.pdf(position, L);
        }
        return pdf;
    }
    RT_FUNCTION float rrRate(float3 color)
    {
        float rr_rate = fmaxf(color);

#ifdef RR_MIN_LIMIT
        rr_rate = rr_rate < MIN_RR_RATE ? MIN_RR_RATE : rr_rate;
#endif
        // if (rr_rate > 0.5)return 0.5;
        return rr_rate;
    }
    RT_FUNCTION float rrRate(const MaterialData::Pbr& pbr)
    {
        return rrRate(make_float3(pbr.base_color));
        }
    RT_FUNCTION float3 contriCompute(const BDPTVertex* path, int path_size)
    {
        //要求：第0个顶点为eye，第size-1个顶点为light
        float3 throughput = make_float3(1);
        const BDPTVertex& light = path[path_size - 1];
        const BDPTVertex& lastMidPoint = path[path_size - 2];
        float3 lightLine = lastMidPoint.position - light.position;
        float3 lightDirection = normalize(lightLine);
        float lAng = dot(light.normal, lightDirection);
        if (lAng < 0.0f)
        {
            return make_float3(0.0f);
        }
        float3 Le = light.flux * lAng;
        throughput *= Le;
        for (int i = 1; i < path_size; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            throughput /= dot(line, line);
        }
        for (int i = 1; i < path_size - 1; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            const BDPTVertex& nextPoint = path[i + 1];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);
             
            MaterialData::Pbr mat = VERTEX_MAT(midPoint);
            throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
                * Eval(mat, midPoint.normal, lastDirection,nextDirection);
        }
        return throughput;
    }
    RT_FUNCTION float SPCBPT_MIS_sum_compute(const BDPTVertex* path, int path_size)
    {
        float eye_pdf[MAX_PATH_LENGTH_FOR_MIS];
        float3 light_contri[MAX_PATH_LENGTH_FOR_MIS];
        light_contri[path_size - 1] = path[path_size - 1].flux;
        eye_pdf[0] = 1;
        eye_pdf[1] = path[1].pdf;
        float pdf_sum = 0;
        for (int i = 2; i < path_size; i++)
        {
            const BDPTVertex& midVertex = path[i];
            const BDPTVertex& lastVertex = path[i - 1];
            float3 LL_pos = path[i - 2].position;
            float3 diff = midVertex.position - lastVertex.position;
            float3 in_dir = normalize(LL_pos - lastVertex.position);
            float3 out_dir = normalize(diff);
            MaterialData::Pbr mat = VERTEX_MAT(lastVertex);
            float rr_rate = Tracer::rrRate(mat);
            eye_pdf[i] = eye_pdf[i - 1] * Tracer::Pdf(mat, lastVertex.normal, in_dir, out_dir, lastVertex.position, true)
                / dot(diff,diff) * abs(dot(out_dir,midVertex.normal)) * rr_rate;
        }
        for (int i = path_size - 2; i >= 2; i--)
        {
            const BDPTVertex& midVertex = path[i];
            const BDPTVertex& lastVertex = path[i + 1];
            float3 LL_pos = path[i + 2].position;
              
            float3 diff = midVertex.position - lastVertex.position;
            float3 in_dir = normalize(LL_pos - lastVertex.position);
            float3 out_dir = normalize(diff);
            float G = abs(dot(out_dir, midVertex.normal)) * abs(dot(out_dir, lastVertex.normal)) / dot(diff, diff);
            if (i == path_size - 2)light_contri[i] = light_contri[i + 1] * G * M_1_PI;
            else
            {
                MaterialData::Pbr mat = VERTEX_MAT(lastVertex);
                float3 f = Tracer::Eval(mat, lastVertex.normal, in_dir, out_dir);
                light_contri[i] = light_contri[i + 1] * G * f;
            }
        }
        for (int i = 2; i <= path_size - 2; i++)
        {
            //if (Shift::glossy(path[i]) || Shift::glossy(path[i - 1]))
            //    continue;
            labelUnit eye_label_unit(path[i].position, path[i].normal, normalize(path[i - 1].position - path[i].position), false);
            int eye_label = eye_label_unit.getLabel(); 

            labelUnit light_label_unit(path[i + 1].position, path[i + 1].normal, normalize(path[i + 2].position - path[i + 1].position), false);
            int light_label = i == path_size - 2 ? path[path_size - 1].subspaceId : light_label_unit.getLabel();
            pdf_sum += eye_pdf[i] * connectRate_SOL(eye_label, light_label, float3weight(light_contri[i + 1]));
            //if (eye_pdf[i] * connectRate_SOL(eye_label, light_label, float3weight(light_contri[i + 1])) < 0)
            //{
            //    printf("zero minus%d %d %f %f %f \n", i, path_size - i - 1, eye_pdf[i] * connectRate_SOL(eye_label, light_label, float3weight(light_contri[i + 1])),
            //        eye_pdf[i], connectRate_SOL(eye_label, light_label, float3weight(light_contri[i + 1]))
            //    );
            //}
        }
        pdf_sum += eye_pdf[path_size - 1];
        return pdf_sum;
    }

    RT_FUNCTION float pdfCompute(const BDPTVertex* path, int path_size, int strategy_id)
    {

        int eyePathLength = strategy_id;
        int lightPathLength = path_size - eyePathLength; 
        /*光源默认为面光源上一点，因此可以用cos来近似模拟其光照效果，如果是点光源需要修改以下代码*/

        float pdf = 1.0;
        if (lightPathLength > 0)
        {
            const BDPTVertex& light = path[path_size - 1];
            pdf *= light.pdf;
        }
        if (lightPathLength > 1)
        {
            const BDPTVertex& light = path[path_size - 1];
            const BDPTVertex& lastMidPoint = path[path_size - 2];
            float3 lightLine = lastMidPoint.position - light.position;
            float3 lightDirection = normalize(lightLine);
            pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

            /*因距离和倾角导致的pdf*/
            for (int i = 1; i < lightPathLength; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                float3 line = midPoint.position - lastPoint.position;
                float3 lineDirection = normalize(line);
                pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
            }

            for (int i = 1; i < lightPathLength - 1; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                const BDPTVertex& nextPoint = path[path_size - i - 2];
                float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                 
                MaterialData::Pbr mat = VERTEX_MAT(midPoint);
                float rr_rate = rrRate(mat);
                pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
            }

        }
        /*由于投影角导致的pdf变化*/
        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*采样方向的概率*/
        for (int i = 1; i < eyePathLength - 1; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            const BDPTVertex& nextPoint = path[i + 1];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);
             
            MaterialData::Pbr mat = VERTEX_MAT(midPoint);
            float rr_rate = rrRate(mat);
            pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position, true) * rr_rate;
        }
        return pdf;
    }

    RT_FUNCTION float MISWeight_SPCBPT(const BDPTVertex* path, int path_size, int strategy_id)
    {
        if (strategy_id <= 1 || strategy_id == path_size)
        {
            return pdfCompute(path, path_size, strategy_id);
        }
        int eyePathLength = strategy_id;
        int lightPathLength = path_size - eyePathLength;
        float pdf = 1.0;

        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*采样方向的概率*/
        for (int i = 1; i < eyePathLength - 1; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            const BDPTVertex& nextPoint = path[i + 1];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);
             
            MaterialData::Pbr mat = VERTEX_MAT(midPoint);
            float rr_rate = fmaxf(midPoint.color);
            pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
        }

        float3 light_contri = make_float3(1.0);

        if (lightPathLength > 0)
        {
            const BDPTVertex& light = path[path_size - 1];
            light_contri *= light.flux;
        }

        if (lightPathLength > 1)
        {
            const BDPTVertex& light = path[path_size - 1];
            const BDPTVertex& lastMidPoint = path[path_size - 2];

            /*因距离和倾角导致的pdf*/
            for (int i = 1; i < lightPathLength; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                float3 line = midPoint.position - lastPoint.position;
                float3 lineDirection = normalize(line);
                light_contri *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection)) * abs(dot(lastMidPoint.normal, lineDirection));
            }

            for (int i = 1; i < lightPathLength - 1; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                const BDPTVertex& nextPoint = path[path_size - i - 2];
                float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                 
                MaterialData::Pbr mat = VERTEX_MAT(midPoint);
                light_contri *= Tracer::Eval(mat, midPoint.normal, lastDirection, nextDirection);
            }

        }

        float3 position;
        float3 dir;
        float3 normal;
        int eye_subspace_id = 0;
        int light_subspace_id = 0;
        position = path[strategy_id - 1].position;
        normal = path[strategy_id - 1].normal;
        dir = normalize(path[strategy_id - 2].position - path[strategy_id - 1].position);
        labelUnit lu(position, normal, dir, false);
        eye_subspace_id = lu.getLabel();

        if (strategy_id == path_size - 1)
        {
            light_subspace_id = path[strategy_id].subspaceId;
        }
        else
        {
            position = path[strategy_id].position;
            normal = path[strategy_id].normal;
            dir = normalize(path[strategy_id + 1].position - path[strategy_id].position);
            labelUnit lu(position, normal, dir, true);
            light_subspace_id = lu.getLabel();
        }
        return pdf * float3weight(connectRate_SOL(eye_subspace_id, light_subspace_id, light_contri));
    }
} // namespace Tracer
RT_FUNCTION int labelUnit::getLabel()
{
    if (light_side)
    {
        if (Tracer::params.subspace_info.light_tree)
            return classTree::getLabel(Tracer::params.subspace_info.light_tree, position, normal, dir);
    }
    else
    {
        if (Tracer::params.subspace_info.eye_tree)
            return classTree::getLabel(Tracer::params.subspace_info.eye_tree, position, normal, dir);
    }
    return 0;

}

namespace TrainData
{

    struct nVertex_device :nVertex
    {
        RT_FUNCTION nVertex_device(const nVertex& a, const nVertex_device& b, bool eye_side)
        {
            position = a.position;
            dir = normalize(b.position - a.position);
            normal = a.normal;
            weight = eye_side ? b.forward_eye(a) : b.forward_light(a);
            pdf = eye_side ? weight.x : b.forward_light_pdf(a);
            color = a.color;
            materialId = a.materialId;
            valid = true;
            label_id = a.label_id;
            isBrdf = a.isBrdf;

            depth = b.depth + 1;
            //save_t = 0;

            //brdf_tracing_branch(a, b, eye_side);

        }
        RT_FUNCTION void brdf_tracing_branch(const nVertex& a, const nVertex_device& b, bool eye_side)
        {
            if (b.isBrdf == false && a.isBrdf == true)
            {
                save_t = length(a.position - b.position);
            }
            else if (b.isBrdf == true)
            {
                save_t = b.save_t + length(a.position - b.position);
            }
            if (a.isBrdf == true && b.isBrdf == false)
            {
                if (eye_side == false)
                {
                    weight = b.weight * abs(dot(dir, b.normal)) * (b.isLightSource() ? make_float3(1.0) : b.color);
                }
                else
                {
                    weight = b.weight * b.color;
                }
            }
            else if (a.isBrdf == true && b.isBrdf == true)
            {
                weight = b.weight * b.color;
            }
            else if (a.isBrdf == false && b.isBrdf == true)
            {
                weight = b.weight * abs(dot(a.normal, dir)) * b.color;
                weight /= save_t * save_t;
                save_t = 0;
            }

        }
        RT_FUNCTION int get_label()
        {
            if (isLightSource())
            {
                return label_id;
            }
            return 0;//to be rewrote
        }
        RT_FUNCTION nVertex_device(const BDPTVertex& a, bool eye_side) :nVertex(a, eye_side) {}
        RT_FUNCTION nVertex_device() {}

        RT_FUNCTION float forward_light_pdf(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);

            if (isLightSource())
            {
                if (isAreaLight())
                {
                    g *= abs(dot(normal, c_dir));
                    return pdf * g * 1.0 / M_PI;
                }
                if (isDirLight())
                {
                    g = abs(dot(c_dir, b.normal));
                    return pdf * g * Tracer::params.sky.projectPdf();
                }
            }
             
            MaterialData::Pbr mat = getMat(Tracer::params.materials);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, false);
            float RR_rate = Tracer::rrRate(mat);
            return pdf * d_pdf * RR_rate * g;
        }

        RT_FUNCTION float3 forward_eye(const nVertex& b)const
        {
            if (b.isDirLight())
            {
                float3 c_dir = -b.normal;
                 
                MaterialData::Pbr mat = getMat(Tracer::params.materials);
                float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
                float RR_rate = Tracer::rrRate(color);
                return weight * d_pdf * RR_rate;
            }

            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);
              
            MaterialData::Pbr mat = getMat(Tracer::params.materials);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
            float RR_rate = Tracer::rrRate(color);
            //d_pdf /= isBrdf ? abs(dot(normal, c_dir)) : 1;
            return weight * d_pdf * RR_rate * g;
        }

        RT_FUNCTION float3 forward_areaLight(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) * abs(dot(c_dir, normal)) / dot(vec, vec);

            return weight * g;
        }

        RT_FUNCTION float3 forward_dirLight(const nVertex& b)const
        {
            return weight * abs(dot(normal, b.normal));
        }

        RT_FUNCTION float3 forward_light_general(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);

            float g = isBrdf ?
                abs(dot(c_dir, b.normal)) / dot(vec, vec) :
                abs(dot(c_dir, b.normal)) * abs(dot(c_dir, normal)) / dot(vec, vec);


            MaterialData::Pbr mat = getMat(Tracer::params.materials); 
            float3 d_contri = Tracer::Eval(mat, normal, dir, c_dir);
            return weight * g * d_contri;

        }
        RT_FUNCTION float3 forward_light(const nVertex& b)const
        { 
            if (isAreaLight())
                return forward_areaLight(b);
            if (isDirLight())
                return forward_dirLight(b);
            return forward_light_general(b);
        }
        RT_FUNCTION float3 local_contri(const nVertex_device& b) const
        {
            float3 vec = b.position - position;
            float3 c_dir = b.isDirLight() ? -b.normal : normalize(vec);
            MaterialData::Pbr mat = getMat(Tracer::params.materials);
            return Tracer::Eval(mat, normal,c_dir, dir);
        }

    };

}

RT_FUNCTION void init_EyeSubpath(BDPTPath& p, float3 origin, float3 direction)
{
    p.push();
    p.currentVertex().position = origin;
    p.currentVertex().flux = make_float3(1.0);
    p.currentVertex().pdf = 1.0f;
    p.currentVertex().RMIS_pointer = 0;
    p.currentVertex().normal = direction;
    p.currentVertex().isOrigin = true;
    p.currentVertex().depth = 0;
    p.currentVertex().singlePdf = 1.0;

    p.nextVertex().singlePdf = 1.0f;
}

/* 把光采样信息装到一个bdpt顶点中 */
RT_FUNCTION void init_vertex_from_lightSample(Tracer::lightSample& light_sample, BDPTVertex& v)
{
    v.position = light_sample.position;
    v.normal = light_sample.normal();
    v.flux = light_sample.emission;
    v.pdf = light_sample.pdf;
    v.singlePdf = v.pdf;
    v.isOrigin = true;
    v.isBrdf = false;
    v.subspaceId = light_sample.subspaceId;
    v.depth = 0;
    v.materialId = light_sample.bindLight->id;
    v.RMIS_pointer = 1;
    v.uv = light_sample.uv;
    if (light_sample.bindLight->type == Light::Type::QUAD)
    {
        v.type = BDPTVertex::Type::QUAD;
    }
    else if (light_sample.bindLight->type == Light::Type::ENV)
    {
        v.type = BDPTVertex::Type::ENV;
    }
    //其他光源的状况待补充
}
namespace Tracer
{
    /* Latest Update! */
    RT_FUNCTION BDPTVertex FastTrace(const BDPTVertex& a, float3 direction, bool& success)
    {
        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = direction;
        payload.origin = a.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);
        
        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath_simple(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            success = 0;
            return BDPTVertex();
        }
        success = 1;
        return payload.path.currentVertex();
    }
}

RT_FUNCTION void DOT_pushRecordToBuffer(DOT_record& record, unsigned int& putId, int bufferBias)
{
    const DropOutTracing_params& dot_params = Tracer::params.dot_params;
    if (putId > dot_params.record_buffer_padding)
    { 
        printf("error in drop out tracing: pushing more record than the record buffer padding\n \
            will discord the over-flowing record\n \
            consider to increase record_buffer_width in dropOutTracing_common.h to assign more memory for record buffer\n");
        return;
    }
    dot_params.record_buffer[putId + bufferBias] = record;
    putId++;
}

struct statistic_payload
{
    unsigned* putId;
    unsigned bufferBias;
    unsigned SP_label;
    unsigned CP_label;
    bool uvvalid;
    dropOut_tracing::DropOutType type;
    dropOut_tracing::statistics_data_struct data;
    bool CP_NOVERTEX;
    dropOut_tracing::PGParams* pg_p;
    RT_FUNCTION dropOut_tracing::statistic_record generate_record(dropOut_tracing::SlotUsage usage)
    {
        dropOut_tracing::statistic_record record(type, SP_label, CP_label, usage); 
        return record;
    }
    RT_FUNCTION bool subspace_valid()const
    {
        if (data.valid)
        {
            if (isnan(data.average) || isinf(data.average) || isnan(data.bound) || isinf(data.bound))
            {
                return false;
            }
        }
        return true;
    }
    RT_FUNCTION void build(BDPTVertex& SP, BDPTVertex& CP, float3 WC, int u)
    {
        DropOutTracing_params& dot_params = Tracer::params.dot_params;
        if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX)
        {
            CP_label = DOT_EMPTY_SURFACEID;
            CP_NOVERTEX = true;
        }
        else
        {
            CP_NOVERTEX = false;
            CP_label = CP.depth == 0 ? dot_params.get_surface_label(CP.position, CP.normal) :
                dot_params.get_surface_label(CP.position, CP.normal, WC);
        }
        SP_label = dot_params.get_specular_label(SP.position, SP.normal);

        if (dot_params.statistic_available())
        {
            data = dot_params.get_statistic_data(dropOut_tracing::pathLengthToDropOutType(u), SP_label, CP_label);
        }
        else
        {
            data = dropOut_tracing::statistics_data_struct();
            data.bound = 1;
        }
        type = dropOut_tracing::pathLengthToDropOutType(u);
        if (dropOut_tracing::PG_reciprocal_estimation_enable == true)pg_p = dot_params.get_PGParams_pointer(type, SP_label, CP_label);

        uvvalid = pg_p->hasLoadln;
        if (dropOut_tracing::PG_reciprocal_estimation_enable == false)uvvalid = false;
     }
    RT_FUNCTION float3 getInitialDirection(float3 normal, unsigned& seed)
    {
        float3 dir;
        if (!uvvalid) {
            Onb onb(RR_TEST(seed, 0.5) ? normal : -normal);
            cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
            onb.inverse_transform(dir);
        }
        else
        {
            float2 uv{0.54,0.14};
           pg_p->predict(uv,seed);
           //printf("predict uv is %f %f and hasloadin is %d\n", uv.x, uv.y,pg_p->hasLoadln);
           dir = uv2dir(uv);
        }
        return dir;
    }
    RT_FUNCTION float getInitialDirectionPdf(float3 normal, float3 direction)
    {
        if (!uvvalid) {
            //Sample in two hemisphere
            return 1.0 / M_PI / 2;
        }
        else {
            return pg_p->pdf(dir2uv(direction));
        }
    }

    RT_FUNCTION bool NO_CP()
    {
        return CP_NOVERTEX;
    }
};
RT_FUNCTION void DOT_pushRecordToBuffer(DOT_record& record, statistic_payload& prd)
{
    DOT_pushRecordToBuffer(record, *prd.putId, prd.bufferBias);
}
namespace Shift
{
    RT_FUNCTION float dh_dwi_estimate(float a, float3 normal, float3 half, float3 v)
    {
        float3 wo = 2.0f * dot(v, half) * half - v;
        float2 origin_uv = sample_reverse_half(a,  half);
        float scale = .0001;
        float2 uv1 = origin_uv + make_float2(scale, 0);
        float2 uv2 = origin_uv + make_float2(0, scale);
        float3 half2 = sample_half(a,  uv2);
        float3 half1 = sample_half(a,  uv1);
        float3 v1 = half1;//2.0 * dot(wo, half1) * half1 - wo;
        float3 v2 = half2;//2.0 * dot(wo, half2) * half2 - wo;
        float3 vv = half;//2.0 * dot(wo, half) * half - wo;
        float area = length(cross(v1 - vv, v2 - vv));//dh duv
        area *= (4 * dot(v, half));
        //printf("dir compare %f %f\n", float3weight(v),float3weight(vv));

        v1 = 2.0 * dot(wo, half1) * half1 - wo;
        v2 = 2.0 * dot(wo, half2) * half2 - wo;
        vv = 2.0 * dot(wo, half) * half - wo;
        float area2 = length(cross(v1 - vv, v2 - vv));//dw duv
        if (origin_uv.y < .7)
        {
            //printf("dh dwi compare %f %f %f\n", area / scale / scale, area2 / scale / scale, area / area2);
        }
        return area2 / scale / scale;
        //float3 reflect = 2.0f * dot(in_dir, normal) * normal - in_dir;

    }
    RT_FUNCTION float robePdf(float a, float3 normal, float3 half, float3 V)
    {
        //return dh_dwi_estimate(a, normal, half, V);
        float cosTheta = abs(dot(half, normal));
        float pdfGTR2 = Tracer::GTR2(cosTheta, a) * cosTheta;
        return pdfGTR2 / (4.0 * abs(dot(half, V)));
    }
    RT_FUNCTION void pdf_valid_compare(float a, float b, float3 normal, float3 halfa, float3 halfb, float3 v1 ,float3 v2)
    {
        float cosThetaa = abs(dot(halfa, normal));
        float cosThetab = abs(dot(halfb, normal));
        float pdfGTR2a = Tracer::GTR2(cosThetaa, a) ;
        float pdfGTR2b = Tracer::GTR2(cosThetab, b) ;
        float o1 = robePdf(a, normal, halfa, v1);// *(4.0 * abs(dot(halfa, v1)));
        float o2 = robePdf(b, normal, halfb, v2);// *(4.0 * abs(dot(halfb, v2)));
        float t1 = dh_dwi_estimate(a, normal, halfa, v1);
        float t2 = dh_dwi_estimate(b, normal, halfb, v2);
        float2 uv1 = sample_reverse_half(a,  halfa);
        float2 uv2 = sample_reverse_half(b,  halfb);
        if(uv2.y<.7 &&uv1.y<.7)
            printf("pdf compare %f %f %f %f %f %f uv info %f %f %f %f\n", o1, o2, t1, t2, o1 / o2, t1 / t2, uv1.x, uv2.y, uv2.x, uv2.y );

    }

    RT_FUNCTION bool RefractionCase(const MaterialData::Pbr& mat)
    {
        return mat.trans > 0.9;
    }

    RT_FUNCTION bool IsCausticPath(const BDPTVertex* path, int path_size)
    {
        for (int i = 1; i < path_size - 1; i++)
        {
            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == false)
                break;

            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == true)
                return true;
        }
        return false;
    }


    RT_FUNCTION bool RefractionCase(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && RefractionCase(Tracer::params.materials[v.materialId]);
    }
     
    RT_FUNCTION float3 SampleControlled(float& duv_dwi, const MaterialData::Pbr& mat, const float3& N, const float3& V, float2 uv, bool is_refract, bool& sample_good)
    {
        sample_good = true;
        float a = max(mat.roughness, ROUGHNESS_A_LIMIT);
        float3 half = sample_half(a, N, uv);
        float cosTheta = abs(dot(half, N));
        float duv_dwh = Tracer::GTR2(cosTheta, a) * cosTheta;
        float3 out_dir;
        if (is_refract)
        {
            sample_good = refract(out_dir, V, half, dot(V, N) > 0 ? mat.eta : 1 / mat.eta);
        }
        else
        {
            out_dir = reflect(-V, half);
        }
        float dwi_dwh = is_refract ? 1 / dwh_dwi_refract(half, V, out_dir, mat.eta) : 4 * abs(dot(half, V));
        duv_dwi = duv_dwh / dwi_dwh;
        return out_dir;
    }

    //简单的路径偏移实现
    //首先统计需要偏移的顶点数目，这个数目等同于从末端往前数的光滑顶点数目
    //计算原先的路径pdf
    //根据偏移顶点数目从末端开始重新追踪
    //重映射时要在最后注意可见性
    struct PathContainer
    {
        BDPTVertex* v;
        int it_step;// +1 or -1
        int m_size;
        RT_FUNCTION PathContainer(BDPTVertex* _v, int _step)
        {
            v = _v;
            it_step = _step;
        }
        RT_FUNCTION PathContainer(BDPTVertex* _v, int _step, int size)
        {
            v = _v;
            it_step = _step;
            m_size = size;
        }
        RT_FUNCTION int size()const { return m_size; }
        RT_FUNCTION int setSize(int i) { m_size = i; }
        RT_FUNCTION BDPTVertex& get(int i)
        {
#define SHIFT_VALID_SIZE 6
            if (i < 0)
            {
                i = m_size + i;
            }

            if (i >= m_size || i < 0)
            {
                printf("wrong Path Container index %d\n", i);
                return *v;
            }
            if (i >= SHIFT_VALID_SIZE)
            {
                printf("wrong path container index - invalid address\n");
            }
            return *(v + i * it_step);
        }
    };

    RT_FUNCTION float3 etaCheck(float3 in_dir, float3 ref_dir, float3 normal)
    {
        float cosA = abs(dot(ref_dir, normal));
        float cosB = abs(dot(in_dir, normal));
        float sin_A = sqrt(1 - cosA * cosA);
        float sin_B = sqrt(1 - cosB * cosB);
        return make_float3(sin_A, sin_B, sin_A / sin_B);
        //printf("refract test %f %f %f\n", sin_A, sin_B, sin_A / sin_B);
    }
    RT_FUNCTION float3 etaCheck(PathContainer& a, float3 anchor)
    {
        float3 out_dir = normalize(a.get(1).position - a.get(0).position);
        float3 in_dir = normalize(anchor - a.get(0).position);
        
        return etaCheck(in_dir, out_dir, a.get(0).normal);
    }

    RT_FUNCTION float map_function(float x, float ratio, float& pdf)
    {
        pdf = 1 / ratio / ratio;
        return x * ratio;
    }
    RT_FUNCTION void refract_state_fill(bool* refract_state, PathContainer& path,float3 anchor, int g)
    { 
        for (int i = 0; i < g; i++)
        {
            float3 in_dir = (i == 0 ? anchor : path.get(i - 1).position) - path.get(i).position;
            float3 out_dir = path.get(i + 1).position - path.get(i).position;
            refract_state[i] = isRefract(path.get(i).normal, in_dir, out_dir);
        }
    }



    RT_FUNCTION float2 scale_map(float2 center, float2 origin, float& jacob)
    {
        float r = .1f;
        jacob = r * r * 4;
        if (center.x < r)center.x = r;
        if (center.y < r)center.y = r;
        if (center.x > 1 - r)center.x = 1 - r;
        if (center.y > 1 - r)center.y = 1 - r;
        if (abs(origin.x - center.x) > r || abs(origin.y - center.y) > r)
        {
            jacob = -1;
            return origin;
        }
        float2 anchor = make_float2(center.x - r, center.y - r);
        return (origin - anchor) / (2 * r);
    }


    struct fractChannelsFlag
    {
        bool xyz[3];
        RT_FUNCTION fractChannelsFlag()
        {
            xyz[0] = false;
            xyz[1] = false;
            xyz[2] = false;
        }
        RT_FUNCTION bool needCheck(float3 f, float rr)
        {
            if (xyz[0] == false && rr < f.x) return true;
            if (xyz[1] == false && rr < f.y) return true;
            if (xyz[2] == false && rr < f.z) return true;
            return false;
        }
        RT_FUNCTION void update(float3 f, float rr)
        {
            if (rr < f.x) xyz[0] = true;
            if (rr < f.y) xyz[1] = true;
            if (rr < f.z) xyz[2] = true;
        }

        RT_FUNCTION bool finish()
        {
            if (xyz[0] == false) return false;
            if (xyz[1] == false) return false;
            if (xyz[2] == false) return false;
            return false;
        }

        RT_FUNCTION float3 valid_channel()
        {
            float3 ans = make_float3(0.0);
            if (xyz[0] == false) ans.x = 1;
            if (xyz[1] == false) ans.y = 1;
            if (xyz[2] == false) ans.z = 1;
            return ans;
        }
    };
    RT_FUNCTION void info_print(PathContainer& path, float3 anchor, float3 dir)
    {  
        return;
        if (path.size() == 3)
        {
            printf("\n\nbegin \n%f %f %f to\n%f %f %f to\n%f %f %f to\n%f %f %f\n\n%f %f %f normal\n\n",
                anchor.x, -anchor.z, anchor.y,
                path.get(0).position.x, -path.get(0).position.z, path.get(0).position.y,
                path.get(1).position.x, -path.get(1).position.z, path.get(1).position.y,
                dir.x, -dir.z, dir.y,
                path.get(0).normal.x, -path.get(0).normal.z, path.get(0).normal.y
                );
        }
    }

    RT_FUNCTION float3 BSDF(const BDPTVertex& LastVertex, const BDPTVertex& MidVertex, const BDPTVertex& NextVertex)
    {
        MaterialData::Pbr mat = MidVertex.getMat(Tracer::params.materials);
        float3 dir_A = LastVertex.position - MidVertex.position;
        float3 dir_B = NextVertex.position - MidVertex.position;
        float3 normal = MidVertex.normal;
        return Tracer::Eval(mat, normal, dir_A, dir_B);
    }
    RT_FUNCTION float GeometryTerm(const BDPTVertex& a, const BDPTVertex& b)
    {
        if (a.type == BDPTVertex::Type::ENV || b.type == BDPTVertex::Type::ENV || a.type == BDPTVertex::Type::ENV_MISS || b.type == BDPTVertex::Type::ENV_MISS)
        {
            printf("Geometry Term call in Env light but we haven't implement it");
        }
        float3 diff = a.position - b.position;
        float3 dir = normalize(diff);
        return abs(dot(dir, a.normal) * dot(dir, b.normal)) / dot(diff, diff);
    }
    RT_FUNCTION float tracingPdf(const BDPTVertex& a, const BDPTVertex& b)
    {
        if (a.depth == 0&& a.type == BDPTVertex::Type::QUAD)// a为面光源，半球面采样
        {
            return GeometryTerm(a, b) * 1 / M_PI * Tracer::visibilityTest(Tracer::params.handle, a, b);
        }
        else
        {
            //TBD 
            return GeometryTerm(a, b) * 1 / M_PI * Tracer::visibilityTest(Tracer::params.handle, a, b);
        }
    }
    struct getClosestPointFunction
    {
        float3 a;
        float3 b;
        float3 c; 
        float3 p;
        RT_FUNCTION getClosestPointFunction(float3 a, float3 b, float3 c, float3 p) :a(a), b(b), c(c), p(p) {}
        RT_FUNCTION float3 getSegDis(float3 aa, float3 bb, float3 pp, float3 normal_plane)
        {
            float3 ap = aa - pp;
            float3 normal = normalize(cross(normal_plane, aa - bb));
            float dis = dot(ap, normal);
            float3 projectPoint = pp + dis * normal;

            float3 aapp = projectPoint - aa;
            float3 bbpp = projectPoint - bb;
            if (dot(aapp, bbpp) < 0)return projectPoint;
            if (dot(aapp, aapp) < dot(bbpp, bbpp))return aa;
            return bb;
        }
        RT_FUNCTION float3 operator()()
        {
            float3 normal = normalize(cross(b - a, c - a));
            float3 ap = a - p;
            
            float dis = dot(ap, normal);
            float3 projectPoint = p + dis * normal;

            float3 aP = a - projectPoint;
            float3 bP = b - projectPoint;
            float3 cP = c - projectPoint;

            float3 ab = a - b;
            float3 bc = b - c;
            float3 ca = c - a;
            if (dot(cross(aP, -ab), cross(bP, -bc)) > 0 && dot(cross(bP, -bc), cross(cP, -ca)) > 0) return projectPoint;

            float3 Ca = getSegDis(a, b, projectPoint, normal);
            float3 Cb = getSegDis(b, c, projectPoint, normal);
            float3 Cc = getSegDis(c, a, projectPoint, normal);

            float3 ans = Ca;
            if (length(Ca - p) > length(Cb - p))ans = Cb;
            if (length(Cb - p) > length(Cc - p))ans = Cc;
            return ans;
        } 

    };

    RT_FUNCTION float getClosestGeometry_upperBound(const Light &light, float3 position,float3 normal,float3& sP)
    {
        Tracer::lightSample light_sample;
        float3 ca = getClosestPointFunction(light.quad.corner, light.quad.u, light.quad.v,position)();
        float3 cb = getClosestPointFunction(-light.quad.corner + light.quad.u + light.quad.v, light.quad.u, light.quad.v, position)();
        float3 c = length(ca - position) < length(cb - position) ? ca: cb;
        float3 diff = c - position;
        float3 dir = normalize(diff);
        sP = c;

        float cos_bound = abs(dot(normal,dir));
        if (abs(dot(normalize(light.quad.corner - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.corner - position), normal));
        if (abs(dot(normalize(light.quad.u - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.u - position), normal));
        if (abs(dot(normalize(light.quad.v - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.v - position), normal));
        if (abs(dot(normalize(-light.quad.corner + light.quad.u + light.quad.v - position), normal)) > cos_bound)
            cos_bound = abs(dot(normalize(-light.quad.corner + light.quad.u + light.quad.v - position), normal));


        return abs(dot(dir, light.quad.normal) ) / dot(diff, diff) * 1 / M_PI * cos_bound;

    }

    /*use to genertate low_difference seq*/
    RT_FUNCTION double halton(int index, int base) {
        double result = 0;
        double f = 1.0 / base;
        int i = index;

        while (i > 0) {
            result += f * (i % base);
            i /= base;
            f /= base;
        }

        return result;
    }

    RT_FUNCTION void printFloat3(float3 v)
    {
        printf("%f %f %f \n", v.x, v.y, v.z);
    }

    //test code 
    //can't ensure its performance
    RT_FUNCTION float get_duv_dwi(const MaterialData::Pbr& mat, const float3& N, const float3& V, const float3& L, bool is_refract)
    {
        float3 half;
        if (is_refract)
        {
            half = refract_half(V,L,N,mat.eta);

//            return 0;
        }
        else
        {
            half = normalize(V + L); 
        }
        float cosTheta = abs(dot(N, half));
        float a = max(mat.roughness, ROUGHNESS_A_LIMIT);

        float duv_dwh = Tracer::GTR2(cosTheta, a) * cosTheta;
        float dwi_dwh = is_refract ? dwh_dwi_refract(half, V, L, mat.eta) : 4 * abs(dot(half, V));
        return duv_dwh / dwi_dwh;
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////Drop Out Tracing Code////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    RT_FUNCTION float tracingPdf(const BDPTVertex& a, const BDPTVertex& b, float3 direction, bool skip_rr, bool skip_visibility = false)
    {
        float pdf = 1;
        if (a.type == BDPTVertex::Type::NORMALHIT)
        {
            MaterialData::Pbr mat = VERTEX_MAT(a);
            float3 diff = b.position - a.position;
            float3 out_dir = normalize(diff);
            pdf = Tracer::Pdf(mat, a.normal, direction, out_dir) * abs(dot(out_dir, b.normal)) / dot(diff, diff);

            if (!skip_rr)
            {
                pdf *= Tracer::rrRate(mat);
            }
            return skip_visibility ? pdf : pdf * Tracer::visibilityTest(Tracer::params.handle, a, b);
        }
        else if (a.type == BDPTVertex::Type::QUAD) {
            return tracingPdf(a, b);
        }
        else {
            printf("tracing pdf is called at undesigned case：%i\n", a.type);
        }
        return pdf;
    }
    // This function retraces the path and stores it in the path container. Returns false if retracing fails.
    // seed is the random number seed. 
    // SP is the specular point
    // incident is the incident direction
    // u is the step size
    //  
    // @param seed: unsigned integer, random number seed
    // @param path: PathContainer, path container to store the retraced path
    // @param SP: BDPTVertex, specular point
    // @param incident: float3, incident direction
    // @param u: integer, step size
    // @param statistic_prd: statistic_payload, interface for using statistic information
    // @return: bool, true if the path is retraced successfully, false otherwise
    RT_FUNCTION bool retracing(unsigned& seed, PathContainer& path, const BDPTVertex& SP, const BDPTVertex& CP, float3 incident, int u, float& pdf)
    {
        path.setSize(u);
        pdf = 1;
        float3 in_dir = incident;
        for (int i = 0; i < u; i++)
        {
            const BDPTVertex& currentVertex = i == 0 ? SP : path.get(i - 1);
            MaterialData::Pbr mat = VERTEX_MAT(currentVertex);
            float3 out_dir = Tracer::Sample(mat, currentVertex.normal, in_dir, seed);

            bool success_hit;
            path.get(i) = Tracer::FastTrace(currentVertex, out_dir, success_hit);
            if (success_hit == false)
            {
                return false;
            }
            if (i == u - 1 && Shift::glossy(path.get(i)))
            {
                return false;
            }
            if (i != u - 1 && !Shift::glossy(path.get(i)))
            {
                return false;
            }
            if (path.get(i).type == BDPTVertex::HIT_LIGHT_SOURCE)
            {
                if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX && i == u - 1)
                {
                    int light_id = path.get(i).materialId;
                    const Light& light = Tracer::params.lights[light_id];
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, path.get(i).uv);
                    init_vertex_from_lightSample(light_sample, path.get(i));
                }
                else
                {
                    return false;
                }
            }
            else if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX && i == u - 1)
            {
                return false;
            }
            //float3 diff = currentVertex.position - path.get(i).position;
            //pdf *= Tracer::Pdf(mat, currentVertex.normal, in_dir, out_dir);
            //pdf *= abs(dot(out_dir, path.get(i).normal)) / dot(diff, diff);
            pdf *= tracingPdf(currentVertex, path.get(i), in_dir, true, true); 
            in_dir = -out_dir;
        }
        if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && !Tracer::visibilityTest(Tracer::params.handle, path.get(-1), CP))
        {
            return false;
        }
        return true;
    }

    RT_FUNCTION bool retracing_general(unsigned& seed, PathContainer& path, const BDPTVertex& SP, const BDPTVertex& CP, float3 incident, float& pdf)
    {
        //printf("retracing CP.type %d    CP.depth %d     u %d\n", CP.type,CP.depth, u);
        path.setSize(SHIFT_VALID_SIZE);
        pdf = 1;
        float3 in_dir = incident;
        int cnt = 0;
        bool success = 0;
        while ( true ) {
            if (cnt >= SHIFT_VALID_SIZE) break;
            const BDPTVertex& currentVertex = ((cnt == 0) ? SP : path.get(cnt - 1));
            MaterialData::Pbr mat = VERTEX_MAT(currentVertex);
            float3 out_dir = Tracer::Sample(mat, currentVertex.normal, in_dir, seed);
            bool success_hit;
            path.get(cnt) = Tracer::FastTrace(currentVertex, out_dir, success_hit);
            if (success_hit == false) break;

            pdf *= tracingPdf(currentVertex, path.get(cnt), in_dir, true, true);

            /* 打到光源 */
            if (path.get(cnt).type == BDPTVertex::HIT_LIGHT_SOURCE) {
                /* L(S)*S */
                if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX) {
                    int light_id = path.get(cnt).materialId;
                    const Light& light = Tracer::params.lights[light_id];
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, path.get(cnt).uv);
                    init_vertex_from_lightSample(light_sample, path.get(cnt));
                    success = 1;
                }
                break;
            }
            else if (!Shift::glossy(path.get(cnt))) {
                /* 打到 diffuse */;
                if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX) {
                    success = 1;
                }
                break;
            }
            in_dir = -out_dir;
            ++cnt;
        }
        if (!success) {
            return false;
        }
        path.setSize(cnt + 1);
        if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && !Tracer::visibilityTest(Tracer::params.handle, path.get(-1), CP))
        {
            return false;
        }
        return true;
    }


    RT_FUNCTION bool retracing_with_reference(unsigned& seed, PathContainer& path, 
        const BDPTVertex& SP, const BDPTVertex& CP, float3 incident, int u, float& pdf, PathContainer& path_ref)
    {
        path.setSize(u);
        pdf = 1;
        float3 in_dir = incident;
        for (int i = 0; i < u; i++)
        {
            const BDPTVertex& currentRef = path_ref.get(i);
            float3 ref_dir = i == 0 ? in_dir : normalize(path_ref.get(i - 1).position - currentRef.position);
            float3 ref_dir2 = normalize(path_ref.get(i + 1).position - currentRef.position);
            bool is_refract_ref = isRefract(currentRef.normal, ref_dir, ref_dir2);
            float refract_force = is_refract_ref ? 1 : 0;

            const BDPTVertex& currentVertex = i == 0 ? SP : path.get(i - 1);
            
            MaterialData::Pbr mat = VERTEX_MAT(currentVertex); 
            float3 out_dir = Tracer::Sample_force(mat, currentVertex.normal, in_dir, seed, refract_force);

            bool success_hit;
            path.get(i) = Tracer::FastTrace(currentVertex, out_dir, success_hit);
            if (success_hit == false)
            {
                return false;
            }
            if (i == u - 1 && Shift::glossy(path.get(i)))
            {
                return false;
            }
            if (i != u - 1 && !Shift::glossy(path.get(i)))
            {
                return false;
            }
            if (path.get(i).type == BDPTVertex::HIT_LIGHT_SOURCE)
            {
                if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX && i == u - 1)
                {
                    int light_id = path.get(i).materialId;
                    const Light& light = Tracer::params.lights[light_id];
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, path.get(i).uv);
                    init_vertex_from_lightSample(light_sample, path.get(i));
                }
                else
                {
                    return false;
                }
            }
            else if (CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX && i == u - 1)
            {
                return false;
            }
            float3 diff = currentVertex.position - path.get(i).position;
            pdf *= Tracer::Pdf_force(mat, currentVertex.normal, in_dir, out_dir, refract_force);
            pdf *= abs(dot(out_dir, path.get(i).normal)) / dot(diff, diff);
            //pdf *= tracingPdf(currentVertex, path.get(i), in_dir, true, true);
            in_dir = -out_dir;
        }
        if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && !Tracer::visibilityTest(Tracer::params.handle, path.get(-1), CP))
        {
            return false;
        }
        return true;
    }


    #define DOT_INVALIDATE_ALTERNATE_PATH(path) (path.setSize(0))
    #define DOT_IS_ALTERNATE_PATH_INVALID(path) (path.size() == 0)
    #define DOT_INVALID_ALTERNATE_PATH_PDF 1
    #define DOT_SP_RATIO 0.0
    /**
     * This function samples an alternate path and stores it in the path container. Returns false if sampling fails.
     * CP stands for control point, SP stands for specular point, u is the step size, and WC stands for control direction.
     * seed is the random number seed.
     *
     * @param seed: unsigned integer, random number seed
     * @param path: PathContainer, path container to store the sampled alternate path, initial size = 0
     * @param CP: BDPTVertex, control point
     * @param SP: BDPTVertex, specular point
     * @param WC: float3, control direction
     * @param u: integer, step size
     * @param statistic_prd: statistic_payload, interface for using statistic information
     * @return: bool, true if the path is sampled successfully, false otherwise
     */
    RT_FUNCTION bool alternate_path_sample(unsigned& seed, PathContainer& path, const BDPTVertex& CP, const BDPTVertex& SP, float3 WC, int u, statistic_payload& statistic_prd)
    {
        path.setSize(u);
        bool SP_SAMPLE_ONLY = u != 1;
        if (!SP_SAMPLE_ONLY && RR_TEST(seed, 1 - DOT_SP_RATIO))//forward sampling
        {
            if (statistic_prd.NO_CP())//light source sampling
            { 
                Tracer::lightSample light_sample;
                light_sample(seed);
                init_vertex_from_lightSample(light_sample, path.get(0));
            }
            else
            {
                float3 out_direction;
                //Control Point is located on light source
                //light source is assume to be surface light source
                //sample by cosine
                if (CP.depth == 0)
                {
                    Onb onb(CP.normal);
                    cosine_sample_hemisphere(rnd(seed), rnd(seed), out_direction);
                    onb.inverse_transform(out_direction); 
                }
                //Control Point is a normal surface vertex
                //sample by BSDF at Control Point
                else
                {
                    MaterialData::Pbr mat = VERTEX_MAT(CP);
                    float3 out_direction = Tracer::Sample(mat, CP.normal, WC, seed);
                }

                bool success_hit;
                path.get(0) = Tracer::FastTrace(CP, out_direction, success_hit);
                if (success_hit == false || path.get(0).type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                    Shift::glossy(path.get(0)))
                {
                    DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                }
            }
        }
        //trace from Specular Point
        else
        {
            float3 out_direction = statistic_prd.getInitialDirection(SP.normal, seed);
            ////////////TBD:get the initial direction by path guiding/////////////

            for (int i = 0; i < u; i++)
            {
                const BDPTVertex& currentVertex = (i == 0 ? SP : path.get(i - 1));
                bool success_hit;
                path.get(i) = Tracer::FastTrace(currentVertex, out_direction, success_hit);
                if (success_hit == false)
                {
                    DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                }
                if (i == u - 1 && Shift::glossy(path.get(i)))
                {
                    DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                }
                if (i != u - 1 && !Shift::glossy(path.get(i)))
                {
                    DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                }
                if (path.get(i).type == BDPTVertex::HIT_LIGHT_SOURCE)
                {
                    if (statistic_prd.NO_CP() && i == u - 1)
                    {
                        int light_id = path.get(i).materialId;
                        const Light& light = Tracer::params.lights[light_id];
                        Tracer::lightSample light_sample;
                        light_sample.ReverseSample(light, path.get(i).uv);
                        init_vertex_from_lightSample(light_sample, path.get(i));
                    }
                    else
                    {
                        DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                    }
                }
                else if (statistic_prd.NO_CP() && i == u - 1)
                {
                    DOT_INVALIDATE_ALTERNATE_PATH(path); return false;
                }
                if (i != u - 1)
                {
                    MaterialData::Pbr mat = VERTEX_MAT(path.get(i));
                    out_direction = Tracer::Sample(mat, path.get(i).normal, -out_direction, seed);
                }
            }
        }

        return true;
     }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////Supporting Distribution PDF Compute below////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This function evaluates the supporting alternate path PDF for given parameters.
     * CP stands for control point, SP stands for specular point, u is the step size, and WC stands for control direction.
     *
     * @param path: PathContainer, path container of the alternate path
     * @param CP: BDPTVertex, control point
     * @param SP: BDPTVertex, specular point
     * @param WC: float3, control direction
     * @param u: integer, step size
     * @param statistic_prd: statistic_payload, interface for using statistic information
     * @return: float, supporting PDF for given parameters
     */
    RT_FUNCTION float alternate_path_pdf(PathContainer& path, const BDPTVertex& CP, const BDPTVertex& SP, float3 WC, int u, statistic_payload& statistic_prd)
    {
        if (DOT_IS_ALTERNATE_PATH_INVALID(path))return DOT_INVALID_ALTERNATE_PATH_PDF;

        bool SP_SAMPLE_ONLY = u != 1;

        float SP_SAMPLE_pdf = 1;
        SP_SAMPLE_pdf *= tracingPdf(SP, path.get(0)) * M_PI * statistic_prd.getInitialDirectionPdf(SP.normal, normalize(path.get(0).position - SP.position));
        for (int i = 1; i < u; i++)
        {
            const BDPTVertex& LL_Vertex = i == 1 ? SP : path.get(i - 2);
            SP_SAMPLE_pdf *= tracingPdf(path.get(i - 1), path.get(i), normalize(LL_Vertex.position - path.get(i - 1).position), true);
        }
        float CP_SAMPLE_pdf;
        if (!SP_SAMPLE_ONLY)
        {
            if (statistic_prd.NO_CP())//light source sampling
            {
                CP_SAMPLE_pdf = path.get(0).pdf;
            }
            else
            {
                //Control Point is located on light source
                //light source is assumed to be surface light source
                //sample by cosine
                if (CP.depth == 0)
                {
                    CP_SAMPLE_pdf = tracingPdf(CP, path.get(0));
                }
                else
                {
                    MaterialData::Pbr mat = VERTEX_MAT(CP);
                    float3 diff = path.get(0).position - CP.position;
                    float3 out_dir = normalize(diff);
                    CP_SAMPLE_pdf = tracingPdf(CP, path.get(0), WC, true);
                }
            } 
        }
        return SP_SAMPLE_ONLY ? SP_SAMPLE_pdf : SP_SAMPLE_pdf * DOT_SP_RATIO + CP_SAMPLE_pdf * (1 - DOT_SP_RATIO);
    }

    /**
     * This function evaluates the alternate path tracing PDF for given parameters.
     * CP stands for control point, SP stands for specular point, u is the step size, and WC stands for control direction.
     *
     * @param path: PathContainer, path container of the alternate path
     * @param CP: BDPTVertex, control point
     * @param SP: BDPTVertex, specular point
     * @param WC: float3, control direction
     * @param u: integer, step size
     * @param statistic_prd: statistic_payload, interface for using statistic information
     * @return: float, alternate path tracing PDF in normal sub-path tracing for given parameters
     */
    RT_FUNCTION float alternate_path_eval(PathContainer& path, const BDPTVertex& CP, const BDPTVertex& SP, float3 WC, int u, statistic_payload& statistic_prd)
    {
        if (DOT_IS_ALTERNATE_PATH_INVALID(path))return 0;

        float pdf = 1;
        if (statistic_prd.NO_CP())
        {
            pdf = path.get(-1).pdf;
        }
        else if (CP.depth == 0)
        {
            pdf = tracingPdf(CP, path.get(-1));
        }
        else
        {
            pdf = tracingPdf(CP, path.get(-1), WC, false, false);
        }
        for (int i = 1; i < u; i++)
        {
            if (statistic_prd.NO_CP() && i == 1)
                pdf *= tracingPdf(path.get(-i), path.get(-i - 1));
            else
            {
                const BDPTVertex& LL_Vertex = i == 1 ? CP : path.get(-i + 1);
                pdf *= tracingPdf(path.get(-i), path.get(-i - 1), normalize(LL_Vertex.position - path.get(-i).position), false);
            }
        }
        
        if (statistic_prd.NO_CP() && u == 1)
        {
            pdf *= tracingPdf(path.get(0), SP);
        }
        else
        {
            float3 dir = u == 1 ? normalize(CP.position - path.get(0).position) : normalize(path.get(1).position - path.get(0).position);
            pdf *= tracingPdf(path.get(0), SP, dir, false);

        }
        return pdf;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////Reciprocal Estimation Code Below/////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct splitingStack
    {
        int positive_stack;
        int negative_stack;
        RT_FUNCTION splitingStack() :positive_stack(0), negative_stack(0) {}
        RT_FUNCTION void push(int sign, int size)
        {
            if (sign == 1)
                positive_stack += size;
            else if (sign == -1)
                negative_stack += size;
            else
            {
                printf("spliting stack error: push invalid sign");
            }
        }
        RT_FUNCTION bool empty() { return positive_stack <= 0 && negative_stack <= 0; }
        RT_FUNCTION int pop()
        {
            if (positive_stack > 0)
            {
                positive_stack -= 1;
                return 1;
            }
            else if (negative_stack > 0)
            {
                negative_stack -= 1;
                return -1;
            }
            else
            {
                printf("spliting stack error: pop order for a empty stack %d %d\n", positive_stack,negative_stack);
            }
        }
    }; 
    // This function evaluates the reciprocal of the path PDF integral for given parameters.
    // CP stands for control point, SP stands for specular point, u is the step size, and WC stands for control direction.
    // seed is the random number seed.
    // 
    // @param seed: unsigned integer, random number seed
    // @param CP: BDPTVertex, control point
    // @param SP: BDPTVertex, specular point
    // @param WC: float3, control direction
    // @param u: integer, step size
    // @param statistic_prd: statistic_payload, interface for using statistic information
    // @return: float, reciprocal of the path PDF integral for given parameters
    RT_FUNCTION float reciprocal_estimation(unsigned& seed, BDPTVertex CP, BDPTVertex SP, float3 WC, int u, statistic_payload& statistic_prd)
    {
        float B = 1;
        int loop_limit = dropOut_tracing::max_loop;
        B = statistic_prd.data.bound;
        float max_B = 0;
        //above code: information setup

        float res = 1 / B;
        res = 0;
        BDPTVertex buffer[SHIFT_VALID_SIZE];
        PathContainer path(buffer, 1, 0);
        splitingStack spliting_stack;
        spliting_stack.push(1, 1); 
        while (spliting_stack.empty() == false && loop_limit > 0)
        {
            int sign = spliting_stack.pop();
            bool sample_success = alternate_path_sample(seed, path, CP, SP, WC, u, statistic_prd);
            float p = alternate_path_eval(path, CP, SP, WC, u, statistic_prd);
            float q = alternate_path_pdf(path, CP, SP, WC, u, statistic_prd);

            if (p>0 && dropOut_tracing::PG_reciprocal_estimation_enable && !statistic_prd.pg_p->trainEnd) {
                ////statistic collection
                dropOut_tracing::statistic_record dirction_record = statistic_prd.generate_record(dropOut_tracing::SlotUsage::Dirction);
                float3 dir = normalize(path.get(0).position - SP.position);
                float2 uv = dir2uv(dir);
                dirction_record.data = uv.x;
                dirction_record.data2 = uv.y;
                DOT_pushRecordToBuffer(dirction_record, statistic_prd);
                ////statistic collection end
            }
            
            float factor = 1 - p / (B * q);
    //        res += factor / B * sign;
            res += 1 / B * sign;
            //if (sample_success)printf("p %f q%f B%f u%d\n", p, q, B, path.size());
            float RRS = abs(factor);
            float rr_rate = RRS - int(RRS);
            int next_sign = factor > 0 ? sign : sign * -1;
            if (RR_TEST(seed, rr_rate))
            {
                spliting_stack.push(next_sign, int(RRS) + 1);
            }
            else
            {
                spliting_stack.push(next_sign, int(RRS));
            }

            loop_limit--;
            path.setSize(0);
            ////statistic collection
            max_B = max_B > abs(p / q) ? max_B : abs(p / q); 
        }


        ////statistic collection
        dropOut_tracing::statistic_record bound_record = statistic_prd.generate_record(dropOut_tracing::SlotUsage::Bound);
        bound_record.data = max_B;
        DOT_pushRecordToBuffer(bound_record, statistic_prd);
        ////statistic collection end


        return res;
    }

    //return u, vertex number of alternate path 
    //path: light sub-path
    //find the Specular Point(SP), Control Point(CP), Control Direction(WC) and u for the given light sub-path
    RT_FUNCTION int get_imcomplete_subpath_info(PathContainer path, BDPTVertex& SP, BDPTVertex& CP, float3& WC)
    {
        int u = 1;
        SP = path.get(0);        
        CP.type = BDPTVertex::Type::DROPOUT_NOVERTEX;
        CP.pdf = 1;
        for (int i = 1; i < path.size() - 1; i++)
        {
            if (Shift::glossy(path.get(i)))
                u += 1;
            else
            { 
                CP = path.get(i + 1);
                if (CP.depth != 0)
                {
                    WC = normalize(path.get(i + 2).position - path.get(i + 1).position);
                }
                break;
            }
        }
        return u;
    }
    /* This function concatenates paths gand h_y together.
    *@param buffer : an array of BDPTVertex objects to store the concatenated path
    * @param begin : the number of vertices in the subpath(and is therefore skipped)
    * @param u : the number of vertices skipped in h_y.In an implementation where the replacement path is the same length as the original path, u is equal to the size of g.However, in future implementations where the replacement path length may vary, u may not be equal to the size of g.
    * @param g : the PathContainer object representing the first path to be concatenated
    * @param h_y : the PathContainer object representing the second path to be concatenated
    * @return: the total number of vertices in the concatenated path
    */
    RT_FUNCTION int dropoutTracing_concatenate(BDPTVertex * buffer, int begin, int u, PathContainer g, PathContainer h_y)
    {
        int count = begin;
        buffer[count] = h_y.get(0);
        count += 1;
        for (int i = 0; i < g.size(); i++)
        {
            buffer[count] = g.get(i);
            count += 1;
        }        
        for (int i = u + 1; i < h_y.size(); i++)
        {
            buffer[count] = h_y.get(i);
            count += 1;
        }
        return count;
    }
    RT_FUNCTION bool valid_specular(BDPTVertex& CP, BDPTVertex& SP, int u, float3 WC)
    {
        if (dropOut_tracing::CP_disable && CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX)return false;
        if (dropOut_tracing::multi_bounce_disable && u != 1)return false;
        if (dropOut_tracing::CP_lightsource_only && CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && CP.depth != 0)return false;
        if (dropOut_tracing::CP_lightsource_disable && CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && CP.depth == 0)return false;
        if (dropOut_tracing::CP_require && CP.type == BDPTVertex::Type::DROPOUT_NOVERTEX) return false;
        if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX && Shift::glossy(CP))return false;
        if (u > dropOut_tracing::max_u)return false; 
        return true;
    }
    RT_FUNCTION bool pathRecord_is_causticEyesubpath(long long record, int depth)
    {
        if (depth > 32)return false;
        for (int i = 0; i < depth; i++)
        {
            if (i != depth - 1 && record % 2 != 1)return false;
            if (i == depth - 1 && record % 2 != 0)return false;
            record = record >> 1;
        }
        return true;
    }
    RT_FUNCTION bool vertex_very_close(const BDPTVertex& a, const BDPTVertex& b)
    {
        float3 diff = a.position - b.position;  
        //if (dot(diff, diff) < DOT_VERY_CLOSE_DISTANCE2 || (a.depth > 0 && Tracer::params.materials[a.materialId].roughness < 0.01)) return true;
        if (dot(diff, diff) < DOT_VERY_CLOSE_DISTANCE2 ) return true;
        return false;
    }

    RT_FUNCTION bool getCausticPathInfo(const BDPTVertex* path, int path_size, BDPTVertex& SP, BDPTVertex& CP, int& u, float3& WC)
    {
        for (int i = 1; i < path_size - 1; i++)
        {
            if (path_size - i - 1 >= SHIFT_VALID_SIZE)return false;
            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == false &&  vertex_very_close(path[i], path[i + 1]) == false)
                break;

            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == true)
            {
                SP = path[i + 1];
                CP.type = BDPTVertex::Type::DROPOUT_NOVERTEX;
                u = 1;
                for (u = 1; i + u + 1 < path_size; u++)
                {
                    if (!glossy(path[i + u + 1]))
                    {
                        if (i + u + 2 < path_size)
                        {
                            CP = path[i + u + 2];
                        }
                        if (i + u + 3 < path_size)
                        {
                            WC = normalize(path[i + u + 3].position - path[i + u + 2].position);
                        }
                        break;
                    }
                }
                return true;
            }
        }
        return false;
    }
    RT_FUNCTION bool path_alreadyCaustic(const BDPTVertex* eye_buffer, int buffer_size, PathContainer light_subpath, int light_depth)
    {
        if (light_subpath.size() >= SHIFT_VALID_SIZE)return false;
        int size = buffer_size + light_depth + 1;
        for (int i = 1; i < size - 1; i++)
        { 
            const BDPTVertex& a = i < buffer_size ? eye_buffer[i] : light_subpath.get(i - buffer_size);
            const BDPTVertex& b = i + 1 < buffer_size ? eye_buffer[i + 1] : light_subpath.get(i - buffer_size + 1);
            const BDPTVertex& c = i - 1 < buffer_size ? eye_buffer[i - 1] : light_subpath.get(i - buffer_size - 1);
            if (Shift::glossy(a) == false)
            {
                if (Shift::glossy(b) == true && size - i < SHIFT_VALID_SIZE)
                {
                    return true;
                }
                if (vertex_very_close(a, b) == false)
                    return false;
            } 
        }
        return false;
    }
}
#endif // !CUPROG_H