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

#include "whitted.h"
#include"optixPathTracer.h"
#include"BDPTVertex.h" 
#include"decisionTree/classTree_device.h"
#define SCENE_EPSILON 1e-3f


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

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
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
        return make_float3(tex2D<float4>(tex, uv.x, uv.y)); 
    }

    RT_FUNCTION float3 color(float3 dir)const
    {
        return getColor(dir);
    } 
    RT_FUNCTION float pdf(float3 dir)const
    {
        float2 uv = dir2uv(dir);
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

struct SubspaceSampler_device:public SubspaceSampler
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


    RT_FUNCTION const BDPTVertex& uniformSample(unsigned int& seed, float& sample_pmf)
    { 
        sample_pmf = 1.0 / vertex_count;
        int index = rnd(seed) * vertex_count; 

        return LVC[jump_buffer[index]];
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
    RT_FUNCTION void clear()
    {
        path.clear();
        depth = 0;
        done = false;
        throughput = make_float3(1);
        result = make_float3(0.0);
    }
};

//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
//
//------------------------------------------------------------------------------

__device__ __forceinline__ float3 schlick( const float3 spec_color, const float V_dot_H )
{
    return spec_color + ( make_float3( 1.0f ) - spec_color ) * powf( 1.0f - V_dot_H, 5.0f );
}

__device__ __forceinline__ float vis( const float N_dot_L, const float N_dot_V, const float alpha )
{
    const float alpha_sq = alpha*alpha;

    const float ggx0 = N_dot_L * sqrtf( N_dot_V*N_dot_V * ( 1.0f - alpha_sq ) + alpha_sq );
    const float ggx1 = N_dot_V * sqrtf( N_dot_L*N_dot_L * ( 1.0f - alpha_sq ) + alpha_sq );

    return 2.0f * N_dot_L * N_dot_V / (ggx0+ggx1);
}


__device__ __forceinline__ float ggxNormal( const float N_dot_H, const float alpha )
{
    const float alpha_sq   = alpha*alpha;
    const float N_dot_H_sq = N_dot_H*N_dot_H;
    const float x          = N_dot_H_sq*( alpha_sq - 1.0f ) + 1.0f;
    return alpha_sq/( M_PIf*x*x );
}

__device__ __forceinline__ float float3sum(float3 c)
{
    return c.x + c.y + c.z;
}

__device__ __forceinline__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
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
        whitted::PayloadRadiance*   payload
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
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_RADIANCE,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
     
}
static __forceinline__ __device__ void traceLightSubPath(
    OptixTraversableHandle      handle,
    float3                      ray_origin,
    float3                      ray_direction,
    float                       tmin,
    float                       tmax,
    PayloadBDPTVertex*          payload
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
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RayType::RAY_TYPE_OCCLUSION,      // SBT offset
            RayType::RAY_TYPE_COUNT,          // SBT stride
            RayType::RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
    return __uint_as_float( u0 );
}


__forceinline__ __device__ void setPayloadResult( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

__forceinline__ __device__ float getPayloadOcclusion()
{
    return __uint_as_float( optixGetPayload_0() );
}

__forceinline__ __device__ void setPayloadOcclusion( float attenuation )
{
    optixSetPayload_0( __float_as_uint( attenuation ) );
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
            position = light.quad.u * r1 + light.quad.v * r2 + light.quad.corner * r3;
            emission = light.quad.emission;
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
        else if (light.type == Light::Type::QUAD)
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
    RT_FUNCTION void traceMode(unsigned int &seed)
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
RT_FUNCTION float lerp(const float &a, const float &b, const float t)
{
    return a + t * (b - a);
}


RT_FUNCTION float3 Eval_Transmit(const MaterialData::Pbr& mat, const float3& normal, const float3& V, const float3& L)
{
    float3 N = normal;
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);

    if (NDotL == 0 || NDotV == 0) return make_float3(0);

    float3 Cdlin = make_float3(mat.base_color);
    float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.
    float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    float mateta = 1.01f;
    float eta = NDotV > 0 ? (mateta / 1.0f) : (1.0f / mateta);
    float3 wh = normalize(V + L * eta);
    if (dot(wh,N) < 0) wh = -wh;

    // Same side?
    if (dot(L, wh) * dot(V, wh) > 0) return make_float3(0);

    float sqrtDenom = dot(V, wh) + eta * dot(L, wh);
    float factor =  1 / eta;
    float mattrans = 0.999;
    float3 T = mattrans * make_float3(sqrt(mat.base_color.x), sqrt(mat.base_color.y), sqrt(mat.base_color.z));
    
    float roughg = sqr(mat.roughness * 0.5f + 0.5f);
    float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);
    float a = max(0.001f, mat.roughness);
    float Ds = GTR2(dot(wh,N), a);
    float FH = SchlickFresnel(dot(V, wh));
    float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);

    return (make_float3(1.f) - Fs) * T *
        std::abs(Ds * Gs * eta * eta *
            abs(dot(L, wh)) * abs(dot(V, wh)) * factor * factor /
            (NDotL * NDotL * sqrtDenom * sqrtDenom));

}
RT_FUNCTION float3 Eval(const MaterialData::Pbr& mat, const float3& normal, const float3& V, const float3& L)
{
#ifdef BRDF
    if (mat.brdf)
    {
        return mat.color;
    }
#endif
    float3 N = normal;
 
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    if (NDotL <= 0.0f && NDotV < 0.0f)
        return make_float3(0);

    if (NDotL <= 0.0f || NDotV <= 0.0f) return Eval_Transmit(mat,normal,V,L);

    float3 H = normalize(L + V);
    float NDotH = dot(N, H);
    float LDotH = dot(L, H);

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
    trans = 0.999f;

    float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
        * (1.0f - mat.metallic)*(1-trans)
        + Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;

    return out;
}


RT_FUNCTION float3 Sample_shift_metallic(const MaterialData::Pbr& mat, const float3& N, const float3& V, float r1, float r2)
{
  
    float3 dir; 
    Onb onb(N); // basis
     
    {
        float a = max(0.001f, mat.roughness);

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
RT_FUNCTION float3 Sample(const MaterialData::Pbr& mat, const float3& N, const float3& V, unsigned int &seed)
{
 
    //float3 N = normal;
    //float3 V = in_dir;
    //prd.origin = state.fhp;
    float transRatio = mat.trans;
    transRatio = 0.999f;
    float transprob = rnd(seed);
    if (transprob < transRatio) // sample transmit
    {
        if (dot(V,N) == 0) return -V;
        float mateta = 1.01f;
        float eta = dot(V, N) > 0 ? (1 / mateta) : (mateta);

        float cosThetaI = dot(N, V);
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = eta * eta * sin2ThetaI;
        if (sin2ThetaT >= 1)
        {
            return N * 2 * dot(N, V) - V;
        }
        float cosThetaT = sqrt(1 - sin2ThetaT);
        return  eta * -V + (eta * cosThetaI - cosThetaT) * N;
    }
    float3 dir;

    float probability = rnd(seed);
    float diffuseRatio = 0.5f * (1.0f - mat.metallic);

    float r1 = rnd(seed);
    float r2 = rnd(seed);

    Onb onb(N); // basis

    if (probability < diffuseRatio) // sample diffuse
    {
        cosine_sample_hemisphere(r1, r2, dir);
        onb.inverse_transform(dir);
    }
    else
    {
        float a = max(0.001f, mat.roughness);

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

RT_FUNCTION float Pdf(MaterialData::Pbr& mat, float3 normal, float3 V, float3 L, float3 position = make_float3(0.0), bool eye_side = false)
{
#ifdef BRDF
    if (mat.brdf)
        return 1.0f;// return abs(dot(L, normal));
#endif

    float mattrans = 0.99f;
    float transRatio = mattrans;
    float3 n = normal;
    float pdf;
    if (dot(n, V) * dot(n, L) > 0)
    {
        float specularAlpha = max(0.001f, mat.roughness);
        float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

        float diffuseRatio = 0.5f * (1.f - mat.metallic);
        float specularRatio = 1.f - diffuseRatio;

        float3 half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        float ratio = 1.0f / (1.0f + mat.clearcoat);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PIf);

        pdf = (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * (1 - transRatio);
    }
    else
    {
        float mateta = 1.01f;
        float eta = dot(V, n) > 0 ? (1 / mateta) : (mateta);
        float3 wh = normalize(V + L * eta);
        if (dot(V, wh) * dot(L, wh) > 0) return 0;

        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        float sqrtDenom = dot(V, wh) + eta * dot(L, wh);
        float dwh_dwi =
            std::abs((eta * eta * dot(L, wh)) / (sqrtDenom * sqrtDenom));
        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(dot(wh, n), a);
        float pdfTrans = Ds * abs(dot(n,wh)) * dwh_dwi;

        pdf = transRatio * pdfTrans;
    }
    
    return pdf;
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

        MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
        mat.base_color = make_float4(midPoint.color, 1.0);
        throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
            * Eval(mat, midPoint.normal, lastDirection, nextDirection);
    }
    return throughput;
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

            MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
            mat.base_color = make_float4(midPoint.color, 1.0);
            float rr_rate = fmaxf(midPoint.color);
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

        MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
        mat.base_color = make_float4(midPoint.color, 1.0);
        float rr_rate = fmaxf(midPoint.color);
        pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
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

            MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
            mat.base_color = make_float4(midPoint.color, 1.0);
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

                MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
                mat.base_color = make_float4(midPoint.color, 1.0);
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
            
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color,1.0);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, false);
            float RR_rate = max(fmaxf(color), MIN_RR_RATE);
            return pdf * d_pdf * RR_rate * g;
        }

        RT_FUNCTION float3 forward_eye(const nVertex& b)const
        {
            if (b.isDirLight())
            {
                float3 c_dir = -b.normal;

                MaterialData::Pbr mat = Tracer::params.materials[materialId];
                mat.base_color = make_float4(color, 1.0);
                float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
                float RR_rate = max(fmaxf(color), MIN_RR_RATE);
                return weight * d_pdf * RR_rate;
            }

            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);
             
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
            float RR_rate = max(fmaxf(color), MIN_RR_RATE);
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


            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
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
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
            return Tracer::Eval(mat, normal, dir, c_dir);
        }

    };

}

#endif // !CUPROG_H