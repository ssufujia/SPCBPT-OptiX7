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


#define ISINVALIDVALUE(ans) (ans.x>100000.0f|| isnan(ans.x)||ans.y>100000.0f|| isnan(ans.y)||ans.z>100000.0f|| isnan(ans.z))
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


    RT_FUNCTION const BDPTVertex& uniformSampleGlossy(unsigned int& seed, float& sample_pmf)
    {
        sample_pmf = 1.0 / glossy_count;
        int index = rnd(seed) * glossy_count;

        return LVC[glossy_index[index]];
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
{//�ǵó�ʼ��
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
RT_FUNCTION float lerp(const float &a, const float &b, const float t)
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
RT_FUNCTION float3 Eval_Transmit(const MaterialData::Pbr& mat, const float3& normal, const float3& V_vec, const float3& L_vec)
{
    float3 N = normal;
    float3 V = V_vec;
    float3 L = L_vec;
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);

#define ETA_DEFAULT 1.5

    float mateta = mat.eta;
    float eta = 1 / mateta;
    if (NDotL > 0)
    {
        eta = 1 / eta;
        N = -N;
    }

    if (NDotL == 0 || NDotV == 0) return make_float3(0);

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
    //printf("%f\n", mat.trans);
    float roughg = sqr(mat.roughness * 0.5f + 0.5f);
    float Gs = 1 / (1 + Lambda(V, N) + Lambda(L, N));
    float a = max(0.001f, mat.roughness);
    float Ds = GTR2(dot(N, wh), a);//D(wh, N);
        //GTR2(dot(wh,N), a);
    float FH = SchlickFresnel(dot(V, wh));
    float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
    float F = fresnel(dot(L, wh), dot(V, wh), eta);
    //printf("Fresnel: %f\n", F);
    float3 out = (make_float3(1.f) - Fs) * T *
        std::abs(Ds * Gs * eta * eta *
            abs(dot(L, wh)) * abs(dot(V, wh)) / // * factor * factor /
            (NDotL * NDotL * sqrtDenom * sqrtDenom));
    //if(out.x!=0)
    //printf("trans: %f,%f,%f\n", out.x, out.y, out.z);


    return out;

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

    if (NDotL * NDotV <= 0.0f)
        return Eval_Transmit(mat, normal, V, L);
    //return make_float3(0);

    if (NDotL < 0.0f && NDotV < 0.0f)
    {
        N = -normal;
        NDotL *= -1;
        NDotV *= -1;
    }
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

    float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
        * (1.0f - mat.metallic)
        + Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;
    //printf("eval: %f,%f,%f\n", out.x, out.y, out.z);
    return out * (1 - trans);
}



RT_FUNCTION float3 Sample_shift_refract(const MaterialData::Pbr& mat, const float3& N, const float3& V, float r1, float r2, bool &refract_good)
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


        refract_good = refract(dir, half, V, dot(N, V) > 0 ? mat.eta : 1 / mat.eta); //reflection vector 

    }
    return dir;
}

RT_FUNCTION float2 sample_reverse_refract(const MaterialData::Pbr& mat, const float3& N, const float3& V, float3 dir)
{
    Onb onb(N); // basis

    float a = max(0.001f, mat.roughness);
    float eta = dot(N, V) > 0 ? mat.eta : 1.0 / mat.eta;
    float3 half = normalize(V + eta * dir);
    if (dot(half, N) < 0) half = -half;
    onb.transform(half);

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
    //dir = 2.0f * dot(V, half) * half - V; //reflection vector

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
RT_FUNCTION float2 sample_reverse_metallic(const MaterialData::Pbr& mat, const float3& N, const float3& V, float3 dir)
{
    Onb onb(N); // basis

    float a = max(0.001f, mat.roughness);
    float3 half;
     
    half = normalize(V + dir);

    onb.transform(half);
    float cosTheta = half.z;
    float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
    float sinPhi = half.y / sinTheta;
    float cosPhi = half.x / sinTheta;
    float phiA = acos(cosPhi);
    float phiB = asin(sinPhi);
    float phi = phiB > 0 ? phiA : 2 * M_PI - phiA;

    float r1 = phi / 2.0f / M_PIf;
    float A = cosTheta * cosTheta;

    float r2 = (1- A) / (A * a * a - A + 1);
    return make_float2(r1, r2);
    //dir = 2.0f * dot(V, half) * half - V; //reflection vector

}
RT_FUNCTION float3 Sample(const MaterialData::Pbr& mat, const float3& N, const float3& V, unsigned int& seed)
{

    //float3 N = normal;
    //float3 V = in_dir;
    //prd.origin = state.fhp;
    float transRatio = mat.trans;
    float transprob = rnd(seed);
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float3 dir;
    float3 normal = N;
    if (transprob < transRatio) // sample transmit
    {
        float mateta = mat.eta;
        float eta = 1 / mateta;
        if (dot(N, V) < 0)
        {
            eta = 1 / eta;
            normal = -N;
        }
        Onb onb(normal); // basis
        float a = mat.roughness;

        float phi = r1 * 2.0f * M_PIf;

        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);
        float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        onb.inverse_transform(half);

        if (dot(V, normal) == 0) return -V;

        float cosThetaI = dot(half, V);
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = eta * eta * sin2ThetaI;
        if (sin2ThetaT >= 1)
        {
            return half * 2 * dot(half, V) - V;
        }
        float cosThetaT = sqrt(1 - sin2ThetaT);
        return  eta * -V + (eta * cosThetaI - cosThetaT) * half;
    }
    if (dot(normal, V) < 0)
    {
        normal = -N;
    }
    Onb onb(normal); // basis
    float probability = rnd(seed);
    float diffuseRatio = 0.5f * (1.0f - mat.metallic);

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
    float transRatio = mat.trans;
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
    else if (dot(n, V) * dot(n, L) < 0)
    {

        float mateta = mat.eta;
        //        float eta = dot(L, n) > 0 ? (mateta) : (1/mateta);     
        float eta = 1 / mateta;
        if (dot(n, V) < 0)
        {
            eta = 1 / eta;
            n = -normal;
        }
        float3 wh = normalize(L + V * eta);
        if (dot(V, wh) * dot(L, wh) > 0) return 0;

        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        float sqrtDenom = dot(L, wh) + eta * dot(V, wh);
        float dwh_dwi =
            std::abs((eta * eta * dot(V, wh)) / (sqrtDenom * sqrtDenom));
        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(dot(wh, n), a);
        float pdfTrans = Ds * abs(dot(n, wh)) * dwh_dwi;

        pdf = transRatio * pdfTrans;
    }

    return pdf;
}


RT_FUNCTION float3 contriCompute(const BDPTVertex* path, int path_size)
{
    //Ҫ�󣺵�0������Ϊeye����size-1������Ϊlight
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
    /*��ԴĬ��Ϊ���Դ��һ�㣬��˿�����cos������ģ�������Ч��������ǵ��Դ��Ҫ�޸����´���*/

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

        /*��������ǵ��µ�pdf*/
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
    /*����ͶӰ�ǵ��µ�pdf�仯*/
    for (int i = 1; i < eyePathLength; i++)
    {
        const BDPTVertex& midPoint = path[i];
        const BDPTVertex& lastPoint = path[i - 1];
        float3 line = midPoint.position - lastPoint.position;
        float3 lineDirection = normalize(line);
        pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
    }
    /*��������ĸ���*/
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
        /*��������ĸ���*/
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

            /*��������ǵ��µ�pdf*/
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
            float RR_rate = Tracer::rrRate(mat);
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
                float RR_rate = Tracer::rrRate(color);
                return weight * d_pdf * RR_rate;
            }

            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);
             
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
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
    //������Դ��״��������
}
namespace Shift
{

    RT_FUNCTION float2 sample_reverse_half(float a, const float3& N, float3 half)
    {
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
    RT_FUNCTION float3 sample_half(float a, const float3& N, float2 uv)
    {
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
    RT_FUNCTION float dh_dwi_estimate(float a, float3 normal, float3 half, float3 v)
    {
        float3 wo = 2.0f * dot(v, half) * half - v;
        float2 origin_uv = sample_reverse_half(a, normal, half);
        float scale = .0001;
        float2 uv1 = origin_uv + make_float2(scale, 0);
        float2 uv2 = origin_uv + make_float2(0, scale);
        float3 half2 = sample_half(a, normal, uv2);
        float3 half1 = sample_half(a, normal, uv1);
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
        float2 uv1 = sample_reverse_half(a, normal, halfa);
        float2 uv2 = sample_reverse_half(b, normal, halfb);
        if(uv2.y<.7 &&uv1.y<.7)
            printf("pdf compare %f %f %f %f %f %f uv info %f %f %f %f\n", o1, o2, t1, t2, o1 / o2, t1 / t2, uv1.x, uv2.y, uv2.x, uv2.y );

    }

    RT_FUNCTION bool back_trace_robeScale(const BDPTVertex& midVertex, const BDPTVertex& originLast, 
        float3 anchor, float robe_scale, BDPTVertex& new_vertex, float& pdf, bool reverse = false)
    {
        float3 in_dir = normalize(anchor - midVertex.position);
        float3 normal = midVertex.normal;

        //float3 reflect = 2.0f * dot(in_dir, normal) * normal - in_dir;

        Onb onb(normal);

        float3 origin_out = normalize(originLast.position - midVertex.position);
        float3 half_global_origin = normalize(origin_out + in_dir);
        if (dot(half_global_origin, normal) < 0) normal = -normal;
        float3 half = half_global_origin;
        onb.transform(half);

        float a_new = max(0.001f, Tracer::params.materials[midVertex.materialId].roughness); 
        robe_scale = .5;
        //a_new = .1;// robe_scale;
        if (reverse)
        {
            float t = a_new;
            a_new = robe_scale;
            robe_scale = t;
            
        }
        float2 uv_sample = sample_reverse_half(robe_scale, normal, half);
        //if (uv_sample.y > 0.8)return false;
        float3 temp_half = sample_half(robe_scale, normal, uv_sample);

        float3 half_new = sample_half(a_new, normal, uv_sample);        
         
        if(false)
        {
            float2 uv2 = sample_reverse_half(robe_scale, normal, half);
            float3 half_2 = sample_half(robe_scale, normal, uv2);
            float2 uv3 = sample_reverse_half(robe_scale, normal, half_2);
            float3 half_3 = sample_half(robe_scale, normal, uv2);
            onb.inverse_transform(half_2);
            onb.inverse_transform(half_3); 
            printf("%f %f %f %f uv check \n", float3weight(half_2), float3weight(half_3),float3weight(half_global_origin), dot(half_global_origin,normal));
            //printf("out_direction check %f %f %f-%f\n", float3weight(half_origin_), float3weight(half_global_origin), uv_sample.x, uv_sample.y);
        }
        float3 half_global = half_new;//
        onb.inverse_transform(half_global);

        //float3 new_out = cosProject + tan_scale * biasVector;
        float3 new_dir = 2.0f * dot(in_dir, half_global) * half_global - in_dir;
        //printf("half compare %f %f\n", float3weight(new_dir), float3weight(origin_out));
        if (dot(new_dir, normal) < 0) { 
            pdf = dot(origin_out,normal)<0? 2: .5;
            return false;
        }

        //new_dir = origin_out;
        //printf("direction compare %f %f %f-%f %f %f\n", origin_out.x, origin_out.y, origin_out.z, new_dir.x, new_dir.y, new_dir.z);
        //MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        //mat.base_color = make_float4(midVertex.color, 1.0);
        //float3 new_dir; //Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, uv.x, uv.y);
        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            return false;
        }
        new_vertex = payload.path.currentVertex();
        float3 dirVec = new_vertex.position - midVertex.position;
        float3 originVec = originLast.position - midVertex.position;

        pdf = 1.0;
        //return true;
        //pdf /= robePdf(robe_scale, normal, half_global_origin, origin_out);
        //pdf *= robePdf(a_new, normal, half_global, new_dir);
        MaterialData::Pbr mat = midVertex.getMat(Tracer::params.materials);
        mat.roughness = robe_scale;
        pdf /= Tracer::Pdf(mat, normal, in_dir, origin_out);
        mat.roughness = a_new;
        pdf *= Tracer::Pdf(mat, normal, in_dir, new_dir);
        
        pdf /= 1.0 / dot(originVec, originVec) * abs(dot(normalize(originVec), originLast.normal));
        pdf *= 1.0 / dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));
        //pdf_valid_compare(robe_scale, a_new, normal, half_global_origin, half_global, origin_out, new_dir);
        return true;
    }

    RT_FUNCTION bool glossy(const MaterialData::Pbr& mat)
    {
        return mat.roughness < 0.1 && max(mat.metallic, mat.trans) >= 0.99;
    }
    RT_FUNCTION bool glossy(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && glossy(Tracer::params.materials[v.materialId]);
    }
    RT_FUNCTION bool RefractionCase(const MaterialData::Pbr& mat)
    {
        return mat.trans > 0.9;
    }
    RT_FUNCTION bool RefractionCase(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && RefractionCase(Tracer::params.materials[v.materialId]);
    }
    RT_FUNCTION float dwh_dwi_refract(float3 wh, float3 V, float3 L, float eta)
    {
        float sqrtDenom = dot(L, wh) + eta * dot(V, wh);
        float dwh_dwi =
            std::abs((eta * eta * dot(V, wh)) / (sqrtDenom * sqrtDenom));
        return dwh_dwi;
    }

    //RT_FUNCTION bool back_trace_tanScale(const BDPTVertex& midVertex, const BDPTVertex& originLast, float3 anchor, float tan_scale, BDPTVertex& new_vertex, float& pdf)
    //{
    //    float3 in_dir = normalize(anchor - midVertex.position);
    //    float3 normal = midVertex.normal;
    //    float3 reflect = 2.0f * dot(in_dir, normal) * normal - in_dir;
    //    
    //    float3 origin_out = normalize(originLast.position - midVertex.position); 
    //    float cosTheta = dot(origin_out, reflect);
    //    if (cosTheta < 0.0)return false;
    //    float3 cosProject = cosTheta * reflect;
    //    float3 biasVector = origin_out - cosProject; 
    //    float3 new_out = cosProject + tan_scale * biasVector;
    //    float3 new_dir = normalize(new_out); 
    //    //MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
    //    //mat.base_color = make_float4(midVertex.color, 1.0);
    //    //float3 new_dir; //Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, uv.x, uv.y);
    //    Tracer::PayloadBDPTVertex payload;
    //    payload.clear();
    //    payload.seed = 0;
    //    payload.ray_direction = new_dir;
    //    payload.origin = midVertex.position;
    //    init_EyeSubpath(payload.path, payload.origin, payload.ray_direction); 
    //    float3 ray_direction = payload.ray_direction;
    //    float3 ray_origin = payload.origin;
    //    int begin_depth = payload.path.size;
    //    Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
    //        SCENE_EPSILON,  // tmin
    //        1e16f,  // tmax
    //        &payload);
    //    if (payload.path.size == begin_depth)
    //    {
    //        return false;
    //    }
    //    new_vertex = payload.path.currentVertex();
    //    float3 dirVec = new_vertex.position - midVertex.position;
    //    float3 originVec = originLast.position - midVertex.position; 
    //    pdf = 1.0 / tan_scale / tan_scale * dot(new_out, new_out) / dot(origin_out, origin_out) * abs(dot(normal, origin_out)) / abs(dot(normal, new_dir));
    //    pdf /= 1.0 / dot(originVec, originVec) * abs(dot(origin_out, originLast.normal));
    //    pdf *= 1.0 / dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));
    //    return true;
    //}
    RT_FUNCTION bool back_trace(const BDPTVertex& midVertex, float2 uv, float3 anchor, BDPTVertex& new_vertex, float& pdf, bool is_refract = false)
    {
        if (is_refract == true && RefractionCase(midVertex))
        {
         //   printf("I haven't implemented the refraction case of the uv remapping. So you call the uv remapping in undesigned cases\n");
        }
        
        float3 in_dir = normalize(anchor - midVertex.position);
        MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        mat.base_color = make_float4(midVertex.color, 1.0);
        bool refract_good = true;
        float3 new_dir = is_refract ? Tracer::Sample_shift_refract(mat, midVertex.normal, in_dir, uv.x, uv.y,refract_good) :
            Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, uv.x, uv.y);
        if (refract_good == false)return false;
 
        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            return false;
        }
        new_vertex = payload.path.currentVertex();
        float3 dirVec = new_vertex.position - midVertex.position;
        pdf = Tracer::Pdf(mat, midVertex.normal, in_dir, new_dir) / dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));
        return true;
    }

    RT_FUNCTION bool shift_type_compare(const BDPTVertex& a, const BDPTVertex& b)
    {
        if(a.type == BDPTVertex::Type::NORMALHIT && b.type == BDPTVertex::Type::NORMALHIT)
            return glossy(a) == glossy(b);
        else if (a.type != BDPTVertex::Type::NORMALHIT && b.type != BDPTVertex::Type::NORMALHIT)
        {
            return a.materialId == b.materialId;
        }
        return false;
        if (a.depth == 0)
        {
            return b.hit_lightSource() && a.materialId == b.materialId;
            //            if(a.type == BDPTVertex::Type::ENV && b.type == BDPTVertex::Type::ENV_MISS)return true;
            //            if(a.type == BDPTVertex::Type::QUAD && b.type == BDPTVertex::Type::HIT_LIGHTSOURCE) return true;
        }
        else if (b.hit_lightSource())
        {
            return false;
        }

        return glossy(a) == glossy(b);
    }

    //�򵥵�·��ƫ��ʵ��
    //����ͳ����Ҫƫ�ƵĶ�����Ŀ�������Ŀ��ͬ�ڴ�ĩ����ǰ���Ĺ⻬������Ŀ
    //����ԭ�ȵ�·��pdf
    //����ƫ�ƶ�����Ŀ��ĩ�˿�ʼ����׷��
    //��ӳ��ʱҪ�����ע��ɼ���
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
        RT_FUNCTION int size() { return m_size; }
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
                printf("wrong Path Container index\n");
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

    RT_FUNCTION bool back_trace_half_vector(const BDPTVertex& midVertex, const BDPTVertex& originLast, const BDPTVertex& originMid,
        const BDPTVertex& originNext, float3 anchor, BDPTVertex& new_vertex, float& pdf)
    {
        float3 origin_half;
        float3 in_dir = normalize(anchor - midVertex.position);
        float3 normal = originMid.normal;

        float3 origin_in = normalize(originNext.position - originMid.position);
        float3 origin_out = normalize(originLast.position - originMid.position);
        bool is_refract = isRefract(normal, origin_in, origin_out);

        float3 half;
        float origin_eta;
        if (is_refract == false)
        {
            half = normalize(origin_in + origin_out);
        }
        else
        {
            origin_eta = originMid.getMat(Tracer::params.materials).eta;
            if (dot(normal, origin_in) < 0) origin_eta = 1 / origin_eta;
            half = normalize(origin_in + origin_eta * origin_out);
        }
        if (dot(half, normal) < 0) half = -half;

        origin_half = half;
        Onb onb_origin(originMid.normal), onb_new(midVertex.normal);
        onb_origin.transform(half);
        onb_new.inverse_transform(half);

        float3 new_dir;
        float dh_dwi_ratio;
        if (is_refract)
        {
            float new_eta = midVertex.getMat(Tracer::params.materials).eta;
            if (dot(midVertex.normal, in_dir) < 0) new_eta = 1 / new_eta;
            //bool refract_good = refract(new_dir, half, in_dir, new_eta);
            //if (refract_good == false) return false;
                        //refraction replace

            float eta = 1 / new_eta;
            float3 n2 = half;
            if (dot(half, in_dir) > 0) 
            { }
            else { n2 = -n2; 
            }
            float costhetai = dot(n2, in_dir);
            float sin2thetai = 1 - costhetai * costhetai;
            float sin2thetat = eta * eta * sin2thetai;
            if (sin2thetat >= 1)
            {
                return false;
            }

            float costhetat = sqrt(1 - sin2thetat);

            new_dir = eta * -in_dir + (eta * costhetai - costhetat) * n2;

            float dh_dwi_old = dwh_dwi_refract(origin_half, origin_in, origin_out, origin_eta);
            float dh_dwi_new = dwh_dwi_refract(half, in_dir, new_dir, new_eta);
            dh_dwi_ratio = dh_dwi_new / dh_dwi_old;
            
            if (false&&dh_dwi_ratio + 1 / dh_dwi_ratio > 2.01)
            {
                printf("info compare %f-%f %f-%f %f-%f %f-%f %f-%f\n",
                    dh_dwi_old, dh_dwi_new,
                    float3weight(origin_half), float3weight(half),
                    float3weight(origin_in), float3weight(in_dir),
                    float3weight(origin_out), float3weight(new_dir),
                    origin_eta, new_eta
                );
            }
            //printf("dir compare %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n %f %f %f\n",
            //    etaCheck(origin_in, origin_out, origin_half).z, etaCheck(in_dir, new_dir, half).z,
            //    origin_out.x, origin_out.y, origin_out.z,
            //    new_dir.x, new_dir.y, new_dir.z,
            //    normal.x, normal.y, normal.z,
            //    in_dir.x, in_dir.y, in_dir.z
            //);
        }
        else
        {
            float3 reflect = 2.0f * dot(in_dir, half) * half - in_dir;
            new_dir = reflect;
            dh_dwi_ratio = 1;
            dh_dwi_ratio /= 1.0 / (4.0 * abs(dot(origin_out, origin_half)));
            dh_dwi_ratio *= 1.0 / (4.0 * abs(dot(new_dir, half)));
        }
        if (isRefract(midVertex.normal, in_dir, new_dir) != is_refract) return false;

        //new_dir = origin_out;

        //printf("input check %f %f \n", float3weight(new_dir), float3weight(origin_out));


        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            return false;
        } 
        new_vertex = payload.path.currentVertex();

        float3 dirVec = new_vertex.position - midVertex.position;
        float3 originVec = originLast.position - originMid.position;
        pdf = dh_dwi_ratio;
        pdf /= 1.0 / dot(originVec, originVec) * abs(dot(origin_out, originLast.normal));
        pdf *= 1.0 / dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));

//        pdf = 1;
        return true;
    }
    RT_FUNCTION float map_function(float x, float ratio, float& pdf)
    {
        pdf = 1 / ratio / ratio;
        return x * ratio;
    }
    RT_FUNCTION bool back_trace_tanScale(const BDPTVertex& midVertex, const BDPTVertex& originLast, 
        float3 anchor, float tan_scale, BDPTVertex& new_vertex, float& pdf)
    {
        float3 in_dir = normalize(anchor - midVertex.position);
        float3 normal = midVertex.normal;
        float3 origin_out = normalize(originLast.position - midVertex.position);
        //first, check the original bounce type
        //is it a reflection or refraction?
        bool is_refract = isRefract(normal, in_dir, origin_out);
        float3 target_direction;
        if (is_refract == false)
        {//reflection 
            float3 reflect = 2.0f * dot(in_dir, normal) * normal - in_dir;
            target_direction = reflect;
        }
        else
        {
            MaterialData::Pbr mat = midVertex.getMat(Tracer::params.materials);
            float eta = mat.eta;
            //            float3 refraction;
            //            bool refract_good;
            //            refraction = refract(normal, in_dir, eta, refract_good);
            //            if (refract_good == false) return false;
            //            target_direction = refraction;
                        //refraction replace

            float3 n2 = normal;
            if (dot(normal, in_dir) > 0) eta = 1 / eta;
            else { n2 = -n2; }
            float cosThetaI = dot(n2, in_dir);
            float sin2ThetaI = 1 - cosThetaI * cosThetaI;
            float sin2ThetaT = eta * eta * sin2ThetaI;
            if (sin2ThetaT >= 1)
            { 
                return false;
            }

            float cosThetaT = sqrt(1 - sin2ThetaT);

            target_direction = eta * -in_dir + (eta * cosThetaI - cosThetaT) * n2;

        }


        float cosTheta = dot(origin_out, target_direction);
        if (cosTheta < 0.0) {
            return false;
        }
        float3 cosProject = cosTheta * target_direction;
        float3 biasVector = origin_out - cosProject;

        float mapping_pdf;
        float map_distance = map_function(length(biasVector), tan_scale, mapping_pdf);

        float3 new_out = cosProject + map_distance * normalize(biasVector);
        float3 new_dir = normalize(new_out);
        pdf = mapping_pdf * dot(new_out, new_out) / dot(origin_out, origin_out) * abs(dot(normal, origin_out)) / abs(dot(normal, new_dir));
         
        //printf("dir %f %f %f - %f %f %f\n", new_dir.x, new_dir.y, new_dir.z, origin_out.x, origin_out.y, origin_out.z);
        //MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        //mat.base_color = make_float4(midVertex.color, 1.0);
        //float3 new_dir; //Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, uv.x, uv.y);
        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            return false;
        }
        new_vertex = payload.path.currentVertex();
        float3 dirVec = new_vertex.position - midVertex.position;
        float3 originVec = originLast.position - midVertex.position;

        pdf /= 1.0 / dot(originVec, originVec) * abs(dot(normalize(originVec), originLast.normal));
        pdf *= 1.0 / dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));
        return true;
    }
    RT_FUNCTION float2 get_origin_uv(PathContainer& p, int index)// ��index+1�������������ɵ�index�������uv //����Դ���裿    
    {
        //���������
        //��ѯ���ڹ�Դ�ϣ�ֱ�ӷ���uv�����ڵ���Դ���ǳ����ģ����Դ�������Ҫ�����ӳ�䣬��Ҫע��
        //Դ���ڻ������ϣ����ʱ��uvȡ���ڲ�ѯ���λ�ú�Դ������Ļ����ⷽ��
        //Դ���ڱ�����ϣ�������diffuse�����ϣ����ʱ��ֱ�Ӹ��ݷ��ߺͽǶȴ���
        //Դ���ڽ��������ϣ����ʱ����Ҫ����׷��
        //�򵽹�Դ������
        if (index + 1 == p.size()) return p.get(index).uv;
        if (index + 2 == p.size() && p.get(index + 1).type == BDPTVertex::Type::ENV)
        {
            return SKY.trace_reverse_uv(p.get(index).position, -p.get(index + 1).normal);
        }
        if ((index + 2 == p.size() && p.get(index + 1).type == BDPTVertex::Type::QUAD) || (!(glossy(p.get(index + 1)))))
        {
            float3 normal = p.get(index + 1).normal;
            float3 dir = normalize(p.get(index).position - p.get(index + 1).position);
            return sample_reverse_cosine(normal, dir);
        }
        else
        {
            BDPTVertex& lastVertex = p.get(index + 1);
            BDPTVertex& midVertex = p.get(index);
            MaterialData::Pbr mat = Tracer::params.materials[lastVertex.materialId];
            mat.base_color = make_float4(lastVertex.color, 1.0);
            float3 last_incident;
            if (index + 3 == p.size() && p.get(index + 2).type == BDPTVertex::Type::ENV)
            {
                last_incident = p.get(index + 2).normal;
            }
            else
            {
                last_incident = normalize(p.get(index + 2).position - p.get(index + 1).position);
            }

            return Tracer::sample_reverse_metallic(mat, lastVertex.normal, last_incident, normalize(midVertex.position - lastVertex.position));
        }
    }
    RT_FUNCTION bool path_shift_classical(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian)
    {
        int glossy_count = 0;
        Jacobian = 1;
        //get glossy count
        while (glossy(originPath.get(glossy_count)))
        {
            glossy_count++;
        }
        //copy the vertex that is no need for shifting
        newPath.setSize(originPath.size());
        newPath.get(0) = originPath.get(0);
        for (int i = glossy_count + 1; i < originPath.size(); i++)
        {
            newPath.get(i) = originPath.get(i);
        }

        //recompute jacobian part one in the dominator
        for (int i = 0; i < glossy_count; i++)
        {
            //����������з��յģ���Ϊ�ڷ�RMIS�İ汾��ʵ��������singlePdf���Բ���ǿ��Ҫ�������������·��׷�ٵ�һ�£�
            //��ʵ������������ص�·���������ɶ����singlePdf�Ͳ�������Ķ��Ƿ����
            Jacobian /= originPath.get(i + 1).singlePdf;

        }

        //get uv for regeneration
        //retrace from anchor, update the jacobian value
        float3 local_anchor = anchor;
        for (int i = 0; i < glossy_count; i++)
        {
            float2 back_trace_uv = get_origin_uv(originPath, i + 1);//��ȡ�ӵ�i + 2 ��������������ԭ��·���ĵ�����i + 1 �������uv��
            float local_jacobian;
            bool trace_hit = back_trace(originPath.get(i), back_trace_uv, local_anchor, newPath.get(i + 1), local_jacobian);

            //���׷��ʧ��
            if (trace_hit == false) return false;

            //���ǰ�����Ͳ�ͬ��Ҳ��ӳ��ʧ��
            if (shift_type_compare(originPath.get(i + 1), newPath.get(i + 1)) == false) return false;
            local_anchor = newPath.get(i).position;
            Jacobian *= local_jacobian;
        }
        //�ɼ��Բ���

        if (originPath.size() == glossy_count + 1)
        {
            Tracer::lightSample light_sample;
            light_sample.ReverseSample(Tracer::params.lights[newPath.get(glossy_count).materialId], newPath.get(glossy_count).uv);
            newPath.get(glossy_count).flux = light_sample.emission;
            return true;
        }//�������·�������������ɵ��ǾͲ���Ҫ�ɼ��Բ��� 
        return Tracer::visibilityTest(Tracer::params.handle, newPath.get(glossy_count + 1), newPath.get(glossy_count));


        return true;
    }



    RT_FUNCTION bool path_shift_tanScale(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian, bool reverse = false)
    { 
        float scale_rate = .02;
        scale_rate = reverse ? 1.0 / scale_rate : scale_rate;

        int glossy_count = 0;
        Jacobian = 1;
        //get glossy count
        while (glossy(originPath.get(glossy_count)))
        {
            glossy_count++;
        }
        //copy the vertex that is no need for shifting
        newPath.setSize(originPath.size());
        newPath.get(0) = originPath.get(0);
        for (int i = glossy_count + 1; i < originPath.size(); i++)
        {
            newPath.get(i) = originPath.get(i);
        }
           
        //retrace from anchor, update the jacobian value
        float3 local_anchor = anchor;
        for (int i = 0; i < glossy_count; i++)
        {
            float local_jacobian = 1;
            bool trace_hit = true;

            if (i == 0)
            {
                //trace_hit = back_trace_tanScale(newPath.get(i), originPath.get(i + 1), local_anchor, scale_rate, newPath.get(i + 1), local_jacobian);
                trace_hit = back_trace_robeScale(newPath.get(i), originPath.get(i + 1),  local_anchor, .2, newPath.get(i + 1), local_jacobian, reverse);
                //newPath.get(i + 1) = originPath.get(i + 1);
            }
            else
            {
                trace_hit = back_trace_half_vector(newPath.get(i), originPath.get(i + 1), originPath.get(i), originPath.get(i - 1), local_anchor, newPath.get(i + 1), local_jacobian); 
            }
             
            
            //���׷��ʧ��
            if (trace_hit == false) {
                //printf(" equal map fails for miss trace\n");
                Jacobian = local_jacobian;
                return false;
            }
            //���ǰ�����Ͳ�ͬ��Ҳ��ӳ��ʧ��
            if (shift_type_compare(originPath.get(i + 1), newPath.get(i + 1)) == false) {
                //printf(" equal map fails for type difference\n");
                return false; 
            }
            local_anchor = newPath.get(i).position;
            Jacobian *= local_jacobian;
        }
        if (originPath.size() == 3)
        {
           // printf("%f info compare %f %f %f %f\n",Jacobian, float3weight(originPath.get(1).position), float3weight(originPath.get(2).position), float3weight(newPath.get(1).position), float3weight(newPath.get(2).position));
        }
        //�����еĺ͹⻬�����޹ص�·���Ŀɼ��Բ���
        //�������������·�������������ɵ��ǾͲ���Ҫ�ɼ��Բ��� 
        if (originPath.size() == glossy_count + 1)
        {
            Tracer::lightSample light_sample;
            light_sample.ReverseSample(Tracer::params.lights[newPath.get(glossy_count).materialId], newPath.get(glossy_count).uv);
            newPath.get(glossy_count).flux = light_sample.emission;
              

            init_vertex_from_lightSample(light_sample, newPath.get(glossy_count));

 
            return true;
        }
        return Tracer::visibilityTest(Tracer::params.handle, newPath.get(glossy_count + 1), newPath.get(glossy_count));


        return true;
    }



    //bounce����Ϊ1���������Դ�ĳ���Ϊ2�Ĺ�·�������ñ�������uv���������ӳ�䷽��
    RT_FUNCTION bool path_shift_uvRemap_1bounce_old(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian)
    {
        if (originPath.size() != 2 || glossy(originPath.get(0)) == false || originPath.get(1).type != BDPTVertex::Type::QUAD)
        {
            printf("path shift uvRemap 1bounce is called in undesigned case\n");
            return false;
        }
        int glossy_count = 1;
        Jacobian = 1;

        //copy the vertex that is no need for shifting
        newPath.setSize(originPath.size());
        newPath.get(0) = originPath.get(0);
        for (int i = glossy_count + 1; i < originPath.size(); i++)
        {
            newPath.get(i) = originPath.get(i);
        }

        //recompute jacobian part one in the dominator
        Tracer::lightSample light_sample;
        light_sample.ReverseSample(Tracer::params.lights[originPath.get(1).materialId], originPath.get(1).uv);
        Jacobian /= light_sample.pdf;

        //get uv for regeneration
        //retrace from anchor, update the jacobian value
        float3 local_anchor = anchor;
        for (int i = 0; i < glossy_count; i++)
        {
            float2 back_trace_uv = originPath.get(1).uv;
            float local_jacobian;
            bool is_refract = isRefract(originPath.get(i).normal, local_anchor - originPath.get(i).position,
                originPath.get(i + 1).position - originPath.get(i).position);
            bool trace_hit = back_trace(originPath.get(i), back_trace_uv, local_anchor, newPath.get(i + 1), local_jacobian, is_refract);

            //���׷��ʧ��
            if (trace_hit == false) return false;

            //���ǰ�����Ͳ�ͬ��Ҳ��ӳ��ʧ��
            if (shift_type_compare(originPath.get(i + 1), newPath.get(i + 1)) == false) return false;
            local_anchor = newPath.get(i).position;
            Jacobian *= local_jacobian;
        }
        //�ɼ��Բ���

        if (originPath.size() == glossy_count + 1)
        {
            Tracer::lightSample light_sample;
            light_sample.ReverseSample(Tracer::params.lights[newPath.get(glossy_count).materialId], newPath.get(glossy_count).uv);
            //newPath.get(glossy_count).flux = light_sample.emission;

            init_vertex_from_lightSample(light_sample, newPath.get(glossy_count));
            return true;
        }//�������·�������������ɵ��ǾͲ���Ҫ�ɼ��Բ��� 
        return Tracer::visibilityTest(Tracer::params.handle, newPath.get(glossy_count + 1), newPath.get(glossy_count));


        return true;
    }
    RT_FUNCTION bool path_shift_uvRemap_1bounce(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian)
    {
        if (originPath.size() != 2 || glossy(originPath.get(0)) == false || originPath.get(1).type != BDPTVertex::Type::QUAD)
        {
            printf("path shift uvRemap 1bounce is called in undesigned case\n");
            return false;
        }
        int glossy_count = 1;
        Jacobian = 1;

        //copy the vertex that is no need for shifting
        newPath.setSize(2);
        newPath.get(0) = originPath.get(0);

        //get uv for regeneration
        float2 remap_uv = originPath.get(1).uv;
        //recompute jacobian part one in the dominator
        Tracer::lightSample light_sample;
        light_sample.ReverseSample(Tracer::params.lights[originPath.get(1).materialId], remap_uv);
        Jacobian /= light_sample.pdf;

        //retrace from anchor, update the jacobian value 
        float local_jacobian;
        bool is_refract = isRefract(originPath.get(0).normal, anchor - originPath.get(0).position,
            originPath.get(1).position - originPath.get(0).position);
        bool trace_hit = back_trace(originPath.get(0), remap_uv, anchor, newPath.get(1), local_jacobian, is_refract);

        //���׷��ʧ��
        if (trace_hit == false) return false;

        //���ǰ�����Ͳ�ͬ��Ҳ��ӳ��ʧ��
        if (newPath.get(1).type != BDPTVertex::Type::HIT_LIGHT_SOURCE) return false;
        Jacobian *= local_jacobian;
        int light_id = newPath.get(1).materialId;
        float2 new_uv = newPath.get(1).uv;
        //}
        //�ɼ��Բ���

        //Tracer::lightSample light_sample;
        light_sample.ReverseSample(Tracer::params.lights[light_id], new_uv);
        //newPath.get(glossy_count).flux = light_sample.emission;

        init_vertex_from_lightSample(light_sample, newPath.get(1));
        return true;

    }

    RT_FUNCTION bool path_shift_inverse_uvRemap_1bounce(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian)
    {
        if (originPath.size() != 2 || glossy(originPath.get(0)) == false || originPath.get(1).type != BDPTVertex::Type::QUAD)
        {
            printf("path shift inverse uvRemap 1bounce is called in undesigned case\n");
            return false;
        }
        newPath.get(0) = originPath.get(0);
        MaterialData::Pbr mat = Tracer::params.materials[originPath.get(0).materialId];
        mat.base_color = make_float4(originPath.get(0).color, 1.0);
        float3 out_vec = originPath.get(1).position - originPath.get(0).position;
        float3 out_dir = normalize(out_vec);
        float3 in_dir = normalize(anchor - originPath.get(0).position);
        bool is_refract = isRefract(originPath.get(0).normal, in_dir, out_dir);
        float2 map_uv = is_refract ? Tracer::sample_reverse_refract(mat, originPath.get(0).normal, in_dir, out_dir)
            : Tracer::sample_reverse_metallic(mat, originPath.get(0).normal, in_dir, out_dir);

        Jacobian = 1;
        Tracer::lightSample light_sample;
        light_sample.ReverseSample(Tracer::params.lights[originPath.get(1).materialId], map_uv);
        init_vertex_from_lightSample(light_sample, newPath.get(1));
        Jacobian *= light_sample.pdf;

        Jacobian /= Tracer::Pdf(mat, originPath.get(0).normal, in_dir, out_dir);
        Jacobian /= 1.0 / dot(out_vec, out_vec) * abs(dot(out_dir, newPath.get(1).normal));
        bool map_suc = Tracer::visibilityTest(Tracer::params.handle, newPath.get(0), newPath.get(1));
        return map_suc;
    }

    RT_FUNCTION bool shiftPathType(PathContainer& a)
    {
        if (a.size() <= 1)return false;
        if (glossy(a.get(0)) == false) return false;
        return true;
    }
    //����false��ʾӳ�䵽��һ������ֵΪ0��·��
    //����true���ʾ����ӳ�䣬ע������ǲ�Ӧ�ñ�ӳ���·���������˸ú�����Ҳ�᷵��true��·�������ᷢ���κθı䡣
    RT_FUNCTION bool path_shift(PathContainer& originPath, PathContainer& newPath, float3 anchor, float& Jacobian, bool reverse = false)
    {
        if (shiftPathType(originPath) == false)
        {
            Jacobian = 1;
            newPath.setSize(originPath.size());
            for (int i = 0; i < originPath.size(); i++)
            {
                newPath.get(i) = originPath.get(i);
            }
            //            newPath = originPath;
            return true;
        }
        if (originPath.size() == 2 && glossy(originPath.get(0)) && originPath.get(1).type == BDPTVertex::Type::QUAD)
        {
            return reverse ? path_shift_inverse_uvRemap_1bounce(originPath, newPath, anchor, Jacobian) :
                path_shift_uvRemap_1bounce(originPath, newPath, anchor, Jacobian);
        } 

        bool ans = path_shift_tanScale(originPath, newPath, anchor, Jacobian, reverse);
        if (originPath.size() == 2)
        {
           // printf("pos compare %f %f, jacobian %f\n", float3weight(originPath.get(1).position), float3weight(newPath.get(1).position), Jacobian);
        }
        return ans;
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
    RT_FUNCTION bool path_shift_scale(const BDPTVertex* originPath, int path_size, BDPTVertex* newPath, float3 anchor, float& Jacobian)
    {
        float r1 = originPath[0].uv.x;
        float r2 = originPath[0].uv.y;
        newPath[1] = originPath[1];

        const BDPTVertex& midVertex = originPath[1];
        float3 in_dir = normalize(anchor - midVertex.position);
        MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        mat.base_color = make_float4(midVertex.color, 1.0);

        //nomal mapping
        float2 normal_map_uv;
        bool normal_map_hit = true;
        {
            float3 new_dir = Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, 0.5, 0.5);
            Tracer::PayloadBDPTVertex payload;
            payload.clear();
            payload.seed = 0;
            payload.ray_direction = new_dir;
            payload.origin = midVertex.position;
            init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


            float3 ray_direction = payload.ray_direction;
            float3 ray_origin = payload.origin;
            int begin_depth = payload.path.size;
            Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
                SCENE_EPSILON,  // tmin
                1e16f,  // tmax
                &payload);
            if (payload.path.size == begin_depth || payload.path.hit_lightSource() == false)
            {
                normal_map_hit = false;
            }
            else
            {
                normal_map_uv = payload.path.currentVertex().uv;
            }

        }

        float2 origin_uv = originPath[0].uv;
        float2 mapped_uv;
        float map_jacob;

        if (normal_map_hit)
        {
            map_jacob = 1;
            mapped_uv = scale_map(normal_map_uv, origin_uv, map_jacob);
            if (map_jacob < 0)return false;
        }
        else
        {
            map_jacob = 1;
            mapped_uv = origin_uv;
        }

        {
            float3 new_dir = Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, mapped_uv.x, mapped_uv.y);
            Tracer::PayloadBDPTVertex payload;
            payload.clear();
            payload.seed = 0;
            payload.ray_direction = new_dir;
            payload.origin = midVertex.position;
            init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


            float3 ray_direction = payload.ray_direction;
            float3 ray_origin = payload.origin;
            int begin_depth = payload.path.size;
            Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
                SCENE_EPSILON,  // tmin
                1e16f,  // tmax
                &payload);
            if (payload.path.size == begin_depth || payload.path.hit_lightSource() == false)
            {
                return false;
            }
            normal_map_uv = payload.path.currentVertex().uv;

            newPath[0] = originPath[0];

            newPath[0].position = payload.path.currentVertex().position;
            newPath[0].uv = payload.path.currentVertex().uv;
            float3 dirVec = newPath[0].position - midVertex.position;
            float pdf2 = Tracer::Pdf(mat, midVertex.normal, in_dir, new_dir) / dot(dirVec, dirVec) * abs(dot(new_dir, newPath[0].normal));
            float pdf1 = originPath[0].singlePdf;

            Jacobian = pdf2 / pdf1 * map_jacob;
        }

        return true;
    }
    RT_FUNCTION bool path_shift_quick_dirty(const BDPTVertex* originPath, int path_size, BDPTVertex* newPath, float3 anchor, float& Jacobian)
    {
        float r1 = originPath[0].uv.x;
        float r2 = originPath[0].uv.y;
        newPath[1] = originPath[1];

        const BDPTVertex& midVertex = originPath[1];
        float3 in_dir = normalize(anchor - midVertex.position);
        MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        mat.base_color = make_float4(midVertex.color, 1.0);
        float3 new_dir = Tracer::Sample_shift_metallic(mat, midVertex.normal, in_dir, r1, r2);

        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth || payload.path.hit_lightSource() == false)
        {
            return false;
        }
        newPath[0] = originPath[0];

        newPath[0].position = payload.path.currentVertex().position;
        newPath[0].uv = payload.path.currentVertex().uv;
        float3 dirVec = newPath[0].position - midVertex.position;
        float pdf2 = Tracer::Pdf(mat, midVertex.normal, in_dir, new_dir) / dot(dirVec, dirVec) * abs(dot(new_dir, newPath[0].normal));
        float pdf1 = originPath[0].singlePdf;

        Jacobian = pdf2 / pdf1;
        return true;
    }

    RT_FUNCTION float pdfCompute_pathShift_activate(const BDPTVertex* path, int path_size, int strategy_id)
    {
        int eyePathLength = strategy_id;
        int lightPathLength = path_size - eyePathLength;
        float pdf = 1.0;
        bool isShift = true;
        if (strategy_id < 2 || strategy_id >= path_size - 1) isShift = false;
        if (glossy(path[strategy_id]) == false) isShift = false;
        if (glossy(path[strategy_id - 1]) == true) isShift = false;
        for (int i = 1; i < strategy_id - 1; i++)
        {
            if (glossy(path[i]) == false) isShift = false;
        }
        if (lightPathLength > SHIFT_VALID_SIZE) isShift = false;
        if (isShift)
        {
            BDPTVertex newPathBuffer[SHIFT_VALID_SIZE];
            PathContainer newPath(newPathBuffer, 1, lightPathLength);
            PathContainer originPath(const_cast<BDPTVertex*>(path + strategy_id), 1, lightPathLength);
            float local_jacobian = 1;
            bool shift_good = path_shift(originPath, newPath, path[eyePathLength - 1].position, local_jacobian, true);
            pdf *= 1.0 / local_jacobian;
            if (shift_good == false) return 0;


            if (lightPathLength > 0)
            {
                const BDPTVertex& light = newPath.get(-1);
                pdf *= light.pdf;
            }
            if (lightPathLength > 1)
            {
                const BDPTVertex& light = newPath.get(-1);
                const BDPTVertex& lastMidPoint = newPath.get(-2);
                float3 lightLine = lastMidPoint.position - light.position;
                float3 lightDirection = normalize(lightLine);
                pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

                /*��������ǵ��µ�pdf*/
                for (int i = 1; i < lightPathLength; i++)
                {
                    const BDPTVertex& midPoint = newPath.get(-1 - i);
                    const BDPTVertex& lastPoint = newPath.get(-i);
                    float3 line = midPoint.position - lastPoint.position;
                    float3 lineDirection = normalize(line);
                    pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
                }

                for (int i = 1; i < lightPathLength - 1; i++)
                {
                    const BDPTVertex& midPoint = newPath.get(-1 - i);
                    const BDPTVertex& lastPoint = newPath.get(-i);
                    const BDPTVertex& nextPoint = newPath.get(-i - 2);
                    float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                    float3 nextDirection = normalize(nextPoint.position - midPoint.position);

                    MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
                    mat.base_color = make_float4(midPoint.color, 1.0);
                    float rr_rate = fmaxf(midPoint.color);
                    pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
                }
            }
        }
        else
        {
            /*��ԴĬ��Ϊ���Դ��һ�㣬��˿�����cos������ģ�������Ч��������ǵ��Դ��Ҫ�޸����´���*/
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

                /*��������ǵ��µ�pdf*/
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
        }
        /*����ͶӰ�ǵ��µ�pdf�仯*/
        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*��������ĸ���*/
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


    RT_FUNCTION float MISWeight_SPCBPT_pathShift_activate(const BDPTVertex* path, int path_size, int strategy_id)
    {
        if (strategy_id <= 1 || strategy_id == path_size)
        {
            return Tracer::pdfCompute(path, path_size, strategy_id);
        }
        int eyePathLength = strategy_id;
        int lightPathLength = path_size - eyePathLength;
        float pdf = 1.0;


        bool isShift = true;
        if (strategy_id < 2 || strategy_id >= path_size - 1) isShift = false;
        if (glossy(path[strategy_id]) == false) isShift = false;
        if (glossy(path[strategy_id - 1]) == true) isShift = false;
        for (int i = 1; i < strategy_id - 1; i++)
        {
            if (glossy(path[i]) == false) isShift = false;
        }
        if (lightPathLength > SHIFT_VALID_SIZE) isShift = false;



        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*��������ĸ���*/
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


        if (isShift)
        {
            BDPTVertex newPathBuffer[SHIFT_VALID_SIZE];
            PathContainer newPath(newPathBuffer, 1, lightPathLength);
            PathContainer originPath(const_cast<BDPTVertex*>(path + strategy_id), 1, lightPathLength);
            float local_jacobian = 1;
            bool shift_good = path_shift(originPath, newPath, path[eyePathLength - 1].position, local_jacobian, true);
            pdf *= 1.0 / local_jacobian;
            if (shift_good == false) return 0;


            float3 light_contri = make_float3(1.0);

            if (lightPathLength > 0)
            {
                const BDPTVertex& light = newPath.get(-1);
                light_contri *= light.flux;
            }

            if (lightPathLength > 1)
            {
                const BDPTVertex& light = newPath.get(-1);
                const BDPTVertex& lastMidPoint = newPath.get(-2);

                /*��������ǵ��µ�pdf*/
                for (int i = 1; i < lightPathLength; i++)
                {
                    const BDPTVertex& midPoint = newPath.get(-1 - i);
                    const BDPTVertex& lastPoint = newPath.get(-i);
                    float3 line = midPoint.position - lastPoint.position;
                    float3 lineDirection = normalize(line);
                    light_contri *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection)) * abs(dot(lastMidPoint.normal, lineDirection));
                }

                for (int i = 1; i < lightPathLength - 1; i++)
                {
                    const BDPTVertex& midPoint = newPath.get(-1 - i);
                    const BDPTVertex& lastPoint = newPath.get(-i);
                    const BDPTVertex& nextPoint = newPath.get(-i - 2);
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

            {
                position = newPath.get(0).position;
                normal = newPath.get(0).normal;
                dir = normalize(newPath.get(1).position - newPath.get(0).position);
                labelUnit lu(position, normal, dir, true);
                light_subspace_id = lu.getLabel();
            }
            return pdf * float3weight(connectRate_SOL(eye_subspace_id, light_subspace_id, light_contri));
        }
        else
        {
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

                /*��������ǵ��µ�pdf*/
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
    }

    RT_FUNCTION float3 eval_path(const BDPTVertex* path, int path_size, int strategy_id)
    {
        //return Tracer::contriCompute(path,path_size);
        float pdf = pdfCompute_pathShift_activate(path, path_size, strategy_id);
        float3 contri = Tracer::contriCompute(path, path_size);

        //float MIS_weight_not_normalize = MISWeight_SPCBPT_pathShift_activate(path, path_size, strategy_id);
        float MIS_weight_not_normalize = pdfCompute_pathShift_activate(path, path_size, strategy_id);
        float MIS_weight_dominator = 0.0;
        for (int i = 2; i <= path_size; i++)
        {
            MIS_weight_dominator += pdfCompute_pathShift_activate(path, path_size, i);
            //MIS_weight_dominator += MISWeight_SPCBPT_pathShift_activate(path, path_size, i);
        }

        float3 ans = contri / pdf * (MIS_weight_not_normalize / MIS_weight_dominator);
        if (ISINVALIDVALUE(ans))
        {
            return make_float3(0.0f);
        }
        return ans;
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
    RT_FUNCTION float3 evalFract_quick_and_dirty(PathContainer& path, float3 anchor, unsigned & seed)
    {
        if (path.size() != 2)
        {
            printf("eval Fract call at undesigned cases");
            return make_float3(1);
        }
        MaterialData::Pbr mat;
        mat = Tracer::params.materials[path.get(0).materialId];
        mat.base_color = make_float4(1.0);
        //mat.base_color = make_float4(path.get(0).color, 1.0);
        float3 normal = path.get(0).normal;
        float3 ans = make_float3(0.0);
        float3 factor = make_float3(1);
        float bound = 1.2;
//        fractChannelsFlag fcf;
        int it = 0;
        while (true)
        {
            it++;
            float3 in_dir = normalize(anchor - path.get(0).position);
            float3 new_dir = Tracer::Sample(mat, normal, in_dir, seed);

            Tracer::PayloadBDPTVertex payload;
            payload.clear();
            payload.seed = 0;
            payload.ray_direction = new_dir;
            payload.origin = path.get(0).position;
            init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


            float3 ray_direction = payload.ray_direction;
            float3 ray_origin = payload.origin;
            int begin_depth = payload.path.size;
            Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
                SCENE_EPSILON,  // tmin
                1e16f,  // tmax
                &payload);

            if (it > 100) {
                //printf("trace rays more than expectation\n");
                break;
            }
            if (payload.path.size == begin_depth)
            {
                continue;
            }

            if (payload.path.currentVertex().hit_lightSource() == false) continue;

            Tracer::lightSample light_sample;
            light_sample.ReverseSample(Tracer::params.lights[payload.path.currentVertex().materialId], payload.path.currentVertex().uv);
            init_vertex_from_lightSample(light_sample, path.get(1));

            float3 bsdf = Tracer::Eval(mat, normal, in_dir, new_dir);
            float3 out_vec = light_sample.position - path.get(0).position;
            float3 contri = bsdf * abs(dot(new_dir, normal) * dot(new_dir, light_sample.normal())) / dot(out_vec, out_vec);// *light_sample.emission;
            float pdf = Tracer::Pdf(mat, normal, in_dir, new_dir) * abs(dot(new_dir, light_sample.normal())) / dot(out_vec, out_vec);
            
//            bound = fmaxf(light_sample.emission);

            float3 fdp = contri / pdf / bound;
            ans += fdp * factor;
            float3 p_factor = make_float3(1) - fdp;
            float ac_rate = fmaxf(make_float3(abs(p_factor.x), abs(p_factor.y), abs(p_factor.z)));
            if (ac_rate > .5) ac_rate = .5;
            //printf("fdp example %f %f %f %f\n", ac_rate,fdp.x, fdp.y, fdp.z);
            BDPTVertex vBuffer[SHIFT_VALID_SIZE];
            PathContainer t_container(vBuffer, 1, path.size());
            float Jacobian;
            if (path_shift(path, t_container, anchor, Jacobian, true) == true)
            {
                float rr = rnd(seed);
                if (rr < ac_rate)
                {
                    factor *= p_factor / ac_rate;
                }
                else
                {
                    break;
                } 
            } 
            else
            {
               // printf("map failed\n");
            }
        }
        
        //printf("eval fract %f in %d times\n", fmaxf(ans),it);
        return ans;
    }
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
        if (a.type == ENV || b.type == ENV)
        {
            printf("Geometry Term call in Env light but we haven't implement it");
        }
        float3 diff = a.position - b.position;
        float3 dir = normalize(diff);
        return abs(dot(dir, a.normal) * dot(dir, b.normal)) / dot(diff, diff);
    }
    RT_FUNCTION bool retracing(PathContainer& path, float3 anchor, unsigned& seed, float3& contri, float& pdf, int re_trace_length)
    {
        MaterialData::Pbr mat;
        mat = Tracer::params.materials[path.get(0).materialId];
        mat.base_color = make_float4(path.get(0).color, 1.0);

        float3 in_dir = normalize(anchor - path.get(0).position);
        float3 new_dir = Tracer::Sample(mat, path.get(0).normal, in_dir, seed);

        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = seed;
        payload.ray_direction = new_dir;
        payload.origin = path.get(0).position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);

        contri = Tracer::Eval(mat, path.get(0).normal, in_dir, new_dir) * abs(dot(path.get(0).normal, new_dir));
        pdf = Tracer::Pdf(mat, path.get(0).normal, in_dir, new_dir);
        float3 ray_direction;
        float3 ray_origin;
        for (int i = 1; i < re_trace_length; i++)
        {
            ray_direction = payload.ray_direction;
            ray_origin = payload.origin;

            int begin_depth = payload.path.size;
            Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
                SCENE_EPSILON,  // tmin
                1e16f,  // tmax
                &payload);

            if (payload.path.size == begin_depth)//miss��
            { 
                //if (i == 2)
                //    info_print(path, anchor, ray_direction);
                return false;
            }
            if (payload.done == true && i != re_trace_length - 1)//��ǰ�����ˣ�����׷����·��������
            { 
                return false;
            }
            if (payload.path.hit_lightSource() == true && i != path.size() - 1)
            {
                return false;
            }

            if (i == path.size() - 1)
            {
                if (payload.path.hit_lightSource() == true)
                {
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(Tracer::params.lights[payload.path.currentVertex().materialId], payload.path.currentVertex().uv);
                    init_vertex_from_lightSample(light_sample, path.get(-1));
                    //printf("light_sourcce_pos %f %f %f\n",light_sample.position.x, light_sample.position.y,light_sample.position.z);
                    contri *= light_sample.emission;
                }
                else//׷�ٵ���·��û�򵽹�Դ��
                { 
                    return false;
                }
            }
            else if(i!= re_trace_length - 1)
            {
                if (glossy(payload.path.currentVertex()) == false)//δ��glossy·��
                {
                    return false;
                }
                //if (payload.path.hit_lightSource() == true)return false;
                mat = Tracer::params.materials[payload.path.currentVertex().materialId];
                mat.base_color = make_float4(payload.path.currentVertex().color, 1.0);
                
                float3 normal = payload.path.currentVertex().normal;
                float3 in_dir = -ray_direction;
                float3 out_dir = payload.ray_direction;
//                float3 out_dir = Tracer::Sample(mat, normal, in_dir, payload.seed);//payload.ray_direction;

                pdf *= Tracer::Pdf(mat,normal,in_dir,out_dir);
                contri *= Tracer::Eval(mat, normal, in_dir, out_dir) * abs(dot(normal, out_dir));
//                payload.origin = payload.path.currentVertex().position;
//                payload.ray_direction = out_dir;
                path.get(i) = payload.path.currentVertex();
            }
            else if(i == re_trace_length - 1)
            {
                path.get(i) = payload.path.currentVertex();
                if (Tracer::visibilityTest(Tracer::params.handle, path.get(i), path.get(i + 1)) == false)
                    return false;
                contri *= BSDF(path.get(i + 1), path.get(i), path.get(i - 1));
                if (path.size() -  re_trace_length >= 2)
                    contri *= BSDF(path.get(i + 2), path.get(i + 1), path.get(i));
                contri *= GeometryTerm(path.get(i), path.get(i + 1));
            }
            else
            {
                printf("retracing get a undesigned case\n");
                return false;
            }
        }


        return true;
    }
    RT_FUNCTION float3 evalFract(PathContainer& path, float3 anchor, unsigned& seed)
    { 
        //mat.base_color = make_float4(path.get(0).color, 1.0);
        int glossy_count = 1;
        for (int i = 1; i < path.size() - 1; i++)
        {
            if (glossy(path.get(i))) glossy_count++;
            else break;
        }
        int re_trace_length = glossy_count + 1; 
        float3 ans = make_float3(0.0);
        float3 factor = make_float3(1);
        float3 bound = make_float3(1.2);
        bool bound_set = false;
        if (re_trace_length == path.size() || re_trace_length == path.size() - 1) bound *= path.get(-1).flux;
        else
        {
            bound *= BSDF(path.get(re_trace_length + 1), path.get(re_trace_length), path.get(re_trace_length - 1));
        }
        if (re_trace_length != path.size())
        {
            bound *= GeometryTerm(path.get(re_trace_length), path.get(re_trace_length - 1));
        }

        for (int i = 0; i < re_trace_length - 1; i++)
        {
            bound *= fmaxf(path.get(i).color);
        }
        //        fractChannelsFlag fcf;
        int it = 0;
        int s_it = 0;
        int b_it = 0;
        float records[4];
        while (true)
        {
            it++;
            if (it > 20) {
                //printf("trace rays more than expectation\n");
                break;
            }

            float3 contri;
            float pdf;
            bool retracing_good = retracing(path, anchor, seed, contri, pdf, re_trace_length);
            if (retracing_good == false)
            {
                continue;
            }
            s_it++;
            if (bound_set == false)
            {
                bound = contri / pdf;
                bound_set = true;
                continue;
            }
            float3 fdp = contri / pdf / bound;
            //records[(s_it - 1) % 4] = float3weight(contri / pdf);
            ans += fdp * factor;
            float3 p_factor = make_float3(1) - fdp;
            float ac_rate = fmaxf(make_float3(abs(p_factor.x), abs(p_factor.y), abs(p_factor.z)));
            if (ac_rate > 0.8)
            {
                bound = contri / pdf;
                ans = make_float3(0);
                continue;
            }
            if (ac_rate > .5) ac_rate = .5;

            
            //if(path.size()!=2 && path.size() != re_trace_length)
            //    printf("fdp example %f %f %f %f\n", ac_rate, p_factor.x, p_factor.y, p_factor.z);
            BDPTVertex vBuffer[SHIFT_VALID_SIZE];
            PathContainer t_container(vBuffer, 1, path.size());
            float Jacobian;
            if (path_shift(path, t_container, anchor, Jacobian, true) == true)
            {
                b_it++;
                float rr = rnd(seed);
                if (rr < ac_rate)
                {
                    factor *= p_factor / ac_rate;
                }
                else
                {
                    break;
                }
            }
            else
            {
                // printf("map failed\n");
            }
        }
        
        //if (path.size() == 3) printf(" %f  %d %d %d %d %d\n", float3weight(ans),  it, s_it, b_it, path.size(), re_trace_length);
        //printf("%d %d %d record %f %f %f %f\n", re_trace_length, path.size(), s_it, s_it > 0 ? records[0] : 0, 
         //   s_it > 1 ? records[1] : 0, s_it > 2 ? records[2] : 0, s_it > 3 ? records[3] : 0);
        return ans;
    }



}

#endif // !CUPROG_H