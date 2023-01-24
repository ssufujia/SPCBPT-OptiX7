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
#ifndef WHITTED_H
#define WHITTED_H


#include <vector_types.h>
#include<optix.h>
#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/Light.h>
#include <cuda/MaterialData.h>

#define RT_FUNCTION __device__ __forceinline__ 
namespace whitted
{

const unsigned int NUM_PAYLOAD_VALUES = 4u;
const unsigned int MAX_TRACE_DEPTH    = 8u;


struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};


//enum RayType
//{
//    RAY_TYPE_RADIANCE  = 0,
//    RAY_TYPE_OCCLUSION = 1,
//    RAY_TYPE_LIGHTSUBPATH = 2,
//    RAY_TYPE_COUNT = 3
//};


struct LaunchParams
{
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;
    float4*                  accum_buffer;
    uchar4*                  frame_buffer;
    int                      max_depth;
    //float                    scene_epsilon;
    //float                    scene_maximum;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<Light>        lights;
    BufferView<MaterialData::Pbr> materials;
    float3                   miss_color;
    OptixTraversableHandle   handle;

};

struct PayloadRadiance
{
    float3 vis_pos_A;
    float3 vis_pos_B;
    float3 currentResult;

    float3 result;
    float3 origin;
    float3 ray_direction;
    float3 throughput;
    float pdf;
    int    depth;
    unsigned int seed;
    bool done;
    bool glossy_bounce;
    short caustic_bounce_state;
    RT_FUNCTION PayloadRadiance()
    {
        result = make_float3(0); 
        throughput = make_float3(1.0);
        depth = 0;
        done = false;
        glossy_bounce = true;
        caustic_bounce_state = 0;
    }
};


struct PayloadOcclusion
{
    float3 result;
};


} // end namespace whitted

#endif // !WHITTED_H