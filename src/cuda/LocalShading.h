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

#include <cuda/LocalGeometry.h>
#include <cuda/MaterialData.h>
//#include <cuda/SimpleSample.h>
//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ T sampleTexture(MaterialData::Texture tex, const float2 uv)
{
    if (tex.tex)
    {
        const float2 UV = uv * tex.texcoord_scale;
        const float2 rotation = tex.texcoord_rotation;
        const float2 UV_trans = make_float2(
            dot(UV, make_float2(rotation.y, rotation.x)),
            dot(UV, make_float2(-rotation.x, rotation.y))) + tex.texcoord_offset;
        return tex2D<float4>(tex.tex, UV_trans.x, UV_trans.y);
    }
    else
    {
        return T();
    }
}

__device__ __forceinline__ void RoughnessAndMetallicTexSample(const float2 uv, MaterialData::Pbr& pbr)
{
    //float  metallic  = hit_group_data->material_data.pbr.metallic;
    //float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4(1.0f);
    if (pbr.metallic_roughness_tex)
        mr_tex = sampleTexture<float4>(pbr.metallic_roughness_tex, uv);
    pbr.roughness *= mr_tex.y;
    pbr.metallic *= mr_tex.z;
    pbr.roughness = max(0.001f, pbr.roughness);
    return;
}
template<typename T>
__device__ __forceinline__ T sampleTexture( MaterialData::Texture tex, const LocalGeometry &geom )
{
    //float2 uv = geom.texcoord[tex.texcoord].UV;
    //return sampleTexture<T>(tex, uv);
    if( tex.tex )
    {
        const float2 UV = geom.texcoord[tex.texcoord].UV * tex.texcoord_scale;
        const float2 rotation = tex.texcoord_rotation;
        const float2 UV_trans = make_float2(
            dot( UV, make_float2( rotation.y, rotation.x ) ),
            dot( UV, make_float2( -rotation.x, rotation.y ) ) ) + tex.texcoord_offset;
        return tex2D<float4>( tex.tex, UV_trans.x, UV_trans.y );
    }
    else
    {
        return T();
    }
}

//template<typename T>
//__device__ __forceinline__ T sampleTexture(MaterialData::Texture tex, const float2 uv)
//{
//    if (tex.tex)
//    {
//        const float2 UV = uv * tex.texcoord_scale;
//        const float2 rotation = tex.texcoord_rotation;
//        const float2 UV_trans = make_float2(
//            dot(UV, make_float2(rotation.y, rotation.x)),
//            dot(UV, make_float2(-rotation.x, rotation.y))) + tex.texcoord_offset;
//        return tex2D<float4>(tex.tex, UV_trans.x, UV_trans.y);
//    }
//    else
//    {
//        return T();
//    }
//}
//__device__ __forceinline__ void RoughnessAndMetallicTexSample(const float2 uv, MaterialData::Pbr& pbr)
//{
//    //float  metallic  = hit_group_data->material_data.pbr.metallic;
//    //float  roughness = hit_group_data->material_data.pbr.roughness;
//    float4 mr_tex = make_float4(1.0f);
//    if (pbr.metallic_roughness_tex)
//        mr_tex = sampleTexture<float4>(pbr.metallic_roughness_tex, uv);
//    pbr.roughness *= mr_tex.y;
//    pbr.metallic *= mr_tex.z;
//    return;
//}

__device__ __forceinline__ void RoughnessAndMetallicTexSample(const LocalGeometry& geom, MaterialData::Pbr& pbr)
{
    //float2 uv = geom.texcoord[pbr.metallic_roughness_tex.texcoord].UV;
    //RoughnessAndMetallicTexSample(uv, pbr);
    //return;

    //float  metallic  = hit_group_data->material_data.pbr.metallic;
    //float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4(1.0f);
    if (pbr.metallic_roughness_tex)
        mr_tex = sampleTexture<float4>(pbr.metallic_roughness_tex, geom);
    pbr.roughness *= mr_tex.y;
    pbr.metallic *= mr_tex.z;
    pbr.roughness = max(0.001f, pbr.roughness);
    return;
}