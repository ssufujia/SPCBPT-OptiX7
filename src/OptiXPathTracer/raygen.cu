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
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/LocalShading.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "BDPTVertex.h"
#include "cuProg.h"
#include "pathControl.h"
#include "rmis.h"
 
__device__ inline float4 ToneMap_exposure(const float4& c, float exposure)
{
    float3 ldr = make_float3(1.0) - make_float3(exp(-c.x * exposure), exp(-c.y * exposure), exp(-c.z * exposure));
    return make_float4(ldr.x, ldr.y, ldr.z, 1.0f);
}
__device__ inline float4 ToneMap(const float4& c, float limit)
{
    //return ToneMap_exposure(c,limit);

    float luminance = 0.3f * c.x + 0.6f * c.y + 0.1f * c.z;

    float4 col = c * 1.0f / (1.0f + 1 * luminance / limit);
    return make_float4(col.x, col.y, col.z, 1.0f);
}

__device__ float3 hsv2rgb(int h, float s, float v)
{
    float C = v * s;
    float X = C * (1 - abs((float(h % 120) / 60) - 1));
    float m = v - C;
    float3 rgb_;
    if (h < 60)
    {
        rgb_ = make_float3(C, X, 0);
    }
    else if (h < 120)
    {
        rgb_ = make_float3(X, C, 0);
    }
    else if (h < 180)
    {
        rgb_ = make_float3(0, C, X);
    }
    else if (h < 240)
    {
        rgb_ = make_float3(0, X, C);
    }
    else if (h < 300)
    {
        rgb_ = make_float3(X, 0, C);
    }
    else
    {
        rgb_ = make_float3(C, 0, X);
    }
    return make_float3(m) + rgb_;
}

RT_FUNCTION float3 HeatMap(float e)
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (e < 0.25f)
    {
        r = 0.0f;
        g = 4.0f * e;
        b = 1.0f;
    }
    else if (e < 0.5f)
    {
        r = 0.0f;
        g = 1.0f;
        b = 2.0f - 4.0f * e;
    }
    else if (e < 0.75f)
    {
        r = 4.0f * e - 2.0f;
        g = 1.0f;
        b = 0.0f;
    }
    else
    {
        r = 1.0f;
        g = 4.0f - 4.0f * e;
        b = 0.0f;
    }

    return make_float3(r, g, b);
}

RT_FUNCTION uchar4 get_error_heat(float4 ref, float3 current)
{
    float3 bias = make_float3(ref) - current;
    float3 r_bias = (bias) / (make_float3(ref) + make_float3(Tracer::params.estimate_pr.min_limit));
    float diff = (abs(r_bias.x) + abs(r_bias.y) + abs(r_bias.z)) / 3;
    diff *= 100;
    diff = min(int(diff), 290);
    return make_color(hsv2rgb(((-int(diff) + 240) % 360), 1.0, 1.0));
}


extern "C" __global__ void __raygen__pinhole()
{

    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = Tracer::params.eye;
    const float3 U = Tracer::params.U;
    const float3 V = Tracer::params.V;
    const float3 W = Tracer::params.W;
    const int    subframe_index = Tracer::params.subframe_index;
     
    float3 normalizeV = normalize(V);

    /* Generate camera ray */
    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, subframe_index );

    /* The center of each pixel is at fraction(0.5, 0.5) */
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    const float2 d =
        2.0f * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                           ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) ) - 1.0f;

    float3 ray_direction = normalize( d.x * U + d.y * V + W );
    float3 ray_origin    = eye;

    /* Trace camera ray */

    Tracer::PayloadRadiance payload; 
    payload.seed          = seed; 
    payload.origin        = eye;
    payload.ray_direction = ray_direction;
    payload.currentResult = make_float3(0);
    while (true)
    {
        ray_direction = payload.ray_direction;
        ray_origin = payload.origin; 
        Tracer::traceRadiance(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload
        );

        payload.depth += 1;

        if (float3weight(payload.currentResult) > 0.0 && (payload.depth + 2 <= MAX_PATH_LENGTH_FOR_MIS || !LIMIT_PATH_TERMINATE))
        {
            const float  L_dist = length(payload.vis_pos_A- payload.vis_pos_B);
            const float3 L = (payload.vis_pos_B - payload.vis_pos_A) / L_dist;
            if (Tracer::visibilityTest(Tracer::params.handle, payload.vis_pos_A, payload.vis_pos_B))
                payload.result += payload.currentResult; 
            payload.currentResult = make_float3(0);
        }
        if (payload.done || (payload.depth + 1 >= MAX_PATH_LENGTH_FOR_MIS && LIMIT_PATH_TERMINATE)) {
            break;
        }
        
    }

    /* Update results */
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = payload.result;

    if( subframe_index > 0 )
    {
        const float  a                = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = make_float3( Tracer::params.accum_buffer[image_index] );
        accum_color                   = lerp( accum_color_prev, accum_color, a );
    }
    if (FIX_ITERATION&& subframe_index > 500)return;
    Tracer::params.accum_buffer[image_index] = make_float4( accum_color, 1.0f );

    float4 val = ToneMap(make_float4(accum_color, 0.0), 1.5);
    Tracer::params.frame_buffer[image_index] = make_color( make_float3(val) );

    if (Tracer::params.error_heat_visual && Tracer::params.estimate_pr.ready &&
        Tracer::params.estimate_pr.height == launch_dims.y && Tracer::params.estimate_pr.width == launch_dims.x)
    {
        float4 ref = Tracer::params.estimate_pr.ref_buffer[image_index];
        Tracer::params.frame_buffer[image_index] = get_error_heat(ref, accum_color);
    }
} 

RT_FUNCTION void init_lightSubPath_from_lightSample(Tracer::lightSample& light_sample, BDPTPath& p)
{
    p.clear();
    p.push();
    BDPTVertex& v = p.currentVertex();
     
    if (light_sample.bindLight->type == Light::Type::QUAD)
    {
        p.nextVertex().singlePdf = light_sample.dir_pdf;
    }
    else if (light_sample.bindLight->type == Light::Type::ENV)
    {
        p.nextVertex().singlePdf = light_sample.dir_pos_pdf;
    }

    init_vertex_from_lightSample(light_sample, v);
}


__device__ float3 direction_connect_SPCBPT(const BDPTVertex& a, const BDPTVertex& b)
{
    float3 L = make_float3(0.0f);
    float3 connectDir = -b.normal; 
    if (dot(a.normal, connectDir) > 0.0)
    { 
        MaterialData::Pbr mat_a = VERTEX_MAT(a); 
        float3 f = Tracer::Eval(mat_a, a.normal, normalize(a.lastPosition - a.position), connectDir)
            * dot(a.normal, connectDir);
        L = a.flux / a.pdf * f * b.flux / b.pdf * rmis::connection_direction_lightSource(a, b);
    }
    if (ISINVALIDVALUE(L))
    {
        return make_float3(0.0f);
    }
    return L;

}
__device__  float3 connectVertex_SPCBPT(const BDPTVertex& a, const BDPTVertex& b)
{
    if (b.is_DIRECTION())
    {
        return direction_connect_SPCBPT(a, b);
    }
    float3 connectVec = a.position - b.position;
    float3 connectDir = normalize(connectVec);
    float G = abs(dot(a.normal, connectDir)) * abs(dot(b.normal, connectDir)) / dot(connectVec, connectVec);
    float3 LA = a.lastPosition - a.position;
    float3 LA_DIR = normalize(LA);
    float3 LB = b.lastPosition - b.position;
    float3 LB_DIR = normalize(LB); 

    float3 fa, fb;
    MaterialData::Pbr mat_a = VERTEX_MAT(a);
    fa = Tracer::Eval(mat_a, a.normal, LA_DIR, -connectDir) ;

    MaterialData::Pbr mat_b;
    if (!b.isOrigin)
    {
        mat_b = VERTEX_MAT(b);
        fb = Tracer::Eval(mat_b, b.normal, connectDir, LB_DIR);
    }
    else
    {
        if (dot(b.normal, -connectDir) > 0.0f)
        {
            fb = make_float3(0.0f);
        }
        else
        {
            fb = make_float3(1.0f);
        }
    }

    float3 contri = a.flux * b.flux * fa * fb * G;
    float pdf = a.pdf * b.pdf; 
    //float3 ans = contri / pdf;// *(b.depth == 0 ? rmis::connection_lightSource(a, b) : rmis::general_connection(a, b));
    float3 ans = (a.flux / a.pdf) * (b.flux / b.pdf) * fa * fb * G
         *(b.depth == 0 ? rmis::connection_lightSource(a, b) : rmis::general_connection(a, b));

    if (ISINVALIDVALUE(ans))
    {
        return make_float3(0.0f);
    }
    return  ans;
}

RT_FUNCTION float3 lightStraghtHit(BDPTVertex& a)
{
    float3 contri = a.flux;
    float pdf = a.pdf;
    float inver_weight = a.RMIS_pointer;

    float3 ans = contri / pdf / inver_weight;
    if (ISINVALIDVALUE(ans))
    {
        return make_float3(0.0f);
    }
    return  ans;
}

extern "C" __global__ void __raygen__SPCBPT()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = Tracer::params.eye;
    const float3 U = Tracer::params.U;
    const float3 V = Tracer::params.V;
    const float3 W = Tracer::params.W;
    const int    subframe_index = Tracer::params.subframe_index;

    float3 normalizeV = normalize(V); 
    /* Generate camera ray */
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    /* The center of each pixel is at fraction(0.5, 0.5) */
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2(0.5f, 0.5f) : make_float2(rnd(seed), rnd(seed));

    const float2 d = 2.0f * make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
            (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y)) - 1.0f;

    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;
    float3 result = make_float3(0);

    Tracer::PayloadBDPTVertex payload;
    payload.clear();
    payload.seed = seed; 
    payload.ray_direction = ray_direction;
    payload.origin = ray_origin;
    
    init_EyeSubpath(payload.path, ray_origin, ray_direction);
       
    unsigned first_hit_id;
    while (true)
    {
        ray_direction = payload.ray_direction;
        ray_origin = payload.origin;
        if (payload.done || payload.depth > 50)
        {
            break;
        }
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            break;
        }
        payload.depth += 1;


        if (payload.path.hit_lightSource())
        {
            float3 res = lightStraghtHit(payload.path.currentVertex());
            if (payload.depth < MAX_PATH_LENGTH_FOR_MIS || !LIMIT_PATH_TERMINATE)
                result += res;
            break;
        }
        if (payload.depth >= MAX_PATH_LENGTH_FOR_MIS && SPCBPT_TERMINATE_EARLY)break;
        BDPTVertex& eye_subpath = payload.path.currentVertex();

        for (int it = 0; it < CONNECTION_N; it++)
        {

            int light_id = 0;
            float pmf_firstStage = 1;
            if (Tracer::params.subspace_info.light_tree)
            {
                light_id =
                    reinterpret_cast<Tracer::SubspaceSampler_device*>(&Tracer::params.sampler)->sampleFirstStage(eye_subpath.subspaceId, payload.seed, pmf_firstStage);
            }
            if (Tracer::params.sampler.subspace[light_id].size == 0)
            {
                continue;
            }
            float pmf_secondStage;
            const BDPTVertex& light_subpath =
                reinterpret_cast<Tracer::SubspaceSampler_device*>(&Tracer::params.sampler)->sampleSecondStage(light_id, payload.seed, pmf_secondStage);

            if (Tracer::visibilityTest(Tracer::params.handle, eye_subpath, light_subpath))
            { 
                //printf("debug info %f\n", float3weight(tmp_float3));
                float pmf = Tracer::params.sampler.path_count * pmf_secondStage * pmf_firstStage;
                float3 res = connectVertex_SPCBPT(eye_subpath, light_subpath) / pmf;
                if (!ISINVALIDVALUE(res) &&
                    (eye_subpath.depth + light_subpath.depth + 2 <= MAX_PATH_LENGTH_FOR_MIS || !LIMIT_PATH_TERMINATE))
                {
                    result += res / CONNECTION_N;
                }
            }
        } 
    } 
    
    /* env map */
    result += payload.result;

    /* Update results */
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = result;

    if (subframe_index > 0)
    {
        const float  a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(Tracer::params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    if (FIX_ITERATION && subframe_index > 100)return;
    Tracer::params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
     
    float4 val = ToneMap(make_float4(accum_color, 0.0), 1.5);
    Tracer::params.frame_buffer[image_index] = make_color(make_float3(val));   

    if (Tracer::params.error_heat_visual && Tracer::params.estimate_pr.ready &&
        Tracer::params.estimate_pr.height == launch_dims.y && Tracer::params.estimate_pr.width == launch_dims.x)
    {
        float4 ref = Tracer::params.estimate_pr.ref_buffer[image_index];
        Tracer::params.frame_buffer[image_index] = get_error_heat(ref, accum_color);
    }
}

RT_FUNCTION float dropOutTracing_MISWeight_non_normalize(const BDPTVertex* path, int path_size)
{
    int specular_index = -1;
    int surface_index = -1;
    int u = 1;
    for (int i = 1; i < path_size - 1; i++)
    {
        if (Shift::glossy(path[i + 1]) && !Shift::glossy(path[i]))
        {
            specular_index = i + 1;
            break;
        }
    }
    if (specular_index == -1)
    {
        printf("Error: non-caustic path is cast to the drop out tracing computation\n"); 
        return 0;
    }
    for (int i = specular_index + 1; i < path_size - 1; i++)
    {
        if (!Shift::glossy(path[i]))
        {
            surface_index = i + 1;
            break;
        }
        else
        {
            u += 1;
        }
    } 
    DropOutTracing_params& dot_params = Tracer::params.dot_params;
    int specular_subspace = dot_params.get_specular_label(path[specular_index].position, path[specular_index].normal);
    int surface_subspace = surface_index  == -1? DOT_EMPTY_SURFACEID: 
        dot_params.get_surface_label(path[surface_index].position, path[surface_index].normal, 
        normalize( path[surface_index+1].position - path[surface_index].position));
    
    float pdf = 1;
    int eye_subpath_end_index = surface_index == -1 ? path_size: surface_index;
    int h_star = surface_index;

    // Eye sub-path Pdf Computation
    // Retracing pdf is also computed
    {
        for (int i = 1; i < eye_subpath_end_index; i++)
        {
            if (i == specular_index) continue;
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        } 
        for (int i = 1; i < eye_subpath_end_index - 1; i++)
        {
            if (i == specular_index - 1) continue;
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            const BDPTVertex& nextPoint = path[i + 1];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);
             
            MaterialData::Pbr mat = VERTEX_MAT(midPoint);
            float rr_rate = i>=specular_index?1: Tracer::rrRate(mat);
            pdf *= i >= specular_index ? 
                Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate:
                Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position, true) * rr_rate;
        }
    }
    // Light sub-path Pdf Computation
    {
        int lightPathLength = path_size - eye_subpath_end_index;

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
                float rr_rate = Tracer::rrRate(mat);
                pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
            } 
        }
    }
    float dropOutTracingPdf = 0;
    if (dot_params.statistics_iteration_count != 0)
    {
        dropOut_tracing::statistics_data_struct& dot_statistic_data =
            dot_params.get_statistic_data(dropOut_tracing::pathLengthToDropOutType(u), specular_subspace, surface_subspace);
        dropOutTracingPdf = 1.0 / dot_statistic_data.average;
        if (DOT_LESS_MIS_WEIGHT)
        {
            float dropOutTracingPdf2 = dot_statistic_data.average / dot_statistic_data.variance;
            dropOutTracingPdf = dropOutTracingPdf2 * 2 < dropOutTracingPdf ? dropOutTracingPdf / 2 : dropOutTracingPdf;
        }
    }
     
    if (isinf(dropOutTracingPdf) || isnan(dropOutTracingPdf) || (dropOutTracingPdf < 0)) dropOutTracingPdf = 0.0;

    //printf("%f %f %f\n", pdf, dropOutTracingPdf, dot_params.selection_ratio(dropOut_tracing::pathLengthToDropOutType(u), specular_index, surface_index));

    labelUnit lu(path[specular_index - 1].position, path[specular_index - 1].normal, normalize(path[specular_index - 2].position - path[specular_index - 1].position), false);
    int eye_subspace_id = lu.getLabel();
    int specular_subspace_id = dot_params.get_specular_label(path[specular_index].position, path[specular_index].normal);
    return pdf * dropOutTracingPdf * dot_params.selection_ratio(eye_subspace_id, specular_subspace_id);
}

RT_FUNCTION float dropOutTracing_MISWeight(const BDPTVertex* path, int path_size)
{
    if (dropOut_tracing::debug_PT_ONLY) return 0;
    if (!dropOut_tracing::MIS_COMBINATION) return 1;

    float SPCBPT_pdf = Tracer::SPCBPT_MIS_sum_compute(path, path_size);
    float other_pdf = SPCBPT_pdf;
    float MIS_weight_not_normalize = dropOutTracing_MISWeight_non_normalize(path, path_size);
    float MIS_weight_denominator = other_pdf + MIS_weight_not_normalize; 
    return MIS_weight_not_normalize / MIS_weight_denominator; 
} 

RT_FUNCTION float3 eval_path(const BDPTVertex* path, int path_size, int strategy_id)
{
    //return Tracer::contriCompute(path,path_size);
    float pdf = Tracer::pdfCompute(path, path_size, strategy_id);
    float3 contri = Tracer::contriCompute(path, path_size);
    float3 ans = contri / pdf;

    //mis computation
    {
        float MIS_weight_not_normalize = Tracer::MISWeight_SPCBPT(path, path_size, strategy_id);
        float MIS_weight_dominator = 0.0;
        for (int i = 2; i <= path_size; i++)
        {
            MIS_weight_dominator += Tracer::MISWeight_SPCBPT(path, path_size, i);
        }

        ans = ans * (MIS_weight_not_normalize / MIS_weight_dominator);
    } 
    if (ISINVALIDVALUE(ans))
    {
        return make_float3(0.0f);
    }
    return ans;
}

extern "C" __global__ void __raygen__DropoutTracing()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = Tracer::params.eye;
    const float3 U = Tracer::params.U;
    const float3 V = Tracer::params.V;
    const float3 W = Tracer::params.W;
    const int    subframe_index = Tracer::params.subframe_index;

    float3 normalizeV = normalize(V);
    // Generate camera ray
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2(0.5f, 0.5f) : make_float2(rnd(seed), rnd(seed));

    const float2 d =
        2.0f * make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
            (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y)) - 1.0f;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;
    float3 result = make_float3(0);

    Tracer::PayloadBDPTVertex payload;
    payload.clear();
    payload.seed = seed;
    payload.ray_direction = ray_direction;
    payload.origin = ray_origin;
    /* 视子路初始化 */
    init_EyeSubpath(payload.path, ray_origin, ray_direction);

    BDPTVertex pathBuffer[MAX_PATH_LENGTH_FOR_MIS];
    int buffer_size = 0;
    pathBuffer[buffer_size++] = payload.path.currentVertex();

    unsigned first_hit_id;
    BDPTVertex SP;
    BDPTVertex CP;
    int u;
    float3 WC;

    /* 视子路追踪主循环 */
    while (true)
    {
        ray_direction = payload.ray_direction;
        ray_origin = payload.origin;
        if (payload.done || payload.depth > 50)
            break;
        int begin_depth = payload.path.size;
        /* 视子路追踪 */
        Tracer::traceEyeSubPath(
            Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload
        );

        /* 没打中 */
        if (payload.path.size == begin_depth)
            break;

        /* 记录一下历史路径 */
        payload.path_record = (payload.path_record) |
            ((long long)Shift::glossy(payload.path.currentVertex()) << payload.depth);
        if (buffer_size < MAX_PATH_LENGTH_FOR_MIS)
        {
            pathBuffer[buffer_size++] = payload.path.currentVertex();
            if (buffer_size > 2 && Shift::vertex_very_close(pathBuffer[buffer_size - 1], pathBuffer[buffer_size - 2]))
            {
                payload.path_record = (payload.path_record) |
                    (((long long)true) << (payload.depth - 1));
            }
        }
        payload.depth += 1;

        /* 如果打中了光源 */
        float3 res = make_float3(0.0);
        if (payload.path.hit_lightSource())
        {
            if (dropOut_tracing::debug_PT_ONLY)
            {
                Tracer::lightSample light_sample;
                light_sample.ReverseSample(Tracer::params.lights[payload.path.currentVertex().materialId], payload.path.currentVertex().uv);

                BDPTVertex light_vertex;
                init_vertex_from_lightSample(light_sample, light_vertex);
                pathBuffer[buffer_size - 1] = light_vertex;
                if (Shift::getCausticPathInfo(pathBuffer, buffer_size, SP, CP, u, WC) && Shift::valid_specular(CP, SP, u, WC))
                {
                    float pdf = Tracer::pdfCompute(pathBuffer, buffer_size, buffer_size);
                    float3 contri = Tracer::contriCompute(pathBuffer, buffer_size);
                    res = contri / pdf * (1 - dropOutTracing_MISWeight(pathBuffer, buffer_size));
                }
            }
            else
            {
                Tracer::lightSample light_sample;
                light_sample.ReverseSample(Tracer::params.lights[payload.path.currentVertex().materialId], payload.path.currentVertex().uv);

                BDPTVertex light_vertex;
                init_vertex_from_lightSample(light_sample, light_vertex);
                pathBuffer[buffer_size - 1] = light_vertex;

                if (Shift::getCausticPathInfo(pathBuffer, buffer_size, SP, CP, u, WC) && Shift::valid_specular(CP, SP, u, WC) && !Tracer::params.spcbpt_pure)
                {
                    //float pdf = Tracer::pdfCompute(pathBuffer, buffer_size, buffer_size);
                    //float3 contri = Tracer::contriCompute(pathBuffer, buffer_size);
                    //res = contri / pdf * (1 - dropOutTracing_MISWeight(pathBuffer, buffer_size));
                    res = lightStraghtHit(payload.path.currentVertex()) * (1 - dropOutTracing_MISWeight(pathBuffer, buffer_size));
                    //if (Tracer::params.caustic_path_only) res = make_float3(float3weight(res), 0, float3weight(res));
                }
                else if (!Tracer::params.caustic_path_only)
                {
                    res = lightStraghtHit(payload.path.currentVertex());
                }
            }
            if (ISINVALIDVALUE(res))
            {
                res = make_float3(0.0f);
            }
            result += res;
            break;
        }

        if (buffer_size >= MAX_PATH_LENGTH_FOR_MIS)
            break;

        BDPTVertex& eye_vertex = payload.path.currentVertex();

        for (int it = 0; it < CONNECTION_N; it++)
        {
            if (dropOut_tracing::debug_PT_ONLY) continue;
            /* 暂时取消LSDE这样的光路筛选方式，改用dropoutTracing_common.h里根据四参数的形式来决定是否连接的方式，仅要求视子路为diffuse */

            dropOut_tracing::pixelRecord& pixel_record =
                Tracer::params.dot_params.pixel_record[Tracer::params.dot_params.pixel2Id(make_uint2(launch_idx.x, launch_idx.y), make_uint2(launch_dims.x, launch_dims.y))];

            bool caustic_eye = Shift::pathRecord_is_causticEyesubpath(payload.path_record, payload.depth);

            int light_id = 0;
            float pmf_firstStage = 1;
            if (Tracer::params.subspace_info.light_tree)
            {
                light_id =
                    reinterpret_cast<Tracer::SubspaceSampler_device*>(&Tracer::params.sampler)->sampleFirstStage(eye_vertex.subspaceId, payload.seed, pmf_firstStage);
            }
            if (Tracer::params.sampler.subspace[light_id].size == 0)
            {
                continue;
            }
            float pmf_secondStage;
            const BDPTVertex& light_subpath =
                reinterpret_cast<Tracer::SubspaceSampler_device*>(&Tracer::params.sampler)->sampleSecondStage(light_id, payload.seed, pmf_secondStage);
            //if (Shift::glossy(light_subpath))continue;
            //if (Shift::glossy(eye_vertex))continue;
            if (Tracer::visibilityTest(Tracer::params.handle, eye_vertex.position, light_subpath.position) &&
                (eye_vertex.depth + light_subpath.depth + 2 <= MAX_PATH_LENGTH_FOR_MIS || !LIMIT_PATH_TERMINATE))
            {
                float pmf = Tracer::params.sampler.path_count * pmf_secondStage * pmf_firstStage;

                float3 res;
                res = connectVertex_SPCBPT(eye_vertex, light_subpath) / pmf;


                Shift::PathContainer originPath(const_cast<BDPTVertex*>(&light_subpath), -1, light_subpath.depth + 1);
                bool caustic_flag = false;
                if (!Tracer::params.spcbpt_pure && Shift::path_alreadyCaustic(pathBuffer, buffer_size, originPath, light_subpath.depth) &&
                    light_subpath.depth + 1 < SHIFT_VALID_SIZE && eye_vertex.depth + light_subpath.depth + 2 <= MAX_PATH_LENGTH_FOR_MIS)
                {
                    Shift::PathContainer tempPath(const_cast<BDPTVertex*>(&light_subpath), -1, 0);
                    int path_size = Shift::dropoutTracing_concatenate(pathBuffer, buffer_size, 0, tempPath, originPath);

                    if (Shift::getCausticPathInfo(pathBuffer, path_size, SP, CP, u, WC) && Shift::valid_specular(CP, SP, u, WC))
                    {
                        res *= (1 - dropOutTracing_MISWeight(pathBuffer, path_size));
                        caustic_flag = true;
                    }
                }

                if (!ISINVALIDVALUE(res))
                {
                    if (!(Tracer::params.caustic_path_only && caustic_flag == false))
                    {
                        //if (Tracer::params.caustic_path_only && caustic_flag == true)
                        //    res = make_float3(float3weight(res), 0, float3weight(res));
                        result += res / CONNECTION_N;    
                    }
                    if (caustic_eye)
                    {
                        pixel_record.is_valid = true;
                        pixel_record.is_caustic_record = false;
                        pixel_record.record = float3weight(res);
                    }
                }
            }
        }


        const int num_specular_conn = dropOut_tracing::specular_connection_N;
        for (int it = 0; it < num_specular_conn; it++)
        {
            if (!Shift::pathRecord_is_causticEyesubpath(payload.path_record, payload.depth)) break;
            if (Tracer::params.spcbpt_pure) break;
            float caustic_connection_prob = Tracer::params.dot_params.get_caustic_prob(
                make_uint2(launch_idx.x, launch_idx.y), make_uint2(launch_dims.x, launch_dims.y));
            //caustic_connection_prob = 1;
            float rr_k = caustic_connection_prob;// caustic_connection_prob > 0.01 ? 1 : caustic_connection_prob;
            if (RR_TEST(seed, rr_k) == false) continue;
            dropOut_tracing::pixelRecord& pixel_record =
                Tracer::params.dot_params.pixel_record[Tracer::params.dot_params.pixel2Id(make_uint2(launch_idx.x, launch_idx.y), make_uint2(launch_dims.x, launch_dims.y))];
            pixel_record.is_valid = true;
            pixel_record.record = 0;
            pixel_record.is_caustic_record = true;
            pixel_record.eyeId = eye_vertex.subspaceId;
        

            {
#define MMIS_N 16
                Tracer::SubspaceSampler_device*  LVC_sampler = reinterpret_cast<Tracer::SubspaceSampler_device*>(&Tracer::params.sampler);
                int light_subpath_index[MMIS_N];
                const BDPTVertex* light_subpath[MMIS_N];
                float final_pmf[MMIS_N];

                for (int i = 0; i < MMIS_N; ++i) {

                    float pmf_firstStage = 1;
                    float pmf_secondStage;

                    int light_subspaceId = LVC_sampler->SampleGlossyFirstStage(eye_vertex.subspaceId, payload.seed, pmf_firstStage);


                    if (Tracer::params.sampler.glossy_subspace_num[light_subspaceId] == 0 || light_subspaceId == DOT_INVALID_SPECULARID) {
                        light_subpath_index[i] = -1;
                        light_subpath[i] = nullptr;
                        continue; 
                    }

                    light_subpath_index[i] = LVC_sampler->SampleGlossySecondStageIndex(light_subspaceId, payload.seed, pmf_secondStage);
                    light_subpath[i] = &LVC_sampler->getVertexByIndex(light_subpath_index[i]);

                    final_pmf[i] = pmf_firstStage * pmf_secondStage;

                    pixel_record.specularId = abs(LVC_sampler->getVertexByIndex(light_subpath_index[i]).get_specular_id());
                } 

                for (int i = 0; i < MMIS_N; ++i) {
                    if (light_subpath_index[i] == -1) continue;
                    if (
                        (eye_vertex.depth + light_subpath[i]->depth + 2 <= MAX_PATH_LENGTH_FOR_MIS) &&
                        (Tracer::visibilityTest(Tracer::params.handle, eye_vertex.position, light_subpath[i]->position)))
                    {
                        float pmf = Tracer::params.sampler.path_count * (1 - Tracer::params.dot_params.discard_ratio)
                            * final_pmf[i] * caustic_connection_prob;

                        if (light_subpath[i]->depth>0 && light_subpath[i]->depth < SHIFT_VALID_SIZE - 1) {
                            BDPTVertex light_sub_new[SHIFT_VALID_SIZE];
                            Shift::PathContainer originPath(const_cast<BDPTVertex*>(light_subpath[i]), -1, light_subpath[i]->depth + 1);
                            Shift::PathContainer finalPath(light_sub_new, 1);

                            //if CP=NO Vertex, CP.pdf = 1 is set at get_imcomplete_subpath_info 
                            u = Shift::get_imcomplete_subpath_info(originPath, SP, CP, WC);
                            if (!Shift::valid_specular(CP, SP, u, WC)) continue;

                            float retracing_pdf;
                            //bool retrace_success = Shift::retracing(payload.seed, finalPath, light_subpath, CP, normalize(eye_vertex.position - light_subpath.position), u, retracing_pdf);
                            bool retrace_success = Shift::retracing_with_reference(
                                payload.seed, finalPath, *light_subpath[i], CP, normalize(eye_vertex.position - light_subpath[i]->position), u, retracing_pdf, originPath);
                            if (retrace_success == false) continue;

                            /* This function put the eye-subpath and light-subpath in pathBuffer */
                            int path_size = Shift::dropoutTracing_concatenate(pathBuffer, buffer_size, u, finalPath, originPath);

                            float pdf = eye_vertex.pdf * retracing_pdf * CP.pdf;

                            float3 contri = Tracer::contriCompute(pathBuffer, path_size);

                            /* ------------------------------------------------------------------------------ */

                            float3 res = (contri / pdf / pmf) / SP.singlePdf;
                            // printf("%f %f\n", SP.singlePdf, light_subpath.inverPdfEst);
                            // float3 res = (contri / pdf / pmf) * light_subpath[i]->inverPdfEst;

                            /* ------------------------------------------------------------------------------ */

                            pixel_record.record = float3weight(res);
                            float tmp = dropOutTracing_MISWeight(pathBuffer, path_size);
                            //printf("%f\n", tmp);
                            res *= tmp;
                            
                            float samplePdf[MMIS_N];
                            for (int j = 0; j < MMIS_N; ++j) {
                                if (j == i) {
                                    samplePdf[j] = SP.singlePdf;
                                }
                                else if (light_subpath[j] == nullptr) {
                                    samplePdf[j] = 0;
                                }
                                else if (CP.type != BDPTVertex::Type::DROPOUT_NOVERTEX) {
                                    samplePdf[j] = 0;
                                }
                                else if (light_subpath[j]->depth <= 0) {
                                    samplePdf[j] = 0;
                                }
                                else {
                                    float retracing_pdf;
                                    Shift::PathContainer originPath_i(const_cast<BDPTVertex*>(light_subpath[j]), -1, light_subpath[j]->depth + 1);
                                    BDPTVertex SP_i, CP_i;
                                    int u_i;
                                    float3 WC_i;
                                    u_i = Shift::get_imcomplete_subpath_info(originPath_i, SP_i, CP_i, WC_i);
                                    if (CP_i.type != BDPTVertex::Type::DROPOUT_NOVERTEX) {
                                        samplePdf[j] = 0;
                                    }
                                    else {
                                        const BDPTVertex& lastVertex = LVC_sampler->getVertexByIndex(light_subpath_index[j] - 1);
                                        samplePdf[j] = Shift::tracingPdf(lastVertex, SP,  normalize(lastVertex.lastPosition- lastVertex.position), true);
                                    }
                                }
                            }
                            
                            float pdfSum = 0;
                            for (int j = 0; j < MMIS_N; ++j) {
                                //printf("i:%i  j:%i  pdf:%f\n", i, j, samplePdf[j]);
                                pdfSum += samplePdf[j];
                            }

                            float cmisWeight = samplePdf[i] / pdfSum;
                            //printf("%f\n", cmisWeight);

                            res *= cmisWeight;


                            if (!ISINVALIDVALUE(res)) {
                                //if (Tracer::params.caustic_path_only) res = make_float3(0, float3weight(res), float3weight(res));
                                result += res / num_specular_conn;
                            }
                            else {
                                pixel_record.record = 0.0;
                            }
                        }
                    }
                }
            }
        }

        if (eye_vertex.depth == 1 && Tracer::params.eye_subspace_visualize || Tracer::params.light_subspace_visualize || Tracer::params.specular_subspace_visualize
            || Tracer::params.caustic_prob_visualize || Tracer::params.PG_grid_visualize)
        {
            unsigned vis_id = 0;
            if (Tracer::params.eye_subspace_visualize) {
                labelUnit lu(eye_vertex.position, eye_vertex.normal, eye_vertex.normal, false);
                vis_id = lu.getLabel();
            }
            else if (Tracer::params.light_subspace_visualize) {
                labelUnit lu(eye_vertex.position, eye_vertex.normal, eye_vertex.normal, true);
                vis_id = lu.getLabel();
            }
            else if (Tracer::params.specular_subspace_visualize) {
                vis_id = Tracer::params.dot_params.get_specular_label(eye_vertex.position, eye_vertex.normal);
            }
            else if (Tracer::params.PG_grid_visualize && Tracer::params.pg_params.pg_enable)
            {
                vis_id = Tracer::params.pg_params.getStreeId(eye_vertex.position);
            }
            result = make_float3(rnd(vis_id), rnd(vis_id), rnd(vis_id));
            if (Tracer::params.caustic_prob_visualize)
            {
                float caustic_connection_prob = Tracer::params.dot_params.get_caustic_prob(
                    make_uint2(launch_idx.x, launch_idx.y), make_uint2(launch_dims.x, launch_dims.y));
                result = make_float3(caustic_connection_prob);
            }
            break;
        }
    }
    
    /* Update results */
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = result;

    if (subframe_index > 0)
    {
        const float  a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(Tracer::params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);

        if (accum_color.x < 0 || accum_color.y < 0 || accum_color.z < 0)
        {
            accum_color = accum_color_prev;
        }
    }
    if (accum_color.x < 0 || accum_color.y < 0 || accum_color.z < 0)
    {
        accum_color = make_float3(0);
    }
    if (FIX_ITERATION && subframe_index > 100)return;
    Tracer::params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);

    float4 val = ToneMap(make_float4(accum_color, 0.0), 1.5);
    Tracer::params.frame_buffer[image_index] = make_color(make_float3(val));

    if (Tracer::params.error_heat_visual && Tracer::params.estimate_pr.ready &&
        Tracer::params.estimate_pr.height == launch_dims.y && Tracer::params.estimate_pr.width == launch_dims.x)
    {
        float4 ref = Tracer::params.estimate_pr.ref_buffer[image_index];
        Tracer::params.frame_buffer[image_index] = get_error_heat(ref, accum_color);
    }

    //if (Tracer::params.caustic_path_only)
    //{
    //    float diff = accum_color.y / accum_color.z * 290;
    //    Tracer::params.frame_buffer[image_index] = make_color(hsv2rgb(((-int(diff) + 240) % 360), 1.0, 1.0));
    //}
}

#define CheckLightBufferState if(!(lightVertexCount<lt_params.core_padding)) break; 

RT_FUNCTION void pushVertexToLVC(BDPTVertex& v, unsigned int& putId, int bufferBias)
{
    const LightTraceParams& lt_params = Tracer::params.lt;
    lt_params.ans[putId + bufferBias] = v;
    lt_params.validState[putId + bufferBias] = true;
    putId++;
} 

extern "C" __global__ void __raygen__lightTrace()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const int    subframe_index = Tracer::params.lt.launch_frame;
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    Tracer::PayloadBDPTVertex payload;
    payload.seed = seed;

    const LightTraceParams& lt_params = Tracer::params.lt;
    int launch_index = launch_idx.x;
    unsigned int bufferBias = lt_params.core_padding * launch_index;
    unsigned int lightVertexCount = 0;
    unsigned int lightPathCount = 0;

    unsigned int DOT_record_count = 0;
    unsigned int DOT_buffer_bias = Tracer::params.dot_params.record_buffer_padding * launch_index;
    for (int i = 0; i < lt_params.core_padding; i++)
    {
        lt_params.validState[i + bufferBias] = false;
        lt_params.ans[i + bufferBias].specular_record = DOT_INVALID_SPECULARID;
        //lightVertexCount++;
    }

    while (true)
    {
        payload.clear();
        int light_id = clamp(static_cast<int>(floorf(rnd(seed) * Tracer::params.lights.count)), 
                                int(0), int(Tracer::params.lights.count - 1));
        const Light& light = Tracer::params.lights[light_id];
        Tracer::lightSample light_sample; 
        light_sample(light, seed); 

        light_sample.traceMode(seed);
        float3 ray_direction = light_sample.trace_direction();
        float3 ray_origin = light_sample.position; 
        init_lightSubPath_from_lightSample(light_sample, payload.path);
        /* lightVertexCount 在经过这个函数后会加 1 */
        pushVertexToLVC(payload.path.currentVertex(), lightVertexCount, bufferBias); 
        CheckLightBufferState;

        /* 光子路追踪 */
        while (true)
        {
            int begin_depth = payload.path.size;
            Tracer::traceLightSubPath(
                Tracer::params.handle, 
                ray_origin, 
                ray_direction,
                SCENE_EPSILON,  // tmin
                1e16f,  // tmax
                &payload
            );
            /* 如果打中了面片 */
            if (payload.path.size > begin_depth) 
            {
                BDPTVertex& curVertex = payload.path.currentVertex();
                /* 将历史信息记录进 pathrecord */
                payload.path_record = (payload.path_record) |
                    ((long long)Shift::glossy(curVertex) << payload.depth);
                curVertex.path_record = payload.path_record;
                 
                float e = curVertex.contri_float();

                if (e < 0.00001) 
                    break;
                /* lightVertexCount 在经过这个函数后会加 1 */
                pushVertexToLVC(curVertex, lightVertexCount, bufferBias); 
                CheckLightBufferState;
                 
                if (!Tracer::params.spcbpt_pure&&Shift::glossy(curVertex) && curVertex.depth < SHIFT_VALID_SIZE&&
                    RR_TEST(seed, 1 - Tracer::params.dot_params.discard_ratio))
                {
                    BDPTVertex& glossy_vertex = Tracer::params.lt.ans[lightVertexCount + bufferBias - 1];
                    Shift::PathContainer light_subpath(&(glossy_vertex), -1, curVertex.depth + 1);
                     
                    // @param CP: BDPTVertex, control point
                    // @param SP: BDPTVertex, specular point
                    // @param WC: float3, control direction
                    // @param u: integer, step size
                    // @param statistic_prd: statistic_payload, interface for using statistic information
                    BDPTVertex CP,SP; 
                    float3 WC;
                    int u;
                    //if CP=NO Vertex, CP.pdf = 1 is set at get_imcomplete_subpath_info
                    u = Shift::get_imcomplete_subpath_info(light_subpath, SP, CP, WC); 

                    if (Shift::valid_specular(CP, SP, u, WC) && dropOut_tracing::debug_PT_ONLY == false)
                    {
                        statistic_payload statistic_prd;
                        statistic_prd.build(curVertex, CP, WC, u);
                        statistic_prd.putId = &DOT_record_count;
                        statistic_prd.bufferBias = DOT_buffer_bias;
                        float3 tempV;
                        if (statistic_prd.data.valid == false)
                            statistic_prd.data.bound = 1;
                        if (statistic_prd.data.bound > dropOut_tracing::max_bound&&statistic_prd.type!=dropOut_tracing::DropOutType::LS)
                            statistic_prd.data.bound = dropOut_tracing::max_bound;

                        if (false&&statistic_prd.type == dropOut_tracing::pathLengthToDropOutType(1))
                        { 
                            statistic_prd.data.bound = Shift::getClosestGeometry_upperBound(Tracer::params.lights[0], SP.position, SP.normal, tempV);
                        }

                        if (statistic_prd.subspace_valid())
                        {
                            float pdf_inverse = 0;
                            glossy_vertex.inverPdfEst = 0;
                            
                            /*
                            for (int it = 0; it < dropOut_tracing::reciprocal_iteration; it++)
                            {
                                float t = 1.0 / (1 + it);
                                float recip = Shift::reciprocal_estimation(payload.seed, CP, SP, WC, u, statistic_prd);
                                pdf_inverse = pdf_inverse * (1 - t) + recip * t;
                            }                                                        
                            if (isnan(pdf_inverse) || isinf(pdf_inverse) || pdf_inverse < 0)glossy_vertex.inverPdfEst = 0;
                            else glossy_vertex.inverPdfEst = pdf_inverse;
                            */

                            /* Average Record Create */
                            DOT_record dot_record = statistic_prd.generate_record(DOT_usage::Average);
                            dot_record = pdf_inverse;
                            DOT_pushRecordToBuffer(dot_record, DOT_record_count, DOT_buffer_bias);
                            /* Average Record Finish */
                        }
                        else
                        {
                            glossy_vertex.inverPdfEst = 0;
                        } 
                        glossy_vertex.set_specular_id(statistic_prd.SP_label);
                    }
                }
            }
            ray_direction = payload.ray_direction;
            ray_origin = payload.origin;
            if (payload.done || payload.depth > 50)
                break;
            payload.depth += 1;
        }
        lightPathCount++;
        if (lightPathCount >= lt_params.M_per_core)
            break;
        CheckLightBufferState; 
    } 

}

extern "C" __global__ void __miss__constant_radiance()
{
    Tracer::PayloadRadiance* prd = Tracer::getPRD();
    prd->done = true;
    prd->currentResult = make_float3(0);
    if (prd->depth == 0&&SKY.valid)
    {
        prd->result = prd->throughput* SKY.color(prd->ray_direction); 
    }
}


extern "C" __global__ void __miss__env__BDPTVertex()
{
    if (SKY.valid == false)
    {
        Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
        prd->done = true;
        return;
    }
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    prd->done = true;

    if (prd->path.size == 1)
    {
        prd->result = SKY.color(prd->ray_direction);
        return;
    }
    //return;
    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex(); 
    BDPTVertex& LastVertex = prd->path.lastVertex(); 
    MidVertex.normal = -prd->ray_direction;

    MidVertex.type = BDPTVertex::Type::ENV_MISS;
    MidVertex.uv = dir2uv(prd->ray_direction); 
    Tracer::lightSample light_sample;
    light_sample.ReverseSample(Tracer::params.lights[SKY.light_id], MidVertex.uv);

    float lightPdf = light_sample.pdf;

    float pdf_G = abs(dot(MidVertex.normal, prd->ray_direction) * dot(LastVertex.normal, prd->ray_direction));
    if (LastVertex.isOrigin)
    {
        MidVertex.flux = LastVertex.flux * pdf_G * SKY.color(prd->ray_direction);
    }
    else
    {
        MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G * SKY.color(prd->ray_direction);
    }

    MidVertex.lastPosition = LastVertex.position;
    MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, prd->ray_direction));

    //MidVertex.zoneId = SUBSPACE_NUM - lightMaterialId - 1;
    MidVertex.subspaceId = SKY.getLabel(prd->ray_direction);
    //MidVertex.zoneId = -1;
    MidVertex.lastZoneId = LastVertex.subspaceId;


    MidVertex.singlePdf = MidVertex.singlePdf;
    MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

    //MidVertex.dLast = LastVertex.d;

    MidVertex.depth = LastVertex.depth + 1;
      
    float3 dir = -MidVertex.normal;
    BDPTVertex virtual_light;
    rmis::construct_virtual_env_light(virtual_light, SKY.color(dir), light_sample.pdf, dir, SKY.getLabel(dir));
    float dd = rmis::light_hit_env(LastVertex, virtual_light);
    //printf("env hit compare %f %f\n", dd, 1.0 / MidVertex.d);
    MidVertex.RMIS_pointer = 1.0 / dd;

}

extern "C" __global__ void __miss__BDPTVertex()
{
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    prd->done = true;
//    prd->result = SKY.color(prd->ray_direction);
}



RT_FUNCTION void PreTrace_buildPathInfo(BDPTVertex* eye, TrainData::nVertex_device light, preTracePath* path, preTraceConnection* conn, int pathSize)
{
    float3 light_contri = light.weight;
    BDPTVertex* eye_end = eye;
    //check if the path is caustic path
    {
        path->is_caustic = false;
        BDPTVertex* eye_subpath_it = eye - eye->depth + 1;
        for (int i = 1; i < eye->depth; i++, eye_subpath_it++)
        {
            if (Shift::glossy(*eye_subpath_it))continue;
            else if (Shift::glossy(*(eye_subpath_it + 1)))
            {
                path->is_caustic = true;
                path->caustic_id = i - 1;
            }
            else
            {
                break;
            }
        }
    }

    path->valid = true;
    
    path->begin_ind = 0;
    path->end_ind = pathSize - 1;
    //path->end_ind = 0;
    //return;

    path->sample_pdf = 0;
     
    TrainData::nVertex_device n_eye = TrainData::nVertex_device(*eye, true);

    //printf("material ID %d %d %d\n", n_eye.materialId, eye->materialId, eye->depth);
    TrainData::nVertex_device n_next_eye = TrainData::nVertex_device(light, n_eye, true); 
    float3 seg_contri = n_eye.local_contri(light);

    path->sample_pdf = n_next_eye.pdf; 
    path->sample_pdf += n_eye.pdf * light.pdf;
    path->fix_pdf = n_next_eye.pdf;
    path->contri = eye->flux * light.forward_light(n_eye) * seg_contri;
    for (int i = 0; i < path->end_ind; i++)
    {
        conn[path->end_ind - i - 1] = preTraceConnection(n_eye, light);
        eye--;
        light = TrainData::nVertex_device(n_eye, light, false);
        n_eye = TrainData::nVertex_device(*eye, true);
    }
    float weight = (float3weight(path->contri) / path-> sample_pdf);
    if (isnan(weight))path->contri = make_float3(0);
    if (isinf(weight))path->contri = make_float3(0);
//    if (path->is_caustic == false) path->valid = false;// *= 100;
    //printf("pretrace path info%f %f %f\n", float3weight(path->contri), path->fix_pdf, weight);

    if (Tracer::params.pre_tracer.PG_mode)
    {
        float accm_pdf = 1;
        float3 accm_flux = light_contri;
        for (int i = 0; i < path->end_ind; i++)
        {
            BDPTVertex& midVertex = *(eye_end - i);
            float3 nextPos = i == 0 ? light.position : (eye_end - i + 1)->position;
            float3 lastPos = (eye_end - i - 1)->position;
            MaterialData::Pbr mat = VERTEX_MAT(midVertex);
            float3 in_dir = normalize(lastPos - midVertex.position);
            float3 out_dir = normalize(nextPos - midVertex.position); 
            accm_pdf *= Tracer::Pdf(mat, midVertex.normal, in_dir, out_dir, midVertex.position, true);
            float3 f = Tracer::Eval(mat, midVertex.normal, in_dir, out_dir) * abs(dot(out_dir, midVertex.normal));
            float r = Tracer::rrRate(mat);

            conn[i].set_PG_weight(float3weight(accm_flux / accm_pdf) * abs(dot(midVertex.normal, out_dir)));
            accm_flux *= f / r;
        }
    }
}
RT_FUNCTION bool rr_acc_accept(int acc_num, unsigned int& seed)
{
    float r = rnd(seed); 
    if (1.0f / (acc_num + 1) > r)
    {
        return true;
    }
    return false;
}
#define PRETRACER_PADDING_VERTICES_CHECK(a) if (a >= pretracer_params.padding)break;

extern "C" __global__ void __raygen__TrainData()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const PreTraceParams& pretracer_params = Tracer::params.pre_tracer;
    const int    subframe_index = pretracer_params.iteration;
    unsigned int seed = tea<16>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    const float3 eye = Tracer::params.eye;
    const float3 U = Tracer::params.U;
    const float3 V = Tracer::params.V;
    const float3 W = Tracer::params.W; 

    float3 normalizeV = normalize(V);

    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

    const float2 d = 2.0f * make_float2(subpixel_jitter.x, subpixel_jitter.y) - 1.0f;

    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;

    BDPTVertex buffer[PRETRACE_CONN_PADDING];
    int buffer_size = 0;

    int resample_number = 0;
     
    Tracer::PayloadBDPTVertex payload;
    payload.clear();
    payload.seed = seed;
    init_EyeSubpath(payload.path, ray_origin, ray_direction);

    int launch_index = launch_idx.x;
    unsigned int bufferBias = launch_index * pretracer_params.padding;
    preTracePath* currentPath = pretracer_params.paths + launch_index;
    preTraceConnection* currentConn = pretracer_params.conns + bufferBias;
    currentPath->valid = false;
    buffer[buffer_size] = payload.path.currentVertex();
    buffer_size++;

    while (true)
    { 
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload); 
        if (payload.path.size == begin_depth)
        {
            break;
        } 
        if (payload.path.hit_lightSource())
        { 
            if (payload.path.currentVertex().type != BDPTVertex::Type::ENV_MISS && payload.path.size > 2 && rr_acc_accept(resample_number, payload.seed))
            {
                Tracer::lightSample light_sample;
                int light_id = payload.path.currentVertex().materialId;
                light_sample.ReverseSample(Tracer::params.lights[light_id], payload.path.currentVertex().uv);

                BDPTVertex light_vertex;
                init_vertex_from_lightSample(light_sample, light_vertex);
                PreTrace_buildPathInfo(buffer + buffer_size - 1, TrainData::nVertex_device(light_vertex ,false), currentPath, currentConn, buffer_size);
                resample_number++;
            }  
            break;
        } 
        buffer[buffer_size] = payload.path.currentVertex();
        buffer_size++;

        BDPTVertex& eye_subpath = payload.path.currentVertex();
        Tracer::lightSample light_sample;
        light_sample(payload.seed);
        float3 vis_vec = light_sample.position - eye_subpath.position;
        BDPTVertex light_vertex;
        init_vertex_from_lightSample(light_sample, light_vertex);
        if (Tracer::visibilityTest(Tracer::params.handle, eye_subpath, light_vertex)
            && rr_acc_accept(resample_number, payload.seed) && Tracer::params.pre_tracer.PG_mode == false)
        {
            if ((light_vertex.is_DIRECTION() && dot(light_vertex.normal, eye_subpath.normal) < 0) ||
                (!light_vertex.is_DIRECTION() && dot(vis_vec, light_sample.normal()) < 0))
            {
                PreTrace_buildPathInfo(buffer + buffer_size - 1, TrainData::nVertex_device(light_vertex, false), currentPath, currentConn, buffer_size); 
                resample_number++; 
            }
             
        }
         
        if (payload.done || payload.depth > 50)
        {
            break;
        }
        PRETRACER_PADDING_VERTICES_CHECK(buffer_size);
        ray_direction = payload.ray_direction;
        ray_origin = payload.origin;
        payload.depth += 1;
    }

    int beginIndex = 0;
    if (currentPath->valid)
    {
        beginIndex += currentPath->end_ind - currentPath->begin_ind;
    }
    for (int i = beginIndex; i < pretracer_params.padding; i++)
    {
        currentConn[i].valid = false;        
    }
     
    currentPath->sample_pdf/= resample_number;
    currentPath->begin_ind += bufferBias;
    currentPath->end_ind += bufferBias; 
    currentPath->pixel_id = make_int2(Tracer::params.width * subpixel_jitter.x, Tracer::params.height * subpixel_jitter.y);
    if (currentPath->begin_ind == currentPath->end_ind && currentPath->valid == true)
    {
        currentPath->valid = false;
    }
}
