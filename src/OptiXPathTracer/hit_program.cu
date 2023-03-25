
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

#define PT_BRDF_STRATEGY_ONLY
//#define PT_NEE_STRATEGY_ONLY

extern "C" __global__ void __anyhit__radiance()
{
    //optixIgnoreIntersection();
    return;
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    if (hit_group_data->material_data.pbr.base_color_tex)
    {
        const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);
        const float         base_alpha = sampleTexture<float4>(hit_group_data->material_data.pbr.base_color_tex, geom).w;
        // force mask mode, even for blend mode, as we don't do recursive traversal.
        if (base_alpha < hit_group_data->material_data.alpha_cutoff)
            optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__occlusion()
{
    Tracer::setPayloadOcclusion(0.f);
    return;
    optixTerminateRay();
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    if (hit_group_data->material_data.pbr.base_color_tex)
    {
        const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);
        const float         base_alpha = sampleTexture<float4>(hit_group_data->material_data.pbr.base_color_tex, geom).w;

        if (hit_group_data->material_data.alpha_mode != MaterialData::ALPHA_MODE_OPAQUE)
        {
            if (hit_group_data->material_data.alpha_mode == MaterialData::ALPHA_MODE_MASK)
            {
                if (base_alpha < hit_group_data->material_data.alpha_cutoff)
                    optixIgnoreIntersection();
            }

            float attenuation = Tracer::getPayloadOcclusion() * (1.f - base_alpha);

            if (attenuation > 0.f)
            {
                Tracer::setPayloadOcclusion(attenuation);
                optixIgnoreIntersection();
            }
        }
    }
}

extern "C" __global__ void __closesthit__occlusion()
{
    Tracer::setPayloadOcclusion(0.f);
}
extern "C" __global__ void __closesthit__eyeSubpath_LightSource()
{
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    prd->done = true;
    const Light& light = Tracer::params.lights[hit_group_data->material_data.light_id];
    if (dot(prd->ray_direction, light.quad.normal) > 0)
    { 
        return;
    }

    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    float t_hit = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 inver_ray_direction = -ray_direction;  
 

    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex();// prd.stackP->v[(prd.stackP->size) % STACKSIZE];
    BDPTVertex& LastVertex = prd->path.lastVertex();// prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
    
    MidVertex.position = geom.P;
    MidVertex.normal = light.quad.normal; 
    MidVertex.type = BDPTVertex::Type::HIT_LIGHT_SOURCE;
    MidVertex.uv = geom.texcoord->UV;
    Tracer::lightSample light_sample;
    light_sample.ReverseSample(light, MidVertex.uv);
    float lightPdf = light_sample.pdf;
     
    
    float pdf_G = abs(dot(MidVertex.normal, ray_direction) * dot(LastVertex.normal, ray_direction)) / (t_hit * t_hit);
    if (LastVertex.isOrigin)
    {
        MidVertex.flux = LastVertex.flux * pdf_G * light_sample.emission;
    }
    else
    {
        MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G * light_sample.emission;
    }


    MidVertex.lastPosition = LastVertex.position;
    MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, ray_direction));

    //MidVertex.zoneId = SUBSPACE_NUM - lightMaterialId - 1;
    MidVertex.subspaceId = light_sample.subspaceId;  
    MidVertex.lastZoneId = LastVertex.subspaceId;


    MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal, ray_direction));
    MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

    //MidVertex.dLast = LastVertex.d;
    MidVertex.materialId = hit_group_data->material_data.light_id;

    MidVertex.depth = LastVertex.depth + 1;

    if (MidVertex.depth == 1)
    {
        MidVertex.RMIS_pointer = 1.0;         
        return;
    }

    BDPTVertex virtual_light;
    virtual_light.position = MidVertex.position;
    virtual_light.RMIS_pointer = 1;
    virtual_light.normal = MidVertex.normal;
    virtual_light.pdf = lightPdf;
    virtual_light.singlePdf = lightPdf;
    virtual_light.flux = light_sample.emission;
    virtual_light.subspaceId = MidVertex.subspaceId;
    virtual_light.isBrdf = false;
    //rtPrintf("%f %f\n", 1.0 / MidVertex.d, light_hit(LastVertex, virtual_light)); 
    MidVertex.RMIS_pointer = 1.0 / rmis::light_hit(LastVertex, virtual_light);   
}
extern "C" __global__ void __closesthit__lightsource()
{
    Tracer::PayloadRadiance* prd = Tracer::getPRD();

    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    /* 打到了哪个光源? */
    const Light& light = Tracer::params.lights[hit_group_data->material_data.light_id];
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    /* 算一算打到点的采样 pdf（用于MIS） */
    Tracer::lightSample light_sample;
    light_sample.ReverseSample(light, geom.texcoord->UV);
    float t_hit = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();

    if ( /* 打中的光源法向要求与光线方向相反 */
        (dot(prd->ray_direction, light_sample.normal()) <= 0 ) && (
        /* 光源直击, L - E */
            (LE_ENABLE && prd->depth == 0) || (
            /* 是否是 S_ONLY */
                (!S_ONLY || prd->path_record) && (
                /* L - * - E，所有光路 */
                    LAE_ENABLE ||
                /* L - D - E */
                    (LDE_ENABLE && prd->depth == 1 && prd->path_record == 0b0) || 
                /* L - D - S - E */
                    (LDSE_ENABLE && prd->depth == 2 && prd->path_record == 0b01) ||
                /* L - D - S - D - E */
                    (LDSDE_ENABLE && prd->depth == 3 && prd->path_record == 0b010) ||
                /* L - S - * - E */
                    (LSAE_ENABLE && prd->depth > 0 && (prd->path_record & (1ll << (prd->depth-1)))) ||
                /* L - S - E */
                    (LSE_ENABLE && prd->depth == 1 && prd->path_record == 0b1) ||
                /* L - S - D - E */
                    (LSDE_ENABLE && prd->depth == 2 && prd->path_record == 0b10) 
                )
            )
        )
    )
    {
        /* PT 加 NEE 的 MIS */
        float MIS_weight = 1;
        if (prd->depth != 0 )
        {
            float pdf_hit = prd->pdf * abs(dot(ray_direction,light_sample.normal())) / (t_hit * t_hit);

            float pdf_area = light_sample.pdf;
            MIS_weight = pdf_hit / (pdf_area + pdf_hit);
        }
#ifdef PT_BRDF_STRATEGY_ONLY 
        MIS_weight = 1;
#endif 
#ifdef PT_NEE_STRATEGY_ONLY 
        MIS_weight = 0;
#endif 
        prd->result += prd->throughput * light_sample.emission * MIS_weight;
    }
    prd->done = true;
    return;
}

RT_FUNCTION void ColorTexSample(const LocalGeometry& geom, MaterialData::Pbr& pbr)
{
    float4 base_color = pbr.base_color;// *geom.color;
    if (pbr.base_color_tex)
    {
        const float4 base_color_tex = sampleTexture<float4>(pbr.base_color_tex, geom);

        // don't gamma correct the alpha channel.
        const float3 base_color_tex_linear = Tracer::linearize(make_float3(base_color_tex));
        //const float3 base_color_tex_linear = make_float3(base_color_tex);

        base_color = make_float4(base_color_tex_linear.x, base_color_tex_linear.y, base_color_tex_linear.z, base_color_tex.w);
    }
    pbr.base_color = base_color;

    return;
}
RT_FUNCTION void RoughnessAndMetallicTexSample(const LocalGeometry& geom, MaterialData::Pbr& pbr)
{
    //float  metallic  = hit_group_data->material_data.pbr.metallic;
    //float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4(1.0f);
    if (pbr.metallic_roughness_tex) 
        mr_tex = sampleTexture<float4>(pbr.metallic_roughness_tex, geom);
    pbr.roughness *= mr_tex.y;
    pbr.metallic *= mr_tex.z;
    return;
}
RT_FUNCTION float3 NormalTexSample(const LocalGeometry& geom, const MaterialData& matData)
{
    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if (matData.normal_tex)
    {
        const int texcoord_idx = matData.normal_tex.texcoord;
        const float4 NN =
            2.0f * sampleTexture<float4>(matData.normal_tex, geom) - make_float4(1.0f);

        // Transform normal from texture space to rotated UV space.
        const float2 rotation = matData.normal_tex.texcoord_rotation;
        const float2 NN_proj = make_float2(NN.x, NN.y);
        const float3 NN_trns = make_float3(
            dot(NN_proj, make_float2(rotation.y, -rotation.x)),
            dot(NN_proj, make_float2(rotation.x, rotation.y)),
            NN.z);

        N = normalize(NN_trns.x * normalize(geom.texcoord[texcoord_idx].dpdu) + NN_trns.y * normalize(geom.texcoord[texcoord_idx].dpdv) + NN_trns.z * geom.N);
    }

    // Flip normal to the side of the incomming ray
    if (dot(N, optixGetWorldRayDirection()) > 0.f)
        N = -N;
    return N;
}
extern "C" __global__ void __closesthit__lightSource_subpath()
{
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    prd->done = true;

}

extern "C" __global__ void __closesthit__eyeSubpath()
{
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    float t_hit = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 inver_ray_direction = -ray_direction;
    MaterialData::Pbr currentPbr = hit_group_data->material_data.pbr;
    ColorTexSample(geom, currentPbr);
    RoughnessAndMetallicTexSample(geom, currentPbr);
    float3 N = geom.N;// NormalTexSample(geom, hit_group_data->material_data);
//    if (dot(N, ray_direction) > 0.f)
//        N = -N;
    prd->ray_direction = Tracer::Sample(currentPbr, N, inver_ray_direction, prd->seed);
    prd->pdf = Tracer::Pdf(currentPbr, N, inver_ray_direction, prd->ray_direction);
    prd->origin = geom.P;
    if (!(prd->pdf > 0.0f))
        prd->done = true;

    //    prd->path.size += 1;
    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex();
    BDPTVertex& NextVertex = prd->path.nextVertex();
    BDPTVertex& LastVertex = prd->path.lastVertex();
    MidVertex.position = geom.P;
    MidVertex.normal = N;//这个在折射场景里需要进一步讨论
    MidVertex.type = BDPTVertex::Type::NORMALHIT;
    float pdf_G = abs(dot(MidVertex.normal, ray_direction) * dot(LastVertex.normal, ray_direction)) / (t_hit * t_hit);
 
    if (LastVertex.isOrigin)
    {
        MidVertex.flux = LastVertex.flux * pdf_G;
    }
    else
    {
        MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G;
    }
    NextVertex.flux = Tracer::Eval(currentPbr, N, inver_ray_direction, prd->ray_direction) / (currentPbr.brdf ? abs(dot(MidVertex.normal, prd->ray_direction)) : 1.0f);
    NextVertex.singlePdf = prd->pdf;

    MidVertex.lastPosition = LastVertex.position;
    if (LastVertex.is_DIRECTION())
    {
        MidVertex.lastPosition = MidVertex.position - ray_direction;
    }

    MidVertex.color = make_float3(currentPbr.base_color);
    MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, ray_direction));
    MidVertex.materialId = hit_group_data->material_data.id;

    labelUnit lu(MidVertex.position, MidVertex.normal, -ray_direction, false);
    MidVertex.subspaceId = lu.getLabel();
    MidVertex.lastZoneId = LastVertex.subspaceId;
    MidVertex.lastBrdf = LastVertex.isBrdf;
    MidVertex.isOrigin = false;
    MidVertex.depth = LastVertex.depth + 1;
    MidVertex.uv = geom.texcoord[0].UV;

    MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal, ray_direction));
    MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;
     
    //MidVertex.last_lum = Tracer::float3sum(LastVertex.flux / LastVertex.pdf);

    {
        MidVertex.lastSinglePdf = LastVertex.singlePdf;
        MidVertex.isLastVertex_direction = LastVertex.depth == 0 && (LastVertex.is_DIRECTION());
        if (MidVertex.depth == 1)
        {
            rmis::tracing_init_eye(MidVertex, LastVertex);
        }
        else
        {
            rmis::tracing_update_eye(MidVertex, LastVertex);
        }

        float r = rnd(prd->seed);

        float rr_rate = Tracer::rrRate(currentPbr);
        if (r > rr_rate)
        {
            prd->done = true;
        }
        else
        {
            NextVertex.singlePdf *= rr_rate;
            prd->throughput *= NextVertex.flux / prd->pdf / rr_rate * dot(N, prd->ray_direction);
        }
        return;
    }
}


extern "C" __global__ void __closesthit__eyeSubpath_LightSource_simple()
{
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    prd->done = true;
    const Light& light = Tracer::params.lights[hit_group_data->material_data.light_id];
    if (dot(prd->ray_direction, light.quad.normal) > 0)
    {
        return;
    }


    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);

    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex();// prd.stackP->v[(prd.stackP->size) % STACKSIZE];
    BDPTVertex& LastVertex = prd->path.lastVertex();// prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];

    MidVertex.position = geom.P;
    MidVertex.normal = light.quad.normal;
    MidVertex.type = BDPTVertex::Type::HIT_LIGHT_SOURCE;
    MidVertex.uv = geom.texcoord->UV;


    MidVertex.materialId = hit_group_data->material_data.light_id;

    MidVertex.depth = LastVertex.depth + 1;

}
extern "C" __global__ void __closesthit__eyeSubpath_simple()
{ 
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    float t_hit = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 inver_ray_direction = -ray_direction;
    MaterialData::Pbr currentPbr = hit_group_data->material_data.pbr;
    ColorTexSample(geom, currentPbr);
    RoughnessAndMetallicTexSample(geom, currentPbr);
    float3 N = geom.N;// NormalTexSample(geom, hit_group_data->material_data);
//    if (dot(N, ray_direction) > 0.f)
//        N = -N;
    //prd->ray_direction = Tracer::Sample(currentPbr, N, inver_ray_direction, prd->seed); 
    prd->origin = geom.P;
    if (!(prd->pdf > 0.0f))
        prd->done = true;

    //    prd->path.size += 1;
    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex();
    BDPTVertex& NextVertex = prd->path.nextVertex();
    BDPTVertex& LastVertex = prd->path.lastVertex();
    MidVertex.position = geom.P;
    MidVertex.normal = N;//这个在折射场景里需要进一步讨论
    MidVertex.type = BDPTVertex::Type::NORMALHIT; 
    MidVertex.color = make_float3(currentPbr.base_color); 

    MidVertex.materialId = hit_group_data->material_data.id;
      
    MidVertex.depth = LastVertex.depth + 1;
    MidVertex.uv = geom.texcoord[0].UV;
     
    //MidVertex.last_lum = Tracer::float3sum(LastVertex.flux / LastVertex.pdf);
     
}
extern "C" __global__ void __closesthit__lightSubpath()
{
    // printf("lightSubpath\n");
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    Tracer::PayloadBDPTVertex* prd = Tracer::getPRD<Tracer::PayloadBDPTVertex>();
    float t_hit = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 inver_ray_direction = -ray_direction;
    MaterialData::Pbr currentPbr = hit_group_data->material_data.pbr;
    ColorTexSample(geom, currentPbr);
    RoughnessAndMetallicTexSample(geom, currentPbr);
    float3 N = geom.N;
    // NormalTexSample(geom, hit_group_data->material_data);
    // if (dot(N, ray_direction) > 0.f)
    // N = -N;
    prd->ray_direction = Tracer::Sample(currentPbr, N, inver_ray_direction, prd->seed); 
    prd->pdf           = Tracer::Pdf(currentPbr, N, inver_ray_direction, prd->ray_direction);
    prd->origin        = geom.P;
    if (!(prd->pdf > 0.0f))
        prd->done = true;
    
//    prd->path.size += 1;
    prd->path.push();
    BDPTVertex& MidVertex = prd->path.currentVertex();
    BDPTVertex& NextVertex = prd->path.nextVertex();
    BDPTVertex& LastVertex = prd->path.lastVertex();
    MidVertex.position = geom.P;
    MidVertex.normal = N;//这个在折射场景里需要进一步讨论
    MidVertex.type = BDPTVertex::Type::NORMALHIT;
    float pdf_G = abs(dot(MidVertex.normal, ray_direction) * dot(LastVertex.normal, ray_direction)) / (t_hit * t_hit);

    if (LastVertex.is_DIRECTION())
        pdf_G = abs(dot(MidVertex.normal, ray_direction) * dot(LastVertex.normal, ray_direction));
    if (LastVertex.isOrigin)
        MidVertex.flux = LastVertex.flux * pdf_G;
    else
        MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G;

    NextVertex.flux = Tracer::Eval(currentPbr, N, prd->ray_direction, -ray_direction) / (currentPbr.brdf ? abs(dot(MidVertex.normal, prd->ray_direction)) : 1.0f);

    NextVertex.singlePdf = prd->pdf;
     
    MidVertex.lastPosition = LastVertex.position;
    if (LastVertex.is_DIRECTION())
        MidVertex.lastPosition = MidVertex.position - ray_direction;

    MidVertex.color = make_float3(currentPbr.base_color);
    MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, ray_direction));
    MidVertex.materialId = hit_group_data->material_data.id;

    labelUnit lu(MidVertex.position, MidVertex.normal, -ray_direction, true);
    MidVertex.subspaceId = lu.getLabel();
    MidVertex.lastZoneId = LastVertex.subspaceId;
    MidVertex.lastBrdf = LastVertex.isBrdf;
    MidVertex.isOrigin = false;
    MidVertex.depth = LastVertex.depth + 1;
    MidVertex.uv = geom.texcoord[0].UV;

    MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal, ray_direction));
    MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

    MidVertex.last_lum = Tracer::float3sum(LastVertex.flux / LastVertex.pdf);

    MidVertex.lastSinglePdf = LastVertex.singlePdf;
    MidVertex.isLastVertex_direction = LastVertex.depth == 0 && (LastVertex.is_DIRECTION());
    if (LastVertex.isOrigin)
        rmis::tracing_init_light(MidVertex, LastVertex);
    else
        rmis::tracing_update_light(MidVertex, LastVertex);

    float r = rnd(prd->seed);
    float rr_rate = Tracer::rrRate(currentPbr);
    if (r > rr_rate)
        prd->done = true;
    else
        NextVertex.singlePdf *= rr_rate;
    return;
}
/* 这个函数应该是 PT 在打到普通面片时被调用 */
extern "C" __global__ void __closesthit__radiance() 
{
    // printf("__closesthit__radiance()\n");
    const Tracer::HitGroupData* hit_group_data = reinterpret_cast<Tracer::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );
    Tracer::PayloadRadiance* prd = Tracer::getPRD();

    /* Retrieve material data */
    MaterialData::Pbr currentPbr = hit_group_data->material_data.pbr;
    ColorTexSample(geom, currentPbr);
    RoughnessAndMetallicTexSample(geom, currentPbr);
    float3 N = geom.N;
    float3 in_dir = -prd->ray_direction;
    float3 result = make_float3( 0.0f );

    float rr_rate = Tracer::rrRate(currentPbr);
    prd->glossy_bounce = Shift::glossy(currentPbr) ? prd->glossy_bounce : false;


    /*  prd->path_record 用二进制按LSB到MSB的顺序编码了当前路径，0 代表 D, 1 代表 S */
    /* 比如 path_record 为 0010，depth 为 4，说明当前路径为 D - D - S - D - E */
    /* path_record 大小为 long long 以保证够用 */
    prd->path_record = (prd->path_record) | 
        ((long long) Shift::glossy(currentPbr) << prd->depth);
 
    /* 计算 NEE */
    int light_id = clamp(static_cast<int>(floorf(rnd(prd->seed) * Tracer::params.lights.count)), int(0), int(Tracer::params.lights.count - 1));
    Light light = Tracer::params.lights[light_id];
    if (light.type == Light::Type::QUAD)
    {
        Tracer::lightSample light_sample;
        light_sample(light, prd->seed);

        // TODO: optimize
        const float  L_dist = length(light_sample.position - geom.P);
        const float3 L = (light_sample.position - geom.P) / L_dist;
        const float3 V = -normalize(optixGetWorldRayDirection());
        const float3 H = normalize(L + V);
        const float3 LN = light.quad.normal;
        const float  L_dot_LN = dot(-L, LN);
        const float  N_dot_L = abs(dot(N, L));
        const float  N_dot_V = abs(dot(N, V));
        const float  N_dot_H  = dot(N, H);
        const float  V_dot_H  = dot(V, H);
        if (N_dot_L > 0.0f && N_dot_V > 0.0f && L_dot_LN > 0.0f)
        {
            const float tmin = 0.001f;           // TODO
            const float tmax = L_dist - 0.001f;  // TODO
            const float attenuation = 1;//// Tracer::traceOcclusion(Tracer::params.handle, geom.P, L, tmin, tmax);
            if (attenuation > 0.f)
            {
                prd->vis_pos_A = geom.P;
                prd->vis_pos_B = light_sample.position;
                float3 eval = Tracer::Eval(currentPbr, N, L, V);

                float MIS_weight = 1;
                {
                    float pdf_area = light_sample.pdf;
                    float pdf_hit = Tracer::Pdf(currentPbr, N, V, L, geom.P, true) * abs(L_dot_LN) / (L_dist * L_dist) * rr_rate;
                    MIS_weight = pdf_area / (pdf_hit + pdf_area);
                }
#ifdef PT_BRDF_STRATEGY_ONLY 
                MIS_weight = 0;
#endif // PT_BRDF_STRATEGY_ONLY 

#ifdef PT_NEE_STRATEGY_ONLY 
                MIS_weight = 1;
#endif // PT_BRDF_STRATEGY_ONLY 
                result += prd->throughput * light_sample.emission * attenuation / light_sample.pdf 
                    * N_dot_L * L_dot_LN / L_dist / L_dist * eval * MIS_weight ;// *make_float3(1.0, 0.0, 1.0);
            }
        }
    }
    else if (light.type == Light::Type::ENV)
    {
        Tracer::lightSample light_sample;
        light_sample(light, prd->seed);

        const float3 V = -normalize(optixGetWorldRayDirection());
        const float3 L = light_sample.direction;
        float L_dot_N = dot(light_sample.direction, N);
        if ( L_dot_N > 0.0)
        {
            prd->vis_pos_A = geom.P;
            prd->vis_pos_B = geom.P + light_sample.direction * SKY.r * 10;
//            printf("light_sample dir %f %f %f\n", light_sample.direction.x, light_sample.direction.y, light_sample.direction.z);
            float3 eval = Tracer::Eval(currentPbr, N, L, V);
            result += prd->throughput * light_sample.emission / light_sample.pdf * eval * L_dot_N;
        }
    }

    //const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);
    //result = make_float3(geom.texcoord[1].UV.x, geom.texcoord[0].UV.y ,0.0) + make_float3(1.0,1.0,0);
    //result = make_float3(currentPbr.base_color);  
     
    //prd->done = true;
    //prd->depth += 1;  
    //prd->result += result;


    //if (prd->depth > 5) result *= 0;

    prd->currentResult += result;
    prd->origin = geom.P;

    if (rnd(prd->seed) > rr_rate)
    {
        prd->done = true;
    }
    else
    {
        prd->ray_direction = Tracer::Sample(currentPbr, N, in_dir, prd->seed, geom.P, true); 
        float pdf = Tracer::Pdf(currentPbr, N, in_dir, prd->ray_direction, geom.P, true);

        if (isRefract(N, in_dir, prd->ray_direction) && prd->depth == 1 && dot(in_dir, N) > 0 && Shift::glossy(currentPbr))
        {
            float3 bsdf = Tracer::Eval(currentPbr, N, in_dir, prd->ray_direction);
            float cos_in = abs(dot(in_dir, N));
            float cos_out = abs(dot(prd->ray_direction, N));
            float sin_in = sqrt(1 - cos_in * cos_in);
            float sin_out = sqrt(1 - cos_out * cos_out);
        }
        if (pdf > 0.0f)
        {
            prd->throughput *= Tracer::Eval(currentPbr, N, in_dir, prd->ray_direction) * abs(dot(prd->ray_direction, N)) / pdf / rr_rate;
            prd->pdf = pdf * rr_rate;
        }
        else
        {
            prd->done = true;
        }
    }
     
    //Tracer::setPayloadResult( result );
}
