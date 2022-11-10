#ifndef RMIS_H
#define RMIS_H
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "light_parameters.h"
#include "state.h"
#include "BDPT.h"
#include "ZGC_device.h"
#include"classTree_device.h"
using namespace optix;

RT_FUNCTION MaterialParameter& getMat(BDPTVertex& Vertex)
{
    MaterialParameter mat = sysMaterialParameters[Vertex.materialId];
    mat.color = Vertex.color;
    return mat;
}
RT_FUNCTION float float3sum(float3 a)
{
    return a.x + a.y + a.z;
}
RT_FUNCTION float getRR(BDPTVertex& vertex)
{
    float rr_rate = fmaxf(vertex.color);


#ifdef RR_MIN_LIMIT
    rr_rate = max(rr_rate, MIN_RR_RATE);
#endif
#ifdef RR_DISABLE
    rr_rate = 1.0f;
#endif
    return rr_rate;
}

RT_FUNCTION float3 getFluxMultiplier(BDPTVertex& vertex,float3 in_dir, float3 out_dir)
{
    MaterialParameter mat = getMat(vertex); 
    float3 flux_ratio = DisneyEval(mat, vertex.normal, in_dir, out_dir) / (mat.brdf ? abs(dot(vertex.normal, out_dir)) : 1.0f);
    float pdf_ratio = DisneyPdf(mat, vertex.normal, in_dir, out_dir, vertex.position, false);
    float rr = getRR(vertex);
    float cos_theta = abs(dot(vertex.normal, out_dir));

    //rtPrintf("%f %f %f %f %d\n", float3sum(flux_ratio) , in_dir.x , out_dir.x , rr,vertex.depth);
    return flux_ratio * cos_theta / pdf_ratio / rr;
}
RT_FUNCTION float3 getFluxMultiplier(BDPTVertex& vertex, float3 in_dir)
{ 
    float3 out_vec = vertex.lastPosition - vertex.position;
    float3 out_dir = normalize(out_vec); 
    return getFluxMultiplier(vertex, in_dir, out_dir);
}
//necessary attribute for tracing light:
//position
//zoneId
//lastZoneId
//materialId
//singlePdf
//color
//normal
//lastNormalProjection
//brdf
//lastBrdf
RT_FUNCTION float tracing_weight_light(BDPTVertex& MidVertex, BDPTVertex& LastVertex)
{

    //if ((LastVertex.inBrdf) != (LastVertex.lastBrdf || LastVertex.isBrdf))
    //{
    //    printf("get error %d %d %d %d\n",LastVertex.depth,LastVertex.inBrdf,LastVertex.isBrdf,LastVertex.lastBrdf);
    //}
    if (LastVertex.lastBrdf || LastVertex.isBrdf)
    {
        return 0.0;
    }
    float3 inver_dir = normalize(MidVertex.position - LastVertex.position);
    
    int eye_label = LastVertex.zoneId;//GetLabel(LastVertex.position,inver_dir)

    labelUnit label_unit(LastVertex.position, LastVertex.normal, inver_dir, false);
    eye_label = label_unit.getLabel();// classTree::getEyeLabel(LastVertex.position);
    int light_label = LastVertex.lastZoneId;

    float lum_sum = LastVertex.last_lum;
    return  connectRate_SOL(eye_label, light_label, lum_sum);
}

RT_FUNCTION float3 tracing_weight_eye(BDPTVertex& MidVertex, BDPTVertex& LastVertex)
{
    if (LastVertex.lastBrdf || LastVertex.isBrdf)
    {
        return make_float3(0.0);
    }
    if (LastVertex.depth == 1)//no light tracing strategy
    {
        return make_float3(0.0);
    }
    float3 inver_dir = MidVertex.is_DIRECTION() ? -MidVertex.normal : normalize(MidVertex.position - LastVertex.position);

    int eye_label = LastVertex.lastZoneId;

    int light_label = LastVertex.zoneId;//GetLabel(LastVertex.position,inver_dir)

    labelUnit label_unit(LastVertex.position, LastVertex.normal, inver_dir, true);
    light_label = label_unit.getLabel();// classTree::getLightLabel(LastVertex.position);
    float3 lum = make_float3(1.0); 
    return  connectRate_SOL(eye_label, light_label, lum);
}

RT_FUNCTION float getLast_pdf(BDPTVertex& MidVertex, float3 in_dir, bool eye_side = true)
{
    MaterialParameter mat = getMat(MidVertex); 
    float3 out_vec = MidVertex.lastPosition - MidVertex.position;
    float3 out_dir = normalize(out_vec);
    float pdf = MidVertex.is_LL_DIRECTION()?
        DisneyPdf(mat, MidVertex.normal, in_dir, out_dir, MidVertex.position, eye_side) :
        DisneyPdf(mat, MidVertex.normal, in_dir, out_dir, MidVertex.position, eye_side) / dot(out_vec, out_vec) * MidVertex.lastNormalProjection;
    pdf *= getRR(MidVertex);
    return pdf;
}
RT_FUNCTION float getPdf(BDPTVertex& beginVertex, BDPTVertex& endVertex, float3 in_dir, bool eye_side = true)//pdf compute:generate endVertex from beginVertex
{
    MaterialParameter mat = getMat(beginVertex);
    float pdf;
    if (endVertex.is_DIRECTION())
    {
        float3 out_dir = -endVertex.normal;
        pdf = DisneyPdf(mat, beginVertex.normal, in_dir, out_dir, beginVertex.position, eye_side);
    }
    else
    {
        float3 out_vec = endVertex.position - beginVertex.position;
        float3 out_dir = normalize(out_vec);
        pdf = DisneyPdf(mat, beginVertex.normal, in_dir, out_dir, beginVertex.position, eye_side) / dot(out_vec, out_vec) * abs(dot(out_dir, endVertex.normal));
    }  
    pdf *= getRR(beginVertex);

    return pdf;
}
RT_FUNCTION float getPdf_from_light_source(BDPTVertex& lightVertex, BDPTVertex& endVertex)
{
    if (!lightVertex.is_DIRECTION())
    {
        float3 conn_vec = endVertex.position - lightVertex.position;
        float3 conn_dir = normalize(conn_vec);
        float pdf_angle = abs(dot(lightVertex.normal,conn_dir)) / M_PI ;
        float angle2a = abs(dot(endVertex.normal, conn_dir)) / (dot(conn_vec, conn_vec));
        return pdf_angle * angle2a;
    }
    else
    {
        float3 dir = lightVertex.normal;
        return sky.projectPdf() * abs(dot(dir, endVertex.normal));
    }
}
RT_FUNCTION float getLL_pdf(BDPTVertex& MidVertex, BDPTVertex& LastVertex, bool eye_side = true)
{
    float3 in_dir = normalize(MidVertex.position - LastVertex.position);

    return getLast_pdf(LastVertex, in_dir, eye_side);
}
RT_FUNCTION void tracing_update_light(BDPTVertex& MidVertex, BDPTVertex& LastVertex)//assumption:lastvertex.depth>0
{
    //what we need to compute
    float LL_pdf;//pdf for midvertex->lastvertex->lastlastvertex
    float weight;//mis weight for current strategy
    float last_single_pdf;//MidVertex pdf-----lastlastvertex -> lastvertex

    LL_pdf = getLL_pdf(MidVertex, LastVertex);

    weight = tracing_weight_light(MidVertex, LastVertex);

    last_single_pdf = LastVertex.singlePdf;

    MidVertex.RMIS_pointer = ((LastVertex.RMIS_pointer * LL_pdf) + weight) / last_single_pdf;

}

RT_FUNCTION void tracing_init_light(BDPTVertex& MidVertex, BDPTVertex& LastVertex)//assumption:lastVertex is the light source
{
//    LastVertex.RMIS_pointer = LastVertex.d;
    MidVertex.RMIS_pointer = LastVertex.RMIS_pointer / LastVertex.singlePdf;
}
RT_FUNCTION void tracing_update_eye(BDPTVertex& MidVertex, BDPTVertex& LastVertex)//assumption.lastVertex.depth>0
{
    //what we need to compute
    float LL_pdf;//pdf for midvertex->lastvertex->lastlastvertex
    float3 weight;//mis weight for current strategy
    float last_single_pdf;//MidVertex pdf-----lastlastvertex -> lastvertex
    float3 flux_multiplier;
    LL_pdf = getLL_pdf(MidVertex, LastVertex,false);
    weight = tracing_weight_eye(MidVertex, LastVertex);
    last_single_pdf = LastVertex.singlePdf;
    flux_multiplier = getFluxMultiplier(LastVertex, normalize(MidVertex.position - LastVertex.position));

    MidVertex.RMIS_pointer_3 = ((LastVertex.RMIS_pointer_3 * LL_pdf * flux_multiplier) + weight) / last_single_pdf;

}
RT_FUNCTION void tracing_init_eye(BDPTVertex& MidVertex, BDPTVertex& LastVertex)//lastVertes is the camera
{
    MidVertex.RMIS_pointer_3 = make_float3(0.0);
}
//////////////////////////////////////////////////////////////////////////////////
////////////////////////eval mis weight///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

RT_FUNCTION float general_connection(BDPTVertex& eyeVertex, BDPTVertex& lightVertex)
{
    if (eyeVertex.isBrdf || lightVertex.isBrdf)
    {
        return 0.0;
    }
    float3 connect_vec = eyeVertex.position - lightVertex.position;
    float3 connect_dir = normalize(connect_vec);
    float3 flux = lightVertex.flux / lightVertex.pdf;
    
    float LL_pdf_A = getLL_pdf(lightVertex, eyeVertex,false);
    float3 flux_multiplier_0 = getFluxMultiplier(eyeVertex, -connect_dir);
    float3 weight_A = tracing_weight_eye(lightVertex, eyeVertex);
    float3 D_A_0 = ((eyeVertex.RMIS_pointer_3 * LL_pdf_A * flux_multiplier_0) + weight_A);

    float3 LA = normalize(lightVertex.lastPosition - lightVertex.position);
    float pdf_A = getPdf(lightVertex, eyeVertex, LA,false);
    float3 flux_multiplier_1 = getFluxMultiplier(lightVertex, LA, connect_dir); 
    float D_A = float3sum(D_A_0 * pdf_A * flux_multiplier_1 * flux / eyeVertex.singlePdf);
 

    float weight = float3sum(connectRate_SOL(eyeVertex.zoneId, lightVertex.zoneId, flux));

    ////light side 

    float LL_pdf_B = getLL_pdf(eyeVertex, lightVertex,true);
    float weight_B = tracing_weight_light(eyeVertex, lightVertex);
    float D_B_0 = (lightVertex.RMIS_pointer * LL_pdf_B) + weight_B;

    float3 LB = normalize(eyeVertex.lastPosition - eyeVertex.position);
    float pdf_B = getPdf(eyeVertex, lightVertex, LB, true);
    float D_B = D_B_0 * pdf_B / lightVertex.singlePdf;

    //rtPrintf("%f %f %f %f\n", pdf_A, pdf_B, weight_B,float3sum(LB));
    return weight / (weight + D_A + D_B);
}

RT_FUNCTION float connection_direction_lightSource(BDPTVertex& eyeVertex, BDPTVertex& lightVertex)//only for area light
{
    if (eyeVertex.isBrdf || lightVertex.isBrdf)
    {
        return 0.0;
    } 
    float3 connect_dir = lightVertex.normal;
    float3 flux = lightVertex.flux / lightVertex.pdf;

    float LL_pdf_A = getLL_pdf(lightVertex, eyeVertex, false);
    float3 flux_multiplier_0 = getFluxMultiplier(eyeVertex, -connect_dir);
    float3 weight_A = tracing_weight_eye(lightVertex, eyeVertex);
    float3 D_A_0 = ((eyeVertex.RMIS_pointer_3 * LL_pdf_A * flux_multiplier_0) + weight_A);

    float pdf_A = getPdf_from_light_source(lightVertex, eyeVertex);
    float flux_multiplier_1 = lightVertex.is_DIRECTION() ? 1.0 / sky.projectPdf() : M_PIf;
    float D_A = float3sum(D_A_0 * pdf_A * flux_multiplier_1 * flux / eyeVertex.singlePdf);


    float weight = float3sum(connectRate_SOL(eyeVertex.zoneId, lightVertex.zoneId, flux));

    ////light side 

    float D_B_0 = lightVertex.RMIS_pointer;//= lightVertex.d;

    float3 LB = normalize(eyeVertex.lastPosition - eyeVertex.position);
    float pdf_B = getPdf(eyeVertex, lightVertex, LB, true);
    float D_B = D_B_0 * pdf_B / lightVertex.singlePdf;

    //rtPrintf("%f %f %f %f\n", pdf_A, pdf_B, weight_B,float3sum(LB));
    return weight / (weight + D_A + D_B);
}
RT_FUNCTION float connection_lightSource(BDPTVertex& eyeVertex, BDPTVertex& lightVertex)//only for area light
{ 
    if (eyeVertex.isBrdf || lightVertex.isBrdf)
    {
        return 0.0;
    }
    float3 connect_vec = eyeVertex.position - lightVertex.position;
    float3 connect_dir = normalize(connect_vec);
    float3 flux = lightVertex.flux / lightVertex.pdf;

    float LL_pdf_A = getLL_pdf(lightVertex, eyeVertex, false);
    float3 flux_multiplier_0 = getFluxMultiplier(eyeVertex, -connect_dir);
    float3 weight_A = tracing_weight_eye(lightVertex, eyeVertex);
    float3 D_A_0 = ((eyeVertex.RMIS_pointer_3 * LL_pdf_A * flux_multiplier_0) + weight_A);
     
    float pdf_A = getPdf_from_light_source(lightVertex, eyeVertex);
    float flux_multiplier_1 = lightVertex.is_DIRECTION() ? 1.0 / sky.projectPdf() : M_PIf;
    float D_A = float3sum(D_A_0 * pdf_A * flux_multiplier_1 * flux / eyeVertex.singlePdf);


    float weight = float3sum(connectRate_SOL(eyeVertex.zoneId, lightVertex.zoneId, flux));

    ////light side 
     
    float D_B_0 = lightVertex.RMIS_pointer;// = lightVertex.d;

    float3 LB = normalize(eyeVertex.lastPosition - eyeVertex.position); 
    float pdf_B = getPdf(eyeVertex, lightVertex, LB, true);
    float D_B = D_B_0 * pdf_B / lightVertex.singlePdf;

    //rtPrintf("%f %f %f %f\n", pdf_A, pdf_B, weight_B,float3sum(LB)); 
    return weight / (weight + D_A + D_B);
}
RT_FUNCTION void construct_virtual_env_light(BDPTVertex& lightVertex, float3 flux, float pdf, float3 direction, int label)
{
    lightVertex.type = DIRECTION;
    lightVertex.flux = flux;
    lightVertex.pdf = pdf;
    lightVertex.singlePdf = pdf;
    lightVertex.normal = -direction;
    lightVertex.RMIS_pointer = 1;
    lightVertex.zoneId = label;
    lightVertex.isBrdf = false; 
}
RT_FUNCTION float light_hit_env(BDPTVertex& eyeVertex, BDPTVertex& lightVertex)
{
    float3 connect_dir = -lightVertex.normal;
    float3 flux = lightVertex.flux / lightVertex.pdf;
    float LL_pdf_A = getLast_pdf(eyeVertex, connect_dir, false);
    float3 flux_multiplier_0 = getFluxMultiplier(eyeVertex, -connect_dir);
    float3 weight_A = tracing_weight_eye(lightVertex, eyeVertex);
    float3 D_A_0 = ((eyeVertex.RMIS_pointer_3 * LL_pdf_A * flux_multiplier_0) + weight_A);

    float pdf_A = getPdf_from_light_source(lightVertex, eyeVertex);
    float flux_multiplier_1 = 1.0 / sky.projectPdf();
    float D_A = float3sum(D_A_0 * pdf_A * flux_multiplier_1 * flux / eyeVertex.singlePdf);


    float weight = float3sum(connectRate_SOL(eyeVertex.zoneId, lightVertex.zoneId, flux));

    if (eyeVertex.isBrdf || lightVertex.isBrdf)
    {
        weight = 0.0;
    }
    ////light side 

    float D_B = lightVertex.RMIS_pointer;// = lightVertex.d;

    float3 LB = normalize(eyeVertex.lastPosition - eyeVertex.position);
    float pdf_B = getPdf(eyeVertex, lightVertex, LB, true);
    //float D_B = D_B_0 * pdf_B / lightVertex.singlePdf;

    //rtPrintf("%f %f %f %f\n", pdf_A, pdf_B, weight_B,float3sum(LB));
    //return  pdf_B;
    return D_B / ((weight + D_A) / pdf_B * lightVertex.singlePdf + D_B);
    //return weight / (weight + D_A + D_B);

}
RT_FUNCTION float light_hit(BDPTVertex& eyeVertex, BDPTVertex& lightVertex)//lastvertex and virtual lightVertex
{
    float3 connect_vec = eyeVertex.position - lightVertex.position;
    float3 connect_dir = normalize(connect_vec);
    float3 flux = lightVertex.flux / lightVertex.pdf;

    float LL_pdf_A = getLL_pdf(lightVertex, eyeVertex, false);
    float3 flux_multiplier_0 = getFluxMultiplier(eyeVertex, -connect_dir);
    float3 weight_A = tracing_weight_eye(lightVertex, eyeVertex);
    float3 D_A_0 = ((eyeVertex.RMIS_pointer_3 * LL_pdf_A * flux_multiplier_0) + weight_A);
     
    float pdf_A = getPdf_from_light_source(lightVertex, eyeVertex);
    float flux_multiplier_1 = M_PIf;
    float D_A = float3sum(D_A_0 * pdf_A * flux_multiplier_1 * flux / eyeVertex.singlePdf); 
    float weight = float3sum(connectRate_SOL(eyeVertex.zoneId, lightVertex.zoneId, flux));

    if (eyeVertex.isBrdf || lightVertex.isBrdf)
    {
        weight = 0.0;
    }
    ////light side 

    float D_B = lightVertex.RMIS_pointer;// = lightVertex.d;

    float3 LB = normalize(eyeVertex.lastPosition - eyeVertex.position);
    float pdf_B = getPdf(eyeVertex, lightVertex, LB, true);
    float ra =  pdf_B / lightVertex.singlePdf;

    //rtPrintf("%f %f %f %f\n", pdf_A, pdf_B, weight_B,float3sum(LB));
    return D_B / ((weight + D_A) / pdf_B * lightVertex.singlePdf + D_B);
}

#endif