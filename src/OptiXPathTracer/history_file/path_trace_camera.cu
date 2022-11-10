/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 #include <optix.h>
 #include <optixu/optixu_math_namespace.h>
 #include <optixu/optixu_matrix_namespace.h> 
 #include "helpers.h"
 #include "prd.h"
 #include "rt_function.h"
 #include "random.h"
 #include "light_parameters.h"
 #include "PG_device.h"
 #include "ZGC_device.h"
 #include "SVM_device.h"
 #include "rmis.h"
 #include "MLP_device.h" 
 using namespace optix;
 
 
 rtDeclareVariable(float3,        bad_color, , );
 rtDeclareVariable(float,         scene_epsilon, , );
 rtDeclareVariable(float3,        cutoff_color, , );
 rtDeclareVariable(int,           max_depth, , );
 rtBuffer<uchar4, 2>              output_buffer;
 rtBuffer<uchar4, 2>              false_buffer;
 rtBuffer<uchar4, 2>              false_mse_buffer;
 rtBuffer<uchar4, 2>              standrd_buffer;
 rtBuffer<float4, 2>              accum_buffer;
 rtBuffer<float4, 2>              tonemapped_buffer;
 rtBuffer<float4, 2>              input_albedo_buffer;
 rtBuffer<float4, 2>              input_normal_buffer;
 rtBuffer<float4, 2>              denoised_buffer;
 rtBuffer<float4, 2>              standrd_float_buffer;
 rtBuffer<float,2>                visibility_buffer;
 rtBuffer<float4, 2>              LT_result_buffer;
 rtBuffer<eyeResultRecord,3>                result_record_buffer;
 
 rtBuffer<triangleStruct,1>       triangle_samples;
 rtBuffer<triangleStruct,1>       triangle_targets;
 rtBuffer<UberZoneLVC,1>          uberLVC; 
 rtDeclareVariable(float, scene_area, , ) = {1.0};
 
 rtBuffer<RAWVertex,1>            raw_LVC;
 rtBuffer<BDPTVertex,1>           LVC;
 rtBuffer<BDPTVertex,1>           PM;  
 rtBuffer<ZoneMatrix,1>           M3_buffer;
 rtBuffer<KDPos,1>        Kd_position;
 rtBuffer<KDPos,1>        last_kd_position;
 rtBuffer<KDPos,1>        KdPM;
 rtBuffer<TestSetting, 1>     test_setting;
 
 rtBuffer<PMFCache,3>        PMFCaches;
 rtDeclareVariable(rtObject,      top_object, , );
 rtDeclareVariable(unsigned int,  frame, , );
 rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );
 rtDeclareVariable(float,           estimate_min_limit, , ) = { 0.01 };
 rtDeclareVariable(int,           KD_SET, , ) = { 0 };
 rtDeclareVariable(int,           LTC_path_count, , ) = { 1 };
 
 rtBuffer< rtCallableProgramId<void(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)> > sysLightSample;
 rtBuffer<LightParameter> sysLightParameters;
 rtBuffer<LightTraceCache> LTC;
 
#define ISINVALIDVALUE(ans) (ans.x>100000.0f|| isnan(ans.x)||ans.y>100000.0f|| isnan(ans.y)||ans.z>100000.0f|| isnan(ans.z))
//#define ISLIGHTSOURCE(a) (a.zoneId >= SUBSPACE_NUM - MAX_LIGHT)
#define ISLIGHTSOURCE(a) (a.type == HIT_LIGHT_SOURCE||a.type == ENV_MISS)
#define ISVALIDVERTEX(a) (fmaxf(a.flux / a.pdf)>= 0.00000001f)

__device__ inline float4 ToneMap_exposure(const float4& c, float exposure)
{  
  float3 ldr = make_float3(1.0) - make_float3(exp(-c.x * exposure),exp(-c.y * exposure),exp(-c.z * exposure));
  return make_float4(ldr.x,ldr.y,ldr.z,1.0f);
}
 __device__ inline float4 ToneMap(const float4& c, float limit)
 {
   //return ToneMap_exposure(c,limit);

   float luminance = 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;
 
   float4 col = c * 1.0f / (1.0f + 1 * luminance / limit);
   return make_float4(col.x, col.y, col.z, 1.0f);
 }
 
__device__ inline float color2luminance(const float3 &c)
{
	return 0.3f*c.x + 0.6f*c.y + 0.1f*c.z;
}
 __device__ inline float4 LinearToSrgb(const float4& c)
 {
   const float kInvGamma = 1.0f / 2.2f;
   return make_float4(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w);
 }
 
 
RT_FUNCTION optix::float3 contriCompute(BDPTVertexStack &path)
{
    //要求：第0个顶点为eye，第size-1个顶点为light
    optix::float3 throughput = make_float3(pow(M,path.size));
    BDPTVertex & light = path.v[path.size - 1];
    BDPTVertex & lastMidPoint = path.v[path.size - 2];
    optix::float3 lightLine = lastMidPoint.position - light.position;
    optix::float3 lightDirection = normalize(lightLine);
    float lAng = dot(light.normal, lightDirection);
    if (lAng < 0.0f)
    {
        return make_float3(0.0f);
    }
    optix::float3 Le = light.flux* lAng;
    throughput *= Le;
    for (int i = 1; i < path.size; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        optix::float3 line = midPoint.position - lastPoint.position;
        throughput /= dot(line, line);
    }
    for (int i = 1; i < path.size - 1; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        BDPTVertex &nextPoint = path.v[i + 1];
        optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);

        MaterialParameter mat = sysMaterialParameters[midPoint.materialId];
        mat.color = midPoint.color;

        throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
            * DisneyEval(mat, midPoint.normal, lastDirection, nextDirection);
    }
    return throughput;
}
 
RT_FUNCTION optix::float3 n_pow_evalPath(BDPTVertexStack &path, int maxLength,int eyePathLen,float bufferWeight)
{
    optix::float3 contri;
    contri = contriCompute(path);
    //path.v[0].dLast = 1.0f;
    for (int i = 1; i < path.size; i++)
    {
        int li = path.size - i - 1;
        //path.v[i].dLast = M;
        path.v[li].pdf = M;

        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            //path.v[i].dLast *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                
                MaterialParameter mat = sysMaterialParameters[midPoint.materialId];
                mat.color = midPoint.color;
                //path.v[i].dLast *= DisneyPdf(mat, midPoint.normal, lastDirection, nextDirection,midPoint.position,true);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                
                MaterialParameter mat = sysMaterialParameters[midPoint.materialId];
                mat.color = midPoint.color;
                path.v[li].pdf *= DisneyPdf(mat, midPoint.normal, lastDirection, nextDirection,midPoint.position);
            }
        }
        //path.v[i].dLast *= path.v[i - 1].dLast;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float pdf = 0.0f;
    float currentPdf;
    float currentWeight = 0.0f;
    float weight = 0.0f;
    float powRate = POWER_RATE;
    if (eyePathLen == path.size)
    {
        //currentPdf = path.v[path.size - 1].dLast * M;
        currentWeight = currentPdf;
    }
    else
    {
        //currentPdf = path.v[eyePathLen - 1].dLast * path.v[eyePathLen].pdf * M * M * bufferWeight;
        currentWeight = currentPdf;
    }


    for (int i = 1; i < path.size - 1; i++)
    {
       // weight += pow( path.v[i].dLast * path.v[i + 1].pdf * M * M * bufferWeight,powRate);
    }
   // weight += pow(path.v[path.size - 1].dLast * M,powRate);
    optix::float3 ans = contri *(currentWeight/ weight) / currentPdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z) || ans.x > DISCARD_VALUE || ans.y > DISCARD_VALUE || ans.z > DISCARD_VALUE)
    {
        return make_float3(0.0f);
    }
    return ans;
}

__device__ float3 Direct_light(BDPTVertex &a,PerRayData_radiance * p)
{


  float3 fa,fb;
  float3 LA = a.lastPosition - a.position;
  float3 LA_DIR = normalize(LA);
  

  PerRayData_radiance prd = *p;

  float3 L = make_float3(0.0f);
  int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
  LightParameter light = sysLightParameters[index];
  LightSample lightSample;
  
 sysLightSample[light.lightType](light, prd, lightSample);
 BDPTVertex b;
 b.position = lightSample.surfacePos;
 b.normal = lightSample.normal;
 b.flux = light.emission;

  float3 lightDir = b.position - a.position;
  float lightDist = length(lightDir);
  float lightDistSq = lightDist * lightDist;
  lightDir /= sqrtf(lightDistSq);
  float NdotL = dot(b.normal, -lightDir);
  float lightPdf = lightDistSq * b.pdf / ( NdotL);

 
  if (dot(lightDir, a.normal) <= 0.0f || dot(lightDir, b.normal) >= 0.0f )
     return make_float3(0.0f);
 
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay = optix::make_Ray(a.position, lightDir, 1, scene_epsilon, lightDist - scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  
  if (!prd_shadow.inShadow)
  {
    float NdotL = dot(lightSample.normal, -lightDir);
    float lightPdf = lightDistSq / (light.area * NdotL);

    MaterialParameter mat_a = sysMaterialParameters[a.materialId];
    mat_a.color = a.color;
    fa = DisneyEval(mat_a,a.normal,lightDir,LA_DIR);

    float3 f = fa * abs(dot(a.normal,lightDir));

    L =  (a.flux / a.pdf) * f * light.emission / max(0.001f, lightPdf);
  }

    return L;

}
rtBuffer<float,1>          corput_buffer; 
__device__ float corput_sequence(int index ,int base)
{
  return corput_buffer[index];

  float result = 0;
  float f = 1.0f / base;
  int i = index;
  while(i>0)
  {
    result = result + f * (i % base);
    i = i / base;
    f = f / base;
  }
  return result;
}
__device__ float bias_corput(int index, float r)
{
  float result = corput_sequence(index,2) + r;
  return result > 1 ? result - 1 : result;
}
__device__ float3 zoneColor(int zoneNum)
{
  int sum = SUBSPACE_NUM;
  int h = zoneNum % 30 * 12;
  float s = (zoneNum) / float(SUBSPACE_NUM);
  float v = 1 - s;
  return hsv2rgb(h,s,v);
}

RT_FUNCTION float3 weight_visualization(float weight)//weight~(0,1)
{
    weight = min(weight, 1.0);
    float c = weight;// ToneMap(make_float4(weight), 1.5).x;
    float diff = c * 290;

    float3 colors = hsv2rgb(((-int(diff) + 290) % 360), 1.0, 1.0);
    float lum = 0.6 * colors.x + 0.3 * colors.y + 0.1 * colors.z;
    float target = 0.6 * c;
    float ratio = lum / target;
    colors /= ratio;
    //float4 colors_4 = make_float4(colors, 0.0);
    float4 val = LinearToSrgb(ToneMap(make_float4(colors,0.0), 1.5));
    return make_float3(val);
}
__device__  float3 connectVertex_ZGCBPT(BDPTVertex &a,BDPTVertex &b,float selectRate = 1.0f)
{ 
  float3 connectVec = a.position - b.position;
  float3 connectDir = normalize(connectVec);
  float G = abs(dot(a.normal,connectDir)) * abs(dot(b.normal,connectDir)) / dot(connectVec,connectVec);
  float3 LA = a.lastPosition - a.position;
  float3 LA_DIR = normalize(LA);
  float3 LB = b.lastPosition - b.position;
  float3 LB_DIR = normalize(LB);
  float pg_pdfRate = b.depth==0? pg_quadPdfRate(b, connectDir):1.0;

  float3 fa,fb;
  float3 ADcolor;
  MaterialParameter mat_a = sysMaterialParameters[a.materialId];
  mat_a.color = a.color;
  fa = DisneyEval(mat_a,a.normal,-connectDir,LA_DIR) / (mat_a.brdf ? abs(dot(a.normal, connectDir)) : 1.0f);
  
  MaterialParameter mat_b;
  if(!b.isOrigin)
  {
    mat_b = sysMaterialParameters[b.materialId];
    mat_b.color = b.color;
    fb = DisneyEval(mat_b,b.normal,connectDir,LB_DIR) / (mat_b.brdf ? abs(dot(b.normal, connectDir)) : 1.0f);
  }
  else
  {
    if(dot(b.normal,-connectDir)>0.0f)
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
#ifdef ZGCBPT
  //contribution, pdf and RMIS computation
  float3 ans = contri / pdf * (b.depth == 0 ? connection_lightSource(a, b) : general_connection(a, b)); 
#else
  float3 ans = contri / pdf;
#endif


  if(ISINVALIDVALUE(ans))
  {
    return make_float3(0.0f);
  }
  return  ans;
}

 __device__  float3 connectVertex(BDPTVertex &a,BDPTVertex &b,float selectRate = 1.0f)
 {
   //if(b.depth!=0)
   //{
   //  return make_float3(0.0);
   //}
   float3 connectVec = a.position - b.position;
   float3 connectDir = normalize(connectVec);
   float G = abs(dot(a.normal,connectDir)) * abs(dot(b.normal,connectDir)) / dot(connectVec,connectVec);
   float3 LA = a.lastPosition - a.position;
   float3 LA_DIR = normalize(LA);
   float3 LB = b.lastPosition - b.position;
   float3 LB_DIR = normalize(LB);
   float pg_pdfRate = b.depth==0? pg_quadPdfRate(b, connectDir):1.0;

   float3 fa,fb;
   float3 ADcolor;
   MaterialParameter mat_a = sysMaterialParameters[a.materialId];
   mat_a.color = a.color;
   fa = DisneyEval(mat_a,a.normal,-connectDir,LA_DIR) / (mat_a.brdf ? abs(dot(a.normal, connectDir)) : 1.0f);
   
   MaterialParameter mat_b;
   if(!b.isOrigin)
   {
     mat_b = sysMaterialParameters[b.materialId];
     mat_b.color = b.color;
     fb = DisneyEval(mat_b,b.normal,connectDir,LB_DIR) / (mat_b.brdf ? abs(dot(b.normal, connectDir)) : 1.0f);
   }
   else
   {
     if(dot(b.normal,-connectDir)>0.0f)
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

   float3 ans = contri / pdf * (b.depth == 0 ? connection_lightSource(a, b) : general_connection(a, b)); 
   if(ISINVALIDVALUE(ans))
   {
     return make_float3(0.0f);
   }
   return  ans;
 }

 __device__ float3 connectVertex_uber(BDPTVertex &a,UberLightVertex &b)
 {
   float3 res = make_float3(0.0f);
  float3 connectVec = a.position - b.position;
  float3 connectDir = normalize(connectVec);
  float G = abs(dot(a.normal,connectDir)) * abs(dot(b.normal,connectDir)) / dot(connectVec,connectVec);
  float3 LA = a.lastPosition - a.position;
  float3 LA_DIR = normalize(LA);
  
  float3 fa,fb;
  
  MaterialParameter mat_a = sysMaterialParameters[a.materialId];
  mat_a.color = a.color;
  fa = DisneyEval(mat_a,a.normal,-connectDir,LA_DIR);
  
  MaterialParameter mat_b;
  mat_b.color = b.color;

  float3 normal = dot(connectDir,b.normal) > 0 ? b.normal : -b.normal;
  for(int i = 0; i < b.size; i++)
  {
    SubLightVertex &v = b.son[i]; 
    fb = DisneyEval(mat_b,normal,connectDir,v.dir);
    
    float3 contri = a.flux * v.contri * fa * fb * G;
    float pdf = a.pdf * v.pdf;
    //float ratio = (v.origin_pdf * connectRate(a.zoneId,b.zoneId)) / uber_pdf(a.zoneId,b.zoneId,v.zoneId);
    //if(ratio<0.1)
      res += contri / pdf / uber_pdf(a.zoneId,b.zoneId,v.zoneId);
    //rtPrintf("%f\n",ratio);
  }
  return res;
 }

 __device__  float3 connectVertex_pmf(BDPTVertex &a,BDPTVertex &b)
 {
   float3 connectVec = a.position - b.position;
   float3 connectDir = normalize(connectVec);
   float G = abs(dot(a.normal,connectDir)) * abs(dot(b.normal,connectDir)) / dot(connectVec,connectVec);

   float3 LB = b.lastPosition - b.position;
   float3 LB_DIR = normalize(LB);
   
   float3 fa,fb;
   
   
   MaterialParameter mat_b;
   if(!b.isOrigin)
   {
     mat_b = sysMaterialParameters[b.materialId];
     mat_b.color = b.color;
     fb = DisneyEval(mat_b,b.normal,connectDir,LB_DIR);
   }
   else
   {
     if(dot(b.normal,-connectDir)>0.0f)
     {
       fb = make_float3(0.0f);
     }
     else
     { 
        fb = make_float3(1.0f / PCBPT_DIRECT_FACTOR);
     }
   }
 
   float3 contri =  fb * G *b.flux;
   float pdf = b.pdf;

   
   
   float3 ans = fb*G  ;
   
   #ifdef PCBPT_STANDARD_MIS
   ans = contri / pdf;
   #endif
   if(ISINVALIDVALUE(ans))
   {
     return make_float3(0.0f);
   }
   return  ans;
 }
#define PCBPT_FAIL_K 0.02
//#define PCBPT_FAIL_K 0.001            
__device__ float3 direction_connect_pmf(BDPTVertex& a,BDPTVertex &b)
{
  float3 L = make_float3(0.0f);
  float3 connectDir = -b.normal; 
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay (a.position, connectDir, ShadowRay, scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  float G = abs(dot(a.normal,connectDir));
  if (!prd_shadow.inShadow && dot(a.normal,connectDir)>0.0)
  {
    L =   b.flux / b.pdf * G;
  }   
  else 
  {
    L =   b.flux / b.pdf * G * PCBPT_FAIL_K;

  }
  if(ISINVALIDVALUE(L))
  {
    return make_float3(0.0f);
  }
  return L;
  
}


__device__ float3 direction_connect(BDPTVertex& a,BDPTVertex &b)
{
  float3 L = make_float3(0.0f);
  float3 connectDir = -b.normal; 
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay (a.position, connectDir, ShadowRay, scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  if (!prd_shadow.inShadow && dot(a.normal,connectDir)>0.0)
  {
    MaterialParameter mat = sysMaterialParameters[a.materialId];
    mat.color = a.color;
    float3 f = DisneyEval(mat,a.normal,normalize(a.lastPosition - a.position),connectDir) 
    * dot(a.normal,connectDir);
    //float lightPdf =abs(dot(a.normal,connectDir)) * 4.0 / (M_PIf * sceneMaxLength * sceneMaxLength);
    float projectPdf = b.type == DIRECTION ? 1.0 / DirProjectionArea : sky.projectPdf();
    float lightPdf =abs(dot(a.normal,connectDir)) * projectPdf;

    float da = 0.0;
    

    L = a.flux / a.pdf * f * b.flux / b.pdf * connection_direction_lightSource(a, b);
  }   
  if(ISINVALIDVALUE(L))
  {
    return make_float3(0.0f);
  } 
  return L;
  
}
 

__device__ float3 direction_connect_ZGCBPT(BDPTVertex& a,BDPTVertex &b)
{
  float3 L = make_float3(0.0f);
  float3 connectDir = -b.normal; 
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay (a.position, connectDir, ShadowRay, scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  if (!prd_shadow.inShadow && dot(a.normal,connectDir)>0.0)
  {
    MaterialParameter mat = sysMaterialParameters[a.materialId];
    mat.color = a.color;
    float3 f = DisneyEval(mat,a.normal,normalize(a.lastPosition - a.position),connectDir) 
    * dot(a.normal,connectDir); 

    //RMIS, pdf and contribution computation
    L = a.flux / a.pdf * f * b.flux / b.pdf * connection_direction_lightSource(a, b); 
  }   
  if(ISINVALIDVALUE(L))
  {
    return make_float3(0.0f);
  }
  return L;
  
}
 
 
 
 __device__  float3 lightStraghtHit(BDPTVertex &a)
 {
   float3 contri = a.flux;
   float pdf = a.pdf;
   float inver_weight = a.RMIS_pointer;
   
   float3 ans = contri / pdf / inver_weight;
   if(ISINVALIDVALUE(ans))
   {
     return make_float3(0.0f);
   }
   return  ans;
 }
 

 RT_PROGRAM void pinhole_camera()
 {
 
   size_t2 screen = output_buffer.size();
   unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
 
   // Subpixel jitter: send the ray through a different position inside the pixel each time,
   // to provide antialiasing.
   float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);
 
   float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
   float3 ray_origin = eye;
   float3 ray_direction = normalize(d.x*U + d.y*V + W);
 
   PerRayData_radiance prd;
   prd.depth = 0;
   prd.seed = seed;
   prd.done = false;
   prd.pdf = 0.0f;
   prd.specularBounce = false;
 
   // These represent the current shading state and will be set by the closest-hit or miss program
 
   // attenuation (<= 1) from surface interaction.
   prd.throughput = make_float3( 1.0f );
 
   // light from a light source or miss program
   prd.radiance = make_float3( 0.0f );
 
   // next ray to be traced
   prd.origin = make_float3( 0.0f );
   prd.direction = make_float3( 0.0f );
   float3 result = make_float3( 0.0f );
 
   // Main render loop. This is not recursive, and for high ray depths
   // will generally perform better than tracing radiance rays recursively
   // in closest hit programs.
   for(;;) {
       optix::Ray ray(ray_origin, ray_direction, /*ray type*/ 0, scene_epsilon );
       rtTrace(top_object, ray, prd);
 
       if ( prd.done || prd.depth >= max_depth)
           break;
 
       prd.depth++;
 
       // Update ray data for the next path segment
       ray_origin = prd.origin;
       ray_direction = prd.direction;
   }
   result = prd.radiance;
 
 
 
 
   float4 acc_val = accum_buffer[launch_index];
   if( frame > 0 ) {
     acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame+1 ) );
   } else {
     acc_val = make_float4( result, 0.f );
   }
 
   float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
   //float4 val = LinearToSrgb(acc_val);
 
   output_buffer[launch_index] = make_color(make_float3(val));
   accum_buffer[launch_index] = acc_val;

   #ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  
  
#endif
 }
 
 RT_PROGRAM void exception()
 {
   const unsigned int code = rtGetExceptionCode();
   rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
   output_buffer[launch_index] = make_color( bad_color );
 }
 
 RT_PROGRAM void BDPT_pinhole_camera()
 {
   
   size_t2 screen = output_buffer.size();
   unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
 
   // Subpixel jitter: send the ray through a different position inside the pixel each time,
   // to provide antialiasing.
   float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);
 
   float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
   float3 ray_origin = eye;
   float3 ray_direction = normalize(d.x*U + d.y*V + W);
 
   BDPTVertexStack EStack;
   EStack.size = 1;
   EStack.v[0].position = eye;
   EStack.v[0].flux = make_float3(1.0f);
   EStack.v[0].pdf = 1.0f;
   EStack.v[0].RMIS_pointer = 0;
   EStack.v[0].normal = ray_direction;
   EStack.v[0].isOrigin = true;
   EStack.v[0].depth = 0;
 
   EStack.v[1].singlePdf = 1.0f;
   PerRayData_radiance prd;
   prd.depth = 0;
   prd.seed = seed;
   prd.done = false;
   prd.pdf = 0.0f;
   prd.specularBounce = false;
 
   prd.stackP = &EStack;
   // These represent the current shading state and will be set by the closest-hit or miss program
 
   // attenuation (<= 1) from surface interaction.
   prd.throughput = make_float3( 1.0f );
 
   // light from a light source or miss program
   prd.radiance = make_float3( 0.0f );
 
   // next ray to be traced
   prd.origin = make_float3( 0.0f );
   prd.direction = make_float3( 0.0f );
   float3 result = make_float3( 0.0f );
 
   // Main render loop. This is not recursive, and for high ray depths
   // will generally perform better than tracing radiance rays recursively
   // in closest hit programs.
  for(;;) 
  {
    if ( prd.done || prd.depth >= max_depth)
      break;
    
    optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
    rtTrace(top_object, ray, prd);
    
    prd.depth++;
 
    // Update ray data for the next path segment
    ray_origin = prd.origin;
    ray_direction = prd.direction;
   }
 
   /*light path Trace*/
   BDPTVertexStack LStack;
   LStack.size = 0;
 
   int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
   LightParameter light = sysLightParameters[index];
   LightSample lightSample;
   sysLightSample[light.lightType](light, prd, lightSample);
 
   float lightPdf = 1.0  / sysNumberOfLights / light.area;
   float3 BDPTResult = make_float3(0.0f);
   BDPTVertex &lightVertex = LStack.v[LStack.size];
   BDPTVertex &nextVertex = LStack.v[LStack.size + 1];
   lightVertex.position = lightSample.surfacePos;
   lightVertex.normal = lightSample.normal;
   lightVertex.flux = lightSample.emission;
   lightVertex.pdf = lightPdf;
   lightVertex.singlePdf = lightPdf;
   lightVertex.isOrigin = true;
   lightVertex.isBrdf = false;
   lightVertex.depth =  0;
 
   float3 dir;
   float r1 = rnd(prd.seed);
   float r2 = rnd(prd.seed); 
   optix::Onb onb( lightVertex.normal );
 
   cosine_sample_hemisphere(r1, r2, dir); 
   onb.inverse_transform(dir);
   
   lightVertex.RMIS_pointer = 1;
   
   LStack.v[1].singlePdf = abs(dot(light.normal , dir)) * (1.0f / M_PIf);
   
   if(light.lightType == DIRECTION)
   {
     dir = lightSample.normal;
     lightVertex.RMIS_pointer = 0;
     lightVertex.type = DIRECTION;
     LStack.v[1].singlePdf  = 1.0 / DirProjectionArea;
   }
   LStack.size++;
   
   prd.depth = 0;
   prd.done = false;
   prd.specularBounce = false;
   prd.stackP = &LStack;
 
 
   ray_origin = lightVertex.position;
   ray_direction = dir;
   for(;;) {
    if ( prd.done || prd.depth >= max_depth)
    break;
       optix::Ray ray(ray_origin, ray_direction, /*ray type*/  BDPT_L_Ray, scene_epsilon );
       rtTrace(top_object, ray, prd);
 
 
       prd.depth++;
 
       // Update ray data for the next path segment
       ray_origin = prd.origin;
       ray_direction = prd.direction;
   }
 /*light path trace end*/
   if(EStack.size > 1 && ISLIGHTSOURCE(EStack.v[EStack.size - 1]))
   {
     BDPTResult +=   lightStraghtHit(EStack.v[EStack.size - 1]);
     
     EStack.size--;
   }
   for(int i=1;i<EStack.size;i++)
   {
     for(int j=0;j<LStack.size;j++)
     {
       
       //get vertex
       BDPTVertex &eyeEndVertex = EStack.v[i];
       BDPTVertex &lightEndVertex = LStack.v[j];
       if(!ISVALIDVERTEX(eyeEndVertex)|| !ISVALIDVERTEX(lightEndVertex))
       {
         continue;
       }
       //if(lightEndVertex.type == DIRECTION)
       //{
       //  BDPTResult += direction_connect(eyeEndVertex,lightEndVertex) ;
       //  continue;
       //}  
       //connect vertex
       float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
       float3 connectDir = normalize(connectVec);
       PerRayData_shadow prd_shadow;
       prd_shadow.inShadow = false;
       optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
          length(connectVec) - scene_epsilon);
       rtTrace(top_object, shadowRay, prd_shadow);
       if (prd_shadow.inShadow)
       {
           continue;
       }

      // BDPTVertexStack computeStack;
      // computeStack.size = i + j + 2;
      // for(int k = 0;k<computeStack.size;k++)
      // {
      //  if(k<i + 1)
      //  {
      //    computeStack.v[k] = EStack.v[k];
      //  }
      //  else
      //  {
      //    int nk = k - i - 1;
      //    computeStack.v[computeStack.size - nk - 1] = LStack.v[nk]; 
      //  }
      //}
      //
      //BDPTResult += n_pow_evalPath(computeStack,0,i + 1,1);

      
      BDPTResult += connectVertex(eyeEndVertex,lightEndVertex);
     }
   }
   result = BDPTResult;
   result += prd.radiance;
   
 /*computation*/
 
 
 
   float4 acc_val = accum_buffer[launch_index];
   if( frame > 0 ) {
     acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame+1 ) );
   } else {
     acc_val = make_float4( result, 0.f );
   }
 
   float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
   //float4 val = LinearToSrgb(acc_val);
 
   output_buffer[launch_index] = make_color(make_float3(val));
   accum_buffer[launch_index] = acc_val;
 #ifdef USE_DENOISER
   tonemapped_buffer[launch_index] = val;
   if(EStack.size>1)
   {
     input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
     input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
   }
 #endif
 }
 
 __device__ uint2 dir2pix(float3 dir)
 {
  //avaliable value: eye U V W
  float3 mid = eye + W;
  float3 leftUp = mid - U - V;
  float3 right = U;
  float3 down  = V;
  float l2w_array[16] = {
		right.x,down.x,dir.x,0,
		right.y,down.y,dir.y,0,
		right.z,down.z,dir.z,0,
    0      ,0     ,0    ,1};
  Matrix<4,4> L2W(l2w_array);
  Matrix<4,4> W2L = L2W.inverse();
  float4 W = make_float4(eye - leftUp,1); 
  float4 L = W2L * W;
  
  size_t2 screen = output_buffer.size();
  return make_uint2(L.x * screen.x / 2 + 0.5 ,L.y * screen.y / 2 + 0.5);
 }
 __device__ bool isVisible(float3 pos)
 {
  float3 connectVec = eye - pos;
  float3 connectDir = normalize(connectVec);
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay = optix::make_Ray(pos, connectDir, ShadowRay, scene_epsilon,
     length(connectVec) - scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  if (!prd_shadow.inShadow)
  {
    return true;
  }
  return false;
 }
 __device__ bool isVisible(float3 pos,float3 pos2)
 {
  float3 connectVec = pos2 - pos;
  float3 connectDir = normalize(connectVec);
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay = optix::make_Ray(pos, connectDir, ShadowRay, scene_epsilon,
     length(connectVec) - scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  if (!prd_shadow.inShadow)
  {
    return true;
  }
  return false;
 }
 
__device__ bool isVisible_direction(float3 position,float3 direction)
{
  PerRayData_shadow prd_shadow;
  prd_shadow.inShadow = false;
  optix::Ray shadowRay (position, direction, ShadowRay, scene_epsilon);
  rtTrace(top_object, shadowRay, prd_shadow);
  return !prd_shadow.inShadow;

}

__device__ bool isVisible_direction(float3 position, float3 normal, float3 direction)
{
  //if() return 0;
  bool normal_flag = dot(normal,direction) > 0;

  return normal_flag && isVisible_direction(position,direction);

}
RT_FUNCTION float3 accum_pm(MaterialParameter & mat, BDPTVertex &a,BDPTVertex &b,float radius)
{
  if(dot(a.normal,b.normal)<0.01)
    return make_float3(0.0f);
  float3 f = DisneyEval(mat,a.normal,normalize(a.lastPosition - a.position),normalize(b.lastPosition - b.position));
  return f * b.flux / b.pdf / (M_PIf * radius * radius) / LTC_path_count;
}
RT_FUNCTION float3 PM_gather(BDPTVertex &a,float radius)
{

  MaterialParameter mat = sysMaterialParameters[a.materialId];
  mat.color = a.color;

  float3 res = make_float3(0.0f); 
  float closest_dis2 = radius * radius;
  float3 position = a.position;
  unsigned int stack[25]; 
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]
 
  push_node( 0 );

  do { 
    KDPos& currentVDirector = KdPM[node];
    uint axis = currentVDirector.axis;
    if( !( axis & PPM_NULL ) ) {

      float3 vd_position = currentVDirector.position;
      float3 diff = position - vd_position;
      float distance2 = dot(diff, diff);
      if(distance2 < closest_dis2)
      {
        res += accum_pm(mat,a,PM[currentVDirector.original_p],radius);
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < closest_dis2 ) { 

          push_node( (node<<1) + 2 - selector );
        }

        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node(); 
      }
    } else {
      node = pop_node(); 
    }
  } while ( node );
  return res;
}
rtDeclareVariable(float,           idea_energy, , ) = { 1 };
rtDeclareVariable(int  ,           LVC_frame, , ) = {1};
rtDeclareVariable(int,            M_FIX, ,) = {1};
rtDeclareVariable(int ,            LT_CORE_NUM, ,) = {PCPT_CORE_NUM};
rtDeclareVariable(int ,            LT_CORE_SIZE, ,) = {LIGHT_VERTEX_PER_CORE};

RT_PROGRAM void light_vertex_launch()
{ 
  //注意，launch_index.y是个常数——这里之所以launch的是二维只是为了能够与之前的raygen函数兼容,实际上变化的只有launch_index.x
  //launch_index.y always equals to 0.
  //launch_index is actually one-dimension. launch_index.x refers to the ID of tracing core.
  //it is a int2 var in order to be compatible with other raygen programs in this file 
    unsigned int seed = (launch_index.y + launch_index.x) + (LT_CORE_NUM * LT_CORE_SIZE) * (LVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);
        // = tea<16>((launch_index.y + launch_index.x) + (LT_CORE_NUM * LT_CORE_SIZE) * LVC_frame, LVC_frame); 


  int lightVertexCount = 0;
  int lightPathCount = 0;
  int buffer_bias = launch_index.x * LT_CORE_SIZE;
  PerRayData_radiance prd;
  prd.seed = seed;
  BDPTVertexStack LStack;
  int seed_bias = 0;
  while(true)
  { 

//initialize the light source vertex and the light sub-path
    prd.seed = tea<16>(buffer_bias + seed_bias,LVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);
    seed_bias++;
    LStack.size = 0;
    LStack.clear();
    int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);  
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;
    sysLightSample[light.lightType](light, prd, lightSample);

    float lightPdf = 1.0  / sysNumberOfLights * lightSample.pdf;
    
    float3 BDPTResult = make_float3(0.0f);
    BDPTVertex &lightVertex = LStack.v[LStack.size];
    BDPTVertex &nextVertex = LStack.v[LStack.size + 1];
    lightVertex.position = lightSample.surfacePos;
    lightVertex.normal = lightSample.normal;
    lightVertex.flux = lightSample.emission; 
    lightVertex.pdf = lightPdf;
    lightVertex.singlePdf = lightPdf;
    lightVertex.isOrigin = true;
    lightVertex.isBrdf = false;
    lightVertex.depth =  0;
    //lightVertex.zoneId = SUBSPACE_NUM - index - 1;
    lightVertex.zoneId = lightSample.zoneId; 
    lightVertex.type = light.lightType;
    lightVertex.materialId = index; 
    float3 dir = lightSample.dir;
     

    lightVertex.RMIS_pointer = 1; 
    LStack.v[1].singlePdf = lightSample.pdf_dir;

    if(light.lightType == DIRECTION)
    { 
      
      lightVertex.RMIS_pointer = light.lightType == DIRECTION?0.0:lightVertex.RMIS_pointer;
       
      lightVertex.type = DIRECTION;
      LStack.v[1].singlePdf  = 1.0 / DirProjectionArea;
    }
    else if (light.lightType == ENV)
    { 
        lightVertex.RMIS_pointer = lightVertex.RMIS_pointer;
        lightVertex.type = ENV;
        LStack.v[1].singlePdf = sky.projectPdf();
    }
    else if(light.lightType == QUAD)
    {
        lightVertex.uv = lightSample.uv;
        lightVertex.pg_lightPdf = make_float2(lightSample.pdf, lightSample.pdf_dir);
           // abs(dot(light.normal, dir)) * (1.0f / M_PIf);
    }
    LStack.size++;
    
    prd.depth = 0;
    prd.done = false;
    prd.specularBounce = false;
    prd.stackP = &LStack;
    float3 ray_origin = lightVertex.position;
    float3 ray_direction = dir;
    
    lightPathCount++;
    RAWVertex &v = raw_LVC[buffer_bias + lightVertexCount];
    v.v = LStack.v[0];
    v.lastZoneId = -1;
    v.valid = true;
    

    lightVertexCount++; 
    if(lightVertexCount > LT_CORE_SIZE)break;
    for(;;) 
    {
      #ifndef UNBIAS_RENDERING
         if ( prd.done || prd.depth >= max_depth)
         break;
      #else
      if ( prd.done)
         break;
      #endif

      //begin tracing
        optix::Ray ray(ray_origin, ray_direction, /*ray type*/  BDPT_L_Ray, scene_epsilon );

        int origin_depth = LStack.size;
        rtTrace(top_object, ray, prd);  
        if(LStack.size == origin_depth)
        {
          //miss
          break;
        }
  
        if(!ISVALIDVERTEX(LStack.v[(LStack.size - 1) % STACKSIZE]))
          break; 
        RAWVertex &v = raw_LVC[buffer_bias + lightVertexCount];
        v.v = LStack.v[(LStack.size - 1) % STACKSIZE];
        if(v.v.depth!=0)
        {
          v.v.type = QUAD;
           
        }
        v.lastZoneId = LStack.v[(LStack.size - 2) % STACKSIZE].zoneId;
        v.valid = true; 
 
 
        lightVertexCount++;
        if(lightVertexCount >= LT_CORE_SIZE)break;
  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    }

    if(lightVertexCount >= LT_CORE_SIZE)
    {
      while(raw_LVC[buffer_bias + lightVertexCount].v.depth!= 0)
      {
        lightVertexCount --;
      }
      lightVertexCount--;
      lightPathCount--;
      break;
    } 

    if (lightPathCount >= M_PER_CORE)
    {
        break;
    }
    if (M_FIX == 1)
    {
        break;
    }
  }  
  for(int i = lightVertexCount;i<LT_CORE_SIZE;i++)
  {
    raw_LVC[buffer_bias +i].valid = false;
    //LTC[buffer_bias + i].valid = false;
  }
  
}

RT_FUNCTION float p_visual_eval(RAWVertex& v)
{ 
    size_t2 screen = output_buffer.size();
    if (isVisible(v.v.position) && dot(W, v.v.position - eye) > 0.0 && dot(v.v.normal, v.v.position - eye) < 0)
    {
        if (v.v.depth != 0)
        {
            MaterialParameter mat = sysMaterialParameters[v.v.materialId];
            mat.color = v.v.color;
            if (mat.brdf)return 0.0;
        }
        float3 vec = eye - v.v.position;
        float3 dir = normalize(vec);
        float len = length(vec);  
        //flux *= abs(dot(v.v.normal,dir)) / (len * len) ;

        float dot1 = abs(dot(v.v.normal, dir));
        float dot2 = abs(dot(dir, normalize(W)));
        float len2 = abs(dot(vec, W)) / length(W) / length(W);
        float plane_area = length(U) * length(V) * 4;

        float pdf2 = (dot1 / dot2) / (len2 * len2) / plane_area;

        if (frame > 1)
        {
            if (true||isVisible(v.v.position, test_setting[0].v.position))
            {
                float weight;
                float3 result = connectVertex(test_setting[0].v, v.v);
                weight = result.x + result.y + result.z ;
#ifdef ZGCBPT
                result = test_setting[0].v.flux / test_setting[0].v.pdf;
                weight = result.x + result.y + result.z;
                weight = connectRate_SOL( test_setting[0].v.zoneId, v.v.zoneId, weight) / iterNum * light_path_count ;
#endif
#ifdef LVCBPT
                weight = result.x + result.y + result.z; 
#endif // LVCBPT 
                //weight = light_path_count/ float(light_vertex_count) ;

//                float selectRate = weight / M2_buffer[test_setting[0].v.zoneId].Q;
//                float solrate = M2_buffer[test_setting[0].v.zoneId].r[v.v.zoneId] / M2_buffer[test_setting[0].v.zoneId].sum * light_path_count;
                return screen.x * screen.y * weight * pdf2 * 100;
            }
        }
    }
    return 0.0;
}
RT_PROGRAM void LTC_launch()
{
  //注意，launch_index.y是个常数——这里之所以launch的是二维只是为了能够与之前的raygen函数兼容,实际上变化的只有launch_index.x
  unsigned int seed = tea<17>(launch_index.y+launch_index.x, LVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);
  int lightVertexCount = 0;
  int lightPathCount = 0;
  int buffer_bias = launch_index.x * LTC_SPC;
  PerRayData_radiance prd;
  prd.seed = seed;
  BDPTVertexStack LStack;
  int seed_bias = 0;
  while(true)
  {  
    prd.seed = tea<16>(buffer_bias + seed_bias,LVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);
    seed_bias++;
    LStack.size = 0;
    LStack.clear();

    int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1); 
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;
    sysLightSample[light.lightType](light, prd, lightSample);

    float lightPdf = 1.0  / sysNumberOfLights * lightSample.pdf;
    
    float3 BDPTResult = make_float3(0.0f);
    BDPTVertex &lightVertex = LStack.v[LStack.size];
    BDPTVertex &nextVertex = LStack.v[LStack.size + 1];
    lightVertex.position = lightSample.surfacePos;
    lightVertex.normal = lightSample.normal;
    lightVertex.flux = lightSample.emission; 
    lightVertex.pdf = lightPdf;
    lightVertex.singlePdf = lightPdf;
    lightVertex.isOrigin = true;
    lightVertex.isBrdf = false;
    lightVertex.depth =  0;
    //lightVertex.zoneId = SUBSPACE_NUM - index - 1;
    lightVertex.zoneId = lightSample.zoneId;
    lightVertex.type = light.lightType;
    lightVertex.materialId = index;
    float3 dir = lightSample.dir; 
    
    lightVertex.RMIS_pointer = 1;
    #ifdef LVCBPT
    lightVertex.d = average_light_length / iterNum;
    #endif 

    LStack.v[1].singlePdf = lightSample.pdf_dir;

    if (light.lightType == DIRECTION)
    {

        lightVertex.RMIS_pointer = light.lightType == DIRECTION ? 0.0 : lightVertex.RMIS_pointer;
         
        lightVertex.type = DIRECTION;
        LStack.v[1].singlePdf = 1.0 / DirProjectionArea;
    }
    else if (light.lightType == ENV)
    { 

        lightVertex.type = ENV;
        LStack.v[1].singlePdf = sky.projectPdf();
    }
    else if (light.lightType == QUAD)
    {
        lightVertex.uv = lightSample.uv;
        lightVertex.pg_lightPdf = make_float2(lightSample.pdf, lightSample.pdf_dir);
        // abs(dot(light.normal, dir)) * (1.0f / M_PIf);
    }
    LStack.size++;
    
    prd.depth = 0;
    prd.done = false;
    prd.specularBounce = false;
    prd.stackP = &LStack;
    float3 ray_origin = lightVertex.position;
    float3 ray_direction = dir;
    
    lightPathCount++;
    RAWVertex v;
    v.v = LStack.v[0];
    v.lastZoneId = -1;
    v.valid = true;
    LTC[buffer_bias + lightVertexCount].valid = true;
    LTC[buffer_bias + lightVertexCount].origin = true;
    if(false&&v.v.depth == 0)
    {
      size_t2 screen = output_buffer.size();
      LightTraceCache &ltc = LTC[buffer_bias + lightVertexCount];
      ltc.pixiv_loc = dir2pix(normalize(v.v.position - eye));
 
      if( ltc.pixiv_loc.x >= 0 && ltc.pixiv_loc.x < screen.x &&
         ltc.pixiv_loc.y >=0  && ltc.pixiv_loc.y < screen.y
          && isVisible(v.v.position) && dot(W,v.v.position - eye)>0.0 && dot(v.v.normal,v.v.position - eye)<0)
      { 
        ltc.result = make_float3(p_visual_eval(v));
        ltc.valid = true;
      }
      else
      {
        ltc.valid = false;
      }
    } 
    lightVertexCount++;
    if(lightVertexCount > LTC_SPC)break;
    for(;;) 
    {
      #ifndef UNBIAS_RENDERING
         if ( prd.done || prd.depth >= max_depth)
         break;
      #else
      if ( prd.done)
         break;
      #endif
        optix::Ray ray(ray_origin, ray_direction, /*ray type*/  BDPT_L_Ray, scene_epsilon );
        
        int origin_depth = LStack.size;
        rtTrace(top_object, ray, prd);
        if(LStack.size == origin_depth)
        {
          //miss
          break;
        }
  
        if(!ISVALIDVERTEX(LStack.v[(LStack.size - 1) % STACKSIZE]))
          break; 
        RAWVertex v;
        v.v = LStack.v[(LStack.size - 1) % STACKSIZE]; 
        v.lastZoneId = LStack.v[(LStack.size - 2) % STACKSIZE].zoneId;
        v.valid = true;
 
        //LTC code
        if(true )
        {
          
          //int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
          //LightParameter light = sysLightParameters[index];
          //LightSample lightSample;
          //sysLightSample[light.lightType](light, prd, lightSample);
          //v.v.normal = lightSample.normal;
          //v.v.position = lightSample.surfacePos;

          size_t2 screen = output_buffer.size();
          LightTraceCache &ltc = LTC[buffer_bias + lightVertexCount];
          ltc.pixiv_loc = dir2pix(normalize(v.v.position - eye));
    
          MaterialParameter mat = sysMaterialParameters[v.v.materialId];
          mat.color = v.v.color;
          if( ltc.pixiv_loc.x >= 0 && ltc.pixiv_loc.x < screen.x &&
             ltc.pixiv_loc.y >=0  && ltc.pixiv_loc.y < screen.y
              && isVisible(v.v.position) && dot(W,v.v.position - eye)>0.0 &&
              !mat.brdf && dot(v.v.normal,v.v.position - eye)<0)
          {
            float3 vec = eye - v.v.position;
            float3 dir = normalize(vec);
            float len = length(vec);
            ltc.valid = true;
            float3 flux = v.v.flux * DisneyEval(mat,v.v.normal,dir,normalize(v.v.lastPosition - v.v.position));
            //flux *= abs(dot(v.v.normal,dir)) / (len * len) ;
            
            float dot1 = abs(dot(v.v.normal,dir));
            float dot2 = abs(dot(dir,normalize(W))); 
            float len2 = abs(dot(vec,W) )/ length(W) /length(W);
            float plane_area = length(U) * length(V)*4;

            float pdf2 = (dot1 / dot2)/(len2 * len2) / plane_area;
            float d_refine;
            {   
                BDPTVertex &b = v.v;
                float3 LB = b.lastPosition - b.position;
                float3 LB_DIR = normalize(LB);    
                  
                
                float L0,L1,R1,R0;
                float db;
                float rr_b = fmaxf(b.color);
                
              #ifdef RR_MIN_LIMIT 
                  rr_b = max(rr_b,MIN_RR_RATE);
              #endif
              #ifdef RR_DISABLE
                    rr_b = 1.0;
              #endif 
              //   rr_a = RR_RATE,rr_b = RR_RATE;
                  if(!b.isOrigin)
                  { 
                    R0 = DisneyPdf(mat ,b.normal,dir,LB_DIR,b.position,true)  / dot(LB,LB) * b.lastNormalProjection * rr_b; 
                    float tb_rate = b.lastSinglePdf / R0;
                
                  float b_weight = b.inBrdf?0.0f:1.0f;
              #ifdef ZGCBPT
              #ifdef ZGC_SAMPLE_ON_LUM 
                  b_weight = 1;//b.SOL_rate;
                  ////////////////////
                  //to be rewritee
                  ////////////////////
              #else 
                    b_weight *= connectRate(b.zoneId,b.lastZoneId);
              #endif
              #endif
              #ifdef PCBPT_MIS
                  if(KD_SET)
                  { 
                    b_weight *= b.pmf_link_rate;
                  }
              #endif
                  
              #ifdef INDIRECT_ONLY
                  if(b.depth == 1)
                    b_weight = b.inBrdf?0.0f:1.0f;
              #endif 
                   // db = (b.dLast / tb_rate) + b_weight; 
                  }    
                  d_refine = db;
            }   

            float t_pdf = abs(dot(v.v.normal,dir)) / len / len;
            float t_rate = v.v.singlePdf / t_pdf;


            float current_weight = get_LTC_weight(vec);
             float d = d_refine / t_rate + current_weight; 
            ltc.result = screen.x * screen.y * flux / v.v.pdf * pdf2 * current_weight/ d;

            //ltc.result = make_float3(p_visual_eval(v));
            float3  s = ltc.result / float(light_path_count) / LTC_SAVE_SUM * LIGHT_VERTEX_NUM;
            if(ISINVALIDVALUE(s))
            {
              ltc.valid = false;
            }
          }
          else
          {
            ltc.valid = false;
          }
        }
        else
        {
          LTC[buffer_bias + lightVertexCount].valid = false;
        } 
        lightVertexCount++;
        if(lightVertexCount >= LTC_SPC)break;
  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    }
    
    if(lightVertexCount >= LTC_SPC)
    {
      while(LTC[buffer_bias + lightVertexCount].origin!= true)
      {
        lightVertexCount --;
      }
      lightVertexCount--;
      lightPathCount--;
      break;
    } 
  }   
  
  for(int i = lightVertexCount;i<LTC_SPC;i++)
  {
    LTC[buffer_bias + i].valid = false;
    LTC[buffer_bias + i].origin = false;
  }
  
}

RT_PROGRAM void LVCBPT_pinhole_camera()
{
  int trace_num = iterNum;
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame + CONTINUE_RENDERING_BEGIN_FRAME);

  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;
  // These represent the current shading state and will be set by the closest-hit or miss program

  // attenuation (<= 1) from surface interaction.
  prd.throughput = make_float3( 1.0f );

  // light from a light source or miss program
  prd.radiance = make_float3( 0.0f );

  // next ray to be traced
  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  float3 PMResult = make_float3(0.0f);
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for(;;) {
#ifndef UNBIAS_RENDERING
   if ( prd.done || prd.depth >= max_depth)
   break;
#else
if ( prd.done)
   break;
#endif

      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }
      //BDPTResult += EStack.v[(EStack.size - 1)].normal;
      //break;
      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {
        float &inver_weight = EStack.v[(EStack.size - 1) % STACKSIZE].RMIS_pointer;
        inver_weight -=1;
        inver_weight /= average_light_length /trace_num;
        inver_weight +=1;
        
        BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
        EStack.size --;
        break;
      }

      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE];
      //BDPTResult = make_float3(1.0f) + eyeEndVertex.normal;
      //BDPTResult = eyeEndVertex.color;
      //break;
      if(!ISVALIDVERTEX(eyeEndVertex))
        break; 
 
      for(int i=0;i<trace_num;i++)
      {
        int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * light_vertex_count)), 0, light_vertex_count - 1);
        
        BDPTVertex &lightEndVertex = LVC[index]; 
        
        //if(eyeEndVertex.depth + lightEndVertex.depth > 3)
        //{
        //  prd.depth++;

          // Update ray data for the next path segment
        //  ray_origin = prd.origin;
        //  ray_direction = prd.direction;
        //  continue;
        //}
        if(lightEndVertex.type == DIRECTION || lightEndVertex.type == ENV)
        {
          BDPTResult += direction_connect(eyeEndVertex,lightEndVertex) * average_light_length  / trace_num;
          //prd.depth++;

          // Update ray data for the next path segment
          //ray_origin = prd.origin;
          //ray_direction = prd.direction;
          continue;
        }  
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
          length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow)
        {
          float k = float(light_vertex_count) / light_path_count;
  #ifdef LVC_RR
          float w = ENERGY_WEIGHT((lightEndVertex.flux / lightEndVertex.pdf));
          k = k * max(1.0f,idea_energy / w); 
  #endif
          BDPTResult += connectVertex(eyeEndVertex,lightEndVertex) * k / trace_num;
         //rtPrintf("%f %d %d\n", k, light_vertex_count, light_path_count);
        }
      
    //    BDPTResult = eyeEndVertex.color;
  //      break;
        if(eyeEndVertex.depth == 1)
        {
        // PMResult = PM_gather(eyeEndVertex,0.001 / sqrt(eyeEndVertex.singlePdf));
        }
      }
      prd.depth++;

      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
  }
  


  
  result =  BDPTResult+  prd.radiance  ;
  //result += PMResult;
  result += make_float3(LT_result_buffer[launch_index])  ;
  //result = make_float3(LT_result_buffer[launch_index]);
  
/*computation*/
int acc_frame_weight = frame ;
#ifdef CONTINUE_RENDERING
  if(frame == 3)
  {
    accum_buffer[launch_index] = standrd_float_buffer[launch_index];
  }
  acc_frame_weight += CONTINUE_RENDERING_BEGIN_FRAME;
#endif
   

  float4 acc_val = accum_buffer[launch_index];
  if( frame > 2 ) {
    acc_val = lerp( acc_val, make_float4( result, 0.f ), 
    1.0f / static_cast<float>( acc_frame_weight-1 ) );
  } else {
    acc_val = make_float4( result, 0.f );
  }
  //if(launch_index.x == 960 && launch_index.y == 800)
  //{
  //  rtPrintf("%f\n",acc_val.x / 0.5);
  //}
  
  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
  //float4 val = LinearToSrgb(acc_val);

  output_buffer[launch_index] = make_color(make_float3(val));

  float diff = 0;
  {
    float min_limit = estimate_min_limit;
    diff += abs(accum_buffer[launch_index].x - standrd_float_buffer[launch_index].x);
    diff += abs(accum_buffer[launch_index].y - standrd_float_buffer[launch_index].y);
    diff += abs(accum_buffer[launch_index].z - standrd_float_buffer[launch_index].z);
    float diff2 = diff;
    diff /= luminance(make_float3(standrd_float_buffer[launch_index])) + min_limit;
    diff *= 100;
    diff = min(int(diff),290);
    false_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));

    diff2 *= 1000;
    diff2 = min(int(diff2),290);
    false_mse_buffer[launch_index] = make_color(hsv2rgb(((-int(diff2) + 240)%360),1.0,1.0));
  }
  #ifdef VIEW_HEAT
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));
  #endif

  accum_buffer[launch_index] = acc_val;
#ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  if(EStack.size>1)
  {
    input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
    input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
  }
#endif
}




RT_PROGRAM void ZGCBPT_pinhole_camera()
{
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);


  //initialize the eye sub-path
  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;

  prd.throughput = make_float3( 1.0f );


  prd.radiance = make_float3( 0.0f );


  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  int prim_zone =0;
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for(;;) {
    #ifndef UNBIAS_RENDERING
       if ( prd.done || prd.depth >=  max_depth)
       break;
    #else
    if ( prd.done)
       break;
    #endif
    //begin tracing
      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
     
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }

      //if eye sub-path hit the light source
      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {
          //if the light source is env light
          //this code should be in the background.cu file 
          //I place it here to avoid some compilation problems
          if (EStack.v[(EStack.size - 1) % STACKSIZE].type == ENV_MISS && EStack.size > 2)
          {
              BDPTVertex& MidVertex = EStack.v[(EStack.size - 1) % STACKSIZE];
              float3 dir = -MidVertex.normal;
              BDPTVertex virtual_light;
              construct_virtual_env_light(virtual_light, sky.color(dir), 1.0 / sysNumberOfLights * sky.pdf(dir), dir, sky.getLabel(dir));
              
              //RMIS computation
              float dd = light_hit_env(EStack.v[(EStack.size - 2) % STACKSIZE],virtual_light);
              //printf("env hit compare %f %f\n", dd, 1.0 / MidVertex.d);
              MidVertex.RMIS_pointer = 1.0 / dd;
          }

        //float &inver_weight = EStack.v[(EStack.size - 1) % STACKSIZE].d; 
 
        //inver_weight *= 100; 
        BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
         
        //if (prd.depth > 5)BDPTResult *= 0;
        EStack.size --;
        break;
      }
      
      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE]; 
      if(eyeEndVertex.depth == 1)
        prim_zone = eyeEndVertex.zoneId;

       

      if(!ISVALIDVERTEX(eyeEndVertex))
        {
          break;
        }

      // if the eye subspace is invalid
      // actually I think this is a dead code....
      // just in case
      if(M2_buffer[eyeEndVertex.zoneId].sum <scene_epsilon )
       {  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        continue;
       } 


      int lightZone;
      int iter_num = iterNum;
      for(int iter=0;iter<iter_num;iter++)
    {
           

      //first stage sampling
      {
        unsigned int b_val = tea<16>(screen.x*launch_index.y+launch_index.x, 0);
        lightZone = randomSampleZoneMatrix(M3_buffer[eyeEndVertex.zoneId],bias_corput(frame * iter_num  +iter,rnd(b_val)));
      }
       

       if(vertex_sampler[lightZone].empty())continue;

       float SOL_rate; 

       //second stage sampling
       BDPTVertex& lightEndVertex = vertex_sampler[lightZone].sample(prd.seed, SOL_rate);
       //if (eyeEndVertex.depth + lightEndVertex.depth == 1)continue;

       //pdf computation
       SOL_rate *= M3_buffer[eyeEndVertex.zoneId].r[lightZone] / M3_buffer[eyeEndVertex.zoneId].sum * light_path_count * iterNum;
        
      if(lightEndVertex.type == DIRECTION || lightEndVertex.type == ENV)
      { 
        float3 res = direction_connect_ZGCBPT(eyeEndVertex,lightEndVertex) / SOL_rate; 
        if(ISINVALIDVALUE(res))
        {
          continue;
        } 
        BDPTResult += res  ;  
        
        
      }  
      else
      { 
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
           length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow   )
        {
          float3 simple_gamma = make_float3(0.0f); 
          #ifdef ZGC_SAMPLE_ON_LUM
          float3 res = connectVertex_ZGCBPT(eyeEndVertex,lightEndVertex) / SOL_rate; 
          #else 
          float3 res = connectVertex_ZGCBPT(eyeEndVertex,lightEndVertex) / (connectRate_2(eyeEndVertex.zoneId,lightEndVertex.zoneId));  
          #endif 
          if(ISINVALIDVALUE(res))
          { 
            continue;
          }
          BDPTResult += res; 
            
      }
      else
      { 
        continue;
      }
      }
      
       
      
    }
      prd.depth++;
#ifdef PRIM_DIRECT_VIEW
      break;
#endif
      //if (prd.depth == 5)break;
      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction; 
  }

   
      
  //radiance is 0 in most cases.
  //only in scenes WITHOUT env light and camera ray get a miss in it's first tracing, we fill the prd.radiance
  result = BDPTResult + prd.radiance;
  //light tracing strategy, equals 0 because we have disabled this strategy.
  result +=  make_float3(LT_result_buffer[launch_index]);
/*computation*/

 
   
  int acc_frame_weight = frame;
#ifdef CONTINUE_RENDERING
  if (frame == 2)
  {
      accum_buffer[launch_index] = standrd_float_buffer[launch_index];
  }
  acc_frame_weight += CONTINUE_RENDERING_BEGIN_FRAME;
#endif


  float4 acc_val = accum_buffer[launch_index];
  if (frame > 2) {
      acc_val = lerp(acc_val, make_float4(result, 0.f),
          1.0f / static_cast<float>(acc_frame_weight-1 ));
  }
  else {
      acc_val = make_float4(result, 0.f);
  } 
  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5)); 

  output_buffer[launch_index] = make_color(make_float3(val)); 
  float diff = 0;
  {
    float min_limit = estimate_min_limit;
    diff += abs(accum_buffer[launch_index].x - standrd_float_buffer[launch_index].x);
    diff += abs(accum_buffer[launch_index].y - standrd_float_buffer[launch_index].y);
    diff += abs(accum_buffer[launch_index].z - standrd_float_buffer[launch_index].z);
    float diff2 = diff;
    diff /= luminance(make_float3(standrd_float_buffer[launch_index])) + min_limit;
    diff *= 100;
    diff = min(int(diff),290);
    false_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0)); 

    diff2 *= 1000;
    diff2 = min(int(diff2),290);

    //write the heat map
    false_mse_buffer[launch_index] = make_color(hsv2rgb(((-int(diff2) + 240)%360),1.0,1.0));
  }
  #ifdef VIEW_HEAT
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));
  #endif

  accum_buffer[launch_index] = acc_val;
#ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  if(EStack.size>1)
  {
    input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
    input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
  }
#endif
    

}

RT_FUNCTION int find_closest_pmfCache(float3 position)
{
  int closest_index = 0;
  float closest_dis2 = dot(Kd_position[0].position - position, Kd_position[0].position - position);
  unsigned int stack[25];
  float dis_stack[25];
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  float block_min = 0.0;
  dis_stack[stack_current] = 0.0;
  push_node( 0 );

  do {
    if(closest_dis2 < block_min)
    {
      node = pop_node();
      block_min = dis_stack[stack_current];
      continue;
    }
    KDPos& currentVDirector = Kd_position[node];
    uint axis = currentVDirector.axis;
    if( !( axis & PPM_NULL ) ) {

      float3 vd_position = currentVDirector.position;
      float3 diff = position - vd_position;
      float distance2 = dot(diff, diff);

      if (distance2 < closest_dis2) {
        closest_dis2 = distance2;
        closest_index = node;
      
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < closest_dis2 ) {
          dis_stack[stack_current] = d*d;
          push_node( (node<<1) + 2 - selector );
        }

        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node();
        block_min = dis_stack[stack_current];
      }
    } else {
      node = pop_node();
      block_min = dis_stack[stack_current];
    }
  } while ( node );
  return closest_index;
}


RT_FUNCTION uint3 find_close3_pmfCache(float3 position)
{
  uint3 closest_index = make_uint3(0,1,2);
  float3 closest_dis2 = make_float3(
    dot(last_kd_position[0].position - position, last_kd_position[0].position - position),
    dot(last_kd_position[1].position - position, last_kd_position[1].position - position),
    dot(last_kd_position[2].position - position, last_kd_position[2].position - position)
    );
  unsigned int stack[25];
  float dis_stack[25];
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  float block_min = 0.0;
  dis_stack[stack_current] = 0.0;
  push_node( 0 );

  do {
    if(fmaxf(closest_dis2) < block_min)
    {
      node = pop_node();
      block_min = dis_stack[stack_current];
      continue;
    }
    KDPos& currentVDirector = last_kd_position[node];
    uint axis = currentVDirector.axis;
    if( !( axis & PPM_NULL ) ) {

      float3 vd_position = currentVDirector.position;
      float3 diff = position - vd_position;
      float distance2 = dot(diff, diff);

      if (distance2 < fmaxf(closest_dis2)) {
        int tmp = node;
        float tmp_d = distance2;
        if(tmp_d<closest_dis2.x)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.x;
          tmp_d = closest_dis2.x;
          closest_index.x = tt;
          closest_dis2.x = ttd;
        }
        if(tmp_d<closest_dis2.y)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.y;
          tmp_d = closest_dis2.y;
          closest_index.y = tt;
          closest_dis2.y = ttd;
        }
        if(tmp_d<closest_dis2.z)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.z;
          tmp_d = closest_dis2.z;
          closest_index.z = tt;
          closest_dis2.z = ttd;
        }
      
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < fmaxf(closest_dis2 )) {
          dis_stack[stack_current] = d*d;
          push_node( (node<<1) + 2 - selector );
        }

        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node();
        block_min = dis_stack[stack_current];
      }
    } else {
      node = pop_node();
      block_min = dis_stack[stack_current];
    }
  } while ( node );
  return closest_index;
}
RT_FUNCTION uint3 find_close3_now(float3 position)
{
  uint3 closest_index = make_uint3(0,1,2);
  float3 closest_dis2 = make_float3(
    dot(Kd_position[0].position - position, Kd_position[0].position - position),
    dot(Kd_position[1].position - position, Kd_position[1].position - position),
    dot(Kd_position[2].position - position, Kd_position[2].position - position)
    );
  unsigned int stack[25];
  float dis_stack[25];
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  float block_min = 0.0;
  dis_stack[stack_current] = 0.0;
  push_node( 0 );

  do {
    if(fmaxf(closest_dis2) < block_min)
    {
      node = pop_node();
      block_min = dis_stack[stack_current];
      continue;
    }
    KDPos& currentVDirector = Kd_position[node];
    uint axis = currentVDirector.axis;
    if( !( axis & PPM_NULL ) ) {

      float3 vd_position = currentVDirector.position;
      float3 diff = position - vd_position;
      float distance2 = dot(diff, diff);

      if (distance2 < fmaxf(closest_dis2)) {
        int tmp = node;
        float tmp_d = distance2;
        if(tmp_d<closest_dis2.x)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.x;
          tmp_d = closest_dis2.x;
          closest_index.x = tt;
          closest_dis2.x = ttd;
        }
        if(tmp_d<closest_dis2.y)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.y;
          tmp_d = closest_dis2.y;
          closest_index.y = tt;
          closest_dis2.y = ttd;
        }
        if(tmp_d<closest_dis2.z)
        {
          int tt = tmp;
          float ttd = tmp_d;
          tmp = closest_index.z;
          tmp_d = closest_dis2.z;
          closest_index.z = tt;
          closest_dis2.z = ttd;
        }
      
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < fmaxf(closest_dis2 )) {
          dis_stack[stack_current] = d*d;
          push_node( (node<<1) + 2 - selector );
        }

        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node();
        block_min = dis_stack[stack_current];
      }
    } else {
      node = pop_node();
      block_min = dis_stack[stack_current];
    }
  } while ( node );
  return closest_index;
}
RT_PROGRAM void PMFCaches_launch3()
{
  size_t2 screen = make_uint2(PMFCaches.size().x,PMFCaches.size().y);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  
  float3 result = make_float3( 0.0f );

  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  int valid_depth = 0;
  BDPTVertex eyeEndVertexs[PMF_DEPTH];

    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;
  
    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;
  
    prd.stackP = &EStack;
    prd.throughput = make_float3( 1.0f );
  
    // light from a light source or miss program
    prd.radiance = make_float3( 0.0f );
  
    // next ray to be traced
    prd.origin = make_float3( 0.0f );
    prd.direction = make_float3( 0.0f );
    for(int i=0;i<PMF_DEPTH;i++)
    {
      uint3 pmf_index = make_uint3(launch_index,i);
      PMFCaches[pmf_index ].valid = false;

    }
    for(;;) 
    {
      if ( prd.done || prd.depth >= PMF_DEPTH)
          break;

      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
      rtTrace(top_object, ray, prd);
      
      BDPTVertex & v = EStack.v[(EStack.size - 1) % STACKSIZE];
      if(ISLIGHTSOURCE(v) || !ISVALIDVERTEX(v)||v.depth == 0)
        break;
		
    
        eyeEndVertexs[valid_depth] = v;
        valid_depth++;
      prd.depth++;

      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
    }
  for(int r=0;r<valid_depth;r++)
{
  BDPTVertex &eyeEndVertex = eyeEndVertexs[r];
    uint3 pmf_index = make_uint3(launch_index,r);
    PMFCaches[pmf_index ].valid = true;
    PMFCaches[pmf_index ].sum = 0.0f;
    for(int i=0;i<light_vertex_count;i++)
    {
      PMFCaches[pmf_index ].r[i] = 0.0f;
    }

  /* pcpt code */
  optix::float3 BDPTResult = make_float3(0.0f);


  PMFCaches[pmf_index ].position = eyeEndVertex.position;
  PMFCaches[pmf_index ].in_direction = normalize(eyeEndVertex.position - eyeEndVertex.lastPosition);
  PMFCaches[pmf_index ].normal   = eyeEndVertex.normal;
  for(int j=0;j<light_vertex_count;j++)
  {
    int index = j;
    BDPTVertex &lightEndVertex = LVC[index];
    float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
    float3 connectDir = normalize(connectVec);

    PerRayData_shadow prd_shadow;
    prd_shadow.inShadow = false;
    optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, 1, scene_epsilon,
      length(connectVec) - scene_epsilon);
    rtTrace(top_object, shadowRay, prd_shadow);
    if (prd_shadow.inShadow)
    {
      PMFCaches[pmf_index ].r[index] = 0.0f;
      continue;
    }
    
    if(lightEndVertex.type == DIRECTION)
    {
      BDPTResult = direction_connect(eyeEndVertex,lightEndVertex);
  
    }
    else{
      BDPTResult = connectVertex_pmf(eyeEndVertex,lightEndVertex);
    }
    float weight = BDPTResult.x + BDPTResult.y + BDPTResult.z;
    //weight = color2luminance(BDPTResult);
    PMFCaches[pmf_index ].r[j] = weight;
    PMFCaches[pmf_index ].sum += weight;
  }

  for(int i=0;i<LIGHT_VERTEX_NUM;i++)
  {
    if(i > 0)
    {
      PMFCaches[pmf_index ].m[i] = PMFCaches[pmf_index ].r[i] + PMFCaches[pmf_index ].m[i - 1];
    }
    else
    {
      PMFCaches[pmf_index ].m[i] = PMFCaches[pmf_index ].r[0];
    }
  }
}
 
}


RT_PROGRAM void PMFCaches_launch2()
{
  size_t2 screen = make_uint2(PMFCaches.size().x,PMFCaches.size().y);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  uint3 pmf_index = make_uint3(launch_index,0);

  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  
  float3 result = make_float3( 0.0f );

  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  bool vertexFound = false;
  BDPTVertex eyeEndVertex;
  int loopTime = 0;
  while(!vertexFound&&loopTime<20)
  {
    loopTime++;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;
  
    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;
  
    prd.stackP = &EStack;
    prd.throughput = make_float3( 1.0f );
  
    // light from a light source or miss program
    prd.radiance = make_float3( 0.0f );
  
    // next ray to be traced
    prd.origin = make_float3( 0.0f );
    prd.direction = make_float3( 0.0f );
    float selectRate = 0.7f;
    float lastFlux = -1.0f;
    float r1 = rnd(prd.seed);
    for(;;) 
    {
      if ( prd.done || prd.depth >= max_depth)
          break;

      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
      rtTrace(top_object, ray, prd);
      
      BDPTVertex & v = EStack.v[(EStack.size - 1) % STACKSIZE];
      if(ISLIGHTSOURCE(v) || !ISVALIDVERTEX(v)||v.depth == 0)
        break;
		
      float flux = v.flux.x + v.flux.y + v.flux.z;
      if(lastFlux>0.0f)
      {
         selectRate *= flux / lastFlux;
      }
      lastFlux = flux;
    
      float r1 = rnd(prd.seed);
      if(r1<selectRate)
      {
        eyeEndVertex = v;
        vertexFound = true;
        break;
      }
      selectRate /= (1.0f - selectRate);


      prd.depth++;

      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
    }

  }
 
  if(vertexFound)
  {
    PMFCaches[pmf_index ].valid = true;
    PMFCaches[pmf_index ].sum = 0.0f;
    for(int i=0;i<light_vertex_count;i++)
    {
      PMFCaches[pmf_index ].r[i] = 0.0f;
    }
  }
  else
  {
    PMFCaches[pmf_index ].valid = false;
    return;
  }
  /* pcpt code */
  optix::float3 BDPTResult = make_float3(0.0f);
  

  PMFCaches[pmf_index ].position = eyeEndVertex.position;
  PMFCaches[pmf_index ].in_direction = normalize(eyeEndVertex.position - eyeEndVertex.lastPosition);
  PMFCaches[pmf_index ].normal   = eyeEndVertex.normal;
  for(int j=0;j<light_vertex_count;j++)
  {
    int index = j;
    BDPTVertex &lightEndVertex = LVC[index];
    float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
    float3 connectDir = normalize(connectVec);

    PerRayData_shadow prd_shadow;
    prd_shadow.inShadow = false;
    optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, 1, scene_epsilon,
       length(connectVec) - scene_epsilon);
    rtTrace(top_object, shadowRay, prd_shadow);
    if (prd_shadow.inShadow)
    {
      PMFCaches[pmf_index ].r[index] = 0.0f;
      continue;
    }
    if(lightEndVertex.type == DIRECTION)
    {
      BDPTResult = direction_connect(eyeEndVertex,lightEndVertex);
    }
    else
    {
      BDPTResult = connectVertex(eyeEndVertex,lightEndVertex);
    }
    float weight = BDPTResult.x + BDPTResult.y + BDPTResult.z;
    PMFCaches[pmf_index ].r[j] = weight;
    PMFCaches[pmf_index ].sum += weight;
  }

  for(int i=0;i<LIGHT_VERTEX_NUM;i++)
  {
    if(i > 0)
    {
      PMFCaches[pmf_index ].m[i] = PMFCaches[pmf_index ].r[i] + PMFCaches[pmf_index ].m[i - 1];
    }
    else
    {
      PMFCaches[pmf_index ].m[i] = PMFCaches[pmf_index ].r[0];
    }
  }
}

RT_FUNCTION void PMF_redirect(PMFCache& p, int size, unsigned int& seed)
{
    p.size = size;
    return;
    for (int i = 0; i < size; i++)
    {
        p.re_direct[i] = i;
        p.stage_1_pdf[i] = float(size) / light_vertex_count;
    }
    //p.size = size;
    //return;
    access_light_cut(p.position, p.normal, classTree::light_tree_dev, p.re_direct, p.stage_1_pdf, p.size); 
    interior_node_decent(classTree::light_tree_dev, p.re_direct, p.stage_1_pdf, p.size, seed);
}
RT_PROGRAM void PMFCaches_launch()
{
  unsigned int seed = tea<16>(launch_index.x, frame);
  
  BDPTVertex eyeEndVertex;
  PMFCache & c = KDPMFCaches[launch_index.x];
  if(!c.valid)
  {
    return;
  }
  eyeEndVertex.position = c.position;
  eyeEndVertex.normal = c.normal;
  PMF_redirect(c, light_vertex_count, seed);

  c.sum = 0.0f;
  c.sum_refine = 0.0f;
  for(int i=0;i<c.size;i++)
  {
    c.r[i] = 0.0f;
    c.hit_success[i] = false;
  }
  c.shadow_success = 0;
  /* pcpt code */
  optix::float3 BDPTResult = make_float3(0.0f);

  for(int j=0;j<c.size;j++)
  {
      int index = j;//c.re_direct[j];
    //int index = c.re_direct[j];
    BDPTVertex &lightEndVertex = LVC[index];
    if(lightEndVertex.type == DIRECTION|| lightEndVertex.type == ENV)
    {
      BDPTResult = direction_connect_pmf(eyeEndVertex,lightEndVertex);
    }
    else{
      float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
      float3 connectDir = normalize(connectVec);
  
      PerRayData_shadow prd_shadow;
      prd_shadow.inShadow = false;
      optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, 1, scene_epsilon,
        length(connectVec) - scene_epsilon);
      rtTrace(top_object, shadowRay, prd_shadow);

      
      bool mat_brdfFlag = false;
      if (lightEndVertex.depth > 0)
      {
          MaterialParameter mat_light = sysMaterialParameters[lightEndVertex.materialId];
          if (mat_light.brdf == true)
          {
              mat_brdfFlag = true;
          }
      }
      //if (prd_shadow.inShadow|| mat_brdfFlag)
      //{
      //  float min_limit = 0.000001;
      //  c.r[index] = 0.0f + min_limit;
      //  c.sum += min_limit;
      //  c.sum_refine += min_limit;
      //  continue;
      //}
      
      BDPTResult = connectVertex_pmf(eyeEndVertex,lightEndVertex)  ; 
      if (prd_shadow.inShadow)
      { 
          float ratio_k = PCBPT_FAIL_K;
#ifdef KITCHEN
          ratio_k = 0.01;
#endif
          BDPTResult *= ratio_k;

      }
      if (mat_brdfFlag)
      {
          BDPTResult *= 0;
      }
      //BDPTResult /= c.stage_1_pdf[j];
      c.shadow_success += 1;
      c.hit_success[j] = true;
      //c.shadow_success +=1;
    }
    
    float weight = BDPTResult.x + BDPTResult.y + BDPTResult.z;
    //weight = color2luminance(BDPTResult);
    c.r[j] = weight;
    c.sum += weight;
    if(lightEndVertex.depth!=0)
      c.sum_refine +=weight;
  }

  //if(c.shadow_success == 0)
  //{
  //  for(int i=0;i<light_vertex_count;i++)
  //  {
  //    c.r[i] = 0.01;
  //    c.sum += c.r[i];
  //  }
  //}
  for(int i=0;i< c.size;i++)
  {
    if(i > 0)
    {
      c.m[i] = c.r[i] + c.m[i - 1];
    }
    else
    {
      c.m[i] = c.r[0];
    }
  }
  if(KD_SET&&frame>0)
  {
    uint3 kdQ_idx = find_close3_pmfCache(c.position); 
    float QLast = (last_kd_position[kdQ_idx.x].Q + last_kd_position[kdQ_idx.y].Q + last_kd_position[kdQ_idx.z].Q)/3 ;
    c.Q = lerp(QLast,c.sum,1.0f / (frame + 1));

    float3 shadow_success_diff = make_float3(float(last_kd_position[kdQ_idx.x].shadow_success),
        float(last_kd_position[kdQ_idx.y].shadow_success), float(last_kd_position[kdQ_idx.z].shadow_success));
    float aver_ss = (shadow_success_diff.x + shadow_success_diff.y + shadow_success_diff.z) / 3;
    float var_ss = pow(shadow_success_diff.x -aver_ss,2)+pow(shadow_success_diff.y -aver_ss,2)+pow(shadow_success_diff.z -aver_ss,2);
    var_ss /= (aver_ss + 0.01);
    //if((last_kd_position[kdQ_idx.x].shadow_success == 0 ||last_kd_position[kdQ_idx.y].shadow_success == 0||last_kd_position[kdQ_idx.z].shadow_success == 0) &&aver_ss>0.1)
    //{
    //  var_ss += 100;
    //}
    c.variance = var_ss;
    float var_last = (last_kd_position[kdQ_idx.x].Q_variance+last_kd_position[kdQ_idx.y].Q_variance+last_kd_position[kdQ_idx.z].Q_variance)/3;
    c.Q_variance = lerp(var_last,c.variance,1.0f / (frame + 1));
//    c.Q = QLast;
    //rtPrintf("%d %f %d\n",launch_index.x,c.Q / QLast,Kd_position.size());
  }
  else
  {
    c.Q = c.sum;
    c.variance = 0;
    c.Q_variance = 0;
  }
}
 


RT_PROGRAM void PCBPT_pinhole_camera()
{
  
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
 
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;
  // These represent the current shading state and will be set by the closest-hit or miss program

  // attenuation (<= 1) from surface interaction.
  prd.throughput = make_float3( 1.0f );

  // light from a light source or miss program
  prd.radiance = make_float3( 0.0f );

  // next ray to be traced
  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs. 
  for(;;) {

#ifndef UNBIAS_RENDERING
    if ( prd.done || prd.depth >= max_depth)
    break;
 #else
 if ( prd.done|| prd.depth >= 20)
    break;
 #endif

      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }

      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {
        
      #ifdef PCBPT_STANDARD_MIS
      D_reset_straight(EStack,EStack.size);
      #endif
        float &inver_weight = EStack.v[(EStack.size - 1) % STACKSIZE].RMIS_pointer;
        inver_weight -=1;
        float k = average_light_length;
        
    #ifdef PCBPT_MIS
        k = 1;
#endif
        inver_weight /= k;
        inver_weight +=1;
       // if(EStack.v[(EStack.size - 1) % STACKSIZE].depth==3)
          BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
        EStack.size --;
        break;
      }

      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE];
      if(!ISVALIDVERTEX(eyeEndVertex))
        break;
      //PMFCache &pmf = KDPMFCaches[find_closest_pmfCache(eyeEndVertex.position)];

#ifdef INDIRECT_ONLY
      {
        int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
        LightParameter light = sysLightParameters[index];
        LightSample lightSample;
        sysLightSample[light.lightType](light, prd, lightSample);
    
        float lightPdf = 1.0  / sysNumberOfLights * lightSample.pdf; 
        BDPTVertex  lightVertex  ; 
        lightVertex.position = lightSample.surfacePos;
        lightVertex.normal = lightSample.normal;
        lightVertex.flux = lightSample.emission;
        lightVertex.pdf = lightPdf;
        lightVertex.singlePdf = lightPdf;
        lightVertex.isOrigin = true;
        lightVertex.isBrdf = false;
        lightVertex.depth =  0;
        //lightVertex.zoneId = SUBSPACE_NUM - index - 1;
        lightVertex.zoneId = lightSample.zoneId;
        lightVertex.type = light.lightType;  
        
        lightVertex.d = average_light_length;
        #ifdef PCBPT_MIS
            lightVertex.d = 1;
        #endif
        #ifdef ZGCBPT
            lightVertex.d = 1;
        #endif
        if(light.lightType == DIRECTION||light.lightType == ENV)
        { 
          lightVertex.d = light.lightType == DIRECTION?0.0:lightVertex.d;
          lightVertex.type = DIRECTION;
        }

        BDPTVertex &lightEndVertex = lightVertex;

        if(lightEndVertex.type == DIRECTION)
        {
          #ifdef PCBPT_STANDARD_MIS
          D_reset(EStack,lightEndVertex,EStack.size);
          #endif
          BDPTResult += direction_connect(eyeEndVertex,lightEndVertex)  ;
        
        }  
        else{
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
          length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow )
        {
          #ifdef PCBPT_STANDARD_MIS
          D_reset(EStack,lightEndVertex,EStack.size);
          #endif
          float3 eval = connectVertex(eyeEndVertex,lightEndVertex);
  
          //eval = make_float3(abs(pmf.normal.x));
          if(!ISINVALIDVALUE(eval))
            BDPTResult += eval; 
        }
      }
    }
#endif 
      for(int iter = 0;iter<iterNum;iter++)
      {
        uint3 c3;
        int pmf_idx;
        bool use_uniform = false;
  #ifdef PCBPT
  #ifdef KD_3 
        c3 = eyeEndVertex.pmf_kd3;
        

        pmf_idx = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * 3)), 0, 2);
        //BDPTResult = make_float3((KDPMFCaches[c3.x].Q_variance + KDPMFCaches[c3.y].Q_variance + KDPMFCaches[c3.z].Q_variance) / 1.0);
        //break;
        if(pmf_idx == 0)
          pmf_idx = c3.x;
        else if(pmf_idx == 1)
          pmf_idx = c3.y;
        else if(pmf_idx == 2)
          pmf_idx = c3.z; 
        PMFCache &pmf = rnd(prd.seed) < UNIFORM_GUIDE_RATE? KDPMFCaches[virtual_pmf_id]: KDPMFCaches[pmf_idx];
  #else
        PMFCache &pmf = KDPMFCaches[eyeEndVertex.pmf_id];
  #endif
        #else
          pmf_idx = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * 3)), 0, 2);
          pmf_idx = find_closest_pmfCache(eyeEndVertex.position);
          PMFCache &pmf = KDPMFCaches[pmf_idx ];
        #endif

        int index;//= randomSelectVertexFrstage_1_pdfomPMFCache(pmf,rnd(prd.seed));
        float lightVertexSelectPdf;
        float apply_rmis_ratio = 1;
        
        BDPTVertex lightEndVertex;// = LVC[index];
        //BDPTVertex &lightEndVertex = LVC[index];
        //if(false&&dot(eyeEndVertex.normal,pmf.normal)<0.1f)
        //{      
        //  index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * light_vertex_count)), 0, light_vertex_count - 1);
        //  lightVertexSelectPdf = 1.0f / light_vertex_count;
        //}       
        { 
            //index = randomSelectVertexFromPMFCache(pmf,rnd(prd.seed));
            //lightVertexSelectPdf = pmf.r[index] / pmf.sum;
            lightEndVertex = pmf.sample(&LVC[0], rnd(prd.seed), lightVertexSelectPdf, index);
          //else
          //{   
          //  index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * light_vertex_count)), 0, light_vertex_count - 1);
            //use_uniform = true;
          //  lightEndVertex = LVC[index];
          //  lightVertexSelectPdf = 1.0 / light_vertex_count;

            
            //float ris_lum = RIS_light_subpath_lum(LVC[index],eyeEndVertex.position,eyeEndVertex.normal);
            //apply_rmis_ratio = PCBPT_UNIFORM_RIS_WEIGHT() / PCBPT_RIS_WEIGHT(ris_lum,c3);
          //}
     
            //pdf recompute
            if (false)
            {
                float ris_lum = RIS_light_subpath_lum(lightEndVertex, eyeEndVertex.position, eyeEndVertex.normal);
                apply_rmis_ratio = PCBPT_RIS_WEIGHT(ris_lum, pmf) / PCBPT_RIS_WEIGHT(ris_lum, c3);

            }
            else
            {
#ifdef KD_3
                lightVertexSelectPdf = (KDPMFCaches[c3.x].r[index] / KDPMFCaches[c3.x].sum + KDPMFCaches[c3.y].r[index] / KDPMFCaches[c3.y].sum +
                    KDPMFCaches[c3.z].r[index] / KDPMFCaches[c3.z].sum) / 3;
                if (isinf(lightVertexSelectPdf))
                {
                    rtPrintf("error\n");
                }
                lightVertexSelectPdf = lightVertexSelectPdf * (1.0f - UNIFORM_GUIDE_RATE) + 1.0f / light_vertex_count * UNIFORM_GUIDE_RATE;
#else
                lightVertexSelectPdf = pmf.r[index] / pmf.sum * (1.0f - UNIFORM_GUIDE_RATE) + 1.0f / light_vertex_count * UNIFORM_GUIDE_RATE;
#endif 

            }
        }

        if(lightEndVertex.type == DIRECTION|| lightEndVertex.type == ENV)
        {
          #ifdef PCBPT_STANDARD_MIS
          D_reset(EStack,lightEndVertex,EStack.size);
          #endif 
          BDPTResult += direction_connect(eyeEndVertex,lightEndVertex) * average_light_length/ (light_vertex_count * lightVertexSelectPdf)/ iterNum * apply_rmis_ratio;
          //prd.depth++;

          // Update ray data for the next path segment
          //ray_origin = prd.origin;
          //ray_direction = prd.direction;
          continue;
        }  
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
          length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow )
        {

        #ifdef PCBPT_STANDARD_MIS
        D_reset(EStack,lightEndVertex,EStack.size);
        #endif 
          float3 eval = connectVertex(eyeEndVertex,lightEndVertex,lightVertexSelectPdf) / (light_path_count * lightVertexSelectPdf) * apply_rmis_ratio;
        #ifdef NOISE_DISCARD
          if(luminance(eval)>20)
          {
           // eval = make_float3(0.0);
          }
          if(pmf_idx == 3||KDPMFCaches[c3.x].shadow_success == 0 ||KDPMFCaches[c3.y].shadow_success == 0 ||KDPMFCaches[c3.z].shadow_success == 0 )
          {
            if(lightVertexSelectPdf * light_path_count < 1.0 / light_vertex_count * light_path_count)
            {
              eval = make_float3(0.0f);
            }
            //rtPrintf("%f\n",1.0f / (lightVertexSelectPdf * light_path_count));
          }
        #endif  
          if(!ISINVALIDVALUE(eval) )
            BDPTResult += eval / iterNum;  
        }
    }
    //break;
      prd.depth++;

      #ifdef PRIM_DIRECT_VIEW
      break;
#endif
      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
  }




  result = BDPTResult + make_float3(LT_result_buffer[launch_index]) ;
  result += prd.radiance;
  
/*computation*/



  float4 acc_val = accum_buffer[launch_index];
  if(frame >2  ) {
    acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame-1 ) );
  } else {
    acc_val = make_float4( result, 0.f );
  }

  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
  //float4 val = LinearToSrgb(acc_val);

  output_buffer[launch_index] = make_color(make_float3(val));
  accum_buffer[launch_index] = acc_val;
  
  float diff = 0;
  {
    float min_limit = estimate_min_limit;
    diff += abs(accum_buffer[launch_index].x - standrd_float_buffer[launch_index].x);
    diff += abs(accum_buffer[launch_index].y - standrd_float_buffer[launch_index].y);
    diff += abs(accum_buffer[launch_index].z - standrd_float_buffer[launch_index].z);
    float diff2 = diff;
    diff /= luminance(make_float3(standrd_float_buffer[launch_index])) + min_limit;
    diff *= 100;
    diff = min(int(diff),290);
    false_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));

    diff2 *= 1000;
    diff2 = min(int(diff2),290);
    false_mse_buffer[launch_index] = make_color(hsv2rgb(((-int(diff2) + 240)%360),1.0,1.0));
  }
  #ifdef VIEW_HEAT
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));
  #endif


#ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  if(EStack.size>1)
  {
    input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
    input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
  }
#endif
}


rtDeclareVariable(uint,         EVC_height, , )={40};
rtDeclareVariable(uint,         EVC_width, , )={40};
rtDeclareVariable(uint,         EVC_max_depth, ,) = {10};
rtDeclareVariable(uint,         EVC_frame, ,) = {0};
RT_PROGRAM void EVC_launch()
{
  size_t2 screen = make_uint2(EVC_width,EVC_height);
//  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, EVC_frame + frame+55);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, EVC_frame + frame+55421187);
  uint launchBase = (screen.x*launch_index.y+launch_index.x) * EVC_max_depth;
  uint evc_num = 0;
  for(int i=0;i<EVC_max_depth;i++)
  {
    raw_LVC[launchBase + i].valid = false;
  }
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  
  float3 result = make_float3( 0.0f );

  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs. 
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
   
    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;
  
    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;
  
    prd.stackP = &EStack;
    prd.throughput = make_float3( 1.0f );
  
    // light from a light source or miss program
    prd.radiance = make_float3( 0.0f );
  
    // next ray to be traced
    prd.origin = make_float3( 0.0f );
    prd.direction = make_float3( 0.0f );
    float selectRate = 0.1f;
    float lastFlux = -1.0f;
 
    bool hit_next = true;
    for(;;) 
    {
      if ( prd.done || prd.depth >= EVC_max_depth)
          break;
      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );

      hit_next = false;
      int t_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(t_depth < EStack.size)
      {
        hit_next = true;
      }

      BDPTVertex & v = EStack.v[(EStack.size - 1) % STACKSIZE];
      if(ISLIGHTSOURCE(v) || !ISVALIDVERTEX(v)||v.depth == 0)
        break;
		
        raw_LVC[launchBase + evc_num].valid = true;
        raw_LVC[launchBase + evc_num].v = v;
         
        if (gamma_need_dense)
        {

            int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * SVM_Target_buffer.size())), 0, SVM_Target_buffer.size() - 1);

            BDPTVertex& target = SVM_Target_buffer[index];
            if (!isVisible(v.position, target.position))
            {
                raw_LVC[launchBase + evc_num].valid = false;
            }
        }

        evc_num++;

      prd.depth++;

      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
    
    }

  


}

 RT_PROGRAM void format_transform()
 {
   #ifdef USE_DENOISER
     //float4 val = LinearToSrgb(ToneMap(denoised_buffer[launch_index], 1.5));
     output_buffer[launch_index] = make_color(make_float3(denoised_buffer[launch_index]));
   
  #endif

   
 }

 
 RT_PROGRAM void unbias_pt_pinhole_camera()
 {
   size_t2 screen = output_buffer.size();
   unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame + 1800);
   //unsigned int seed = tea<16>((screen.x/10) + (launch_index.y /50) * 10000, frame);
 
   // Subpixel jitter: send the ray through a different position inside the pixel each time,
   // to provide antialiasing.
   float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);
 
   float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
   float3 ray_origin = eye;
   float3 ray_direction = normalize(d.x*U + d.y*V + W);
 
   PerRayData_radiance prd;
   prd.depth = 0;
   prd.seed = seed;
   prd.done = false;
   prd.pdf = 0.0f;
   prd.specularBounce = false;
 
   // These represent the current shading state and will be set by the closest-hit or miss program
 
   // attenuation (<= 1) from surface interaction.
   prd.throughput = make_float3( 1.0f );
 
   // light from a light source or miss program
   prd.radiance = make_float3( 0.0f );
 
   // next ray to be traced
   prd.origin = make_float3( 0.0f );
   prd.direction = make_float3( 0.0f );
   float3 result = make_float3( 0.0f );
 
   // Main render loop. This is not recursive, and for high ray depths
   // will generally perform better than tracing radiance rays recursively
   // in closest hit programs.
   for(;;) {
#ifndef UNBIAS_RENDERING
        if ( prd.done || prd.depth >= max_depth)
        break;
#else
        if ( prd.done )
        break;
#endif
       optix::Ray ray(ray_origin, ray_direction, /*ray type*/ PT_RR_RAY, scene_epsilon );
       rtTrace(top_object, ray, prd);
  
        if(prd.depth>50)
        {
          rtPrintf("errrrrr %d\n",prd.seed);
          break;
        }
       prd.depth++;
 
       // Update ray data for the next path segment
       ray_origin = prd.origin;
       ray_direction = prd.direction;
   }
   result = prd.radiance;
 
 
 
 
   float4 acc_val = accum_buffer[launch_index];
   if( frame > 0 ) {
     acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame+1 ) );
   } else {
     acc_val = make_float4( result, 0.f );
   }
 
   float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
   //float4 val = LinearToSrgb(acc_val);
 
   output_buffer[launch_index] = make_color(make_float3(val));
   accum_buffer[launch_index] = acc_val;

   #ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  
  
#endif
 }


 RT_PROGRAM void ZGCBPT_V2_pinhole_camera()
{
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;
  // These represent the current shading state and will be set by the closest-hit or miss program

  // attenuation (<= 1) from surface interaction.
  prd.throughput = make_float3( 1.0f );

  // light from a light source or miss program
  prd.radiance = make_float3( 0.0f );

  // next ray to be traced
  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  int prim_zone =0;
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for(;;) {
    #ifndef UNBIAS_RENDERING
       if ( prd.done || prd.depth >=  max_depth)
       break;
    #else
    if ( prd.done)
       break;
    #endif
      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
     
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }
      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {
        float &inver_weight = EStack.v[(EStack.size - 1) % STACKSIZE].RMIS_pointer;
        inver_weight -=1;
        #ifndef ZGCBPT
        inver_weight /= average_light_length;
        #endif
        //inver_weight *= 100;
        inver_weight +=1;
        
        BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
        EStack.size --;
        break;
      }
      
      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE]; 
      if(!ISVALIDVERTEX(eyeEndVertex))
        {
          break;
        }

        
      if(M2_buffer[eyeEndVertex.zoneId].sum <scene_epsilon )
       {  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        continue;
       } 
       // Update ray data for the next path segment
      int lightZone;
      int iter_num = 1;
      for(int iter=0;iter<iter_num;iter++)
    {
      float3 TRAIN_SQ = make_float3(0.0);
      if(eyeEndVertex.depth != 1)
      {
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],rnd(prd.seed));
      }
      else
      {
        unsigned int b_val = tea<16>(screen.x*launch_index.y+launch_index.x, 0);
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],bias_corput(frame * iter_num  +iter,rnd(b_val)));
      } 
      if(eyeEndVertex.depth == 1)
      {
        UberZoneLVC &z = uberLVC[lightZone];
        if(z.realSize>0)
        {

          int validSize = min(z.realSize,z.maxSize);
	        int u_index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * validSize)), 0, validSize - 1);
          UberLightVertex &u = z.v[u_index];

            
          if(isVisible(eyeEndVertex.position,u.position))
              {
                float3 uber_res = connectVertex_uber(eyeEndVertex,u)  ;
                if(!ISINVALIDVALUE(uber_res)&& luminance(uber_res)<10)
                  {
//                    if(launch_index.x<800)
  //                    rtPrintf("%d %f %f\n",lightZone,zone_var_get(lightZone),M2_buffer[eyeEndVertex.zoneId].r[lightZone] / M2_buffer[eyeEndVertex.zoneId].sum * SUBSPACE_NUM);
                    BDPTResult += uber_res; 
                    TRAIN_SQ = uber_res;
                                
                    {
                      eyeResultRecord &record = result_record_buffer[make_uint3(launch_index,0)];
                      record.valid = true;
                      record.result = TRAIN_SQ ;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
                      if(luminance(TRAIN_SQ) > 1)
                        record.result = make_float3(1.0,0.0,0.0); 
                      record.eyeZone = eyeEndVertex.zoneId;
                      record.lightZone = lightZone;
                    }
                  }
              }
        } 
      } 
      
      BDPTVertex &lightEndVertex = randomSampleZoneLVC(zoneLVC[lightZone],rnd(prd.seed)); 
      


      if(lightEndVertex.type == DIRECTION)
      {
        float3 res = direction_connect(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
        if(ISINVALIDVALUE(res))
        {
          prd.depth++;
          // Update ray data for the next path segment
          ray_origin = prd.origin;
          ray_direction = prd.direction;
          continue;
        } 
        TRAIN_SQ = res;
      }  
      else
      { 
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
           length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow  )
        {
          //rtPrintf("%f\n",light_path_sum / connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
           
          float3 TSQ = connectVertex(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
       
          if(ISINVALIDVALUE(TSQ))
          {
            prd.depth++;
            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
            continue;
          } 
          TRAIN_SQ = TSQ;
      }
      else
      {
        continue;
      }
      #ifdef EYE_DIRECT 
        float max_limit = 100.0;
        float change_rate = 0.2;
        eyeResultRecord &record = result_record_buffer[make_uint3(launch_index,0)];
      
        if(record.valid == false)
        {
          record.result = TRAIN_SQ ;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > max_limit)
            record.result = make_float3(max_limit,0.0,0.0);
          record.valid = true;
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightEndVertex.zoneId;
        }
        else if(rnd(prd.seed)>change_rate)
        {
          record.result /= 1 - change_rate;
        }
        else
        {
          record.result = TRAIN_SQ / change_rate;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > max_limit)
            record.result = make_float3(max_limit,0.0,0.0)/ change_rate; 
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightEndVertex.zoneId;
        }
      #endif

      }
    }
      prd.depth++; 
      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
  }

 
      

  result = BDPTResult + prd.radiance;
  
/*computation*/


  float4 acc_val = accum_buffer[launch_index];
  if(  frame > 0 ) {
    acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame+1 ) );
  } else {
    acc_val = make_float4( result, 0.f );
  }

  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
  //float4 val = LinearToSrgb(acc_val);

  output_buffer[launch_index] = make_color(make_float3(val));
  
  float diff = 0;
  {
    float min_limit = estimate_min_limit;
    diff += abs(accum_buffer[launch_index].x - standrd_float_buffer[launch_index].x);
    diff += abs(accum_buffer[launch_index].y - standrd_float_buffer[launch_index].y);
    diff += abs(accum_buffer[launch_index].z - standrd_float_buffer[launch_index].z);
    float diff2 = diff;
    diff /= luminance(make_float3(standrd_float_buffer[launch_index])) + min_limit;
    diff *= 100;
    diff = min(int(diff),290);
    false_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));

    diff2 *= 1000;
    diff2 = min(int(diff2),290);
    false_mse_buffer[launch_index] = make_color(hsv2rgb(((-int(diff2) + 240)%360),1.0,1.0));
  }
  #ifdef VIEW_HEAT
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));
  #endif

  accum_buffer[launch_index] = acc_val;
#ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  if(EStack.size>1)
  {
    input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
    input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
  }
#endif
}



RT_PROGRAM void Uber_vertex_C()
{
  if(uberLVC[launch_index.x].realSize <= launch_index.y)
    return;
  UberLightVertex &U = uberLVC[launch_index.x].v[launch_index.y];
  
  unsigned int seed = tea<15>( launch_index.y *UBER_VERTEX_NUM +launch_index.x, frame);
  float3 normal = U.normal;
  float3 position = U.position;

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye; 
  EStack.v[0].depth = 0;
   

  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = 0;
  prd.stackP = &EStack;

  optix::Ray infoRay(position + scene_epsilon * (5) * normal  , -normal,  BDPTRay, scene_epsilon );
 rtTrace(top_object, infoRay, prd);
 normal = EStack.v[1].normal;
 U.normal = normal;
 U.color = EStack.v[1].color;
 U.materialId = EStack.v[1].materialId;
 U.inBrdf = EStack.v[1].inBrdf;
 U.size = 0;
 int vain_count = 0;
  for(int i=0;i<UberWidth;i++)
  {
    SubLightVertex & v = U.son[U.size];
    unsigned int b_val = tea<16>( launch_index.y *UBER_VERTEX_NUM+ launch_index.x,frame);
    int lightZone = randomSampleZoneMatrix(M2_buffer[U.zoneId],bias_corput(i,rnd(b_val)));

    if(zoneLVC[lightZone].size==0)
    {
        vain_count++;
        continue;
    }
    BDPTVertex & lightEndVertex = randomSampleZoneLVC(zoneLVC[lightZone],rnd(prd.seed));
    
    if(isVisible(position,lightEndVertex.position))
    {
      float3 connectVec = position - lightEndVertex.position;
      float3 connectDir = normalize(connectVec);
      float len2 = dot(connectVec,connectVec);
      float G = abs(dot(connectDir,lightEndVertex.normal)) * abs(dot(connectDir,normal))/len2;
      if(lightEndVertex.depth==0)
      {
        v.contri = G * lightEndVertex.flux;
        v.origin_pdf = 1.0 / M_PIf *G; 
      }
      else
      {
        MaterialParameter mat = sysMaterialParameters[lightEndVertex.materialId];
        mat.color = lightEndVertex.color;
        float3 f = DisneyEval(mat,lightEndVertex.normal,
          normalize(lightEndVertex.lastPosition - lightEndVertex.position),
          connectDir);
        v.contri = G * f * lightEndVertex.flux;
        v.origin_pdf = DisneyPdf(mat,lightEndVertex.normal,
          normalize(lightEndVertex.lastPosition - lightEndVertex.position),
          connectDir,lightEndVertex.position) * abs(dot(connectDir,normal))/len2 * fmaxf(mat.color);
      }
      v.distance = length(connectVec);
      v.pdf = lightEndVertex.pdf ;//*uber_pdf_1(U.zoneId,lightEndVertex.zoneId);//* connectRate(U.zoneId,lightEndVertex.zoneId) * UberWidth; 
      v.dir = normalize(lightEndVertex.position - position);
      v.zoneId = lightEndVertex.zoneId;
      v.lastZoneId = lightEndVertex.lastZoneId;
      v.pdf_00 = lightEndVertex.singlePdf;
      v.lastNormalProjection = abs(dot(connectDir,lightEndVertex.normal));
       
      
      v.depth = lightEndVertex.depth + 1;
      U.size++;
    }
  } 
}



__device__ float var_get(float *p,int size ,int zoneId = 0)
{
  float sum = 0;
  for(int i=0;i<size;i++)
  {
    sum += p[i];
  }
  float aver = sum / size;
  float var = 0;
  for(int i=0;i<size;i++)
  {
    float diff = (aver - p[i]);
    var += diff * diff;
  }
  var /= size * sum * sum;
  if(true||var > 240)
  {
    float min_distance2 = 100000000;
    float md_e = 0;
    float max_e = 0.0;
    float me_d = 0;
    UberZoneLVC &z = uberLVC[zoneId];
    for(int i=0;i<min(z.realSize,z.maxSize);i++)
    {
      UberLightVertex &U = z.v[i];
      for(int j = 0;j < U.size;j++)
      {
        SubLightVertex &v = U.son[j];
        if(min_distance2 > v.distance * v.distance)
        {
          min_distance2 = v.distance * v.distance;
          md_e = luminance(v.contri / v.pdf);
        }
        if(max_e<luminance(v.contri / v.pdf))
        {
          max_e = luminance(v.contri / v.pdf);
          me_d = v.distance * v.distance;
        }
      }
    }
    for(int i=0;i<size;i++)
    {
      rtPrintf("%f %f   %d  |%f %f |%f %f\n",aver,p[i], zoneId,min_distance2,md_e,me_d,max_e);
    }
  }
  return var;
}
__device__ float zone_var_get(int zoneId)
{
  UberZoneLVC &z = uberLVC[zoneId]; 
  {
    float e[UBER_VERTEX_PER_ZONE] = {0};
    float t[UBER_VERTEX_PER_ZONE] = {0};
    int s = min(z.realSize,z.maxSize); 
    float sum=0;
    int valid_s = 0;
    for(int i=0;i<s;i++)
    {
      UberLightVertex &U = z.v[i];
      if(U.size == 0)
      {
        e[i] = -1; 
        continue;
      }
      for(int j=0;j<U.size;j++)
      {
        SubLightVertex &v = U.son[j];
        e[i] += luminance(v.contri / v.pdf);
      }
      t[valid_s] = e[i];
      valid_s++;
    }
    float var = var_get(t,valid_s,zoneId); 
    return var;
  }
}


__device__  float3 connectVertex_uber_mis(BDPTVertex &a,BDPTVertex &b)
{ 
  return make_float3(0.0);
}

__device__ float3 connectVertex_uber_mis(BDPTVertex &a,UberLightVertex &b)
{ 
 return make_float3(0.0f);
}
RT_PROGRAM void ZGCBPT_V3_pinhole_camera()
{
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;
  // These represent the current shading state and will be set by the closest-hit or miss program

  // attenuation (<= 1) from surface interaction.
  prd.throughput = make_float3( 1.0f );

  // light from a light source or miss program
  prd.radiance = make_float3( 0.0f );

  // next ray to be traced
  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  int prim_zone =0;
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for(;;) {
    #ifndef UNBIAS_RENDERING
       if ( prd.done || prd.depth >=  max_depth)
       break;
    #else
    if ( prd.done)
       break;
    #endif
      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
     
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }
      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {  
        BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
        EStack.size --;
        break;
      }
      
      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE]; 
      if(!ISVALIDVERTEX(eyeEndVertex))
        {
          break;
        }

        
      if(M2_buffer[eyeEndVertex.zoneId].sum <scene_epsilon )
       {  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        continue;
       } 
       // Update ray data for the next path segment
      int lightZone;
      int iter_num = 1;
      for(int iter=0;iter<iter_num;iter++)
    { 
      float3 TRAIN_SQ = make_float3(0.0);
      if(eyeEndVertex.depth != 1)
      {
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],rnd(prd.seed));
      }
      else
      {
        unsigned int b_val = tea<16>(screen.x*launch_index.y+launch_index.x, 0);
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],bias_corput(frame * iter_num  +iter,rnd(b_val)));
      } 
      if(lightZone < SUBSPACE_NUM - MAX_LIGHT)
      {
        UberZoneLVC &z = uberLVC[lightZone];
        if(z.realSize>0)
        {

          int validSize = min(z.realSize,z.maxSize);
	        int u_index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * validSize)), 0, validSize - 1);
          UberLightVertex &u = z.v[u_index];

            
          if(isVisible(eyeEndVertex.position,u.position))
          {
              float3 uber_res = connectVertex_uber_mis(eyeEndVertex,u);// /pdf_2;
              if(!ISINVALIDVALUE(uber_res))
              { 
                  BDPTResult += uber_res; 
                  TRAIN_SQ = uber_res; 
              }
          }
        } 
      } 
      else
      { 
        BDPTVertex &lightEndVertex = randomSampleZoneLVC(zoneLVC[lightZone],rnd(prd.seed)); 
      
        if(lightEndVertex.type == DIRECTION)
        {
          rtPrintf("direction light is unfinished\n");
          float3 res = direction_connect(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
          if(ISINVALIDVALUE(res))
          {
            prd.depth++;
            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
            continue;
          } 
          TRAIN_SQ = res;
        }
        else
        { 
          float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
          float3 connectDir = normalize(connectVec);
          PerRayData_shadow prd_shadow;
          prd_shadow.inShadow = false;
          optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
             length(connectVec) - scene_epsilon);
          rtTrace(top_object, shadowRay, prd_shadow);
          if (!prd_shadow.inShadow  )
          { 
            float3 TSQ = connectVertex_uber_mis(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId)); 
            if(ISINVALIDVALUE(TSQ))
            {
              prd.depth++;
              // Update ray data for the next path segment
              ray_origin = prd.origin;
              ray_direction = prd.direction;
              continue;
            } 
            TRAIN_SQ = TSQ;
            BDPTResult += TRAIN_SQ;
          } 
        }
      



      }
      #ifdef EYE_DIRECT 
        float max_limit = 100.0;
        float change_rate = 0.2;
        eyeResultRecord &record = result_record_buffer[make_uint3(launch_index,0)];
      
        if(luminance(TRAIN_SQ)<0.00000001)
        {
          continue;
        }
        else if(record.valid == false)
        {
          record.result = TRAIN_SQ ;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > max_limit)
            record.result = make_float3(max_limit,0.0,0.0);
          record.valid = true;
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightZone;
        }
        else if(rnd(prd.seed)>change_rate)
        {
          record.result /= 1 - change_rate;
        }
        else
        {
          record.result = TRAIN_SQ / change_rate;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > max_limit)
            record.result = make_float3(max_limit,0.0,0.0)/ change_rate; 
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightZone;
        }
      #endif

    }
      prd.depth++; 
      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction; 
  }

 
      

  result = BDPTResult + prd.radiance;
  
/*computation*/


  float4 acc_val = accum_buffer[launch_index];
  if(  frame > 0 ) {
    acc_val = lerp( acc_val, make_float4( result, 0.f ), 1.0f / static_cast<float>( frame+1 ) );
  } else {
    acc_val = make_float4( result, 0.f );
  }

  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
  //float4 val = LinearToSrgb(acc_val);

  output_buffer[launch_index] = make_color(make_float3(val));
 
  float diff = 0;
  {
    float min_limit = estimate_min_limit;
    diff += abs(accum_buffer[launch_index].x - standrd_float_buffer[launch_index].x);
    diff += abs(accum_buffer[launch_index].y - standrd_float_buffer[launch_index].y);
    diff += abs(accum_buffer[launch_index].z - standrd_float_buffer[launch_index].z);
    float diff2 = diff;
    diff /= luminance(make_float3(standrd_float_buffer[launch_index])) + min_limit;
    diff *= 100;
    diff = min(int(diff),290);
    false_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));

    diff2 *= 1000;
    diff2 = min(int(diff2),290);
    false_mse_buffer[launch_index] = make_color(hsv2rgb(((-int(diff2) + 240)%360),1.0,1.0));
  }
  #ifdef VIEW_HEAT
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(diff) + 240)%360),1.0,1.0));
  #endif

  accum_buffer[launch_index] = acc_val;
#ifdef USE_DENOISER
  tonemapped_buffer[launch_index] = val;
  if(EStack.size>1)
  {
    input_albedo_buffer[launch_index] = make_float4(EStack.v[1].color,1.0f);
    input_normal_buffer[launch_index] = make_float4(EStack.v[1].normal,1.0f);
  }
#endif
}


rtDeclareVariable(int,           vp_x, , ) = {1499};
rtDeclareVariable(int,           vp_y, , ) = {892};
RT_PROGRAM void sampler_p_visualize()
{

    size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);
    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing.
    float2 subpixel_jitter = frame == 0 ? make_float2(0.0f) : make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);

    BDPTVertex first_eye_v;
    first_eye_v.normal = normalize(U);
    first_eye_v.zoneId = 0;

    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;

    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;

    prd.stackP = &EStack;
    // These represent the current shading state and will be set by the closest-hit or miss program

    // attenuation (<= 1) from surface interaction.
    prd.throughput = make_float3(1.0f);

    // light from a light source or miss program
    prd.radiance = make_float3(0.0f);

    // next ray to be traced
    prd.origin = make_float3(0.0f);
    prd.direction = make_float3(0.0f);
    float3 result = make_float3(0.0f);
    float3 BDPTResult = make_float3(0.0f);
    int prim_zone = 0;
    // Main render loop. This is not recursive, and for high ray depths
    // will generally perform better than tracing radiance rays recursively
    // in closest hit programs.
    for (;;) {

        optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon);

        int origin_depth = EStack.size;
        rtTrace(top_object, ray, prd);
        if (EStack.size == origin_depth)break;

        BDPTVertex& eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE];
        first_eye_v = eyeEndVertex;
        break;
    }

     
    if (launch_index.x == vp_x && launch_index.y == vp_y)
    {
        test_setting[0].v = first_eye_v;
    }
    result = make_float3(LT_result_buffer[launch_index]);

    float4 acc_val = accum_buffer[launch_index];
    if (frame > 0) {
        acc_val = lerp(acc_val, make_float4(result, 0.f), 1.0f / static_cast<float>(frame + 1));
    }
    else {
        acc_val = make_float4(result, 0.f);
    }

    float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
    //float4 val = LinearToSrgb(acc_val); 
    output_buffer[launch_index] = make_color(hsv2rgb(((-int(acc_val.x) + 240) % 360), 1.0, 1.0));
    //output_buffer[launch_index] = make_color(make_float3(val));
    accum_buffer[launch_index] = acc_val;
}
RT_PROGRAM void ZGCBPT_test_pinhole_camera()
{
  size_t2 screen = output_buffer.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);
  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  float2 subpixel_jitter = frame == 0 ? make_float2( 0.0f ) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  BDPTVertex first_eye_v;
  first_eye_v.normal = normalize(U);
  first_eye_v.zoneId = 0;

  BDPTVertexStack EStack;
  EStack.size = 1;
  EStack.v[0].position = eye;
  EStack.v[0].flux = make_float3(1.0f);
  EStack.v[0].pdf = 1.0f;
  EStack.v[0].RMIS_pointer = 0;
  EStack.v[0].normal = ray_direction;
  EStack.v[0].isOrigin = true;
  EStack.v[0].depth = 0;

  EStack.v[1].singlePdf = 1.0f;
  PerRayData_radiance prd;
  prd.depth = 0;
  prd.seed = seed;
  prd.done = false;
  prd.pdf = 0.0f;
  prd.specularBounce = false;

  prd.stackP = &EStack;
  // These represent the current shading state and will be set by the closest-hit or miss program

  // attenuation (<= 1) from surface interaction.
  prd.throughput = make_float3( 1.0f );

  // light from a light source or miss program
  prd.radiance = make_float3( 0.0f );

  // next ray to be traced
  prd.origin = make_float3( 0.0f );
  prd.direction = make_float3( 0.0f );
  float3 result = make_float3( 0.0f );
  float3 BDPTResult = make_float3( 0.0f );
  int prim_zone =0;
  // Main render loop. This is not recursive, and for high ray depths
  // will generally perform better than tracing radiance rays recursively
  // in closest hit programs.
  for(;;) {
    
    #ifndef UNBIAS_RENDERING
       if ( prd.done || prd.depth >=  max_depth)
       break;
    #else
    if ( prd.done)
       break;
    #endif
      optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon );
     
      int origin_depth = EStack.size;
      rtTrace(top_object, ray, prd);
      if(EStack.size == origin_depth)
      {
        //miss
        break;
      }
      if(ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
      {
        float &inver_weight = EStack.v[(EStack.size - 1) % STACKSIZE].RMIS_pointer;
        inver_weight -=1;
        #ifndef ZGCBPT
        inver_weight /= average_light_length;
        #endif
        //inver_weight *= 100;
        inver_weight +=1;
        
        BDPTResult += lightStraghtHit(EStack.v[(EStack.size - 1) % STACKSIZE]);
        EStack.size --;
        break;
      }
      
      BDPTVertex &eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE]; 
      if(!ISVALIDVERTEX(eyeEndVertex))
        {
          break;
        }

        
      if(M2_buffer[eyeEndVertex.zoneId].sum <scene_epsilon )
       {  
        prd.depth++;
  
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        continue;
       } 
       // Update ray data for the next path segment
      int lightZone;
      int iter_num = 1;
      for(int iter=0;iter<iter_num;iter++)
    {
      float3 TRAIN_SQ = make_float3(0.0);
      if(true||eyeEndVertex.depth != 1)
      {
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],rnd(prd.seed));
      }
      else
      {
        unsigned int b_val = tea<16>(screen.x*launch_index.y+launch_index.x, 0);
        lightZone = randomSampleZoneMatrix(M2_buffer[eyeEndVertex.zoneId],bias_corput(frame * iter_num  +iter,rnd(b_val)));
      } 
      if(eyeEndVertex.depth == 1)
      {
        first_eye_v = eyeEndVertex;
        if(frame>10)
        {
          break;
        }
        UberZoneLVC &z = uberLVC[lightZone];
        if(z.realSize>0)
        {
          int validSize = min(z.realSize,z.maxSize);
	        int u_index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * validSize)), 0, validSize - 1);
          UberLightVertex &u = z.v[u_index];
          float pdf_2 =   M2_buffer[eyeEndVertex.zoneId].r[lightZone] / M2_buffer[eyeEndVertex.zoneId].sum * iter_num / z.realSize * u.singlePdf; 
          if(isVisible(eyeEndVertex.position,u.position))
              {
                float3 uber_res = connectVertex_uber(eyeEndVertex,u) /pdf_2;
                if(!ISINVALIDVALUE(uber_res))
                  {
                    BDPTResult += uber_res; 
                    TRAIN_SQ = uber_res;
                                
                    {
                      eyeResultRecord &record = result_record_buffer[make_uint3(launch_index,0)];
                      record.valid = true;
                      record.result = TRAIN_SQ ;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
                      if(luminance(TRAIN_SQ) > 1)
                        record.result = make_float3(1.0,0.0,0.0); 
                      record.eyeZone = eyeEndVertex.zoneId;
                      record.lightZone = lightZone;
                    }
                  }
              }
        } 
      } 
      
      BDPTVertex &lightEndVertex = randomSampleZoneLVC(zoneLVC[lightZone],rnd(prd.seed)); 
      


      if(lightEndVertex.type == DIRECTION)
      {
        float3 res = direction_connect(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
        if(ISINVALIDVALUE(res))
        {
          prd.depth++;
          // Update ray data for the next path segment
          ray_origin = prd.origin;
          ray_direction = prd.direction;
          continue;
        } 
        TRAIN_SQ = res;
      }  
      else
      { 
        float3 connectVec = eyeEndVertex.position - lightEndVertex.position;
        float3 connectDir = normalize(connectVec);
        PerRayData_shadow prd_shadow;
        prd_shadow.inShadow = false;
        optix::Ray shadowRay = optix::make_Ray(lightEndVertex.position, connectDir, ShadowRay, scene_epsilon,
           length(connectVec) - scene_epsilon);
        rtTrace(top_object, shadowRay, prd_shadow);
        if (!prd_shadow.inShadow  )
        {
          //rtPrintf("%f\n",light_path_sum / connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
          
          float3 TSQ = connectVertex(eyeEndVertex,lightEndVertex) / (connectRate(eyeEndVertex.zoneId,lightEndVertex.zoneId));
  
          if(ISINVALIDVALUE(TSQ))
          {
            prd.depth++;
            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;
            continue;
          } 
          TRAIN_SQ = TSQ;
      }
      else
      {
        continue;
      }
      #ifdef EYE_DIRECT 
        float change_rate = 0.2;
        eyeResultRecord &record = result_record_buffer[make_uint3(launch_index,0)];
      
        if(record.valid == false)
        {
          record.result = TRAIN_SQ ;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > 1)
            record.result = make_float3(1.0,0.0,0.0);
          record.valid = true;
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightEndVertex.zoneId;
        }
        else if(rnd(prd.seed)>change_rate)
        {
          record.result /= 1 - change_rate;
        }
        else
        {
          record.result = TRAIN_SQ / change_rate;/// (eyeEndVertex.flux / eyeEndVertex.pdf);
          if(luminance(TRAIN_SQ) > 1)
            record.result = make_float3(1.0,0.0,0.0)/ change_rate; 
          record.eyeZone = eyeEndVertex.zoneId;
          record.lightZone = lightEndVertex.zoneId;
        }
      #endif

      }
    }
      prd.depth++; 
      // Update ray data for the next path segment
      ray_origin = prd.origin;
      ray_direction = prd.direction;
  }

 
  result = make_float3(abs(dot(first_eye_v.normal,normalize(W)))) / 100; 
  if(launch_index.x == vp_x - 1 && launch_index.y == vp_y - 1)
  {
    test_setting[0].vpZone = first_eye_v.zoneId;
    rtPrintf("pointer is locate on surface %d\n",first_eye_v.zoneId);
  }
  if(first_eye_v.zoneId == test_setting[0].vpZone)
  {
    result += make_float3(1.0,1.0,1.0) * 1000000;
  }
  float w = M2_buffer[test_setting[0].vpZone].r[first_eye_v.zoneId] / M2_buffer[test_setting[0].vpZone].sum * SUBSPACE_NUM;
  //w = (zoneLVC[first_eye_v.zoneId].size / 50.0 / (frame+1));
  //w = w > 1 ? 1:w;
  float t = w;
  w *= 10;
  w = min(int(w),280);
  


  UberZoneLVC &z = uberLVC[first_eye_v.zoneId];
  float var_in_zone = 0;
  if(false)
  {
    float e[UBER_VERTEX_PER_ZONE] = {0};
    float t[UBER_VERTEX_PER_ZONE] = {0};
    int s = min(z.realSize,z.maxSize); 
    float sum=0;
    int valid_s = 0;
    for(int i=0;i<s;i++)
    {
      UberLightVertex &U = z.v[i];
      if(U.size == 0)
      {
        e[i] = -1; 
        continue;
      }
      for(int j=0;j<U.size;j++)
      {
        SubLightVertex &v = U.son[j];
        e[i] += luminance(v.contri / v.pdf);
      }
      t[valid_s] = e[i];
      valid_s++;
    }
    float var = var_get(t,valid_s,first_eye_v.zoneId);
    w = var;
  }
  
  result += hsv2rgb(((-int(w) + 240)%360),0.7,0.7) ;
/*computation*/


  if (launch_index.x == vp_x && launch_index.y == vp_y)
  {
      test_setting[0].v = first_eye_v;
  }

  result = make_float3(LT_result_buffer[launch_index]);
  float4 acc_val = accum_buffer[launch_index]; 
//  acc_val = make_float4( result, 0.f );
  if (frame > 0) {
      acc_val = lerp(acc_val, make_float4(result, 0.f), 1.0f / static_cast<float>(frame + 1));
  }
  else {
      acc_val = make_float4(result, 0.f);
  }
  float4 val = LinearToSrgb(ToneMap(acc_val, 1.5));
  //float4 val = LinearToSrgb(acc_val);

  output_buffer[launch_index] = make_color(make_float3(val)); 
  accum_buffer[launch_index] = acc_val; 

  if(abs(float(vp_x - launch_index.x)) + abs(float(launch_index.y - vp_y)) < 30)
  { 
    output_buffer[launch_index] = make_color(make_float3(0.0,0.0,1.0)); 
  }  
  //output_buffer[launch_index] = make_color(make_float3(t));  
}




rtBuffer<PG_training_mat, 1>  PG_mats;
rtDeclareVariable(int, PG_max_step, , );
rtDeclareVariable(int, PG_seed, , ) = {0};
rtDeclareVariable(int, PG_inverse_mat, , ) = {0};
RT_FUNCTION void PG_matRecord(BDPTVertexStack& stack, int stack_depth)
{
    if (stack_depth <= 2)
        return;
    if (stack_depth >= STACKSIZE)
        return; 
    float lightPdf;
    float3 Le_flux; 

    if (stack.v[stack_depth - 1].type != ENV_MISS)
    {
        int light_id = stack.v[stack_depth - 1].materialId;
        LightParameter light = sysLightParameters[light_id];
        lightPdf = 1.0 / sysNumberOfLights *stack.v[stack_depth - 1].pg_lightPdf.x;
        Le_flux = light.emission; 
         
         
    }
    else
    {
        float3 dir = -stack.v[stack_depth - 1].normal;
        lightPdf = 1.0 / sysNumberOfLights * sky.pdf(dir);
        Le_flux = sky.color(dir); 
    }

    float accm_pdf = 1.0;
    float3 accm_flux = Le_flux;
    int accm_idx = 0;
    float l = luminance(stack.v[stack_depth - 1].flux / stack.v[stack_depth - 1].pdf);
    if (isnan(l)||isinf(l))
        return;
    if (PG_inverse_mat == 0 && stack.v[stack_depth - 1].type != ENV_MISS)
    { 
        float3 dir = normalize(stack.v[stack_depth - 2].position - stack.v[stack_depth - 1].position);
        PG_training_mat& m = PG_mats[launch_index.x * PG_max_step + accm_idx];
        m.light_source = true;
        m.valid = false;
        m.uv_light = stack.v[stack_depth - 1].uv;
        m.uv = sky.dir2uv(dir);
        m.lum = luminance(stack.v[stack_depth - 1].flux / stack.v[stack_depth - 1].pdf);
        m.light_id = stack.v[stack_depth - 1].materialId;
        accm_idx++;
    }
    for (int i = stack_depth - 2; i > 0 ; i--)
    {
        if (accm_idx >= PG_max_step)
            continue;  
        BDPTVertex& midVertex = stack.v[i];
        BDPTVertex& nextVertex = stack.v[i+1];
        BDPTVertex& lastVertex = stack.v[i-1];
        float3 in_dir = normalize(lastVertex.position - midVertex.position);
        float3 out_dir = normalize(nextVertex.position - midVertex.position);
        MaterialParameter mat = sysMaterialParameters[midVertex.materialId];
        mat.color = midVertex.color;
        float pdf = DisneyPdf(mat, midVertex.normal, in_dir, out_dir, midVertex.position, true);
        float3 f = DisneyEval(mat, midVertex.normal, in_dir, out_dir) * abs(dot(out_dir, midVertex.normal));
        accm_pdf *= pdf;

        float rr_rate = fmaxf(mat.color);
#ifdef RR_MIN_LIMIT
        rr_rate = max(rr_rate, MIN_RR_RATE); 
#endif
#ifdef RR_DISABLE
        rr_rate = 1.0;
#endif



        if (PG_inverse_mat == 0)
        {
            PG_training_mat& m = PG_mats[launch_index.x * PG_max_step + accm_idx];
            m.valid = true;
            m.light_source = false;
            m.position = stack.v[i].position;
            m.uv = sky.dir2uv(out_dir);
            m.lum = abs(luminance(accm_flux / accm_pdf) *abs(dot(stack.v[i].normal, out_dir)));
            accm_flux *= f/rr_rate;
            accm_idx++;
        }
        else if(i > 1)
        {
            PG_training_mat& m = PG_mats[launch_index.x * PG_max_step + accm_idx];
            m.valid = true;
            m.light_source = false;
            m.position = stack.v[i].position;
            m.uv = sky.dir2uv(in_dir);
            m.lum = abs(luminance(midVertex.flux / midVertex.pdf));
            accm_idx++;
        }
    }
    for (int i = accm_idx; i < PG_max_step; i++)
    { 
        PG_training_mat& m = PG_mats[launch_index.x * PG_max_step + i];
        m.valid = false;
        m.light_source = false;
    }
}

RT_PROGRAM void PG_training()
{ 
    size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, PG_seed);

    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing.
    float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    float2 d = (subpixel_jitter) * 2.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);

    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;

    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;

    prd.stackP = &EStack;
    // These represent the current shading state and will be set by the closest-hit or miss program

    // attenuation (<= 1) from surface interaction.
    prd.throughput = make_float3(1.0f);

    // light from a light source or miss program
    prd.radiance = make_float3(0.0f);

    // next ray to be traced
    prd.origin = make_float3(0.0f);
    prd.direction = make_float3(0.0f);
    float3 result = make_float3(0.0f);
    float3 BDPTResult = make_float3(0.0f);
    float3 PMResult = make_float3(0.0f);
    // Main render loop. This is not recursive, and for high ray depths
    // will generally perform better than tracing radiance rays recursively
    // in closest hit programs.
    for (;;) {
#ifndef UNBIAS_RENDERING
        if (prd.done || prd.depth >= max_depth)
            break;
#else
        if (prd.done)
            break;
#endif

        optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon);
        int origin_depth = EStack.size;
        rtTrace(top_object, ray, prd);
        if (EStack.size == origin_depth)
        {
            //miss
            break;
        }

        if (ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
        {  
            PG_matRecord(EStack, EStack.size);
            break;
        }

        BDPTVertex& eyeEndVertex = EStack.v[(EStack.size - 1) % STACKSIZE];
        if (!ISVALIDVERTEX(eyeEndVertex))
        {
            break;
        }

        prd.depth++;
        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    }
}
RT_PROGRAM void quick_slic()
{
    int tri_id = slic_unlabel_index_buffer[launch_index.x];
    triangleStruct& tri = slic_tris[tri_id];

    float min_d = slic_diff( slic_cluster_center_buffer[0], tri);
    int min_id = 0;
    for (int i = 1; i < slic_cluster_num; i++)
    {
        float d = slic_diff( slic_cluster_center_buffer[i], tri);
        if (d < min_d)
        {
            min_d = d;
            min_id = i;
        }
    }
    slic_label_buffer[launch_index.x] = min_id;

}


RT_PROGRAM void get_OPT_Info()
{ 
    unsigned int seed = tea<16>(launch_index.x, frame + EVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);

    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing.
    float2 subpixel_jitter =  make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    float2 d = subpixel_jitter * 2.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);

    uint2 pixiv_loc = dir2pix(ray_direction);
    float pixiv_lum = luminance(make_float3(standrd_float_buffer[pixiv_loc]));

    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;

    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;

    prd.stackP = &EStack;
    // These represent the current shading state and will be set by the closest-hit or miss program

    // attenuation (<= 1) from surface interaction.
    prd.throughput = make_float3(1.0f);

    // light from a light source or miss program
    prd.radiance = make_float3(0.0f);

    // next ray to be traced
    prd.origin = make_float3(0.0f);
    prd.direction = make_float3(0.0f);
    // Main render loop. This is not recursive, and for high ray depths
    // will generally perform better than tracing radiance rays recursively
    // in closest hit programs.
    for (;;) {
#ifndef UNBIAS_RENDERING
        if (prd.done || prd.depth >= max_depth)
            break;
#else
        if (prd.done)
            break;
#endif

        optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon);
        int origin_depth = EStack.size;
        rtTrace(top_object, ray, prd);
        if (EStack.size == origin_depth)
        {
            //miss
            break;
        }

        if (ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]))
        { 
            break;
        }
        prd.depth++;

        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    }
    if (ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]) && EStack.size >= 3 && EStack.size < OPT_PATH_LENGTH)
    {
        SVM_OPTP_buffer[launch_index.x] = OPT_info_from_path(EStack); 
        SVM_OPTP_buffer[launch_index.x].pixiv_lum = pixiv_lum; 
        //rtPrintf("%f\n", pixiv_lum);
        SVM_OPTP_buffer[launch_index.x].valid = true; 

        //BDPTVertex& v = EStack.v[(EStack.size - 1) % STACKSIZE];
        //rtPrintf("%f\n", luminance(v.flux / v.pdf * v.flux / v.pdf));
    }
    else
    {
        SVM_OPTP_buffer[launch_index.x].valid = false;
    }
}

RT_FUNCTION BDPTVertex& sample_light_vertex(PerRayData_radiance & prd)
{
    int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
    LightParameter light = sysLightParameters[index];
    LightSample lightSample;
    sysLightSample[light.lightType](light, prd, lightSample);

    float lightPdf = 1.0 / sysNumberOfLights * lightSample.pdf;
    BDPTVertex  lightVertex;
    lightVertex.position = lightSample.surfacePos;
    lightVertex.normal = lightSample.normal;
    lightVertex.flux = lightSample.emission;
    lightVertex.pdf = lightPdf;
    lightVertex.singlePdf = lightPdf;
    lightVertex.isOrigin = true;
    lightVertex.isBrdf = false;
    lightVertex.depth = 0;
    //lightVertex.zoneId = SUBSPACE_NUM - index - 1;
    lightVertex.zoneId = lightSample.zoneId;
    lightVertex.type = light.lightType;

    lightVertex.RMIS_pointer = average_light_length;
#ifdef PCBPT_MIS
    lightVertex.RMIS_pointer = 1;
#endif
#ifdef ZGCBPT
    lightVertex.RMIS_pointer = 1;
#endif
    if (light.lightType == DIRECTION || light.lightType == ENV)
    {
        lightVertex.RMIS_pointer = light.lightType == DIRECTION ? 0.0 : lightVertex.RMIS_pointer;
        lightVertex.type = DIRECTION;
    }

    return lightVertex;

}

RT_PROGRAM void get_OPT_Info_NEE()
{
    unsigned int seed = tea<16>(launch_index.x, frame + EVC_frame + CONTINUE_RENDERING_BEGIN_FRAME);

    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing.
    float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    float2 d = subpixel_jitter * 2.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);

    uint2 pixiv_loc = dir2pix(ray_direction);
    float pixiv_lum = luminance(make_float3(standrd_float_buffer[pixiv_loc]));

    BDPTVertexStack EStack;
    EStack.size = 1;
    EStack.v[0].position = eye;
    EStack.v[0].flux = make_float3(1.0f);
    EStack.v[0].pdf = 1.0f;
    EStack.v[0].RMIS_pointer = 0;
    EStack.v[0].normal = ray_direction;
    EStack.v[0].isOrigin = true;
    EStack.v[0].depth = 0;

    EStack.v[1].singlePdf = 1.0f;
    PerRayData_radiance prd;
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;

    prd.stackP = &EStack;
    // These represent the current shading state and will be set by the closest-hit or miss program

    // attenuation (<= 1) from surface interaction.
    prd.throughput = make_float3(1.0f);

    // light from a light source or miss program
    prd.radiance = make_float3(0.0f);

    // next ray to be traced
    prd.origin = make_float3(0.0f);
    prd.direction = make_float3(0.0f);
    // Main render loop. This is not recursive, and for high ray depths
    // will generally perform better than tracing radiance rays recursively
    // in closest hit programs.
    int success_count = 0;
    SVM_OPTP_buffer[launch_index.x].valid = false;
    for (;;) {
#ifndef UNBIAS_RENDERING
        if (prd.done || prd.depth >= max_depth)
            break;
#else
        if (prd.done)
            break;
#endif

        optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon);
        int origin_depth = EStack.size;
        rtTrace(top_object, ray, prd);
        if (EStack.size == origin_depth)
        {
            //miss
            break;
        }

        if (ISLIGHTSOURCE(EStack.v[(EStack.size - 1) % STACKSIZE]) || EStack.size>=OPT_PATH_LENGTH - 1)
        {
            break;
        }

        if (!ISVALIDVERTEX(EStack.v[(EStack.size - 1) % STACKSIZE]))
        {
            break;
        }


        BDPTVertex lightVertex = sample_light_vertex(prd);

        if (isVisible(lightVertex.position, EStack.v[EStack.size - 1].position))
        {
            success_count++;
            if (rnd(prd.seed) < 1.0 / success_count)
            {
                SVM_OPTP_buffer[launch_index.x] = OPT_info_from_path_NEE(EStack,lightVertex);
                SVM_OPTP_buffer[launch_index.x].pixiv_lum = pixiv_lum;
                
                
                if (ISINVALIDVALUE(make_float3(SVM_OPTP_buffer[launch_index.x].contri / SVM_OPTP_buffer[launch_index.x].actual_pdf)))
                {
                    SVM_OPTP_buffer[launch_index.x].contri = 0;
                }
                
                SVM_OPTP_buffer[launch_index.x].actual_pdf /= success_count;

                SVM_OPTP_buffer[launch_index.x].valid = true;
            }
        }

        prd.depth++;

        // Update ray data for the next path segment
        ray_origin = prd.origin;
        ray_direction = prd.direction;
    } 
}

RT_PROGRAM void MLP_forward()
{ 
    int id = launch_index.x + batch_size * batch_id;
    feed_token &p = feed_buffer[id];
    MLP_network& net = MLP_buffer[p.grid_label];
    BP_token& t = BP_buffer[launch_index.x]; 
    
    t.clear();
    float3 current_cos = make_float3(cos(p.position.x * M_PI), cos(p.position.y * M_PI), cos(p.position.z * M_PI));
    float3 current_sin = make_float3(sin(p.position.x * M_PI), sin(p.position.y * M_PI), sin(p.position.z * M_PI));
     
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            //rtPrintf("%d\n", i * 6 + 5);
            t.feature_0[j] += 
                current_cos.x * net.layer_0[i * 6 + 0][j] +
                current_cos.y * net.layer_0[i * 6 + 1][j] +
                current_cos.z * net.layer_0[i * 6 + 2][j] +
                current_sin.x * net.layer_0[i * 6 + 3][j] +
                current_sin.y * net.layer_0[i * 6 + 4][j] +
                current_sin.z * net.layer_0[i * 6 + 5][j];

        }
        current_sin = current_sin * current_cos * 2;
        current_cos = 1 - current_cos * current_cos;
    } 

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            t.feature_1[j] += t.feature_0[i] * net.layer_1[i][j];
        }
    }

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            t.res[j] += t.feature_1[i] * net.layer_2[i][j];
        }
    } 
}


RT_PROGRAM void MLP_backward()
{
    int id = launch_index.x;
    int2 BP_index = MLP_index_L1_buffer[make_uint2(batch_id,id)];      //
    MLP_network& net = MLP_buffer[id];                      
    MLP_network& dnet = gradient_buffer[id];                //
    dnet.clear();
    for (int it = BP_index.x; it < BP_index.y; it++)
    {
        int bp_id = MLP_index_L2_buffer[it];                //
        int feed_id = bp_id + batch_size * batch_id;
        BP_token& t = BP_buffer[bp_id];
        feed_token& p = feed_buffer[feed_id];

        for (int j = 0; j < 32; j++)
        {
            dnet.bias_2[j] = -1;
        }

        for (int i = 0; i < 16; i++)
        { 
            for (int j = 0; j < 32; j++)
            {
                dnet.layer_2[i][j] += t.feature_1[i] * dnet.bias_2[j];
                dnet.bias_1[i] += dnet.bias_2[j] * net.layer_2[i][j]; 
            }
        }
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                dnet.layer_1[i][j] += t.feature_0[i] * dnet.bias_1[j];
                dnet.bias_0[i] += dnet.bias_1[j] * net.layer_1[i][j];
            }
        }


        float3 current_cos = make_float3(cos(p.position.x * M_PI), cos(p.position.y * M_PI), cos(p.position.z * M_PI));
        float3 current_sin = make_float3(sin(p.position.x * M_PI), sin(p.position.y * M_PI), sin(p.position.z * M_PI));
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                dnet.layer_0[i * 6 + 0][j] += current_cos.x * dnet.bias_0[j];
                dnet.layer_0[i * 6 + 1][j] += current_cos.y * dnet.bias_0[j];
                dnet.layer_0[i * 6 + 2][j] += current_cos.z * dnet.bias_0[j];
                dnet.layer_0[i * 6 + 3][j] += current_sin.x * dnet.bias_0[j];
                dnet.layer_0[i * 6 + 4][j] += current_sin.y * dnet.bias_0[j];
                dnet.layer_0[i * 6 + 5][j] += current_sin.z * dnet.bias_0[j];
            } 
            current_sin = current_sin * current_cos * 2;
            current_cos = 1 - current_cos * current_cos;
        }
    }

    net.update(dnet);
}



RT_FUNCTION bool eye_step(PerRayData_radiance& prd)
{
    float3 ray_origin = prd.origin;
    float3 ray_direction = prd.direction;
    optix::Ray ray(ray_origin, ray_direction, /*ray type*/ BDPTRay, scene_epsilon);

    int origin_depth = prd.stackP->size;
    rtTrace(top_object, ray, prd);

    prd.depth++;
    if (prd.stackP->size == origin_depth)
    {
        //miss
        prd.done = true;
        return false;
    }

    if (ISLIGHTSOURCE(prd.stackP->back()))
    {
        prd.done = true;
        return true;
    }

    if (!ISVALIDVERTEX(prd.stackP->back()))
    {
        prd.done = true;
        return false;
    }
    return true;

}
rtDeclareVariable(MLP::data_buffer, mlp_data_buffer, , ) = { }; 
RT_PROGRAM void MLP_path_construct()
{
    int2 launch_size = mlp_data_buffer.launch_size;
    int id = launch_size.x * launch_index.y + launch_index.x;
    unsigned int seed = tea<16>(id + mlp_data_buffer.construct_frame * mlp_data_buffer.construct_frame , mlp_data_buffer.launch_seed);

    //if(id==0)
    //printf("launch_id%d\n",mlp_data_buffer.launch_seed);
    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing.

    // Main render loop. This is not recursive, and for high ray depths
    // will generally perform better than tracing radiance rays recursively
    // in closest hit programs.
 
    int2 inch = mlp_data_buffer.res_padding;
    int2 base_ind = id * inch;
    for (int i = base_ind.x; i < base_ind.x + inch.x; i++)
    {
        mlp_data_buffer.p_samples[i].valid = false;
    }
    
    for (int i = base_ind.y; i < base_ind.y + inch.y; i++)
    {
        mlp_data_buffer.p_nodes[i].valid = false;
    }
    int2 counter = make_int2(0, 0); 
    int try_count = 0;
    int try_limit = 1;
    bool get_success_flag = false;
    int2 pixiv_id;
    while (try_count<try_limit && get_success_flag == false)
    {
        try_count++;

        float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        float2 d = subpixel_jitter * 2.f;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);

        pixiv_id = make_int2(dir2pix(ray_direction).x, dir2pix(ray_direction).y);
        BDPTVertexStack EStack(eye, ray_direction);

        PerRayData_radiance prd(seed, &EStack);

        while (prd.done == false)
        {
            bool hit_success = eye_step(prd);
            if (!hit_success || ISLIGHTSOURCE(prd.stackP->back()))break;
            //rtPrintf("%d AA\n",id);

            //int light_id = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * mlp_data_buffer.samples.size)), 0, mlp_data_buffer.samples.size - 1);

            int light_id = mlp_data_buffer.light_sample(rnd(prd.seed));
            //rtPrintf("pick light id%d %d\n",light_id, mlp_data_buffer.samples.size);
            //continue;
            //random sample, to be rewrite
            MLP::nVertex& light_vertex = mlp_data_buffer.samples[light_id];
#ifdef SAMPLE_ONLY_SOURCE
            while (light_vertex.isLightSource() == false)
            {
                light_id = mlp_data_buffer.light_sample(rnd(prd.seed));
                light_vertex = mlp_data_buffer.samples[light_id];
            }
#endif

            bool visible_flag = light_vertex.isDirLight() ?
                isVisible_direction(prd.stackP->back().position, prd.stackP->back().normal, -light_vertex.normal) :
                isVisible(prd.stackP->back().position, light_vertex.position);
            bool succes_flag = prd.stackP->back().isBrdf == false && visible_flag;

#ifdef GLASSROOM
#define PATHLIMIT 7
#else
#define PATHLIMIT 10
#endif
            if (succes_flag)
            {
                get_success_flag = true;
                int brdf_count = 0;
                int back_count = 0;
                int back_current = 1;
                MLP::nVertex_device buffer_vertexs[PATHLIMIT];
                if (mlp_data_buffer.get_sample_depth(light_id) >= PATHLIMIT)
                {
                    continue;
                }
                if (false)
                {
                    MLP::nVertex_device* light_nVertex_p = (MLP::nVertex_device*) & light_vertex;
                    buffer_vertexs[back_count] = *light_nVertex_p;
                    back_count++;

                    while (true)
                    {
                        if (buffer_vertexs[back_count - 1].isLightSource())
                        {
                            //back_count++;
                            break;
                        }
                        else
                        {
                            light_nVertex_p = (MLP::nVertex_device*) & mlp_data_buffer.LVC[light_nVertex_p->last_id];
                            buffer_vertexs[back_count] = *light_nVertex_p;
                            back_count++;
                        }
                    }
                    //rtPrintf("stack size %d %d\n",back_count,mlp_data_buffer.get_sample_depth(light_id));
                    for (int i = back_count - 2; i >= 0; i--)
                    {
                        buffer_vertexs[i] = MLP::nVertex_device(buffer_vertexs[i], buffer_vertexs[i + 1], false);
                    }


                    MLP::nVertex_device eye_res = MLP::nVertex_device(prd.stackP->v[1], true);
                    for (int i = 2; i < prd.stackP->size; i++)
                    {
                        if (i > 20)break;
                        MLP::nVertex_device eye_res2 = MLP::nVertex_device(prd.stackP->v[i], true);

                        eye_res = MLP::nVertex_device(eye_res2, eye_res, true);
                        prd.stackP->v[i].pdf = eye_res.weight.x;
                    }
                }


                MLP::pathInfo_sample& sample = mlp_data_buffer.p_samples.v[(base_ind + counter).x];
                counter.x++;
                sample.valid = true;
                sample.pixiv_id = pixiv_id;
                MLP::nVertex_device* light_nVertex_p = (MLP::nVertex_device*) & light_vertex;
                //            MLP::nVertex_device * light_nVertex_p = (MLP::nVertex_device*)&buffer_vertexs[0];
                MLP::nVertex_device   eye_nVertex = MLP::nVertex_device(EStack.back(), true);
                float3 seg_contri = eye_nVertex.local_contri(*light_nVertex_p);
                sample.contri = EStack.back().flux * light_nVertex_p->forward_light(eye_nVertex) * seg_contri;
                float3 stand_light_flux = EStack.back().flux;
                sample.choice_id = light_id;
                if (mlp_data_buffer.get_sample_depth(light_id) + EStack.size - 1 >= PATHLIMIT)
                {
                    sample.valid = false;
                    counter.x--;
                    break;
                }
                if (!(ENERGY_WEIGHT(sample.contri) > 0.0))
                {
                    //printf("%d invalid count %f %f %f\n", light_nVertex_p->isDirLight(),
                    // ENERGY_WEIGHT(seg_contri), ENERGY_WEIGHT(EStack.back().flux), ENERGY_WEIGHT(light_nVertex_p->forward_light(eye_nVertex)));
                    sample.valid = false;
                    counter.x--;
                    continue;
                }
                MLP::contri_reseter reseter_light(eye_nVertex, *light_nVertex_p);
                MLP::contri_reseter reseter_eye(*light_nVertex_p, eye_nVertex);

                sample.sample_pdf = EStack.back().pdf * light_nVertex_p->pdf * mlp_data_buffer.light_select_pdf(light_nVertex_p->get_label())
                    * light_nVertex_p->brdf_weight() * eye_nVertex.brdf_weight();
                float init_sample_pdf = sample.sample_pdf;
                //eye forward, light backward
                sample.begin_ind = base_ind.y + counter.y;
                while (true)
                {
                    mlp_data_buffer.p_nodes[base_ind.y + counter.y] = MLP::pathInfo_node(eye_nVertex, *light_nVertex_p);
                    counter.y++;
                    if (light_nVertex_p->isLightSource())
                    {
                        sample.fix_pdf = eye_nVertex.forward_eye(*light_nVertex_p).x;
                        reseter_light.hit_light(light_nVertex_p->weight);
                        //rtPrintf("contri compare %f %f\n", ENERGY_WEIGHT(stand_light_flux),ENERGY_WEIGHT(n_contri));
                        break;
                    }
                    else
                    {
                        eye_nVertex = MLP::nVertex_device(*light_nVertex_p, eye_nVertex, true);
                        light_nVertex_p = (MLP::nVertex_device*) & mlp_data_buffer.LVC[light_nVertex_p->last_id];
                        //light_nVertex_p = (MLP::nVertex_device*)&buffer_vertexs[back_current]; 
                        //back_current++;
                        sample.sample_pdf += eye_nVertex.weight.x * light_nVertex_p->pdf * mlp_data_buffer.light_select_pdf(light_nVertex_p->get_label())
                            * light_nVertex_p->brdf_weight() * eye_nVertex.brdf_weight();

                        reseter_light.iteration(*light_nVertex_p);
                        //if(light_nVertex_p->isBrdf)brdf_count ++;

                    }
                }
                //light_nVertex_p = (MLP::nVertex_device*)&buffer_vertexs[0];
                light_nVertex_p = (MLP::nVertex_device*) & light_vertex;

                eye_nVertex = MLP::nVertex_device(EStack.back(), true);
                MLP::nVertex_device light_nVertex;

                for (int i = 1; EStack.size - 1 - i > 0; i++)
                {
                    light_nVertex = MLP::nVertex_device(eye_nVertex, *light_nVertex_p, false);

                    eye_nVertex = MLP::nVertex_device(EStack.back(i), true);
                    reseter_eye.iteration(eye_nVertex);
                    if (eye_nVertex.isBrdf)brdf_count++;
                    light_nVertex_p = &light_nVertex;

                    //if(light_nVertex_p->isBrdf||eye_nVertex.isBrdf)continue;

                    sample.sample_pdf += eye_nVertex.weight.x * light_nVertex_p->pdf * mlp_data_buffer.light_select_pdf(light_nVertex_p->get_label())
                        * light_nVertex_p->brdf_weight() * eye_nVertex.brdf_weight();
                    mlp_data_buffer.p_nodes[base_ind.y + counter.y] = MLP::pathInfo_node(eye_nVertex, *light_nVertex_p);
                    //if(eye_nVertex.isBrdf)
                    //  rtPrintf("%d %d %f\n",light_nVertex_p->brdf_weight(), eye_nVertex.brdf_weight(), mlp_data_buffer.p_nodes[base_ind.y + counter.y].peak_pdf);
                    counter.y++;
                }

                sample.end_ind = base_ind.y + counter.y;


                eye_nVertex = MLP::nVertex_device(EStack.v[0], true);
                reseter_eye.iteration(eye_nVertex);
                reseter_eye.hit_light(make_float3(1.0));
                //rtPrintf("%d contri compare %f %f\n", prd.stackP->size, ENERGY_WEIGHT(stand_light_flux),ENERGY_WEIGHT(n_contri));
                float3 real_contri = reseter_light.merge_with(reseter_eye);
                float real_fake_ratio = ENERGY_WEIGHT(real_contri) / ENERGY_WEIGHT(sample.contri);
                //rtPrintf("real fake compare %f \n", real_fake_ratio);

                real_fake_ratio = max(real_fake_ratio, 0.00001);
#ifdef SAMPLE_ONLY_SOURCE
                sample.sample_pdf = init_sample_pdf;
#endif
                sample.contri *= real_fake_ratio; 
                sample.fix_pdf *= real_fake_ratio;
                sample.sample_pdf *= real_fake_ratio;
                for (int i = sample.begin_ind; i < sample.end_ind; i++)
                {
                    mlp_data_buffer.p_nodes[i].peak_pdf *= real_fake_ratio;
                }

                //printf("fix pdf %e\n",sample.fix_pdf);

                if (sample.fix_pdf < ENERGY_WEIGHT(sample.contri) * 0.00001)
                {
                    sample.fix_pdf = ENERGY_WEIGHT(sample.contri) * 0.00001;
                    //printf("AAA");
                }
                //if(brdf_count>0)rtPrintf("brdf found"); 
                if (light_vertex.isLightSource())
                {
                    BDPTVertex temple_light;
                    temple_light.position = light_vertex.position;
                    temple_light.normal = light_vertex.normal;
                    temple_light.flux = light_vertex.weight;
                    temple_light.materialId = 0;

                    //float3 pdf0 = MLP::nVertex_device(EStack.back(),true).forward_eye(light_vertex);
                    //OptimizationPathInfo optp = OPT_info_from_path_NEE(EStack,temple_light);

                    //rtPrintf("counter %d %d\n\n",counter.y,EStack.size - 1);
                }

            }
            if (counter.x >= inch.x || counter.y >= inch.y - PATHLIMIT)break;

        }
        seed = prd.seed;
    }

}