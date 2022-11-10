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
#include "rmis.h"
using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float2, beta_gamma,         attribute beta_gamma, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );
rtDeclareVariable( float3, uv, attribute uv, );
rtDeclareVariable( int, p_idx,  attribute p_idx, ); 
rtDeclareVariable( int, base_PrimIdx,  , ); 

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(int, max_depth, , );
rtBuffer<triangleStruct,1> triangle_samples;

rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFPdf;
rtBuffer< rtCallableProgramId<void(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFSample;
rtBuffer< rtCallableProgramId<float3(MaterialParameter &mat, State &state, PerRayData_radiance &prd)> > sysBRDFEval;
rtBuffer< rtCallableProgramId<void(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)> > sysLightSample;

rtBuffer<UberZoneLVC,1>          uberLVC;
rtDeclareVariable(float, scene_area, , ) = {1.0};
 
rtDeclareVariable(int, materialId, , ); 
rtDeclareVariable(int, programId, , );
rtDeclareVariable(int,           KD_SET, , ) = { 0 };

rtBuffer<LightParameter> sysLightParameters;


rtBuffer<KDPos,1>        Kd_position;   
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
RT_FUNCTION float3 DirectLight(MaterialParameter &mat, State &state)
{
	float3 L = make_float3(0.0f);

	//Pick a light to sample
	int index = optix::clamp(static_cast<int>(floorf(rnd(prd.seed) * sysNumberOfLights)), 0, sysNumberOfLights - 1);
	LightParameter light = sysLightParameters[index];
	LightSample lightSample;

	float3 surfacePos = state.fhp;
	float3 surfaceNormal = state.ffnormal;

	if (light.lightType == ENV) 
	{
		sysLightSample[light.lightType](light, prd, lightSample);

		//float lightPdf = 1.0 / sysNumberOfLights * lightSample.pdf;

		float3 lightDir = -lightSample.normal;
		PerRayData_shadow prd_shadow;
		prd_shadow.inShadow = false;
		optix::Ray shadowRay(surfacePos, lightDir, 1, scene_epsilon);
		rtTrace(top_object, shadowRay, prd_shadow);
		if (!prd_shadow.inShadow && dot(surfaceNormal, lightDir) > 0.0)
		{
			prd.direction = lightDir;
			float3 f = sysBRDFEval[programId](mat, state, prd);
			L = prd.throughput * f * lightSample.emission / lightSample.pdf * sysNumberOfLights ;
		}
		return L;
	}
	if(light.lightType == DIRECTION)
	{
		float3 lightDir = -light.direction;
	
		PerRayData_shadow prd_shadow;
		prd_shadow.inShadow = false;
		optix::Ray shadowRay(surfacePos, lightDir, 1, scene_epsilon);
		rtTrace(top_object, shadowRay, prd_shadow);
		if(!prd_shadow.inShadow && dot(surfaceNormal,lightDir)>0.0)
		{
			prd.direction = lightDir;
			float3 f = sysBRDFEval[programId](mat, state, prd);
			L = prd.throughput * f * light.emission * sysNumberOfLights;
		}
		return L;
	}

	sysLightSample[light.lightType](light, prd, lightSample);
    
	float3 lightDir = lightSample.surfacePos - surfacePos;
	float lightDist = length(lightDir);
	float lightDistSq = lightDist * lightDist;
	lightDir /= sqrtf(lightDistSq);

	if (dot(lightDir, surfaceNormal) <= 0.0f || dot(lightDir, lightSample.normal) >= 0.0f )
		return L;

	PerRayData_shadow prd_shadow;
	prd_shadow.inShadow = false;
	optix::Ray shadowRay = optix::make_Ray(surfacePos, lightDir, 1, scene_epsilon, lightDist - scene_epsilon);
	rtTrace(top_object, shadowRay, prd_shadow);

	if (!prd_shadow.inShadow)
	{
		float NdotL = dot(lightSample.normal, -lightDir);
		float lightPdf = lightDistSq / (light.area * NdotL);

		prd.direction = lightDir;

		sysBRDFPdf[programId](mat, state, prd);
		float3 f = sysBRDFEval[programId](mat, state, prd);

		L = powerHeuristic(lightPdf, prd.pdf) * prd.throughput * f * lightSample.emission / lightPdf * sysNumberOfLights;
	}

	return L;
}
 
RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
	int zoneNum = triangle_samples[p_idx + base_PrimIdx].zoneNum;

	MaterialParameter mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
		mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}
	int div_block = 0;
	int div_class = 7;
	float beta = beta_gamma.x;
	float gamma = beta_gamma.y;
	while(div_class!=0)
	{
		div_block = div_block<<1;
		div_class -= 1;
		float n_gamma  = abs((beta + gamma - 1));
		float n_beta = abs((beta - gamma));
		if(beta<gamma)
		{
			div_block += 1;
			gamma = n_beta;
			beta = n_gamma;
		} 
		else
		{
			beta = n_beta;
			gamma = n_gamma;

		}
	} 

	State state;
	state.fhp = front_hit_point;
	state.bhp = back_hit_point;
	state.normal = world_shading_normal;
	state.ffnormal = ffnormal;
	state.eye_side = true;

	prd.radiance += mat.emission * prd.throughput;

	//TODO: Clean up handling of specular bounces
	prd.specularBounce = mat.brdf == GLASS? true : false;

	// Direct light Sampling
	if (!prd.specularBounce && prd.depth < max_depth)
	{
		prd.radiance += DirectLight(mat, state);
	}
	//prd.radiance = hsv2rgb(div_block / 128.0 * 180,1.0, 1.0 );
	//prd.done = true;
	// BRDF Sampling
	sysBRDFSample[programId](mat, state, prd);
	sysBRDFPdf[programId](mat, state, prd);

	float3 f = sysBRDFEval[programId](mat, state, prd);

	if (prd.pdf > 0.0f)
		prd.throughput *= f / prd.pdf; 
	else
		prd.done = true;
	#ifdef NO_COLOR
		prd.radiance = ffnormal + make_float3(1.0);
		prd.radiance = mat.color;
		//prd.radiance = texcoord ;
		prd.done = true;
	#endif
	
}

rtDeclareVariable(unsigned int, frame, , );
RT_PROGRAM void rr_closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	const float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

#ifdef NO_COLOR
	prd.radiance = ffnormal + make_float3(1.0);
	prd.done = true;
	return;
#endif
	//int zoneNum = triangle_samples[p_idx + base_PrimIdx].zoneNum;
	{
		MaterialParameter mat = sysMaterialParameters[materialId];

		if (mat.albedoID != RT_TEXTURE_ID_NULL)
		{
			const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
			mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
		}

		prd.radiance = mat.color * abs(dot(ffnormal, ray.direction));
		return;
	}

	labelUnit vertex_label(front_hit_point, ffnormal, -ray.direction, true);
	int zoneNum = vertex_label.getLabel();
	//int zoneNum = tree_index_node_id(classTree::temple_eye, front_hit_point);
	unsigned int z2 = zoneNum;
	prd.radiance = make_float3(rnd(z2), rnd(z2), rnd(z2));
	prd.done = true;

	return;


	MaterialParameter mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
		mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}
	

	State state;
	state.fhp = front_hit_point;
	state.bhp = back_hit_point;
	state.normal = world_shading_normal;
	state.ffnormal = ffnormal;
	state.eye_side = true;

	prd.radiance += mat.emission * prd.throughput;

	//TODO: Clean up handling of specular bounces
	prd.specularBounce = mat.brdf == GLASS? true : false;

	// Direct light Sampling
	if (!prd.specularBounce)
	{
		prd.radiance += DirectLight(mat, state);
	}
	// BRDF Sampling
	sysBRDFSample[programId](mat, state, prd);
	sysBRDFPdf[programId](mat, state, prd);
	float3 f = sysBRDFEval[programId](mat, state, prd);

float rr_rate = fmaxf(mat.color);
#ifdef RR_MIN_LIMIT
	rr_rate = max(rr_rate,MIN_RR_RATE);
#endif
#ifdef RR_DISABLE
	rr_rate = 1.0f;
#endif

	if (prd.pdf > 0.0f && rnd(prd.seed)< rr_rate)
		prd.throughput *= f / prd.pdf / rr_rate; 
	else
		prd.done = true; 
	

	if (false)
	{
		int grid = 100;
		int2 grid_id = make_int2(uv.x * grid, uv.y * grid);
		int black = (int(uv.x * grid) + int(uv.y * grid)) % 2;
		if (black)
		{
			mat.color = make_float3(0.0);
		}
		else
		{
			mat.color = make_float3(1.0);
		}
		if ((grid_id.x *grid + grid_id.y) == 0 )
		{
			mat.color = make_float3(1.0, 0.5, 0.3);
		}
	}
	//prd.radiance = make_float3(uv.x, uv.y, 0.2);
	//prd.radiance = mat.color;
	//prd.done = true;

}

RT_PROGRAM void any_hit()
{ 
	MaterialParameter &mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float alpha = optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y).w;
		if (alpha < scene_epsilon )
		{ 
			rtIgnoreIntersection(); 
		}
	} 
	prd_shadow.inShadow = true;
	rtTerminateRay(); 
}


RT_PROGRAM void light_any_hit()
{ 
	prd_shadow.inShadow = true;
	rtTerminateRay();
}
RT_PROGRAM void trans_any_hit()
{
	MaterialParameter& mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float alpha = optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y).w;
		if (alpha < scene_epsilon)
		{
			rtIgnoreIntersection();
		}
	} 
}
RT_PROGRAM void BDPT_closest_hit()
{   
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
	int tri_id = p_idx + base_PrimIdx;
	int zoneNum;// = triangle_samples[p_idx + base_PrimIdx].zoneNum;
	//int div_tri_id = triBase[tri_id] + FindDivTriNum(divLevel[tri_id], beta_gamma);
	//zoneNum = div_tris[div_tri_id].zoneNum;

	//find the subspace
	labelUnit vertex_label(front_hit_point, ffnormal, -ray.direction, make_float2(texcoord), beta_gamma, materialId, tri_id,false);
	zoneNum = vertex_label.getLabel();
 
	MaterialParameter mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
		mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}

	
	State state;
	state.fhp = front_hit_point;
	state.bhp = back_hit_point;
	state.normal = world_shading_normal;
	state.ffnormal = ffnormal;
	state.eye_side = true;

	prd.specularBounce = mat.brdf == GLASS? true : false;

	// BRDF Sampling
	sysBRDFSample[programId](mat, state, prd);
	sysBRDFPdf[programId](mat, state, prd);

	if (!(prd.pdf > 0.0f))
		prd.done = true;

	/*for a BDPT Version*/
	BDPTVertex &MidVertex  = prd.stackP->v[(prd.stackP->size)     % STACKSIZE];
	BDPTVertex &NextVertex = prd.stackP->v[(prd.stackP->size + 1) % STACKSIZE];
	BDPTVertex &LastVertex = prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
	//warning , I am not sure about the true meaning of 'front_hit_point'

	//需要为下一个顶点填写：singlePdf的方向部分；flux的本顶点brdf部分
	//需要为本顶点填写    optix::float3 position;
    //					 optix::float3 normal;
    //					 optix::float3 flux;
    //					 optix::float3 color;
    //					 optix::float3 lastPosition;
    //					 optix::float3 lastNormal;
    //							 float pdf;
    //							 float singlePdf;

    //							 float d;
    //							 float dLast;

    //							 int materialId; yes

    //							 int zoneId; yes
    
    //							 bool isOrigin; yes
	//							 bool inBrdf; yes
	if(prd.specularBounce)
	{
		MidVertex.inBrdf = true;
		MidVertex.isBrdf = true;
		NextVertex.inBrdf = true;
	}
	else
	{
		MidVertex.isBrdf = false;
		NextVertex.inBrdf = false;
	}
	float rr_rate = fmaxf(mat.color);


#ifdef RR_MIN_LIMIT
	rr_rate = max(rr_rate,MIN_RR_RATE);
#endif
#ifdef RR_DISABLE
	rr_rate = 1.0f;
#endif
	MidVertex.position = ray.origin + t_hit * ray.direction;
	MidVertex.normal = MidVertex.isBrdf? world_shading_normal :ffnormal;


	float pdf_G = abs(dot(MidVertex.normal,ray.direction) * dot(LastVertex.normal,ray.direction)) / (t_hit * t_hit);
	if(LastVertex.isOrigin)
	{
		MidVertex.flux = LastVertex.flux * pdf_G;
	}
	else
	{
		MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G;
	}
	NextVertex.flux = DisneyEval(mat,ffnormal,-ray.direction,prd.direction) / (mat.brdf?abs(dot(MidVertex.normal,prd.direction)):1.0f);
	NextVertex.singlePdf = prd.pdf;


	MidVertex.color = mat.color;
	MidVertex.lastPosition = LastVertex.position;
	MidVertex.lastNormalProjection = abs(dot(LastVertex.normal,ray.direction));
	MidVertex.materialId = materialId;
	MidVertex.zoneId = zoneNum;
	MidVertex.lastZoneId = LastVertex.zoneId;
	MidVertex.lastBrdf = LastVertex.isBrdf;
	MidVertex.isOrigin = false;
	MidVertex.depth = LastVertex.depth + 1;
	MidVertex.uv = make_float2(texcoord);


	MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal,ray.direction));
	MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;
	 
	{
		MidVertex.lastSinglePdf = LastVertex.singlePdf;
		MidVertex.isLastVertex_direction = LastVertex.depth == 0 && (LastVertex.type == DIRECTION || LastVertex.type == ENV);
		if (MidVertex.depth == 1)
		{
			tracing_init_eye(MidVertex, LastVertex);
		}
		else
		{
			tracing_update_eye(MidVertex, LastVertex);
		}
		prd.stackP->size++;


		float r = rnd(prd.seed);
		if (r > rr_rate)
		{
			prd.done = true;
		}
		else
		{
			NextVertex.singlePdf *= rr_rate;
			prd.throughput *= NextVertex.flux / prd.pdf / rr_rate * dot(ffnormal, prd.direction);
		}
		return;
	} 

}
RT_PROGRAM void BDPT_L_closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
	int tri_id = p_idx + base_PrimIdx;
	int zoneNum;

	//find the subspace
	labelUnit vertex_label(front_hit_point, ffnormal, -ray.direction, make_float2(texcoord), beta_gamma, materialId, tri_id,true);
	zoneNum = vertex_label.getLabel(); 
	MaterialParameter mat = sysMaterialParameters[materialId];

	if (mat.albedoID != RT_TEXTURE_ID_NULL)
	{
		const float3 texColor = make_float3(optix::rtTex2D<float4>(mat.albedoID, texcoord.x, texcoord.y));
		mat.color = make_float3(powf(texColor.x, 2.2f), powf(texColor.y, 2.2f), powf(texColor.z, 2.2f));
	}
	State state;
	state.fhp = front_hit_point;
	state.bhp = back_hit_point;
	state.normal = world_shading_normal;
	state.ffnormal = ffnormal;
	state.eye_side = false;

	// BRDF Sampling
	sysBRDFSample[programId](mat, state, prd);
	sysBRDFPdf[programId](mat, state, prd);
	if (!(prd.pdf > 0.0f))
		prd.done = true;
		
        /*for a BDPT Version*/
	BDPTVertex &MidVertex  = prd.stackP->v[(prd.stackP->size)     % STACKSIZE];
	BDPTVertex &NextVertex = prd.stackP->v[(prd.stackP->size + 1) % STACKSIZE];
	BDPTVertex &LastVertex = prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
	//warning , I am not sure about the true meaning of 'front_hit_point'

	//需要为下一个顶点填写：singlePdf的方向部分；flux的本顶点brdf部分
	//需要为本顶点填写    optix::float3 position;
    //					 optix::float3 normal;
    //					 optix::float3 flux;
    //					 optix::float3 color;
    //					 optix::float3 lastPosition;
    //					 optix::float3 lastNormal;
    //							 float pdf;
    //							 float singlePdf;

    //							 float d;
    //							 float dLast;

    //							 int materialId; yes

    //							 int zoneId; yes
    
    //							 bool isOrigin; yes
	prd.specularBounce = mat.brdf == GLASS ? true : false;
	if(prd.specularBounce)
	{
		MidVertex.inBrdf = true;
		MidVertex.isBrdf = true;
		NextVertex.inBrdf = true;
	}
	else
	{
		MidVertex.isBrdf = false;
		NextVertex.inBrdf = false;
	}	

	float rr_rate = fmaxf(mat.color);

#ifdef RR_MIN_LIMIT
	rr_rate = max(rr_rate,MIN_RR_RATE);
#endif
#ifdef RR_DISABLE
	rr_rate = 1.0f;
#endif
	MidVertex.position = ray.origin + t_hit * ray.direction;
	MidVertex.normal = MidVertex.isBrdf ? world_shading_normal : ffnormal;


	float pdf_G = abs(dot(MidVertex.normal,ray.direction) * dot(LastVertex.normal,ray.direction)) / (t_hit * t_hit);
	if(LastVertex.type == DIRECTION|| LastVertex.type == ENV)
	{
		pdf_G = abs(dot(MidVertex.normal,ray.direction) * dot(LastVertex.normal,ray.direction));
	}
	if(LastVertex.isOrigin)
	{
		MidVertex.flux = LastVertex.flux * pdf_G;

	}
	else
	{
		MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G;
	}
	NextVertex.flux = DisneyEval(mat,ffnormal,-ray.direction,prd.direction) / (mat.brdf?abs(dot(MidVertex.normal,prd.direction)):1.0f);
	NextVertex.singlePdf = prd.pdf;

	MidVertex.color = mat.color;
	MidVertex.lastPosition = LastVertex.position;
	if (LastVertex.type == DIRECTION || LastVertex.type == ENV)
	{
		MidVertex.lastPosition = MidVertex.position - ray.direction;
	}

	MidVertex.lastNormalProjection = abs(dot(LastVertex.normal,ray.direction));
	MidVertex.materialId = materialId;
	MidVertex.zoneId = zoneNum;
	MidVertex.lastZoneId = LastVertex.zoneId;
	MidVertex.lastBrdf = LastVertex.isBrdf;
	MidVertex.isOrigin = false;
	MidVertex.depth = LastVertex.depth + 1;
	MidVertex.uv = make_float2(texcoord);

	MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal,ray.direction));
	MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

	MidVertex.last_lum = float3sum(LastVertex.flux / LastVertex.pdf);


#ifdef ZGCBPT
	{
		MidVertex.lastSinglePdf = LastVertex.singlePdf;
		MidVertex.isLastVertex_direction = LastVertex.depth == 0 && (LastVertex.type == DIRECTION || LastVertex.type == ENV);
		if (LastVertex.isOrigin)
		{
			tracing_init_light(MidVertex, LastVertex);
		}
		else
		{
			tracing_update_light(MidVertex, LastVertex);
		}
		prd.stackP->size++;

		float r = rnd(prd.seed);
		if (r > rr_rate)
		{
			prd.done = true;
		}
		else
		{
			NextVertex.singlePdf *= rr_rate;
		}
		return;
	}
#endif // ZGCBPT

}