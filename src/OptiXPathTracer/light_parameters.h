#pragma once

#ifndef LIGHT_PARAMETER_H
#define LIGHT_PARAMETER_H

#include"rt_function.h"
#include <sutil/vec_math.h> 
enum LightType
{
	SPHERE, QUAD, DIRECTION,ENV, LightTypeNum
//	SPHERE, QUAD, DIRECTION,ENV,HIT_LIGHT_SOURCE,ENV_MISS, NORMALHIT,LightTypeNum
};
enum MyRayType
{
    PTRay,ShadowRay,BDPTRay, BDPT_L_Ray, PT_RR_RAY, RayTypeCount
};
struct LightParameter
{
	float3 position;
	float3 normal;
	float3 emission;
	float3 u;
	float3 v;
    float3 direction;
	LightType lightType;
	float area;
	float radius;
	int albedoID;
	int divBase;
	int divLevel;
	int id;
};
 

#endif
