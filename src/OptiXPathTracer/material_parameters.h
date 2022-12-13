#pragma once

#ifndef MATERIAL_PARAMETER_H
#define MATERIAL_PARAMETER_H

#include"rt_function.h"

enum BrdfType
{
	DISNEY, GLASS
};

struct MaterialParameter
{
#define RT_TEXTURE_ID_NULL 0
	RT_FUNCTION MaterialParameter()
	{
		color = make_float3(1.0f, 1.0f, 1.0f);
		emission = make_float3(0.0f);
		metallic = 0.0;
		subsurface = 0.0f;
		specular = 0.5f;
		roughness = 0.5f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.5f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
		brdf = DISNEY;
		albedoID = RT_TEXTURE_ID_NULL;
	}

	int albedoID;
	float3 color;
	float3 emission;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float trans;
	BrdfType brdf;
};

#endif
