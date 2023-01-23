/*Copyright (c) 2016 Miles Macklin

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.*/

#include"sceneLoader.h"

static const int kMaxLineLength = 2048;
std::vector<LightParameter> breakLight(LightParameter& a,int divLevel)
{
	std::vector<LightParameter> ans;
	if (a.lightType != QUAD)
	{
		ans.push_back(a);
		return ans;
	}
	float3 sig_v1 = a.u / divLevel;
	float3 sig_v2 = a.v / divLevel;
	for (int i = 0; i < divLevel; i++)
		for (int j = 0; j < divLevel; j++)
		{
			LightParameter sigL(a);
			sigL.position = a.position + i * sig_v1 + j * sig_v2;
			sigL.u = sig_v1;
			sigL.v = sig_v2;
			sigL.area = a.area / divLevel / divLevel;
			//sigL.normal = a.normal;
			//sigL.lightType = a.lightType;
			ans.push_back(sigL);
		}
	return ans;
}

Scene* LoadScene(const char* filename)
{
	Scene *scene = new Scene;
	int tex_id = 0;
	FILE* file = fopen(filename, "r");

	if (!file)
	{
		printf("Couldn't open %s for reading.", filename);
		return NULL;
	}

	std::map<std::string, MaterialParameter> materials_map;
	std::map<std::string, int> texture_ids;

	char line[kMaxLineLength];

	while (fgets(line, kMaxLineLength, file))
	{
		// skip comments
		if (line[0] == '#')
			continue;

		// name used for materials and meshes
		char name[kMaxLineLength] = { 0 };


		//--------------------------------------------
		// Material

		if (sscanf(line, " material %s", name) == 1)
		{
			printf("%s", line);

			MaterialParameter material;
			char tex_name[kMaxLineLength] = "None";

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " name %s", name);
				sscanf(line, " color %f %f %f", &material.color.x, &material.color.y,
                    &material.color.z);
				sscanf(line, " albedoTex %s", &tex_name);
				sscanf(line, " emission %f %f %f", &material.emission.x, &material.emission.y,
                    &material.emission.z);

				sscanf(line, " metallic %f", &material.metallic);
				sscanf(line, " subsurface %f", &material.subsurface);
				sscanf(line, " specular %f", &material.specular);
				sscanf(line, " specularTint %f", &material.specularTint);
				sscanf(line, " roughness %f", &material.roughness);
				sscanf(line, " anisotropic %f", &material.anisotropic);
				sscanf(line, " sheen %f", &material.sheen);
				sscanf(line, " sheenTint %f", &material.sheenTint);
				sscanf(line, " clearcoat %f", &material.clearcoat);
				sscanf(line, " clearcoatGloss %f", &material.clearcoatGloss);
				sscanf(line, " brdf %i", &material.brdf);

				sscanf(line, " trans %f", &material.trans);
				sscanf(line, " eta %f", &material.eta);
				if (material.roughness < .001)material.roughness = .001;
			}

			// Check if texture is already loaded
			if (texture_ids.find(tex_name) != texture_ids.end()) // Found Texture
			{
				material.albedoID = texture_ids[tex_name];
			}
			else if(strcmp(tex_name, "None") != 0)
			{
				tex_id++;
				texture_ids[tex_name] = tex_id;
				scene->texture_map[tex_id - 1] = tex_name;
				material.albedoID = tex_id;
			}

			// add material to map
			materials_map[name] = material;
		}

		//--------------------------------------------
		// Light

		if (strstr(line, "light"))
		{
			LightParameter light;
			float3 v1, v2;
			char light_type[20] = "None";
			int lightDivLevel = 1;
			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " position %f %f %f", &light.position.x, &light.position.y,
                    &light.position.z);
				sscanf(line, " emission %f %f %f", &light.emission.x, &light.emission.y,
                    &light.emission.z);
				sscanf(line, " normal %f %f %f", &light.normal.x, &light.normal.y,
                    &light.normal.z);

                sscanf(line, " direction %f %f %f", &light.direction.x, &light.direction.y,
                    &light.direction.z);
				sscanf(line, " radius %f", &light.radius);
				//sscanf(line, " u %f %f %f", &light.v1.x, &light.v1.y, &light.v1.z);
				//sscanf(line, " v %f %f %f", &light.v2.x, &light.v2.y, &light.v2.z);
				sscanf(line, " v1 %f %f %f", &v1.x, &v1.y, &v1.z);
				sscanf(line, " v2 %f %f %f", &v2.x, &v2.y, &v2.z);
				sscanf(line, " type %s", light_type);
				sscanf(line, " divLevel %d", &lightDivLevel);
			}
			light.divLevel = lightDivLevel;
			if (strcmp(light_type, "Quad") == 0)
			{
				light.lightType = QUAD;
				light.u = v1 - light.position;
				light.v = v2 - light.position;
				light.area = length(cross(light.u, light.v));
				light.normal = normalize(cross(light.u, light.v));
			}
			else if (strcmp(light_type, "Sphere") == 0)
			{
				light.lightType = SPHERE;
				light.normal = normalize(light.normal);
				light.area = 4.0f * M_PIf * light.radius * light.radius;
			}
            else if (strcmp(light_type, "Direction") == 0)
            {
                light.lightType = DIRECTION;
                light.direction = normalize(light.direction);
                
            }
            else if (strcmp(light_type, "Env") == 0)
            {
                light.lightType = ENV;

            }
			if (light.lightType == DIRECTION)
			{
				scene->dirLightDir = light.direction;
			}
			//printf("%d\n", lightDivLevel);
			//auto local_lights = breakLight(light, lightDivLevel);
			//scene->lights.insert(scene->lights.end(), local_lights.begin(), local_lights.end());
			//printf("insert lights %d\n", local_lights.size());
			scene->lights.push_back(light);
		}

		//--------------------------------------------
		// Properties

		//Defaults
		Properties prop;
		prop.width = 1920;
		prop.height = 1001;
		scene->properties = prop;

		if (strstr(line, "properties"))
		{

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " width %i", &prop.width);
				sscanf(line, " height %i", &prop.height);
			}
			scene->properties = prop;
		}

		//--------------------------------------------
        if (strstr(line, "cameraSetting"))
        {
            scene->use_camera = true;
            float3 eye, lookat,up = make_float3(0,1,0);
			std::string env_file = "";
			char env_file_c[256] = "";
            float fov = 35.0f;
			int geo_normal = 0;
			float env_lum = 1;
            while (fgets(line, kMaxLineLength, file))
            {
                // end group
                if (strchr(line, '}'))
                    break;

                sscanf(line, " eye %f %f %f", &eye.x,&eye.y,&eye.z);
                sscanf(line, " lookat %f %f %f", &lookat.x, &lookat.y, &lookat.z);
                sscanf(line, " up %f %f %f", &up.x, &up.y, &up.z);
                sscanf(line, " fov %f", &fov);
				sscanf(line, " geo_normal %d", &geo_normal);
				sscanf(line, " env_lum %f", &env_lum);
				sscanf(line, " env_file %s", env_file_c); 
            }
			if (geo_normal == 1)
			{
				scene->use_geometry_normal = true;
			}
            scene->eye = eye;
            scene->lookat = lookat;
            scene->up = up;
            scene->fov = fov;
			scene->env_file = std::string(env_file_c);
			scene->env_factor = env_lum;
        }
        
        // Mesh

		if (strstr(line, "mesh"))
		{ 
			while (fgets(line, kMaxLineLength, file))
			{
				// end group
                if (strchr(line, '}'))
                { 
                    break;
                }
				int count = 0;

				char path[2048];

				if (sscanf(line, " file %s", path) == 1)
				{
					scene->mesh_names.push_back(std::string(SAMPLES_DIR) + "/data/" + path);
					scene->uv_mesh_names.push_back(std::string(SAMPLES_DIR) + "/data/" + path);
				}
				if (sscanf(line, " uv_file %s", path) == 1)
				{ 
					scene->uv_mesh_names.back() = (std::string(SAMPLES_DIR) + "/data/" + path);
				}
				if (sscanf(line, " material %s", path) == 1)
				{
					// look up material in dictionary
					if (materials_map.find(path) != materials_map.end())
					{
						scene->materials.push_back(materials_map[path]);
					}
					else
					{
						printf("Could not find material %s\n", path);
					}
				}
                float m_data[16];
                if (strstr(line, "transform" ))
                {
                    int t = sscanf(line, " transform [ %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f ]", &m_data[0], &m_data[1], &m_data[2], &m_data[3], &m_data[4],
                        &m_data[5], &m_data[6], &m_data[7], &m_data[8], &m_data[9], &m_data[10], &m_data[11], &m_data[12], &m_data[13], &m_data[14], &m_data[15]);
                    if (t != 16)
                    {
                        printf("error\n");
                    }

                    
                }
			}
		}
	}
	return scene;
}

std::string directoryOfFilePath(const std::string& filepath)
{
	size_t slash_pos, backslash_pos;
	slash_pos = filepath.find_last_of('/');
	backslash_pos = filepath.find_last_of('\\');

	size_t break_pos;
	if (slash_pos == std::string::npos && backslash_pos == std::string::npos) {
		return std::string();
	}
	else if (slash_pos == std::string::npos) {
		break_pos = backslash_pos;
	}
	else if (backslash_pos == std::string::npos) {
		break_pos = slash_pos;
	}
	else {
		break_pos = std::max(slash_pos, backslash_pos);
	}

	// Include the final slash                                                   
	return filepath.substr(0, break_pos + 1);
}
 void Scene::getMeshData(int id, std::vector<tinyobj::shape_t> &shapes, std::vector<tinyobj::material_t> & mats)
{

	 //std::vector<tinyobj::shape_t>       m_shapes;
	 //std::vector<tinyobj::material_t>    m_materials;
	 std::string err;
	 std::string mesh_name = mesh_names[id];
	 int b = tinyobj::LoadObj(shapes, mats,err,mesh_name.c_str());
	 return ;
}
 void Scene::getMeshData(int id)
 {

	 std::vector<tinyobj::shape_t>       m_shapes;
	 std::vector<tinyobj::material_t>    m_materials;
	 std::string err;
	 std::string mesh_name = mesh_names[id];
	 int b = tinyobj::LoadObj(m_shapes, m_materials, err, mesh_name.c_str());
	 return ;
 }
