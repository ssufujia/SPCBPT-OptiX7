#ifndef PG_COMMON
#define PG_COMMON 
#include"rt_function.h"
#include <optix.h>
#include"random.h"
//#include"sutil/vec_math.h"
//using namespace optix;
namespace path_guiding
{

    struct PG_training_mat
    {
        float3 position;
        float2 uv;
        //float2 uv_light;
        float lum;
        bool valid;
        //bool light_source;
        //int light_id; 
        RT_FUNCTION __host__ PG_training_mat() :valid(false) {}
    };

    struct quad_tree_node
    {
        float2 m_min;
        float2 m_max;
        float2 m_mid;
        int child[4];
        int quad_tree_id;
        int count;
        bool leaf;
        float lum;
        RT_FUNCTION quad_tree_node(float2 m_min, float2 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2), leaf(true), lum(0) {}
        RT_FUNCTION int whichChild(float2 uv)
        {

            int base = 1;
            int index = 0;
            if (uv.x > m_mid.x)
            {
                index += base;
            }
            base *= 2;
            if (uv.y > m_mid.y)
            {
                index += base;
            }
            base *= 2;
            return index;
        }

        RT_FUNCTION int getChild(int index)
        {
            return child[index];
        }
        RT_FUNCTION int traverse(float2 uv)
        {
            return getChild(whichChild(uv));
        }
        RT_FUNCTION float area()
        {
            float2 d = m_max - m_min;
            return d.x * d.y;
        }
    };

    struct Spatio_tree_node
    {
        float3 m_min;
        float3 m_max;
        float3 m_mid;
        bool leaf;
        int count;
        int child[8];
        int quad_tree_id;
        RT_FUNCTION Spatio_tree_node(float3 m_min, float3 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2), leaf(true), count(0), quad_tree_id(0) {}
        RT_FUNCTION int whichChild(float3 pos)
        {
            int base = 1;
            int index = 0;
            if (pos.x > m_mid.x)
            {
                index += base;
            }
            base *= 2;
            if (pos.y > m_mid.y)
            {
                index += base;
            }
            base *= 2;
            if (pos.z > m_mid.z)
            {
                index += base;
            }
            base *= 2;
            return index;
        }
        RT_FUNCTION int getChild(int index)
        {
            return child[index];
        }
        RT_FUNCTION int traverse(float3 pos)
        {
            return getChild(whichChild(pos));
        }
    };
#define GMM_CORE 3


    struct PathGuiding_params
    {
        //For PreTracing
        //PG_training_mat* mats;
        //int mats_length;
        Spatio_tree_node* spatio_trees;
        quad_tree_node* quad_trees;
        //float epsilon_lum = 0.001; 
        float epsilon_lum; 
        int pg_enable;
        float guide_ratio;

        RT_FUNCTION __host__ float lerp(const float a, const float b, const float t)
        {
            return a + t * (b - a);
        }
        RT_FUNCTION float3 uv2dir(float2 uv)
        {
            float3 dir;

            float theta, phi, x, y, z;
            float u = uv.x;
            float v = uv.y;

            phi = asinf(2 * v - 1.0);
            theta = u / (0.5 * M_1_PIf) - M_PIf;

            dir.y = cos(M_PIf * 0.5f - phi);
            dir.x = cos(phi) * sin(theta);
            dir.z = cos(phi) * cos(theta);
            return dir;
        }
        RT_FUNCTION float2 dir2uv(float3 dir)
        {
            float2 uv;
            float theta = atan2f(dir.x, dir.z);
            float phi = M_PIf * 0.5f - acosf(dir.y);
            float u = (theta + M_PIf) * (0.5f * M_1_PIf);
            float v = 0.5f * (1.0f + sin(phi));
            uv = make_float2(u, v);

            return uv;
        }
        RT_FUNCTION int quad_random_walk(int id, float lum)
        {
            if (quad_trees[id].leaf == false)
            {
                float lum_header = quad_trees[id].lum;
                float lum_sum =
                    quad_trees[quad_trees[id].child[0]].lum +
                    quad_trees[quad_trees[id].child[1]].lum +
                    quad_trees[quad_trees[id].child[2]].lum +
                    quad_trees[quad_trees[id].child[3]].lum;
                for (int i = 0; i < 4; i++)
                {
                    int c_id = quad_trees[id].child[i];
                    if (lum < quad_trees[c_id].lum)
                    {
                        //rtPrintf("%f %f %f\n",lum_header, lum_sum, lum);

                        return c_id;
                    }
                    else
                    {
                        lum -= quad_trees[c_id].lum;
                    }
                }
            }
            //rtPrintf("%f %f %f %f %f\n", quad_trees[id].lum, quad_trees[quad_trees[id].child[0]].lum,
            //    quad_trees[quad_trees[id].child[1]].lum, quad_trees[quad_trees[id].child[2]].lum,
            //    quad_trees[quad_trees[id].child[3]].lum);
            return id;
        }
        RT_FUNCTION int getStreeId(float3 position)
        {
            int c_id = 0;
            while (spatio_trees[c_id].leaf == false)
            {
                c_id = spatio_trees[c_id].traverse(position);
            }
            return c_id;
        }

        RT_FUNCTION float pdf(float3 position, float3 dir)
        {
            int c_id = getStreeId(position);
            int q_id = spatio_trees[c_id].quad_tree_id;
            if (quad_trees[q_id].lum < epsilon_lum)
            {
                return 1.0 / (4 * M_PI);
            }
            float lum_sum = quad_trees[q_id].lum;
            float2 uv = dir2uv(dir);
            while (quad_trees[q_id].leaf == false)
            {
                q_id = quad_trees[q_id].traverse(uv);
            }
            float pdf1 = quad_trees[q_id].lum / lum_sum;
            float area = quad_trees[q_id].area();
            //if(q_id == 2)
              //  rtPrintf("%d %f %f %f\n",q_id, pdf1, quad_trees[2].lum,lum_sum);
            return pdf1 / (area * 4 * M_PI);
        }
        RT_FUNCTION float3 sample(unsigned int& seed, float3 position)
        {
            int c_id = getStreeId(position);
            int q_id = spatio_trees[c_id].quad_tree_id;
            float lum_sum = quad_trees[q_id].lum;
            if (quad_trees[q_id].lum < epsilon_lum)
            {
                float t1 = rnd(seed);
                float t2 = rnd(seed);
                float2 uv = make_float2(t1, t2);
                return uv2dir(uv);
            }
            while (quad_trees[q_id].leaf == false)
            {
                float t_lum = rnd(seed) * quad_trees[q_id].lum;
                q_id = quad_random_walk(q_id, t_lum);
                //            q_id = quad_trees[q_id].child[optix::clamp(static_cast<int>(floorf(rnd(seed) * 4)), 0, 3)];
            }
            float t1 = rnd(seed);
            float t2 = rnd(seed);
            float2 uv = make_float2( 
                lerp(quad_trees[q_id].m_min.x, quad_trees[q_id].m_max.x, t1),
                lerp(quad_trees[q_id].m_min.y, quad_trees[q_id].m_max.y, t2));
            //float pdf2 = pdf(position, uv2dir(uv));
            //rtPrintf("%f %f %f \n", quad_trees[q_id].lum / lum_sum /(quad_trees[q_id].area() * 4 * M_PI),pdf2, quad_trees[q_id].area());
            return uv2dir(uv);
        }




        RT_FUNCTION int quad_tree_traverse(quad_tree_node* p, float2 uv, int root_id)
        {
            int c_id = root_id;
            while (p[c_id].leaf == false)
            {
                c_id = p[c_id].traverse(uv);
            }
            return c_id;
        }
        RT_FUNCTION int quad_tree_random_walk(quad_tree_node* p, int id, float lum)
        {
            if (p[id].leaf == false)
            {
                float lum_header = quad_trees[id].lum;
                float lum_sum =
                    p[p[id].child[0]].lum +
                    p[p[id].child[1]].lum +
                    p[p[id].child[2]].lum +
                    p[p[id].child[3]].lum;


                for (int i = 0; i < 4; i++)
                {
                    int c_id = p[id].child[i];
                    if (lum < p[c_id].lum)
                    {
                        return c_id;
                    }
                    else
                    {
                        lum -= p[c_id].lum;
                    }
                }
            }
            printf("random walk error %f %f %f %f %f \n", p[p[id].child[0]].lum, p[p[id].child[1]].lum, p[p[id].child[2]].lum, p[p[id].child[3]].lum, quad_trees[id].lum);
            return p[id].child[0];
        }
        RT_FUNCTION float quad_tree_pdf(quad_tree_node* p, float2 uv, int id)
        {
            if (p[id].lum < epsilon_lum)
            {
                return 1.0;
            }
            int leaf_id = quad_tree_traverse(p, uv, id);
            return p[leaf_id].lum / p[id].lum / p[leaf_id].area();
        }
        RT_FUNCTION float2 quad_sample(quad_tree_node* p, unsigned int& seed, int id)
        {
            float lum_sum = p[id].lum;
            if (p[id].lum < epsilon_lum)
            {
                float t1 = rnd(seed);
                float t2 = rnd(seed);
                float2 uv = make_float2(t1, t2);
                return uv;
            }
            while (p[id].leaf == false)
            {
                float t_lum = rnd(seed) * p[id].lum;
                id = quad_tree_random_walk(p, id, t_lum);
                //            q_id = quad_trees_light[q_id].child[optix::clamp(static_cast<int>(floorf(rnd(seed) * 4)), 0, 3)];
            }
            float t1 = rnd(seed);
            float t2 = rnd(seed);
            float2 uv = make_float2(
                lerp(p[id].m_min.x, p[id].m_max.x, t1),
                lerp(p[id].m_min.y, p[id].m_max.y, t2));
            //float pdf2 = pdf(position, uv2dir(uv));
            //rtPrintf("%f %f %f \n", quad_trees_light[q_id].lum / lum_sum /(quad_trees_light[q_id].area() * 4 * M_PI),pdf2, quad_trees_light[q_id].area());
            return uv;
        }
    };
}
typedef path_guiding::PathGuiding_params PG_params;
#endif