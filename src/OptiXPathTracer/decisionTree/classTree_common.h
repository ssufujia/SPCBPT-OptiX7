#ifndef CLASSTREE_COMMON
#define CLASSTREE_COMMON
#include<optix.h>
#include"../rt_function.h"
namespace classTree
{
    enum tree_node_type
    {
        type_position = 0,type_normal = 1,type_direction = 2
    };
    struct tree_node
    {
        float3 mid;
        int child[8];
        int label;
        int type;
        bool leaf;
        
        tree_node():leaf(true),label(0),type(type_position){}
        __host__ RT_FUNCTION int getChild(float3 position)
        {
            int ind = 0;
            ind += position.x > mid.x ? 1 : 0;
            ind += position.y > mid.y ? 2 : 0;
            ind += position.z > mid.z ? 4 : 0; 
            return child[ind];
        }
        //__host__ RT_FUNCTION int operator()(float3 position)
        //{
        //    return getChild(position);
        //}

        __host__ RT_FUNCTION int operator()(float3 position, float3 normal = make_float3(0),float3 direction = make_float3(0))
        {
            return type == type_position ? getChild(position) : (type == type_normal? getChild(normal):getChild(direction));
        }

    }; 
    __host__ RT_FUNCTION int tree_index(tree_node* root, float3 position,float3 normal = make_float3(0.0),float3 direction = make_float3(0.0))
    {
        int node_id = 0;
        int l_id = node_id;
        while (root[node_id].leaf == false)
        {
            l_id = node_id;
            node_id = root[node_id](position, normal, direction);
        }
        //rtPrintf("%d %f %f %f %d\n", loop_time,position.x,position.y,position.z,node_id);
 
        return root[node_id].label;
    }

    __host__ RT_FUNCTION int tree_index_node_id(tree_node* root, float3 position, float3 normal = make_float3(0.0), float3 direction = make_float3(0.0))
    {
        int node_id = 0;
        while (root[node_id].leaf == false)
        {
            node_id = root[node_id](position,normal,direction);
        }
        //rtPrintf("%d %f %f %f %d\n", loop_time,position.x,position.y,position.z,node_id);
        return node_id;
    }
    struct tree
    {
        tree_node* v;
        float3* center;
        float3 bbox_max;
        float3 bbox_min;
        int size;
        int max_label;
        bool in_gpu;
        
        tree() :size(0), in_gpu(false) {}
    };

    struct divide_weight
    {
        float3 position;
        float3 dir;
        float3 normal;
        float weight;
        __host__ RT_FUNCTION float d(const divide_weight& a, float diag2)const
        {
            float k = DIR_JUDGE;
            float3 diff = a.position - position;
            float d_a = dot(diff, diff);
            float diff_direction = dot(dir, a.dir);
            float diff_normal = dot(normal, a.normal);
            return d_a + diag2 * ((1 - diff_normal) + (1 - diff_direction) * k);
        }
    };
    struct divide_weight_with_label: divide_weight
    {
        int label;
    };


    struct VPL
    {
        float3 color;
        float3 position;
        float3 normal;

        float3 dir;
        float weight;
    };
    struct aabb
    {
        float3 aa;
        float3 bb;
        __host__ __device__
            aabb make_union(aabb& a)
        {
            aabb c;
            c.aa = fminf(a.aa, aa);
            c.bb = fmaxf(a.bb, bb);
            return c;
        }
        __host__ __device__
            float3 diag()
        {
            return aa - bb;
        }
        __host__ __device__
        float3 dir_mid()
        {
            return normalize(aa + bb);
        }
        __host__ __device__
        float half_angle_cos()
        {
            float3 a = dir_mid();
            return abs(dot(normalize(aa), a));

        }
        __host__ __device__
        float3 mid_pos()
        {
            return (aa + bb) / 2;
        }
    };
    struct lightTreeNode:VPL
    {
        aabb pos_box;
        bool leaf;
        int left_id;
        int right_id;
    };
    
    struct light_tree_api
    {
        lightTreeNode* p;
        int size; 
        __host__ RT_FUNCTION lightTreeNode& get_root()
        {
            return p[size - 1];
        }
        __host__ RT_FUNCTION lightTreeNode& operator[](int id)
        { 
            return p[id];
        }
        __host__ RT_FUNCTION int root_id()
        {
            return size - 1;
        }
    };
    struct KD_node
    {
        float3 position;
        int axis; 
        int index;
        bool valid;
    };


} 
#endif