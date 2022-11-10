#ifndef CLASSTREE_DEVICE
#define CLASSTREE_DEVICE 
#include "classTree_common.h"   
#include"random.h"
namespace classTree
{ 

    RT_FUNCTION int getLabel(tree_node* root, float3 position, float3 normal = make_float3(0.0), float3 direction = make_float3(0.0))
    {
        return tree_index(root, position, normal, direction);
        int node_id = 0;
        int loop_time = 0;
        while (root[node_id].leaf == false)
        {
            loop_time += 1;
            node_id = root[node_id](position,normal,direction);
        }
        //rtPrintf("%d %f %f %f %d\n", loop_time,position.x,position.y,position.z,node_id);
        return root[node_id].label;
    }
     
}

#endif // !CLASSTREE_HOST
