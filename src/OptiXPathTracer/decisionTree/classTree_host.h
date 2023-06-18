#ifndef CLASSTREE_HOST
#define CLASSTREE_HOST

#include "classTree_common.h" 
#include<random>
#include<vector> 
#include<fstream>
#include<algorithm>
#include<map> 
#include<optix.h>
#include<vector>
#include"../rt_function.h"
#include"../optixPathTracer.h"
namespace classTree
{ 
    void tree_load(std::vector<tree_node>& eye, std::vector<tree_node>& light)
    {
        eye.clear();
        light.clear();
        {

            std::ifstream inFile;
            inFile.open("tree_eye.txt");
            bool leaf;
            while (inFile >> leaf)
            {
                tree_node node;
                inFile >> node.label;
                node.leaf = leaf;
                if (!leaf)
                {
                    inFile >> node.type >> node.mid.x >> node.mid.y >> node.mid.z;
                    for (int i = 0; i < 8; i++)
                    {
                        inFile >> node.child[i];
                    }
                }
                eye.push_back(node);
            }
        }
        {
            std::ifstream inFile;
            inFile.open("tree_light.txt");
            bool leaf;
            while (inFile >> leaf)
            {
                tree_node node;
                inFile >> node.label;
                node.leaf = leaf;
                if (!leaf)
                {
                    inFile >> node.type >> node.mid.x >> node.mid.y >> node.mid.z;
                    for (int i = 0; i < 8; i++)
                    {
                        inFile >> node.child[i];
                    }
                }
                light.push_back(node);
            }
        }
    }
    struct buildTreeBaseOnExistSample
    {
        struct devide_node :tree_node
        {
            std::vector<divide_weight_with_label> v;
            int depth;
            float weight;
            float correct_weight;
            int father;
            int position_depth;
            int normal_depth;
            int dir_depth;

            devide_node() :weight(0), v(0), father(0), depth(0), correct_weight(0), position_depth(0), normal_depth(0), dir_depth(0)
            {

            }

            void add_sample(divide_weight_with_label w)
            {
                v.push_back(w);
                weight += w.weight;
            }
            bool need_split()
            { 
                //printf("%f %f\n", correct_weight, weight);
                return v.size() != 0 && correct_weight < weight;
            }
        };
        std::vector<devide_node> v;
        int label_id;
        int max_label;
        std::vector<float3> block_size;
        std::vector<float3> direction_block_size;

        float3 bbox_min;// = make_float3(FLT_MAX);
        float3 bbox_max;// = make_float3(FLT_MIN);

        buildTreeBaseOnExistSample() :v(0), label_id(0)
        {
            bbox_min = make_float3(FLT_MAX);
            bbox_max = make_float3(FLT_MIN);
        }
        float split(int id)
        { 
            auto split_type = (v[id].depth % 2 == 0 || v[id].normal_depth > 3) ? tree_node_type::type_position : tree_node_type::type_normal;
            if (v[id].depth == 7 || v[id].depth == 9)
            {
                if (DIR_JUDGE)
                    split_type = tree_node_type::type_direction;
            }
            //split_type = tree_node_type::type_position;
            int back = v.size();
            //devide_node& node = v[id];

            v[id].leaf = false;
            float3 inch;
            if (split_type == tree_node_type::type_position)
            {
                inch = block_size[v[id].position_depth + 1];
            }
            else if (split_type == tree_node_type::type_normal)
            {
                inch = direction_block_size[v[id].normal_depth + 1];
            }
            else if (split_type == tree_node_type::type_direction)
            {
                inch = direction_block_size[v[id].dir_depth + 1];
            }

            float3 mid;
            
            if (v[id].normal_depth == 0 && split_type == tree_node_type::type_normal)
            { 
                mid = make_float3(0.0); 
            }
            else if (v[id].dir_depth == 0 && split_type == tree_node_type::type_direction)
            {
                mid = make_float3(0.0);
            }
            else if (v[id].position_depth == 0)
            {
                mid = v[id].mid;
            }
            else
            {  
                int L_id = id;
                int t_id = v[id].father;

                while (t_id != 0 && v[t_id].type != split_type)
                {
                    L_id = t_id;
                    t_id = v[t_id].father;
                }

                mid = v[t_id].mid;
                int c_id_local = 0;
                for (; c_id_local < 8; c_id_local++)
                {
                    if (v[t_id].child[c_id_local] == L_id)break;
                }

                float3 delta_mid = make_float3(
                    (c_id_local >> 0) % 2 == 0 ? -inch.x : inch.x,
                    (c_id_local >> 1) % 2 == 0 ? -inch.y : inch.y,
                    (c_id_local >> 2) % 2 == 0 ? -inch.z : inch.z
                );
                mid += delta_mid;
            }
            v[id].mid = mid;
            v[id].type = split_type;

            for (int i = 0; i < 8; i++)
            {
                v[id].child[i] = back + i;
                v.push_back(devide_node());
                //printf("sizeeee %d %d\n", v[id].v.size());

                v.back().father = id;
                v.back().depth = v[id].depth + 1;

                //float3 delta_mid = make_float3(
                //    (i >> 0) % 2 == 0 ? -inch.x : inch.x,
                //    (i >> 1) % 2 == 0 ? -inch.y : inch.y,
                //    (i >> 2) % 2 == 0 ? -inch.z : inch.z
                //);
                //v.back().mid = mid + delta_mid;
                v.back().label = v[id].label;
                
                v.back().position_depth = v[id].position_depth + (split_type == tree_node_type::type_position);
                v.back().normal_depth = v[id].normal_depth + (split_type == tree_node_type::type_normal);
                v.back().dir_depth = v[id].dir_depth + (split_type == tree_node_type::type_direction);
                
            }
            for (auto p = v[id].v.begin(); p != v[id].v.end(); p++)
            {
                // printf("%d\n", v[v[id](p->position)]);
                //printf("A %d",id);
                v[v[id](p->position,p->normal,p->dir)].add_sample(*p);
                //printf("B");
            }

            float n_correct_weight = 0.0;
            for (int i = 0; i < 8; i++)
            {
                color(v[id].child[i]); 
                n_correct_weight += v[v[id].child[i]].correct_weight;
            }
            v[id].weight = 0;
            v[id].v.clear();
            return n_correct_weight;
        } 
        template<typename T>
        void para_initial(std::vector<T>& samples, int max_depth)
        {
            float unoramlize_weight = 0.0;
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                unoramlize_weight += p->weight;
                //printf("%f\n", p->weight);
                bbox_min = fminf(bbox_min, p->position);
                bbox_max = fmaxf(bbox_max, p->position);
            }
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                p->weight /= unoramlize_weight;
            }
            //printf("unormal weight %f\n", unoramlize_weight);
            float3 bbox_block = (bbox_max - bbox_min) ;
        
            for (int i = 0; i < max_depth + 10; i++)
            {
                block_size.push_back(bbox_block);
                bbox_block /= 2;
            }
            float3 direction_block = make_float3(2.0);
            for (int i = 0; i < 15; i++)
            {
                direction_block_size.push_back(direction_block);
                direction_block /= 2;
            }
        }

        void color(int id)
        {
            auto& t = v[id];
            if (t.v.size() == 0)
            {
                t.correct_weight = 0.0; 
                return;
            }
            bool need_split = false;
            t.label = t.v[0].label;
            for (int i = 0; i < t.v.size(); i++)
            {
                if (t.v[i].label != t.label)
                {
                    need_split = true;
                    break;
                }
            }
            if (need_split)
            {
                std::vector<float> weights(max_label + 1,0);
                float max_weight = 0.0;
                int max_weight_id = t.label;
                for (int i = 0; i < t.v.size(); i++)
                {
                    weights[t.v[i].label] += t.v[i].weight;
                    if (max_weight < weights[t.v[i].label])
                    {
                        max_weight = weights[t.v[i].label];
                        max_weight_id = t.v[i].label;

                    } 
                } 
                t.label = max_weight_id;
                t.correct_weight = max_weight;

            } 
            else  
            {
                t.correct_weight = t.weight;
            }
        }
         
        template<typename T>
        float get_position_variance(std::vector<T>& pos, int it)
        {           
            float3 mean = make_float3(0.0);
            for (int i = 0; i < it; i++)
            {
                mean += pos[i].position / it;
            }
            float3 var = make_float3(0);
            for (int i = 0; i < it; i++)
            {
                float3 diff = mean - pos[i].position;
                var += diff * diff / (it - 1);
            }
            return max(var.x, max(var.y, var.z));
        }
        tree operator()(std::vector<divide_weight>& samples, int subspaceSize, int labelBias = 0)
        {
            std::vector<divide_weight> centers;
            std::vector<divide_weight_with_label> labeled_samples;
            float weight_sum = 0;
            float scene_diversity2 = get_position_variance(samples,samples.size());
            printf("scene diversity %f\n", scene_diversity2);
            for (auto p : samples)
            {
                weight_sum += p.weight;
            } 
            float acc = 0;
            for (auto p : samples)
            {
                acc += p.weight;
                if (acc > weight_sum / subspaceSize)
                {
                    acc -= weight_sum / subspaceSize;
                    centers.push_back(p);
                }
            }
            for (auto p : samples)
            {
                float min_distance = FLT_MAX;
                int subspaceId = 0;
                for (int i = 0; i < centers.size(); i++)
                {
                    if (p.d(centers[i],scene_diversity2) < min_distance)
                    {
                        min_distance = p.d(centers[i], scene_diversity2);
                        subspaceId = i + labelBias;
                    }
                }
                divide_weight_with_label t;
                t.dir = p.dir; t.normal = p.normal; t.position = p.position; t.weight = p.weight;
                t.label = subspaceId;
                //printf("centroid number %d subspace id %d\n", centers.size(), subspaceId);
                labeled_samples.push_back(t);
            }
            return operator()(labeled_samples, 0.99);
        }
        
        tree operator()(std::vector<divide_weight_with_label>& samples,float threshold, int max_depth = 12, int refer_num_class = 100)
        {
            max_label = 0;
            //weight normalization
            para_initial(samples, max_depth);

            float max_weight = 0.1 / refer_num_class;
            float min_weight = 1.0 / refer_num_class;

            v.push_back(devide_node());
            v[0].v = std::vector<divide_weight_with_label>(samples.begin(), samples.end());
            v[0].weight = 1;
            v[0].mid = (bbox_max + bbox_min) / 2;
            printf("bounding box max: %f %f %f\n", bbox_max.x, bbox_max.y, bbox_max.z);
            printf("bounding box min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);

            for (int i = 0; i < samples.size(); i++)
                max_label = max(samples[i].label, max_label);
            color(0);
            //printf("root correct_weight %f\n", v[0].correct_weight);
            float c_w = v[0].correct_weight;

            for (int i = 0; i < v.size(); i++)
            {
                if (v[i].need_split() && v[i].depth < max_depth && threshold>c_w)
                {
                    c_w -= v[i].correct_weight;
                    c_w += split(i);

                } 
            }
            //printf("bbox_min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);
            //printf("%d", v[0].v.size());
            //exit(0);

            int valid_id = 0;
            int sum_id = 0;

            for (int t = 0; t < v.size(); t++)
            {
                if (!v[t].leaf)continue;
                for (int j = 0; j < v[t].v.size(); j++)
                {
                    if (v[t].label == v[t].v[j].label)
                    {
                        valid_id++;
                    }
                    sum_id++;
                }
            }
            printf("acc:%d/%d %e %e\n", valid_id, sum_id, c_w, float(valid_id) / sum_id);

            tree_node* p = new tree_node[v.size()];
            float3* centers = new float3[max_label + 1];
            for (int i = 0; i < max_label + 1; i++)
            {
                centers[i] = make_float3(0.0);
            }
            float* center_weight = new float[max_label + 1];
            for (int i = 0; i < max_label + 1; i++)center_weight[i] = 0;

            for (int i = 0; i < v.size(); i++)
            {
                p[i] = v[i];
                if (v[i].leaf == true)
                {
                    float tt = center_weight[v[i].label];
                    float3 ttt = centers[v[i].label];

                    center_weight[v[i].label] += v[i].weight;
                    centers[v[i].label] += v[i].weight * v[i].mid;

                    float3 t = centers[v[i].label] / center_weight[v[i].label];
                }
            }
            for (int i = 0; i < max_label + 1; i++)
            {
                centers[i] = center_weight[i] == 0.0 ? make_float3(0.0) : centers[i] / center_weight[i];
            } 
            delete[] center_weight;
            tree t;
            t.v = p;
            t.center = centers;
            t.size = v.size();
            t.in_gpu = false;
            t.max_label = max_label;
            t.bbox_max = bbox_max;
            t.bbox_min = bbox_min;
            printf("class tree building complete:size %d and max-label %d\n", t.size, t.max_label);
            return t;
        }

    };


} 
#endif // !CLASSTREE_HOST
