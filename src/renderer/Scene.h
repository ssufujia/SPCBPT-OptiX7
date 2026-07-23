//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/Light.h>
#include <cuda/MaterialData.h>
#include <renderer/Aabb.h>
#include <renderer/Camera.h>
#include <renderer/Matrix.h>
#include <renderer/Preprocessor.h>
#include <renderer/Texture.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <renderer/core/renderer_config.h>

namespace sutil
{

    enum programType
    {
        rayGenProg, missProg, lightHitProg, normalHitProg, SPCBPT_SPECIAL_rayGen, progTypeNum
    };

class Scene
{
public:
    Scene();
    ~Scene();

    struct Instance
    {
        Matrix4x4                         transform;
        Aabb                              world_aabb;

        int                               mesh_idx;
    };

    struct MeshGroup
    {
        std::string                       name;

        std::vector<GenericBufferView>    indices;
        std::vector<BufferView<float3> >  positions;
        std::vector<BufferView<float3> >  normals;
        std::vector<BufferView<Vec2f> >   texcoords[GeometryData::num_textcoords];
        std::vector<BufferView<Vec4f> >   colors;

        std::vector<int32_t>              material_idx;

        OptixTraversableHandle            gas_handle = 0;
        CUdeviceptr                       d_gas_output = 0;

        Aabb                              object_aabb;
    };


    void addCamera  ( const Camera& camera            )    { m_cameras.push_back( camera );     }
    void addInstance( std::shared_ptr<Instance> instance ) { m_instances.push_back( instance ); }
    void addMesh    ( std::shared_ptr<MeshGroup> mesh )    { m_meshes.push_back( mesh );        }
    void addMaterial( const MaterialData& mtl    )         { m_materials.push_back( mtl );      }
    void addLight   ( const Light& light    )              { m_lights.push_back( light );       }
    void addBuffer  ( const uint64_t buf_size, const void* data );
    void trackBuffer( CUdeviceptr buffer ) { m_buffers.push_back( buffer ); }
    void addImage(
                const int32_t width,
                const int32_t height,
                const int32_t bits_per_component,
                const int32_t num_components,
                const void*   data
                );
    void addSampler(
                cudaTextureAddressMode address_s,
                cudaTextureAddressMode address_t,
                cudaTextureFilterMode  filter_mode,
                const int32_t          image_idx
                );
    void adoptTexture( const spcbpt::Texture& texture );

    CUdeviceptr                    getBuffer ( int32_t buffer_index  )const;
    cudaArray_t                    getImage  ( int32_t image_index   )const;
    cudaTextureObject_t            getSampler( int32_t sampler_index )const;

    void createPTXModule(std::string fileName);

    void                           finalize();
    void                           cleanup();

    Camera                                         camera()const;
    OptixPipeline                                  pipeline()const           { return m_pipeline;   } 
    const OptixShaderBindingTable*                 sbt()const                { return &m_sbt;       }
    OptixTraversableHandle                         traversableHandle() const { return m_ias_handle; }
    sutil::Aabb                                    aabb() const              { return m_scene_aabb; }
    OptixDeviceContext                             context() const           { return m_context;    }
    const std::vector<MaterialData>&               materials() const         { return m_materials;  }
    const std::vector<std::shared_ptr<MeshGroup>>& meshes() const            { return m_meshes;     }
    const std::vector<std::shared_ptr<Instance>>&  instances() const         { return m_instances;  }

    void                                           removeCurrent()           { m_meshes.clear(); m_instances.clear(); }
    size_t                                         ImagesSize() const        { return m_images.size(); }
    cudaTextureObject_t                            SamplerCurrent() const    { return m_samplers.back(); }
    size_t                                         MaterialsSize() const     { return m_materials.size(); }

    void createContext();
    void buildMeshAccels();
    void buildInstanceAccel( int rayTypeCount = RayType::RAY_TYPE_COUNT );
    void switchRaygen(std::string raygenName);
    void setSpcbptPure( bool enabled ) { m_spcbpt_pure = enabled; }
    void setEnvFilePath(std::string envFileName) { m_env_file_name = envFileName; }
    std::string getEnvFilePath()const { return m_env_file_name; }
    void setResourceRoot( std::string resourceRoot ) { m_resource_root = std::move( resourceRoot ); }
    const std::string& getResourceRoot() const { return m_resource_root; }
    void addDirectionalLight(float3 dir, float3 intensity) { return dir_lights.push_back(std::make_pair(dir,intensity)); }

    std::vector<std::pair<float3, float3>> dir_lights;//directional light
private:
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // TODO: custom geometry support

    std::vector<Camera>                      m_cameras;
    std::vector<std::shared_ptr<Instance> >  m_instances;
    std::vector<std::shared_ptr<MeshGroup> > m_meshes;
    std::vector<MaterialData>                m_materials;
    std::vector<CUdeviceptr>                 m_buffers;
    std::vector<CUdeviceptr>                 m_gas_buffers;
    std::vector<cudaTextureObject_t>         m_samplers;
    std::vector<cudaArray_t>                 m_images;
    std::vector<Light>                       m_lights;
    sutil::Aabb                              m_scene_aabb;
    //EmptyRecord                              m_rg_sbt_host;
    //EmptyRecord                              m_ms_sbt_host[RayType::RAY_TYPE_COUNT];
    //EmptyRecord                              m_hit_sbt_host[RayHitType::RAYHIT_TYPE_COUNT][RayType::RAY_TYPE_COUNT];

    OptixDeviceContext                   m_context                  = 0;
    OptixShaderBindingTable              m_sbt                      = {};
    OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    OptixPipeline                        m_pipeline                 = 0; 
    OptixModule                          m_ptx_module               = 0;
    OptixModule                          m_ptx_module_hit           = 0;

    OptixProgramGroup                    m_raygen_prog_group        = 0; 
    OptixProgramGroup                    m_raygen_prog_pretrace     = 0; 
    OptixProgramGroup                    m_radiance_miss_group      = 0;
    OptixProgramGroup                    m_occlusion_miss_group     = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixProgramGroup                    m_lightsource_hit_group    = 0;
    OptixProgramGroup                    m_occlusion_hit_group      = 0;

    OptixProgramGroup                    m_light_trace_ray_group[progTypeNum] = {};
    OptixProgramGroup                    m_SPCBPT_eye_subpath_group[progTypeNum] = {};
    OptixProgramGroup                    m_eye_subpath_group_simple[progTypeNum] = {};
    //0 for raygen, 1 for normal closest_hit, 2 for light source closest hit, 3 for miss

    OptixTraversableHandle               m_ias_handle               = 0;
    CUdeviceptr                          m_d_ias_output_buffer      = 0;
    bool                                 m_spcbpt_pure              = true;
    std::string                          m_env_file_name = {};
    std::string                          m_resource_root = {};
};


void loadScene( const std::string& filename, Scene& scene );

} // end namespace sutil
