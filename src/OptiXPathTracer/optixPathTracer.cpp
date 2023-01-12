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

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include<sutil/Scene.h> 
#include"sceneLoader.h"
#include"scene_shift.h"
#include<sutil/Record.h>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"cuda_thrust/device_thrust.h"
#include"decisionTree/classTree_host.h"
#include"frame_estimation.h"
using namespace std;
 

bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
MyParams* d_params = nullptr; 
sutil::Camera    camera;
sutil::Trackball trackball;
MyParams   params = {};
LightTraceParams& lt_params = params.lt;
PreTraceParams& pr_params = params.pre_tracer;
subspaceMacroInfo& subspaceInfo = params.subspace_info;
int32_t                 width = 1920;
int32_t                 height = 1000; 
// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 1; 

std::vector< std::string> render_alg = { std::string("pt"), std::string("SPCBPT_eye")};
int render_alg_id = 0;
bool one_frame_render_only = false;
float render_fps = 60;
//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

  

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    MyParams* params = static_cast<MyParams*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


void img_save()
{
    sutil::ImageBuffer outputbuffer;

    auto host_buffer = MyThrustOp::copy_to_host(params.frame_buffer, params.height * params.width);
    outputbuffer.data = host_buffer.data();
    outputbuffer.width = params.width;
    outputbuffer.height = params.height;
    outputbuffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::saveImage("image.png", outputbuffer, true);


    auto p = MyThrustOp::copy_to_host(params.accum_buffer, params.height * params.width);
    std::ofstream outFile;
    outFile.open("./standard.txt");

    outFile << params.width << " " << params.height << std::endl;
    for (int i = 0; i < params.width * params.height; i++)
    {
        outFile << p[i].x << " ";
        outFile << p[i].y << " ";
        outFile << p[i].z << " ";
        outFile << p[i].w << std::endl;

    }
    outFile.close();

}

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }

        else if (key == GLFW_KEY_C)
        {
            printf("Camera Info:\n");
            printf("up      %f %f %f\n", camera.up().x, camera.up().y, camera.up().z);
            printf("eye     %f %f %f\n", camera.eye().x, camera.eye().y, camera.eye().z);
            printf("lookat  %f %f %f\n", camera.lookat().x, camera.lookat().y, camera.lookat().z);
            // toggle UI draw
        }

        else if (key == GLFW_KEY_S)
        {
            img_save();
        }
        else if (key == GLFW_KEY_SPACE)
        {
            render_alg_id++;
            if (render_alg_id >= render_alg.size())
            {
                render_alg_id = 0;
            }

            camera_changed = true;
            resize_dirty = true;
        } 
        else if (key == GLFW_KEY_P)
        {
            one_frame_render_only = one_frame_render_only ^ true;
        }

    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }

    if (key == GLFW_KEY_W)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 dir = normalize(lookat - eye);
        float speed = 0.5;
        eye += dir / render_fps * speed;
        lookat += dir / render_fps * speed;
        camera.setEye(eye);
        camera.setLookat(lookat);
        camera_changed = true;
        resize_dirty = true;
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 1920x1000\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams(const sutil::Scene& scene) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width * height * sizeof(float4)
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();
    
    std::vector<MaterialData::Pbr> material_vec;
    for (int i = 0; i < scene.materials().size(); i++)
    {
        material_vec.push_back(scene.materials()[i].pbr);
    }
    
    params.materials = HostToDeviceBuffer(material_vec.data(), material_vec.size());

    // TODO: add light support to sutil::Scene
    //std::vector<Light> lights(2);
    //lights[0].type = Light::Type::POINT;
    //lights[0].point.color = { 1.0f, 1.0f, 0.8f };
    //lights[0].point.intensity = 5.0f;
    //lights[0].point.position = scene.aabb().center() + make_float3(loffset);
    //lights[0].point.falloff = Light::Falloff::QUADRATIC;
    //lights[1].type = Light::Type::POINT;
    //lights[1].point.color = { 0.8f, 0.8f, 1.0f };
    //lights[1].point.intensity = 3.0f;
    //lights[1].point.position = scene.aabb().center() + make_float3(-loffset, 0.5f * loffset, -0.5f * loffset);
    //lights[1].point.falloff = Light::Falloff::QUADRATIC;

    //params.lights.count = static_cast<uint32_t>(lights.size());
    //CUDA_CHECK(cudaMalloc(
    //    reinterpret_cast<void**>(&params.lights.data),
    //    lights.size() * sizeof(Light)
    //));
    //CUDA_CHECK(cudaMemcpy(
    //    reinterpret_cast<void*>(params.lights.data),
    //    lights.data(),
    //    lights.size() * sizeof(Light),
    //    cudaMemcpyHostToDevice
    //));

    params.miss_color = make_float3(0.1f);
    //params.scene_epsilon = 1e-3f;
    //params.scene_maximum = 1e16f;
    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(MyParams)));

    params.handle = scene.traversableHandle();
    subspaceInfo.eye_tree = nullptr;
    subspaceInfo.light_tree = nullptr;
    subspaceInfo.Q = nullptr;
    subspaceInfo.CMFGamma = nullptr;
}

 

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, MyParams& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                params.width * params.height * sizeof( float4 )
                ) );
}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(width, height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width * height * sizeof(float4)
    ));
}

void handleCameraUpdate(MyParams& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    //printf("camera ratio%f, %d %d\n", camera.aspectRatio(), params.width, params.height);
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
    /*
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        */

}
void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, MyParams& params)
{
    // Update params on device
    if (camera_changed || resize_dirty || one_frame_render_only)
        params.subframe_index = 0;
     
    handleCameraUpdate( params );
    handleResize( output_buffer, params );
     
}

std::vector<int> surroundsIndex(int index, const envInfo& infos)
{
    int2 coord = infos.index2coord(index);
    std::vector<int2> s_coord;
    for (int dx = -2; dx <= 2; dx++)
    {
        for (int dy = -2; dy <= 2; dy++)
        {
            if (abs(dx) + abs(dy) <= 2)
                s_coord.push_back(coord + make_int2(dx, dy));
        }
    }
    std::vector<int> ans;
    for (int i = 0; i < s_coord.size(); i++)
    {
        if (s_coord[i].x >= 0 && s_coord[i].y >= 0 && s_coord[i].x < infos.width && s_coord[i].y < infos.height)
        {
            ans.push_back(infos.coord2index(s_coord[i]));
        }
    }
    return ans;
}
thrust::device_ptr<float> envMapCMFBuild(float4* lum, int size, const envInfo& infos)
{ 

    std::vector<float> p2(size);
    float uniform_rate = 0.25;
    float uniform_pdf = 1.0 / size;
    auto p = lum;
    for (int i = 0; i < size; i++)
    {
        auto s_index = surroundsIndex(i, infos);
        p2[i] = float3weight(make_float3(p[i]));
        for (auto ii = s_index.begin(); ii != s_index.end(); ii++)
        {
            p2[i] += float3weight(make_float3(p[*ii])) / s_index.size();
        }
        if (i >= 1)
        {
            p2[i] += p2[i - 1];
        }
    }
    float sum = p2[size - 1];
    for (int i = 0; i < size; i++)
    {
        p2[i] /= sum;
        p2[i] = p2[i] * (1 - uniform_rate) + (uniform_pdf * (i + 1) * uniform_rate);
    }
    return MyThrustOp::envMapCMFBuild(p2.data(), size);
}
void env_params_setup(const sutil::Scene& scene)
{ 
    if (scene.getEnvFilePath() == std::string(""))
    {
        params.sky.valid = false;
        return;
    }
    printf("load and build sampling cmf from file %s\n",scene.getEnvFilePath());
    HDRLoader hdr_env((string(SAMPLES_DIR) + string("/data/") + scene.getEnvFilePath()));

    float3 default_color = make_float3(1.0);
    params.sky.height = hdr_env.height();
    params.sky.width = hdr_env.width();
    params.sky.divLevel = sqrt(0.5 * NUM_SUBSPACE_LIGHTSOURCE);
    params.sky.ssBase = 0;
    params.sky.size = hdr_env.height() * hdr_env.width();



    float4* hdr_m_raster = reinterpret_cast<float4*>(hdr_env.raster());
    for (int i = 0; i < scene.dir_lights.size(); i++)
    {
        auto& dir_light = scene.dir_lights[i];
        float3 dir = dir_light.first;
        dir.y = -dir.y;
        auto uv = dir2uv(-dir);
        auto coord = params.sky.uv2coord(uv);
        auto index = params.sky.coord2index(coord);
        hdr_m_raster[index] += make_float4(dir_light.second * params.sky.size / (4 * M_PI), 0.0);
        printf("Add directional light %f %f %f in index %d\n", dir_light.first.x, dir_light.first.y, dir_light.first.z, index);
    }
    auto env_tex = hdr_env.loadTexture(default_color, nullptr);
    params.sky.tex = env_tex.texture;
    params.sky.cmf = thrust::raw_pointer_cast(envMapCMFBuild(hdr_m_raster, hdr_env.height() * hdr_env.width(), params.sky));
    params.sky.center = scene.aabb().center();
    params.sky.r = length(scene.aabb().m_min - scene.aabb().m_max);
    params.sky.valid = true;
    params.sky.light_id = params.lights.count - 1;
}
void lt_params_setup(const sutil::Scene& scene)
{
    lt_params.M_per_core = 100;
    lt_params.core_padding = 800;
    lt_params.num_core = 1000;
    lt_params.M = lt_params.M_per_core * lt_params.num_core;
    lt_params.launch_frame = 0;

    BDPTVertex* LVC_ptr;
    bool* valid_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&LVC_ptr),   sizeof(BDPTVertex) * lt_params.get_element_count()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&valid_ptr), sizeof(bool)       * lt_params.get_element_count())); 
    lt_params.ans = LVC_ptr;// BufferView<BDPTVertex>(LVC_ptr, lt_params.get_element_count());
    lt_params.validState = valid_ptr;// BufferView<bool>(valid_ptr, lt_params.get_element_count());
    //params.lt = lt_params; 
}

void preTracer_params_setup(const sutil::Scene& scene)
{
    pr_params.num_core = 10000;
    pr_params.padding = 10;
    pr_params.iteration = 0;
    preTracePath*       pretrace_path_ptr;
    preTraceConnection* pretrace_conn_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pretrace_path_ptr), sizeof(preTracePath)       * pr_params.num_core));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pretrace_conn_ptr), sizeof(preTraceConnection) * pr_params.get_element_count()));
    pr_params.paths = pretrace_path_ptr;
    pr_params.conns = pretrace_conn_ptr;
}
void launchLightTrace(sutil::Scene& scene)
{
    lt_params.launch_frame += 1;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(MyParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    scene.switchRaygen(std::string("light trace"));
    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        0,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        scene.sbt(),
        lt_params.num_core,
        1,
        1
    ));
    CUDA_SYNC_CHECK();  
}
void launchLVCTrace(sutil::Scene& scene)
{
    launchLightTrace(scene);
    auto p_v = thrust::device_pointer_cast(params.lt.ans);
    auto p_valid = thrust::device_pointer_cast(params.lt.validState);
    auto sampler = MyThrustOp::LVC_Process(p_v, p_valid, params.lt.get_element_count());
    params.sampler = sampler;
    sampler = MyThrustOp::LVC_Process_glossyOnly(p_v, p_valid, params.lt.get_element_count(), params.materials);
    params.sampler.glossy_count = sampler.glossy_count;
    params.sampler.glossy_index = sampler.glossy_index;
    params.sampler.glossy_subspace_bias = sampler.glossy_subspace_bias;
    params.sampler.glossy_subspace_num = sampler.glossy_subspace_num;

}
int launchPretrace(sutil::Scene& scene)
{
    pr_params.iteration += 1;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(MyParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    scene.switchRaygen(std::string("pretrace"));
    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        0,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        scene.sbt(),
        pr_params.num_core,
        1,
        1
    ));
    CUDA_SYNC_CHECK();
    int validSample = MyThrustOp::valid_sample_gather(
        thrust::device_pointer_cast(pr_params.paths), pr_params.num_core,
        thrust::device_pointer_cast(pr_params.conns), pr_params.get_element_count()
        );
    return validSample;
}
void preprocessing(sutil::Scene& scene)
{
    printf("BDPTVertex Size %d\n", sizeof(BDPTVertex));
    const int target_sample_count = 1000000;
    int current_sample_count = 0;
    while (current_sample_count < target_sample_count)
    {
        current_sample_count += launchPretrace(scene);
    }

    MyThrustOp::sample_reweight();
    auto unlabeled_samples = MyThrustOp::get_weighted_point_for_tree_building(true, 10000);
    auto h_eye_tree = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, min(300, NUM_SUBSPACE), 0);

    unlabeled_samples = MyThrustOp::get_weighted_point_for_tree_building(false, 10000);
    auto h_light_tree = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, min(500, NUM_SUBSPACE - NUM_SUBSPACE_LIGHTSOURCE), 0);

    auto d_DecisionTree = MyThrustOp::eye_tree_to_device(h_eye_tree.v, h_eye_tree.size);
    subspaceInfo.eye_tree = d_DecisionTree;
    d_DecisionTree = MyThrustOp::light_tree_to_device(h_light_tree.v, h_light_tree.size);
    subspaceInfo.light_tree = d_DecisionTree;
    //{
    //    std::vector<classTree::tree_node> eye_load, light_load;
    //    classTree::tree_load(eye_load, light_load); 
    //    printf("load tree size %d %d\n", eye_load.size(), light_load.size());
    //    auto d_DecisionTree = MyThrustOp::eye_tree_to_device(eye_load.data(), eye_load.size());
    //    subspaceInfo.eye_tree = d_DecisionTree;
    //    d_DecisionTree = MyThrustOp::light_tree_to_device(light_load.data(), light_load.size());
    //    subspaceInfo.light_tree = d_DecisionTree; 
    //}

    const int target_Q_samples = 2000000;
    int current_Q_samples = 0;
    thrust::device_ptr<float> Q_star = nullptr;
    while (current_Q_samples < target_Q_samples)
    {
        launchLightTrace(scene);
        auto p_v = thrust::device_pointer_cast(params.lt.ans);
        auto p_valid = thrust::device_pointer_cast(params.lt.validState); 
        current_Q_samples += MyThrustOp::preprocess_getQ(p_v, p_valid, params.lt.get_element_count(), Q_star);
    }
    MyThrustOp::Q_zero_handle(Q_star); 
    MyThrustOp::node_label(subspaceInfo.eye_tree, subspaceInfo.light_tree);

    thrust::device_ptr<float> Gamma;
    //MyThrustOp::load_Q_file(Q_star);
    MyThrustOp::build_optimal_E_train_data(target_sample_count);
    MyThrustOp::preprocess_getGamma(Gamma);
    MyThrustOp::train_optimal_E(Gamma);

    //MyThrustOp::load_Gamma_file(Gamma); 
    //MyThrustOp::train_optimal_E(Gamma);

    subspaceInfo.Q = thrust::raw_pointer_cast(Q_star);
    subspaceInfo.CMFGamma = thrust::raw_pointer_cast(MyThrustOp::Gamma2CMFGamma(Gamma));

    thrust::device_ptr<float> CausticGamma;
    MyThrustOp::preprocess_getGamma(CausticGamma, true);
    subspaceInfo.CMFCausticGamma = thrust::raw_pointer_cast(MyThrustOp::Gamma2CMFGamma(CausticGamma, true));

    thrust::device_ptr<float> CausticRatio;
    MyThrustOp::get_caustic_frac(CausticRatio);
    subspaceInfo.caustic_ratio = thrust::raw_pointer_cast(CausticRatio);
}
void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::Scene& scene)
{
    //printf("subframe id %d\n", params.subframe_index);
    scene.switchRaygen(render_alg[render_alg_id]);
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(MyParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        scene.sbt(),
        params.width,  // launch width
        params.height, // launch height
        1       // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

 
void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

 

void initCameraState(const sutil::Scene& scene)
{
    camera = scene.camera();
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}
 
 
 
//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------ 
int main( int argc, char* argv[] )
{ 
    //Cthrust;
    //PathTracerState state;
    //state.params.width                             = 1920;
    //state.params.height                            = 1000;
    params.width = 1920;
    params.height = 1000;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            params.width  = w;
            params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
//        string scenePath = string(SAMPLES_DIR) + string("/data/house/house_uvrefine2.scene"); 
//         string scenePath = string(SAMPLES_DIR) + string("/data/cornell_box/cornell_icoBall.scene");
         string scenePath = string(SAMPLES_DIR) + string("/data/cornell_box/cornell_refract.scene");
//         string scenePath = string(SAMPLES_DIR) + string("/data/glossy_kitchen/glossy_kitchen.scene");
//        string scenePath = string(SAMPLES_DIR) + string("/data/glassroom/glassroom_simple.scene");
//        string scenePath = string(SAMPLES_DIR) + string("/data/hallway/hallway_env2.scene");


        auto myScene = LoadScene(scenePath.c_str()); 
        
        myScene->getMeshData(0);
        //cout << scenePath << std::endl;
        
        sutil::Scene TScene;
        //char scene_path2[] = "D:/optix7PlayGround/OptiX SDK 7.5.0/SDK/data/house/Victorian House Blendswap.gltf";
        //sutil::loadScene(scene_path2, TScene); 

        LightSource_shift(*myScene, params, TScene);
        Scene_shift(*myScene, TScene);
        
        TScene.finalize();
        
        //initCameraState();

        //
        // Set up OptiX state
        //
        //createContext( state );
        //buildMeshAccel( state );
        //createModule( state );
        //createProgramGroups( state );
        //createPipeline( state );
        //createSBT( state );
        //initLaunchParams( state );
        OPTIX_CHECK(optixInit()); // Need to initialize function table
        initCameraState(TScene);
        //initCameraState(*myScene);
        initLaunchParams(TScene);
        lt_params_setup(TScene);
        preTracer_params_setup(TScene);
        env_params_setup(TScene);
        //pre tracing
        { 
            handleCameraUpdate(params);
            preprocessing(TScene);
        }

        //if( outfile.empty() )
        if(true)
        {
            GLFWwindow* window = sutil::initUI( "optixPathTracer", width, height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &params );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    if(render_alg[render_alg_id] == std::string("SPCBPT_eye"))
                        launchLVCTrace(TScene);

                    launchSubframe(output_buffer, TScene);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;                    
                    



                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);
                    render_fps = 1.0 / (display_time.count() + render_time.count() + state_update_time.count()); 

                    glfwSwapBuffers(window);

                    estimation::es.estimation_mode = false;
                    if (estimation::es.estimation_mode == true)
                    {
                        float error = estimation::es.relMse_estimate(MyThrustOp::copy_to_host(params.accum_buffer, params.width * params.height), params);
                        printf("frame %d relMse %f\n", params.subframe_index, error); 
                    }
                    else
                    {
                        printf("frame %d\n", params.subframe_index);
                    }
                    ++params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }
            sutil::cleanupUI( window );
        }

       // cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
