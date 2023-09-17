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
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include<sutil/Scene.h> 
#include"sceneLoader.h"
#include"scene_shift.h"
#include<sutil/Record.h>
#include <io.h>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"cuda_thrust/device_thrust.h"
#include"decisionTree/classTree_host.h"
#include"PG_host.h"
#include"frame_estimation.h"
#include <direct.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include"cjson/cJSON.cpp"
#define NUM_SUBSPACE params.subspace_info.num_subspace
#define NUM_SUBSPACE_LIGHTSOURCE params.subspace_info.num_subspace_lightsource
using namespace std;

static double render_time_record = 0;
static int render_frame_record = 0;

bool resize_dirty = false;
bool minimized = false;
// Camera state
bool             camera_changed = true;
MyParams* d_params = nullptr;
sutil::Camera    camera;
sutil::Trackball trackball;
MyParams   params = {};
LightTraceParams& lt_params = params.lt;
PreTraceParams& pr_params = params.pre_tracer;
subspaceMacroInfo& subspaceInfo = params.subspace_info;
DropOutTracing_params& dot_params = params.dot_params;
// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 1;

string initial_algrithm;
std::vector<sutil::renderMainProg> render_main_progs;
//std::vector< std::string> render_alg = { std::string("pt"), std::string("SPCBPT_eye"), std::string("SPCBPT_eye_ForcePure") };
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
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

//typedef Record<RayGenData>   RayGenRecord;
//typedef Record<MissData>     MissRecord;
//typedef Record<HitGroupData> HitGroupRecord;



//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    MyParams* params = static_cast<MyParams*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
        params->image_resize();
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
        params->image_resize();
    }
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    MyParams* params = static_cast<MyParams*>(glfwGetWindowUserPointer(window));
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
    params->image_resize();
}


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}


void img_save(double render_time = -1, int frame = 0)
{
    sutil::ImageBuffer outputbuffer;

    auto host_buffer = MyThrustOp::copy_to_host(params.frame_buffer, params.height * params.width);
    outputbuffer.data = host_buffer.data();
    outputbuffer.width = params.width;
    outputbuffer.height = params.height;
    outputbuffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // 检测当前目录下是否存在名为 "data" 的文件夹
    if (_access_s("data", 0) != 0) {
        _mkdir("data");
    }
    else
        std::cout << "already has data" << std::endl;

    // 将时间格式化为字符串
    std::stringstream ss;
    ss << "./data/" << std::put_time(std::localtime(&now_time_t), "%Y_%m_%d%H_%M_%S") << "_" << frame << "iterations_" << render_time << "time";

    // 获取格式化后的文件名
    std::string filename = ss.str();
    //printf("save %s\n",filename);

    sutil::saveImage((filename + ".png").c_str(), outputbuffer, true);


    auto p = MyThrustOp::copy_to_host(params.accum_buffer, params.height * params.width);
    std::ofstream outFile;
    outFile.open((filename + ".txt").c_str());
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

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
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
            img_save(render_time_record, render_frame_record);
        }
        else if (key == GLFW_KEY_SPACE)
        {
            render_alg_id++;
            if (render_alg_id >= render_main_progs.size())
            {
                render_alg_id = 0;
            }
            printf("raygen switching to %s\n", render_main_progs[render_alg_id].mainProgName.c_str());
            camera_changed = true;
            resize_dirty = true;
        }
        else if (key == GLFW_KEY_P)
        {
            one_frame_render_only = one_frame_render_only ^ true;
        }

    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }

    if (key == GLFW_KEY_W)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 dir = normalize(lookat - eye);
        float speed = 1;
        eye += dir / render_fps * speed;
        lookat += dir / render_fps * speed;
        camera.setEye(eye);
        camera.setLookat(lookat);
        camera_changed = true;
        resize_dirty = true;
        params.image_resize();
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
    {
        camera_changed = true;
        params.image_resize();
    }
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

std::vector<string> mainProgNames()
{
    std::vector<string> ans;
    for (int i = 0; i < render_main_progs.size(); i++) ans.push_back(render_main_progs[i].mainProgName);
    return ans;
}
void program_setup(sutil::Scene& scene)
{
    auto& prog_set_maps = scene.m_progsets;

    //program setting for spcbpt
    //also used as the template setting for the other algorithm
    sutil::ProgSet spcbpt_progset;
    spcbpt_progset.setRaygen("raygen.cu", "__raygen__SPCBPT");

    spcbpt_progset.setMiss(RayType::RAY_TYPE_EYESUBPATH, "raygen.cu", "__miss__env__BDPTVertex");
    spcbpt_progset.setMiss(RayType::RAY_TYPE_OCCLUSION, "raygen.cu", "__miss__none");//no miss for occulsion
    spcbpt_progset.setMiss(RayType::RAY_TYPE_EYESUBPATH_SIMPLE, "raygen.cu", "__miss__env__BDPTVertex");

    spcbpt_progset.setRayHit(RayType::RAY_TYPE_OCCLUSION, RayHitType::RAYHIT_TYPE_NORMAL, "hit_program.cu", "__closesthit__occlusion", "__anyhit__occlusion");
    spcbpt_progset.setRayHit(RayType::RAY_TYPE_OCCLUSION, RayHitType::RAYHIT_TYPE_LIGHTSOURCE, "hit_program.cu", "__closesthit__occlusion", "__anyhit__occlusion");
    spcbpt_progset.setRayHit(RayType::RAY_TYPE_EYESUBPATH, RayHitType::RAYHIT_TYPE_NORMAL, "hit_program.cu", "__closesthit__eyeSubpath", "__anyhit__none");
    spcbpt_progset.setRayHit(RayType::RAY_TYPE_EYESUBPATH, RayHitType::RAYHIT_TYPE_LIGHTSOURCE, "hit_program.cu", "__closesthit__eyeSubpath_LightSource", "__anyhit__none");
    spcbpt_progset.setRayHit(RayType::RAY_TYPE_EYESUBPATH_SIMPLE, RayHitType::RAYHIT_TYPE_NORMAL, "hit_program.cu", "__closesthit__eyeSubpath_simple", "__anyhit__none");
    spcbpt_progset.setRayHit(RayType::RAY_TYPE_EYESUBPATH_SIMPLE, RayHitType::RAYHIT_TYPE_LIGHTSOURCE, "hit_program.cu", "__closesthit__eyeSubpath_LightSource_simple", "__anyhit__none");
    prog_set_maps[std::string("SPCBPT_eye")] = spcbpt_progset;

    if (params.enable_proxy_sampling)
    {
        sutil::ProgSet proxy_progset = spcbpt_progset;
        proxy_progset.setRaygen("raygen.cu", "__raygen__shift_combine");
        prog_set_maps[std::string("proxy_SPCBPT_eye")] = proxy_progset;

    }

    sutil::ProgSet dot_progset = spcbpt_progset;
    dot_progset.setRaygen("raygen.cu", "__raygen__shift_combine");
    prog_set_maps[std::string("ProxyTracing_eye")] = dot_progset;

    //program setting for pt
    sutil::ProgSet pt_progset = spcbpt_progset;
    pt_progset.setRaygen("raygen.cu", "__raygen__pinhole");
    pt_progset.setMiss(RayType::RAY_TYPE_RADIANCE, "raygen.cu", "__miss__constant_radiance");
    pt_progset.setRayHit(RayType::RAY_TYPE_RADIANCE, RayHitType::RAYHIT_TYPE_LIGHTSOURCE, "hit_program.cu", "__closesthit__lightsource", "__anyhit__none");
    pt_progset.setRayHit(RayType::RAY_TYPE_RADIANCE, RayHitType::RAYHIT_TYPE_NORMAL, "hit_program.cu", "__closesthit__radiance", "__anyhit__none");
    prog_set_maps[std::string("pt")] = pt_progset;

    //program setting for light subpath tracing
    sutil::ProgSet lt_progset = spcbpt_progset;
    lt_progset.setRaygen("raygen.cu", "__raygen__lightTrace");
    lt_progset.setMiss(RayType::RAY_TYPE_LIGHTSUBPATH, "raygen.cu", "__miss__BDPTVertex");
    lt_progset.setRayHit(RayType::RAY_TYPE_LIGHTSUBPATH, RayHitType::RAYHIT_TYPE_NORMAL, "hit_program.cu", "__closesthit__lightSubpath", "__anyhit__none");
    lt_progset.setRayHit(RayType::RAY_TYPE_EYESUBPATH, RayHitType::RAYHIT_TYPE_LIGHTSOURCE, "hit_program.cu", "__closesthit__lightSource_subpath", "__anyhit__none");
    prog_set_maps[std::string("light trace")] = lt_progset;

    //program setting for pretracing
    sutil::ProgSet pretrace_progset = spcbpt_progset;
    pretrace_progset.setRaygen("raygen.cu", "__raygen__TrainData");
    prog_set_maps[std::string("pretrace")] = pretrace_progset;

    //compile the program
    scene.loadModules();
    for (auto p = prog_set_maps.begin(); p != prog_set_maps.end(); p++) {
        p->second.compile(scene);
    }

    for (auto p = prog_set_maps.begin(); p != prog_set_maps.end(); p++)
    {
        scene.createPipeline(p->second);
        scene.createSBT(p->second);
    }


    //注册渲染算法以供运行时切换
    sutil::renderMainProg render_prog;
    render_prog.eyeRaygenProgName = "pt";
    render_prog.mainProgName = "PT";
    render_prog.need_light_trace = false;
    render_main_progs.push_back(render_prog);


    render_prog.eyeRaygenProgName = "SPCBPT_eye";
    render_prog.mainProgName = "SPCBPT";
    render_prog.need_light_trace = true;
    render_main_progs.push_back(render_prog);

    if (params.enable_proxy_sampling)
    {
        sutil::renderMainProg render_prog;
        render_prog.eyeRaygenProgName = "proxy_SPCBPT_eye";
        render_prog.mainProgName = "proxy_SPCBPT";
        render_prog.need_light_trace = true;
        render_prog.is_dropout_tracing_proxy_sampling = true;
        render_main_progs.push_back(render_prog);
    }

    for (int i = 0; i < render_main_progs.size(); i++)
    {
        if (initial_algrithm == render_main_progs[i].mainProgName)
            render_alg_id = i;
    }
    printf("set %s as the initial rendering algorithm\n", render_main_progs[render_alg_id].mainProgName);
}
void initLaunchParams(const sutil::Scene& scene) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
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


    params.miss_color = make_float3(0.1f);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(MyParams)));

    params.handle = scene.traversableHandle();
    subspaceInfo.eye_tree = nullptr;
    subspaceInfo.light_tree = nullptr;
    subspaceInfo.Q = nullptr;
    subspaceInfo.CMFGamma = nullptr;

    params.estimate_pr.ready = false;
    params.estimate_pr.ref_buffer = nullptr;
    if (estimation::es.estimation_mode == true)
    {
        params.estimate_pr.ref_buffer = estimation::es.ref_ptr;
        params.estimate_pr.height = estimation::es.ref_height;
        params.estimate_pr.width = estimation::es.ref_width;
        params.estimate_pr.ready = true;
    }
    params.no_subspace = NO_SUBSPACE;
}



void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, MyParams& params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));
}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
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

    handleCameraUpdate(params);
    handleResize(output_buffer, params);

    params.skip_proxy_dropout_tracing = !render_main_progs[render_alg_id].is_dropout_tracing_proxy_sampling;


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
void env_params_setup(Scene& Src, const sutil::Scene& scene)
{
    if (scene.getEnvFilePath() == std::string(""))
    {
        params.sky.valid = false;
        return;
    }
    printf("load and build sampling cmf from file %s\n", scene.getEnvFilePath());
    HDRLoader hdr_env((string(SAMPLES_DIR) + string("/data/") + scene.getEnvFilePath()));

    params.sky.light_id = Src.env_light_id;
    Light& light = Src.optix_lights[params.sky.light_id];
    params.sky.height = hdr_env.height();
    params.sky.width = hdr_env.width();
    params.sky.divLevel = light.divLevel;
    params.sky.ssBase = light.ssBase;
    params.sky.size = hdr_env.height() * hdr_env.width();



    float4* hdr_m_raster = reinterpret_cast<float4*>(hdr_env.raster());
    for (int i = 0; i < Src.dirLights.size(); i++)
    {
        auto& dir_light = Src.dirLights[i];
        float3 dir = dir_light.first;
        //dir.y = -dir.y;
        auto uv = dir2uv(dir);
        auto coord = params.sky.uv2coord(uv);
        auto index = params.sky.coord2index(coord);
        hdr_m_raster[index] += make_float4(dir_light.second * params.sky.size / (4 * M_PI), 0.0);
        printf("Add directional light %f %f %f in index %d\n", dir_light.first.x, dir_light.first.y, dir_light.first.z, index);

    }
    auto env_tex = hdr_env.loadTexture(light.env.backgroundColor, nullptr);
    params.sky.tex = env_tex.texture;
    light.emissionID = params.sky.tex;
    params.sky.cmf = thrust::raw_pointer_cast(envMapCMFBuild(hdr_m_raster, hdr_env.height() * hdr_env.width(), params.sky));
    params.sky.center = scene.aabb().center();
    params.sky.r = length(scene.aabb().m_min - scene.aabb().m_max);
    params.sky.valid = true;
    params.sky.sampler_scale = make_float2(1, 1);
    params.sky.emission_scale = light.env.emission_scale;
}
void lt_params_setup(const sutil::Scene& scene)
{
    lt_params.M_per_core = 25;
    lt_params.core_padding = 200;
    lt_params.num_core = 4000;
    lt_params.M = lt_params.M_per_core * lt_params.num_core;
    lt_params.launch_frame = 0;

    BDPTVertex* LVC_ptr;
    bool* valid_ptr;
    float3* Light_image;
    int* pixel_id;
    float3* Light_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&LVC_ptr), sizeof(BDPTVertex) * lt_params.get_element_count()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&valid_ptr), sizeof(bool) * lt_params.get_element_count()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Light_image), sizeof(float3) * params.width * params.height));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pixel_id), sizeof(int) * lt_params.get_element_count()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Light_buffer), sizeof(float3) * lt_params.get_element_count()));

    lt_params.ans = LVC_ptr;// BufferView<BDPTVertex>(LVC_ptr, lt_params.get_element_count());
    lt_params.validState = valid_ptr;// BufferView<bool>(valid_ptr, lt_params.get_element_count());
    lt_params.lightImage = Light_image;
    lt_params.lightBuffer = Light_buffer;
    lt_params.lightIndex = pixel_id;
    //params.lt = lt_params; 
}
void estimation_setup(const string& path) {
    string algo = "";
    switch (render_alg_id)
    {
    case 0:algo += "pt"; break;
    case 1:algo += "dropout"; break;
    case 2:algo += "spcbpt"; break;
    default:algo += "error"; break;
    }
    string name = path.substr(path.rfind('/') + 1);
    name = name.substr(0, name.rfind('\.'));
    string outpath = name + "_" + algo + ".txt";
    cout << "save our estimate to " << outpath << endl;
    estimation::es.outputFile.open(outpath);
    estimation::es.outputFile << "{\n"
        << "name:" << name << endl
        << "height:" << params.height << endl
        << "width:" << params.width << endl
        << "algo:" << algo << endl
        << "}" << endl;


    estimation::es.estimation_update("./ref/" + name + ".txt", false);
}

void preTracer_params_setup(const sutil::Scene& scene)
{
    pr_params.num_core = 10000;
    pr_params.padding = 10;
    pr_params.iteration = 0;
    preTracePath* pretrace_path_ptr;
    preTraceConnection* pretrace_conn_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pretrace_path_ptr), sizeof(preTracePath) * pr_params.num_core));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pretrace_conn_ptr), sizeof(preTraceConnection) * pr_params.get_element_count()));
    pr_params.paths = pretrace_path_ptr;
    pr_params.conns = pretrace_conn_ptr;
    pr_params.PG_mode = false;
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

    //scene.switchRaygen(std::string("light trace"));
    auto prog_set = scene.m_progsets[std::string("light trace")];
    OPTIX_CHECK(optixLaunch(
        //scene.pipeline(),
        prog_set.m_pipeline,
        0,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        //scene.sbt(),
        &prog_set.m_sbt,
        lt_params.num_core,
        1,
        1
    ));
    CUDA_SYNC_CHECK();
}
void setLightImage()
{
    float3* light_buffer = new float3[params.lt.get_element_count()];
    int* index = new int[params.lt.get_element_count()];
    float3* light_image = new float3[params.width * params.height];
    bool* valid = new bool[params.lt.get_element_count()];

    cudaMemcpy(light_buffer, params.lt.lightBuffer, params.lt.get_element_count() * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(index, params.lt.lightIndex, params.lt.get_element_count() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(valid, params.lt.validState, params.lt.get_element_count() * sizeof(bool), cudaMemcpyDeviceToHost);

    memset(light_image, 0, params.width * params.height * sizeof(float3));
 
    for (int i = 0; i < params.lt.get_element_count(); i++)
    {
        if (!valid[i])
            continue;
        int id = index[i];
        if (id > 0)
        {
            light_image[id] += light_buffer[i];
        }            
    }
    for (int i = 0; i < params.width * params.height; i++)
    {
        light_image[i] /= params.lt.M;
    }
    cudaMemcpy(params.lt.lightImage, light_image, params.width * params.height * sizeof(float3), cudaMemcpyHostToDevice);

    delete[] valid;
    delete[] light_buffer;
    delete[] index;
    delete[] light_image;

}
void launchLVCTrace(sutil::Scene& scene)
{
    if (!params.skip_proxy_dropout_tracing) { dot_params.discard_ratio = dot_params.discard_ratio_next; }
    launchLightTrace(scene);
    auto p_v = thrust::device_pointer_cast(params.lt.ans);
    auto p_valid = thrust::device_pointer_cast(params.lt.validState);
    auto sampler = MyThrustOp::LVC_Process(p_v, p_valid, params.lt.get_element_count());
    setLightImage();

    params.sampler = sampler;
    if (!params.skip_proxy_dropout_tracing)
    {
        sampler = MyThrustOp::LVC_Process_glossyOnly(p_v, p_valid, params.lt.get_element_count(), params.materials);
        params.sampler.glossy_count = sampler.glossy_count;
        params.sampler.glossy_index = sampler.glossy_index;
        params.sampler.glossy_subspace_bias = sampler.glossy_subspace_bias;
        params.sampler.glossy_subspace_num = sampler.glossy_subspace_num;
    }

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

    //scene.switchRaygen(std::string("pretrace"));
    auto prog_set = scene.m_progsets[std::string("pretrace")];
    OPTIX_CHECK(optixLaunch(
        prog_set.m_pipeline,
        0,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        //scene.sbt(),
        &prog_set.m_sbt,
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
void path_guiding_params_setup(sutil::Scene& scene)
{
    int pg_training_data_batch = 10;
    int pg_training_data_online_batch = 0;
    const int batch_sample_count = 1000000;

    if (!PG_ENABLE) {
        params.pg_params.pg_enable = 0;
        return;
    }
    //pr_params.PG_mode = true;


    std::vector<path_guiding::PG_training_mat> g_mats;
    int build_iteration_max = 12;
    //g_mats = MyThrustOp::get_data_for_path_guiding();

    //是否使用path guiding自己的训练结果来引导path guiding的训练
    if (PG_SELF_TRAIN)
    {
        build_iteration_max = 12;
        int initial_path = 1000;
        int split_limit = initial_path;
        int target_path = initial_path;
        if (PG_MORE_TRAINING)
        {
            build_iteration_max += 4;
            split_limit *= 2;
        }
        PGTrainer_api.init(scene.aabb());
        for (int i = 0; i < build_iteration_max; i++)
        {
            MyThrustOp::clear_training_set();
            int current_sample_count = 0;
            int accm_sample_count = 0;
            int accm_it = 0;
            while (current_sample_count + accm_sample_count < target_path)
            {
                current_sample_count += launchPretrace(scene);
                accm_it++;
                //printf("regenerate data for pg %d %d\n", current_sample_count + accm_sample_count, g_mats.size());

                if (current_sample_count > batch_sample_count)
                {
                    accm_sample_count += current_sample_count;
                    current_sample_count = 0;
                    auto n_mats = MyThrustOp::get_data_for_path_guiding(-1, pr_params.PG_mode);
                    g_mats.insert(g_mats.end(), n_mats.begin(), n_mats.end());
                    MyThrustOp::clear_training_set();
                }
            }
            auto n_mats = MyThrustOp::get_data_for_path_guiding(-1, pr_params.PG_mode);
            g_mats.insert(g_mats.end(), n_mats.begin(), n_mats.end());
            MyThrustOp::clear_training_set();


            printf("get %d samples for pg building iteration %d; target path%d; split-limit %d; average nodes %f \n",
                g_mats.size(), i, target_path, split_limit, float(g_mats.size()) / accm_it);
            PGTrainer_api.set_training_set(g_mats);
            PGTrainer_api.build_tree(split_limit, g_mats.size());
            params.pg_params.spatio_trees = MyThrustOp::spatio_tree_to_device(PGTrainer_api.s_tree.nodes.data(), PGTrainer_api.s_tree.nodes.size());
            params.pg_params.quad_trees = MyThrustOp::quad_tree_to_device(PGTrainer_api.q_tree_group.nodes.data(), PGTrainer_api.q_tree_group.nodes.size());
            params.pg_params.pg_enable = 1;
            params.pg_params.epsilon_lum = 0.001;
            params.pg_params.guide_ratio = 0.5;

            target_path *= 2;
            split_limit *= sqrt(2);
            g_mats.clear();
        }
    }
    else
    {
        for (int i = 0; i < pg_training_data_batch; i++)
        {
            MyThrustOp::clear_training_set();
            int current_sample_count = 0;
            while (current_sample_count < batch_sample_count)
            {
                current_sample_count += launchPretrace(scene);
                printf("regenerate data for pg %d %d\n", current_sample_count, g_mats.size());
            }
            auto n_mats = MyThrustOp::get_data_for_path_guiding(-1, pr_params.PG_mode);
            g_mats.insert(g_mats.end(), n_mats.begin(), n_mats.end());
            MyThrustOp::clear_training_set();
        }
        printf("get mats size %d\n", g_mats.size());
        PGTrainer_api.set_training_set(g_mats);
        PGTrainer_api.init(scene.aabb());
        //build the tree until we reach max iteration (and the function return false) 
        for (int i = 0; i < build_iteration_max; i++) { PGTrainer_api.build_tree(); }

        for (int i = 0; i < pg_training_data_online_batch; i++)
        {
            g_mats.clear();
            MyThrustOp::clear_training_set();
            int current_sample_count = 0;
            while (current_sample_count < batch_sample_count)
            {
                current_sample_count += launchPretrace(scene);
            }
            printf("online training for pg batch %d \n", i);
            auto n_mats = MyThrustOp::get_data_for_path_guiding();
            PGTrainer_api.set_training_set(n_mats);
            PGTrainer_api.online_training();
        }
    }
    PGTrainer_api.mats_cache.clear();
    PGTrainer_api.mats.clear();
    g_mats.clear();
    params.pg_params.spatio_trees = MyThrustOp::spatio_tree_to_device(PGTrainer_api.s_tree.nodes.data(), PGTrainer_api.s_tree.nodes.size());
    params.pg_params.quad_trees = MyThrustOp::quad_tree_to_device(PGTrainer_api.q_tree_group.nodes.data(), PGTrainer_api.q_tree_group.nodes.size());
    params.pg_params.pg_enable = 1;
    params.pg_params.epsilon_lum = 0.001;
    params.pg_params.guide_ratio = 0.5;
    pr_params.PG_mode = false;
    //printf("pg tree check %f %f %f\n", PGTrainer_api.s_tree.getNode(0).m_mid.x, PGTrainer_api.s_tree.getNode(0).m_mid.y, PGTrainer_api.s_tree.getNode(0).m_mid.z);
}
void dropOutTracingParamsInit()
{
    dot_params.is_init = false;
    dot_params.specularSubSpace = nullptr;
    dot_params.surfaceSubSpace = nullptr;
    dot_params.record_buffer = nullptr;
}
void dropOutTracingParamsSetup(sutil::Scene& scene)
{
    if (params.skip_proxy_dropout_tracing)return;

    dot_params.pixel_dirty = true;
    dot_params.discard_ratio = dropOut_tracing::light_subpath_caustic_discard_ratio;
    dot_params.discard_ratio_next = dropOut_tracing::light_subpath_caustic_discard_ratio;
    dot_params.specularSubSpaceNumber = dropOut_tracing::default_specularSubSpaceNumber;
    dot_params.surfaceSubSpaceNumber = dropOut_tracing::default_surfaceSubSpaceNumber;
    dot_params.data.on_GPU = false;
    ///////////////////////////////////////
    ///////Build Small Subspace////////////
    ///////////////////////////////////////
    MyThrustOp::clear_training_set();
    const int target_sample_count = 100000;
    int current_sample_count = 0;
    while (current_sample_count < target_sample_count)
    {
        //printf("% d\n", current_sample_count);
        current_sample_count += launchPretrace(scene);
    }

    auto unlabeled_samples = MyThrustOp::getCausticCentroidCandidate(false, 100000);
    auto specular_subspace = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, dot_params.specularSubSpaceNumber - 1, 1);
    dot_params.specularSubSpace = MyThrustOp::DOT_specular_tree_to_device(specular_subspace.v, specular_subspace.size);

    unlabeled_samples = MyThrustOp::get_weighted_point_for_tree_building(false, 10000);
    // surface Id 0 is remain for EMPTY SURFACEID
    auto normalsurfaceSubspace = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, dot_params.surfaceSubSpaceNumber - 1, 1);
    dot_params.surfaceSubSpace = MyThrustOp::DOT_surface_tree_to_device(normalsurfaceSubspace.v, normalsurfaceSubspace.size);

    /////////////////////////////////////////////////////////
    ////////////Subspace Build Finish////////////////////////
    /////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////
    ////////////Assign the Memory For Statistics/////////////
    ////////////////为统计数据的存储分配显存空间////////////////
    /////////////////////////////////////////////////////////
    dot_params.data.size =
        dot_params.specularSubSpaceNumber * dot_params.surfaceSubSpaceNumber * dropOut_tracing::slot_number * int(dropOut_tracing::DropOutType::DropOutTypeNumber);
    thrust::host_vector<dropOut_tracing::statistics_data_struct> DOT_statics_data(dot_params.data.size);
    thrust::fill(DOT_statics_data.begin(), DOT_statics_data.end(), dropOut_tracing::statistics_data_struct());
    dot_params.data.host_data = DOT_statics_data.data();
    dot_params.data.device_data = MyThrustOp::DOT_statistics_data_to_device(dot_params.data.host_data, dot_params.data.size);

    thrust::host_vector<dropOut_tracing::PGParams> DOT_PG_data(dot_params.specularSubSpaceNumber * dot_params.surfaceSubSpaceNumber * dropOut_tracing::max_u);
    dot_params.data.device_PGParams = MyThrustOp::DOT_PG_data_to_device(DOT_PG_data);
    dot_params.data.on_GPU = true;



    //thrust::host_vector<dropOut_tracing::statistic_record> h_record(lt_params.get_element_count());
    dot_params.record_buffer_core = lt_params.num_core;
    dot_params.record_buffer_padding = lt_params.core_padding * dropOut_tracing::record_buffer_width;
    dot_params.record_buffer = MyThrustOp::DOT_get_statistic_record_buffer(dot_params.record_buffer_core * dot_params.record_buffer_padding);
    dot_params.statistics_iteration_count = 0;


    if (dot_params.pixel_dirty)
    {
        thrust::host_vector<float> h_frac(params.width * params.height, 0.5);
        dot_params.pixel_caustic_refract = MyThrustOp::DOT_causticFrac_to_device(h_frac);
        dot_params.pixel_record = MyThrustOp::DOT_set_pixelRecords_size(params.width * params.height);

        //thrust::host_vector<dropOut_tracing::pixelRecord> h_record(params.width * params.height);
        dot_params.pixel_dirty = false;
    }
    //initial finished
    dot_params.is_init = true;
    dot_params.selection_const = 0.0;
}
//update the probability for caustic subspace sampling and caustic frac
void updateDropOutTracingCombineWeight()
{
    if (params.skip_proxy_dropout_tracing) return;
    static int train_iter = 0;
    if (train_iter > 0 && train_iter > dropOut_tracing::iteration_stop_learning)return;
    train_iter++;
    static thrust::host_vector<float> h_frac(params.width * params.height, 0.5);
    static thrust::host_vector<float> h_caustic_gamma(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE, 1.0 / dropOut_tracing::default_specularSubSpaceNumber);
    static vector<float> normal_weight(params.width * params.height, 0);
    static vector<int> normal_count(params.width * params.height, 0);
    static vector<float> caustic_weight(params.width * params.height, 0);
    static vector<int> caustic_count(params.width * params.height, 0);
    static vector<float> gamma_non_normalized(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE, 0.000001);
    static vector<float> gamma_non_normalized_single(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE, 0.000001);
    static vector<int> gamma_count(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE, 0);
    if (dot_params.pixel_dirty)
    {
        h_frac.resize(params.width * params.height);
        thrust::fill(h_frac.begin(), h_frac.end(), 0.5);
        normal_weight.resize(params.width * params.height);
        thrust::fill(normal_weight.begin(), normal_weight.end(), 0);
        normal_count.resize(params.width * params.height);
        thrust::fill(normal_count.begin(), normal_count.end(), 0);
        caustic_weight.resize(params.width * params.height);
        thrust::fill(caustic_weight.begin(), caustic_weight.end(), 0);
        caustic_count.resize(params.width * params.height);
        thrust::fill(caustic_count.begin(), caustic_count.end(), 0);

        dot_params.pixel_caustic_refract = MyThrustOp::DOT_causticFrac_to_device(h_frac);
        dot_params.pixel_record = MyThrustOp::DOT_set_pixelRecords_size(params.width * params.height);
        dot_params.pixel_dirty = false;
    }
    else
    {
        auto h_record = MyThrustOp::DOT_get_pixelRecords();
        for (int i = 0; i < h_record.size(); i++)
        {
            if (h_record[i].valid() == false || h_record[i].is_caustic() == false)continue;
            if (isnan(h_record[i].record) || isinf(h_record[i].record))continue;
            float weight = abs(h_record[i].record) * h_caustic_gamma[h_record[i].eyeId * dropOut_tracing::default_specularSubSpaceNumber + h_record[i].specularId];
            if (weight > 1000000) weight = 1000000;
            unsigned id = h_record[i].eyeId * dropOut_tracing::default_specularSubSpaceNumber + h_record[i].specularId;
            gamma_count[id] += 1;
            gamma_non_normalized[id] += weight * weight;
            gamma_non_normalized_single[id] = lerp(gamma_non_normalized_single[id], weight * weight, 1.0 / gamma_count[id]);
            //gamma_non_normalized_single[id] = gamma_non_normalized[id];

        }
        vector<float> gamma_sum(NUM_SUBSPACE, 0);
        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; j++)
            {
                unsigned id = j + i * dropOut_tracing::default_specularSubSpaceNumber;
                //gamma_sum[i] +=sqrt( gamma_non_normalized[j + i * dropOut_tracing::default_specularSubSpaceNumber]);
                //gamma_sum[i] += gamma_non_normalized[j + i * dropOut_tracing::default_specularSubSpaceNumber];
                gamma_sum[i] += sqrt(gamma_non_normalized_single[id]);
            }
        }

        for (int i = 0; i < NUM_SUBSPACE; i++)
        {
            for (int j = 0; j < dropOut_tracing::default_specularSubSpaceNumber; j++)
            {
                unsigned id = j + i * dropOut_tracing::default_specularSubSpaceNumber;
                h_caustic_gamma[id] =
                    //sqrt(gamma_non_normalized[i * dropOut_tracing::default_specularSubSpaceNumber + j]) / gamma_sum[i] * (1-CONSERVATIVE_RATE) +
                    sqrt(gamma_non_normalized_single[id]) / gamma_sum[i] * (1 - CONSERVATIVE_RATE) +
                    1.0 / dropOut_tracing::default_specularSubSpaceNumber * (CONSERVATIVE_RATE);
                //printf("eye %d-%d pmf %f\n",i , j, h_caustic_gamma[i * dropOut_tracing::default_specularSubSpaceNumber + j]);
            }
        }

        for (int i = 0; i < h_record.size(); i++)
        {
            if (!h_record[i].valid())continue;
            if (h_record[i].record < 0)continue;
            //if (abs(h_record[i].record) == 0)continue;
            uint2 pixel_label = dot_params.Id2pixel(i, make_uint2(params.width, params.height));
            int final_label = dot_params.pixel2unitId(pixel_label, make_uint2(params.width, params.height));

            if (!h_record[i].is_caustic())
            {
                normal_count[final_label]++;
                normal_weight[final_label] += h_record[i].record;
            }
            else
            {
                caustic_count[final_label]++;
                caustic_weight[final_label] += h_record[i].record;
            }
        }
        for (int i = 0; i < h_frac.size(); i++)
        {
            int valid_size = 10;
            float t = lerp(1, CONSERVATIVE_RATE, min(1.0f, float(normal_count[i] + caustic_count[i]) / valid_size));
            if (caustic_count[i] + normal_count[i] == 0)
                h_frac[i] = 0.5;
            else
            {
                float recommend = (caustic_weight[i] / (normal_weight[i] + caustic_weight[i]));
                if (recommend < 0.05)
                    h_frac[i] = CONSERVATIVE_RATE;
                else h_frac[i] = 1;

            }
        }
        subspaceInfo.CMFCausticGamma = MyThrustOp::DOT_causticCMFGamma_to_device(h_caustic_gamma);
        dot_params.CMF_Gamma = subspaceInfo.CMFCausticGamma;
        dot_params.pixel_caustic_refract = MyThrustOp::DOT_causticFrac_to_device(h_frac);
    }
}

std::vector<std::vector<std::vector<std::vector<float2>>>> TrainVector(dropOut_tracing::max_u,
    std::vector<std::vector<std::vector<float2>>>(dropOut_tracing::default_specularSubSpaceNumber,
        std::vector<std::vector<float2>>(dropOut_tracing::default_surfaceSubSpaceNumber,
            std::vector<float2>())));

std::vector<std::vector<std::vector<bool>>> TrainFinish(dropOut_tracing::max_u,
    std::vector<std::vector<bool>>(dropOut_tracing::default_specularSubSpaceNumber,
        std::vector<bool>(dropOut_tracing::default_surfaceSubSpaceNumber,
            0)));

const int capacity = 30000;

void updateDropOutTracingParams()
{
    if (params.skip_proxy_dropout_tracing) return;
    static int train_iter = 0;
    //迭代数目超过指定数目后终止统计
    if (train_iter > 0 && train_iter > dropOut_tracing::iteration_stop_learning)
    {
        if (train_iter == dropOut_tracing::iteration_stop_learning + 1)
        {
            train_iter++;
            printf("iteration more than stop point, stop params learning\n");
        }
        return;
    }
    train_iter++;
    bool disable_print = !DOT_DEBUG_INFO_ENABLE;
    thrust::host_vector<dropOut_tracing::statistics_data_struct>statics_data = MyThrustOp::DOT_statistics_data_to_host();
    thrust::host_vector<dropOut_tracing::PGParams> pg_data = MyThrustOp::DOT_PG_data_to_host();
    dot_params.data.host_data = statics_data.data();
    dot_params.data.host_PGParams = pg_data.data();
    dot_params.data.on_GPU = false;
    auto records = MyThrustOp::DOT_get_host_statistic_record_buffer();
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////Update the Statstics Data In The Fllowing Code Block///////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////// 

    //Average Computation
    //基于子空间统计倒数评估的平均值和方差
    {
        // Create temporary vectors to store the counter and data for each subspace 
        static std::vector<std::vector<std::vector<int>>> tempVector_counter(dropOut_tracing::max_u,
            std::vector<std::vector<int>>(dot_params.specularSubSpaceNumber,
                std::vector<int>(dot_params.surfaceSubSpaceNumber, 0)));
        static std::vector<std::vector<std::vector<float>>> tempVector(dropOut_tracing::max_u,
            std::vector<std::vector<float>>(dot_params.specularSubSpaceNumber,
                std::vector<float>(dot_params.surfaceSubSpaceNumber, 0)));

        static std::vector<std::vector<std::vector<float>>> variance_vector(dropOut_tracing::max_u,
            std::vector<std::vector<float>>(dot_params.specularSubSpaceNumber,
                std::vector<float>(dot_params.surfaceSubSpaceNumber, 0)));
        // Iterate through each record compute the summary reciprocal
        for (int i = 0; i < records.size(); i++)
        {
            auto& record = records[i];
            if (record.data_slot == DOT_usage::Average)
            {
                float& average = tempVector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId];
                float& variance = variance_vector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId];
                int& count = tempVector_counter[int(record.type)][record.specular_subspaceId][record.surface_subspaceId];
                average = lerp(average, float(record), 1.0 / (count + 1));
                variance = lerp(variance, float(record) * float(record), 1.0 / (count + 1));
                count++;
            }
        }
        for (int i = 0; i < dropOut_tracing::max_u; i++)
            for (int j = 0; j < dot_params.specularSubSpaceNumber; j++)
                for (int k = 0; k < dot_params.surfaceSubSpaceNumber; k++)
                {
                    auto& statistic_data = dot_params.get_statistic_data(dropOut_tracing::DropOutType(i), j, k);

                    if (statistic_data.valid)
                    {
                        // Linearly interpolate the data in dot_params with the new data from tempVector
                        statistic_data.average = tempVector[i][j][k];// lerp(statistic_data.average, tempVector[i][j][k], 1.0 / float(dot_params.statistics_iteration_count + 1));
                        statistic_data.variance = variance_vector[i][j][k];
                        if (statistic_data.valid && !disable_print)
                            printf("Average reciprocal PDF for ID S:%d C:%d U:%d is %f , sqrt variance %f\n", j, k, i, statistic_data.average, sqrt(statistic_data.variance));
                    }
                    if (tempVector_counter[i][j][k] != 0)
                    {
                        statistic_data.valid = true;
                    }
                }
    }

    //Bound Process
    //基于子空间，统计评估的上界
    {
        std::vector<std::vector<std::vector<float>>> tempVector(dropOut_tracing::max_u,
            std::vector<std::vector<float>>(dot_params.specularSubSpaceNumber,
                std::vector<float>(dot_params.surfaceSubSpaceNumber, 0)));


        // Iterate through each record compute the summary reciprocal
        for (int i = 0; i < records.size(); i++)
        {
            auto& record = records[i];
            if (record.data_slot == DOT_usage::Bound)
            {
                if (isinf(record))continue;
                tempVector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId] =
                    max(float(record), tempVector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId]);

            }
        }
        for (int i = 0; i < dropOut_tracing::max_u; i++)
            for (int j = 0; j < dot_params.specularSubSpaceNumber; j++)
                for (int k = 0; k < dot_params.surfaceSubSpaceNumber; k++)
                {
                    auto& statistic_data = dot_params.get_statistic_data(dropOut_tracing::DropOutType(i), j, k);
                    statistic_data.bound = max(statistic_data.bound, tempVector[i][j][k]);
                    if (statistic_data.valid && !disable_print)
                        printf("Bound Setting for ID S:%d C:%d U:%d is %f\n", j, k, i, statistic_data.bound);
                }
    }

    //PG training
    //并未经过测试的，使用path guiding来加速倒数评估流程的代码段
    if (dropOut_tracing::PG_reciprocal_estimation_enable)
    {
        int count = 0;

        for (int i = 0; i < records.size(); i++)
        {
            auto& record = records[i];
            if (record.data_slot == DOT_usage::Dirction)
            {
                //printf("PG record for ID S:%d C:%d U:%d with uv %f %f\n", record.specular_subspaceId, record.surface_subspaceId, int(record.type), record.data, record.data2);
                count++;
                if (TrainVector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId].size() == capacity) continue;
                TrainVector[int(record.type)][record.specular_subspaceId][record.surface_subspaceId].push_back(float2{ record.data,record.data2 });
            }
        }

        for (int i = 0; i < dropOut_tracing::max_u; i++)
            for (int j = 0; j < dot_params.specularSubSpaceNumber; j++)
                for (int k = 0; k < dot_params.surfaceSubSpaceNumber; k++)
                {
                    int num = TrainVector[i][j][k].size();
                    if (num == 0) continue;
                    if (num == capacity && TrainFinish[i][j][k]) continue;
                    if (num == capacity) {
                        TrainFinish[i][j][k] = true; printf("S:%d C:%d U:%d train end!!!\n", j, k, i);
                        dot_params.get_PGParams_pointer(dropOut_tracing::DropOutType(i), j, k)->trainEnd = 1;
                    }
                    dot_params.get_PGParams_pointer(dropOut_tracing::DropOutType(i), j, k)->loadIn(TrainVector[i][j][k]);
                    if (!disable_print) {
                        printf("PG traning for ID S:%d C:%d U:%d with size %d\n", j, k, i, num);
                    }
                }
        printf("we get %d record success\n", count);
    }

    dot_params.statistics_iteration_count++;
    printf("received %lld valid records in the Light Tracing\n", records.size());

    // Bad Block Ratio
    //如果统计数据出现异常，则被视为坏块，这些坏块的比率在本段代码中被计算
    float sum = dropOut_tracing::max_u * dot_params.specularSubSpaceNumber * dot_params.surfaceSubSpaceNumber;
    int bad_cnt = 0;
    for (int i = 0; i < dropOut_tracing::max_u; i++)
        for (int j = 0; j < dot_params.specularSubSpaceNumber; j++)
            for (int k = 0; k < dot_params.surfaceSubSpaceNumber; k++)
            {
                auto& data = dot_params.get_statistic_data(dropOut_tracing::DropOutType(i), j, k);
                if (isnan(data.average) || isinf(data.average) || isnan(data.bound) || isinf(data.bound))
                {
                    ++bad_cnt;
                }
            }

    printf("Bad Block: %.2f%%\n", 100.0f * bad_cnt / sum);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////Update the Statstics Data In The Above Data Block//////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////



    dot_params.data.device_data = MyThrustOp::DOT_statistics_data_to_device(statics_data);
    dot_params.data.device_PGParams = MyThrustOp::DOT_PG_data_to_device(pg_data);
    dot_params.data.on_GPU = true;


    //selection const被用于MIS计算中
    dot_params.selection_const = dropOut_tracing::connection_uniform_sample ? lt_params.M_per_core * lt_params.num_core / float(params.sampler.glossy_count)
        : lt_params.M_per_core * lt_params.num_core;
    // dot_params.selection_const *= DOT_LESS_MIS_WEIGHT ? (1 - dot_params.discard_ratio) : 1;
    dot_params.selection_const *= (1 - dot_params.discard_ratio);
    dot_params.specular_Q = MyThrustOp::DOT_get_Q();

    printf("get %d specular subpath, discard ratio %f\n", params.sampler.glossy_count, dot_params.discard_ratio);
    static float average_incomplete_ratio = 0;
    float incomplete_ratio = float(params.sampler.glossy_count + 1) / (1 - dot_params.discard_ratio) / (lt_params.M_per_core * lt_params.num_core);
    float train_t = 1.0 / train_iter;
    average_incomplete_ratio = average_incomplete_ratio * (1 - train_t) + incomplete_ratio * train_t;
    dot_params.discard_ratio_next = clamp(1 - dropOut_tracing::target_num_incomplete_subpath / (average_incomplete_ratio * lt_params.M_per_core * lt_params.num_core), 0.0, 0.99);
}

void preprocessing(sutil::Scene& scene)
{
    printf("begin pretracing\n");
    MyThrustOp::clear_training_set();
    const int target_sample_count = 1000000;
    int current_sample_count = 0;
    //用PT追踪训练数据直到一百万条完整路径被追踪出来
    while (current_sample_count < target_sample_count)
    {
        current_sample_count += launchPretrace(scene);
    }
    printf("pretracing complete\n");
    //MyThrustOp::sample_reweight();
    //从完整路径中取出能用于子空间分类的样本
    auto unlabeled_samples = MyThrustOp::get_weighted_point_for_tree_building(true, 10000);
    //根据这些样本建立子空间决策树
    auto h_eye_tree = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, NUM_SUBSPACE, 0);

    unlabeled_samples = MyThrustOp::get_weighted_point_for_tree_building(false, 10000);
    auto h_light_tree = classTree::buildTreeBaseOnExistSample()(unlabeled_samples, NUM_SUBSPACE - NUM_SUBSPACE_LIGHTSOURCE, 0);


    //数据分配到GPU中
    auto d_DecisionTree = MyThrustOp::eye_tree_to_device(h_eye_tree.v, h_eye_tree.size);
    subspaceInfo.eye_tree = d_DecisionTree;
    d_DecisionTree = MyThrustOp::light_tree_to_device(h_light_tree.v, h_light_tree.size);
    subspaceInfo.light_tree = d_DecisionTree;


    const int target_Q_samples = 2000000;
    int current_Q_samples = 0;
    thrust::device_ptr<float> Q_star = nullptr;
    while (current_Q_samples < target_Q_samples)
    {
        //追踪光子路以确定每个光子空间的能量值Q的大小
        //launchLightTrace(scene);
        launchLVCTrace(scene);
        auto p_v = thrust::device_pointer_cast(params.lt.ans);
        auto p_valid = thrust::device_pointer_cast(params.lt.validState);
        current_Q_samples += MyThrustOp::preprocess_getQ(p_v, p_valid, params.lt.get_element_count(), Q_star);

        //在预处理阶段可以顺带把统计数据算一下
        updateDropOutTracingParams();//update the statistic data for drop out sampling

    }
    //如果有能量为0的光子空间，将其能量设为无穷大以避免评估的时候出现问题
    MyThrustOp::Q_zero_handle(Q_star);
    //根据光子空间树和视子空间树来给完整路径打上标识
    MyThrustOp::node_label(subspaceInfo.eye_tree, subspaceInfo.light_tree);

    thrust::device_ptr<float> Gamma;
    //MyThrustOp::load_Q_file(Q_star);
    //将最优化所需的各种数据统一装配以供训练
    MyThrustOp::build_optimal_E_train_data(target_sample_count);
    //计算基于贡献值的初始化的Gamma矩阵
    MyThrustOp::preprocess_getGamma(Gamma);
    //训练全局方差最优化的Gamma矩阵
    MyThrustOp::train_optimal_E(Gamma);

    //MyThrustOp::load_Gamma_file(Gamma); 
    //MyThrustOp::train_optimal_E(Gamma);

    subspaceInfo.Q = thrust::raw_pointer_cast(Q_star);
    subspaceInfo.CMFGamma = thrust::raw_pointer_cast(MyThrustOp::Gamma2CMFGamma(Gamma));

    //thrust::device_ptr<float> CausticGamma;
    //MyThrustOp::preprocess_getGamma(CausticGamma, true);

    if (!params.skip_proxy_dropout_tracing)
    {
        //如果启用了proxy sampling，则初始化镜面子空间的Gamma矩阵
        thrust::host_vector<float> h_caustic_gamma(dropOut_tracing::default_specularSubSpaceNumber * NUM_SUBSPACE);
        thrust::fill(h_caustic_gamma.begin(), h_caustic_gamma.end(), 1.0 / dropOut_tracing::default_specularSubSpaceNumber);
        subspaceInfo.CMFCausticGamma = MyThrustOp::DOT_causticCMFGamma_to_device(h_caustic_gamma);
        dot_params.CMF_Gamma = subspaceInfo.CMFCausticGamma;

        thrust::device_ptr<float> CausticRatio;
        MyThrustOp::get_caustic_frac(CausticRatio);
        subspaceInfo.caustic_ratio = thrust::raw_pointer_cast(CausticRatio);
    }
}
void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::Scene& scene)
{
    //printf("subframe id %d\n", params.subframe_index);
    //scene.switchRaygen(render_alg[render_alg_id]);
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(MyParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    auto prog_set = scene.m_progsets[render_main_progs[render_alg_id].eyeRaygenProgName];
    OPTIX_CHECK(optixLaunch(
        //scene.pipeline(),
        prog_set.m_pipeline,
        0,             // stream
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(MyParams),
        //scene.sbt(),
        &prog_set.m_sbt,
        params.width,  // launch width
        params.height, // launch height
        1       // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}



void initCameraState(const sutil::Scene& scene)
{
    camera = scene.camera();
    camera_changed = true;
    params.image_resize();
    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}

string scenePath = " ";
void readConfig()
{
    string config_path = string(SAMPLES_DIR) + string("/config.json");
    FILE* config_fp = fopen(config_path.c_str(), "r");
    char buf[1024] = { 0 };
    fread(buf, 1, sizeof(buf), config_fp);
    cJSON* root = cJSON_Parse(buf);
    auto out = cJSON_Print(root);

    params.width = cJSON_GetObjectItem(root, "width")->valueint;
    params.height = cJSON_GetObjectItem(root, "height")->valueint;
    params.conservative_rate = cJSON_GetObjectItem(root, "conservative_rate")->valuedouble;
    params.n_connections = cJSON_GetObjectItem(root, "n_connections")->valueint;
    scenePath = string(SAMPLES_DIR) + string(cJSON_GetObjectItem(root, "scene_path")->valuestring);
    params.path_length_limited = static_cast<bool> (cJSON_GetObjectItem(root, "path_length_limited")->valueint);
    params.min_RR_rate = cJSON_GetObjectItem(root, "min_RR_rate")->valuedouble;
    params.enable_proxy_sampling = static_cast<bool> (cJSON_GetObjectItem(root, "enable_proxy_sampling")->valueint);

    params.subspace_info.num_subspace = cJSON_GetObjectItem(root, "num_subspace")->valueint;
    params.subspace_info.num_subspace_lightsource = cJSON_GetObjectItem(root, "num_subspace_lightsource")->valueint;

    params.skip_proxy_dropout_tracing = !params.enable_proxy_sampling;

    initial_algrithm = cJSON_GetObjectItem(root, "initial_algorithm")->valuestring;
    MyThrustOp::params_to_thrust(params);
    //std::cout << out;
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------ 
int main(int argc, char* argv[])
{
    //读取config.json获知渲染配置
    readConfig();
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
    _putenv("OPTIX_FORCE_DEPRECATED_LAUNCHER=1");



    try
    {
        //scenePath = string(SAMPLES_DIR) + string("/data/door/door.scene");   
        //scenePath = string(SAMPLES_DIR) + string("/data/hallway/hallway_teaser_env2.scene");   
        //scenePath = string(SAMPLES_DIR) + string("/data/glassroom/glassroom_project.scene");   
        //scenePath = string(SAMPLES_DIR) + string("/data/house/house_uvrefine2.scene");   
        scenePath = string(SAMPLES_DIR) + string("/data/cornell_box/cornell_env.scene");
        //scenePath = string(SAMPLES_DIR) + string("/data/bedroom_b/scene_v4.scene");  
        //读取场景文件
        auto myScene = LoadScene(scenePath.c_str());

        //将场景配置文件中的信息传到optix后端
        sutil::Scene OptiXScene;
        Scene_shift(*myScene, OptiXScene);//场景几何处理
        LightSource_shift(*myScene, params, OptiXScene,params.subspace_info.num_subspace_lightsource);//光源信息处理
        OptiXScene.finalize();//创建optix context和加速结构
        env_params_setup(*myScene, OptiXScene);//环境贴图设置
        program_setup(OptiXScene);//加载device端的cuda程序，包括装配SBT等


        OPTIX_CHECK(optixInit()); // Need to initialize function table
        initCameraState(OptiXScene); //摄像机参数初始化
        //estimation_setup(scenePath);
        initLaunchParams(OptiXScene);//params通用参数设置
        dropOutTracingParamsInit();//用于proxy sampling的参数params简单初始化，只是把一些指针置空，真正的参数设置在下面的dropOutTracingParamsSetup函数里
        lt_params_setup(OptiXScene);//用于light trace的参数params设置
        preTracer_params_setup(OptiXScene);//用于预追踪的参数params设置

        //pre tracing
        {
            handleCameraUpdate(params);//更新相机参数
            path_guiding_params_setup(OptiXScene);//用于path guiding的参数params设置
            dropOutTracingParamsSetup(OptiXScene);//用于proxy sampling的参数params设置
            preprocessing(OptiXScene);//预处理，追踪完整路径，建立子空间，训练Gamma矩阵（在代码里也被称为E矩阵），统计Q向量，
            //如果启用了proxy sampling，那么也会统计proxy sampling的数据，
            //得到的大部分信息都会传入device端，在params里存储device端指针
        }

        if (true)
        {
            //设置各种callback
            GLFWwindow* window = sutil::initUI("optixPathTracer", params.width, params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            //
            // Render loop
            // 主循环
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, params.width, params.height);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> light_tracing_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);
                std::chrono::duration<double> sum_render_time(0.0);
                std::chrono::duration<double> print_time(10.0);
                bool print = false;
                bool setting_changed = false;
                do
                {
                    glfwPollEvents();

                    updateState(output_buffer, params);
                    if (setting_changed) { params.subframe_index = 0; }
                    if (params.subframe_index == 0) { sum_render_time = std::chrono::duration<double>::zero(); render_time_record = 0; }

                    auto t0 = std::chrono::steady_clock::now();
                    if (render_main_progs[render_alg_id].need_light_trace)
                    {
                        //光子路追踪
                        //追踪到的光子路会存到LVC里
                        launchLVCTrace(OptiXScene);
                        //统计proxy sampling信息
                        updateDropOutTracingParams();
                        //统计proxy sampling启用时，SPCBPT用于重要性采样镜面顶点的CMF
                        updateDropOutTracingCombineWeight();
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    light_tracing_time += t1 - t0;
                    t0 = t1;

                    //调用渲染算法的视子路追踪部分
                    launchSubframe(output_buffer, OptiXScene);

                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    sum_render_time += t1 - t0;
                    t0 = t1;


                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    //控件操作
                    setting_changed = sutil::displayStatsControls(light_tracing_time, render_time, display_time,
                        params.subframe_index,
                        render_alg_id, mainProgNames(),
                        params.eye_subspace_visualize, params.light_subspace_visualize,
                        //从这里以下的几个参数都是暂时不用的
                        //或者说只有在proxy sampling(DOT)里会用到，真要用到的话进函数里把相应注释解除吧
                        params.caustic_path_only,
                        params.specular_subspace_visualize, params.caustic_prob_visualize, params.PG_grid_visualize,
                        params.pg_params.pg_enable,
                        params.error_heat_visual
                    );
                    glfwSwapBuffers(window);

                    //评估代码，暂时注释
                    estimation::es.estimation_mode = false;
                    if (estimation::es.estimation_mode == true)
                    {
                        float error = estimation::es.relMse_estimate(MyThrustOp::copy_to_host(params.accum_buffer, params.width * params.height), params);
                        printf("render time sum %f frame %d relMse %f\n", sum_render_time.count(), params.subframe_index, error);

                        error = estimation::es.MAPE_estimate(MyThrustOp::copy_to_host(params.accum_buffer, params.width * params.height), params);
                        printf("render time sum %f frame %d MAPE %f %%\n", sum_render_time, params.subframe_index, error * 100);

                        if (estimation_save) {
                            estimation::es.outputFile << params.subframe_index << " " << sum_render_time.count() << " " << error << endl;
                        }
                    }
                    else
                    {
                        printf("frame %d time %f\n", params.subframe_index, sum_render_time.count());
                    }
                    render_fps = 1.0 / (sum_render_time.count() - render_time_record);
                    render_time_record = sum_render_time.count();
                    render_frame_record = params.subframe_index;

                    ++params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }
            sutil::cleanupUI(window);
        }

        // cleanupState( state );
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
