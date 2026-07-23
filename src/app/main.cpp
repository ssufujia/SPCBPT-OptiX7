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

#include <glad/gl.h> // Needs to be included before cuda_gl_interop.
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <spcbptConfig.h>

#include <GLFW/glfw3.h>
#include <renderer/RendererRuntime.h>
#include <renderer/RendererWorkflow.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Scene.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "cuda_thrust/device_thrust.h"
#include "frame_estimation.h"

#include <chrono>
#include <cstdlib>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <io.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static double render_time_record = 0.0;
static int render_frame_record = 0;
static constexpr bool ESTIMATION_SAVE = true;

bool resize_dirty = false;
bool minimized = false;
bool camera_changed = true;

spcbpt::RendererRuntime renderer;
spcbpt::RendererWorkflow workflow( renderer );
MyParams& params = renderer.params();
sutil::Camera camera;
sutil::Trackball trackball;

int32_t width = 1920;
int32_t height = 1000;
int32_t mouse_button = -1;

std::vector<std::string> render_alg = {
    std::string( "pt" ),
    std::string( "SPCBPT_eye" ),
    std::string( "SPCBPT_eye_ForcePure" )
};
int render_alg_id = 1;
bool one_frame_render_only = false;
float render_fps = 60.0f;

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos = 0.0;
    double ypos = 0.0;
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
    MyParams* window_params =
        static_cast<MyParams*>( glfwGetWindowUserPointer( window ) );
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking(
            static_cast<int>( xpos ),
            static_cast<int>( ypos ),
            window_params->width,
            window_params->height
        );
        camera_changed = true;
        renderer.markImageDirty();
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking(
            static_cast<int>( xpos ),
            static_cast<int>( ypos ),
            window_params->width,
            window_params->height
        );
        camera_changed = true;
        renderer.markImageDirty();
    }
}

static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    if( minimized )
        return;

    sutil::ensureMinimumSize( res_x, res_y );
    MyParams* window_params =
        static_cast<MyParams*>( glfwGetWindowUserPointer( window ) );
    window_params->width  = res_x;
    window_params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
    renderer.markImageDirty();
}

static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = iconified > 0;
}

void img_save( double render_time = -1.0, int frame = 0 )
{
    sutil::ImageBuffer output_buffer;
    auto host_buffer =
        MyThrustOp::copy_to_host( params.frame_buffer, params.height * params.width );
    output_buffer.data = host_buffer.data();
    output_buffer.width = params.width;
    output_buffer.height = params.height;
    output_buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t( now );
    if( _access_s( "data", 0 ) != 0 )
        _mkdir( "data" );
    else
        std::cout << "already has data" << std::endl;

    std::stringstream filename;
    filename << "./data/"
             << std::put_time( std::localtime( &now_time ), "%Y_%m_%d%H_%M_%S" )
             << "_" << frame << "iterations_" << render_time << "time";
    sutil::saveImage( ( filename.str() + ".png" ).c_str(), output_buffer, true );

    const auto accumulation =
        MyThrustOp::copy_to_host( params.accum_buffer, params.height * params.width );
    std::ofstream output_file( filename.str() + ".txt" );
    output_file << params.width << " " << params.height << std::endl;
    for( unsigned int i = 0; i < params.width * params.height; ++i )
    {
        output_file << accumulation[i].x << " "
                    << accumulation[i].y << " "
                    << accumulation[i].z << " "
                    << accumulation[i].w << std::endl;
    }
}

static void keyCallback(
    GLFWwindow* window,
    int32_t key,
    int32_t /*scancode*/,
    int32_t action,
    int32_t /*mods*/
)
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
        else if( key == GLFW_KEY_C )
        {
            std::printf( "Camera Info:\n" );
            std::printf( "up      %f %f %f\n", camera.up().x, camera.up().y, camera.up().z );
            std::printf( "eye     %f %f %f\n", camera.eye().x, camera.eye().y, camera.eye().z );
            std::printf(
                "lookat  %f %f %f\n",
                camera.lookat().x,
                camera.lookat().y,
                camera.lookat().z
            );
        }
        else if( key == GLFW_KEY_S )
        {
            img_save( render_time_record, render_frame_record );
        }
        else if( key == GLFW_KEY_SPACE )
        {
            render_alg_id = ( render_alg_id + 1 ) % render_alg.size();
            params.spcbpt_pure =
                render_alg[render_alg_id] == "SPCBPT_eye_ForcePure"
                    ? true
                    : renderer.config().spcbpt_pure;
            std::printf( "raygen switching to %s\n", render_alg[render_alg_id].c_str() );
            camera_changed = true;
            resize_dirty = true;
        }
        else if( key == GLFW_KEY_P )
        {
            one_frame_render_only = !one_frame_render_only;
        }
    }

    if( key == GLFW_KEY_W )
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        const float3 direction = normalize( lookat - eye );
        constexpr float speed = 0.5f;
        eye += direction / render_fps * speed;
        lookat += direction / render_fps * speed;
        camera.setEye( eye );
        camera.setLookat( lookat );
        camera_changed = true;
        resize_dirty = true;
        renderer.markImageDirty();
    }
}

static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( static_cast<int>( yscroll ) ) )
    {
        camera_changed = true;
        renderer.markImageDirty();
    }
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 1920x1000\n";
    std::cerr << "         --scene=<path>              Override the default scene file\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::exit( 0 );
}

void handleResize(
    sutil::CUDAOutputBuffer<uchar4>& output_buffer,
    MyParams& render_params
)
{
    if( !resize_dirty )
        return;
    resize_dirty = false;
    output_buffer.resize( render_params.width, render_params.height );
    renderer.resize( render_params.width, render_params.height );
}

void handleCameraUpdate( MyParams& render_params )
{
    if( !camera_changed )
        return;
    camera_changed = false;
    camera.setAspectRatio(
        static_cast<float>( render_params.width )
        / static_cast<float>( render_params.height )
    );
    render_params.eye = camera.eye();
    camera.UVWFrame( render_params.U, render_params.V, render_params.W );
}

void updateState(
    sutil::CUDAOutputBuffer<uchar4>& output_buffer,
    MyParams& render_params
)
{
    if( camera_changed || resize_dirty || one_frame_render_only )
        render_params.subframe_index = 0;
    handleCameraUpdate( render_params );
    handleResize( output_buffer, render_params );
}

void estimation_setup( const string& path )
{
    string algorithm;
    switch( render_alg_id )
    {
        case 0:
            algorithm = "pt";
            break;
        case 1:
            algorithm = "lvcbpt+lighttrace";
            break;
        case 2:
            algorithm = "spcbpt";
            break;
        default:
            algorithm = "error";
            break;
    }

    string name = path.substr( path.rfind( '/' ) + 1 );
    name = name.substr( 0, name.rfind( '.' ) );
    const string output_path = name + "_" + algorithm + ".txt";
    cout << "save our estimate to " << output_path << endl;
    estimation::es.outputFile.open( output_path );
    estimation::es.outputFile
        << "{\n"
        << "name:" << name << endl
        << "height:" << params.height << endl
        << "width:" << params.width << endl
        << "algo:" << algorithm << endl
        << "}" << endl;
    estimation::es.estimation_update( "./ref/" + name + ".txt", false );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    uchar4* result_buffer = output_buffer.map();
    workflow.renderFrame( result_buffer, render_alg[render_alg_id] );
    output_buffer.unmap();
}

void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>& output_buffer,
    sutil::GLDisplay& gl_display,
    GLFWwindow* window
)
{
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize( window, &framebuffer_width, &framebuffer_height );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuffer_width,
        framebuffer_height,
        output_buffer.getPBO()
    );
}

void initCameraState( const sutil::Scene& scene )
{
    camera = scene.camera();
    camera_changed = true;
    renderer.markImageDirty();
    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
        make_float3( 1.0f, 0.0f, 0.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 1.0f, 0.0f )
    );
    trackball.setGimbalLock( true );
}

int main( int argc, char* argv[] )
{
    params.caustic_path_only = 1;
    params.width = 1920;
    params.height = 1000;
    sutil::CUDAOutputBufferType output_buffer_type =
        sutil::CUDAOutputBufferType::GL_INTEROP;
    std::string scene_override;

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
            const std::string dimensions = arg.substr( 6 );
            int parsed_width = 0;
            int parsed_height = 0;
            sutil::parseDimensions(
                dimensions.c_str(),
                parsed_width,
                parsed_height
            );
            params.width = parsed_width;
            params.height = parsed_height;
        }
        else if( arg.rfind( "--scene=", 0 ) == 0 )
        {
            scene_override = arg.substr( 8 );
            if( scene_override.empty() )
            {
                std::cerr << "--scene requires a non-empty path\n";
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        spcbpt::SceneConfig scene_config = spcbpt::SceneConfig::defaultScene();
        if( !scene_override.empty() )
            scene_config.path = scene_override;
        const string& scene_path = scene_config.path;

        const unsigned int render_width = params.width;
        const unsigned int render_height = params.height;
        width = static_cast<int32_t>( render_width );
        height = static_cast<int32_t>( render_height );

        renderer.loadScene( scene_config );
        spcbpt::RendererConfig renderer_config;
        renderer_config.width  = render_width;
        renderer_config.height = render_height;
        renderer.initialize( renderer_config );
        sutil::Scene& scene = renderer.scene();

        initCameraState( scene );
        estimation_setup( scene_path );
        if( estimation::es.estimation_mode )
        {
            params.estimate_pr.ref_buffer = estimation::es.ref_ptr;
            params.estimate_pr.height = estimation::es.ref_height;
            params.estimate_pr.width = estimation::es.ref_width;
            params.estimate_pr.ready = true;
        }

        workflow.initializeAlgorithmState();
        handleCameraUpdate( params );
        std::printf(
            "Preprocessing (training) — window opens after this finishes...\n"
        );
        std::fflush( stdout );
        workflow.runPreprocessing();

        std::printf( "Creating GL window %dx%d\n", width, height );
        std::fflush( stdout );
        GLFWwindow* window = sutil::initUI( "optixPathTracer", width, height );
        glfwSetMouseButtonCallback( window, mouseButtonCallback );
        glfwSetCursorPosCallback( window, cursorPosCallback );
        glfwSetWindowSizeCallback( window, windowSizeCallback );
        glfwSetWindowIconifyCallback( window, windowIconifyCallback );
        glfwSetKeyCallback( window, keyCallback );
        glfwSetScrollCallback( window, scrollCallback );
        glfwSetWindowUserPointer( window, &params );
        glfwShowWindow( window );
        glfwFocusWindow( window );

        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                width,
                height
            );
            sutil::GLDisplay gl_display;
            std::chrono::duration<double> state_update_time( 0.0 );
            std::chrono::duration<double> render_time( 0.0 );
            std::chrono::duration<double> display_time( 0.0 );
            std::chrono::duration<double> sum_render_time( 0.0 );
            bool setting_changed = false;

            do
            {
                auto start = std::chrono::steady_clock::now();
                glfwPollEvents();
                updateState( output_buffer, params );
                if( setting_changed )
                    params.subframe_index = 0;
                if( params.subframe_index == 0 )
                    sum_render_time = std::chrono::duration<double>();

                auto end = std::chrono::steady_clock::now();
                state_update_time += end - start;
                start = end;

                launchSubframe( output_buffer );

                end = std::chrono::steady_clock::now();
                render_time += end - start;
                sum_render_time += end - start;
                start = end;

                displaySubframe( output_buffer, gl_display, window );
                end = std::chrono::steady_clock::now();
                display_time += end - start;

                setting_changed = sutil::displayStatsControls(
                    state_update_time,
                    render_time,
                    display_time,
                    params.eye_subspace_visualize,
                    params.light_subspace_visualize,
                    params.caustic_path_only,
                    params.specular_subspace_visualize,
                    params.caustic_prob_visualize,
                    params.PG_grid_visualize,
                    params.pg_params.pg_enable,
                    params.error_heat_visual
                );
                render_fps =
                    1.0f
                    / static_cast<float>(
                        display_time.count()
                        + render_time.count()
                        + state_update_time.count()
                    );
                glfwSwapBuffers( window );

                estimation::es.estimation_mode = 0;
                if( estimation::es.estimation_mode )
                {
                    float error = estimation::es.relMse_estimate(
                        MyThrustOp::copy_to_host(
                            params.accum_buffer,
                            params.width * params.height
                        ),
                        params
                    );
                    std::printf(
                        "render time sum %f frame %d relMse %f\n",
                        sum_render_time.count(),
                        params.subframe_index,
                        error
                    );
                    error = estimation::es.MAPE_estimate(
                        MyThrustOp::copy_to_host(
                            params.accum_buffer,
                            params.width * params.height
                        ),
                        params
                    );
                    std::printf(
                        "render time sum %f frame %d MAPE %f %%\n",
                        sum_render_time.count(),
                        params.subframe_index,
                        error * 100.0f
                    );
                    if( ESTIMATION_SAVE )
                    {
                        estimation::es.outputFile
                            << params.subframe_index << " "
                            << sum_render_time.count() << " "
                            << error << endl;
                    }
                }
                else
                {
                    std::printf(
                        "frame %d time %f\n",
                        params.subframe_index,
                        sum_render_time.count()
                    );
                }

                render_time_record = sum_render_time.count();
                render_frame_record = params.subframe_index;
                ++params.subframe_index;
            } while( !glfwWindowShouldClose( window ) );
            renderer.synchronize();
        }
        sutil::cleanupUI( window );
    }
    catch( const std::exception& error )
    {
        std::cerr << "Caught exception: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
