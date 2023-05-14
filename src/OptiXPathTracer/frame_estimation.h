#ifndef FRAME_ESTIMATION
#define FRAME_ESTIMATION

#include<vector>
#include<thrust/host_vector.h>
#include<string>
#include<iostream>
#include<fstream> 

#include"optixPathTracer.h"
#include"cuda_thrust/device_thrust.h"
namespace estimation
{ 
    struct estimation_status
    {
        thrust::host_vector<float4> reference;
        int ref_width;
        int ref_height;
        bool estimation_mode;
        float4* ref_ptr;
        estimation_status(std::string reference_filepath, bool old_version = false);
        float relMse_estimate(thrust::host_vector<float4> accm, const MyParams& params);
        float MAPE_estimate(thrust::host_vector<float4> accm, const MyParams& params);
    };
    extern estimation_status es;
}

#endif // !FRAME_ESTIMATION
