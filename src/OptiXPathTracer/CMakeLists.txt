#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include_directories(${SAMPLES_INCLUDE_DIR})
 
cmake_minimum_required(VERSION 2.8)


add_subdirectory(cuda_thrust)
set (EXTRA_LIBS ${EXTRA_LIBS} device_thrust)

OPTIX_add_sample_executable( optixPathTracer target_name
  optixPathTracer.cu
  optixPathTracer.cpp
  optixPathTracer.h

  sceneLoader.h
  sceneLoader.cpp
   
  tiny_obj_loader.h
  tiny_obj_loader.cc

  scene_shift.cpp
  scene_shift.h

  cuProg.h
  raygen.cu

  stb_image.h
  stb_image.cpp
  material_parameters.h
  light_parameters.h
  frame_estimation.h
  frame_estimation.cpp

  whitted.h
  BDPTVertex.h
  hit_program.cu
  rmis.h 

  PG_common.h
  PG_host.h
  pathControl.h
  #trainingParams.h
  tester.h
  DOT_PG_trainingParams.h
  decisionTree/classTree_common.h
  decisionTree/classTree_host.h
  decisionTree/classTree_device.h 
  dropOutTracing_common.h

  OPTIONS -rdc true
  )
  
  
target_link_libraries (optixPathTracer ${EXTRA_LIBS})
target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )