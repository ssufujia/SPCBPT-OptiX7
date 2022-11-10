# CMakeLists.txt for G4CU project

#project(device_thrust)
project(device_thrust LANGUAGES CXX CUDA)
# required cmake version
cmake_minimum_required(VERSION 2.8)


# packages
#find_package(CUDA)
find_package(CUDAToolkit)
#include_directories ("${PROJECT_SOURCE_DIR}")

# nvcc flags -g for debug
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 


#cuda_add_library(device_thrust SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(device_thrust STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES}) 
target_link_libraries(device_thrust CUDA::cublas)
#cuda_add_cublas_to_target(device_thrust)
#target_link_libraries(device_thrust PUBLIC 
#cublas)