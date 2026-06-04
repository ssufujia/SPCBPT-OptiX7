# CMakeLists.txt for G4CU project
#
# The parent OptiX SDK CMake uses its bundled FindCUDA module.  Keep this
# target on the same path instead of enabling CMake's native CUDA language in a
# nested project, which would trigger a second CUDA compiler probe.

# nvcc flags -g for debug
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 


cuda_add_library(device_thrust STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(device_thrust ${CUDA_cublas_LIBRARY})
