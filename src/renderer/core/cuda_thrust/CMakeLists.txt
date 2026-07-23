add_library(device_thrust STATIC
  device_thrust.cu
  device_thrust.h
)

target_include_directories(device_thrust PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(device_thrust PUBLIC spcbpt_common CUDA::cublas CUDA::cudart_static)
spcbpt_configure_cuda_target(device_thrust)
