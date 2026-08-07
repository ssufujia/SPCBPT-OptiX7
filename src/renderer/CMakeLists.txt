add_library(spcbpt_optimal_e_optimizer STATIC
  core/cuda_thrust/optimal_e_optimizer.cu
  core/cuda_thrust/optimal_e_optimizer.h
)
target_include_directories(spcbpt_optimal_e_optimizer
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core/cuda_thrust"
)
target_link_libraries(spcbpt_optimal_e_optimizer
  PUBLIC
    ${SPCBPT_CUDART_TARGET}
)
spcbpt_configure_cuda_target(spcbpt_optimal_e_optimizer)
set_target_properties(spcbpt_optimal_e_optimizer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  FOLDER "${OPTIX_IDE_FOLDER}"
)

add_library(spcbpt_renderer STATIC
  Aabb.h
  Camera.cpp
  Camera.h
  Exception.h
  ImageCodecs.cpp
  ImageIO.cpp
  ImageIO.h
  Matrix.h
  Preprocessor.h
  Quaternion.h
  Record.h
  RendererConfig.cpp
  RendererConfig.h
  RendererRuntime.cpp
  RendererRuntime.h
  SamplingProgress.h
  RendererWorkflow.cpp
  RendererWorkflow.h
  Scene.cpp
  Scene.h
  Texture.h
  vec_math.h
  "${CMAKE_CURRENT_SOURCE_DIR}/core/cuda_thrust/device_thrust.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/cuda_thrust/device_thrust.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/renderer_config.h"
)

target_include_directories(spcbpt_renderer
  PUBLIC
    "${SPCBPT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/cuda_thrust"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_link_libraries(spcbpt_renderer
  PUBLIC
    spcbpt_common
  PRIVATE
    CUDA::cuda_driver
    ${SPCBPT_CUDART_TARGET}
    spcbpt_optimal_e_optimizer
)

spcbpt_configure_cuda_target(spcbpt_renderer)
set_target_properties(spcbpt_renderer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_property(TARGET spcbpt_renderer PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

add_subdirectory(core)
