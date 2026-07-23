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
  RendererRuntime.cpp
  RendererRuntime.h
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
    CUDA::cudart_static
    CUDA::cublas
)

spcbpt_configure_cuda_target(spcbpt_renderer)
set_target_properties(spcbpt_renderer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_property(TARGET spcbpt_renderer PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

add_subdirectory(core)
