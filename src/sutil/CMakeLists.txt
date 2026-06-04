set(sutil_sources
  "${SPCBPT_GENERATED_CONFIG}"
  "${SAMPLES_CUDA_DIR}/sphere.cu"
  "${SAMPLES_CUDA_DIR}/whitted.cu"
  "${SAMPLES_CUDA_DIR}/BufferView.h"
  "${SAMPLES_CUDA_DIR}/GeometryData.h"
  "${SAMPLES_CUDA_DIR}/Light.h"
  "${SAMPLES_CUDA_DIR}/LocalGeometry.h"
  "${SAMPLES_CUDA_DIR}/MaterialData.h"
  "${SAMPLES_CUDA_DIR}/util.h"
  "${SAMPLES_CUDA_DIR}/helpers.h"
  Aabb.h
  Camera.cpp
  Camera.h
  CUDAOutputBuffer.h
  Exception.h
  GLDisplay.cpp
  GLDisplay.h
  Matrix.h
  PPMLoader.cpp
  PPMLoader.h
  Preprocessor.h
  Quaternion.h
  Record.h
  Scene.cpp
  Scene.h
  sutilapi.h
  sutil.cpp
  sutil.h
  Trackball.cpp
  Trackball.h
  vec_math.h
)

set(sutil_optix_sources
  "${SAMPLES_CUDA_DIR}/sphere.cu"
  "${SAMPLES_CUDA_DIR}/whitted.cu"
)

spcbpt_add_optix_ir(sutil_optix_ir
  TARGET_PREFIX sutil
  SOURCES ${sutil_optix_sources}
  INCLUDE_DIRS
    ${SPCBPT_OPTIX_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${sutil_sources}
)

set_source_files_properties(${sutil_optix_sources} PROPERTIES HEADER_FILE_ONLY ON)

add_library(sutil_7_sdk SHARED
  ${sutil_sources}
  ${sutil_optix_ir}
)

if(WIN32)
  target_compile_definitions(sutil_7_sdk PUBLIC GLAD_GLAPI_EXPORT)
endif()

target_include_directories(sutil_7_sdk
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(sutil_7_sdk
  PRIVATE
    spcbpt_common
    glfw
    glad
    imgui
    CUDA::cuda_driver
    CUDA::cudart_static
)

if(CUDA_NVRTC_ENABLED)
  target_link_libraries(sutil_7_sdk PRIVATE CUDA::nvrtc)
endif()

if(WIN32)
  target_link_libraries(sutil_7_sdk PRIVATE winmm.lib)
endif()

set_property(TARGET sutil_7_sdk PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
