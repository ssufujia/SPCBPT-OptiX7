set(sutil_sources
  "${SPCBPT_GENERATED_CONFIG}"
  "${SPCBPT_SOURCE_DIR}/cuda/BufferView.h"
  "${SPCBPT_SOURCE_DIR}/cuda/GeometryData.h"
  "${SPCBPT_SOURCE_DIR}/cuda/Light.h"
  "${SPCBPT_SOURCE_DIR}/cuda/LocalGeometry.h"
  "${SPCBPT_SOURCE_DIR}/cuda/MaterialData.h"
  "${SPCBPT_SOURCE_DIR}/cuda/util.h"
  "${SPCBPT_SOURCE_DIR}/cuda/helpers.h"
  Aabb.h
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
  sutilapi.h
  sutil.cpp
  sutil.h
  Trackball.cpp
  Trackball.h
  vec_math.h
)

add_library(spcbpt_viewer STATIC
  ${sutil_sources}
)
add_library(sutil_7_sdk ALIAS spcbpt_viewer)

target_compile_definitions(spcbpt_viewer PUBLIC
  SUTILAPI=
  $<$<PLATFORM_ID:Windows>:GLAD_API_CALL_EXPORT>
)

target_include_directories(spcbpt_viewer
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(spcbpt_viewer
  PUBLIC
    spcbpt_renderer
  PRIVATE
    spcbpt_common
    glfw
    glad
    imgui
    CUDA::cuda_driver
    CUDA::cudart_static
)

if(WIN32)
  target_link_libraries(spcbpt_viewer PRIVATE winmm.lib)
endif()

set_property(TARGET spcbpt_viewer PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
