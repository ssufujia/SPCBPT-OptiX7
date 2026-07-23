set(OPTIX_IDE_FOLDER "SPCBPT")
set(SPCBPT_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")
set(SPCBPT_OPTIX_IR_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optix-ir")

configure_file(spcbptConfig.h.in spcbptConfig.h @ONLY)
set(SPCBPT_GENERATED_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/spcbptConfig.h")

set(SPCBPT_COMMON_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/viewer"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

set(SPCBPT_OPTIX_INCLUDE_DIRS
  "${CUDAToolkit_INCLUDE_DIRS}"
  "${OptiX_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

add_library(spcbpt_common INTERFACE)
target_include_directories(spcbpt_common INTERFACE ${SPCBPT_COMMON_INCLUDE_DIRS})
target_link_libraries(spcbpt_common INTERFACE
  OptiX::OptiX
  spcbpt_project_options
)

add_library(spcbpt_device_api INTERFACE)
target_include_directories(spcbpt_device_api INTERFACE
  ${SPCBPT_OPTIX_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/renderer/core"
)
target_compile_definitions(spcbpt_device_api INTERFACE
  _USE_MATH_DEFINES
  NOMINMAX
)
target_link_libraries(spcbpt_device_api INTERFACE OptiX::OptiX)

include("${PROJECT_SOURCE_DIR}/cmake/OptiXModern.cmake")

if(SPCBPT_BUILD_DEMAND_LOADING)
  add_subdirectory(lib/ImageSource)
  add_subdirectory(lib/DemandLoading)
endif()
add_subdirectory(renderer)
add_subdirectory(viewer)
add_subdirectory(app)
