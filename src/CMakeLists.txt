set(OPTIX_IDE_FOLDER "SPCBPT")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPCBPT_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")
set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
set(SPCBPT_OPTIX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SPCBPT_OPTIX_IR_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optix-ir")

set(SAMPLES_PTX_DIR "${SPCBPT_OPTIX_OUTPUT_DIR}")
if(WIN32)
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR "${SAMPLES_PTX_DIR}")
endif()

set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE_DIR}\", \\
  \"${CUDAToolkit_INCLUDE_DIRS}\", ")

# Runtime loads prebuilt .optixir only (no NVRTC / PTX path).
set(CUDA_NVRTC_ENABLED OFF)
set(SAMPLES_INPUT_GENERATE_OPTIXIR ON)
set(SAMPLES_INPUT_GENERATE_PTX OFF)
set(CUDA_NVRTC_OPTIONS "")

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)
set(SPCBPT_GENERATED_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h")

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
  "${CMAKE_CURRENT_SOURCE_DIR}/device_include"
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
