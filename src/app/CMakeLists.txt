add_executable(optixPathTracer
  main.cpp
)
add_dependencies(optixPathTracer spcbpt_optix_ir_deploy)

target_include_directories(optixPathTracer
  PRIVATE
    "${SPCBPT_SOURCE_DIR}/renderer/core"
    "${SPCBPT_SOURCE_DIR}/cuda"
    "${SPCBPT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_compile_definitions(optixPathTracer
  PRIVATE
    OPTIX_SAMPLE_NAME_DEFINE=optixPathTracer
    OPTIX_SAMPLE_DIR_DEFINE=optixPathTracer
)

target_link_libraries(optixPathTracer
  PRIVATE
    spcbpt_common
    spcbpt_viewer
    glfw
    glad
    imgui
    CUDA::cuda_driver
    ${SPCBPT_CUDART_TARGET}
    CUDA::curand
)

set_target_properties(optixPathTracer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)
