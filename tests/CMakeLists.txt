add_executable(spcbpt_smoke
  smoke_main.cpp
)
add_dependencies(spcbpt_smoke spcbpt_optix_ir_deploy)

target_compile_definitions(spcbpt_smoke
  PRIVATE
    SUTIL_NO_OPENGL
)

target_include_directories(spcbpt_smoke
  PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party"
)

target_link_libraries(spcbpt_smoke
  PRIVATE
    spcbpt_renderer
    CUDA::cuda_driver
    CUDA::cudart_static
)

set_target_properties(spcbpt_smoke PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)

add_test(
  NAME spcbpt.headless_smoke
  COMMAND spcbpt_smoke
)
set_tests_properties(spcbpt.headless_smoke PROPERTIES
  LABELS "gpu;optix;smoke"
  TIMEOUT 120
)
