add_executable(spcbpt_smoke
  smoke_main.cpp
)
add_dependencies(spcbpt_smoke spcbpt_optix_ir_deploy)

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

add_executable(spcbpt_renderer_linkcheck
  renderer_linkcheck.cpp
)
target_link_libraries(spcbpt_renderer_linkcheck
  PRIVATE
    spcbpt_renderer
)
if(MSVC)
  target_link_options(spcbpt_renderer_linkcheck
    PRIVATE
      "$<HOST_LINK:/WHOLEARCHIVE:$<TARGET_FILE:spcbpt_renderer>>"
  )
endif()
set_target_properties(spcbpt_renderer_linkcheck PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)

add_test(
  NAME spcbpt.renderer_linkcheck
  COMMAND spcbpt_renderer_linkcheck
)
set_tests_properties(spcbpt.renderer_linkcheck PROPERTIES
  LABELS "architecture;renderer"
  TIMEOUT 30
)
