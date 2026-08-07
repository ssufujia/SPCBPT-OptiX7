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

add_test(
  NAME spcbpt.default_mode_init
  COMMAND spcbpt_smoke --default-mode-init
)
set_tests_properties(spcbpt.default_mode_init PROPERTIES
  LABELS "gpu;optix;smoke;config"
  TIMEOUT 120
)

add_test(
  NAME spcbpt.reload_smoke
  COMMAND spcbpt_smoke --reload-check
)
set_tests_properties(spcbpt.reload_smoke PROPERTIES
  LABELS "gpu;optix;smoke;reload"
  TIMEOUT 120
)

add_test(
  NAME spcbpt.camera_update_smoke
  COMMAND spcbpt_smoke --camera-update-check
)
set_tests_properties(spcbpt.camera_update_smoke PROPERTIES
  LABELS "gpu;optix;smoke;camera"
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

add_executable(spcbpt_renderer_config_test
  renderer_config_test.cpp
)
target_link_libraries(spcbpt_renderer_config_test PRIVATE spcbpt_renderer)
set_target_properties(spcbpt_renderer_config_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)
add_test(
  NAME spcbpt.renderer_config
  COMMAND spcbpt_renderer_config_test
)
set_tests_properties(spcbpt.renderer_config PROPERTIES
  LABELS "architecture;renderer;config"
  TIMEOUT 30
)

add_executable(spcbpt_optimal_e_optimizer_test
  optimal_e_optimizer_test.cu
)
target_link_libraries(spcbpt_optimal_e_optimizer_test
  PRIVATE
    spcbpt_optimal_e_optimizer
    ${SPCBPT_CUDART_TARGET}
)
spcbpt_configure_cuda_target(spcbpt_optimal_e_optimizer_test)
set_target_properties(spcbpt_optimal_e_optimizer_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)
add_test(
  NAME spcbpt.optimal_e_optimizer
  COMMAND spcbpt_optimal_e_optimizer_test
)
set_tests_properties(spcbpt.optimal_e_optimizer PROPERTIES
  LABELS "gpu;optimizer;training"
  TIMEOUT 30
)

add_executable(spcbpt_legacy_optimal_e_optimizer_test
  legacy_optimal_e_optimizer_test.cu
)
target_link_libraries(spcbpt_legacy_optimal_e_optimizer_test
  PRIVATE
    spcbpt_optimal_e_optimizer
    CUDA::cudart_static
)
spcbpt_configure_cuda_target(spcbpt_legacy_optimal_e_optimizer_test)
set_target_properties(spcbpt_legacy_optimal_e_optimizer_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  FOLDER "${OPTIX_IDE_FOLDER}"
)
add_test(
  NAME spcbpt.legacy_optimal_e_optimizer
  COMMAND spcbpt_legacy_optimal_e_optimizer_test
)
set_tests_properties(spcbpt.legacy_optimal_e_optimizer PROPERTIES
  LABELS "gpu;optimizer;training;legacy"
  TIMEOUT 30
)
