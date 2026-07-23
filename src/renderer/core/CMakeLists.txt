set(optix_path_tracer_sources
  optixPathTracer.h
  launch_params.h
  device_compile_config.h
  renderer_config.h
  sceneLoader.h
  sceneLoader.cpp
  "${PROJECT_SOURCE_DIR}/third_party/tinyobjloader/tiny_obj_loader.h"
  tiny_obj_loader.cc
  scene_shift.cpp
  scene_shift.h
  cuProg.h
  "${PROJECT_SOURCE_DIR}/third_party/stb/stb_image.h"
  material_parameters.h
  light_parameters.h
  frame_estimation.h
  frame_estimation.cpp
  whitted.h
  BDPTVertex.h
  rmis.h
  PG_common.h
  PG_host.h
  pathControl.h
  DOT_PG_trainingParams.h
  decisionTree/classTree_common.h
  decisionTree/classTree_host.h
  decisionTree/classTree_device.h
  dropOutTracing_common.h
)

target_sources(spcbpt_renderer PRIVATE
  ${optix_path_tracer_sources}
)
target_include_directories(spcbpt_renderer PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Runtime loads only raygen + hit (see Scene::createPTXModule).
# optixPathTracer.cu is leftover NVIDIA sample code and is not loaded.
set(optix_path_tracer_shaders
  raygen.cu
  hit_program.cu
)

add_library(spcbpt_optix_ir OBJECT
  ${optix_path_tracer_shaders}
)
set_target_properties(spcbpt_optix_ir PROPERTIES
  CUDA_OPTIX_COMPILATION ON
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(spcbpt_optix_ir PRIVATE spcbpt_device_api)
target_compile_options(spcbpt_optix_ir PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${SPCBPT_OPTIX_INPUT_ARCH}>
)
spcbpt_configure_cuda_target(spcbpt_optix_ir)
set_property(TARGET spcbpt_optix_ir PROPERTY CUDA_ARCHITECTURES OFF)

add_custom_target(spcbpt_optix_ir_deploy
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPCBPT_OPTIX_IR_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          $<TARGET_OBJECTS:spcbpt_optix_ir>
          "${SPCBPT_OPTIX_IR_DIR}"
  DEPENDS spcbpt_optix_ir
  COMMAND_EXPAND_LISTS
  VERBATIM
)
