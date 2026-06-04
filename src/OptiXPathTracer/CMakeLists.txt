add_subdirectory(cuda_thrust)

set(optix_path_tracer_sources
  optixPathTracer.cpp
  optixPathTracer.h
  sceneLoader.h
  sceneLoader.cpp
  tiny_obj_loader.h
  tiny_obj_loader.cc
  scene_shift.cpp
  scene_shift.h
  cuProg.h
  stb_image.h
  stb_image.cpp
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
  tester.h
  DOT_PG_trainingParams.h
  decisionTree/classTree_common.h
  decisionTree/classTree_host.h
  decisionTree/classTree_device.h
  dropOutTracing_common.h
)

set(optix_path_tracer_shaders
  optixPathTracer.cu
  raygen.cu
  hit_program.cu
)

spcbpt_add_optix_ir(optix_path_tracer_ir
  TARGET_PREFIX optixPathTracer
  SOURCES ${optix_path_tracer_shaders}
  INCLUDE_DIRS
    ${SPCBPT_OPTIX_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}"
  OPTIONS -rdc=true
  DEPENDS
    ${optix_path_tracer_sources}
    ${optix_path_tracer_shaders}
)

set_source_files_properties(${optix_path_tracer_shaders} PROPERTIES HEADER_FILE_ONLY ON)

add_executable(optixPathTracer
  ${optix_path_tracer_sources}
  ${optix_path_tracer_shaders}
  ${optix_path_tracer_ir}
)

target_include_directories(optixPathTracer
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cuda"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../device_include"
)

target_compile_definitions(optixPathTracer
  PRIVATE
    OPTIX_SAMPLE_NAME_DEFINE=optixPathTracer
    OPTIX_SAMPLE_DIR_DEFINE=optixPathTracer
)

target_link_libraries(optixPathTracer
  PRIVATE
    spcbpt_common
    sutil_7_sdk
    glfw
    glad
    imgui
    device_thrust
    CUDA::cuda_driver
    CUDA::cudart_static
)

set_property(TARGET optixPathTracer PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
