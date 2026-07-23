function(spcbpt_configure_cuda_target target_name)
  set_target_properties(${target_name} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4819,/wd4244,/wd4267,/wd4305,/wd4189,/wd4101>
  )

  target_compile_definitions(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:NVCC>
  )

  if(MSVC)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    )
    target_compile_definitions(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
    )
  endif()
endfunction()

function(spcbpt_add_optix_ir out_var)
  set(one_value_args TARGET_PREFIX)
  set(multi_value_args SOURCES INCLUDE_DIRS OPTIONS DEPENDS)
  cmake_parse_arguments(ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ARG_TARGET_PREFIX)
    message(FATAL_ERROR "spcbpt_add_optix_ir requires TARGET_PREFIX.")
  endif()

  set(_include_args)
  foreach(_dir IN LISTS ARG_INCLUDE_DIRS)
    if(_dir)
      list(APPEND _include_args "-I${_dir}")
    endif()
  endforeach()

  set(_host_compiler_args)
  if(CMAKE_CUDA_HOST_COMPILER)
    list(APPEND _host_compiler_args "-ccbin" "${CMAKE_CUDA_HOST_COMPILER}")
  endif()

  set(_windows_nvcc_args)
  if(MSVC)
    list(APPEND _windows_nvcc_args
      -allow-unsupported-compiler
      -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
      -Xcompiler=/utf-8,/wd4819
    )
  endif()

  set(_outputs)
  foreach(_source IN LISTS ARG_SOURCES)
    if(IS_ABSOLUTE "${_source}")
      set(_source_abs "${_source}")
    else()
      set(_source_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    endif()

    get_filename_component(_source_name "${_source_abs}" NAME)
    set(_output "${SPCBPT_OPTIX_OUTPUT_DIR}/${ARG_TARGET_PREFIX}_generated_${_source_name}.optixir")

    add_custom_command(
      OUTPUT "${_output}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPCBPT_OPTIX_OUTPUT_DIR}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
        "${_source_abs}"
        -optix-ir
        -o "${_output}"
        -arch=${SPCBPT_OPTIX_INPUT_ARCH}
        --use_fast_math
        -lineinfo
        -Wno-deprecated-gpu-targets
        -DNVCC
        -D_USE_MATH_DEFINES
        -DNOMINMAX
        ${_host_compiler_args}
        ${_windows_nvcc_args}
        ${_include_args}
        ${ARG_OPTIONS}
      DEPENDS "${_source_abs}" ${ARG_DEPENDS}
      COMMENT "Building OptiX IR ${ARG_TARGET_PREFIX}_generated_${_source_name}.optixir"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )

    list(APPEND _outputs "${_output}")
  endforeach()

  set(${out_var} ${_outputs} PARENT_SCOPE)
endfunction()
