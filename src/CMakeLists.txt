cmake_minimum_required(VERSION 3.24)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

set(SPCBPT_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures for host CUDA kernels.")
set(CMAKE_CUDA_ARCHITECTURES "${SPCBPT_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)

if(CMAKE_HOST_WIN32)
  string(APPEND CMAKE_CUDA_FLAGS_INIT " -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
endif()

project(SPCBPT-OptiX LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

find_package(CUDAToolkit REQUIRED)

function(spcbpt_default_optix_root out_var)
  set(_candidates
    "$ENV{OptiX_ROOT}"
    "$ENV{OPTIX_ROOT}"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
  )
  foreach(_candidate IN LISTS _candidates)
    if(_candidate AND EXISTS "${_candidate}/include/optix.h")
      set(${out_var} "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

spcbpt_default_optix_root(_spcbpt_default_optix_root)
set(OptiX_ROOT "${_spcbpt_default_optix_root}" CACHE PATH "Root directory of the OptiX SDK.")

find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  HINTS "${OptiX_ROOT}"
  PATH_SUFFIXES include
)

if(NOT OptiX_INCLUDE_DIR)
  message(FATAL_ERROR "OptiX headers not found. Set OptiX_ROOT to the OptiX SDK directory.")
endif()

file(READ "${OptiX_INCLUDE_DIR}/optix.h" _spcbpt_optix_header)
string(REGEX MATCH "#define[ \t]+OPTIX_VERSION[ \t]+([0-9]+)" _spcbpt_optix_version_match "${_spcbpt_optix_header}")
if(CMAKE_MATCH_1)
  set(OptiX_VERSION_NUMBER "${CMAKE_MATCH_1}")
  math(EXPR OptiX_VERSION_MAJOR "${OptiX_VERSION_NUMBER} / 10000")
  math(EXPR OptiX_VERSION_MINOR "(${OptiX_VERSION_NUMBER} % 10000) / 100")
  math(EXPR OptiX_VERSION_PATCH "${OptiX_VERSION_NUMBER} % 100")
  set(OptiX_VERSION "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_PATCH}")
else()
  set(OptiX_VERSION "unknown")
endif()

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE "${OptiX_INCLUDE_DIR}")

message(STATUS "Found OptiX ${OptiX_VERSION}: ${OptiX_INCLUDE_DIR}")
message(STATUS "Found CUDA Toolkit ${CUDAToolkit_VERSION}: ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/MP>
    $<$<COMPILE_LANGUAGE:C,CXX>:/W3>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4355>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4996>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4800>
    $<$<COMPILE_LANGUAGE:C,CXX>:/w34101>
    $<$<COMPILE_LANGUAGE:C,CXX>:/w34189>
    $<$<COMPILE_LANGUAGE:C,CXX>:/w34018>
    $<$<COMPILE_LANGUAGE:C,CXX>:/w34389>
  )
endif()

add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)

set(OPTIX_IDE_FOLDER "SPCBPT")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
set(SPCBPT_OPTIX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx")

set(SAMPLES_PTX_DIR "${SPCBPT_OPTIX_OUTPUT_DIR}")
if(WIN32)
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR "${SAMPLES_PTX_DIR}")
endif()

set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \"lib/DemandLoading/include\", \\
  \"lib/ImageSource/include\", \\
  \".\", ")

set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE_DIR}\", \\
  \"${CUDAToolkit_INCLUDE_DIRS}\", ")

option(CUDA_NVRTC_ENABLED "Use NVRTC to compile PTX at runtime." OFF)
set(SAMPLES_INPUT_GENERATE_OPTIXIR ON)
set(SAMPLES_INPUT_GENERATE_PTX OFF)
set(CUDA_NVRTC_OPTIONS "\\
  \"-std=c++11\", \\
  \"-arch\", \\
  \"compute_50\", \\
  \"-use_fast_math\", \\
  \"-lineinfo\", \\
  \"-default-device\", \\
  \"-rdc\", \\
  \"true\",")

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)
set(SPCBPT_GENERATED_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h")

set(SPCBPT_COMMON_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}"
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
target_link_libraries(spcbpt_common INTERFACE OptiX::OptiX)

include(CMake/OptiXModern.cmake)

add_subdirectory(support)
add_subdirectory(lib/ImageSource)
add_subdirectory(lib/DemandLoading)
add_subdirectory(sutil)
add_subdirectory(OptiXPathTracer)
