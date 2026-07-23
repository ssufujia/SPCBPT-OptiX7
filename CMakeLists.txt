cmake_minimum_required(VERSION 3.27)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported. "
    "Use a preset or configure into a separate build directory."
  )
endif()

set(SPCBPT_CUDA_ARCHITECTURES "89" CACHE STRING
  "CUDA architectures for native host CUDA kernels."
)
set(CMAKE_CUDA_ARCHITECTURES "${SPCBPT_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
set(SPCBPT_OPTIX_INPUT_ARCH "sm_89" CACHE STRING
  "NVCC -arch used when compiling OptiX-IR (.optixir)."
)

if(CMAKE_HOST_WIN32)
  string(APPEND CMAKE_CUDA_FLAGS_INIT
    " -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  )
endif()

project(SPCBPT-OptiX LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

option(SPCBPT_BUILD_DEMAND_LOADING
  "Build unused OptiX sample DemandLoading/ImageSource libraries."
  OFF
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)

function(spcbpt_default_optix_root out_var)
  set(_candidates
    "$ENV{OptiX_ROOT}"
    "$ENV{OPTIX_ROOT}"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
  )
  foreach(_candidate IN LISTS _candidates)
    if(_candidate AND EXISTS "${_candidate}/include/optix.h")
      set(${out_var} "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

spcbpt_default_optix_root(_spcbpt_default_optix_root)
set(OptiX_ROOT "${_spcbpt_default_optix_root}" CACHE PATH
  "Root directory of the OptiX SDK."
)

find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  HINTS "${OptiX_ROOT}"
  PATH_SUFFIXES include
)

if(NOT OptiX_INCLUDE_DIR)
  message(FATAL_ERROR
    "OptiX headers not found. Set OptiX_ROOT to the OptiX SDK directory."
  )
endif()

file(READ "${OptiX_INCLUDE_DIR}/optix.h" _spcbpt_optix_header)
string(REGEX MATCH
  "#define[ \t]+OPTIX_VERSION[ \t]+([0-9]+)"
  _spcbpt_optix_version_match
  "${_spcbpt_optix_header}"
)
if(CMAKE_MATCH_1)
  set(OptiX_VERSION_NUMBER "${CMAKE_MATCH_1}")
  math(EXPR OptiX_VERSION_MAJOR "${OptiX_VERSION_NUMBER} / 10000")
  math(EXPR OptiX_VERSION_MINOR "(${OptiX_VERSION_NUMBER} % 10000) / 100")
  math(EXPR OptiX_VERSION_PATCH "${OptiX_VERSION_NUMBER} % 100")
  set(OptiX_VERSION
    "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_PATCH}"
  )
else()
  set(OptiX_VERSION "unknown")
endif()

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE "${OptiX_INCLUDE_DIR}")

if(OptiX_VERSION_NUMBER AND OptiX_VERSION_NUMBER LESS 80000)
  message(FATAL_ERROR
    "SPCBPT requires OptiX 8+ (found ${OptiX_VERSION}). "
    "Set OptiX_ROOT to an OptiX 9.x SDK."
  )
endif()

message(STATUS "Found OptiX ${OptiX_VERSION}: ${OptiX_INCLUDE_DIR}")
message(STATUS "Found CUDA Toolkit ${CUDAToolkit_VERSION}: ${CUDAToolkit_BIN_DIR}")
message(STATUS "Host CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OptiX-IR architecture: ${SPCBPT_OPTIX_INPUT_ARCH}")

add_library(spcbpt_project_options INTERFACE)

if(MSVC)
  target_compile_options(spcbpt_project_options INTERFACE
    $<$<COMPILE_LANGUAGE:C,CXX>:/MP>
    $<$<COMPILE_LANGUAGE:C,CXX>:/W3>
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4819>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4828>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4244>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4267>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4305>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4189>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4101>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4389>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4355>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4996>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4800>
  )
endif()

target_compile_definitions(spcbpt_project_options INTERFACE
  _USE_MATH_DEFINES
  NOMINMAX
)

set(OPTIX_IDE_FOLDER "SPCBPT")
set(SPCBPT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

include(CTest)

add_subdirectory(third_party)
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
