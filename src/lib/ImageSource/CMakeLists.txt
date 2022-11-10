#
#  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# CUDA compilation doesn't pick up the ImageSource include path target property.
include_directories( include )

cuda_wrap_srcs( DeviceMandelbrotImageKernels OBJ mandelbrot_generated_files DeviceMandelbrotImageKernels.cu "" OPTIONS "" )
cuda_wrap_srcs( DeviceConstantImageKernels OBJ constant_generated_files DeviceConstantImageKernels.cu "" OPTIONS "" )

add_library( ImageSource STATIC
  CheckerBoardImage.cpp
  DeviceConstantImage.cpp
  DeviceMandelbrotImage.cpp
  Exception.h
  ImageSource.cpp
  Stopwatch.h
  TextureInfo.cpp
  include/ImageSource/CheckerBoardImage.h
  include/ImageSource/DeviceConstantImage.h 
  include/ImageSource/DeviceConstantImageParams.h
  include/ImageSource/DeviceMandelbrotImage.h
  include/ImageSource/DeviceMandelbrotParams.h
  include/ImageSource/ImageSource.h
  include/ImageSource/MultiCheckerImage.h
  include/ImageSource/TextureInfo.h
  ${mandelbrot_generated_files}
  ${constant_generated_files}
  )

target_include_directories( ImageSource PUBLIC
  include
  )

find_package( OpenEXR )
if( NOT OpenEXR_FOUND )
  message( WARNING "OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

find_package( ZlibStatic )
if ( NOT ZlibStatic_FOUND )
  message( WARNING "Zlib not found (see ZlibStatic_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

if( OpenEXR_FOUND AND ZlibStatic_FOUND )
  # Build our EXRImage class if OpenEXR is available, adding the necessary include path and libraries.
  target_sources( ImageSource PRIVATE
    EXRReader.cpp
    include/ImageSource/EXRReader.h )
  target_link_libraries( ImageSource PUBLIC OpenEXR::OpenEXR Zlib::Static )
  target_compile_definitions( ImageSource PUBLIC OPTIX_SAMPLE_USE_OPEN_EXR )

  # The CoreReader depends on OpenEXR 3.1 or newer
  if( ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0" )
    target_sources( ImageSource PRIVATE
      CoreEXRReader.cpp
      include/ImageSource/CoreEXRReader.h )
    target_compile_definitions( ImageSource PUBLIC OPTIX_SAMPLE_USE_CORE_EXR )
  endif()    
endif()

set_property(TARGET ImageSource PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
