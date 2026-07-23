#pragma once

#include <cuda_runtime.h>

namespace spcbpt
{

struct Texture
{
    cudaArray_t          array   = nullptr;
    cudaTextureObject_t  texture = 0;
};

} // namespace spcbpt
