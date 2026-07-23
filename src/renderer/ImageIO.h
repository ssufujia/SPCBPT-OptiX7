#pragma once

#include <cuda_runtime.h>

#include <string>

namespace spcbpt
{

void saveRgba8Png(
    const std::string& path,
    const uchar4* pixels,
    unsigned int width,
    unsigned int height
);

} // namespace spcbpt
