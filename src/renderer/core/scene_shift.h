#ifndef SCENE_SHIFT_H
#define SCENE_SHIFT_H
#include <renderer/Scene.h>
#include <renderer/Exception.h>
#include <renderer/Texture.h>
#include"sceneLoader.h"
#include"optixPathTracer.h"
template<class T>
BufferView<T> HostToDeviceBuffer(
    T* ptr,
    int count,
    int UniNum = 1,
    sutil::Scene* owner = nullptr
)
{
    T* devPtr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&devPtr),
        count * sizeof(T)
    ));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(devPtr),
        ptr,
        count * sizeof(T),
        cudaMemcpyHostToDevice
    ));
    BufferView<T> ans;
    ans.count = count;
    ans.data = reinterpret_cast<CUdeviceptr>(devPtr);
    ans.byte_stride = sizeof(T);
    ans.elmt_byte_size = static_cast<uint16_t>(sizeof(T) / UniNum);
    if (owner)
        owner->trackBuffer(reinterpret_cast<CUdeviceptr>(devPtr));
    return ans;
}
void Geometry_shift(Scene& Src, sutil::Scene& Dst);
void LightSource_shift(Scene& Src, MyParams& params, sutil::Scene& Dst);
void Scene_shift(Scene& Src, sutil::Scene& Dst);


class HDRLoader
{
public:
    HDRLoader(const std::string& filename);
    ~HDRLoader();

    bool           failed()const;
    unsigned int   width()const;
    unsigned int   height()const;
    float* raster()const;

    spcbpt::Texture loadTexture(const float3& default_color, cudaTextureDesc* tex_desc);
private:
    unsigned int   m_nx;
    unsigned int   m_ny;
    float* m_raster;

    static void getLine(std::ifstream& file_in, std::string& s);

};

#endif // !SCENE_SHIFT_H
