#ifndef SCENE_SHIFT_H
#define SCENE_SHIFT_H
#include<sutil/Scene.h>
#include<sutil/Exception.h>
#include"sceneLoader.h"
#include"optixPathTracer.h"
template<class T>
BufferView<T> HostToDeviceBuffer(T* ptr, int count, int UniNum = 1)
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
    ans.elmt_byte_size = sizeof(T) / UniNum;
    return ans;
}
void Geometry_shift(Scene& Src, sutil::Scene& Dst);
void LightSource_shift(Scene& Src, MyParams& params, sutil::Scene& Dst);
void Scene_shift(Scene& Src, sutil::Scene& Dst);


class HDRLoader
{
public:
    SUTILAPI HDRLoader(const std::string& filename);
    SUTILAPI ~HDRLoader();

    SUTILAPI bool           failed()const;
    SUTILAPI unsigned int   width()const;
    SUTILAPI unsigned int   height()const;
    SUTILAPI float* raster()const;

    SUTILAPI sutil::Texture loadTexture(const float3& default_color, cudaTextureDesc* tex_desc);
private:
    unsigned int   m_nx;
    unsigned int   m_ny;
    float* m_raster;

    static void getLine(std::ifstream& file_in, std::string& s);

};

#endif // !SCENE_SHIFT_H
