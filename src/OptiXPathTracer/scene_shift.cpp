#include "scene_shift.h" 
#include<sutil/Exception.h>
#include<optix.h> 
#include"stb_image.h"
#include<direct.h>
#include<map>

sutil::Aabb get_aabb(std::vector<float3> v)
{

    sutil::Aabb ans;
    for (auto p:v)
    {
        ans.include(p);
    }
    return ans;
}
sutil::Aabb get_aabb(std::vector<float> v)
{
    sutil::Aabb ans;
    for (int i = 0; i < v.size() / 3; i += 3)
    {
        int id1 = i;
        int id2 = i + 1;
        int id3 = i + 2;
        auto f3 = make_float3(v[id1], v[id2], v[id3]);
        ans.include(f3);
    }
    return ans;
} 
static std::map<int, int> materialID_remap;
static std::map<int, int> lightsourceID_remap;
static std::map<int, int> sampler_remap;
void Material_shift(Scene& Src, sutil::Scene& Dst)
{
    for (int i = 0; i < Src.texture_map.size(); i++)
    {

        int texWidth, texHeight, texChannels;
        std::string name = std::string(SAMPLES_DIR) + std::string("/data/") + Src.texture_map[i];
        stbi_uc* pixels = stbi_load(name.c_str(),
            &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);


        if (!pixels)
        {
            std::cout << "error image loading" << name<< std::endl;
        }
        auto data_p = reinterpret_cast<uint32_t*>(pixels);

        Dst.addImage(
            texWidth,
            texHeight,
            8,
            4,
            data_p
        );
        int image_id = Dst.ImagesSize() - 1;
        const cudaTextureAddressMode address_s = cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = cudaAddressModeWrap;
        const cudaTextureFilterMode  filter = cudaFilterModeLinear;
        Dst.addSampler(address_s, address_t, filter, image_id); 
        sampler_remap[i + 1] = Dst.SamplerCurrent();
    }
    for (int i = 0; i < Src.optix_materials.size(); i++)
    { 
        auto& p = Src.optix_materials[i];
        MaterialData mtl;
        
        mtl.doubleSided = true;

        mtl.alpha_mode = MaterialData::ALPHA_MODE_OPAQUE;
        mtl.pbr = p;
        //mtl.pbr.base_color = make_float4(p.color,1.0);
        mtl.pbr.base_color.w = 1.0;
        //mtl.pbr.metallic = p.metallic;
        //mtl.pbr.roughness = p.roughness;
        //mtl.pbr.trans = p.trans;
        //mtl.pbr.eta = p.eta;
        //mtl.pbr.brdf = p.brdf; 
        if (p.base_color_tex.tex != 0)
        {
            mtl.pbr.base_color_tex.tex = sampler_remap[p.base_color_tex.tex];
            mtl.pbr.base_color_tex.texcoord = 0;

            float2 offset = { 0, 0 };
            float  rotation = 0;
            float2 scale = { 1, 1 };
            mtl.pbr.base_color_tex.texcoord_offset = offset;
            mtl.pbr.base_color_tex.texcoord_scale = scale;
            mtl.pbr.base_color_tex.texcoord_rotation = make_float2((float)sinf(rotation), (float)cosf(rotation));
        }
        if (p.normal_tex.tex != 0)
        {
            mtl.normal_tex.tex = sampler_remap[p.normal_tex.tex];
            mtl.normal_tex.texcoord = 0;

            float2 offset = { 0, 0 };
            float  rotation = 0;
            float2 scale = { 1, 1 };
            mtl.normal_tex.texcoord_offset = offset;
            mtl.normal_tex.texcoord_scale = scale;
            mtl.normal_tex.texcoord_rotation = make_float2((float)sinf(rotation), (float)cosf(rotation));
        }  
        materialID_remap[i] = Dst.MaterialsSize();
        //printf("material remap id %d -> %d\n", i, materialID_remap[i]);
        Dst.addMaterial(mtl);
    }
    for (int i = 0; i < Src.optix_lights.size(); i++)
    {
        auto &light = Src.optix_lights[i];
        if (light.type != Light::Type::QUAD)
            continue;
        MaterialData mtl;
        mtl.doubleSided = true;
        mtl.emissive_factor = light.quad.emission;
        mtl.light_id = i;
        lightsourceID_remap[i] = Dst.MaterialsSize();
        //printf("lightsource remap id %d -> %d\n", i, lightsourceID_remap[i]);
        Dst.addMaterial(mtl);

    }

    return;

}
void LightSource_shift(Scene& Src, MyParams& params, sutil::Scene& Dst)
{ 
    int ssBase = Src.has_envMap() ? 0.5 * NUM_SUBSPACE_LIGHTSOURCE : 0;

    std::vector<Light>& lights = Src.optix_lights;
    for (int i = 0; i < lights.size(); i++)
    {
        //auto& SL = Src.lights[i];
        Light& light = lights[i]; 
        //if (SL.lightType == LightType::DIRECTION)
        //{
        //    Dst.addDirectionalLight(SL.direction, SL.emission);
        //    continue;
        //    light.type = Light::Type::DIRECTIONAL;
        //    light.directional.intensity = SL.emission;
        //    light.directional.direction = SL.direction;
        //}
        //else if (SL.lightType == LightType::QUAD)
        //{
        //    light.type = Light::Type::QUAD;
        //    light.quad.emission = SL.emission;
        //    light.quad.corner = SL.position;
        //    light.quad.u = SL.position + SL.u;
        //    light.quad.v = SL.position + SL.v;
        //    light.quad.normal = normalize(cross(SL.u, SL.v));
        //    light.quad.area = length(cross(SL.u, SL.v));
        //}
        //else
        //{
        //    continue;
        //}
//        light.type = SL.lightType == LightType::DIRECTION? Light::Type::DIRECTIONAL  
        light.id = i;
        //light.albedoID = sampler_remap[SL.albedoID];
        light.ssBase = ssBase;
        ssBase += light.divLevel * light.divLevel; 
        //printf("light source info %f %f %f\nlight source info u %f %f %f\nlight source info v %f %f %f\n", light.quad.corner.x, light.quad.corner.y, light.quad.corner.z,
        //    light.quad.u.x, light.quad.u.y, light.quad.u.z,
        //    light.quad.v.x, light.quad.v.y, light.quad.v.z
        //    );
    }
    if (Src.has_envMap())
    {
        Light light;
        light.type = Light::Type::ENV;
        light.id = lights.size();
        lights.push_back(light);

    }
    params.lights = HostToDeviceBuffer(lights.data(), lights.size());
}
void Camera_shift(Scene& Src, sutil::Scene& Dst)
{    
    sutil::Camera cam;
    cam.setEye(Src.eye);
//    cam.setDirection(normalize(Src.lookat - Src.eye));
    cam.setLookat(Src.lookat);
    cam.setFovY(Src.fov);
    cam.setUp(Src.up);
    
    Dst.addCamera(cam);
}
Vec2f make_Vec2f(float x, float y)
{
    Vec2f v;
    v.x = x;
    v.y = y;
    return v;
}
void Scene_shift(Scene& Src, sutil::Scene& Dst)
{
    Dst.removeCurrent();
    if (Src.env_file != std::string(""))
    {
        Dst.setEnvFilePath(Src.env_file);
    }
    Material_shift(Src, Dst);
    Camera_shift(Src, Dst);
    Geometry_shift(Src, Dst);
}
void Geometry_shift(Scene& Src, sutil::Scene& Dst)
{ 
    for (int k = 0; k < Src.mesh_names.size(); k++)
    {
        std::vector<tinyobj::shape_t>       m_shapes;
        std::vector<tinyobj::material_t>    m_materials;
        Src.getMeshData(k, m_shapes, m_materials);

        for (int j = 0; j < m_shapes.size(); j++)
        {
            auto mesh_ptr = std::make_shared<sutil::Scene::MeshGroup>();
            auto color_vector = std::vector<float>();
            Dst.addMesh(mesh_ptr);
            sutil::Scene::MeshGroup& a = *mesh_ptr;
            auto c_mesh = m_shapes[j].mesh;
            a.name = m_shapes[j].name;
             
            int num_points = c_mesh.positions.size() / 3;
            int num_faces = c_mesh.indices.size() / 3;
            while (c_mesh.texcoords.size() < c_mesh.positions.size() / 3 * 2)
            {
                c_mesh.texcoords.push_back(0);
            }
            //while (color_vector.size() < c_mesh.positions.size() / 3 * 4)
            //{
            //    color_vector.push_back(1);
            //}


            a.positions.push_back(HostToDeviceBuffer(
                reinterpret_cast<float3*>(c_mesh.positions.data()),
                num_points, 3));
            a.indices.push_back(HostToDeviceBuffer(
                reinterpret_cast<unsigned int*>(c_mesh.indices.data()),
                c_mesh.indices.size()));

            a.colors.push_back(BufferView<Vec4f>());
            auto BV = HostToDeviceBuffer(
                reinterpret_cast<Vec2f*>(c_mesh.texcoords.data()),
                num_points, 2);

            for (int i = 0; i < GeometryData::num_textcoords; i++)
            {
                a.texcoords[i].push_back(BV);

            }
            /* ********************************** */
            if(!Src.use_geometry_normal)
                a.normals.push_back(HostToDeviceBuffer(
                reinterpret_cast<float3*>(c_mesh.normals.data()),
                num_points,3));
            /* ********************************** */
            a.normals.push_back(BufferView<float3>());
            a.material_idx.push_back(materialID_remap[k]);
            a.object_aabb = get_aabb(c_mesh.positions);



            auto instance = std::make_shared<sutil::Scene::Instance>();
            auto T_matrix = sutil::Matrix4x4::identity();
            instance->transform = T_matrix;
            instance->mesh_idx = Dst.meshes().size() - 1;
            instance->world_aabb = a.object_aabb;
            instance->world_aabb.transform(T_matrix);
            Dst.addInstance(instance);
            //printf("obj%d shifted\n", k);
        }
        //break;
    }
    //return;
    for (int i = 0; i < Src.lights.size(); i++)
    {
        //auto &SLight = Src.lights[i];
        auto &SLight = Src.optix_lights[i];
        if (SLight.type != Light::Type::QUAD)
        {
            continue;
        }
        Light light = SLight;  
        
        Dst.addLight(light);


        auto mesh_ptr = std::make_shared<sutil::Scene::MeshGroup>(); 
        Dst.addMesh(mesh_ptr);
        sutil::Scene::MeshGroup& a = *mesh_ptr;

        int num_points = 6;
        int num_faces = 2;
        std::vector<float3> positions;
        positions.push_back(light.quad.corner);
        positions.push_back(light.quad.u);
        positions.push_back(light.quad.v);
        positions.push_back(light.quad.u + light.quad.v - light.quad.corner);
         
        std::vector<unsigned> indices;
        indices.push_back(0);
        indices.push_back(1);
        indices.push_back(3);
        indices.push_back(0);
        indices.push_back(3);
        indices.push_back(2); 
         

        std::vector<Vec2f> texcoords; 
        texcoords.push_back(make_Vec2f(0, 0));
        texcoords.push_back(make_Vec2f(1, 0));
        texcoords.push_back(make_Vec2f(0, 1));
        texcoords.push_back(make_Vec2f(1, 1)); 


        a.positions.push_back(HostToDeviceBuffer(
            reinterpret_cast<float3*>(positions.data()),
            num_points, 3));
        a.indices.push_back(HostToDeviceBuffer(
            reinterpret_cast<unsigned int*>(indices.data()),
            indices.size()));

        a.colors.push_back(BufferView<Vec4f>());
        auto BV = HostToDeviceBuffer(
            reinterpret_cast<Vec2f*>(texcoords.data()),
            num_points, 2);

        for (int i = 0; i < GeometryData::num_textcoords; i++)
        {
            a.texcoords[i].push_back(BV);
        }
        //a.normals.push_back(HostToDeviceBuffer(
        //    reinterpret_cast<float3*>(c_mesh.normals.data()),
        //    num_points,3)); 
        a.normals.push_back(BufferView<float3>());
        a.material_idx.push_back(lightsourceID_remap[i]);
        //printf("light id %d\n",a.material_idx[0]);
        //a.material_idx.push_back(materialID_remap[0]);
        a.object_aabb = get_aabb(positions);



        auto instance = std::make_shared<sutil::Scene::Instance>();
        auto T_matrix = sutil::Matrix4x4::identity();
        instance->transform = T_matrix;
        instance->mesh_idx = Dst.meshes().size() - 1;
        instance->world_aabb = a.object_aabb;
        instance->world_aabb.transform(T_matrix);
        Dst.addInstance(instance);
    }
}

#include <string> 
#include<fstream>

void HDRLoader::getLine(std::ifstream& file_in, std::string& s)
{
    for (;;) {
        if (!std::getline(file_in, s))
            return;
        if (s.empty()) return;
        std::string::size_type index = s.find_first_not_of("\n\r\t ");
        if (index != std::string::npos && s[index] != '#')
            break;
    }
}
namespace {

    // The error class to throw
    struct HDRError {
        std::string Er;
        HDRError(const std::string& st = "HDRLoader error") : Er(st) {}
    };

    union RGBe {
        struct {
            unsigned char r, g, b, e;
        };
        unsigned char v[4];
    };

    inline void RGBEtoFloats(const RGBe& RV, float* FV, float inv_img_exposure)
    {
        if (RV.e == 0)
            FV[0] = FV[1] = FV[2] = 0.0f;
        else {
            const int HDR_EXPON_BIAS = 128;
            float s = (float)ldexp(1.0, (int(RV.e) - (HDR_EXPON_BIAS + 8)));
            s *= inv_img_exposure;
            FV[0] = (RV.r + 0.5f) * s;
            FV[1] = (RV.g + 0.5f) * s;
            FV[2] = (RV.b + 0.5f) * s;
        }
    }


    void ReadScanlineNoRLE(std::ifstream& inf, RGBe* RGBEline, const size_t wid)
    {
        inf.read(reinterpret_cast<char*>(RGBEline), wid * sizeof(RGBe));
        if (inf.eof()) throw HDRError("Premature file end in ReadScanlineNoRLE");
    }

    void ReadScanline(std::ifstream& inf, RGBe* RGBEline, const size_t wid)
    {
        const size_t MinLen = 8, MaxLen = 0x7fff;
        if (wid<MinLen || wid>MaxLen) return ReadScanlineNoRLE(inf, RGBEline, wid);
        char c0, c1, c2, c3;
        inf.get(c0);
        inf.get(c1);
        inf.get(c2);
        inf.get(c3);
        if (inf.eof()) throw HDRError("Premature file end in ReadScanline 1");
        if (c0 != 2 || c1 != 2 || (c2 & 0x80)) {
            inf.putback(c3);
            inf.putback(c2);
            inf.putback(c1);
            inf.putback(c0);
            return ReadScanlineNoRLE(inf, RGBEline, wid); // Found an old-format scanline
        }

        if (size_t(size_t(c2) << 8 | size_t(c3)) != wid) throw HDRError("Scanline width inconsistent");

        // This scanline is RLE.
        for (unsigned int ch = 0; ch < 4; ch++) {
            for (unsigned int x = 0; x < wid; ) {
                unsigned char code;
                inf.get(reinterpret_cast<char&>(code));
                if (inf.eof()) throw HDRError("Premature file end in ReadScanline 2");
                if (code > 0x80) { // RLE span
                    char pix;
                    inf.get(pix);
                    if (inf.eof()) throw HDRError("Premature file end in ReadScanline 3");
                    code = code & 0x7f;
                    while (code--)
                        RGBEline[x++].v[ch] = pix;
                }
                else { // Arbitrary span
                    while (code--) {
                        inf.get(reinterpret_cast<char&>(RGBEline[x++].v[ch]));
                        if (inf.eof()) throw HDRError("Premature file end in ReadScanline 4");
                    }
                }
            }
        }
    }
};
HDRLoader::HDRLoader(const std::string& filename)
    : m_nx(0u), m_ny(0u), m_raster(0)
{
    if (filename.empty()) return;

    // Open file
    try {
        std::ifstream inf(filename.c_str(), std::ios::binary);

        if (!inf.is_open()) throw HDRError("Couldn't open file " + filename);

        std::string magic, comment;
        float exposure = 1.0f; 
        std::getline(inf, magic);
        if (magic != "#?RADIANCE") throw HDRError("File isn't Radiance.");
        for (;;) {
            getLine(inf, comment);

            // VS2010 doesn't let you look at the 0th element of a 0 length string, so this was tripping
            // debug asserts
            if (comment.empty()) break;
            if (comment[0] == '#') continue;

            if (comment.find("FORMAT") != std::string::npos) {
                if (comment != "FORMAT=32-bit_rle_rgbe") throw HDRError("Can only handle RGBe, not XYZe.");
                continue;
            }

            size_t ofs = comment.find("EXPOSURE=");
            if (ofs != std::string::npos) {
                exposure = (float)atof(comment.c_str() + ofs + 9);
            }
        }

        std::string major, minor;
        inf >> minor >> m_ny >> major >> m_nx;
        if (minor != "-Y" || major != "+X") throw "Can only handle -Y +X ordering";
        if (m_nx <= 0 || m_ny <= 0) throw "Invalid image dimensions";
        getLine(inf, comment); // Read the last newline of the header

        RGBe* RGBERaster = new RGBe[m_nx * m_ny];

        for (unsigned int y = 0; y < m_ny; y++) {
            ReadScanline(inf, RGBERaster + m_nx * y, m_nx);
        }

        m_raster = new float[m_nx * m_ny * 4];

        float inv_img_exposure = 1.0f / exposure;
        for (unsigned int i = 0; i < m_nx * m_ny; i++) {
            RGBEtoFloats(RGBERaster[i], m_raster + i * 4, inv_img_exposure);
        }
        delete[] RGBERaster;
    }
    catch (const HDRError& err) {
        std::cerr << "HDRLoader( '" << filename << "' ) failed to load file: " << err.Er << '\n';
        delete[] m_raster;
        m_raster = 0;
    }
    printf("load hdr file %s of %d %d rasters\n", filename.c_str(), m_nx * m_ny, m_nx * m_ny * 4);
}


HDRLoader::~HDRLoader()
{
    delete[] m_raster;
}


bool HDRLoader::failed()const
{
    return m_raster == 0;
}


unsigned int HDRLoader::width()const
{
    return m_nx;
}


unsigned int HDRLoader::height()const
{
    return m_ny;
}


float* HDRLoader::raster()const
{
    return m_raster;
}

sutil::Texture HDRLoader::loadTexture(const float3& default_color, cudaTextureDesc* tex_desc)
{
    std::vector<float> buffer;
    const unsigned int         nx = width();
    const unsigned int         ny = height();
    if (failed())
    {
        buffer.resize(4);
         

        buffer[0] = static_cast<float>(default_color.x);
        buffer[1] = static_cast<float>(default_color.y);
        buffer[2] = static_cast<float>(default_color.z);
        buffer[3] = 1.0;
    }
    else
    {
        printf("env info %d\n", nx * ny);
        buffer.resize(4 * nx * ny);

        for (unsigned int i = 0; i < nx; ++i)
        {
            for (unsigned int j = 0; j < ny; ++j)
            {

                unsigned int hdr_index = ((ny - j - 1) * nx + i) * 4;
                unsigned int buf_index = ((j)*nx + i) * 4;

                buffer[buf_index + 0] = raster()[hdr_index + 0];
                buffer[buf_index + 1] = raster()[hdr_index + 1];
                buffer[buf_index + 2] = raster()[hdr_index + 2];
                buffer[buf_index + 3] = 1.0;
            }
        }
    }

    // Allocate CUDA array in device memory
    int32_t               pitch = nx * 4 * sizeof(float);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, nx, ny));
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, buffer.data(), pitch, pitch, ny, cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc default_tex_desc = {};
    if (tex_desc == nullptr)
    {
        default_tex_desc.addressMode[0] = cudaAddressModeWrap;
        default_tex_desc.addressMode[1] = cudaAddressModeWrap;
        default_tex_desc.filterMode = cudaFilterModeLinear;
        default_tex_desc.readMode = cudaReadModeElementType;
        default_tex_desc.normalizedCoords = 1;
        default_tex_desc.maxAnisotropy = 1;
        default_tex_desc.maxMipmapLevelClamp = 99;
        default_tex_desc.minMipmapLevelClamp = 0;
        default_tex_desc.mipmapFilterMode = cudaFilterModePoint;
        default_tex_desc.borderColor[0] = 1.0f;
        default_tex_desc.sRGB = 0;  

        tex_desc = &default_tex_desc;
    }

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, tex_desc, nullptr));

    sutil::Texture hdr_texture = { cuda_array, cuda_tex };
    return hdr_texture;
}