#include"frame_estimation.h"
#include<sutil/sutil.h>
#include<optix.h>
namespace estimation
{ 
    void estimation_status::estimation_update(std::string reference_filepath, bool old_version)
    {
        estimation_mode = false;

        printf("loading reference img.....\n");
//        thrust::host_vector<float4> reference;
        if (reference_filepath == std::string(""))
        {
            reference_filepath = std::string("ref.txt");
        }  
        float a, b, c, d;

        std::ifstream inFile;
        inFile.open(reference_filepath.c_str());
        if (inFile.good() == false)
        {
            printf("Can't find the referencce file for estimation.\nTurn off the estimation mode\n");
            estimation_mode = false;
            return;
        }
//        reference.resize(params.width * params.height);
        if (old_version == false)
        {
            inFile >> ref_width >> ref_height;
        }
        else
        {
            ref_width = 1920;
            ref_height = 1001;
        }
        while (inFile >> a >> b >> c >> d)
        {
            float4 pixel = make_float4(a, b, c, d);
            reference.push_back(pixel); 
        }
        ref_ptr = MyThrustOp::reference_h2d(reference);
        if (ref_width * ref_height != reference.size())
        {
            printf("find a size dismatch problem in reference loading\n");
            printf("expected reference width : %d\n",ref_height);
            printf("expected reference height: %d\n",ref_width);
            printf("actual reference pixels size: %d\n", reference.size());
        }
        else
        {
            thrust::host_vector<uchar4> ref_img(reference.size());
            for (int i = 0; i < ref_img.size(); i++)
            {
                float4 pixel = reference[i];
                float lum = 0.3 * pixel.x + 0.6 * pixel.y + 0.1 * pixel.z;
                float limit = 1.5;
                float4 tone = pixel * 1.0f / (1.0f + lum / limit);
                float kInvGamma = 1.0f / 2.2f;
                float3 gamma_color = make_float3(pow(tone.x, kInvGamma), pow(tone.y, kInvGamma), pow(tone.z, kInvGamma));
                gamma_color.x = fminf(1.0f, gamma_color.x);
                gamma_color.y = fminf(1.0f, gamma_color.y);
                gamma_color.z = fminf(1.0f, gamma_color.z);

                uchar4 color = make_uchar4(unsigned(gamma_color.x * 255), unsigned(gamma_color.y * 255), unsigned(gamma_color.z * 255), 255); 
                ref_img[i] = color;
            }
            sutil::ImageBuffer outputbuffer;
            outputbuffer.data = ref_img.data();
            outputbuffer.width = ref_width;
            outputbuffer.height = ref_height;
            outputbuffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage("reference_load.png", outputbuffer, false);
        }
        inFile.close();
        estimation_mode = true;
    }
    float estimation_status::MAPE_estimate(thrust::host_vector<float4> accm, const MyParams& params)
    {
        if (estimation_mode == false)
        {
            return 0.0;
        }
        if (params.width != ref_width || params.height != ref_height)
        {
            printf("Dismatch img size found in estimation\n");
            printf("reference size width %d and height %d\n", ref_width, ref_height);
            printf("actual rendering size width %d and height %d\n", params.width, params.height);
            return 0;
        }
        else
        {
            float mape = 0.0;
            float valid_pixels = 0;

            float minLimit = 0.01;
            float3 error3 = make_float3(0);
            for (int i = 0; i < ref_width * ref_height; i++)
            {
                float3 a = make_float3(accm[i]);
                float3 b = make_float3(reference[i]);
                if (b.x + b.y + b.z > 0)
                    valid_pixels += 1;
                else continue;
                float3 bias = a - b;
                float3 r_bias = (a - b) / (b + make_float3(minLimit)); 
                error3 += make_float3(abs(r_bias.x), abs(r_bias.y), abs(r_bias.z));
                float error = (abs(r_bias.x) + abs(r_bias.y) + abs(r_bias.z)) / 3;
                error = min(error, 50);
                mape += error;
            }
            error3 /= valid_pixels;
            printf("no limit mape 3 channels %f %f %f\n", error3.x, error3.y, error3.z);
            return mape / valid_pixels;
        }
    }
    float estimation_status::relMse_estimate(thrust::host_vector<float4> accm, const MyParams& params)
    {
        if (estimation_mode == false)
        {
            return 0.0;
        }
        if (params.width != ref_width || params.height != ref_height)
        {
            printf("Dismatch img size found in estimation\n");
            printf("reference size width %d and height %d\n", ref_width, ref_height);
            printf("actual rendering size width %d and height %d\n", params.width, params.height);
            return 0;
        }
        else
        { 
            float relmse = 0.0;
            float valid_pixels = 0;

            float minLimit = 0.01;
            for (int i = 0; i < ref_width * ref_height; i++)
            { 
                float3 a = make_float3(accm[i]);
                float3 b = make_float3(reference[i]);
                if (b.x + b.y + b.z > 0)
                    valid_pixels += 1;
                float3 bias = a - b;
                float3 r_bias = (a - b) / (b + make_float3(minLimit));
                float3 sqaure_rbias = r_bias * r_bias; 
                float error = (abs(sqaure_rbias.x) + abs(sqaure_rbias.y) + abs(sqaure_rbias.z)) / 3; 
                error = fmin(error, 100);
                relmse += error;
            }
            return relmse / valid_pixels;
        } 
    }
    estimation_status es; 
}