#pragma once

#include <optix.h>
#include <cstring>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include "rt_function.h"

#define GRID_SIZE 1
namespace dropOut_tracing
{
    // 正态分布随机数生成函数
    RT_FUNCTION __host__ float randn(float mean, float stdDev, unsigned int& seed) {
        static bool hasSpare = false;
        static float spare;
        if (hasSpare) {
            hasSpare = false;
            return mean + stdDev * spare;
        }
        hasSpare = true;
        float u, v, s;
        do {
            u = rnd(seed) * 2.0f - 1.0f;
            v = rnd(seed) * 2.0f - 1.0f;
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        s = std::sqrt(-2.0f * std::log(s) / s);
        spare = v * s;
        return mean + stdDev * u * s;
    }

    // 离散分布随机数生成函数
    RT_FUNCTION __host__ int randd(const float* weights, int numWeights, unsigned int& seed) {
        float sum = 0.0f;
        for (int i = 0; i < numWeights; ++i) {
            sum += weights[i];
        }
        float r = rnd(seed) * sum;
        for (int i = 0; i < numWeights; ++i) {
            if (r < weights[i]) {
                return i;
            }
            r -= weights[i];
        }
        return numWeights - 1;
    }

    class PGParams {
    public:
        RT_FUNCTION __host__ PGParams() : hasLoadln(false), trainEnd(false){}
        inline __host__ void loadIn(const std::vector<float2>& points);
        inline __host__ void predict_array(float2* point, int num);
        RT_FUNCTION __host__ float predict(float2& point, unsigned int& seed);
        RT_FUNCTION __host__ float pdf(float2& dirction);
        bool hasLoadln;
        bool trainEnd;
    private:
        inline __host__ void initializeGMM(const std::vector<float2>& newPoints);
        inline __host__ void updateGaussian(const std::vector<float2>& newPoints);
        inline __host__ void generatePoints(float2* point, int num);
        RT_FUNCTION __host__ float gaussianProbability(const float2& point, int component);
        RT_FUNCTION __host__ float lerp(const float& a, const float& b, const float t) { return a + t * (b - a); }

        static constexpr int numComponents = 3;
        static constexpr int maxIterations = 1000;
        static constexpr float convergenceThreshold = 1e-4f;

        float2 means[numComponents];
        float weights[numComponents];
        float2 stdDevs[numComponents];
        float covariances[numComponents][3];
    };

    __host__ void PGParams::loadIn(const std::vector<float2>& points) {
        //if (!hasLoadln)
        initializeGMM(points);
        hasLoadln = 1;
        try { updateGaussian(points); }
        catch (const std::exception& e) { std::cout << "error"<<e.what() << std::endl; hasLoadln = 0; }
    }

    __host__ void PGParams::predict_array(float2* point, int num) {
        generatePoints(point, num);
    }

    __host__ void PGParams::initializeGMM(const std::vector<float2>& points) {
        // Initialize weights with equal probabilities
        for (int i = 0; i < numComponents; ++i) {
            weights[i] = 1.0f / numComponents;
        }

        // Initialize means with random points from the dataset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points.size() - 1);

        for (int i = 0; i < numComponents; ++i) {
            int randomIndex = dis(gen);
            means[i] = points[randomIndex];
        }

        // Initialize standard deviations and covariances with the same value for all components
        float initialStdDev = 1.0f;
        float initialCovariance = 1.0f;
        for (int i = 0; i < numComponents; ++i) {
            stdDevs[i] = { initialStdDev, initialStdDev };

            // Initialize covariances with identity matrix
            covariances[i][0] = initialCovariance;
            covariances[i][1] = 0.0f;
            covariances[i][2] = initialCovariance;
        }
    }

    __host__ void PGParams::updateGaussian(const std::vector<float2>& points) {
        int numPoints = points.size();
        std::vector<std::vector<float>> responsibilities(numPoints, std::vector<float>(numComponents));

        float logLikelihood = 0;
        float previousLogLikelihood = -std::numeric_limits<float>::infinity();

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            // E step
            for (int i = 0; i < numPoints; ++i) {
                float totalProb = 0;
                for (int j = 0; j < numComponents; ++j) {
                    float prob = gaussianProbability(points[i], j) * weights[j];
                    totalProb += prob;
                    responsibilities[i][j] = prob;
                }

                // Normalize the responsibilities
                for (int j = 0; j < numComponents; ++j) {
                    responsibilities[i][j] /= totalProb;
                }
            }

            // M step
            for (int j = 0; j < numComponents; ++j) {
                float sumResponsibilities = 0;
                float2 newMean = { 0, 0 };
                float2 newStdDev = { 0, 0 };
                float newCovariance00 = 0;
                float newCovariance01 = 0;
                float newCovariance11 = 0;

                for (int i = 0; i < numPoints; ++i) {
                    sumResponsibilities += responsibilities[i][j];
                    newMean.x += responsibilities[i][j] * points[i].x;
                    newMean.y += responsibilities[i][j] * points[i].y;
                }

                weights[j] = sumResponsibilities / numPoints;
                means[j] = { newMean.x / sumResponsibilities, newMean.y / sumResponsibilities };

                for (int i = 0; i < numPoints; ++i) {
                    float dx = points[i].x - means[j].x;
                    float dy = points[i].y - means[j].y;
                    newStdDev.x += responsibilities[i][j] * dx * dx;
                    newStdDev.y += responsibilities[i][j] * dy * dy;
                    newCovariance00 += responsibilities[i][j] * dx * dx;
                    newCovariance01 += responsibilities[i][j] * dx * dy;
                    newCovariance11 += responsibilities[i][j] * dy * dy;
                }

                stdDevs[j] = { sqrt(newStdDev.x / sumResponsibilities), sqrt(newStdDev.y / sumResponsibilities) };
                covariances[j][0] = newCovariance00 / sumResponsibilities;
                covariances[j][1] = newCovariance01 / sumResponsibilities; // Covariance matrix is symmetric
                covariances[j][2] = newCovariance11 / sumResponsibilities;
            }

            // Compute log-likelihood
            logLikelihood = 0;
            for (int i = 0; i < numPoints; ++i) {
                float pointProb = 0;
                for (int j = 0; j < numComponents; ++j) {
                    pointProb += weights[j] * gaussianProbability(points[i], j);
                }
                logLikelihood += log(pointProb);
            }

            // Check for convergence
            if (fabs(logLikelihood - previousLogLikelihood) < convergenceThreshold) {
                break;
            }
            previousLogLikelihood = logLikelihood;
        }
    }

    RT_FUNCTION __host__ float PGParams::gaussianProbability(const float2& point, int component) {
        float dx = point.x - means[component].x;
        float dy = point.y - means[component].y;
        float varX = stdDevs[component].x * stdDevs[component].x;
        float varY = stdDevs[component].y * stdDevs[component].y;
        float det = covariances[component][0] * covariances[component][2] - covariances[component][1] * covariances[component][1];

        float exponent = -0.5f * (dx * dx * covariances[component][2] + dy * dy * covariances[component][0] - 2 * covariances[component][1] * dx * dy) / det;
        float normalization = 1.0f / (2 * M_PI * sqrt(det));

        return normalization * exp(exponent);
    }

    __host__ void PGParams::generatePoints(float2* point, int num) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> distX[numComponents], distY[numComponents];

        // Prepare weights for discrete_distribution
        std::vector<float> weights_vec(weights, weights + numComponents);
        std::discrete_distribution<int> componentDist(weights_vec.begin(), weights_vec.end());

        for (int j = 0; j < numComponents; ++j) {
            distX[j] = std::normal_distribution<float>(means[j].x, stdDevs[j].x);
            distY[j] = std::normal_distribution<float>(means[j].y, stdDevs[j].y);
        }

        for (int i = 0; i < num; ++i) {
            int component = componentDist(gen);
            point[i].x = distX[component](gen);
            point[i].y = distY[component](gen);
        }
    }

    RT_FUNCTION __host__ float PGParams::predict(float2& point,unsigned int& seed) {
        if (!hasLoadln) {
            printf("error\n");
            point = make_float2(0.14, 0.54);
            return 0.f;
        }

        int component = randd(weights, numComponents,seed);
        point.x = randn(means[component].x, stdDevs[component].x, seed);
        point.y = randn(means[component].y, stdDevs[component].y, seed);

        point.x = lerp(0.0f, 1.0f, point.x);
        point.y = lerp(0.0f, 1.0f, point.y);

        float pdf = 0;
        for (int j = 0; j < numComponents; ++j) {
            pdf += weights[j] * gaussianProbability(point, j);
        }
        
        return pdf;
    }

    RT_FUNCTION __host__ float PGParams::pdf(float2& dirction) {
        float pdf = 0;
        for (int j = 0; j < numComponents; ++j) {
            pdf += weights[j] * gaussianProbability(dirction, j);
        }
        return pdf;
    }
}