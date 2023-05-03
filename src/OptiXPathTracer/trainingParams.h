#pragma once

#include <optix.h>
#include <cstring>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <stdio.h>
#include <vector>
#include "rt_function.h"

#define GRID_SIZE 4

struct trainingParams
{
	float grid[GRID_SIZE][GRID_SIZE];
	/* prefixSum[i] = grid[0] + grid[1] + ... + grid[i] */
	float prefixSum[GRID_SIZE * GRID_SIZE];

	RT_FUNCTION __host__ trainingParams() {
		memset(grid, 0, sizeof(grid));
		memset(prefixSum, 0, sizeof(prefixSum));
	}

	RT_FUNCTION __host__ void printGrid() {
		printf("Print Grid:\n");
		for (int i = 0; i < GRID_SIZE; ++i) {
			for (int j = 0; j < GRID_SIZE; ++j) {
				printf("%f ", grid[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	RT_FUNCTION __host__ void printPrefixSum() {
		printf("Print PrefixSum:\n");
		for (int i = 0; i < GRID_SIZE; ++i) {
			for (int j = 0; j < GRID_SIZE; ++j) {
				printf("%f ", prefixSum[i * GRID_SIZE + j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	RT_FUNCTION __host__ void printPdf() {
		printf("Print pdf:\n");
		for (int i = 0; i < GRID_SIZE; ++i) {
			for (int j = 0; j < GRID_SIZE; ++j) {
				printf("%f ", pdf(make_float2(((float)i) / GRID_SIZE + 1e-3, ((float)j) / GRID_SIZE + 1e-3)));
			}
			printf("\n");
		}
		printf("\n");
	}

	RT_FUNCTION __host__ void checkSample() {

		int sample_grid[GRID_SIZE][GRID_SIZE];
		memset(sample_grid, 0, sizeof(sample_grid));
		int sampleNum = 10000000;
		unsigned int seed = 114514;
		float2 s;
		for (int i = 0; i < sampleNum; ++i) {
			//printf("sample %d\n", i);
			sample(seed, s);
			int u = s.x * GRID_SIZE;
			int v = s.y * GRID_SIZE;
			sample_grid[u][v] += 1;
		}
		printf("Check sample:\n");
		for (int i = 0; i < GRID_SIZE; ++i) {
			for (int j = 0; j < GRID_SIZE; ++j) {
				printf("%f ", GRID_SIZE* GRID_SIZE*((float)sample_grid[i][j]) / sampleNum);
			}
			printf("\n");
		}
		printf("\n");
	}

	RT_FUNCTION __host__ void checkUV(int u, int v) {
		if (u >= GRID_SIZE || v >= GRID_SIZE) {
			printf("Error in trainingParams -- u,v should be smaller than GRID_SIZE\nu: %d\t\tv: %d\t\tgrid: %d\n", u, v, GRID_SIZE);
			exit(0);
		}
	}

	RT_FUNCTION __host__ void checkGrid() {
		if (prefixSum[GRID_SIZE * GRID_SIZE-1] <1e-9) {
			printf("Error in trainingParams -- prefixSum[%d] is %f. which is too small\n", GRID_SIZE * GRID_SIZE - 1, prefixSum[GRID_SIZE * GRID_SIZE - 1]);
			exit(0);
		}
	}

	RT_FUNCTION __host__ void train(vector<float3> input)
	{
		for (auto i : input) {
			int u = i.x * GRID_SIZE;
			int v = i.y * GRID_SIZE;
			checkUV(u, v);
			grid[u][v] += i.z;
		}
		updatePrefixSum();
	}

	RT_FUNCTION __host__ void train(float3 input)
	{
		int u = input.x * GRID_SIZE;
		int v = input.y * GRID_SIZE;
		checkUV(u, v);
		grid[u][v] += input.z;
		updatePrefixSum();
	}

	RT_FUNCTION __host__ void updatePrefixSum()
	{
		prefixSum[0] = grid[0][0];
		for (int i = 1; i < GRID_SIZE * GRID_SIZE; ++i)
			prefixSum[i] = prefixSum[i - 1] + grid[i / GRID_SIZE][i % GRID_SIZE];
	}

	RT_FUNCTION __host__ void sample(unsigned int& seed, float2& s)
	{
		float a = rnd(seed);
		checkGrid();
		float sum = prefixSum[GRID_SIZE * GRID_SIZE - 1];
		int l = 0, r = GRID_SIZE * GRID_SIZE - 1, mid = (l+r)/2;
		//printf("before \n");
		while (l != r) {
			//printf("l %d\t\tr %d\n", l, r);
			if (a >= (prefixSum[mid] / sum)) {
				l = mid+1;
			}
			else {
				r = mid;
			}
			mid = (l + r) / 2;
		}

		//printf("after \n");
		int u = l / GRID_SIZE;
		int v = l % GRID_SIZE;
		float grid_step = 1.0f / GRID_SIZE;

		float u1 = rnd(seed);
		float v1 = rnd(seed);

		s.x = ((float)u) / GRID_SIZE + u1 * grid_step;
		s.y = ((float)v) / GRID_SIZE + v1 * grid_step;
	}

	RT_FUNCTION __host__ float pdf(float2 sample)
	{
		int u = sample.x * GRID_SIZE;
		int v = sample.y * GRID_SIZE;
		checkUV(u, v);
		checkGrid();
		float sum = prefixSum[GRID_SIZE * GRID_SIZE - 1];
		return GRID_SIZE * GRID_SIZE * grid[u][v] / sum;
	}

};