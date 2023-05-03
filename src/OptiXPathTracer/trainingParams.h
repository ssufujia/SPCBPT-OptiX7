#pragma once

#include <optix.h>
#include <cstring>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <stdio.h>
#include "rt_function.h"

#define GRID_SIZE 10

struct trainingParams
{
	float grid[GRID_SIZE][GRID_SIZE];
	/* prefixSum[i] = grid[0] + grid[1] + ... + grid[i] */
	float prefixSum[GRID_SIZE * GRID_SIZE];

	RT_FUNCTION __host__ trainingParams() {
		memset(grid, 0, sizeof(grid));
		memset(prefixSum, 0, sizeof(prefixSum));
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

	RT_FUNCTION __host__ void train(float3 input)
	{
		int u = input.x * GRID_SIZE;
		int v = input.y * GRID_SIZE;
		checkUV(u, v);
		grid[u][v] += input.z;
		for (int i = 10 * u + v; i < GRID_SIZE * GRID_SIZE; ++i)
			prefixSum[i] += input.z;
	}

	RT_FUNCTION __host__ void sample(unsigned int& seed, float2& s)
	{
		float a = rnd(seed);
		checkGrid();
		float sum = prefixSum[GRID_SIZE * GRID_SIZE - 1];
		int l = 0, r = GRID_SIZE * GRID_SIZE - 1, mid = (l+r)/2;
		while (l != r) {
			if (a >= prefixSum[mid] / sum) {
				l = mid+10;
			}
			else {
				r = mid;
			}
			mid = (l + r) / 2;
		}

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
		return grid[u][v] / sum;
	}

};