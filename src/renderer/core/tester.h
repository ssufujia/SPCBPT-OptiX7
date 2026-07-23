#pragma once
#include "pathControl.h"

#define DEBUG
#define lPath_bSize 32

struct lightSubpathTester {

	int lightSubpath[lPath_bSize];
	RT_FUNCTION __host__ lightSubpathTester() {
		for (int i = 0; ++i; i < lPath_bSize)
			lightSubpath[i] = 0;
	}
	RT_FUNCTION __host__ void reset() {
		for (int i = 0; i < lPath_bSize;++i)
			lightSubpath[i] = 0;
	}
	RT_FUNCTION __host__ void add(int depth, long long pathRecord) {
		if (depth == 1 && pathRecord == 0b1)
			lightSubpath[(int)LightSubpathType::LS] ++;
		else if (depth == 2 && pathRecord == 0b10)
			lightSubpath[(int)LightSubpathType::LDS] ++;
		else if (depth == 2 && pathRecord == 0b11)
			lightSubpath[(int)LightSubpathType::LSS] ++;
		else if (depth == 3 && pathRecord == 0b100)
			lightSubpath[(int)LightSubpathType::LDDS] ++;
		else if (depth == 3 && pathRecord == 0b110)
			lightSubpath[(int)LightSubpathType::LDSS] ++;
		else if (depth == 3 && pathRecord == 0b111)
			lightSubpath[(int)LightSubpathType::LSSS] ++;
		else if (depth == 4 && pathRecord == 0b1000)
			lightSubpath[(int)LightSubpathType::LDDDS] ++;
		else if (depth == 4 && pathRecord == 0b1100)
			lightSubpath[(int)LightSubpathType::LDDSS] ++;
		else if (depth == 4 && pathRecord == 0b1110)
			lightSubpath[(int)LightSubpathType::LDSSS] ++;
		else if (depth == 4 && pathRecord == 0b1001)
			lightSubpath[(int)LightSubpathType::LSDDS] ++;
		else if (depth == 4 && pathRecord == 0b1111)
			lightSubpath[(int)LightSubpathType::LSSSS] ++;
	}
	RT_FUNCTION __host__ void print() {
		printf("LS %d\n", lightSubpath[0]);
		printf("LDS %d\n", lightSubpath[1]);
		printf("LSS %d\n", lightSubpath[2]);
		printf("LDDS %d\n", lightSubpath[3]);
		printf("LDSS %d\n", lightSubpath[4]);
		printf("LSSS %d\n", lightSubpath[5]);
		printf("LDDDS %d\n", lightSubpath[6]);
		printf("LDDSS %d\n", lightSubpath[7]);
		printf("LSDDS %d\n", lightSubpath[8]);
		printf("LDSSS %d\n", lightSubpath[9]);
		printf("LSSSS %d\n", lightSubpath[10]);
	}

};

RT_FUNCTION __host__ void printLightSubpath(int depth, long long pathRecord) {
	if (depth == 1 && pathRecord == 0b1)
		printf("LS\n");
	if (depth == 2 && pathRecord == 0b10)
		printf("LDS\n");
	if (depth == 2 && pathRecord == 0b11)
		printf("LSS\n");
	//if (depth == 3 && pathRecord == 0b100)
	//	printf("LDDS\n");
	/*if (depth == 3 && pathRecord == 0b110)
		printf("LDSS\n");
	if (depth == 3 && pathRecord == 0b111)
		printf("LSSS\n");
	if (depth == 4 && pathRecord == 0b1000)
		printf("LDDDS\n");
	if (depth == 4 && pathRecord == 0b1100)
		printf("LDDSS\n");
	if (depth == 4 && pathRecord == 0b1110)
		printf("LDSSS\n");
	if (depth == 4 && pathRecord == 0b1001)
		printf("LSDDS\n");
	if (depth == 4 && pathRecord == 0b1111)
		printf("LSSSS\n");*/
}

