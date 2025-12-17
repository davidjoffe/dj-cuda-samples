// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
#pragma once

// float4 = x,y,z,w cuda type
// conceptually our renderer should not have cuda-specific stuff in tho ...
// it's a simple struct like the below though
#include <vector_types.h>
/*
struct __device_builtin__ __builtin_align__(16) float4
{
    float x, y, z, w;
};
*/

void djVisualsInit();
void djVisualsInitOnceoff(const int N);
void djVisualsDraw(float4* h_pos, float *h_x, float* h_y, float* h_z, float* radius, const int N,
    float zoom);
 