// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
#pragma once

// CUDA float4 etc.:
#include <vector_types.h>

// Wrapper to call CUDA 
void djCUDA_GPU_InitStep(int    N,
    float4* __restrict__ pos,   // (x,y,z,q)
    float4*       __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2
);

void djCUDA_GPU_Update(int    N,
     float4* __restrict__ pos,   // (x,y,z,q)
    float4*       __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2,
    float dt,
    float4* __restrict__ vel,
    float inv_mass
);
