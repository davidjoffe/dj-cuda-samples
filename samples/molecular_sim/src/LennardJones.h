// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
// ------------------------------------------------------------------------------
// Molecular Sim: Calculate Lennard-Jones forces CUDA GPU kernel function
// ------------------------------------------------------------------------------
// This is meant to be included only from .cu files not normal .cpp as it uses __global__ for nvcc
#pragma once

#include <cuda_runtime.h>

// N: Number of molecules/particles
// NB: pos, force must be GPU device memory pointers (VRAM) e.g. cudaMalloc
__global__
void compute_forces_lj_coulomb(
    int    N,
    const float4* __restrict__ pos,   // (x,y,z,q)
    float4*       __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2
);
