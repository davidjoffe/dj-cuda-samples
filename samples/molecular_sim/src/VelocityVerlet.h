// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
//
// Verlet integration helpers to apply incoming force -> velocity -> position updates
// This is meant to be included only from .cu files not normal .cpp as it usesx __global__ for nvcc (which for C++ is "non-standard")
#pragma once

#include <cuda_runtime.h>

__global__ void verlet_step1(
    int N,
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float inv_mass,
    float dt
);
__global__ void verlet_step2(
    int N,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float inv_mass,
    float dt
);
