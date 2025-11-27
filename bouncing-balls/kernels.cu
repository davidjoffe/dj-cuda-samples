// dj-cuda-samples
// Simple bouncing balls CUDA demo
// https://github.com/davidjoffe/dj-cuda-samples
// Copyright David Joffe 2025

#include "defs.h"
#include <iostream>

// GPU thread update kernel function
// dt = delta time (time between updates/frames)
__global__ void kernelDemoUpdate(StructOfArrays_Balls* balls, int count, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (balls == nullptr) return;//sanity check
    
    // This is a struct of arrays (SoA) approach
    // So we don't have a single container with x,y,z etc.
    // It's in separate arrays for faster access on GPU multi-threading
    // but a bit more tedious and unintuitive to code
    //Ball b = balls[i]; 
    balls->x[i] += dt * balls->vx[i];
    balls->y[i] += dt * balls->vy[i];
    balls->z[i] += dt * balls->vz[i];
    // bounce off walls
    if (balls->x[i] < -1.0f || balls->x[i] > 1.0f) balls->vx[i] = -balls->vx[i];
    if (balls->y[i] < -1.0f || balls->y[i] > 1.0f) balls->vy[i] = -balls->vy[i];
    if (balls->z[i] < -1.0f || balls->z[i] > 1.0f) balls->vz[i] = -balls->vz[i];
}

// NB d_data is device (GPU) pointer. dt is delta time
void djDoUpdate(StructOfArrays_Balls* d_data, float dt)
{
    // Sanity check
    if (d_data == nullptr) {
        std::cerr << "djDoUpdate: Error: d_data is null!" << std::endl;
        return;
    }
    const int NUMBALLS = 1024;
    const int N = NUMBALLS;

    int threadsPerBlock = 256;//16;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel threads
    // we don't necessarily want this much logging unless debugging
    // but leaving it here for now
    // can comment out later if too verbose or add options like "-vv" etc. (low priority for small sample/demo)
    std::cout << "dj:Run GPU update<<<" << blocks << " blocks of " << threadsPerBlock << " threads>>>" << std::endl;
    kernelDemoUpdate<<<blocks, threadsPerBlock>>>(d_data, N, dt);

    //cudaDeviceSynchronize();
}
