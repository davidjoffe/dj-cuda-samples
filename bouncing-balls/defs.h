// dj-cuda-samples
// https://github.com/davidjoffe/dj-cuda-samples
// Copyright David Joffe 2025
#pragma once
#ifndef _DJ_BOUNCING_BALLS_DEFS_H_
#define _DJ_BOUNCING_BALLS_DEFS_H_

// This array of structs may seem inefficient (or rather, not "traditional" type of data structuring), but it works well with CUDA coalesced memory access
// i.e. it is much faster if all threads in a warp access contiguous memory locations
// "Traditionally" data like this would be stored as an array of structs (each with say x,y,z), which is less efficient for CUDA
struct StructOfArrays_Balls
{
    // array of x positions and so on ...
    float* x=nullptr;
    float* y=nullptr;
    float* z=nullptr;
    float* vx=nullptr;
    float* vy=nullptr;
    float* vz=nullptr;
    float* radius=nullptr;
    int    count=0;

    // host/CPU functions
    void init(int n);
    // Note we don't want a destructor here that would automatically cleanup() as that would mean temporaries like 'StructOfArrays_Balls h_balls' would cause cudaFree calls when going out of scope (of the live actual d_balls in GPU memory alloc'd with cudaAlloc etc.)
    void cleanup();
};

// NB d_data is device (GPU) pointer. dt is delta time
extern void djDoUpdate(StructOfArrays_Balls* d_data, float dt);

// future: I think there should be a base class like abstract OOP 'Visualization' with virtual methods for init, draw etc.
// Then we should instantiate and generically allocate/init/draw zero or more instances .. so could have multiple running visualizations
// An ncurses-based visualization for, say, Docker could then also easily be an option ...

extern void djVisualsInit();
extern void djVisualsDraw(float *h_x, float* h_y, float* h_z, float* radius, int N);

#endif // _DJ_BOUNCING_BALLS_DEFS_H_