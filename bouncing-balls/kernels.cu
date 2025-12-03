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

    const float FLOOR = 0.f;//-0.2f;

    if (balls->y[i]<FLOOR) balls->y[i] = -balls->y[i];
    
    // This is a struct of arrays (SoA) approach
    // So we don't have a single container with x,y,z etc.
    // It's in separate arrays for faster access on GPU multi-threading
    // but a bit more tedious and unintuitive to code
    //Ball b = balls[i]; 
    balls->x[i] += dt * balls->vx[i];
    balls->y[i] += dt * balls->vy[i];
    // add gravity effect in y direction
    balls->vy[i] -= dt * 0.98f; // gravity constant (approx Earth gravity) ... later support other planets?
    balls->z[i] += dt * balls->vz[i];
    // bounce off walls
    if (balls->x[i] < -1.0f || balls->x[i] > 1.0f) balls->vx[i] = -balls->vx[i];

    // The updated behavior below is maybe a bit specific-y and non-conventional
    // but it makes some cool effects (almost water-fountain-display vibe) to make it less boring and less uniform ...
    // (We do some things like make y velocity changes etc. x-position-dependent and so on ...
    // a bit non-norm but looks quite cool, more interesting than plain bouncnig balls)

    // TODO also factor in ball radius here
    if (balls->y[i] <= FLOOR)
    {
        // There are two cases here
        // 1. The ball is still moving fast enough to bounce
        // 2. The ball has slowed down enough to be considered 'at rest' and should be reset to the top
        // If absolute value of velocity is low it's 'at rest'
        if (abs(balls->vy[i]) < 0.001f)
        {
            // if the ball has come to a rest set its y up to 'drop' it .. this should have some randomness
            balls->y[i] = 0.85f; // reset to top
            balls->vy[i] = 0.f; // reset velocity to a 'drop'

            // Add a little of the x position to y *velocity* to make it more interesting to look at so they aren't uniformly dropping the same
            // (NB, we are adding it to the *start velocity* not the start position - so won't immediately see effect until update/move)
            balls->vy[i] += balls->x[i] * 0.1f;
            // so those to left start slightly 'going up' still (as if thrown up slightly), in the middle just plain drop, those to right as if slightly heavier (or thrown with a bit of downward thrust to start)
        }
        else // Not at rest, hitting floor, bounce upwards
        {
            // decay after a bounce on floor like a real bouncing ball
            // Make sure we're moving up - so set to use abs() of velocity (positive velocity means 'moving upwards')
            balls->vy[i] = abs(balls->vy[i]) * 
                (0.8f + balls->x[i] * 0.1f) ; // lose some energy on each bounce
        }

        // The higher the abs(y) it means we hit harder/deeper the floor so we could make more bounce
        // Arguably we could either set this to 0 or -y ...
        balls->y[i] = balls->y[i];
    }
    
    if (balls->z[i] < -1.0f || balls->z[i] > 1.0f) balls->vz[i] = -balls->vz[i];
}

// NB d_data is device (GPU) pointer. dt is delta time
void djDoUpdate(StructOfArrays_Balls* d_data, float dt, int N)
{
    // Sanity check
    if (d_data == nullptr) {
        std::cerr << "djDoUpdate: Error: d_data is null!" << std::endl;
        return;
    }
    //const int NUMBALLS = 1024;
    //const int N = NUMBALLS;

    int threadsPerBlock = 256;//16;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel threads
    // we don't necessarily want this much logging unless debugging
    // but leaving it here for now
    // can comment out later if too verbose or add options like "-vv" etc. (low priority for small sample/demo)
    //std::cout << "dj:Run GPU update<<<" << blocks << " blocks of " << threadsPerBlock << " threads>>>" << std::endl;
    kernelDemoUpdate<<<blocks, threadsPerBlock>>>(d_data, N, dt);

    //cudaDeviceSynchronize();
}
