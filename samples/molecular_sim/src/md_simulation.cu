// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software — Business Source License (BSL 1.1). See LICENSE

// Lennard-Jones

#include "VelocityVerlet.h"
#include "LennardJones.h"
#include "stats.h"

void djCUDA_GPU_InitStep(int    N,
    float4* __restrict__ pos,   // (x,y,z,q)
    float4*       __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2
)
{
    int block = 256;
    int grid = (N + block - 1) / block;

    cudaMemset(force, 0, N * sizeof(float4));
    compute_forces_lj_coulomb<<<grid, block>>>(
        N, pos, force,
        epsilon, sigma, k_electric, cutoff2
    );
}

// This function is intended to be called hundreds of thousands of times per second or more 'as fast as possible' so it must be kept as streamlined/fast/optimized as possible!
void djCUDA_GPU_Update(int N,
    float4* __restrict__ pos,   // (x,y,z,q)
    float4* __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2,
    float dt,
    float4* __restrict__ vel,
    float inv_mass
)
{
    // Sanity check
//    if (d_data == nullptr) {
//        std::cerr << "djDoUpdate: Error: d_data is null!" << std::endl;
//        return;
 //   }

    int block = 256;
    int grid = (N + block - 1) / block;

    // launch kernel threads
    // we don't necessarily want this much logging unless debugging
    // but leaving it here for nows
    // can comment out later if too verbose or add options like "-vv" etc. (low priority for small sample/demo)
    //std::cout << "dj:Run GPU update<<<" << blocks << " blocks of " << threadsPerBlock << " threads>>>" << std::endl;

    verlet_step1<<<grid, block>>>(N, pos, vel, force, inv_mass, dt);

    cudaMemset(force, 0, N * sizeof(float4));
    // Note that this does use dt! That's correct .. it just calculates instantaneous forces. The integration step applies them to make things move.
    compute_forces_lj_coulomb<<<grid, block>>>(
        N, pos, force,
        epsilon, sigma, k_electric, cutoff2
    );

    verlet_step2<<<grid, block>>>(N, vel, force, inv_mass, dt);

    ++g_stats.GPUupdates;
}
