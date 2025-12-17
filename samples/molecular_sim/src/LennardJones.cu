// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

#include "LennardJones.h"

__global__
void compute_forces_lj_coulomb(
    int    N,
    const float4* __restrict__ pos,   // (x,y,z,q)
    float4*       __restrict__ force, // (fx,fy,fz,_)
    float epsilon,
    float sigma,
    float k_electric,
    float cutoff2
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi4 = pos[i];
    float3 pi  = make_float3(pi4.x, pi4.y, pi4.z);
    //float  qi  = pi4.w;

    float3 fi  = make_float3(0.0f, 0.0f, 0.0f);


    for (int j = 0; j < N; ++j)
    {
        if (j == i) continue;

        float4 pj4 = pos[j];
        float3 pj  = make_float3(pj4.x, pj4.y, pj4.z);
        //float  qj  = pj4.w;

        float3 rij;
        rij.x = pj.x - pi.x;
        rij.y = pj.y - pi.y;
        rij.z = pj.z - pi.z;

        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 > cutoff2 || r2 == 0.0f) continue;
        // debug safety: prevent singularity .. for stability
        r2 = fmaxf(r2, 0.8f * 0.8f);

        // inverse powers
        float inv_r2 = 1.0f / r2;
        float sr2    = (sigma * sigma) * inv_r2;
        float sr6    = sr2 * sr2 * sr2;
        float sr12   = sr6 * sr6;

        float f_over_r = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6);

        fi.x += f_over_r * rij.x;
        fi.y += f_over_r * rij.y;
        fi.z += f_over_r * rij.z;
    }

    force[i] = make_float4(fi.x, fi.y, fi.z, 0.0f);
}
