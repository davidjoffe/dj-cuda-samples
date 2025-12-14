
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
    float  qi  = pi4.w;

    float3 fi  = make_float3(0.0f, 0.0f, 0.0f);

    float sig2 = sigma * sigma;
    float sig6 = sig2 * sig2 * sig2;

    for (int j = 0; j < N; ++j)
    {
        if (j == i) continue;

        float4 pj4 = pos[j];
        float3 pj  = make_float3(pj4.x, pj4.y, pj4.z);
        float  qj  = pj4.w;

        float3 rij;
        rij.x = pj.x - pi.x;
        rij.y = pj.y - pi.y;
        rij.z = pj.z - pi.z;

        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 > cutoff2 || r2 == 0.0f) continue;

        // inverse powers
        float inv_r2  = 1.0f / r2;
        float inv_r   = rsqrtf(r2);           // 1/r
        float inv_r6  = inv_r2 * inv_r2 * inv_r2;

        // Lennard-Jones: F_LJ = 24 ε [2 (σ^12 / r^13) − (σ^6 / r^7)] r̂
        float sig12 = sig6 * sig6;
        float lj_scalar = 24.0f * epsilon * inv_r2 * inv_r6 * (2.0f * sig12 * inv_r6 - sig6);

        // Coulomb: F_C = k q_i q_j / r^2 * r̂
        float coulomb_scalar = k_electric * qi * qj * inv_r2;

        float scalar = lj_scalar + coulomb_scalar;

        fi.x += scalar * rij.x;
        fi.y += scalar * rij.y;
        fi.z += scalar * rij.z;
    }

    force[i] = make_float4(fi.x, fi.y, fi.z, 0.0f);
}
