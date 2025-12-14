// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

__global__
void verlet_step1(
    int N,
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float inv_mass,
    float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 v = make_float3(vel[i].x,   vel[i].y,   vel[i].z);
    float3 f = make_float3(force[i].x, force[i].y, force[i].z);
    float3 p = make_float3(pos[i].x,   pos[i].y,   pos[i].z);

    // v(t + dt/2)
    v.x += 0.5f * dt * inv_mass * f.x;
    v.y += 0.5f * dt * inv_mass * f.y;
    v.z += 0.5f * dt * inv_mass * f.z;

    // x(t + dt)
    p.x += dt * v.x;
    p.y += dt * v.y;
    p.z += dt * v.z;

    vel[i] = make_float4(v.x, v.y, v.z, 0.0f);
    pos[i] = make_float4(p.x, p.y, p.z, pos[i].w);
}

__global__
void verlet_step2(
    int N,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float inv_mass,
    float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

//    float3 v = make_float3(vel[i]);
//    float3 f = make_float3(force[i]);
    float3 v = make_float3(vel[i].x,   vel[i].y,   vel[i].z);
    float3 f = make_float3(force[i].x, force[i].y, force[i].z);

    // v(t + dt)
    v.x += 0.5f * dt * inv_mass * f.x;
    v.y += 0.5f * dt * inv_mass * f.y;
    v.z += 0.5f * dt * inv_mass * f.z;

    vel[i] = make_float4(v.x, v.y, v.z, 0.0f);
}

