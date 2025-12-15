// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
#pragma once

// Stats like number of frames drawn, total time, updates called etc.
struct Stats
{
    // Sum of total dt (delta time) over all frames (total time elapsed for update/draw loop over application runtime)
    float frameTimeTotal = 0.0f;
    // Total 'virtual simulation time' passed
    float virtualTimeTotal = 0.0f;
    // Total actual human time passed (including things like paused time)
    float runtimeTotal = 0.0f;
    // Updates performed
    int updateCount = 0;
    int updateCountAccum = 0;
    // Frames drawn
    int frameCount = 0;

    // Average Frames Per Second over runtime
    float averageFPS = 0.0f;
    float averageUpdatesPerSecond = 0.0f;

    // This may be redundant, check and clean up later and consolidate if so, but for now it helps us test and be sure the main update call is occurring and how many times
    // Note if we also had a Metal backend then 'CUDAupdates' and 'GPUupdates' would be separate different things - should we just have one 'GPU updates' or both? 
    int GPUupdates = 0;
};

extern Stats g_stats;

