// dj-cuda-samples
// https://github.com/davidjoffe/dj-cuda-samples
//
// Stats like number of frames drawn, total time, updates called etc.
//
// Copyright David Joffe 2025
#pragma once
#ifndef _DJ_STATS_H_
#define _DJ_STATS_H_

// Stats like number of frames drawn, total time, updates called etc.
// This may be a bit confusing to understand, it's not always straightforward what counts as a 'frame' or an 'update' in different situations
// For example, if running in headless mod
// When running with visuals, the viewer typically expects delta-time to cause the balls to move at realistic looking speed.
// Regarding the idea of headless mode, there are two types of situations to consider:
// 1. Imagine this is something like a game server generating visuals, then again it should consider delta time for realistic looking time passing
// 2. Alternatively, we may just want it to 'run the simulation as fast as you can' and finish, e.g. if this was something like molecular simulation, or we just want to get this task done as quickly as possible for the user.
// In such case, we maybe don't want to sleep, and though we may still want to use some sort of constant delta time for the physics e.g. say 120Hz or 100Hz or 60Hz or whatever we need for simulation fidelity, we call the GPU kernel update as fast as possible with this fixed time-step, so we have a 'virtual' delta time and a real delta time which may be either faster or slower depending how fast the hardware is.
// So we have 'virtual simulation time passed' statistics but in headless mode (and/or if running in cloud container that falls back to headless) the actual human time passed may be faster or slower, depending on hardware speed.
// But with default visuals mode, it's more like a game say, then simulation time passed should be same as human render time passed (sum of delta times for each frame rendered)
// 
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
};

#endif // _DJ_STATS_H_
