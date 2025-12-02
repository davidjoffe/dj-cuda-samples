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
struct Stats
{
    // Sum of total dt (delta time) over all frames (total time elapsed for update/draw loop over application runtime)
    float frameTimeTotal = 0.0f;
    // Updates performed
    int updateCount = 0;
    // Frames drawn
    int frameCount = 0;

    // Average Frames Per Second over runtime
    float averageFPS = 0.0f;
};

#endif // _DJ_STATS_H_
