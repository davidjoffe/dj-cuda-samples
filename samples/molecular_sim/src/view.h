// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

// View setting like user zoom .. conceptually probably the renderer shouldn't 'know about this' (or maybe - debatable)
// though the renderer must apply settings like zoom ... design-wise something (controller or app) should pass the relevant
// settings to the visualization/renderer.
#pragma once

struct View
{
    float zoom = 1.0f;//10.f;
};
//User view settings like zoom
extern View g_view;
