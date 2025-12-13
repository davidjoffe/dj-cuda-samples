#!/bin/sh
# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software — Business Source License (BSL 1.1). See LICENSE
#
# Helper script to try help auto-generate new projects from templates or other samples/projects as starting point
# (This script is not used or important for building or running samples.)
# WIP

# shortname newtarget "New Name" "Old Name"

# WIP - dj2025-12 - some of below are still in 'busy thinking about this' stage ...

./make_from_template.sh template_minimal template_gl "OpenGL Template" "DJ Sample" 0.1.0
./make_from_template.sh template_gl template_gl_cuda "CUDA and OpenGL Template" "OpenGL Template" 0.1.0
./make_from_template.sh template_minimal template_headless "Headless Template" "DJ Sample" 0.1.0
./make_from_template.sh template_minimal template_oneapi "oneAPI Template" "DJ Sample" 0.1.0
./make_from_template.sh template_minimal template_sycl "SYCL Template" "DJ Sample" 0.1.0
./make_from_template.sh template_minimal template_cuda "CUDA Template" "DJ Sample" 0.1.0

./make_from_template.sh template_gl_cuda molecular_sim  "Molecular Sim"   "CUDA and OpenGL Template" 0.1.0
./make_from_template.sh template_gl galaxy_explorer     "Galaxy Explorer" "OpenGL Template" 0.1.0
./make_from_template.sh template_gl_cuda perlin_terrain "Perlin Terrain"  "CUDA and OpenGL Template" 0.1.0
