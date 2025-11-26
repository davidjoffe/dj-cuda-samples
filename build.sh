#!/bin/sh
# dj-cuda-samples
# Build helper script: Linux, WSL etc.

# Build
cmake -S . -B build-linux
cmake --build build-linux

# Run
./build-linux/bouncing-balls/djbouncing_balls_demo
