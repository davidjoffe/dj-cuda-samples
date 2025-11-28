#!/bin/sh
# dj-cuda-samples
# Build helper script: Linux, WSL etc.

# Clean previous build if any
rm -rf build-linux

# Build
cmake -S . -B build-linux
cmake --build build-linux

# Run
./build-linux/bouncing-balls/djbouncing_balls_demo
