#!/bin/sh
# dj-cuda-samples
# Build helper script: Linux, WSL etc.
#
# Usage: ./build.sh
#
# Optionally you may also pass in arguments to this build script which will be simply passed on to the built app when it is run.
# Examples:
# ./build.sh -N 50000
# ./build.sh --paused
# ./build.sh -N 1000000 --headless --maxframes 1000

# Clean previous build if any
rm -rf build-linux

# Build
cmake -S . -B build-linux
cmake --build build-linux

# Run
# Just pass in passed-in command line args to this script, that allows us to call build with different args to pass to the auto run here:
#./build-linux/bouncing-balls/djbouncing_balls_demo --paused -N 200000  --headless --maxframes 10000
@echo dj:build-linux: Run app $@
./build-linux/bouncing-balls/djbouncing_balls_demo $@
