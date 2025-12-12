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

# stop on build errors
set -e

# Clean previous build if any
rm -rf build-linux
rm -rf build-linux-debug

# Build
# -j speeds up builds by using multiple CPU cores (nproc retrieves your system's number of CPU cores you are building on so -j tells it to use the number of cores you have)

echo "dj-build-linux: Building Release"
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux --config Release -- -j$(nproc)

echo "dj-build-linux: Building Debug"
cmake -S . -B build-linux-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-linux-debug --config Debug -- -j$(nproc)

# Run
# Just pass in passed-in command line args to this script, that allows us to call build with different args to pass to the auto run here:
echo dj-build-linux: Run app $@

# If building on a non-CUDA-supported platform then it may be normal for this to be not found, else it should be there:
if [ ! -f ./build-linux/samples/bouncing_balls/djbouncing_balls_demo ]; then
    echo "djbouncing_balls_demo not found"
else
    ./build-linux/samples/bouncing_balls/djbouncing_balls_demo $@
fi

if [ ! -f ./build-linux/samples/template_minimal/djtemplate_minimal ]; then
    echo "djtemplate_minimal not found"
else
    ./build-linux/samples/template_minimal/djtemplate_minimal $@
fi
