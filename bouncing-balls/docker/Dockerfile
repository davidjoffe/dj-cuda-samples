# dj-cuda-samples
# Dockerfile for building and running the bouncing balls CUDA sample
# but there is no UI really except maybe via DISPLAY forwarding ...
#
# Requires the NVIDIA Container Toolkit to be installed - see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Note also if you run from Docker desktop it may still fail to pick up the GPU
# in which case try run from command line to force GPU support, e.g.:
# docker run --gpus all --runtime=nvidia dj-cuda-sample1:local

    
#FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libglfw3-dev \
    libgl1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxxf86vm-dev \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

#FROM dj-cuda-base

WORKDIR /app

COPY . /app

RUN cmake -S . -B build-docker && cmake --build build-docker -j 8

CMD ["./build-docker/bouncing-balls/djbouncing_balls_demo"]
