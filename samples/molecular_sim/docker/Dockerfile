# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
#
# Dockerfile for building and running the Molecular Sim CUDA sample
# but there is no UI really except maybe via DISPLAY forwarding ...
#
# Requires the NVIDIA Container Toolkit to be installed - see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Note also if you run from Docker desktop it may still fail to pick up the GPU
# in which case try run from command line to force GPU support, e.g.:
# docker run --gpus all --runtime=nvidia dj-cuda-sample1:local

    
#FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS builder

# Faster apt and smaller layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglfw3-dev libgl1-mesa-dev libx11-dev libxrandr-dev \
    libxi-dev libxxf86vm-dev libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the essential files, not the whole repo
COPY CMakeLists.txt ./
COPY external/ ./external/
COPY src/ ./src/
#COPY samples/molecular_sim/ ./samples/molecular_sim/
# for now copy all though we should probably just build molecular_sim someday ..
COPY samples/ ./samples/

# Build
RUN cmake -S . -B build-docker -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build-docker -j$(nproc)

# ===============================
#   RUNTIME STAGE
# ===============================
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

WORKDIR /

# Only libs required for running GLFW/OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3 \
    libgl1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the final binary to keep the image small
COPY --from=builder /app/build-docker/samples/molecular_sim/djmolecular_sim /usr/local/bin/

CMD ["djmolecular_sim"]
