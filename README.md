# dj CUDA Samples

![CUDA Bouncing Balls Demo](media/dj_cuda_bouncing_balls_demo.png)

A growing collection of small CUDA cross-platform learning samples by David Joffe, starting with a simple GPU-accelerated bouncing-balls demo using CUDA + OpenGL via GLFW.

[CUDA Sample(s)/demo by David Joffe](https://github.com/davidjoffe/dj-cuda-samples)

## Samples:

* Simple bouncing ball sample/demo

Platforms: Windows; Linux/WSL

## Requirements

* Requires NVIDIA CUDA toolkit installed (e.g. via winget or NVIDIA installer).

To install via winget, use ```winget install Nvidia.CUDA```

* CMake 3.2
* OpenGL + GLFW3 dev packages
* A modern NVIDIA GPU with CUDA support

## Build

First install dependencies, as per below instructions.

Then to build on Windows, configure and run ```.\build.bat``` build helper, or:

```
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=c:/your-vcpkg-folder/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

To build on Linux, either run ```./build.sh```, or:

```
cmake -S . -B build-linux
cmake --build build-linux
```

## Run (Windows)

```
.\build\bouncing-balls\Debug\djbouncing_balls_demo.exe
```

## Run (Linux)

```
./build-linux/bouncing-balls/djbouncing_balls_demo
```

## Help

### **Keys**

```
    P     Pause/Unpause
    ESC   Exit
```

### **Command-line Options**

```
   --paused   Start paused
   -N / --n   Number of particles/entities
```

### Installing Dependencies:

This uses glfw to create a window with OpenGL context.

#### Windows

* glfw3

Windows: Install glfw3 with vcpkg:

```
vcpkg install glfw3:x64-windows
```

Make sure your vcpkg installation is integrated: ```vcpkg integrate install```

#### Linux, WSL etc.:

```
sudo apt install libglfw3-dev
```

## Troubleshooting

If you get a build error about glfw3 cmake config not found, first try set your VCPKG folder in build.bat. Alternatively pass " -DCMAKE_TOOLCHAIN_FILE=" to cmake. 

For Windows, you should build from a Developer Command Prompt for VS.

## Docker (Optional)

Note Docker support is completely **optional**.

It provides an alternative, additional way to build and run the CUDA samples — useful for deployment, reproducible builds, or testing.

## Docker Build and Run

The provided Dockerfile builds the CUDA sample inside a GPU-enabled container. It currently has no User Interface unless you forward X11 or VirtualGL.

It requires the NVIDIA Container Toolkit to be installed - see the NVIDIA guide for installing this.

Note that if you have installed the NVIDIA Container Toolkit but still get a warning about the driver failing to load when you run the Docker version in Docker Desktop, try run from command line as per below to force GPU support via command line:


```docker build -t dj-cuda-sample1:local -f .\bouncing-balls\docker\Dockerfile .```

```docker run --gpus all --runtime=nvidia dj-cuda-sample1:local```
