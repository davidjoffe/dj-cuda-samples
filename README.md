# dj CUDA Samples

A growing collection of small CUDA cross-platform learning samples by David Joffe, starting with a simple GPU-accelerated bouncing-balls demo using CUDA + OpenGL via GLFW.

[CUDA Sample(s)/demo by David Joffe](https://github.com/davidjoffe/dj-cuda-samples)

## Samples:

* Simple bouncing ball sample/demo

Platforms: Windows; Linux/WSL

## Requirements

Requires NVIDIA CUDA toolkit installed (e.g. via winget or NVIDIA installer).

To install via winget, use ```winget install Nvidia.CUDA```

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

## Docker Build

Note that the Docker build currently has no User Interface (unless perhaps via exporting X11 DISPLAY)

```docker build -t dj-cuda-sample1:local -f .\bouncing-balls\docker\Dockerfile .```

```docker run --gpus all --runtime=nvidia dj-cuda-sample1:local```
