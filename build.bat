@rem dj CUDA samples
@rem Build script for Windows using vcpkg and cmake
@rem Adjust the VCPKG variable to point to your vcpkg installation
@rem Usage: build.bat

@rem https://github.com/davidjoffe/dj-cuda-samples
@rem Copyright David Joffe 2025
@echo off

rem *** NB: SET YOUR VCPKG PATH HERE FOR CMAKE TO FIND CONFIG FILES FOR LIBS LIKE GLFW ***
rem If you get errors about "FindGLFW3.cmake" and CMAKE_MODULE_PATH try set your vcpkg path below
set VCPKG=C:\v

rem clean
rd /s /q build

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cmake --build build

@rem run the built demo
.\build\bouncing-balls\Debug\djbouncing_balls_demo.exe"
