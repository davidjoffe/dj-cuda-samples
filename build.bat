@rem dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
@rem (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

@rem Build script for Windows using vcpkg and cmake
@rem Adjust the VCPKG variable to point to your vcpkg installation
@rem Usage: build.bat (optional-args-to-pass-to-built-app)

@rem Optionally you may pass in run-time arguments to the built application here, though this is not required:

@rem Examples:

@rem Start paused
@rem build.bat --paused
@rem or with number of items/entities/particles to simulate:
@rem build.bat --paused -N 50000
@rem or for headless mode test, or for benchmark or test runs:
@rem build.bat -N 1000000 --headless --maxframes 10000

@rem https://github.com/davidjoffe/dj-cuda-samples
@rem Copyright David Joffe 2025
@echo off

rem *** NB: SET YOUR VCPKG PATH HERE FOR CMAKE TO FIND CONFIG FILES FOR LIBS LIKE GLFW ***
rem If you get errors about "FindGLFW3.cmake" and CMAKE_MODULE_PATH try set your vcpkg path below
set VCPKG=C:\v
@echo dj-build: VCPKG vcpkg folder setting: %VCPKG%

rem clean
rd /s /q build-windows
rd /s /q build-windows-debug

@echo dj-build: Building dj CUDA samples with vcpkg and cmake
@echo dj-build: Building both debug and release versions

@echo dj-build: Build debug version (slower but include debug info in case we need to debug)
cmake -S . -B build-windows-debug -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cmake --build build-windows-debug

@echo dj-build: Build release version (faster, better for benchmarking and final user runs with more optimal performance)
cmake -S . -B build-windows -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cmake --build build-windows --config Release -j


@rem run the built application
@rem Just pass in passed-in command line args to this script, that allows us to call build with different args to pass to the auto run here:
@echo dj:build: Run djbouncing_balls.exe %*
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe" %*
".\build-windows\samples\template_minimal\Release\djtemplate_minimal.exe" %*

call .\scripts\wrunall.bat %*
