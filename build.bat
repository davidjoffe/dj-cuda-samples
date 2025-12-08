@rem dj CUDA samples
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

rem clean
rd /s /q build

cmake -S . -B build-windows -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cmake --build build-windows

@rem run the built application
@rem Just pass in passed-in command line args to this script, that allows us to call build with different args to pass to the auto run here:
@echo dj:build-windows: Run djbouncing_balls_demo.exe %*
".\build-windows\bouncing-balls\Debug\djbouncing_balls_demo.exe" %*
