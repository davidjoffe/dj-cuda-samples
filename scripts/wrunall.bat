@rem dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
@rem (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
@rem Helper to test run all built applications (say after build)
@rem Windows .bat version
@rem "w" is just short for "windows runall" to make tab auto-completion easier during testing

@echo off
setlocal enabledelayedexpansion

echo "================================"
echo "dj: wrunall.bat: Run all built applications on Windows, such as after building"

set BUILD_DIR=build-windows\samples

for /d %%D in (samples\*) do (
    set SAMPLE_NAME=%%~nxD
    set EXE_PATH=%BUILD_DIR%\!SAMPLE_NAME!\Release\dj!SAMPLE_NAME!.exe

    if not exist "!EXE_PATH!" (
        echo ================================================
        echo dj: WARNING: EXECUTABLE NOT FOUND
        echo dj: EXPECTED: !EXE_PATH!
        echo ================================================
        echo.
    ) else (
        echo ==== dj: RUNNING !SAMPLE_NAME!...
        "!EXE_PATH!" %*
    )
)

endlocal