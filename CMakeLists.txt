cmake_minimum_required(VERSION 3.25)

# Add C for glad GL loader ...
project(dj_cuda_samples LANGUAGES C CXX CUDA)

# ================================================================
# Global Configuration
# ================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default GPU arch = my RTX 4060 laptop (sm_89)
if (NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH 89)
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
message(STATUS "CUDA sm_${CUDA_ARCH}")

include(cmake/warnings.cmake OPTIONAL)

# ============================================================
# GLAD OpenGL loader cmake library
# ============================================================
add_library(glad STATIC
    external/glad/src/glad.c
)

target_include_directories(glad PUBLIC
    external/glad/include
)

# Tell CMake that glad.c is C code
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)

# ============================================================
# Optional small core library (common utils)
# ============================================================

file(GLOB_RECURSE CORE_SRC
    src/*.cpp
    src/*.cu
)

add_library(djcuda_core STATIC ${CORE_SRC})

target_include_directories(djcuda_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    # Below seems to be necessary otherwise docker build fails, unable to find cuda headers - dj2025
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(djcuda_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ================================================================
# Subsamples
# ================================================================
add_subdirectory(bouncing-balls)
