cmake_minimum_required(VERSION 3.18)

project(dj_cuda_samples LANGUAGES C CXX)

# ================================================================
# Global Configuration
# ================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "dj-cmake: CUDA support detected: ${CMAKE_CUDA_COMPILER}")

    enable_language(CUDA)

    # ------------------------------------------------------------
    # CUDA architecture settings (only if CUDA is available)
    # ------------------------------------------------------------
    if (NOT DEFINED CUDA_ARCH)
        # Generate fat binary for multiple GPU architectures:
        # Default: build fat binary for 30xx + 40xx GPUs like RTX 3090 or RTX 4060
        # Adjust if/as needed for your GPU:
        set(CUDA_ARCH "86;89")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
    message(STATUS "dj-cmake: Building CUDA SM architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "dj-cmake: CUDA not found - building CPU/OpenGL-only components")
endif()

include(cmake/warnings.cmake OPTIONAL)

# ============================================================
# GLAD OpenGL loader
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
)

# Add *.cu files **only if CUDA is available**
if (CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CORE_CUDA_SRC src/*.cu)
    list(APPEND CORE_SRC ${CORE_CUDA_SRC})
endif()

add_library(djcuda_core STATIC ${CORE_SRC})

target_include_directories(djcuda_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# CUDA-specific properties only if CUDA exists
if (CMAKE_CUDA_COMPILER)
    target_include_directories(djcuda_core PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    set_target_properties(djcuda_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# ================================================================
# Subsamples (platform independent ones)
# ================================================================
add_subdirectory(samples/template_minimal)

# ================================================================
# CUDA samples (only if CUDA compiler exists)
# ================================================================
if (CMAKE_CUDA_COMPILER)
    add_subdirectory(samples/bouncing_balls)
else()
    message(STATUS "dj-cmake: Skipping CUDA projects (CUDA unavailable)")
endif()
