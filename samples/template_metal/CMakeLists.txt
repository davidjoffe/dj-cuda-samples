# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software — Business Source License (BSL 1.1). See LICENSE

cmake_minimum_required(VERSION 3.20)
project(djtemplate_metal LANGUAGES OBJCXX)
#project(djtemplate_metal VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Metal/Foundation frameworks (macOS only)
find_library(FRAMEWORK_METAL Metal REQUIRED)
find_library(FRAMEWORK_FOUNDATION Foundation REQUIRED)

add_executable(djtemplate_metal
    src/main.mm
)

target_link_libraries(djtemplate_metal
    PRIVATE
    ${FRAMEWORK_METAL}
    ${FRAMEWORK_FOUNDATION}
)

# Helpful warnings
target_compile_options(djtemplate_metal PRIVATE
    -Wall -Wextra -Wpedantic
)
