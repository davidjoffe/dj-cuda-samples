# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software — Business Source License (BSL 1.1). See LICENSE

# Need C for glad GL loader ...
project(djmolecular_sim VERSION 0.1.0 LANGUAGES C CXX CUDA)
add_compile_definitions(DJAPP_VERSION_molecular_sim="${PROJECT_VERSION}")

file(GLOB_RECURSE CORE_SRC
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(djmolecular_sim
    ${CORE_SRC}
)


target_link_libraries(djmolecular_sim PRIVATE djcuda_core)


# Link core shared code
#target_link_libraries(djbouncing_balls_demo PRIVATE djcuda_core)

# If this demo uses glfw:
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(djmolecular_sim PRIVATE
    glad
    glfw
    OpenGL::GL
)