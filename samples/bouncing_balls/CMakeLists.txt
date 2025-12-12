# dj-cuda-samples - Bouncing Balls Demo CMakeLists.txt

# Add C for glad GL loader ...
project(djbouncing_balls VERSION 1.0.0 LANGUAGES C CXX CUDA)
add_compile_definitions(DJAPP_VERSION="${PROJECT_VERSION}")

add_executable(djbouncing_balls_demo
    demo.cpp
    visuals_gl.cpp
    kernels.cu
)

target_link_libraries(djbouncing_balls_demo PRIVATE djcuda_core)


# Link core shared code
#target_link_libraries(djbouncing_balls_demo PRIVATE djcuda_core)

# If this demo uses glfw:
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(djbouncing_balls_demo PRIVATE
    glad
    glfw
    OpenGL::GL
)