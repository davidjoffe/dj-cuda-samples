# dj-cuda-samples - Bouncing Balls Demo CMakeLists.txt
project(djbouncing_balls LANGUAGES CXX CUDA)

add_executable(djbouncing_balls_demo
    demo.cpp
    kernels.cu
)

target_link_libraries(djbouncing_balls_demo PRIVATE djcuda_core)


# Link core shared code
#target_link_libraries(djbouncing_balls_demo PRIVATE djcuda_core)

# If this demo uses glfw:
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(djbouncing_balls_demo PRIVATE
    glfw
    OpenGL::GL
)