// dj CUDA sample
// Demo program for bouncing balls simulation
// dj2025-11
// https://github.com/davidjoffe/dj-cuda-samples
// Copyright David Joffe 2025

#include <iostream>
#include <cuda_runtime.h>
#include "defs.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdlib>

void StructOfArrays_Balls::init(int n)
{
    std::cout << "dj:Initializing " << n << " demo data in GPU VRAM." << std::endl;
    count = n;
    // Keep in mind this part runs on the host (CPU)
    // Array buffer size in bytes ...
    const int nARRSIZE=sizeof(float) * n;
    // The data should be in VRAM accessible to the GPU
    cudaMalloc(&x, nARRSIZE); 
    cudaMalloc(&y, nARRSIZE);
    cudaMalloc(&z, nARRSIZE);
    cudaMalloc(&vx, nARRSIZE);
    cudaMalloc(&vy, nARRSIZE);
    cudaMalloc(&vz, nARRSIZE);
    cudaMalloc(&radius, nARRSIZE);

    cudaMemset(x,0,nARRSIZE);
    cudaMemset(y,0,nARRSIZE);
    cudaMemset(z,0,nARRSIZE);
    cudaMemset(vx,0,nARRSIZE);
    cudaMemset(vy,0,nARRSIZE);
    cudaMemset(vz,0,nARRSIZE);
    cudaMemset(radius,0,nARRSIZE);

    std::cout << "dj:Randomizing initial ball positions and velocities." << std::endl;
    // Randomize position, radius etc. here
    for (int i=0; i<n; i++) {
        float h_radius = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.2f)));
        float h_x = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_y = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_z = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_vx = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        float h_vy = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        float h_vz = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        // todo this has too many small copies, can be more optimal
        cudaMemcpy(&radius[i], &h_radius, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&x[i], &h_x, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&y[i], &h_y, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&z[i], &h_z, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&vx[i], &h_vx, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&vy[i], &h_vy, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&vz[i], &h_vz, sizeof(float), cudaMemcpyHostToDevice);
    }
    std::cout << "dj:Initialization complete." << std::endl;
}

void StructOfArrays_Balls::cleanup()
{
    std::cout << "dj:Cleaning up balls data from GPU VRAM." << std::endl;
    if (x!=nullptr) cudaFree(x);
    if (y!=nullptr) cudaFree(y);
    if (z!=nullptr) cudaFree(z);
    if (vx!=nullptr) cudaFree(vx);
    if (vy!=nullptr) cudaFree(vy);
    if (vz!=nullptr) cudaFree(vz);
    if (radius!=nullptr) cudaFree(radius);
    // set to nullptr after free to avoid dangling pointers
    std::cout << "dj:Cleanup cudaFree complete." << std::endl;
    x = nullptr;
    y = nullptr;
    z = nullptr;
    vx = nullptr;
    vy = nullptr;
    vz = nullptr;
    radius = nullptr;
    count = 0;
}


int main() {
    std::cout << "dj CUDA sample" << std::endl;

    // (1) INIT

    // Init GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // Optional: Request OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    std::cout << "Creating window..." << std::endl;
    GLFWwindow* window = glfwCreateWindow(800, 600, "dj CUDA Sample - Bouncing Balls", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Enable VSYNC (0 for unlimited)
    glfwSwapInterval(1);


    std::cout << "Initializing demo data..." << std::endl;
    const int NUMBALLS = 1024;

    // GPU VRAM top-level data instance of array of structs
    // NB this must live in GPU memory also if we pass it to the kernel update as a pointer, which we do
    // (Otherwise it would be a memory address in CPU space, which GPU can't dereference, even though the struct's members are in GPU)
    StructOfArrays_Balls* d_balls = nullptr;
    std::cout << "dj:Allocating StructOfArrays_Balls in GPU VRAM." << std::endl;
    cudaMalloc(&d_balls, sizeof(StructOfArrays_Balls));

    // Temporary CPU instance to init data before copying to GPU
    StructOfArrays_Balls h_balls;

    // Initialize start positions, radius data etc.
    // Init on CPU then copy initial state to GPU mem to start
    h_balls.init(NUMBALLS);
    // Do shallow copy of struct to GPU.
    // NB we can't just do "*d_balls = h_balls;" as we are used to doing in C/C++!
    // as that would mean dereferencig a GPU pointer here from CPU code, which causes a crash.
    // We must use cudaMemcpy to copy, with cudaMemcpyHostToDevice
    // *** Don't do: *d_balls = h_balls;
    // This final memcpy, while intuitively we might guess it copies a lot (all the ball data) in fact it copies only a tiny amount of data,
    // just the device GPU pointers to the arrays like 'float* x' etc. and a few other basics like the count.
    // The arrays are already in GPU memory.
    cudaMemcpy(d_balls, &h_balls, sizeof(StructOfArrays_Balls),
        cudaMemcpyHostToDevice);


    
    // (2) Main loop
    std::cout << "Starting main loop. Close window or press ESC to exit." << std::endl;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        // Close on ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Calculate delta-time (time passed since last frame) for updates
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        // Run GPU kernel parallel update function
        djDoUpdate(d_balls, dt);//0.016f);

        // Clear
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // TODO: draw balls here (OpenGL or CUDA→GL interop)

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // (3) Cleanup
    // This is a little dicey: Remember, h_balls lives in CPU memory but not only cleans the GPU 
    // memory pointers, but also indirectly (as they are shallow copies of the same pointers) the same array pointers d_balls has!
    // If we are not careful it's easy to end up with double-delete mistake in situations like this ...
    // d_balls pointers will dangle the moment after doing this, and we don't and should not also "cleanup()" on d_balls
    // but we do need to cudaFree d_balls instance itself.
    h_balls.cleanup();
    if (d_balls!=nullptr) cudaFree(d_balls);
    d_balls = nullptr;//<- sanity safety measure to help avoid dangling pointers (even though we're about to exit this is a good habit that can prevent bugs in some cases)

    glfwDestroyWindow(window);
    glfwTerminate(); 

    return 0;
}
