// dj CUDA sample
// Demo program for bouncing balls simulation
//
// Created dj2025-11
//
// https://github.com/davidjoffe/dj-cuda-samples
// Copyright David Joffe 2025

#include <iostream>
#include <cuda_runtime.h>
#include "defs.h"
#include "stats.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>//sleep
#include <cstdlib>
#include <string.h>//memset

// Stats like number of frames drawn, total time, updates called etc.
Stats g_stats;

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
    // Temporary CPU arrays for init, then copy all at once to GPU (since CPU <-> GPU copies are relatively slow and latency-bound, better to do fewer large copies than many small ones)
    StructOfArrays_Balls h_init;
    h_init.radius = new float[n];
    h_init.x = new float[n];
    h_init.y = new float[n];
    h_init.z = new float[n];
    h_init.vx = new float[n];
    h_init.vy = new float[n];
    h_init.vz = new float[n];
    for (int i=0; i<n; ++i) {
        float h_radius = 0.1f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.2f)));
        float h_x = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_y = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_z = -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0f)));
        float h_vx = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        float h_vy = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        float h_vz = -0.01f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.02f)));
        h_init.radius[i] = h_radius;
        h_init.x[i] = h_x;
        h_init.y[i] = h_y;
        h_init.z[i] = h_z;
        h_init.vx[i] = h_vx;
        h_init.vy[i] = h_vy;
        h_init.vz[i] = h_vz;
    }
    // Several large copies instead of many small ones - should be faster than before for large N
    // This is especially true because each copy is CPU <-> GPU hence latency-bound (relatively)
    cudaMemcpy(radius, h_init.radius, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x, h_init.x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_init.y, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(z, h_init.z, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vx, h_init.vx, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vy, h_init.vy, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vy, h_init.vz, n*sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_init.x;
    delete[] h_init.y;
    delete[] h_init.z;
    delete[] h_init.vx;
    delete[] h_init.vy;
    delete[] h_init.vz;
    delete[] h_init.radius;

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


int main(int argc, char** argv) {
    std::cout << "===========================================" << std::endl;
    std::cout << "dj CUDA sample" << std::endl;
    std::cout << "Keys:" << std::endl;
    std::cout << "    P     Pause/Unpause" << std::endl;
    std::cout << "    ESC   Exit" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Command line options:" << std::endl;
    std::cout << "   --paused   Start paused" << std::endl;
    std::cout << "   -N / --n   Number of particles/entities" << std::endl;
    std::cout << "===========================================" << std::endl;

    bool paused = false;
    //const int NUMBALLS = 10000;
    //const int NUMBALLS = 100000;
    //const int N = 1024;
    //int N = 10000;//1024;//NUMBALLS;
    int N = 20000;
    //bool fullsreen=false;
    int w = 800;//1920
    int h = 600;//1080

    // PARSE COMMAND LINE ARGUMENTS the standard old way
    // See comments at https://x.com/d_joffe/status/1997001768384057815
    for (int i=1; i<argc; ++i) {
        //if (std::string(argv[i]) == "--no-gfx")//future? headless ..
        //    visual = false;
        // todo - future (low prio) some combined system to dual-handle these as either say command line args, or say user settings to load/save
        if (std::string(argv[i]) == "--paused")
            paused = true;
        else if (
            (std::string(argv[i]) == "-N" || std::string(argv[i]) == "--n")
            && i+1<argc) {
            N = std::atoi(argv[i+1]);
            ++i; // skip next as we've used it as parameter
        }
        //else // maybe later
            //N = std::atoi(argv[i]);
    }
    
    // Display settings before we start
    if (paused) std::cout << "Starting paused" << std::endl;
    std::cout << "Particles: N=" << N << std::endl;

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
    std::string title = "dj CUDA Sample - Bouncing Balls";
    title = title + " - ";
    title = title + std::to_string(N) + " particles"; // not quite sure about word 'particles' ... these may be more than just particles ...
    GLFWwindow* window = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);
//    GLFWwindow* window = glfwCreateWindow(1920, 1080, title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // The below is actually specific to GL-based visualization(s) ...
    // if we only had, say, an ncurses-based visualization, we wouldn't need OpenGL context at all ... (low/future)
    // For now it's fine.

    // Load OpenGL functions using GLAD (after context creation)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to load OpenGL! gladLoadGLLoader" << std::endl;
        return -1;
    }

    // Enable VSYNC (0 for unlimited)
    // dj2025-11 note - on WSL Linux this seems to have no effect currently, so we get uncapped framerates (other than our sleep below)
    glfwSwapInterval(1);


    std::cout << "Initializing demo data..." << std::endl;

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
    h_balls.init(N);
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

    // Initialize visualization(s)
    djVisualsInit();

    // Copy positions etc. to CPU for visualization
    float* h_x = new float[N];
    float* h_y = new float[N];
    memset(h_x, 0, N*sizeof(float));
    memset(h_y, 0, N*sizeof(float));

    // (2) Main loop
    //bool paused = startpaused; // User may optionally 'start paused' etc.
    std::cout << "Starting main loop. Close window or press ESC to exit." << std::endl;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        // Close on ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        // P for pause
        // Detect 'edge' of keypress (but react on down for fastest response, but we don't want to repeat pause/unpause rapidly while key is held down)
        int keystate = glfwGetKey(window, GLFW_KEY_P);
        static int keystate_last = GLFW_RELEASE;
        if (keystate == GLFW_PRESS && keystate_last != GLFW_PRESS) {
            paused = !paused;
            if (paused)
                std::cout << "Pause" << std::endl;
            else
                std::cout << "Unpause" << std::endl;
        }
        keystate_last = keystate;

        // Calculate delta-time (time passed since last frame) for updates
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        // Run GPU kernel parallel update function
        if (!paused)
            djDoUpdate(d_balls, dt, N);//0.016f);

        // Copy positions etc. from GPU to CPU for visualization
        cudaMemcpy(h_x, h_balls.x, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, h_balls.y, N*sizeof(float), cudaMemcpyDeviceToHost);

        // Clear
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw here
        // Temporarily keep old drawing code commented out for now for testing ... to make sure new one is correct
        //djVisualsDrawOld(h_x, h_y, nullptr, nullptr, N);
        djVisualsDraw(h_x, h_y, nullptr, nullptr, N);

        glfwSwapBuffers(window);

        // Prevent CPU hogging (slightly fudgy/simplistic) with a sleep of 1 millisecond.
        // This helps avoid pegging the CPU at 100% usage in this simple demo app, which can use too much power and make it run hot, since we don't need a thousand frames per second for a simple sample ...
        //
        // NB this is very carefully and deliberately placed after the 'swap' but before 'poll events',
        // since if (say) the user presses a key during sleep, we want to process the user's input as quickly as possible before drawing the next frame (if this were a game) for fastest possible response to user input.
        // If we put the sleep after pollEvents, there would be an unnecessary delay between user input and processing it as it would render an extra frame first and only then process the event.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // STATS
        ++g_stats.updateCount;
        if (!paused)
        {
            ++g_stats.frameCount; // "++foo" may in some cases optimize better than "foo++"
            g_stats.frameTimeTotal += dt;
        }
        //stats.fps = 1.0f / dt;

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

    // STATS/INFO
    std::cout << "Particles: " << N << std::endl;
    // total time
    std::cout << "Total time: " << g_stats.frameTimeTotal;
    if (g_stats.frameTimeTotal > 0.0f) // <- prevent divide by 0
    {
        std::cout << ", average FPS (Frames per Second): ";
        g_stats.averageFPS = (g_stats.frameCount / g_stats.frameTimeTotal);
        std::cout << g_stats.averageFPS;
    }
    std::cout << std::endl;
    if (g_stats.frameCount > 0)
        std::cout << "Average frame time (ms): " << (g_stats.frameTimeTotal / g_stats.frameCount * 1000.0f) << std::endl;
    std::cout << "Total frames: " << g_stats.frameCount << std::endl;
    std::cout << "Total updates: " << g_stats.updateCount << std::endl;

    // Set to null even though we're 'about to exit' is a good habit just in case someone later tries to add code below dereferencing these pointers
    delete[] h_x;
    h_x = nullptr;
    delete[] h_y;
    h_y = nullptr;

    std::cout << "Exiting demo." << std::endl;
    return 0;
}
