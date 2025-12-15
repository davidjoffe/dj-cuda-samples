// dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
// (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

#include "renderer.h"

#include <glad/glad.h>
#include <string.h>//memset

// Small helper to compile a shader
GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}

// Simple struct to hold variables and data for OpenGL visuals, such as shader handle, vertex arrays, etc.
struct GLContext
{
    // Shader handle
    GLuint prog = 0;

    // todo still cleanup below on exit ... djVisualsCleanup() or something? and/or OOP-y for generic visualations system later ... low prio

    // global vertex buffer array for rendering points
    float* vertexarray = nullptr;
    int N = 0;
    GLuint vao=0, vbo=0;
};
GLContext g_GL;

void djVisualsInit()
{
    // Simple shader setup (no error checking here for brevity)
    // vertex shader (vs), fragment shader (fs)
    const char* vs = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        out vec4 vertexColor; // send color to the fragment shader
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            // Originally we just had white vertices, now we make the color position-dependent which looks nicer
            // later we could maybe add some settings if user wants to make it white again or something
            // Positions are currently between -1 to 1 so make sure normalized in that range
            float x = clamp(aPos.x, -1.0, 1.0);
            float y = clamp(aPos.y, -1.0, 1.0);
            //float z = 0.0; // could use aPos.z if we had it
            vertexColor = vec4(
                0.1 + 0.9 * ((x + 1.0) / 2.0), // <- red
                0.1 + 0.9 * ((y + 1.0) / 2.0), // <- green
                0.1 + 0.9 * ((0 + 1.0) / 2.0), // <- blue
                1.0 // alpha
            );
        }
        )";
    const char* fs = R"(
        #version 330 core
        out vec4 FragColor;
        in vec4 vertexColor; // From vertex shader
        void main() {
            //FragColor = vec4(1.0, 1.0, 1.0, 1.0); // white pixel
            //FragColor = vec4(1.0, 1.0, 0.5, 1.0); // off-white pixel
            // make the color depend on position
            FragColor = vertexColor;
        }
        )";

    // Initialize if not initialized
    static bool initialized = false;
    GLuint v = 0;
    GLuint f = 0;
    //GLuint prog=0;
    // Once-off initialization
    if (!initialized)
    {
        // Compile program
        // We want to do things like shader compilation once only on initialization
        v = compile(GL_VERTEX_SHADER, vs);
        f = compile(GL_FRAGMENT_SHADER, fs);
        g_GL.prog = glCreateProgram();
        glAttachShader(g_GL.prog, v);
        glAttachShader(g_GL.prog, f);
        glLinkProgram(g_GL.prog);

        initialized = true;
    }

    glAttachShader(g_GL.prog, v);
    glAttachShader(g_GL.prog, f);
    glLinkProgram(g_GL.prog);

    glUseProgram(g_GL.prog);

    // One “pixel” coord in NDC (-1..1)
    glPointSize(5.0f);  // size of your “pixel”
}

void djVisualsInitOnceoff(const int N)
{
    if (N<0) return;
    if (g_GL.N<=0 || g_GL.vertexarray==nullptr || N!=g_GL.N)
    {
        // Allocate host arrays to hold positions
        if (g_GL.vertexarray) delete[]g_GL.vertexarray;

        // Allocate twice the size for x and y
        // Our incoming data is in separate arrays for x and y (for GPU kernel optimization reasons eg warp threads memory access speed), but we want to interleave them for OpenGL
        // [x0,y0, x1,y1, x2,y2, ...] easier for OpenGL to consume
        g_GL.vertexarray = new float[N * 2]; // x and y
        memset(g_GL.vertexarray, 0, N * 2 * sizeof(float));

        // Save size
        g_GL.N = N;

        glGenVertexArrays(1, &g_GL.vao);
        glGenBuffers(1, &g_GL.vbo);

        glBindVertexArray(g_GL.vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_GL.vbo);

        // allocate the max needed size ONCE
        glBufferData(GL_ARRAY_BUFFER, N * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        // set vertex layout
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }
}

void djVisualsDraw(float4* h_pos, float *h_x, float* h_y, float* h_z, float* radius, const int N)
{
    djVisualsInitOnceoff(N);

    // This scale is/was meant for things like, say, if may be necessary to scale to window width/height ...
    //const float scaleX = 1.f;//800.f;
    //const float scaleY = 1.f;//600.f;
    const float drawoffsetX = -0.1f;
    const float drawoffsetY = -0.85f;// should be near bottom of window

    // If many more points, make the points smaller to scale and fit on screen better
    const float basePointSize = 0.0f;//1.0f;
    if (N>=100000)
        glPointSize(0.5f + basePointSize);
    else if (N>=20000)
        glPointSize(3.0f + basePointSize);
    else if (N>=10000)
        glPointSize(1.0f + basePointSize);
    else if (N>=5000)
        glPointSize(2.0f + basePointSize);
//    else

glPointSize(3.0f);
glPointSize(0.8f);
//glPointSize(1.0f);

//fddsf

    // LOOP THROUGH MOLECULES' POSITIONS AND DRAW
    for ( int i = 0; i < N; ++i ) {
//        g_GL.vertexarray[i*2 + 0] = h_x[i] + drawoffsetX;
    //      g_GL.vertexarray[i*2 + 1] = h_y[i] + drawoffsetY;
        g_GL.vertexarray[i*2 + 0] = h_pos[i].x * 0.01f + drawoffsetX;
        g_GL.vertexarray[i*2 + 1] = h_pos[i].y * 0.01f + drawoffsetY;
        //g_GL.vert
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_GL.vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 2*N*sizeof(float), g_GL.vertexarray);
    glBindVertexArray(g_GL.vao);
    glDrawArrays(GL_POINTS, 0, N);// Draw all points at once
}

