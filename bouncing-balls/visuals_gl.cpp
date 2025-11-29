// dj-cuda-samples
// "Simple" putpixel-style OpenGL-based visuals
//
// Created dj2025-11
// https://github.com/davidjoffe/dj-cuda-samples
// Copyright David Joffe 2025

#include "defs.h"
#include <glad/glad.h>

// Small helper to compile a shader
GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}

// Shader handle
GLuint prog = 0;

void djVisualsInit()
{
    // Simple shader setup (no error checking here for brevity)
    // vertex shader (vs), fragment shader (fs)
    const char* vs = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        )";
    const char* fs = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0); // white pixel
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
        prog = glCreateProgram();
        glAttachShader(prog, v);
        glAttachShader(prog, f);
        glLinkProgram(prog);

        initialized = true;
    }

    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);

    glUseProgram(prog);

    // One “pixel” coord in NDC (-1..1)
    glPointSize(5.0f);  // size of your “pixel”
}

void djVisualsDraw(float *h_x, float* h_y, float* h_z, float* radius, int N)
{
    // One "pixel" coord in NDC (-1..1)
    float px = -0.2f;
    float py =  0.3f;
    float pts[2] = { px, py };

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pts), pts, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Not sure if we might need to glUseProgram(prog) here ... seems not in testing but maybe in future if more visualization stuff causes shaders to change ...
    //glUseProgram(prog);

    // This scale is/was meant for things like, say, if may be necessary to scale to window width/height ...
    const float scaleX = 1.f;//800.f;
    const float scaleY = 1.f;//600.f;
    const float drawoffsetX = 0.f;
    const float drawoffsetY = -0.95f;// should be near bottom of window
    glPointSize(3.0f);

    // LOOP THROUGH BOUNCING BALLS' POSITIONS AND DRAW
    for ( int i = 0; i < N; ++i ) {
        pts[0] = h_x[i] + drawoffsetX;
        pts[1] = h_y[i] + drawoffsetY;
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(pts), pts, GL_STATIC_DRAW);
        //glSetPointSize(5.0f * ());  // size of your "pixel"”" - in future could maybe scale with 'z' so further away look smaller?
        glDrawArrays(GL_POINTS, 0, 1); // <- "put pixel"
    }
}
