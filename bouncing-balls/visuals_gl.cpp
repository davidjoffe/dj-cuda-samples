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
                1.0
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

void djVisualsDraw(float *h_x, float* h_y, float* h_z, float* radius, const int N)
{
    // One "pixel" coord in NDC (-1..1)
    //float px = -0.2f;
    //float py =  0.3f;
    float pts[2] = { 0.0f, 0.0f };

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

    // If many more points, make the points smaller to scale and fit on screen better
    const float basePointSize = 0.0f;//1.0f;
    if (N>=100000)
        glPointSize(0.5f + basePointSize);
    else if (N>=20000)
        glPointSize(0.8f + basePointSize);
    else if (N>=10000)
        glPointSize(1.0f + basePointSize);
    else if (N>=5000)
        glPointSize(2.0f + basePointSize);
    else
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
