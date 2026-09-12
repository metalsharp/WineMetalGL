#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>

#define GL_VERTEX_SHADER 0x8b31
#define GL_FRAGMENT_SHADER 0x8b30
#define GL_COMPILE_STATUS 0x8b81
#define GL_LINK_STATUS 0x8b82
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88e4
#define GL_FLOAT 0x1406
#define GL_UNSIGNED_SHORT 0x1403
#define GL_TRIANGLES 0x0004
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401

typedef GLuint (WINAPI *PFNGLCREATESHADERPROC)(GLenum);
typedef void (WINAPI *PFNGLSHADERSOURCEPROC)(GLuint, GLsizei, const char *const *, const GLint *);
typedef void (WINAPI *PFNGLCOMPILESHADERPROC)(GLuint);
typedef void (WINAPI *PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint *);
typedef GLuint (WINAPI *PFNGLCREATEPROGRAMPROC)(void);
typedef void (WINAPI *PFNGLATTACHSHADERPROC)(GLuint, GLuint);
typedef void (WINAPI *PFNGLLINKPROGRAMPROC)(GLuint);
typedef void (WINAPI *PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint *);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC)(GLuint);
typedef void (WINAPI *PFNGLDELETESHADERPROC)(GLuint);
typedef void (WINAPI *PFNGLDELETEPROGRAMPROC)(GLuint);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC)(GLuint, const char *);
typedef void (WINAPI *PFNGLUNIFORM4FPROC)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
typedef void (WINAPI *PFNGLGENBUFFERSPROC)(GLsizei, GLuint *);
typedef void (WINAPI *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void (WINAPI *PFNGLBUFFERDATAPROC)(GLenum, intptr_t, const void *, GLenum);
typedef void (WINAPI *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void (WINAPI *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint);
typedef void (WINAPI *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
typedef void (WINAPI *PFNGLDRAWELEMENTSPROC)(GLenum, GLsizei, GLenum, const void *);
typedef void (WINAPI *PFNGLDRAWELEMENTSBASEVERTEXPROC)(GLenum, GLsizei, GLenum, const void *, GLint);
typedef void (WINAPI *PFNGLBINDVERTEXBUFFERPROC)(GLuint,GLuint,intptr_t,GLsizei); typedef void (WINAPI *PFNGLVERTEXATTRIBBINDINGPROC)(GLuint,GLuint); typedef void (WINAPI *PFNGLVERTEXATTRIBFORMATPROC)(GLuint,GLint,GLenum,GLboolean,GLuint);

static void *get_proc(const char *name)
{
    PROC proc = wglGetProcAddress(name);
    HMODULE module;
    if (proc) return (void *)proc;
    module = GetModuleHandleA("opengl32.dll");
    return module ? (void *)GetProcAddress(module, name) : NULL;
}

#define LOAD(type, variable, symbol) do { \
    variable = (type)get_proc(symbol); \
    if (!variable) { printf("FAIL missing %s\n", symbol); return 14; } \
} while (0)

int main(void)
{
    static const char vertex_source[] =
        "#version 330 core\n"
        "layout(location = 0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char fragment_source[] =
        "#version 330 core\n"
        "out vec4 color;\n"
        "uniform vec4 tint;\n"
        "void main() { color = tint; }\n";
    static const float vertices[] = {0.0f, 0.0f, -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    static const uint16_t indices[] = {0, 1, 2};
    PIXELFORMATDESCRIPTOR pfd = {0};
    HWND window;
    HDC dc;
    HGLRC context;
    GLuint vertex_shader, fragment_shader, program, vbo, ibo;
    GLint compiled = 0, linked = 0, location;
    unsigned char pixel[4] = {0};
    int format;
    PFNGLCREATESHADERPROC create_shader;
    PFNGLSHADERSOURCEPROC shader_source;
    PFNGLCOMPILESHADERPROC compile_shader;
    PFNGLGETSHADERIVPROC get_shader_iv;
    PFNGLCREATEPROGRAMPROC create_program;
    PFNGLATTACHSHADERPROC attach_shader;
    PFNGLLINKPROGRAMPROC link_program;
    PFNGLGETPROGRAMIVPROC get_program_iv;
    PFNGLUSEPROGRAMPROC use_program;
    PFNGLDELETESHADERPROC delete_shader;
    PFNGLDELETEPROGRAMPROC delete_program;
    PFNGLGETUNIFORMLOCATIONPROC get_uniform_location;
    PFNGLUNIFORM4FPROC uniform4f;
    PFNGLGENBUFFERSPROC gen_buffers;
    PFNGLBINDBUFFERPROC bind_buffer;
    PFNGLBUFFERDATAPROC buffer_data;
    PFNGLDELETEBUFFERSPROC delete_buffers;
    PFNGLENABLEVERTEXATTRIBARRAYPROC enable_attrib;
    PFNGLVERTEXATTRIBPOINTERPROC attrib_pointer;
    PFNGLDRAWELEMENTSPROC draw_elements; PFNGLDRAWELEMENTSBASEVERTEXPROC draw_elements_base; PFNGLBINDVERTEXBUFFERPROC bind_vertex_buffer; PFNGLVERTEXATTRIBBINDINGPROC attrib_binding; PFNGLVERTEXATTRIBFORMATPROC attrib_format;

    setvbuf(stdout, NULL, _IONBF, 0);
    window = CreateWindowA("STATIC", "WineMetalGL GL33 resources", WS_OVERLAPPEDWINDOW,
                           0, 0, 64, 64, NULL, NULL, NULL, NULL);
    if (!window) return 11;
    dc = GetDC(window);
    pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32;
    format = ChoosePixelFormat(dc, &pfd);
    if (!format || !SetPixelFormat(dc, format, &pfd)) return 12;
    context = wglCreateContext(dc);
    if (!context || !wglMakeCurrent(dc, context)) return 13;

    LOAD(PFNGLCREATESHADERPROC, create_shader, "glCreateShader");
    LOAD(PFNGLSHADERSOURCEPROC, shader_source, "glShaderSource");
    LOAD(PFNGLCOMPILESHADERPROC, compile_shader, "glCompileShader");
    LOAD(PFNGLGETSHADERIVPROC, get_shader_iv, "glGetShaderiv");
    LOAD(PFNGLCREATEPROGRAMPROC, create_program, "glCreateProgram");
    LOAD(PFNGLATTACHSHADERPROC, attach_shader, "glAttachShader");
    LOAD(PFNGLLINKPROGRAMPROC, link_program, "glLinkProgram");
    LOAD(PFNGLGETPROGRAMIVPROC, get_program_iv, "glGetProgramiv");
    LOAD(PFNGLUSEPROGRAMPROC, use_program, "glUseProgram");
    LOAD(PFNGLDELETESHADERPROC, delete_shader, "glDeleteShader");
    LOAD(PFNGLDELETEPROGRAMPROC, delete_program, "glDeleteProgram");
    LOAD(PFNGLGETUNIFORMLOCATIONPROC, get_uniform_location, "glGetUniformLocation");
    LOAD(PFNGLUNIFORM4FPROC, uniform4f, "glUniform4f");
    LOAD(PFNGLGENBUFFERSPROC, gen_buffers, "glGenBuffers");
    LOAD(PFNGLBINDBUFFERPROC, bind_buffer, "glBindBuffer");
    LOAD(PFNGLBUFFERDATAPROC, buffer_data, "glBufferData");
    LOAD(PFNGLDELETEBUFFERSPROC, delete_buffers, "glDeleteBuffers");
    LOAD(PFNGLENABLEVERTEXATTRIBARRAYPROC, enable_attrib, "glEnableVertexAttribArray");
    LOAD(PFNGLVERTEXATTRIBPOINTERPROC, attrib_pointer, "glVertexAttribPointer");
    LOAD(PFNGLDRAWELEMENTSPROC, draw_elements, "glDrawElements"); LOAD(PFNGLDRAWELEMENTSBASEVERTEXPROC, draw_elements_base, "glDrawElementsBaseVertex"); LOAD(PFNGLBINDVERTEXBUFFERPROC, bind_vertex_buffer, "glBindVertexBuffer"); LOAD(PFNGLVERTEXATTRIBBINDINGPROC, attrib_binding, "glVertexAttribBinding"); LOAD(PFNGLVERTEXATTRIBFORMATPROC, attrib_format, "glVertexAttribFormat");

    vertex_shader = create_shader(GL_VERTEX_SHADER); fragment_shader = create_shader(GL_FRAGMENT_SHADER);
    { const char *source = vertex_source; shader_source(vertex_shader, 1, &source, NULL); compile_shader(vertex_shader); }
    { const char *source = fragment_source; shader_source(fragment_shader, 1, &source, NULL); compile_shader(fragment_shader); }
    get_shader_iv(vertex_shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) return 15;
    get_shader_iv(fragment_shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) return 16;
    program = create_program(); attach_shader(program, vertex_shader); attach_shader(program, fragment_shader);
    link_program(program); get_program_iv(program, GL_LINK_STATUS, &linked);
    if (!linked) return 17;
    use_program(program);
    location = get_uniform_location(program, "tint");
    if (location < 0) return 18;
    uniform4f(location, 0.2f, 0.4f, 0.6f, 1.0f);

    gen_buffers(1, &vbo); bind_buffer(GL_ARRAY_BUFFER, vbo); buffer_data(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    gen_buffers(1, &ibo); bind_buffer(GL_ELEMENT_ARRAY_BUFFER, ibo); buffer_data(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    enable_attrib(0); attrib_pointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (const void *)0); bind_vertex_buffer(0,vbo,0,2*sizeof(float)); attrib_binding(0,0); attrib_format(0,2,GL_FLOAT,GL_FALSE,0);
    glViewport(0, 0, 64, 64); draw_elements_base(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, (const void *)0, 1);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    printf("GL33 resources readback rgba=%u,%u,%u,%u error=0x%x\n", pixel[0], pixel[1], pixel[2], pixel[3], (unsigned)glGetError());
    if (pixel[0] < 47 || pixel[0] > 55 || pixel[1] < 98 || pixel[1] > 106 || pixel[2] < 149 || pixel[2] > 157 || pixel[3] < 250) return 19;
    printf("WINEMETALGL_GL33_RESOURCES_OK\n");
    SwapBuffers(dc);
    delete_buffers(1, &ibo); delete_buffers(1, &vbo); delete_program(program); delete_shader(fragment_shader); delete_shader(vertex_shader);
    wglMakeCurrent(NULL, NULL); wglDeleteContext(context); ReleaseDC(window, dc); DestroyWindow(window);
    return 0;
}
