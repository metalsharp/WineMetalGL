#include <windows.h>
#include <GL/gl.h>
#include <stdint.h>
#include <stdio.h>

#define GL_VERTEX_SHADER 0x8b31
#define GL_FRAGMENT_SHADER 0x8b30
#define GL_COMPILE_STATUS 0x8b81
#define GL_LINK_STATUS 0x8b82
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0de1
#endif
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_LINEAR 0x2601
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TRIANGLES 0x0004

typedef GLuint (WINAPI *PFNGLCREATESHADERPROC)(GLenum);
typedef void (WINAPI *PFNGLSHADERSOURCEPROC)(GLuint, GLsizei, const char *const *, const GLint *);
typedef void (WINAPI *PFNGLCOMPILESHADERPROC)(GLuint);
typedef void (WINAPI *PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint *);
typedef GLuint (WINAPI *PFNGLCREATEPROGRAMPROC)(void);
typedef void (WINAPI *PFNGLATTACHSHADERPROC)(GLuint, GLuint);
typedef void (WINAPI *PFNGLLINKPROGRAMPROC)(GLuint);
typedef void (WINAPI *PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint *);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC)(GLuint);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC)(GLuint, const char *);
typedef void (WINAPI *PFNGLUNIFORM1IPROC)(GLint, GLint);
typedef void (WINAPI *PFNGLGENTEXTURESPROC)(GLsizei, GLuint *);
typedef void (WINAPI *PFNGLBINDTEXTUREPROC)(GLenum, GLuint);
typedef void (WINAPI *PFNGLTEXPARAMETERIPROC)(GLenum, GLenum, GLint);
typedef void (WINAPI *PFNGLTEXIMAGE2DPROC)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
typedef void (WINAPI *PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint *);

typedef void (WINAPI *PFNGLDRAWAREASPROC)(GLenum, GLint, GLint); /* unused; keeps fixture self-contained */
static void *get_proc(const char *name)
{
    PROC p = wglGetProcAddress(name);
    HMODULE m;
    if (p) return (void *)p;
    m = GetModuleHandleA("opengl32.dll");
    return m ? (void *)GetProcAddress(m, name) : NULL;
}
#define LOAD(type, var, name) do { var = (type)get_proc(name); if (!var) { printf("FAIL missing %s\n", name); return 14; } } while (0)

int main(void)
{
    static const char vs[] = "#version 330 core\nconst vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); void main(){gl_Position=vec4(p[gl_VertexID],0,1);}";
    static const char fs[] = "#version 330 core\nuniform sampler2D tex; out vec4 color; void main(){color=texture(tex,vec2(0.5,0.5));}";
    static const unsigned char rgba[] = {51,102,153,255, 51,102,153,255, 51,102,153,255, 51,102,153,255};
    PIXELFORMATDESCRIPTOR pfd = {0}; HWND window; HDC dc; HGLRC context; int format;
    GLuint v, f, program, texture; GLint compiled, linked, location; unsigned char pixel[4] = {0};
    PFNGLCREATESHADERPROC create_shader; PFNGLSHADERSOURCEPROC shader_source; PFNGLCOMPILESHADERPROC compile_shader;
    PFNGLGETSHADERIVPROC get_shader_iv; PFNGLCREATEPROGRAMPROC create_program; PFNGLATTACHSHADERPROC attach_shader;
    PFNGLLINKPROGRAMPROC link_program; PFNGLGETPROGRAMIVPROC get_program_iv; PFNGLUSEPROGRAMPROC use_program;
    PFNGLGETUNIFORMLOCATIONPROC get_uniform_location; PFNGLUNIFORM1IPROC uniform1i;
    PFNGLGENTEXTURESPROC gen_textures; PFNGLBINDTEXTUREPROC bind_texture; PFNGLTEXPARAMETERIPROC tex_parameteri;
    PFNGLTEXIMAGE2DPROC tex_image_2d; PFNGLDELETETEXTURESPROC delete_textures;
    setvbuf(stdout, NULL, _IONBF, 0);
    window = CreateWindowA("STATIC", "WineMetalGL texture", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, NULL, NULL, NULL, NULL);
    if (!window) return 11; dc = GetDC(window);
    pfd.nSize = sizeof(pfd); pfd.nVersion = 1; pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32;
    format = ChoosePixelFormat(dc, &pfd); if (!format || !SetPixelFormat(dc, format, &pfd)) return 12;
    context = wglCreateContext(dc); if (!context || !wglMakeCurrent(dc, context)) return 13;
    LOAD(PFNGLCREATESHADERPROC, create_shader, "glCreateShader"); LOAD(PFNGLSHADERSOURCEPROC, shader_source, "glShaderSource"); LOAD(PFNGLCOMPILESHADERPROC, compile_shader, "glCompileShader"); LOAD(PFNGLGETSHADERIVPROC, get_shader_iv, "glGetShaderiv");
    LOAD(PFNGLCREATEPROGRAMPROC, create_program, "glCreateProgram"); LOAD(PFNGLATTACHSHADERPROC, attach_shader, "glAttachShader"); LOAD(PFNGLLINKPROGRAMPROC, link_program, "glLinkProgram"); LOAD(PFNGLGETPROGRAMIVPROC, get_program_iv, "glGetProgramiv"); LOAD(PFNGLUSEPROGRAMPROC, use_program, "glUseProgram");
    LOAD(PFNGLGETUNIFORMLOCATIONPROC, get_uniform_location, "glGetUniformLocation"); LOAD(PFNGLUNIFORM1IPROC, uniform1i, "glUniform1i"); LOAD(PFNGLGENTEXTURESPROC, gen_textures, "glGenTextures"); LOAD(PFNGLBINDTEXTUREPROC, bind_texture, "glBindTexture"); LOAD(PFNGLTEXPARAMETERIPROC, tex_parameteri, "glTexParameteri"); LOAD(PFNGLTEXIMAGE2DPROC, tex_image_2d, "glTexImage2D"); LOAD(PFNGLDELETETEXTURESPROC, delete_textures, "glDeleteTextures");
    v = create_shader(GL_VERTEX_SHADER); f = create_shader(GL_FRAGMENT_SHADER); { const char *s = vs; shader_source(v, 1, &s, NULL); } { const char *s = fs; shader_source(f, 1, &s, NULL); } compile_shader(v); compile_shader(f); get_shader_iv(v, GL_COMPILE_STATUS, &compiled); if (!compiled) return 15; get_shader_iv(f, GL_COMPILE_STATUS, &compiled); if (!compiled) return 16;
    program = create_program(); attach_shader(program, v); attach_shader(program, f); link_program(program); get_program_iv(program, GL_LINK_STATUS, &linked); if (!linked) return 17; use_program(program);
    gen_textures(1, &texture); bind_texture(GL_TEXTURE_2D, texture); tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    location = get_uniform_location(program, "tex"); if (location < 0) return 18; uniform1i(location, 0);
    glViewport(0, 0, 64, 64); glDrawArrays(GL_TRIANGLES, 0, 3); glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    printf("GL33 texture readback rgba=%u,%u,%u,%u error=0x%x\n", pixel[0], pixel[1], pixel[2], pixel[3], (unsigned)glGetError());
    if (pixel[0] < 47 || pixel[0] > 55 || pixel[1] < 98 || pixel[1] > 106 || pixel[2] < 149 || pixel[2] > 157 || pixel[3] < 250) return 19;
    printf("WINEMETALGL_GL33_TEXTURE_OK\n"); SwapBuffers(dc); delete_textures(1, &texture); wglMakeCurrent(NULL, NULL); wglDeleteContext(context); ReleaseDC(window, dc); DestroyWindow(window); return 0;
}
