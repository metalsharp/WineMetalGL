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
typedef void (WINAPI *PFNGLTEXSTORAGE2DPROC)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
typedef void (WINAPI *PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint *);
typedef void (WINAPI *PFNGLGENSAMPLERSPROC)(GLsizei, GLuint *);
typedef void (WINAPI *PFNGLBINDSAMPLERPROC)(GLuint, GLuint);
typedef void (WINAPI *PFNGLSAMPLERPARAMETERIPROC)(GLuint, GLenum, GLint);
typedef void (WINAPI *PFNGLGETTEXIMAGEPROC)(GLenum, GLint, GLenum, GLenum, void *);typedef void (WINAPI *PFNGLGETTEXTUREIMAGEPROC)(GLuint,GLint,GLenum,GLenum,GLsizei,void *);
typedef void (WINAPI *PFNGLGENERATEMIPMAPPROC)(GLenum);typedef void (WINAPI *PFNGLCREATETEXTURESPROC)(GLenum,GLsizei,GLuint*);typedef void (WINAPI *PFNGLTEXTURESTORAGE2DPROC)(GLuint,GLsizei,GLenum,GLsizei,GLsizei);typedef void (WINAPI *PFNGLTEXTUREPARAMETERIPROC)(GLuint,GLenum,GLint);typedef void (WINAPI *PFNGLTEXTURESUBIMAGE2DPROC)(GLuint,GLint,GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,const void*);typedef void (WINAPI *PFNGLBINDTEXTUREUNITPROC)(GLuint,GLuint);typedef void (WINAPI *PFNGLGETTEXTURELEVELPARAMETERIVPROC)(GLuint,GLint,GLenum,GLint*);
typedef void (WINAPI *PFNGLCOPYIMAGESUBDATAPROC)(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei);

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
    static const unsigned char rgba[] = {51,102,153, 51,102,153, 51,102,153, 51,102,153};
    static const uint16_t packed565[] = {0x3353, 0x3353, 0x3353, 0x3353};
    static const uint16_t rgba16[] = {13107,26214,39321,65535, 13107,26214,39321,65535, 13107,26214,39321,65535, 13107,26214,39321,65535};
    static const uint16_t rgbaHalf[] = {0x3266,0x3666,0x38cd,0x3c00, 0x3266,0x3666,0x38cd,0x3c00, 0x3266,0x3666,0x38cd,0x3c00, 0x3266,0x3666,0x38cd,0x3c00};
    static const unsigned char bgra[] = {153,102,51,255, 153,102,51,255, 153,102,51,255, 153,102,51,255};
    PIXELFORMATDESCRIPTOR pfd = {0}; HWND window; HDC dc; HGLRC context; int format;
    GLuint v, f, program, texture; GLint compiled, linked, location; PFNGLCOPYIMAGESUBDATAPROC copy_image; PFNGLGENERATEMIPMAPPROC generate_mipmap; PFNGLTEXSTORAGE2DPROC tex_storage; PFNGLCREATETEXTURESPROC create_textures; PFNGLTEXTURESTORAGE2DPROC texture_storage; PFNGLTEXTUREPARAMETERIPROC texture_parameteri; PFNGLTEXTURESUBIMAGE2DPROC texture_subimage; PFNGLBINDTEXTUREUNITPROC bind_texture_unit; PFNGLGETTEXTURELEVELPARAMETERIVPROC get_texture_level; unsigned char pixel[4] = {0};
    PFNGLCREATESHADERPROC create_shader; PFNGLSHADERSOURCEPROC shader_source; PFNGLCOMPILESHADERPROC compile_shader;
    PFNGLGETSHADERIVPROC get_shader_iv; PFNGLCREATEPROGRAMPROC create_program; PFNGLATTACHSHADERPROC attach_shader;
    PFNGLLINKPROGRAMPROC link_program; PFNGLGETPROGRAMIVPROC get_program_iv; PFNGLUSEPROGRAMPROC use_program;
    PFNGLGETUNIFORMLOCATIONPROC get_uniform_location; PFNGLUNIFORM1IPROC uniform1i;
    PFNGLGENTEXTURESPROC gen_textures; PFNGLBINDTEXTUREPROC bind_texture; PFNGLTEXPARAMETERIPROC tex_parameteri;
    PFNGLTEXIMAGE2DPROC tex_image_2d; PFNGLDELETETEXTURESPROC delete_textures;
    PFNGLGENSAMPLERSPROC gen_samplers; PFNGLBINDSAMPLERPROC bind_sampler; PFNGLSAMPLERPARAMETERIPROC sampler_parameteri; PFNGLGETTEXTUREIMAGEPROC get_texture_image; PFNGLGETTEXIMAGEPROC get_tex_image;
    setvbuf(stdout, NULL, _IONBF, 0);
    window = CreateWindowA("STATIC", "WineMetalGL texture", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, NULL, NULL, NULL, NULL);
    if (!window) return 11; dc = GetDC(window);
    pfd.nSize = sizeof(pfd); pfd.nVersion = 1; pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32;
    format = ChoosePixelFormat(dc, &pfd); if (!format || !SetPixelFormat(dc, format, &pfd)) return 12;
    context = wglCreateContext(dc); if (!context || !wglMakeCurrent(dc, context)) return 13;
    LOAD(PFNGLCREATESHADERPROC, create_shader, "glCreateShader"); LOAD(PFNGLSHADERSOURCEPROC, shader_source, "glShaderSource"); LOAD(PFNGLCOMPILESHADERPROC, compile_shader, "glCompileShader"); LOAD(PFNGLGETSHADERIVPROC, get_shader_iv, "glGetShaderiv");
    LOAD(PFNGLCREATEPROGRAMPROC, create_program, "glCreateProgram"); LOAD(PFNGLATTACHSHADERPROC, attach_shader, "glAttachShader"); LOAD(PFNGLLINKPROGRAMPROC, link_program, "glLinkProgram"); LOAD(PFNGLGETPROGRAMIVPROC, get_program_iv, "glGetProgramiv"); LOAD(PFNGLUSEPROGRAMPROC, use_program, "glUseProgram");
    LOAD(PFNGLGETUNIFORMLOCATIONPROC, get_uniform_location, "glGetUniformLocation"); LOAD(PFNGLUNIFORM1IPROC, uniform1i, "glUniform1i"); LOAD(PFNGLGENTEXTURESPROC, gen_textures, "glGenTextures"); LOAD(PFNGLBINDTEXTUREPROC, bind_texture, "glBindTexture"); LOAD(PFNGLTEXPARAMETERIPROC, tex_parameteri, "glTexParameteri"); LOAD(PFNGLTEXIMAGE2DPROC, tex_image_2d, "glTexImage2D"); LOAD(PFNGLTEXSTORAGE2DPROC, tex_storage, "glTexStorage2D"); LOAD(PFNGLCREATETEXTURESPROC, create_textures, "glCreateTextures"); LOAD(PFNGLTEXTURESTORAGE2DPROC, texture_storage, "glTextureStorage2D"); LOAD(PFNGLTEXTUREPARAMETERIPROC, texture_parameteri, "glTextureParameteri"); LOAD(PFNGLTEXTURESUBIMAGE2DPROC, texture_subimage, "glTextureSubImage2D"); LOAD(PFNGLBINDTEXTUREUNITPROC, bind_texture_unit, "glBindTextureUnit"); LOAD(PFNGLGETTEXTURELEVELPARAMETERIVPROC, get_texture_level, "glGetTextureLevelParameteriv"); LOAD(PFNGLDELETETEXTURESPROC, delete_textures, "glDeleteTextures"); LOAD(PFNGLCOPYIMAGESUBDATAPROC, copy_image, "glCopyImageSubData"); LOAD(PFNGLGENERATEMIPMAPPROC, generate_mipmap, "glGenerateMipmap"); LOAD(PFNGLGENSAMPLERSPROC, gen_samplers, "glGenSamplers"); LOAD(PFNGLBINDSAMPLERPROC, bind_sampler, "glBindSampler"); LOAD(PFNGLSAMPLERPARAMETERIPROC, sampler_parameteri, "glSamplerParameteri"); LOAD(PFNGLGETTEXIMAGEPROC, get_tex_image, "glGetTexImage"); LOAD(PFNGLGETTEXTUREIMAGEPROC, get_texture_image, "glGetTextureImage");
    v = create_shader(GL_VERTEX_SHADER); f = create_shader(GL_FRAGMENT_SHADER); { const char *s = vs; shader_source(v, 1, &s, NULL); } { const char *s = fs; shader_source(f, 1, &s, NULL); } compile_shader(v); compile_shader(f); get_shader_iv(v, GL_COMPILE_STATUS, &compiled); if (!compiled) return 15; get_shader_iv(f, GL_COMPILE_STATUS, &compiled); if (!compiled) return 16;
    program = create_program(); attach_shader(program, v); attach_shader(program, f); link_program(program); get_program_iv(program, GL_LINK_STATUS, &linked); if (!linked) return 17; use_program(program);
    gen_textures(1, &texture); bind_texture(GL_TEXTURE_2D, texture); tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); glPixelStorei(0x0CF5,1); tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, 0x1907, GL_UNSIGNED_BYTE, rgba); glPixelStorei(0x0CF5,4); unsigned char texture_readback[16]={0}; get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]!=51||texture_readback[1]!=102||texture_readback[2]!=153||texture_readback[3]!=255)return 18; printf("WINEMETALGL_UNPACK_ALIGNMENT_OK\\n"); generate_mipmap(GL_TEXTURE_2D); printf("WINEMETALGL_MIPMAP_OK\\n"); tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, 0x1907, 0x8363, packed565); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]<47||texture_readback[0]>55||texture_readback[1]<98||texture_readback[1]>106||texture_readback[2]<149||texture_readback[2]>159||texture_readback[3]!=255)return 18; printf("WINEMETALGL_TEXTURE_PACKED_OK\\n"); tex_image_2d(GL_TEXTURE_2D,0,GL_RGBA,2,2,0,GL_RGBA,0x1403,rgba16); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]<47||texture_readback[0]>55||texture_readback[1]<98||texture_readback[1]>106||texture_readback[2]<149||texture_readback[2]>157||texture_readback[3]<250)return 18; printf("WINEMETALGL_TEXTURE_USHORT_OK\\n"); tex_image_2d(GL_TEXTURE_2D,0,GL_RGBA,2,2,0,GL_RGBA,0x140B,rgbaHalf); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]<47||texture_readback[0]>55||texture_readback[1]<98||texture_readback[1]>106||texture_readback[2]<149||texture_readback[2]>157||texture_readback[3]<250)return 18; printf("WINEMETALGL_TEXTURE_HALF_FLOAT_OK\\n"); tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, 0x80E1, GL_UNSIGNED_BYTE, bgra); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]!=51||texture_readback[1]!=102||texture_readback[2]!=153||texture_readback[3]!=255)return 18; printf("WINEMETALGL_TEXTURE_BGRA_OK\\n"); GLuint copied_texture; gen_textures(1, &copied_texture); bind_texture(GL_TEXTURE_2D, copied_texture); tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, 0x1907, GL_UNSIGNED_BYTE, 0); copy_image(texture, GL_TEXTURE_2D, 0, 0, 0, 0, copied_texture, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]!=51||texture_readback[1]!=102||texture_readback[2]!=153||texture_readback[3]!=255)return 18; printf("WINEMETALGL_COPY_IMAGE_OK\\n"); GLuint sampler; gen_samplers(1, &sampler); bind_sampler(0, sampler); sampler_parameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR); sampler_parameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    location = get_uniform_location(program, "tex"); if (location < 0) return 18; uniform1i(location, 0);
    glViewport(0, 0, 64, 64); glDrawArrays(GL_TRIANGLES, 0, 3); glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    printf("GL33 texture readback rgba=%u,%u,%u,%u error=0x%x\n", pixel[0], pixel[1], pixel[2], pixel[3], (unsigned)glGetError());
    if (pixel[0] < 47 || pixel[0] > 55 || pixel[1] < 98 || pixel[1] > 106 || pixel[2] < 149 || pixel[2] > 157 || pixel[3] < 250) return 19;
    uint16_t half_pixel[4]={0}; glReadPixels(32,32,1,1,GL_RGBA,0x140B,half_pixel); if(half_pixel[0]<0x3000||half_pixel[0]>0x3400||half_pixel[1]<0x3500||half_pixel[1]>0x3800||half_pixel[3]<0x3b00)return 19; printf("WINEMETALGL_READBACK_HALF_FLOAT_OK\\n");
    uint16_t packed_pixel = 0; glReadPixels(32, 32, 1, 1, 0x1907, 0x8363, &packed_pixel);
    if (packed_pixel < 0x2400 || packed_pixel > 0x3800) return 19;
    printf("WINEMETALGL_READBACK_PACKED_OK\\n");
    glPixelStorei(0x0D05,8); unsigned char aligned[16]={0}; glReadPixels(32,31,1,2,0x1907,GL_UNSIGNED_BYTE,aligned); if(aligned[0]<47||aligned[1]<98||aligned[2]<149||aligned[8]<47||aligned[9]<98||aligned[10]<149)return 19; printf("WINEMETALGL_PACK_ALIGNMENT_OK\\n"); glPixelStorei(0x0D05,4); glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,0,0,2,2,0); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]<47||texture_readback[0]>55||texture_readback[1]<98||texture_readback[1]>110||texture_readback[2]<149||texture_readback[2]>160||texture_readback[3]<250)return 19; printf("WINEMETALGL_COPY_TEX_OK\\n"); tex_storage(GL_TEXTURE_2D,1,GL_RGBA,2,2); printf("WINEMETALGL_TEX_STORAGE_OK\\n"); GLuint dsa_texture; create_textures(GL_TEXTURE_2D,1,&dsa_texture); texture_storage(dsa_texture,1,GL_RGBA,2,2); texture_parameteri(dsa_texture,GL_TEXTURE_MIN_FILTER,GL_LINEAR); bind_texture_unit(0,dsa_texture); GLint dsa_width=0; get_texture_level(dsa_texture,0,0x1000,&dsa_width); if(dsa_width!=2)return 19; texture_subimage(dsa_texture,0,0,0,2,2,0x80E1,GL_UNSIGNED_BYTE,bgra); bind_texture(GL_TEXTURE_2D,dsa_texture); get_tex_image(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,texture_readback); if(texture_readback[0]!=51||texture_readback[1]!=102||texture_readback[2]!=153||texture_readback[3]!=255)return 19; get_texture_image(dsa_texture,0,GL_RGBA,GL_UNSIGNED_BYTE,sizeof(texture_readback),texture_readback); if(texture_readback[0]!=51||texture_readback[1]!=102||texture_readback[2]!=153||texture_readback[3]!=255)return 19; printf("WINEMETALGL_DSA_TEXTURE_OK\\n");
    printf("WINEMETALGL_GL33_TEXTURE_OK\n"); SwapBuffers(dc); GLuint textures_to_delete[2]={texture,copied_texture}; delete_textures(2, textures_to_delete); wglMakeCurrent(NULL, NULL); wglDeleteContext(context); ReleaseDC(window, dc); DestroyWindow(window); return 0;
}
