#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define GL_VERTEX_SHADER 0x8b31
#define GL_FRAGMENT_SHADER 0x8b30
#define GL_COMPILE_STATUS 0x8b81
#define GL_LINK_STATUS 0x8b82
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0de1
#endif
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FRAMEBUFFER 0x8d40
#define GL_COLOR_ATTACHMENT0 0x8ce0
#define GL_FRAMEBUFFER_COMPLETE 0x8cd5
#define GL_TEXTURE_2D_MULTISAMPLE 0x9100
#define GL_RGBA8 0x8058
#ifndef GL_COLOR_BUFFER_BIT
#define GL_COLOR_BUFFER_BIT 0x4000
#endif
#define GL_READ_FRAMEBUFFER 0x8ca8
#define GL_DRAW_FRAMEBUFFER 0x8ca9
#define GL_NEAREST 0x2600

typedef GLuint (WINAPI *CreateShader)(GLenum); typedef void (WINAPI *ShaderSource)(GLuint,GLsizei,const char *const*,const GLint*); typedef void (WINAPI *CompileShader)(GLuint); typedef void (WINAPI *GetShaderiv)(GLuint,GLenum,GLint*); typedef GLuint (WINAPI *CreateProgram)(void); typedef void (WINAPI *AttachShader)(GLuint,GLuint); typedef void (WINAPI *LinkProgram)(GLuint); typedef void (WINAPI *GetProgramiv)(GLuint,GLenum,GLint*); typedef void (WINAPI *UseProgram)(GLuint); typedef void (WINAPI *GenTextures)(GLsizei,GLuint*); typedef void (WINAPI *BindTexture)(GLenum,GLuint); typedef void (WINAPI *TexImage2D)(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*); typedef void (WINAPI *TexImage2DMultisample)(GLenum,GLsizei,GLenum,GLsizei,GLsizei,GLboolean); typedef void (WINAPI *GenFramebuffers)(GLsizei,GLuint*); typedef void (WINAPI *BindFramebuffer)(GLenum,GLuint); typedef void (WINAPI *FramebufferTexture2D)(GLenum,GLenum,GLenum,GLuint,GLint); typedef GLenum (WINAPI *CheckFramebufferStatus)(GLenum); typedef void (WINAPI *BlitFramebuffer)(GLint,GLint,GLint,GLint,GLint,GLint,GLint,GLint,GLbitfield,GLenum);
static void *p(const char *n){PROC x=wglGetProcAddress(n);HMODULE m;if(x)return(void*)x;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n) do{v=(t)p(n);if(!v){printf("FAIL %s\n",n);return 14;}}while(0)
int main(void){
 static const char vs[]="#version 330 core\nconst vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));void main(){gl_Position=vec4(p[gl_VertexID],0,1);}";
 static const char fs[]="#version 330 core\nout vec4 color;void main(){color=vec4(.2,.4,.6,1);}";
 PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,f,pr,tex,fbo;GLint ok;GLenum status;unsigned char px[4]={0};
 CreateShader cs;ShaderSource ss;CompileShader co;GetShaderiv si;CreateProgram cp;AttachShader at;LinkProgram li;GetProgramiv pi;UseProgram us;GenTextures gt;BindTexture bt;TexImage2D ti;TexImage2DMultisample tm;GenFramebuffers gf;BindFramebuffer bf;FramebufferTexture2D ft;CheckFramebufferStatus fc;BlitFramebuffer blit;
 w=CreateWindowA("STATIC","WineMetalGL FBO",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;
 L(CreateShader,cs,"glCreateShader");L(ShaderSource,ss,"glShaderSource");L(CompileShader,co,"glCompileShader");L(GetShaderiv,si,"glGetShaderiv");L(CreateProgram,cp,"glCreateProgram");L(AttachShader,at,"glAttachShader");L(LinkProgram,li,"glLinkProgram");L(GetProgramiv,pi,"glGetProgramiv");L(UseProgram,us,"glUseProgram");L(GenTextures,gt,"glGenTextures");L(BindTexture,bt,"glBindTexture");L(TexImage2D,ti,"glTexImage2D");L(TexImage2DMultisample,tm,"glTexImage2DMultisample");L(GenFramebuffers,gf,"glGenFramebuffers");L(BindFramebuffer,bf,"glBindFramebuffer");L(FramebufferTexture2D,ft,"glFramebufferTexture2D");L(CheckFramebufferStatus,fc,"glCheckFramebufferStatus");L(BlitFramebuffer,blit,"glBlitFramebuffer");
 v=cs(GL_VERTEX_SHADER);f=cs(GL_FRAGMENT_SHADER);{const char*s=vs;ss(v,1,&s,0);}{const char*s=fs;ss(f,1,&s,0);}co(v);co(f);si(v,GL_COMPILE_STATUS,&ok);if(!ok)return 15;si(f,GL_COMPILE_STATUS,&ok);if(!ok)return 16;pr=cp();at(pr,v);at(pr,f);li(pr);pi(pr,GL_LINK_STATUS,&ok);if(!ok)return 17;us(pr);
 gt(1,&tex);bt(GL_TEXTURE_2D_MULTISAMPLE,tex);tm(GL_TEXTURE_2D_MULTISAMPLE,4,GL_RGBA8,2,2,1);gf(1,&fbo);bf(GL_FRAMEBUFFER,fbo);ft(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D_MULTISAMPLE,tex,0);status=fc(GL_FRAMEBUFFER);if(status!=GL_FRAMEBUFFER_COMPLETE){printf("FAIL FBO status=0x%x\n",status);return 18;}
 glViewport(0,0,2,2);glDrawArrays(GL_TRIANGLES,0,3);glReadPixels(1,1,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("GL33 FBO readback rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 19;printf("WINEMETALGL_MULTISAMPLE_OK\n"); GLuint dstTex,dstFbo; gt(1,&dstTex);bt(GL_TEXTURE_2D,dstTex);ti(GL_TEXTURE_2D,0,GL_RGBA8,2,2,0,GL_RGBA,GL_UNSIGNED_BYTE,0);gf(1,&dstFbo);bf(GL_FRAMEBUFFER,dstFbo);ft(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,dstTex,0);bf(GL_READ_FRAMEBUFFER,fbo);bf(GL_DRAW_FRAMEBUFFER,dstFbo);blit(0,0,2,2,0,0,2,2,GL_COLOR_BUFFER_BIT,GL_NEAREST);bf(GL_READ_FRAMEBUFFER,dstFbo);glReadPixels(1,1,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 20;printf("WINEMETALGL_BLIT_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;
}
