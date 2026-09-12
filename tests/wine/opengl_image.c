#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define GL_COMPUTE_SHADER 0x91b9
#define GL_COMPILE_STATUS 0x8b81
#define GL_LINK_STATUS 0x8b82
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0de1
#endif
#define GL_RGBA 0x1908
#define GL_RGBA8 0x8058
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FRAMEBUFFER 0x8d40
#define GL_COLOR_ATTACHMENT0 0x8ce0
#define GL_FRAMEBUFFER_COMPLETE 0x8cd5
#define GL_WRITE_ONLY 0x88b9

typedef GLuint(WINAPI*CS)(GLenum);typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI*CO)(GLuint);typedef void(WINAPI*SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI*CP)(void);typedef void(WINAPI*AT)(GLuint,GLuint);typedef void(WINAPI*LI)(GLuint);typedef void(WINAPI*PI)(GLuint,GLenum,GLint*);typedef void(WINAPI*US)(GLuint);typedef void(WINAPI*GT)(GLsizei,GLuint*);typedef void(WINAPI*BT)(GLenum,GLuint);typedef void(WINAPI*TI)(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*);typedef void(WINAPI*GI)(GLuint,GLuint,GLint,GLboolean,GLint,GLenum,GLenum);typedef void(WINAPI*DC)(GLuint,GLuint,GLuint);typedef void(WINAPI*GF)(GLsizei,GLuint*);typedef void(WINAPI*BF)(GLenum,GLuint);typedef void(WINAPI*FT)(GLenum,GLenum,GLenum,GLuint,GLint);typedef GLenum(WINAPI*FC)(GLenum);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){static const char src[]="#version 430 core\nlayout(local_size_x=1)in;layout(rgba8,binding=0)uniform writeonly image2D img;void main(){imageStore(img,ivec2(0),vec4(.2,.4,.6,1));}";PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint s,pr,t,fbo;GLint ok;GLenum stat;unsigned char px[4]={0};CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;GT gt;BT bt;TI ti;GI gi;DC dispatch;GF gf;BF bf;FT ft;FC fc;w=CreateWindowA("STATIC","WineMetalGL image",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");L(US,us,"glUseProgram");L(GT,gt,"glGenTextures");L(BT,bt,"glBindTexture");L(TI,ti,"glTexImage2D");L(GI,gi,"glBindImageTexture");L(DC,dispatch,"glDispatchCompute");L(GF,gf,"glGenFramebuffers");L(BF,bf,"glBindFramebuffer");L(FT,ft,"glFramebufferTexture2D");L(FC,fc,"glCheckFramebufferStatus");s=cs(GL_COMPUTE_SHADER);{const char*x=src;ss(s,1,&x,0);}co(s);si(s,GL_COMPILE_STATUS,&ok);if(!ok)return 15;pr=cp();at(pr,s);li(pr);pi(pr,GL_LINK_STATUS,&ok);if(!ok)return 16;us(pr);gt(1,&t);bt(GL_TEXTURE_2D,t);ti(GL_TEXTURE_2D,0,GL_RGBA8,2,2,0,GL_RGBA,GL_UNSIGNED_BYTE,0);gi(0,t,0,0,0,GL_WRITE_ONLY,GL_RGBA8);dispatch(1,1,1);gf(1,&fbo);bf(GL_FRAMEBUFFER,fbo);ft(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t,0);stat=fc(GL_FRAMEBUFFER);if(stat!=GL_FRAMEBUFFER_COMPLETE)return 17;glReadPixels(0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("Image readback rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 18;printf("WINEMETALGL_IMAGE_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
