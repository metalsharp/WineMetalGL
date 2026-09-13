#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define VS 0x8b31
#define FS 0x8b30
#define COMPILE 0x8b81
#define LINK 0x8b82
#define TEXTURE_2D 0x0de1
#define DEPTH_COMPONENT24 0x81a6
#define DEPTH_COMPONENT 0x1902
#define FRAMEBUFFER 0x8d40
#define DEPTH_ATTACHMENT 0x8d00
#define COLOR_ATTACHMENT0 0x8ce0
#define RGBA8 0x8058
#define RGBA 0x1908
#define FRAMEBUFFER_COMPLETE 0x8cd5
#define NEAREST 0x2600
#define TRIANGLES 4
#define DEPTH_BUFFER_BIT 0x0100

typedef GLuint(WINAPI *CS)(GLenum);typedef void(WINAPI *SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI *CO)(GLuint);typedef void(WINAPI *SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI *CP)(void);typedef void(WINAPI *AT)(GLuint,GLuint);typedef void(WINAPI *LI)(GLuint);typedef void(WINAPI *PI)(GLuint,GLenum,GLint*);typedef void(WINAPI *US)(GLuint);typedef void(WINAPI *GT)(GLsizei,GLuint*);typedef void(WINAPI *BT)(GLenum,GLuint);typedef void(WINAPI *TP)(GLenum,GLenum,GLint);typedef void(WINAPI *TI)(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*);typedef void(WINAPI *GF)(GLsizei,GLuint*);typedef void(WINAPI *BF)(GLenum,GLuint);typedef void(WINAPI *FT)(GLenum,GLenum,GLenum,GLuint,GLint);typedef GLenum(WINAPI *FC)(GLenum);typedef GLint(WINAPI *UL)(GLuint,const char*);typedef void(WINAPI *U1)(GLint,GLint);typedef void(WINAPI *DA)(GLenum,GLint,GLint);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m=GetModuleHandleA("opengl32.dll");return q?(void*)q:(m?(void*)GetProcAddress(m,n):0);}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){static const char vs[]="#version 330 core\nconst vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));void main(){gl_Position=vec4(p[gl_VertexID],0,1);}";static const char fs[]="#version 330 core\nuniform sampler2D depthTex;out vec4 color;void main(){float d=texture(depthTex,vec2(.5)).r;color=vec4(d,0,0,1);}";PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,f,pr,t,color,fbo;GLint ok;unsigned char px[4]={0};CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;GT gt;BT bt;TP tp;TI ti;GF gf;BF bf;FT ft;FC fc;UL ul;U1 u1;DA draw;w=CreateWindowA("STATIC","WineMetalGL depth sample",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");L(US,us,"glUseProgram");L(GT,gt,"glGenTextures");L(BT,bt,"glBindTexture");L(TP,tp,"glTexParameteri");L(TI,ti,"glTexImage2D");L(GF,gf,"glGenFramebuffers");L(BF,bf,"glBindFramebuffer");L(FT,ft,"glFramebufferTexture2D");L(FC,fc,"glCheckFramebufferStatus");L(UL,ul,"glGetUniformLocation");L(U1,u1,"glUniform1i");L(DA,draw,"glDrawArrays");v=cs(VS);f=cs(FS);{const char*x=vs;ss(v,1,&x,0);}{const char*x=fs;ss(f,1,&x,0);}co(v);co(f);si(v,COMPILE,&ok);if(!ok)return 15;si(f,COMPILE,&ok);if(!ok)return 16;pr=cp();at(pr,v);at(pr,f);li(pr);pi(pr,LINK,&ok);if(!ok)return 17;us(pr);gt(1,&t);bt(TEXTURE_2D,t);tp(TEXTURE_2D,0x2801,NEAREST);tp(TEXTURE_2D,0x2800,NEAREST);ti(TEXTURE_2D,0,DEPTH_COMPONENT24,2,2,0,DEPTH_COMPONENT,GL_FLOAT,0);gt(1,&color);bt(TEXTURE_2D,color);ti(TEXTURE_2D,0,RGBA8,2,2,0,RGBA,GL_UNSIGNED_BYTE,0);gf(1,&fbo);bf(FRAMEBUFFER,fbo);ft(FRAMEBUFFER,COLOR_ATTACHMENT0,TEXTURE_2D,color,0);ft(FRAMEBUFFER,DEPTH_ATTACHMENT,TEXTURE_2D,t,0);if(fc(FRAMEBUFFER)!=FRAMEBUFFER_COMPLETE)return 18;glClearDepth(.25);glClear(DEPTH_BUFFER_BIT);bf(FRAMEBUFFER,0);bt(TEXTURE_2D,t);GLint loc=ul(pr,"depthTex");if(loc<0)return 19;u1(loc,0);glViewport(0,0,64,64);draw(TRIANGLES,0,3);glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("Depth sample rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<62||px[0]>66||px[1]!=0||px[2]!=0||px[3]<250)return 20;printf("WINEMETALGL_DEPTH_TEXTURE_SAMPLE_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
