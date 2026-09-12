#include <windows.h>
#include <GL/gl.h>
#include <stdint.h>
#include <stdio.h>
#define TF 0x8c8e
#define STATIC_DRAW 0x88e4
#define TRIANGLES 4
typedef void(WINAPI*Gen)(GLsizei,GLuint*);typedef void(WINAPI*Bind)(GLenum,GLuint);typedef void(WINAPI*Data)(GLenum,intptr_t,const void*,GLenum);typedef void(WINAPI*Base)(GLenum,GLuint,GLuint);typedef void(WINAPI*BeginTF)(GLenum);typedef void(WINAPI*EndTF)(void);typedef void(WINAPI*Get)(GLenum,intptr_t,intptr_t,void*);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;Gen gen;Bind bind;Data data;Base base;BeginTF begin;EndTF end;Get get;GLuint b;float out[27]={0};w=CreateWindowA("STATIC","WineMetalGL transform",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(Gen,gen,"glGenBuffers");L(Bind,bind,"glBindBuffer");L(Data,data,"glBufferData");L(Base,base,"glBindBufferBase");L(BeginTF,begin,"glBeginTransformFeedback");L(EndTF,end,"glEndTransformFeedback");L(Get,get,"glGetBufferSubData");gen(1,&b);bind(TF,b);data(TF,sizeof(out),0,STATIC_DRAW);base(TF,0,b);begin(TRIANGLES);glBegin(TRIANGLES);glColor4f(1,0,0,1);glVertex3f(-1,-1,0);glVertex3f(3,-1,0);glVertex3f(-1,3,0);glEnd();end();get(TF,0,sizeof(out),out);printf("TF position=%g,%g,%g color=%g,%g,%g,%g error=0x%x\n",out[0],out[1],out[2],out[3],out[4],out[5],out[6],(unsigned)glGetError());if(out[0] < -1.1f || out[3] < .9f)return 15;printf("WINEMETALGL_TRANSFORM_FIXED_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
