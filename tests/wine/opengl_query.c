#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
typedef void(WINAPI*Gen)(GLsizei,GLuint*);typedef void(WINAPI*Begin)(GLenum,GLuint);typedef void(WINAPI*End)(GLenum);typedef void(WINAPI*Get)(GLuint,GLenum,GLuint*);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint id=0,value=0;Gen gen;Begin begin;End end;Get get;w=CreateWindowA("STATIC","WineMetalGL query",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(Gen,gen,"glGenQueries");L(Begin,begin,"glBeginQuery");L(End,end,"glEndQuery");L(Get,get,"glGetQueryObjectuiv");gen(1,&id);begin(0x8914,id);glClear(GL_COLOR_BUFFER_BIT);end(0x8914);get(id,0x8867,&value);printf("Query samples=%u error=0x%x\n",value,(unsigned)glGetError());if(value!=1)return 15;printf("WINEMETALGL_QUERY_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
