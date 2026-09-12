#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
int main(void) {
    PIXELFORMATDESCRIPTOR p={0}; HWND w; HDC dc; HGLRC first, second; int pf;
    w=CreateWindowA("STATIC","WineMetalGL contexts",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0); if(!w)return 11;
    dc=GetDC(w); p.nSize=sizeof(p); p.nVersion=1; p.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL; p.iPixelType=PFD_TYPE_RGBA; p.cColorBits=32;
    pf=ChoosePixelFormat(dc,&p); if(!pf||!SetPixelFormat(dc,pf,&p))return 12;
    first=wglCreateContext(dc); second=wglCreateContext(dc); if(!first||!second)return 13;
    if(!wglMakeCurrent(dc,first)||wglGetCurrentContext()!=first||wglGetCurrentDC()!=dc)return 14;
    if(!wglMakeCurrent(dc,second)||wglGetCurrentContext()!=second||wglGetCurrentDC()!=dc)return 15;
    if(!wglMakeCurrent(NULL,NULL)||wglGetCurrentContext()!=NULL)return 16;
    printf("WINEMETALGL_WGL_MULTI_CONTEXT_OK\n");
    wglDeleteContext(first); wglDeleteContext(second); ReleaseDC(w,dc); DestroyWindow(w); return 0;
}
