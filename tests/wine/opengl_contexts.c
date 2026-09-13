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
    glViewport(1,2,10,11);
    if(!wglMakeCurrent(dc,second)||wglGetCurrentContext()!=second||wglGetCurrentDC()!=dc)return 15;
    glViewport(3,4,20,21);
    if(!wglMakeCurrent(dc,first)||wglGetCurrentContext()!=first||wglGetCurrentDC()!=dc)return 16;
    GLint viewport[4]={0};glGetIntegerv(0x0BA2,viewport);if(viewport[0]!=1||viewport[1]!=2||viewport[2]!=10||viewport[3]!=11)return 17;
    if(!wglMakeCurrent(NULL,NULL)||wglGetCurrentContext()!=NULL)return 18;
    printf("WINEMETALGL_WGL_MULTI_CONTEXT_OK\n");
    printf("WINEMETALGL_WGL_CONTEXT_STATE_OK\n");
    wglDeleteContext(first); wglDeleteContext(second); ReleaseDC(w,dc); DestroyWindow(w); return 0;
}
