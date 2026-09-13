#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
int main(void){PIXELFORMATDESCRIPTOR p={0};HWND w=CreateWindowA("STATIC","WineMetalGL share",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);HDC d=GetDC(w);p.nSize=sizeof(p);p.nVersion=1;p.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;p.iPixelType=PFD_TYPE_RGBA;p.cColorBits=32;int f=ChoosePixelFormat(d,&p);if(!f||!SetPixelFormat(d,f,&p))return 12;HGLRC a=wglCreateContext(d),b=wglCreateContext(d);if(!a||!b)return 13;if(!wglShareLists(a,b))return 14;printf("WINEMETALGL_WGL_SHARE_LISTS_OK\n");wglDeleteContext(a);wglDeleteContext(b);ReleaseDC(w,d);DestroyWindow(w);return 0;}
