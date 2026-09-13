#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
typedef struct {HDC dc;HGLRC context;int x;int ok;} ThreadContext;
typedef BOOL (WINAPI *SwapInterval)(int);
static DWORD WINAPI context_thread(void *opaque){ThreadContext *ctx=(ThreadContext*)opaque;GLint viewport[4]={0};ctx->ok=wglMakeCurrent(ctx->dc,ctx->context);glViewport(ctx->x,0,8,8);glGetIntegerv(0x0BA2,viewport);ctx->ok=ctx->ok&&viewport[0]==ctx->x&&wglGetCurrentContext()==ctx->context;wglMakeCurrent(NULL,NULL);return 0;}
int main(void) {
    PIXELFORMATDESCRIPTOR p={0}; HWND w; HDC dc; HGLRC first, second; int pf;
    w=CreateWindowA("STATIC","WineMetalGL contexts",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0); if(!w)return 11;
    dc=GetDC(w); p.nSize=sizeof(p); p.nVersion=1; p.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL; p.iPixelType=PFD_TYPE_RGBA; p.cColorBits=32;
    pf=ChoosePixelFormat(dc,&p); if(!pf||!SetPixelFormat(dc,pf,&p))return 12;
    first=wglCreateContext(dc); second=wglCreateContext(dc); if(!first||!second)return 13;
    if(!wglMakeCurrent(dc,first)||wglGetCurrentContext()!=first||wglGetCurrentDC()!=dc)return 14;
    SwapInterval swap_interval=(SwapInterval)wglGetProcAddress("wglSwapIntervalEXT");if(!swap_interval||!swap_interval(0)||!swap_interval(1))return 14;
    printf("WINEMETALGL_WGL_SWAP_INTERVAL_OK\n");
    glViewport(1,2,10,11);
    if(!wglMakeCurrent(dc,second)||wglGetCurrentContext()!=second||wglGetCurrentDC()!=dc)return 15;
    glViewport(3,4,20,21);
    if(!wglMakeCurrent(dc,first)||wglGetCurrentContext()!=first||wglGetCurrentDC()!=dc)return 16;
    GLint viewport[4]={0};glGetIntegerv(0x0BA2,viewport);if(viewport[0]!=1||viewport[1]!=2||viewport[2]!=10||viewport[3]!=11)return 17;
    if(!wglMakeCurrent(NULL,NULL)||wglGetCurrentContext()!=NULL)return 18;
    ThreadContext thread_a={dc,first,7,0},thread_b={dc,second,19,0};HANDLE a=CreateThread(NULL,0,context_thread,&thread_a,0,NULL),b=CreateThread(NULL,0,context_thread,&thread_b,0,NULL);if(!a||!b)return 19;WaitForSingleObject(a,INFINITE);WaitForSingleObject(b,INFINITE);CloseHandle(a);CloseHandle(b);if(!thread_a.ok||!thread_b.ok)return 20;
    printf("WINEMETALGL_WGL_MULTI_CONTEXT_OK\n");
    printf("WINEMETALGL_WGL_CONTEXT_THREADS_OK\n");
    printf("WINEMETALGL_WGL_CONTEXT_STATE_OK\n");
    wglDeleteContext(first); wglDeleteContext(second); ReleaseDC(w,dc); DestroyWindow(w); return 0;
}
