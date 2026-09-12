#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>
#define SYNC_GPU 0x9117
#define ALREADY 0x911a
#define SATISFIED 0x911c
typedef void* GLsync; typedef GLsync (WINAPI *Fence)(GLenum,GLbitfield);typedef GLenum(WINAPI*Wait)(GLsync,GLbitfield,uint64_t);typedef GLboolean(WINAPI*Is)(GLsync);typedef void(WINAPI*Del)(GLsync);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;Fence fence;Wait wait;Is is_sync;Del del;GLsync sync;GLenum result;w=CreateWindowA("STATIC","WineMetalGL sync",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(Fence,fence,"glFenceSync");L(Wait,wait,"glClientWaitSync");L(Is,is_sync,"glIsSync");L(Del,del,"glDeleteSync");sync=fence(SYNC_GPU,0);if(!sync||!is_sync(sync))return 15;result=wait(sync,0,0);printf("Sync result=0x%x error=0x%x\n",result,(unsigned)glGetError());if(result!=ALREADY&&result!=SATISFIED)return 16;del(sync);if(is_sync(sync))return 17;printf("WINEMETALGL_SYNC_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
