#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>
#define SYNC_GPU 0x9117
#define ALREADY 0x911a
#define SATISFIED 0x911c
typedef void* GLsync; typedef GLsync (WINAPI *Fence)(GLenum,GLbitfield);typedef void(WINAPI*Get)(GLsync,GLenum,GLsizei,GLsizei*,GLint*);typedef GLenum(WINAPI*Wait)(GLsync,GLbitfield,uint64_t);typedef GLboolean(WINAPI*Is)(GLsync);typedef void(WINAPI*Del)(GLsync);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;Fence fence;Get get_sync;Wait wait;Is is_sync;Del del;GLsync sync;GLenum result;GLint value=0,length=0;w=CreateWindowA("STATIC","WineMetalGL sync",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(Fence,fence,"glFenceSync");L(Get,get_sync,"glGetSynciv");L(Wait,wait,"glClientWaitSync");L(Is,is_sync,"glIsSync");L(Del,del,"glDeleteSync");sync=fence(SYNC_GPU,0);if(!sync||!is_sync(sync))return 15;get_sync(sync,0x9112,1,&length,&value);if(glGetError()!=0||length!=1||value!=0x9116)return 18;get_sync(sync,0x9114,1,&length,&value);if(glGetError()!=0||length!=1||value!=0x9119)return 19;result=wait(sync,0,0);printf("Sync result=0x%x error=0x%x\n",result,(unsigned)glGetError());if(result!=ALREADY&&result!=SATISFIED)return 16;del(sync);if(is_sync(sync))return 17;printf("WINEMETALGL_SYNC_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
