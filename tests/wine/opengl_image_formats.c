#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>
#define COMPUTE_SHADER 0x91B9
#define COMPILE_STATUS 0x8B81
#define LINK_STATUS 0x8B82
#define TEXTURE_2D 0x0DE1
#define R32UI 0x8236
#define RED_INTEGER 0x8D94
#define UNSIGNED_INT 0x1405
#define IMAGE_2D 0x904D
#define WRITE_ONLY 0x88B9
typedef GLuint(WINAPI*CS)(GLenum);typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI*CO)(GLuint);typedef void(WINAPI*SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI*CP)(void);typedef void(WINAPI*AT)(GLuint,GLuint);typedef void(WINAPI*LI)(GLuint);typedef void(WINAPI*PI)(GLuint,GLenum,GLint*);typedef void(WINAPI*US)(GLuint);typedef void(WINAPI*GT)(GLsizei,GLuint*);typedef void(WINAPI*BT)(GLenum,GLuint);typedef void(WINAPI*TI)(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*);typedef void(WINAPI*GI)(GLuint,GLuint,GLint,GLboolean,GLint,GLenum,GLenum);typedef void(WINAPI*DC)(GLuint,GLuint,GLuint);typedef void(WINAPI*GX)(GLenum,GLint,GLenum,GLenum,void*);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){static const char src[]="#version 430 core\nlayout(local_size_x=1) in;layout(r32ui,binding=0) uniform uimage2D img;void main(){imageStore(img,ivec2(0),uvec4(305419896u,0,0,1));}";PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint s,pr,t;GLint ok;uint32_t value=0;CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;GT gt;BT bt;TI ti;GI gi;DC dispatch;GX get_image;w=CreateWindowA("STATIC","WineMetalGL image formats",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");L(US,us,"glUseProgram");L(GT,gt,"glGenTextures");L(BT,bt,"glBindTexture");L(TI,ti,"glTexImage2D");L(GI,gi,"glBindImageTexture");L(DC,dispatch,"glDispatchCompute");L(GX,get_image,"glGetTexImage");s=cs(COMPUTE_SHADER);{const char*x=src;ss(s,1,&x,0);}co(s);si(s,COMPILE_STATUS,&ok);if(!ok)return 15;pr=cp();at(pr,s);li(pr);pi(pr,LINK_STATUS,&ok);if(!ok)return 16;us(pr);gt(1,&t);bt(TEXTURE_2D,t);ti(TEXTURE_2D,0,R32UI,2,2,0,RED_INTEGER,UNSIGNED_INT,0);gi(0,t,0,0,0,WRITE_ONLY,R32UI);dispatch(1,1,1);get_image(TEXTURE_2D,0,RED_INTEGER,UNSIGNED_INT,&value);printf("R32UI value=0x%08x error=0x%x\n",value,(unsigned)glGetError());if(value!=0x12345678u)return 17;printf("WINEMETALGL_IMAGE_R32UI_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
