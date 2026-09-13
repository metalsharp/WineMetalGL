#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define VS 0x8b31
#define FS 0x8b30
#define COMPILE 0x8b81
#define LINK 0x8b82
typedef GLuint(WINAPI*CS)(GLenum);typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI*CO)(GLuint);typedef void(WINAPI*SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI*CP)(void);typedef void(WINAPI*AT)(GLuint,GLuint);typedef void(WINAPI*LI)(GLuint);typedef void(WINAPI*PI)(GLuint,GLenum,GLint*);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){static const char vs[]="#version 330 core\nout vec3 vColor;void main(){vColor=vec3(1);gl_Position=vec4(0);}";static const char fs[]="#version 330 core\nin vec4 vColor;out vec4 color;void main(){color=vColor;}";PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,f,pr;GLint ok;CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;w=CreateWindowA("STATIC","WineMetalGL interface",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");v=cs(VS);f=cs(FS);{const char*s=vs;ss(v,1,&s,0);}{const char*s=fs;ss(f,1,&s,0);}co(v);co(f);si(v,COMPILE,&ok);if(!ok)return 15;si(f,COMPILE,&ok);if(!ok)return 16;pr=cp();at(pr,v);at(pr,f);li(pr);pi(pr,LINK,&ok);if(ok)return 17;printf("WINEMETALGL_INTERFACE_REJECT_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
