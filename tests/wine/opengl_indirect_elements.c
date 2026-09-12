#include <windows.h>
#include <GL/gl.h>
#include <stdint.h>
#include <stdio.h>
#define VS 0x8b31
#define FS 0x8b30
#define COMPILE 0x8b81
#define LINK 0x8b82
#define ARRAY_BUFFER 0x8892
#define ELEMENT_ARRAY_BUFFER 0x8893
#define DRAW_INDIRECT_BUFFER 0x8f3f
#define STATIC_DRAW 0x88e4
#define TRIANGLES 4
#define FLOAT 0x1406
#define UNSIGNED_SHORT 0x1403
typedef GLuint(WINAPI*CS)(GLenum);typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI*CO)(GLuint);typedef void(WINAPI*SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI*CP)(void);typedef void(WINAPI*AT)(GLuint,GLuint);typedef void(WINAPI*LI)(GLuint);typedef void(WINAPI*PI)(GLuint,GLenum,GLint*);typedef void(WINAPI*US)(GLuint);typedef void(WINAPI*GB)(GLsizei,GLuint*);typedef void(WINAPI*BB)(GLenum,GLuint);typedef void(WINAPI*BD)(GLenum,intptr_t,const void*,GLenum);typedef void(WINAPI*MDI)(GLenum,GLenum,const void*,GLsizei,GLsizei);typedef void(WINAPI*VP)(GLuint,GLint,GLenum,GLboolean,GLsizei,const void*);typedef void(WINAPI*EA)(GLuint);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
struct Cmd{uint32_t count,instances,firstIndex,baseVertex,baseInstance;};
int main(void){static const char vs[]="#version 330 core\nlayout(location=0)in vec2 position;void main(){gl_Position=vec4(position,0,1);}";static const char fs[]="#version 330 core\nout vec4 c;void main(){c=vec4(.2,.4,.6,1);}";static const float vertices[]={-1,-1,3,-1,-1,3};static const uint16_t indices[]={0,1,2};static const struct Cmd cmds[2]={{3,1,0,0,0},{3,1,0,0,0}};PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,f,pr,vbo,ibo,indirect;GLint ok;unsigned char px[4]={0};CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;GB gb;BB bb;BD bd;MDI mdi;VP vp;EA ea;w=CreateWindowA("STATIC","WineMetalGL elements indirect",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");L(US,us,"glUseProgram");L(GB,gb,"glGenBuffers");L(BB,bb,"glBindBuffer");L(BD,bd,"glBufferData");L(MDI,mdi,"glMultiDrawElementsIndirect");L(VP,vp,"glVertexAttribPointer");L(EA,ea,"glEnableVertexAttribArray");v=cs(VS);f=cs(FS);{const char*s=vs;ss(v,1,&s,0);}{const char*s=fs;ss(f,1,&s,0);}co(v);co(f);si(v,COMPILE,&ok);if(!ok)return 15;si(f,COMPILE,&ok);if(!ok)return 16;pr=cp();at(pr,v);at(pr,f);li(pr);pi(pr,LINK,&ok);if(!ok)return 17;us(pr);gb(1,&vbo);bb(ARRAY_BUFFER,vbo);bd(ARRAY_BUFFER,sizeof(vertices),vertices,STATIC_DRAW);ea(0);vp(0,2,FLOAT,0,2*sizeof(float),(const void*)0);gb(1,&ibo);bb(ELEMENT_ARRAY_BUFFER,ibo);bd(ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,STATIC_DRAW);gb(1,&indirect);bb(DRAW_INDIRECT_BUFFER,indirect);bd(DRAW_INDIRECT_BUFFER,sizeof(cmds),cmds,STATIC_DRAW);glViewport(0,0,64,64);mdi(TRIANGLES,UNSIGNED_SHORT,0,2,0);glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("Elements indirect rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 18;printf("WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
