#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>

#define VS 0x8b31
#define FS 0x8b30
#define COMPILE 0x8b81
#define LINK 0x8b82
#define TEXTURE_2D 0x0de1
#define RGBA8UI 0x8d7c
#define RGBA_INTEGER 0x8d99
#define UNSIGNED_BYTE 0x1401
#define NEAREST 0x2600
#define TRIANGLES 4

typedef GLuint(WINAPI *CS)(GLenum); typedef void(WINAPI *SS)(GLuint,GLsizei,const char*const*,const GLint*);
typedef void(WINAPI *CO)(GLuint); typedef void(WINAPI *SI)(GLuint,GLenum,GLint*); typedef GLuint(WINAPI *CP)(void);
typedef void(WINAPI *AT)(GLuint,GLuint); typedef void(WINAPI *LI)(GLuint); typedef void(WINAPI *PI)(GLuint,GLenum,GLint*);
typedef void(WINAPI *US)(GLuint); typedef void(WINAPI *GT)(GLsizei,GLuint*); typedef void(WINAPI *BT)(GLenum,GLuint);
typedef void(WINAPI *TP)(GLenum,GLenum,GLint); typedef void(WINAPI *TI)(GLenum,GLint,GLint,GLsizei,GLsizei,GLint,GLenum,GLenum,const void*);
typedef GLint(WINAPI *UL)(GLuint,const char*); typedef void(WINAPI *U1)(GLint,GLint); typedef void(WINAPI *DA)(GLenum,GLint,GLint);
static void *proc(const char *name){PROC p=wglGetProcAddress(name);HMODULE m=GetModuleHandleA("opengl32.dll");return p?(void*)p:(m?(void*)GetProcAddress(m,name):0);}
#define LOAD(t,v,n) do{v=(t)proc(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){
    static const char vs[]="#version 330 core\nconst vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));void main(){gl_Position=vec4(p[gl_VertexID],0,1);} ";
    static const char fs[]="#version 330 core\nuniform usampler2D tex;out vec4 color;void main(){uvec4 v=texture(tex,vec2(.5));color=vec4(v)/255.0;}";
    static const unsigned char data[]={1,2,3,255,1,2,3,255,1,2,3,255,1,2,3,255};
    PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,f,pr,t;GLint ok;unsigned char pixel[4]={0};
    CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;GT gt;BT bt;TP tp;TI ti;UL ul;U1 u1;DA draw;
    w=CreateWindowA("STATIC","WineMetalGL integer texture",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;
    LOAD(CS,cs,"glCreateShader");LOAD(SS,ss,"glShaderSource");LOAD(CO,co,"glCompileShader");LOAD(SI,si,"glGetShaderiv");LOAD(CP,cp,"glCreateProgram");LOAD(AT,at,"glAttachShader");LOAD(LI,li,"glLinkProgram");LOAD(PI,pi,"glGetProgramiv");LOAD(US,us,"glUseProgram");LOAD(GT,gt,"glGenTextures");LOAD(BT,bt,"glBindTexture");LOAD(TP,tp,"glTexParameteri");LOAD(TI,ti,"glTexImage2D");LOAD(UL,ul,"glGetUniformLocation");LOAD(U1,u1,"glUniform1i");LOAD(DA,draw,"glDrawArrays");
    v=cs(VS);f=cs(FS);{const char*p=vs;ss(v,1,&p,0);}{const char*p=fs;ss(f,1,&p,0);}co(v);co(f);si(v,COMPILE,&ok);if(!ok)return 15;si(f,COMPILE,&ok);if(!ok)return 16;pr=cp();at(pr,v);at(pr,f);li(pr);pi(pr,LINK,&ok);if(!ok)return 17;us(pr);
    gt(1,&t);bt(TEXTURE_2D,t);tp(TEXTURE_2D,0x2801,NEAREST);tp(TEXTURE_2D,0x2800,NEAREST);ti(TEXTURE_2D,0,RGBA8UI,2,2,0,RGBA_INTEGER,UNSIGNED_BYTE,data);GLint loc=ul(pr,"tex");if(loc<0)return 18;u1(loc,0);glViewport(0,0,64,64);draw(TRIANGLES,0,3);glReadPixels(32,32,1,1,GL_RGBA,UNSIGNED_BYTE,pixel);printf("Integer texture rgba=%u,%u,%u,%u error=0x%x\n",pixel[0],pixel[1],pixel[2],pixel[3],(unsigned)glGetError());if(pixel[0]!=1||pixel[1]!=2||pixel[2]!=3||pixel[3]<250)return 19;printf("WINEMETALGL_INTEGER_TEXTURE_SAMPLE_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;
}
