#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define VS 0x8b31
#define FS 0x8b30
#define TESC 0x8e88
#define TESE 0x8e87
#define COMPILE 0x8b81
#define LINK 0x8b82
#define ARRAY_BUFFER 0x8892
#define STATIC_DRAW 0x88e4
#define FLOAT 0x1406
#define PATCHES 0x000e
#define PATCH_VERTICES 0x8e72
typedef GLuint(WINAPI*CS)(GLenum); typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*); typedef void(WINAPI*CO)(GLuint); typedef void(WINAPI*SI)(GLuint,GLenum,GLint*); typedef GLuint(WINAPI*CP)(void); typedef void(WINAPI*AT)(GLuint,GLuint); typedef void(WINAPI*LI)(GLuint); typedef void(WINAPI*PI)(GLuint,GLenum,GLint*); typedef void(WINAPI*US)(GLuint); typedef void(WINAPI*GB)(GLsizei,GLuint*); typedef void(WINAPI*BB)(GLenum,GLuint); typedef void(WINAPI*BD)(GLenum,intptr_t,const void*,GLenum); typedef void(WINAPI*EA)(GLuint); typedef void(WINAPI*VP)(GLuint,GLint,GLenum,GLboolean,GLsizei,const void*); typedef void(WINAPI*DP)(GLenum,GLint,GLint); typedef void(WINAPI*PP)(GLenum,GLint);
static void *p(const char *name) { PROC q=wglGetProcAddress(name); HMODULE m; if(q)return (void*)q; m=GetModuleHandleA("opengl32.dll"); return m?(void*)GetProcAddress(m,name):0; }
#define LOAD(t,v,n) do { v=(t)p(n); if(!v){printf("FAIL missing %s\n",n);return 14;} } while(0)
int main(void) {
    static const char vs[]="#version 450 core\nlayout(location=0)in vec4 position;void main(){gl_Position=position;}";
    static const char tc[]="#version 450 core\nlayout(vertices=4)out;void main(){gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;if(gl_InvocationID==0){gl_TessLevelInner[0]=1.0;gl_TessLevelInner[1]=1.0;gl_TessLevelOuter[0]=1.0;gl_TessLevelOuter[1]=1.0;gl_TessLevelOuter[2]=1.0;gl_TessLevelOuter[3]=1.0;}}";
    static const char te[]="#version 450 core\nlayout(quads,equal_spacing,cw)in;void main(){float u=gl_TessCoord.x,v=gl_TessCoord.y;gl_Position=(1-u)*(1-v)*gl_in[0].gl_Position+u*(1-v)*gl_in[1].gl_Position+u*v*gl_in[2].gl_Position+(1-u)*v*gl_in[3].gl_Position;}";
    static const char fs[]="#version 450 core\nout vec4 color;void main(){color=vec4(.2,.4,.6,1);}";
    static const float vertices[]={-1,-1,0,1,1,-1,0,1,1,1,0,1,-1,1,0,1};
    PIXELFORMATDESCRIPTOR d={0}; HWND w; HDC dc; HGLRC c; int pf; GLuint buf,pr,shaders[4]; GLint ok; unsigned char px[4]={0};
    CS cs; SS ss; CO co; SI si; CP cp; AT at; LI li; PI pi; US us; GB gb; BB bb; BD bd; EA ea; VP vp; DP draw; PP pp;
    w=CreateWindowA("STATIC","WineMetalGL tess quad",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0); if(!w)return 11; dc=GetDC(w); d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;
    LOAD(CS,cs,"glCreateShader"); LOAD(SS,ss,"glShaderSource"); LOAD(CO,co,"glCompileShader"); LOAD(SI,si,"glGetShaderiv"); LOAD(CP,cp,"glCreateProgram"); LOAD(AT,at,"glAttachShader"); LOAD(LI,li,"glLinkProgram"); LOAD(PI,pi,"glGetProgramiv"); LOAD(US,us,"glUseProgram"); LOAD(GB,gb,"glGenBuffers"); LOAD(BB,bb,"glBindBuffer"); LOAD(BD,bd,"glBufferData"); LOAD(EA,ea,"glEnableVertexAttribArray"); LOAD(VP,vp,"glVertexAttribPointer"); LOAD(DP,draw,"glDrawArrays"); LOAD(PP,pp,"glPatchParameteri");
    const char *src[4]={vs,tc,te,fs}; for(int i=0;i<4;++i){shaders[i]=cs((GLenum[]){VS,TESC,TESE,FS}[i]);ss(shaders[i],1,&src[i],0);co(shaders[i]);si(shaders[i],COMPILE,&ok);if(!ok)return 15;} pr=cp();for(int i=0;i<4;++i)at(pr,shaders[i]);li(pr);pi(pr,LINK,&ok);if(!ok)return 16;us(pr);gb(1,&buf);bb(ARRAY_BUFFER,buf);bd(ARRAY_BUFFER,sizeof(vertices),vertices,STATIC_DRAW);ea(0);vp(0,4,FLOAT,0,4*sizeof(float),(const void*)0);pp(PATCH_VERTICES,4);glViewport(0,0,64,64);draw(PATCHES,0,4);glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("Tess quad rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 17;printf("WINEMETALGL_TESSELLATION_QUAD_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;
}
