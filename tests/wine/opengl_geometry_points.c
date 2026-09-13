#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define VS 0x8b31
#define FS 0x8b30
#define GS 0x8dd9
#define COMPILE 0x8b81
#define LINK 0x8b82
#define POINTS 0
#define TRIANGLES 4

typedef GLuint(WINAPI*CS)(GLenum);typedef void(WINAPI*SS)(GLuint,GLsizei,const char*const*,const GLint*);typedef void(WINAPI*CO)(GLuint);typedef void(WINAPI*SI)(GLuint,GLenum,GLint*);typedef GLuint(WINAPI*CP)(void);typedef void(WINAPI*AT)(GLuint,GLuint);typedef void(WINAPI*LI)(GLuint);typedef void(WINAPI*PI)(GLuint,GLenum,GLint*);typedef void(WINAPI*US)(GLuint);typedef void(WINAPI*DA)(GLenum,GLint,GLint);
static void*p(const char*n){PROC q=wglGetProcAddress(n);HMODULE m;if(q)return(void*)q;m=GetModuleHandleA("opengl32.dll");return m?(void*)GetProcAddress(m,n):0;}
#define L(t,v,n)do{v=(t)p(n);if(!v){printf("FAIL missing %s\n",n);return 14;}}while(0)
int main(void){static const char vs[]="#version 330 core\nconst vec2 p[1]=vec2[1](vec2(0,0));void main(){gl_PointSize=8.0;gl_Position=vec4(p[gl_VertexID],0,1);}";static const char gs[]="#version 330 core\nlayout(points)in;layout(points,max_vertices=1)out;void main(){gl_Position=gl_in[0].gl_Position;EmitVertex();EndPrimitive();}";static const char fs[]="#version 330 core\nout vec4 c;void main(){c=vec4(.2,.4,.6,1);}";PIXELFORMATDESCRIPTOR d={0};HWND w;HDC dc;HGLRC c;int pf;GLuint v,g,f,pr;GLint ok;unsigned char px[4]={0};CS cs;SS ss;CO co;SI si;CP cp;AT at;LI li;PI pi;US us;DA draw;w=CreateWindowA("STATIC","WineMetalGL geometry",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0);if(!w)return 11;dc=GetDC(w);d.nSize=sizeof(d);d.nVersion=1;d.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;d.iPixelType=PFD_TYPE_RGBA;d.cColorBits=32;pf=ChoosePixelFormat(dc,&d);if(!pf||!SetPixelFormat(dc,pf,&d))return 12;c=wglCreateContext(dc);if(!c||!wglMakeCurrent(dc,c))return 13;L(CS,cs,"glCreateShader");L(SS,ss,"glShaderSource");L(CO,co,"glCompileShader");L(SI,si,"glGetShaderiv");L(CP,cp,"glCreateProgram");L(AT,at,"glAttachShader");L(LI,li,"glLinkProgram");L(PI,pi,"glGetProgramiv");L(US,us,"glUseProgram");L(DA,draw,"glDrawArrays");v=cs(VS);g=cs(GS);f=cs(FS);const char*src[3]={vs,gs,fs};GLuint sh[3]={v,g,f};for(int i=0;i<3;++i){ss(sh[i],1,&src[i],0);co(sh[i]);si(sh[i],COMPILE,&ok);printf("shader%d id=%u status=%d\\n",i,sh[i],ok);if(!ok)return 15;}pr=cp();at(pr,v);at(pr,g);at(pr,f);li(pr);pi(pr,LINK,&ok);if(!ok)return 16;us(pr);glViewport(0,0,64,64);draw(POINTS,0,1);glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);printf("Geometry rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());if(px[0]<47||px[0]>55||px[1]<98||px[1]>106||px[2]<149||px[2]>157||px[3]<250)return 17;printf("WINEMETALGL_GEOMETRY_PASSTHROUGH_OK\n");printf("WINEMETALGL_GEOMETRY_POINT_PASSTHROUGH_OK\n");wglMakeCurrent(0,0);wglDeleteContext(c);ReleaseDC(w,dc);DestroyWindow(w);return 0;}
