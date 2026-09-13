#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#define GL_CONSTANT_COLOR 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR 0x8002
typedef void (WINAPI *BC)(GLfloat,GLfloat,GLfloat,GLfloat);
int main(void) {
    PIXELFORMATDESCRIPTOR p={0}; HWND w; HDC dc; HGLRC c; int f; unsigned char px[4]={0}; BC blend_color;
    w=CreateWindowA("STATIC","WineMetalGL blend",WS_OVERLAPPEDWINDOW,0,0,64,64,0,0,0,0); if(!w)return 11;
    dc=GetDC(w); p.nSize=sizeof(p); p.nVersion=1; p.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL; p.iPixelType=PFD_TYPE_RGBA; p.cColorBits=32;
    f=ChoosePixelFormat(dc,&p); if(!f||!SetPixelFormat(dc,f,&p))return 12; c=wglCreateContext(dc); if(!c||!wglMakeCurrent(dc,c))return 13; blend_color=(BC)wglGetProcAddress("glBlendColor"); if(!blend_color)return 13;
    glViewport(0,0,64,64); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glColor4f(1,0,0,0.5f);
    glBegin(GL_TRIANGLES); glVertex3f(-1,-1,0); glVertex3f(3,-1,0); glVertex3f(-1,3,0); glEnd();
    glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
    printf("Blend rgba=%u,%u,%u,%u error=0x%x\n",px[0],px[1],px[2],px[3],(unsigned)glGetError());
    if(px[0]<115||px[0]>140||px[1]>8||px[2]>8||px[3]<180||px[3]>205)return 14;
    printf("WINEMETALGL_BLEND_OK\n"); glClearColor(0,0,1,1); glEnable(GL_BLEND); blend_color(.5f,.5f,.5f,1); glBlendFunc(GL_CONSTANT_COLOR,GL_ONE_MINUS_CONSTANT_COLOR); glColor4f(1,0,0,1); glBegin(GL_TRIANGLES); glVertex3f(-1,-1,0); glVertex3f(3,-1,0); glVertex3f(-1,3,0); glEnd(); glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px); printf("Constant blend rgba=%u,%u,%u,%u\n",px[0],px[1],px[2],px[3]); if(px[0]<115||px[0]>140||px[1]>8||px[2]<115||px[2]>140)return 15; printf("WINEMETALGL_BLEND_CONSTANT_OK\\n"); glDisable(GL_BLEND); glClearColor(0,0,1,1); glEnable(GL_SCISSOR_TEST); glScissor(0,0,16,16); glColor4f(0,1,0,1); glBegin(GL_TRIANGLES); glVertex3f(-1,-1,0); glVertex3f(3,-1,0); glVertex3f(-1,3,0); glEnd(); unsigned char inside[4]={0},outside[4]={0}; glReadPixels(8,8,1,1,GL_RGBA,GL_UNSIGNED_BYTE,inside); glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,outside); if(inside[1]<245||inside[0]>8||inside[2]>8||outside[2]<245||outside[0]>8||outside[1]>8)return 15; printf("WINEMETALGL_SCISSOR_OK\\n"); glDisable(GL_SCISSOR_TEST); glClearColor(0,0,0,1); glEnable(GL_CULL_FACE); glCullFace(GL_BACK); glFrontFace(GL_CCW); glColor4f(1,0,0,1); glBegin(GL_TRIANGLES); glVertex3f(-1,-1,0); glVertex3f(3,-1,0); glVertex3f(-1,3,0); glEnd(); glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px); if(px[0]<245||px[1]>8||px[2]>8)return 16; glCullFace(GL_FRONT); glBegin(GL_TRIANGLES); glVertex3f(-1,-1,0); glVertex3f(3,-1,0); glVertex3f(-1,3,0); glEnd(); glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px); if(px[0]>8||px[1]>8||px[2]>8)return 17; printf("WINEMETALGL_CULL_OK\\n"); glDisable(GL_CULL_FACE); SwapBuffers(dc); wglMakeCurrent(0,0); wglDeleteContext(c); ReleaseDC(w,dc); DestroyWindow(w); return 0;
}
