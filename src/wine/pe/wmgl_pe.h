/* winemetalgl.dll: the PE side of the WineMetalGL Wine module (wmgl_wine.h). */
#ifndef WMGL_PE_H
#define WMGL_PE_H

#include <stddef.h>

/* Windows 10 API level (SetThreadDpiAwarenessContext), before any header. */
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#undef WINVER
#define WINVER 0x0A00

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#define _GDI32_  /* wingdi.h: this module defines the wgl functions it declares (no dllimport) */
#include <windows.h>
#include <winternl.h>

NTSTATUS NTAPI NtQueryVirtualMemory( HANDLE process, const void *address, ULONG info_class, void *buffer,
                                     SIZE_T length, SIZE_T *result );

#include "wmgl_wine.h"
#include <GL/wglext.h>

/* WGL_WINE_query_renderer (Wine's wgl.h). */
#define WGL_RENDERER_VENDOR_ID_WINE 0x8183
#define WGL_RENDERER_DEVICE_ID_WINE 0x8184
#define WGL_RENDERER_VERSION_WINE 0x8185
#define WGL_RENDERER_ACCELERATED_WINE 0x8186
#define WGL_RENDERER_VIDEO_MEMORY_WINE 0x8187
#define WGL_RENDERER_UNIFIED_MEMORY_ARCHITECTURE_WINE 0x8188
#define WGL_RENDERER_PREFERRED_PROFILE_WINE 0x8189
#define WGL_RENDERER_OPENGL_CORE_PROFILE_VERSION_WINE 0x818A
#define WGL_RENDERER_OPENGL_COMPATIBILITY_PROFILE_VERSION_WINE 0x818B
#define WGL_RENDERER_OPENGL_ES_PROFILE_VERSION_WINE 0x818C
#define WGL_RENDERER_OPENGL_ES2_PROFILE_VERSION_WINE 0x818D

#ifndef WGL_SWAP_UNDEFINED_ARB
#define WGL_SWAP_UNDEFINED_ARB 0x2063
#endif
#define GL_FALSE 0
#define GL_TRUE 1
#define GL_VERSION 0x1F02
#define GL_EXTENSIONS 0x1F03
#define GL_NUM_EXTENSIONS 0x821D
#define GL_COMPILE 0x1300
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNPACK_ALIGNMENT 0x0CF5
#define GL_UNPACK_ROW_LENGTH 0x0CF2

typedef UINT64 unixlib_handle_t;

/* A unix call (wmgl_wine.h); debug messages it queued reach the
 * application's callback before it returns. */
NTSTATUS wmgl_unix_call( unsigned int code, void *args );
/* Appends a GL call to the thread's batch (wine_gen.py: commands that
 * return nothing and take values), with `data_size` bytes of `data` that the
 * parameter struct points to at `pointer_offset`. FALSE: make the call now. */
BOOL wmgl_batch( unsigned int code, const void *args, size_t size, const void *data, UINT64 data_size,
                 size_t pointer_offset );
#define WMGL_CALL(name, args) wmgl_unix_call( unix_##name, (args) )

struct wmgl_proc
{
    const char *name;
    PROC proc;
    const char *source;  /* "1.0".."4.6", or the extension that provides it */
};
extern const struct wmgl_proc wmgl_procs[];
extern const unsigned int wmgl_proc_count;

/* The pixel formats (wgl_formats.c), fetched once. */
struct wgl_pixel_format
{
    PIXELFORMATDESCRIPTOR pfd;
    int swap_method;
    int transparent;
    int pixel_type;
    int draw_to_pbuffer;
    int max_pbuffer_pixels;
    int max_pbuffer_width;
    int max_pbuffer_height;
    int transparent_red_value;
    int transparent_red_value_valid;
    int transparent_green_value;
    int transparent_green_value_valid;
    int transparent_blue_value;
    int transparent_blue_value_valid;
    int transparent_alpha_value;
    int transparent_alpha_value_valid;
    int transparent_index_value;
    int transparent_index_value_valid;
    int sample_buffers;
    int samples;
    int bind_to_texture_rgb;
    int bind_to_texture_rgba;
    int bind_to_texture_rectangle_rgb;
    int bind_to_texture_rectangle_rgba;
    int framebuffer_srgb_capable;
    int float_components;
};
void wmgl_init_pixel_formats( const struct wmgl_pixel_format *formats, UINT count, UINT onscreen );
INT WINAPI wglDescribePixelFormat( HDC hdc, int index, UINT size, PIXELFORMATDESCRIPTOR *ppfd );
BOOL WINAPI wglChoosePixelFormatARB( HDC hdc, const int *attribs_int, const FLOAT *attribs_float, UINT max_formats,
                                     int *formats, UINT *num_formats );
BOOL WINAPI wglGetPixelFormatAttribivARB( HDC hdc, int index, int plane, UINT count, const int *attributes, int *values );
BOOL WINAPI wglGetPixelFormatAttribfvARB( HDC hdc, int index, int plane, UINT count, const int *attributes,
                                          FLOAT *values );

/* GL entry points of this module the WGL code calls. */
void WINAPI glGetIntegerv( GLenum pname, GLint *data );
const GLubyte *WINAPI glGetString( GLenum name );
const GLubyte *WINAPI glGetStringi( GLenum name, GLuint index );
GLenum WINAPI glGetError( void );
void WINAPI glNewList( GLuint list, GLenum mode );
void WINAPI glEndList( void );
void WINAPI glBitmap( GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove,
                      const GLubyte *bitmap );
void WINAPI glPixelStorei( GLenum pname, GLint param );

#define WMGL_TRACE(...) do { if (wmgl_debug) { fprintf( stderr, "winemetalgl: " __VA_ARGS__ ); } } while (0)
extern BOOL wmgl_debug;

#endif
