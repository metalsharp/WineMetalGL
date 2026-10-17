/* winemetalgl.so: the unix side of the WineMetalGL Wine module (wmgl_wine.h). */
#ifndef WMGL_UNIX_H
#define WMGL_UNIX_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "ntstatus.h"
#define WIN32_NO_STATUS
#include "windef.h"
#include "winbase.h"
#include "winternl.h"
#include "wingdi.h"
#include "wine/unixlib.h"

#include "wmgl_wine.h"
#include "wine_gl_procs.gen.h"
#include "wmgl_window.h"

/* The core (include/winemetalgl/driver.h). */
struct wmgl_core
{
    int (*initialize)( unsigned int abi_version );
    unsigned int (*get_pixel_formats)( struct wmgl_pixel_format *formats, unsigned int max, unsigned int *onscreen );
    void *(*surface_create_window)( void *layer, int format, int width, int height );
    void *(*surface_create_pbuffer)( int format, int width, int height );
    void (*surface_destroy)( void *surface );
    void (*surface_flush)( void *surface, unsigned int flags, int width, int height, int interval );
    int (*surface_swap)( void *surface );
    void *(*context_create)( int format, void *share, const int *attribs, int *error );
    void (*context_destroy)( void *context );
    int (*make_current)( void *context, void *draw, void *read );
    void *(*get_proc_address)( const char *name );
    const char *(*renderer_string)( void );
    unsigned int (*video_memory_mb)( void );
    void (*max_versions)( int *core_major, int *core_minor, int *compat_major, int *compat_minor );
    void (*version)( int *major, int *minor, int *patch );  /* optional */
};
extern struct wmgl_core wmgl;

#define WMGL_DRIVER_ABI_VERSION 1
#define WMGL_FLUSH_INTERVAL 0x02
#define WMGL_FLUSH_UPDATED 0x04

/* An optional symbol of the core, or NULL. */
void *wmgl_core_symbol( const char *name );
/* A pointer for a 32-bit caller (0, logged, when beyond its reach). */
UINT wmgl_wow64_pointer( const void *pointer );

/* The status a GL unix call returns: STATUS_PENDING when this thread has
 * debug messages for the PE side to deliver. */
NTSTATUS wmgl_after_call( void );

/* The current context's debug callback (unix_main.c). */
/* A context's debug callbacks (client pointers, 0: none): glDebugMessageCallback's
 * and glDebugMessageCallbackAMD's. */
struct wmgl_debug_target
{
    UINT64 callback, user;
    UINT64 callback_amd, user_amd;
};
struct wmgl_debug_target *wmgl_current_debug_target( void );
void wmgl_set_debug_callback( UINT64 callback, UINT64 user, BOOL amd );

#define WMGL_ERR(...) fprintf( stderr, "winemetalgl: " __VA_ARGS__ )
extern BOOL wmgl_trace;
#define WMGL_TRACE(...) do { if (wmgl_trace) fprintf( stderr, "winemetalgl: " __VA_ARGS__ ); } while (0)

#endif
