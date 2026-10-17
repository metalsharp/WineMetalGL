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

/* Logging (unix_log.c): into the core's log once it is loaded (category
 * "wine"), to stderr before. Levels: 0 error, 1 warning, 2 info, 3 trace;
 * wmgl_log_max is the highest the log keeps, so the macros skip building
 * messages it would drop. */
extern int wmgl_log_max;
void wmgl_log( int level, const char *format, ... ) __attribute__((format(printf, 2, 3)));
/* Binds the log to the core (NULL: none) and names the program and host. */
void wmgl_log_start( void *core, const char *program, const char *host_version );
/* The highest level the PE side's messages ("pe") are kept at. */
int wmgl_pe_log_level( void );
#define WMGL_ERR(...) wmgl_log( 0, __VA_ARGS__ )
#define WMGL_WARN(...) do { if (wmgl_log_max >= 1) wmgl_log( 1, __VA_ARGS__ ); } while (0)
#define WMGL_INFO(...) do { if (wmgl_log_max >= 2) wmgl_log( 2, __VA_ARGS__ ); } while (0)
#define WMGL_TRACE(...) do { if (wmgl_log_max >= 3) wmgl_log( 3, __VA_ARGS__ ); } while (0)

#endif
