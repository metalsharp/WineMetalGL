/* WGL of winemetalgl.so: contexts, pixel formats, drawables and swaps on the
 * WineMetalGL core (include/winemetalgl/driver.h).
 *
 * - Pixel formats belong to windows (by HWND) or, for other DCs, to DCs.
 * - A window's drawable is made when a context first becomes current on one
 *   of its DCs: a core window surface on the layer unix_window.m provides.
 *   The PE side describes the window on every make-current and swap
 *   (wmgl_window_placement); a new size resizes the surface.
 * - Every context shares objects with a hidden root context, so
 *   wglShareLists always succeeds (as win32u arranges for its drivers).
 * - Handles are small numbers, valid for 32-bit callers too. */

#include "wmgl_unix.h"

#define HANDLE_BASE 0x10000

/* WGL constants (wglext.h; WGL_WINE_query_renderer from Wine's wgl.h). */
#define WGL_TEXTURE_FORMAT_ARB 0x2072
#define WGL_TEXTURE_TARGET_ARB 0x2073
#define WGL_MIPMAP_TEXTURE_ARB 0x2074
#define WGL_TEXTURE_RGB_ARB 0x2075
#define WGL_TEXTURE_RGBA_ARB 0x2076
#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB 0x2078
#define WGL_TEXTURE_1D_ARB 0x2079
#define WGL_TEXTURE_2D_ARB 0x207A
#define WGL_MIPMAP_LEVEL_ARB 0x207B
#define WGL_CUBE_MAP_FACE_ARB 0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_FRONT_LEFT_ARB 0x2083
#define WGL_BACK_LEFT_ARB 0x2085
#define WGL_PBUFFER_LARGEST_ARB 0x2033
#define WGL_PBUFFER_WIDTH_ARB 0x2034
#define WGL_PBUFFER_HEIGHT_ARB 0x2035
#define WGL_PBUFFER_LOST_ARB 0x2036
#define WGL_TEXTURE_RECTANGLE_NV 0x20A2
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x0001
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
#define ERROR_INVALID_VERSION_ARB 0x2095
#define ERROR_INVALID_PROFILE_ARB 0x2096

#define GL_TEXTURE_1D 0x0DE0
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE_RECTANGLE 0x84F5
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GL_RGB 0x1907
#define GL_RGBA 0x1908

static const char wgl_extensions[] =
    "WGL_ARB_context_flush_control WGL_ARB_create_context WGL_ARB_create_context_no_error "
    "WGL_ARB_create_context_profile WGL_ARB_create_context_robustness WGL_ARB_extensions_string "
    "WGL_ARB_framebuffer_sRGB WGL_ARB_make_current_read WGL_ARB_multisample WGL_ARB_pbuffer WGL_ARB_pixel_format "
    "WGL_ARB_pixel_format_float WGL_ARB_render_texture WGL_ATI_pixel_format_float WGL_EXT_extensions_string "
    "WGL_EXT_framebuffer_sRGB WGL_EXT_swap_control WGL_WINE_pixel_format_passthrough WGL_WINE_query_renderer";
/* OpenGL ES contexts, with a core that implements them (wmgl_max_es_version). */
static const char wgl_es_extensions[] = " WGL_EXT_create_context_es2_profile WGL_EXT_create_context_es_profile";

static pthread_mutex_t wgl_lock = PTHREAD_MUTEX_INITIALIZER;
static struct wmgl_pixel_format *formats;
static unsigned int format_count, onscreen_count;
static void *root_context;
static void *metal_device;

/* Allocated one by one, so they stay put while the table grows. */
struct context
{
    void *core;
    int format;
    const void *thread;  /* the thread it is current on (its current_context), or NULL */
    struct wmgl_debug_target debug;
};

struct window
{
    UINT64 hwnd;
    int format;     /* the window's pixel format */
    void *host;     /* unix_window.m */
    void *surface;  /* core window surface, made on first use */
    int width, height;
    int interval;
};

struct dc_format
{
    UINT64 hdc;
    int format;
};

struct pbuffer
{
    void *surface;
    int format, width, height;
    UINT64 hdc;     /* the memory DC the PE side made for it, or 0 */
    int texture_format, texture_target, mipmap_level, cube_face;
    BOOL used;
};

static struct context **contexts;
static unsigned int context_count;
static struct window *windows;
static unsigned int window_count;
static struct dc_format *dc_formats;
static unsigned int dc_format_count;
static struct pbuffer *pbuffers;
static unsigned int pbuffer_count;

/* This thread's current context and drawables. */
static __thread UINT64 current_context;
static __thread struct context *current_object;  /* current_context's, read without the lock */
static __thread void *current_draw, *current_read;
static __thread UINT64 current_draw_hwnd;

static void *grow( void *array, unsigned int *count, size_t size )
{
    void *bigger = realloc( array, (*count + 1) * size );
    if (!bigger) return NULL;
    memset( (char *)bigger + *count * size, 0, size );
    ++*count;
    return bigger;
}

static struct context *context_from_handle( UINT64 handle )
{
    if (handle < HANDLE_BASE || handle - HANDLE_BASE >= context_count) return NULL;
    return contexts[handle - HANDLE_BASE];
}

static struct pbuffer *pbuffer_from_handle( UINT64 handle )
{
    if (handle < HANDLE_BASE || handle - HANDLE_BASE >= pbuffer_count) return NULL;
    return pbuffers[handle - HANDLE_BASE].used ? &pbuffers[handle - HANDLE_BASE] : NULL;
}

static struct pbuffer *pbuffer_from_dc( UINT64 hdc )
{
    unsigned int i;
    if (!hdc) return NULL;
    for (i = 0; i < pbuffer_count; i++)
        if (pbuffers[i].used && pbuffers[i].hdc == hdc) return &pbuffers[i];
    return NULL;
}

static struct window *find_window( UINT64 hwnd, BOOL create )
{
    unsigned int i;
    struct window *bigger;

    if (!hwnd) return NULL;
    for (i = 0; i < window_count; i++)
        if (windows[i].hwnd == hwnd) return &windows[i];
    if (!create) return NULL;
    for (i = 0; i < window_count; i++)
    {
        if (windows[i].hwnd) continue;  /* forgotten */
        windows[i].hwnd = hwnd;
        windows[i].interval = 1;
        return &windows[i];
    }
    if (!(bigger = grow( windows, &window_count, sizeof(*windows) ))) return NULL;
    windows = bigger;
    windows[window_count - 1].hwnd = hwnd;
    windows[window_count - 1].interval = 1;
    return &windows[window_count - 1];
}

static struct dc_format *find_dc_format( UINT64 hdc, BOOL create )
{
    unsigned int i;
    struct dc_format *bigger;

    for (i = 0; i < dc_format_count; i++)
        if (dc_formats[i].hdc == hdc) return &dc_formats[i];
    if (!create || !(bigger = grow( dc_formats, &dc_format_count, sizeof(*dc_formats) ))) return NULL;
    dc_formats = bigger;
    dc_formats[dc_format_count - 1].hdc = hdc;
    return &dc_formats[dc_format_count - 1];
}

/* The pixel format of a DC: its window's, its pbuffer's or its own. */
static int dc_pixel_format( UINT64 hdc, UINT64 hwnd )
{
    struct window *window;
    struct pbuffer *pbuffer;
    struct dc_format *dc;

    if (hwnd) return (window = find_window( hwnd, FALSE )) ? window->format : 0;
    if ((pbuffer = pbuffer_from_dc( hdc ))) return pbuffer->format;
    return (dc = find_dc_format( hdc, FALSE )) ? dc->format : 0;
}

C_ASSERT( WMGL_WINDOW_MAX_HOLES == WMGL_MAX_HOLES );

static void to_placement( const struct wmgl_window_placement *from, struct wmgl_placement *to )
{
    unsigned int i;

    /* Whole, padding and unused holes too: wmgl_window_place compares them. */
    memset( to, 0, sizeof(*to) );
    to->hwnd = from->hwnd;
    to->top_level = from->top_level;
    to->x = from->x;
    to->y = from->y;
    to->width = from->width;
    to->height = from->height;
    to->top_level_width = from->top_level_width;
    to->top_level_height = from->top_level_height;
    to->screen_width = from->screen_width;
    to->clip_x = from->clip_x;
    to->clip_y = from->clip_y;
    to->clip_width = from->clip_width;
    to->clip_height = from->clip_height;
    to->visible = from->visible;
    to->hole_count = from->hole_count < WMGL_MAX_HOLES ? from->hole_count : WMGL_MAX_HOLES;
    for (i = 0; i < to->hole_count; i++)
    {
        to->holes[i].x = from->holes[i].x;
        to->holes[i].y = from->holes[i].y;
        to->holes[i].width = from->holes[i].width;
        to->holes[i].height = from->holes[i].height;
    }
}

/* A window's surface, made on first use and kept on its client area. */
static void *window_surface( const struct wmgl_window_placement *placement )
{
    struct window *window = find_window( placement->hwnd, FALSE );
    struct wmgl_placement where;
    int width = placement->width > 0 ? placement->width : 1, height = placement->height > 0 ? placement->height : 1;
    void *layer = NULL;

    if (!window || !window->format) return NULL;
    to_placement( placement, &where );
    if (!window->surface)
    {
        if (!(window->host = wmgl_window_create( &where, metal_device, &layer )) || !layer)
        {
            WMGL_ERR( "no Cocoa view for window %#llx\n", (unsigned long long)placement->hwnd );
            return NULL;
        }
        if (!(window->surface = wmgl.surface_create_window( layer, window->format, width, height )))
        {
            WMGL_ERR( "window %#llx: the core made no surface (%dx%d, pixel format %d)\n",
                      (unsigned long long)placement->hwnd, width, height, window->format );
            wmgl_window_destroy( window->host );
            window->host = NULL;
            return NULL;
        }
        window->width = width;
        window->height = height;
        return window->surface;
    }
    wmgl_window_place( window->host, &where );
    return window->surface;
}

/* Resizes a window surface to its client area (the core keeps what fits). */
static void window_resize( const struct wmgl_window_placement *placement )
{
    struct window *window = find_window( placement->hwnd, FALSE );
    int width = placement->width > 0 ? placement->width : 1, height = placement->height > 0 ? placement->height : 1;

    if (!window || !window->surface || (window->width == width && window->height == height)) return;
    wmgl.surface_flush( window->surface, WMGL_FLUSH_UPDATED, width, height, window->interval );
    window->width = width;
    window->height = height;
}

/* The surface a DC draws to: its window's or its pbuffer's. */
static void *dc_surface( UINT64 hdc, const struct wmgl_window_placement *placement )
{
    struct pbuffer *pbuffer;
    if (placement->hwnd) return window_surface( placement );
    return (pbuffer = pbuffer_from_dc( hdc )) ? pbuffer->surface : NULL;
}

/* The current context cannot be deleted or made current elsewhere: only
 * this thread uses its debug state. */
struct wmgl_debug_target *wmgl_current_debug_target( void )
{
    return current_object ? &current_object->debug : NULL;
}

void wmgl_set_debug_callback( UINT64 callback, UINT64 user, BOOL amd )
{
    if (!current_object) return;
    if (amd)
    {
        current_object->debug.callback_amd = callback;
        current_object->debug.user_amd = user;
    }
    else
    {
        current_object->debug.callback = callback;
        current_object->debug.user = user;
    }
}

/* ---------------------------------------------------------- pixel formats */

NTSTATUS wmgl_unix_init_formats( void *args )
{
    struct wmgl_init_params *params = args;
    int core_major = 0, core_minor = 0, compat_major = 0, compat_minor = 0;

    pthread_mutex_lock( &wgl_lock );
    if (!formats && wmgl.get_pixel_formats)
    {
        format_count = wmgl.get_pixel_formats( NULL, 0, &onscreen_count );
        if (format_count && (formats = calloc( format_count, sizeof(*formats) )))
            wmgl.get_pixel_formats( formats, format_count, &onscreen_count );
        metal_device = wmgl_window_default_device();
    }
    params->format_count = formats ? format_count : 0;
    params->onscreen_count = formats ? onscreen_count : 0;
    pthread_mutex_unlock( &wgl_lock );
    if (wmgl.max_versions) wmgl.max_versions( &core_major, &core_minor, &compat_major, &compat_minor );
    params->core_major = core_major;
    params->core_minor = core_minor;
    params->ok = formats != NULL;
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_pixel_formats( void *args )
{
    struct wmgl_pixel_formats_params *params = args;
    struct wmgl_pixel_format *out = (struct wmgl_pixel_format *)(ULONG_PTR)params->formats;
    unsigned int count = format_count < params->max ? format_count : params->max;

    if (out && formats) memcpy( out, formats, count * sizeof(*out) );
    params->count = count;
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_get_pixel_format( void *args )
{
    struct wmgl_get_pixel_format_params *params = args;
    pthread_mutex_lock( &wgl_lock );
    params->ret = dc_pixel_format( params->hdc, params->hwnd );
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_set_pixel_format( void *args )
{
    struct wmgl_set_pixel_format_params *params = args;
    int format = params->format;
    struct window *window;
    struct dc_format *dc;

    params->ret = FALSE;
    if (format <= 0 || (unsigned int)format > format_count)
    {
        WMGL_WARN( "SetPixelFormat(dc %#llx, %d): no such pixel format (there are %u)\n",
                  (unsigned long long)params->hdc,
                   format, format_count );
        return STATUS_SUCCESS;
    }
    pthread_mutex_lock( &wgl_lock );
    if (params->hwnd)
    {
        /* A window takes a window format, once (WGL), or again through
         * wglSetPixelFormatWINE before it has a drawable. */
        if ((unsigned int)format <= onscreen_count && (window = find_window( params->hwnd, TRUE )))
        {
            if (!window->format || window->format == format) window->format = format, params->ret = TRUE;
            else if (params->internal && !window->surface) window->format = format, params->ret = TRUE;
        }
    }
    else if (!pbuffer_from_dc( params->hdc ) && (dc = find_dc_format( params->hdc, TRUE )))
    {
        if (!dc->format || dc->format == format) dc->format = format, params->ret = TRUE;
    }
    pthread_mutex_unlock( &wgl_lock );
    if (params->ret)
        WMGL_INFO( "SetPixelFormat(dc %#llx, window %#llx, %d)\n", (unsigned long long)params->hdc,
                   (unsigned long long)params->hwnd, format );
    else
        WMGL_WARN( "SetPixelFormat(dc %#llx, window %#llx, %d) refused: %s\n", (unsigned long long)params->hdc,
                   (unsigned long long)params->hwnd, format,
                   params->hwnd && (unsigned int)format > onscreen_count ? "not a format for windows"
                                                                          : "the DC or window has another format" );
    return STATUS_SUCCESS;
}

/* The highest OpenGL ES version the core implements; FALSE for cores
 * without OpenGL ES contexts. */
static BOOL es_max_version( int *major, int *minor )
{
    static void (*max_es_version)( int *, int * );
    static BOOL looked;
    int a = 0, b = 0;

    if (!looked)
    {
        *(void **)&max_es_version = wmgl_core_symbol( "wmgl_max_es_version" );
        looked = TRUE;
    }
    if (!max_es_version) return FALSE;
    max_es_version( &a, &b );
    if (major) *major = a;
    if (minor) *minor = b;
    return a != 0;
}

/* --------------------------------------------------------------- contexts */

NTSTATUS wmgl_unix_create_context( void *args )
{
    struct wmgl_create_context_params *params = args;
    const int *attribs = (const int *)(ULONG_PTR)params->attribs;
    struct context *share = NULL, *context, **bigger;
    void *core, *share_core;
    int format, error = 0;
    unsigned int i;

    params->ret = 0;
    params->error = 0;
    pthread_mutex_lock( &wgl_lock );
    if (!(format = dc_pixel_format( params->hdc, params->hwnd )))
    {
        WMGL_WARN( "wglCreateContext(dc %#llx): the DC has no pixel format\n", (unsigned long long)params->hdc );
        params->error = ERROR_INVALID_PIXEL_FORMAT;
        goto done;
    }
    if (params->share && !(share = context_from_handle( params->share )))
    {
        WMGL_WARN( "wglCreateContext(dc %#llx): context %#llx to share with is not a context\n",
                   (unsigned long long)params->hdc, (unsigned long long)params->share );
        params->error = ERROR_INVALID_OPERATION;
        goto done;
    }
    if (!root_context) root_context = wmgl.context_create( format, NULL, NULL, &error );
    share_core = share ? share->core : root_context;
    if (!share_core || !(core = wmgl.context_create( format, share_core, attribs, &error )))
    {
        WMGL_WARN( "wglCreateContext(dc %#llx, pixel format %d): the core refused (error %d%s)\n",
                   (unsigned long long)params->hdc, format, error, share_core ? "" : ", no root context" );
        params->error = error == 1 ? ERROR_INVALID_VERSION_ARB : error == 2 ? ERROR_INVALID_PROFILE_ARB
                                                                            : ERROR_INVALID_PARAMETER;
        goto done;
    }
    for (i = 0; i < context_count; i++)
        if (!contexts[i]) break;
    if (i == context_count && (bigger = grow( contexts, &context_count, sizeof(*contexts) ))) contexts = bigger;
    if (i == context_count || !(context = calloc( 1, sizeof(*context) )))
    {
        wmgl.context_destroy( core );
        params->error = ERROR_NOT_ENOUGH_MEMORY;
        goto done;
    }
    context->core = core;
    context->format = format;
    contexts[i] = context;
    params->ret = HANDLE_BASE + i;
    WMGL_INFO( "context %#llx (core %p) on dc %#llx, pixel format %d%s\n", (unsigned long long)params->ret, core,
               (unsigned long long)params->hdc, format, share ? ", shared" : "" );
done:
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_delete_context( void *args )
{
    struct wmgl_delete_context_params *params = args;
    struct context *context;

    pthread_mutex_lock( &wgl_lock );
    if (!(context = context_from_handle( params->context )))
    {
        WMGL_WARN( "wglDeleteContext(%#llx): not a context\n", (unsigned long long)params->context );
        params->error = ERROR_INVALID_HANDLE;
    }
    /* Not while another thread has it current (as Windows). */
    else if (context->thread && context->thread != &current_context)
    {
        WMGL_WARN( "wglDeleteContext(%#llx): current in another thread\n", (unsigned long long)params->context );
        params->error = ERROR_BUSY;
    }
    else
    {
        if (current_context == params->context)
        {
            wmgl.make_current( NULL, NULL, NULL );
            current_context = 0;
            current_object = NULL;
            current_draw = current_read = NULL;
        }
        wmgl.context_destroy( context->core );
        contexts[params->context - HANDLE_BASE] = NULL;
        free( context );
        params->ret = TRUE;
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_make_current( void *args )
{
    struct wmgl_make_current_params *params = args;
    struct context *context;
    void *draw, *read;

    params->ret = FALSE;
    params->error = 0;
    pthread_mutex_lock( &wgl_lock );
    if (!params->context)
    {
        wmgl.make_current( NULL, NULL, NULL );
        params->ret = TRUE;
        if (current_object) current_object->thread = NULL;
        current_context = 0;
        current_object = NULL;
        current_draw = current_read = NULL;
        current_draw_hwnd = 0;
        goto done;
    }
    if (!(context = context_from_handle( params->context )))
    {
        WMGL_WARN( "wglMakeCurrent(%#llx): not a context\n", (unsigned long long)params->context );
        params->error = ERROR_INVALID_HANDLE;
        goto done;
    }
    /* One thread at a time (as Windows). */
    if (context->thread && context->thread != &current_context)
    {
        WMGL_WARN( "wglMakeCurrent(%#llx): current in another thread\n", (unsigned long long)params->context );
        params->error = ERROR_BUSY;
        goto done;
    }
    if (!(draw = dc_surface( params->draw_hdc, &params->draw )) ||
        !(read = dc_surface( params->read_hdc, &params->read )))
    {
        params->error = dc_pixel_format( params->draw_hdc, params->draw.hwnd ) &&
                        dc_pixel_format( params->read_hdc, params->read.hwnd ) ? ERROR_INVALID_HANDLE
                                                                               : ERROR_INVALID_PIXEL_FORMAT;
        WMGL_WARN( "wglMakeCurrent(%#llx, draw dc %#llx window %#llx, read dc %#llx): %s\n",
                   (unsigned long long)params->context, (unsigned long long)params->draw_hdc,
                   (unsigned long long)params->draw.hwnd, (unsigned long long)params->read_hdc,
                   params->error == ERROR_INVALID_PIXEL_FORMAT ? "a DC without a pixel format" : "no surface for a DC" );
        goto done;
    }
    if (!wmgl.make_current( context->core, draw, read))
    {
        WMGL_WARN( "wglMakeCurrent(%#llx): the core refused\n", (unsigned long long)params->context );
        params->error = ERROR_INVALID_OPERATION;
        goto done;
    }
    WMGL_TRACE( "wglMakeCurrent(%#llx, window %#llx)\n", (unsigned long long)params->context,
                (unsigned long long)params->draw.hwnd );
    if (current_object) current_object->thread = NULL;
    context->thread = &current_context;
    current_context = params->context;
    current_object = context;
    current_draw = draw;
    current_read = read;
    current_draw_hwnd = params->draw.hwnd;
    if (params->draw.hwnd) window_resize( &params->draw );
    if (params->read.hwnd && params->read.hwnd != params->draw.hwnd) window_resize( &params->read );
    params->ret = TRUE;
done:
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_share_lists( void *args )
{
    struct wmgl_share_lists_params *params = args;
    /* Every context shares with the root context already. */
    pthread_mutex_lock( &wgl_lock );
    params->ret = context_from_handle( params->source ) && context_from_handle( params->dest );
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_copy_context( void *args )
{
    struct wmgl_copy_context_params *params = args;
    /* wglCopyContext: attribute groups between contexts are not copied. */
    WMGL_WARN( "wglCopyContext(%#llx, %#llx, %#x) is not implemented\n", (unsigned long long)params->source,
               (unsigned long long)params->dest, params->mask );
    params->ret = FALSE;
    return STATUS_SUCCESS;
}

/* A destroyed window (the PE side notices): its surface and layer go.
 * Contexts that drew to it keep the surface until they draw elsewhere. */
NTSTATUS wmgl_unix_forget_window( void *args )
{
    struct wmgl_forget_window_params *params = args;
    struct window *window;

    pthread_mutex_lock( &wgl_lock );
    if ((window = find_window( params->hwnd, FALSE )))
    {
        WMGL_TRACE( "window %#llx is gone: its drawable goes\n", (unsigned long long)params->hwnd );
        if (window->surface) wmgl.surface_destroy( window->surface );
        wmgl_window_destroy( window->host );
        memset( window, 0, sizeof(*window) );
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

/* A window's child windows changed (the PE side notices), which a program
 * that has stopped swapping, to play a movie in one, does not show us
 * otherwise: its layer makes room for them now. */
NTSTATUS wmgl_unix_place_window( void *args )
{
    struct wmgl_place_window_params *params = args;
    struct wmgl_placement where;
    struct window *window;

    pthread_mutex_lock( &wgl_lock );
    if ((window = find_window( params->window.hwnd, FALSE )) && window->host)
    {
        WMGL_TRACE( "window %#llx: %u child windows over it\n", (unsigned long long)params->window.hwnd,
                    params->window.hole_count );
        to_placement( &params->window, &where );
        wmgl_window_place( window->host, &where );
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ swaps */

NTSTATUS wmgl_unix_swap_buffers( void *args )
{
    struct wmgl_swap_buffers_params *params = args;
    struct window *window;
    void *surface;

    params->ret = FALSE;
    pthread_mutex_lock( &wgl_lock );
    if (!params->window.hwnd)
    {
        /* A pbuffer or memory DC: nothing to present. */
        params->ret = pbuffer_from_dc( params->hdc ) != NULL || find_dc_format( params->hdc, FALSE ) != NULL;
        goto done;
    }
    if (!(window = find_window( params->window.hwnd, FALSE )) || !window->surface)
    {
        /* SwapBuffers before any context was current on the window. */
        params->ret = window && window->format;
        goto done;
    }
    window_surface( &params->window );  /* follows the window */
    window_resize( &params->window );
    surface = window->surface;  /* until the window is destroyed (wmgl_unix_forget_window) */
    pthread_mutex_unlock( &wgl_lock );
    /* A frame the core could not present (no drawable) is dropped, as a
     * compositor would; SwapBuffers itself succeeds. Presenting may wait for
     * a drawable: not under the lock. */
    wmgl.surface_swap( surface );
    params->ret = TRUE;
    return STATUS_SUCCESS;
done:
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_swap_interval( void *args )
{
    struct wmgl_swap_interval_params *params = args;
    struct window *window;

    pthread_mutex_lock( &wgl_lock );
    params->ret = FALSE;
    if (current_context && (window = find_window( current_draw_hwnd, FALSE )) && window->surface)
    {
        window->interval = params->interval;
        wmgl.surface_flush( window->surface, WMGL_FLUSH_INTERVAL, window->width, window->height, window->interval );
        params->ret = TRUE;
    }
    else if (current_context) params->ret = TRUE;  /* pbuffers present nothing */
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_get_swap_interval( void *args )
{
    struct wmgl_swap_interval_params *params = args;
    struct window *window;

    pthread_mutex_lock( &wgl_lock );
    window = find_window( current_draw_hwnd, FALSE );
    params->interval = window ? window->interval : 1;
    params->ret = TRUE;
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

/* --------------------------------------------------------------- pbuffers */

NTSTATUS wmgl_unix_create_pbuffer( void *args )
{
    struct wmgl_create_pbuffer_params *params = args;
    const int *attribs = (const int *)(ULONG_PTR)params->attribs;
    struct pbuffer *slot = NULL, *bigger;
    int width = params->width, height = params->height, texture_format = WGL_NO_TEXTURE_ARB;
    int texture_target = WGL_NO_TEXTURE_ARB;
    BOOL largest = FALSE;
    unsigned int i;
    void *surface;

    params->ret = 0;
    params->error = 0;
    if (params->format <= 0 || (unsigned int)params->format > format_count ||
        !formats[params->format - 1].draw_to_pbuffer)
    {
        WMGL_WARN( "wglCreatePbufferARB: pixel format %d is not one for pbuffers\n", params->format );
        params->error = ERROR_INVALID_PIXEL_FORMAT;
        return STATUS_SUCCESS;
    }
    for (i = 0; attribs && attribs[i]; i += 2)
    {
        switch (attribs[i])
        {
        case WGL_PBUFFER_LARGEST_ARB: largest = attribs[i + 1] != 0; break;
        case WGL_TEXTURE_FORMAT_ARB: texture_format = attribs[i + 1]; break;
        case WGL_TEXTURE_TARGET_ARB: texture_target = attribs[i + 1]; break;
        case WGL_MIPMAP_TEXTURE_ARB: break;
        default:
            WMGL_WARN( "wglCreatePbufferARB: unknown attribute %#x\n", attribs[i] );
            params->error = ERROR_INVALID_DATA;
            return STATUS_SUCCESS;
        }
    }
    if (width <= 0 || height <= 0)
    {
        WMGL_WARN( "wglCreatePbufferARB: size %dx%d\n", width, height );
        params->error = ERROR_INVALID_DATA;
        return STATUS_SUCCESS;
    }
    if (largest)
    {
        const struct wmgl_pixel_format *f = &formats[params->format - 1];
        if (f->max_pbuffer_width && width > f->max_pbuffer_width) width = f->max_pbuffer_width;
        if (f->max_pbuffer_height && height > f->max_pbuffer_height) height = f->max_pbuffer_height;
    }
    if (!(surface = wmgl.surface_create_pbuffer( params->format, width, height )))
    {
        WMGL_ERR( "wglCreatePbufferARB: the core made no %dx%d pbuffer (pixel format %d)\n", width, height,
                 params->format );
        params->error = ERROR_NO_SYSTEM_RESOURCES;
        return STATUS_SUCCESS;
    }
    pthread_mutex_lock( &wgl_lock );
    for (i = 0; i < pbuffer_count && !slot; i++)
        if (!pbuffers[i].used) slot = &pbuffers[i];
    if (!slot && (bigger = grow( pbuffers, &pbuffer_count, sizeof(*pbuffers) )))
    {
        pbuffers = bigger;
        slot = &pbuffers[pbuffer_count - 1];
    }
    if (slot)
    {
        memset( slot, 0, sizeof(*slot) );
        slot->surface = surface;
        slot->format = params->format;
        slot->width = width;
        slot->height = height;
        slot->texture_format = texture_format;
        slot->texture_target = texture_target;
        slot->cube_face = WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB;
        slot->used = TRUE;
        params->ret = HANDLE_BASE + (slot - pbuffers);
    }
    else
    {
        wmgl.surface_destroy( surface );
        params->error = ERROR_NOT_ENOUGH_MEMORY;
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_destroy_pbuffer( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    struct pbuffer *pbuffer;

    pthread_mutex_lock( &wgl_lock );
    if ((params->ret = (pbuffer = pbuffer_from_handle( params->pbuffer )) != NULL))
    {
        wmgl.surface_destroy( pbuffer->surface );
        memset( pbuffer, 0, sizeof(*pbuffer) );
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_get_pbuffer_dc( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    struct pbuffer *pbuffer;

    pthread_mutex_lock( &wgl_lock );
    if ((params->ret = (pbuffer = pbuffer_from_handle( params->pbuffer )) != NULL))
    {
        /* One DC per pbuffer until it is released. */
        if (pbuffer->hdc) params->hdc = pbuffer->hdc;
        else pbuffer->hdc = params->hdc;
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_release_pbuffer_dc( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    struct pbuffer *pbuffer;

    pthread_mutex_lock( &wgl_lock );
    params->ret = (pbuffer = pbuffer_from_handle( params->pbuffer )) && pbuffer->hdc == params->hdc;
    if (params->ret) pbuffer->hdc = 0;
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_query_pbuffer( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    struct pbuffer *pbuffer;

    pthread_mutex_lock( &wgl_lock );
    params->ret = FALSE;
    if ((pbuffer = pbuffer_from_handle( params->pbuffer )))
    {
        params->ret = TRUE;
        switch (params->attribute)
        {
        case WGL_PBUFFER_WIDTH_ARB: params->value = pbuffer->width; break;
        case WGL_PBUFFER_HEIGHT_ARB: params->value = pbuffer->height; break;
        case WGL_PBUFFER_LOST_ARB: params->value = FALSE; break;
        case WGL_TEXTURE_FORMAT_ARB: params->value = pbuffer->texture_format; break;
        case WGL_TEXTURE_TARGET_ARB: params->value = pbuffer->texture_target; break;
        case WGL_MIPMAP_TEXTURE_ARB: params->value = FALSE; break;
        case WGL_MIPMAP_LEVEL_ARB: params->value = pbuffer->mipmap_level; break;
        case WGL_CUBE_MAP_FACE_ARB: params->value = pbuffer->cube_face; break;
        default:
            params->ret = FALSE;
            params->error = ERROR_INVALID_DATA;
            break;
        }
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

/* WGL_ARB_render_texture: the pbuffer's colour is copied into the texture
 * bound to the pbuffer's texture target, reading from the pbuffer. */
NTSTATUS wmgl_unix_bind_tex_image( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    struct context *context;
    struct pbuffer *pbuffer;
    GLenum target, internal;

    pthread_mutex_lock( &wgl_lock );
    params->ret = FALSE;
    if (!(pbuffer = pbuffer_from_handle( params->pbuffer )) || pbuffer->texture_format == WGL_NO_TEXTURE_ARB ||
        !(context = context_from_handle( current_context )))
        goto done;
    switch (pbuffer->texture_target)
    {
    case WGL_TEXTURE_1D_ARB: target = GL_TEXTURE_1D; break;
    case WGL_TEXTURE_2D_ARB: target = GL_TEXTURE_2D; break;
    case WGL_TEXTURE_RECTANGLE_NV: target = GL_TEXTURE_RECTANGLE; break;
    case WGL_TEXTURE_CUBE_MAP_ARB:
        target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + (pbuffer->cube_face - WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB);
        break;
    default: goto done;
    }
    internal = pbuffer->texture_format == WGL_TEXTURE_RGB_ARB ? GL_RGB : GL_RGBA;
    if (!wmgl.make_current( context->core, current_draw, pbuffer->surface )) goto done;
    if (target == GL_TEXTURE_1D)
        core.p_glCopyTexImage1D( target, pbuffer->mipmap_level, internal, 0, 0, pbuffer->width, 0 );
    else
        core.p_glCopyTexImage2D( target, pbuffer->mipmap_level, internal, 0, 0, pbuffer->width, pbuffer->height, 0 );
    wmgl.make_current( context->core, current_draw, current_read );
    params->ret = TRUE;
done:
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_release_tex_image( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    pthread_mutex_lock( &wgl_lock );
    params->ret = pbuffer_from_handle( params->pbuffer ) != NULL;
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_set_pbuffer_attrib( void *args )
{
    struct wmgl_pbuffer_params *params = args;
    const int *attribs = (const int *)(ULONG_PTR)params->attribs;
    struct pbuffer *pbuffer;
    unsigned int i;

    pthread_mutex_lock( &wgl_lock );
    if ((params->ret = (pbuffer = pbuffer_from_handle( params->pbuffer )) != NULL))
    {
        for (i = 0; attribs && attribs[i]; i += 2)
        {
            if (attribs[i] == WGL_MIPMAP_LEVEL_ARB) pbuffer->mipmap_level = attribs[i + 1];
            else if (attribs[i] == WGL_CUBE_MAP_FACE_ARB) pbuffer->cube_face = attribs[i + 1];
            else params->ret = FALSE;
        }
    }
    pthread_mutex_unlock( &wgl_lock );
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------- renderer queries */

/* WGL_WINE_query_renderer: one renderer, the Metal device; wined3d reads
 * its vendor, device and video memory. */
NTSTATUS wmgl_unix_query_renderer_integer( void *args )
{
    struct wmgl_query_renderer_params *params = args;
    GLuint *value = (GLuint *)(ULONG_PTR)params->value;
    int core_major = 0, core_minor = 0, compat_major = 0, compat_minor = 0, major = 0, minor = 0, patch = 0;

    params->ret = FALSE;
    if (params->renderer || !value) return STATUS_SUCCESS;
    params->ret = TRUE;
    switch (params->attribute)
    {
    case WGL_RENDERER_ACCELERATED_WINE: *value = 1; break;
    case WGL_RENDERER_UNIFIED_MEMORY_ARCHITECTURE_WINE: *value = 1; break;  /* Apple silicon */
    case WGL_RENDERER_VENDOR_ID_WINE: *value = 0x106b; break;              /* Apple */
    case WGL_RENDERER_DEVICE_ID_WINE: *value = 0xffffffff; break;
    case WGL_RENDERER_OPENGL_CORE_PROFILE_VERSION_WINE:
    case WGL_RENDERER_OPENGL_COMPATIBILITY_PROFILE_VERSION_WINE:
        wmgl.max_versions( &core_major, &core_minor, &compat_major, &compat_minor );
        if (params->attribute == WGL_RENDERER_OPENGL_CORE_PROFILE_VERSION_WINE)
            value[0] = core_major, value[1] = core_minor;
        else
            value[0] = compat_major, value[1] = compat_minor;
        break;
    case WGL_RENDERER_OPENGL_ES_PROFILE_VERSION_WINE:  /* OpenGL ES 1.x */
        value[0] = value[1] = 0;
        break;
    case WGL_RENDERER_OPENGL_ES2_PROFILE_VERSION_WINE:
        if (es_max_version( &major, &minor )) value[0] = major, value[1] = minor;
        else value[0] = value[1] = 0;
        break;
    case WGL_RENDERER_PREFERRED_PROFILE_WINE: *value = WGL_CONTEXT_CORE_PROFILE_BIT_ARB; break;
    case WGL_RENDERER_VERSION_WINE:
        if (wmgl.version) wmgl.version( &major, &minor, &patch );
        value[0] = major;
        value[1] = minor;
        value[2] = patch;
        break;
    case WGL_RENDERER_VIDEO_MEMORY_WINE: *value = wmgl.video_memory_mb(); break;
    default: params->ret = FALSE; break;
    }
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_query_renderer_string( void *args )
{
    struct wmgl_query_renderer_params *params = args;
    char *buffer = (char *)(ULONG_PTR)params->buffer;
    const char *string = NULL, *vendor;

    params->ret = FALSE;
    if (params->renderer || !buffer || !params->size) return STATUS_SUCCESS;
    switch (params->attribute)
    {
    case WGL_RENDERER_DEVICE_ID_WINE: string = wmgl.renderer_string(); break;
    case WGL_RENDERER_VENDOR_ID_WINE:
        /* The core's GL_VENDOR (WINEMETALGL_VENDOR overrides it). */
        vendor = getenv( "WINEMETALGL_VENDOR" );
        string = vendor && *vendor ? vendor : "WineMetalGL";
        break;
    }
    if (!string) return STATUS_SUCCESS;
    snprintf( buffer, params->size, "%s", string );
    params->ret = TRUE;
    return STATUS_SUCCESS;
}

NTSTATUS wmgl_unix_extensions_string( void *args )
{
    struct wmgl_extensions_string_params *params = args;
    char *buffer = (char *)(ULONG_PTR)params->buffer;

    const char *es = es_max_version( NULL, NULL ) ? wgl_es_extensions : "";
    size_t length = strlen( wgl_extensions ), es_length = strlen( es );

    params->needed = length + es_length + 1;
    if (buffer && params->size >= params->needed)
    {
        memcpy( buffer, wgl_extensions, length );
        memcpy( buffer + length, es, es_length + 1 );
    }
    return STATUS_SUCCESS;
}
