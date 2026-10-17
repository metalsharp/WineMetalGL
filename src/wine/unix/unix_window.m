/* Window layers of winemetalgl.so (wmgl_window.h): where a window DC's
 * frames are shown, without changes to winemac.
 *
 * Two ways in:
 * - a winemac that exports macdrv_functions (CrossOver, MetalSharp's Wine),
 *   for a window that already has a client view (Wine 11 makes one with a
 *   client surface, for Vulkan or Wine's GL): a metal view in it, which
 *   follows the window as winemac moves and resizes it;
 * - otherwise (WineHQ's winemac exports nothing; a window without a client
 *   view) a view of ours, with the layer, goes into the content view of the
 *   top-level Cocoa window whose hwnd matches, placed from the window's
 *   client area (wmgl_placement), on the main thread: inside a view that
 *   clips it to what the window's ancestors show, hidden with the window. */

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include "wmgl_window.h"

typedef struct macdrv_opaque_metal_device *macdrv_metal_device;
typedef struct macdrv_opaque_metal_view *macdrv_metal_view;
typedef struct macdrv_opaque_metal_layer *macdrv_metal_layer;
typedef struct macdrv_opaque_view *macdrv_view;
typedef struct macdrv_opaque_window *macdrv_window;
typedef struct opaque_HWND *HWND;

/* The start of winemac's struct macdrv_win_data (Wine 11.x), up to the
 * client area's view. */
struct macdrv_win_data
{
    HWND hwnd;
    macdrv_window cocoa_window;
    macdrv_view client_view;
};

/* winemac's exported function table (CrossOver, MetalSharp's Wine). */
struct macdrv_functions_t
{
    void (*macdrv_init_display_devices)( int );
    struct macdrv_win_data *(*get_win_data)( HWND hwnd );
    void (*release_win_data)( struct macdrv_win_data *data );
    macdrv_window (*macdrv_get_cocoa_window)( HWND hwnd, int require_on_screen );
    macdrv_metal_device (*macdrv_create_metal_device)( void );
    void (*macdrv_release_metal_device)( macdrv_metal_device d );
    macdrv_metal_view (*macdrv_view_create_metal_view)( macdrv_view v, macdrv_metal_device d );
    macdrv_metal_layer (*macdrv_view_get_metal_layer)( macdrv_metal_view v );
    void (*macdrv_view_release_metal_view)( macdrv_metal_view v );
    void (*on_main_thread)( dispatch_block_t b );
};

/* Our view: mouse events go to Wine's views under it. */
@interface WineMetalGLView : NSView
@end

@implementation WineMetalGLView
- (NSView *)hitTest:(NSPoint)point
{
    return nil;
}
@end

@interface WineMetalGLHost : NSObject
{
@public
    CAMetalLayer *layer;
    NSView *clip, *view;            /* ours (no winemac metal view): the layer's view in a clipping one */
    macdrv_metal_view winemac_view; /* winemac's */
    struct wmgl_placement placement;
    int placed;                     /* in a Cocoa window that is open (atomic) */
    id closing;                     /* the observer of that window closing */
    /* Main thread only: */
    struct wmgl_placement latest;   /* the newest placement */
    struct wmgl_hole holes[WMGL_WINDOW_MAX_HOLES];  /* the holes in the mask */
    uint32_t hole_count;
    int hole_wait;                  /* new holes: 1 waiting, 2 due */
    int gone;                       /* wmgl_window_destroy took the view off */
}
@end

@implementation WineMetalGLHost
@end

static struct macdrv_functions_t *winemac_functions(void)
{
    static struct macdrv_functions_t *functions;
    static dispatch_once_t once;
    dispatch_once( &once, ^{ functions = dlsym( RTLD_DEFAULT, "macdrv_functions" ); } );
    return functions;
}

static void trace( const char *what, const struct wmgl_placement *placement )
{
    if (wmgl_log_max >= 3) wmgl_log( 3, "window %#llx: %s", (unsigned long long)placement->hwnd, what );
}

void *wmgl_window_default_device( void )
{
    return (__bridge void *)MTLCreateSystemDefaultDevice();
}

/* A rectangle of the top-level window's client area in the content view,
 * which is the top-level window less what Cocoa draws around it; what Wine
 * draws there itself is a border as wide below as beside the client area,
 * and the rest (caption, menu) above. */
static NSRect content_rect( NSView *content, const struct wmgl_placement *placement, int x, int y, int width,
                            int height )
{
    CGFloat scale = placement->screen_width ? placement->screen_width / NSScreen.screens.firstObject.frame.size.width : 1.0;
    CGFloat border, above;
    NSRect rect;

    if (scale <= 0) scale = 1.0;
    border = (content.bounds.size.width - placement->top_level_width / scale) / 2;
    above = content.bounds.size.height - placement->top_level_height / scale - border;
    rect = NSMakeRect( border + x / scale, above + y / scale, width / scale, height / scale );
    if (!content.isFlipped) rect.origin.y = content.bounds.size.height - NSMaxY( rect );
    return rect;
}

static void place_on_main_thread( WineMetalGLHost *host, struct wmgl_placement placement );

/* How long a new hole waits: a child window that gets a drawable of its own
 * meanwhile (the PE side then sends a placement without it) never shows
 * the white Wine painted under it. */
#define HOLE_DELAY_MS 250

static int has_hole( const struct wmgl_hole *holes, uint32_t count, const struct wmgl_hole *hole )
{
    uint32_t i;
    for (i = 0; i < count; i++)
        if (!memcmp( &holes[i], hole, sizeof(*hole) )) return 1;
    return 0;
}

/* Masks the window's child windows out of its layer (main thread): Wine
 * draws them into the top-level window, under our view. Holes go at once;
 * new ones wait HOLE_DELAY_MS. */
static void make_holes( WineMetalGLHost *host, NSView *content, const struct wmgl_placement *placement )
{
    CGRect bounds = CGRectMake( 0, 0, host->view.bounds.size.width, host->view.bounds.size.height );
    struct wmgl_hole shown[WMGL_WINDOW_MAX_HOLES];
    uint32_t i, count = 0;
    int waiting = 0;
    CAShapeLayer *mask;
    CGMutablePathRef path;

    for (i = 0; i < placement->hole_count && i < WMGL_WINDOW_MAX_HOLES; i++)
    {
        if (host->hole_wait == 2 || has_hole( host->holes, host->hole_count, &placement->holes[i] ))
            shown[count++] = placement->holes[i];
        else
            waiting = 1;
    }
    if (!waiting)
        host->hole_wait = 0;
    else if (!host->hole_wait)
    {
        host->hole_wait = 1;
        dispatch_after( dispatch_time( DISPATCH_TIME_NOW, HOLE_DELAY_MS * NSEC_PER_MSEC ), dispatch_get_main_queue(), ^{
            host->hole_wait = 2;
            place_on_main_thread( host, host->latest );
        } );
    }
    memcpy( host->holes, shown, sizeof(shown[0]) * count );
    host->hole_count = count;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    if (!count)
        host->layer.mask = nil;
    else
    {
        path = CGPathCreateMutable();
        CGPathAddRect( path, NULL, bounds );
        for (i = 0; i < count; i++)
        {
            const struct wmgl_hole *hole = &shown[i];
            NSRect rect = content_rect( content, placement, hole->x, hole->y, hole->width, hole->height );
            /* The layer's geometry is its (unflipped) view's. */
            CGPathAddRect( path, NULL, NSRectToCGRect( [host->view convertRect:rect fromView:content] ) );
        }
        mask = [CAShapeLayer layer];
        mask.frame = bounds;
        mask.path = path;
        mask.fillRule = kCAFillRuleEvenOdd;
        host->layer.mask = mask;
        CGPathRelease( path );
    }
    [CATransaction commit];
}

/* Puts a host's view over its window's client area (main thread). */
static void place_on_main_thread( WineMetalGLHost *host, struct wmgl_placement placement )
{
    if (host->gone) return;
    host->latest = placement;
    for (NSWindow *window in [NSApp windows])
    {
        NSView *content;
        NSRect area, shown, inside;

        /* A 32-bit program's handle is the low half of Wine's. */
        if (![window respondsToSelector:@selector(hwnd)] ||
            (uint32_t)(uintptr_t)((void *(*)(id, SEL))objc_msgSend)( window, @selector(hwnd) ) != (uint32_t)placement.top_level)
            continue;
        content = window.contentView;
        area = content_rect( content, &placement, placement.x, placement.y, placement.width, placement.height );
        shown = content_rect( content, &placement, placement.clip_x, placement.clip_y, placement.clip_width,
                              placement.clip_height );
        if (!host->view)
        {
            host->clip = [[WineMetalGLView alloc] initWithFrame:shown];
            host->clip.clipsToBounds = YES;
            host->view = [[WineMetalGLView alloc] initWithFrame:area];
            host->view.layer = host->layer;
            host->view.wantsLayer = YES;
            [host->clip addSubview:host->view];
        }
        /* A window given another parent takes its view along. */
        if (host->clip.superview != content)
        {
            [content addSubview:host->clip positioned:NSWindowAbove relativeTo:nil];
            if (host->closing) [[NSNotificationCenter defaultCenter] removeObserver:host->closing];
            host->closing = [[NSNotificationCenter defaultCenter]
                addObserverForName:NSWindowWillCloseNotification object:window queue:nil
                        usingBlock:^(NSNotification *note) { __atomic_store_n( &host->placed, 0, __ATOMIC_RELEASE ); }];
        }
        host->clip.frame = shown;
        /* The clipping view is not flipped: its origin is at the bottom. */
        inside = NSMakeRect( area.origin.x - shown.origin.x, area.origin.y - shown.origin.y, area.size.width,
                             area.size.height );
        if (content.isFlipped) inside.origin.y = shown.size.height - NSMaxY( inside );
        host->view.frame = inside;
        make_holes( host, content, &placement );
        host->clip.hidden = !placement.visible || placement.clip_width <= 0 || placement.clip_height <= 0;
        host->layer.contentsScale = window.backingScaleFactor;
        __atomic_store_n( &host->placed, 1, __ATOMIC_RELEASE );
        return;
    }
    /* Not there (yet): the next placement tries again. */
    __atomic_store_n( &host->placed, 0, __ATOMIC_RELEASE );
}

void *wmgl_window_create( const struct wmgl_placement *placement, void *device, void **layer )
{
    struct macdrv_functions_t *functions = winemac_functions();
    WineMetalGLHost *host = [[WineMetalGLHost alloc] init];

    *layer = NULL;
    host->placement = *placement;
    if (functions && functions->get_win_data && functions->macdrv_view_create_metal_view)
    {
        struct macdrv_win_data *data = functions->get_win_data( (HWND)(uintptr_t)placement->hwnd );
        if (data)
        {
            if (data->client_view)
                host->winemac_view = functions->macdrv_view_create_metal_view( data->client_view,
                                                                               (macdrv_metal_device)device );
            functions->release_win_data( data );
        }
        if (host->winemac_view)
        {
            host->layer = (__bridge CAMetalLayer *)(void *)functions->macdrv_view_get_metal_layer( host->winemac_view );
            *layer = (__bridge void *)host->layer;
            trace( "winemac metal view", placement );
            return (__bridge_retained void *)host;
        }
    }
    if (!objc_getClass( "WineWindow" )) return NULL;
    trace( "own view in the Cocoa window", placement );
    /* A layer made outside a committed transaction never shows, and this
     * thread has no run loop to commit one. The main thread places the view
     * later: this thread must not wait for it, which may be waiting on us. */
    [CATransaction begin];
    host->layer = [[CAMetalLayer alloc] init];
    host->layer.device = (__bridge id<MTLDevice>)device;
    [CATransaction commit];
    [CATransaction flush];
    *layer = (__bridge void *)host->layer;
    struct wmgl_placement where = *placement;
    dispatch_async( dispatch_get_main_queue(), ^{ place_on_main_thread( host, where ); } );
    return (__bridge_retained void *)host;
}

void wmgl_window_place( void *handle, const struct wmgl_placement *placement )
{
    WineMetalGLHost *host = (__bridge WineMetalGLHost *)handle;
    if (!host || host->winemac_view) return;  /* winemac's view follows its window */
    if (__atomic_load_n( &host->placed, __ATOMIC_ACQUIRE ) && !memcmp( &host->placement, placement, sizeof(*placement) ))
        return;
    host->placement = *placement;
    struct wmgl_placement where = *placement;
    dispatch_async( dispatch_get_main_queue(), ^{ place_on_main_thread( host, where ); } );
}

void wmgl_window_destroy( void *handle )
{
    WineMetalGLHost *host = (__bridge_transfer WineMetalGLHost *)handle;
    struct macdrv_functions_t *functions = winemac_functions();

    if (!host) return;
    if (host->winemac_view)
    {
        if (functions && functions->macdrv_view_release_metal_view)
            functions->macdrv_view_release_metal_view( host->winemac_view );
        return;
    }
    dispatch_async( dispatch_get_main_queue(), ^{
        /* Placements still queued, or holes waiting, leave it off. */
        host->gone = 1;
        if (host->closing) [[NSNotificationCenter defaultCenter] removeObserver:host->closing];
        [host->clip removeFromSuperview];
    } );
}
