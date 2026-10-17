/* Window layers of winemetalgl.so (unix_window.m). The Objective-C side
 * includes no Wine headers (their BOOL is not Objective-C's); this is its
 * interface in plain C types. */
#ifndef WMGL_WINDOW_H
#define WMGL_WINDOW_H

#include <stdint.h>

/* WMGL_MAX_HOLES and struct wmgl_rect of wmgl_wine.h. */
#define WMGL_WINDOW_MAX_HOLES 8

struct wmgl_hole
{
    int32_t x, y, width, height;
};

/* struct wmgl_window_placement of wmgl_wine.h, in C types. */
struct wmgl_placement
{
    uint64_t hwnd;
    uint64_t top_level;
    int32_t x, y, width, height;
    int32_t top_level_width, top_level_height;
    uint32_t screen_width;
    int32_t clip_x, clip_y, clip_width, clip_height;
    int32_t visible;
    uint32_t hole_count;
    struct wmgl_hole holes[WMGL_WINDOW_MAX_HOLES];
};

/* A CAMetalLayer over the window's client area (*layer), and the host that
 * keeps it there (the result, NULL on failure): winemac's metal view where
 * winemac exports how to make one (macdrv_functions), which follows the
 * window by itself; otherwise a view of ours in the top-level Cocoa window,
 * which wmgl_window_place moves. */
void *wmgl_window_create( const struct wmgl_placement *placement, void *device, void **layer );
void wmgl_window_place( void *host, const struct wmgl_placement *placement );
/* Takes the layer off the window, and frees the host. */
void wmgl_window_destroy( void *host );
/* The Metal device the core renders with. */
void *wmgl_window_default_device( void );

/* Logging of wmgl_unix.h. */
extern int wmgl_log_max;
void wmgl_log( int level, const char *format, ... ) __attribute__((format(printf, 2, 3)));

#endif
