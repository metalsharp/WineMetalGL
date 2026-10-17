/* Pixel format selection and description (WGL, WGL_ARB_pixel_format) of
 * winemetalgl.dll, adapted from Wine 11.0 dlls/opengl32/wgl.c. The formats
 * come from the WineMetalGL core once (wmgl_init_pixel_formats); every DC
 * sees the same list, its window formats first.
 *
 * Copyright (c) 1999 Lionel Ulmer
 * Copyright (c) 2005 Raphael Junqueira
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include <assert.h>

#include "wmgl_pe.h"

#define TRACE(...) do { } while (0)
#define FIXME(...) WMGL_INFO(__VA_ARGS__)
#define WARN(...) WMGL_TRACE(__VA_ARGS__)
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

static struct wgl_pixel_format *wgl_formats_cache;
static UINT wgl_format_count, wgl_onscreen_count;

/* The core's formats (struct wmgl_pixel_format) as WGL describes them: fully
 * accelerated (neither PFD_GENERIC_FORMAT nor PFD_GENERIC_ACCELERATED). */
void wmgl_init_pixel_formats( const struct wmgl_pixel_format *formats, UINT count, UINT onscreen )
{
    UINT i;

    if (!(wgl_formats_cache = calloc( count, sizeof(*wgl_formats_cache) ))) return;
    for (i = 0; i < count; i++)
    {
        const struct wmgl_pixel_format *f = &formats[i];
        struct wgl_pixel_format *d = &wgl_formats_cache[i];

        d->pfd.nSize = sizeof(d->pfd);
        d->pfd.nVersion = 1;
        d->pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_SUPPORT_COMPOSITION;
        if (f->draw_to_window) d->pfd.dwFlags |= PFD_DRAW_TO_WINDOW;
        if (f->double_buffer) d->pfd.dwFlags |= PFD_DOUBLEBUFFER;
        if (f->stereo) d->pfd.dwFlags |= PFD_STEREO;
        d->pfd.iPixelType = PFD_TYPE_RGBA;
        d->pfd.cColorBits = f->color_bits;
        d->pfd.cRedBits = f->red_bits;
        d->pfd.cRedShift = f->red_shift;
        d->pfd.cGreenBits = f->green_bits;
        d->pfd.cGreenShift = f->green_shift;
        d->pfd.cBlueBits = f->blue_bits;
        d->pfd.cBlueShift = f->blue_shift;
        d->pfd.cAlphaBits = f->alpha_bits;
        d->pfd.cAlphaShift = f->alpha_shift;
        d->pfd.cAccumBits = f->accum_bits;
        d->pfd.cAccumRedBits = d->pfd.cAccumGreenBits = d->pfd.cAccumBlueBits = d->pfd.cAccumAlphaBits =
            f->accum_bits / 4;
        d->pfd.cDepthBits = f->depth_bits;
        d->pfd.cStencilBits = f->stencil_bits;
        d->pfd.cAuxBuffers = f->aux_buffers;
        d->pfd.iLayerType = PFD_MAIN_PLANE;
        d->swap_method = WGL_SWAP_UNDEFINED_ARB;
        d->transparent = FALSE;
        d->pixel_type = f->float_components ? WGL_TYPE_RGBA_FLOAT_ARB : WGL_TYPE_RGBA_ARB;
        d->draw_to_pbuffer = f->draw_to_pbuffer;
        d->max_pbuffer_pixels = f->max_pbuffer_pixels;
        d->max_pbuffer_width = f->max_pbuffer_width;
        d->max_pbuffer_height = f->max_pbuffer_height;
        d->sample_buffers = f->sample_buffers;
        d->samples = f->samples;
        d->bind_to_texture_rgb = d->bind_to_texture_rgba = f->draw_to_pbuffer;
        d->bind_to_texture_rectangle_rgb = d->bind_to_texture_rectangle_rgba = f->draw_to_pbuffer;
        d->framebuffer_srgb_capable = f->srgb_capable;
        d->float_components = f->float_components;
    }
    wgl_format_count = count;
    wgl_onscreen_count = onscreen;
}

static BOOL wmgl_is_memdc( HDC hdc, DWORD *is_memdc )
{
    DWORD type = GetObjectType( hdc );
    if (!type) return FALSE;
    *is_memdc = type == OBJ_MEMDC;
    return TRUE;
}

/* A memory DC sees every format; others the window ones. */
static struct wgl_pixel_format *get_pixel_formats( HDC hdc, UINT *num_formats, UINT *num_onscreen_formats )
{
    DWORD is_memdc;

    *num_formats = *num_onscreen_formats = 0;
    if (!wmgl_is_memdc( hdc, &is_memdc ))
    {
        SetLastError( ERROR_INVALID_HANDLE );
        return NULL;
    }
    *num_formats = wgl_format_count;
    *num_onscreen_formats = is_memdc ? wgl_format_count : wgl_onscreen_count;
    return wgl_formats_cache;
}

INT WINAPI wglChoosePixelFormat(HDC hdc, const PIXELFORMATDESCRIPTOR* ppfd)
{
    PIXELFORMATDESCRIPTOR format, best;
    int i, count, best_format;
    int bestDBuffer = -1, bestStereo = -1;
    DWORD is_memdc;

    TRACE( "%p %p: size %u version %u flags %lu type %u color %u %u,%u,%u,%u "
           "accum %u depth %u stencil %u aux %u\n",
           hdc, ppfd, ppfd->nSize, ppfd->nVersion, ppfd->dwFlags, ppfd->iPixelType,
           ppfd->cColorBits, ppfd->cRedBits, ppfd->cGreenBits, ppfd->cBlueBits, ppfd->cAlphaBits,
           ppfd->cAccumBits, ppfd->cDepthBits, ppfd->cStencilBits, ppfd->cAuxBuffers );

    count = wglDescribePixelFormat( hdc, 0, 0, NULL );
    if (!count) return 0;

    if (!wmgl_is_memdc( hdc, &is_memdc )) is_memdc = 0;

    best_format = 0;
    best.dwFlags = 0;
    best.cAlphaBits = -1;
    best.cColorBits = -1;
    best.cDepthBits = -1;
    best.cStencilBits = -1;
    best.cAuxBuffers = -1;
    best.cAccumBits = -1;

    for (i = 1; i <= count; i++)
    {
        if (!wglDescribePixelFormat( hdc, i, sizeof(format), &format )) continue;

        if ((ppfd->iPixelType == PFD_TYPE_COLORINDEX) != (format.iPixelType == PFD_TYPE_COLORINDEX))
        {
            TRACE( "pixel type mismatch for iPixelFormat=%d\n", i );
            continue;
        }

        if ((ppfd->dwFlags & PFD_DRAW_TO_BITMAP) && !(format.dwFlags & PFD_DRAW_TO_BITMAP))
        {
            TRACE( "PFD_DRAW_TO_BITMAP required but not found for iPixelFormat=%d\n", i );
            continue;
        }
        if ((ppfd->dwFlags & PFD_DRAW_TO_WINDOW) && !(format.dwFlags & PFD_DRAW_TO_WINDOW))
        {
            TRACE( "PFD_DRAW_TO_WINDOW required but not found for iPixelFormat=%d\n", i );
            continue;
        }

        if ((ppfd->dwFlags & PFD_SUPPORT_GDI) && !(format.dwFlags & PFD_SUPPORT_GDI))
        {
            TRACE( "PFD_SUPPORT_GDI required but not found for iPixelFormat=%d\n", i );
            continue;
        }
        if ((ppfd->dwFlags & PFD_SUPPORT_OPENGL) && !(format.dwFlags & PFD_SUPPORT_OPENGL))
        {
            TRACE( "PFD_SUPPORT_OPENGL required but not found for iPixelFormat=%d\n", i );
            continue;
        }
        if (is_memdc && ppfd->cColorBits == 16 && ppfd->cAlphaBits != 1)
        {
            TRACE( "Ignoring iPixelFormat=%d RGB565 format on memory DC\n", i );
            continue;
        }

        /* The behavior of PDF_STEREO/PFD_STEREO_DONTCARE and PFD_DOUBLEBUFFER / PFD_DOUBLEBUFFER_DONTCARE
         * is not very clear on MSDN. They specify that ChoosePixelFormat tries to match pixel formats
         * with the flag (PFD_STEREO / PFD_DOUBLEBUFFERING) set. Otherwise it says that it tries to match
         * formats without the given flag set.
         * A test on Windows using a Radeon 9500pro on WinXP (the driver doesn't support Stereo)
         * has indicated that a format without stereo is returned when stereo is unavailable.
         * So in case PFD_STEREO is set, formats that support it should have priority above formats
         * without. In case PFD_STEREO_DONTCARE is set, stereo is ignored.
         *
         * To summarize the following is most likely the correct behavior:
         * stereo not set -> prefer non-stereo formats, but also accept stereo formats
         * stereo set -> prefer stereo formats, but also accept non-stereo formats
         * stereo don't care -> it doesn't matter whether we get stereo or not
         *
         * In Wine we will treat non-stereo the same way as don't care because it makes
         * format selection even more complicated and second drivers with Stereo advertise
         * each format twice anyway.
         */

        /* Doublebuffer, see the comments above */
        if (!(ppfd->dwFlags & PFD_DOUBLEBUFFER_DONTCARE))
        {
            if (((ppfd->dwFlags & PFD_DOUBLEBUFFER) != bestDBuffer) &&
                ((format.dwFlags & PFD_DOUBLEBUFFER) == (ppfd->dwFlags & PFD_DOUBLEBUFFER)))
                goto found;

            if (bestDBuffer != -1 && (format.dwFlags & PFD_DOUBLEBUFFER) != bestDBuffer) continue;
        }
        else if (!best_format)
            goto found;

        /* Stereo, see the comments above. */
        if (!(ppfd->dwFlags & PFD_STEREO_DONTCARE))
        {
            if (((ppfd->dwFlags & PFD_STEREO) != bestStereo) &&
                ((format.dwFlags & PFD_STEREO) == (ppfd->dwFlags & PFD_STEREO)))
                goto found;

            if (bestStereo != -1 && (format.dwFlags & PFD_STEREO) != bestStereo) continue;
        }
        else if (!best_format)
            goto found;

        /* Below we will do a number of checks to select the 'best' pixelformat.
         * We assume the precedence cColorBits > cAlphaBits > cDepthBits > cStencilBits -> cAuxBuffers.
         * The code works by trying to match the most important options as close as possible.
         * When a reasonable format is found, we will try to match more options.
         * It appears (see the opengl32 test) that Windows opengl drivers ignore options
         * like cColorBits, cAlphaBits and friends if they are set to 0, so they are considered
         * as DONTCARE. At least Serious Sam TSE relies on this behavior. */

        if (ppfd->cColorBits)
        {
            if (((ppfd->cColorBits > best.cColorBits) && (format.cColorBits > best.cColorBits)) ||
                ((format.cColorBits >= ppfd->cColorBits) && (format.cColorBits < best.cColorBits)))
                goto found;

            if (best.cColorBits != format.cColorBits)  /* Do further checks if the format is compatible */
            {
                TRACE( "color mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->cAlphaBits)
        {
            if (((ppfd->cAlphaBits > best.cAlphaBits) && (format.cAlphaBits > best.cAlphaBits)) ||
                ((format.cAlphaBits >= ppfd->cAlphaBits) && (format.cAlphaBits < best.cAlphaBits)))
                goto found;

            if (best.cAlphaBits != format.cAlphaBits)
            {
                TRACE( "alpha mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->cStencilBits)
        {
            if (((ppfd->cStencilBits > best.cStencilBits) && (format.cStencilBits > best.cStencilBits)) ||
                ((format.cStencilBits >= ppfd->cStencilBits) && (format.cStencilBits < best.cStencilBits)))
                goto found;

            if (best.cStencilBits != format.cStencilBits)
            {
                TRACE( "stencil mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->cDepthBits && !(ppfd->dwFlags & PFD_DEPTH_DONTCARE))
        {
            if (((ppfd->cDepthBits > best.cDepthBits) && (format.cDepthBits > best.cDepthBits)) ||
                ((format.cDepthBits >= ppfd->cDepthBits) && (format.cDepthBits < best.cDepthBits)))
                goto found;

            if (best.cDepthBits != format.cDepthBits)
            {
                TRACE( "depth mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->cAuxBuffers)
        {
            if (((ppfd->cAuxBuffers > best.cAuxBuffers) && (format.cAuxBuffers > best.cAuxBuffers)) ||
                ((format.cAuxBuffers >= ppfd->cAuxBuffers) && (format.cAuxBuffers < best.cAuxBuffers)))
                goto found;

            if (best.cAuxBuffers != format.cAuxBuffers)
            {
                TRACE( "aux mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->cAccumBits)
        {
            if (((ppfd->cAccumBits > best.cAccumBits) && (format.cAccumBits > best.cAccumBits)) ||
                ((format.cAccumBits >= ppfd->cAccumBits) && (format.cAccumBits < best.cAccumBits)))
                goto found;

            if (best.cAccumBits != format.cAccumBits)
            {
                TRACE( "cAccumBits mismatch for iPixelFormat=%d\n", i );
                continue;
            }
        }
        if (ppfd->dwFlags & PFD_DEPTH_DONTCARE)
        {
            if (format.cDepthBits < best.cDepthBits)
                goto found;
            continue;
        }

        if (!ppfd->cDepthBits && format.cDepthBits > best.cDepthBits)
            goto found;

        continue;

    found:
        best_format = i;
        best = format;
        bestDBuffer = format.dwFlags & PFD_DOUBLEBUFFER;
        bestStereo = format.dwFlags & PFD_STEREO;
    }

    TRACE( "returning %u\n", best_format );
    return best_format;
}

static BOOL wgl_attrib_uses_layer( int attrib )
{
    switch (attrib)
    {
    case WGL_ACCELERATION_ARB:
    case WGL_TRANSPARENT_ARB:
    case WGL_SHARE_DEPTH_ARB:
    case WGL_SHARE_STENCIL_ARB:
    case WGL_SHARE_ACCUM_ARB:
    case WGL_TRANSPARENT_RED_VALUE_ARB:
    case WGL_TRANSPARENT_GREEN_VALUE_ARB:
    case WGL_TRANSPARENT_BLUE_VALUE_ARB:
    case WGL_TRANSPARENT_ALPHA_VALUE_ARB:
    case WGL_TRANSPARENT_INDEX_VALUE_ARB:
    case WGL_SUPPORT_GDI_ARB:
    case WGL_SUPPORT_OPENGL_ARB:
    case WGL_DOUBLE_BUFFER_ARB:
    case WGL_STEREO_ARB:
    case WGL_PIXEL_TYPE_ARB:
    case WGL_COLOR_BITS_ARB:
    case WGL_RED_BITS_ARB:
    case WGL_RED_SHIFT_ARB:
    case WGL_GREEN_BITS_ARB:
    case WGL_GREEN_SHIFT_ARB:
    case WGL_BLUE_BITS_ARB:
    case WGL_BLUE_SHIFT_ARB:
    case WGL_ALPHA_BITS_ARB:
    case WGL_ALPHA_SHIFT_ARB:
    case WGL_ACCUM_BITS_ARB:
    case WGL_ACCUM_RED_BITS_ARB:
    case WGL_ACCUM_GREEN_BITS_ARB:
    case WGL_ACCUM_BLUE_BITS_ARB:
    case WGL_ACCUM_ALPHA_BITS_ARB:
    case WGL_DEPTH_BITS_ARB:
    case WGL_STENCIL_BITS_ARB:
    case WGL_AUX_BUFFERS_ARB:
    case WGL_SAMPLE_BUFFERS_ARB:
    case WGL_SAMPLES_ARB:
    case WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB:
    case WGL_FLOAT_COMPONENTS_NV:
        return TRUE;
    default:
        return FALSE;
    }
}

static BOOL wgl_pixel_format_get_attrib( const struct wgl_pixel_format *fmt, int attrib, int *value )
{
    int val = 0;
    int valid = -1;

    switch (attrib)
    {
    case WGL_DRAW_TO_WINDOW_ARB: val = !!(fmt->pfd.dwFlags & PFD_DRAW_TO_WINDOW); break;
    case WGL_DRAW_TO_BITMAP_ARB: val = !!(fmt->pfd.dwFlags & PFD_DRAW_TO_BITMAP); break;
    case WGL_ACCELERATION_ARB:
        if (fmt->pfd.dwFlags & PFD_GENERIC_ACCELERATED)
            val = WGL_GENERIC_ACCELERATION_ARB;
        else if (fmt->pfd.dwFlags & PFD_GENERIC_FORMAT)
            val = WGL_NO_ACCELERATION_ARB;
        else
            val = WGL_FULL_ACCELERATION_ARB;
        break;
    case WGL_NEED_PALETTE_ARB: val = !!(fmt->pfd.dwFlags & PFD_NEED_PALETTE); break;
    case WGL_NEED_SYSTEM_PALETTE_ARB: val = !!(fmt->pfd.dwFlags & PFD_NEED_SYSTEM_PALETTE); break;
    case WGL_SWAP_LAYER_BUFFERS_ARB: val = !!(fmt->pfd.dwFlags & PFD_SWAP_LAYER_BUFFERS); break;
    case WGL_SWAP_METHOD_ARB: val = fmt->swap_method; break;
    case WGL_NUMBER_OVERLAYS_ARB:
    case WGL_NUMBER_UNDERLAYS_ARB:
        /* We don't support any overlays/underlays. */
        val = 0;
        break;
    case WGL_TRANSPARENT_ARB: val = fmt->transparent; break;
    case WGL_SHARE_DEPTH_ARB:
    case WGL_SHARE_STENCIL_ARB:
    case WGL_SHARE_ACCUM_ARB:
        /* We support only a main plane at the moment which by definition
         * shares the depth/stencil/accum buffers with itself. */
        val = GL_TRUE;
        break;
    case WGL_SUPPORT_GDI_ARB: val = !!(fmt->pfd.dwFlags & PFD_SUPPORT_GDI); break;
    case WGL_SUPPORT_OPENGL_ARB: val = !!(fmt->pfd.dwFlags & PFD_SUPPORT_OPENGL); break;
    case WGL_DOUBLE_BUFFER_ARB: val = !!(fmt->pfd.dwFlags & PFD_DOUBLEBUFFER); break;
    case WGL_STEREO_ARB: val = !!(fmt->pfd.dwFlags & PFD_STEREO); break;
    case WGL_PIXEL_TYPE_ARB: val = fmt->pixel_type; break;
    case WGL_COLOR_BITS_ARB: val = fmt->pfd.cColorBits; break;
    case WGL_RED_BITS_ARB: val = fmt->pfd.cRedBits; break;
    case WGL_RED_SHIFT_ARB: val = fmt->pfd.cRedShift; break;
    case WGL_GREEN_BITS_ARB: val = fmt->pfd.cGreenBits; break;
    case WGL_GREEN_SHIFT_ARB: val = fmt->pfd.cGreenShift; break;
    case WGL_BLUE_BITS_ARB: val = fmt->pfd.cBlueBits; break;
    case WGL_BLUE_SHIFT_ARB: val = fmt->pfd.cBlueShift; break;
    case WGL_ALPHA_BITS_ARB: val = fmt->pfd.cAlphaBits; break;
    case WGL_ALPHA_SHIFT_ARB: val = fmt->pfd.cAlphaShift; break;
    case WGL_ACCUM_BITS_ARB: val = fmt->pfd.cAccumBits; break;
    case WGL_ACCUM_RED_BITS_ARB: val = fmt->pfd.cAccumRedBits; break;
    case WGL_ACCUM_GREEN_BITS_ARB: val = fmt->pfd.cAccumGreenBits; break;
    case WGL_ACCUM_BLUE_BITS_ARB: val = fmt->pfd.cAccumBlueBits; break;
    case WGL_ACCUM_ALPHA_BITS_ARB: val = fmt->pfd.cAccumAlphaBits; break;
    case WGL_DEPTH_BITS_ARB: val = fmt->pfd.cDepthBits; break;
    case WGL_STENCIL_BITS_ARB: val = fmt->pfd.cStencilBits; break;
    case WGL_AUX_BUFFERS_ARB: val = fmt->pfd.cAuxBuffers; break;
    case WGL_DRAW_TO_PBUFFER_ARB: val = fmt->draw_to_pbuffer; break;
    case WGL_MAX_PBUFFER_PIXELS_ARB: val = fmt->max_pbuffer_pixels; break;
    case WGL_MAX_PBUFFER_WIDTH_ARB: val = fmt->max_pbuffer_width; break;
    case WGL_MAX_PBUFFER_HEIGHT_ARB: val = fmt->max_pbuffer_height; break;
    case WGL_TRANSPARENT_RED_VALUE_ARB:
        val = fmt->transparent_red_value;
        valid = !!fmt->transparent_red_value_valid;
        break;
    case WGL_TRANSPARENT_GREEN_VALUE_ARB:
        val = fmt->transparent_green_value;
        valid = !!fmt->transparent_green_value_valid;
        break;
    case WGL_TRANSPARENT_BLUE_VALUE_ARB:
        val = fmt->transparent_blue_value;
        valid = !!fmt->transparent_blue_value_valid;
        break;
    case WGL_TRANSPARENT_ALPHA_VALUE_ARB:
        val = fmt->transparent_alpha_value;
        valid = !!fmt->transparent_alpha_value_valid;
        break;
    case WGL_TRANSPARENT_INDEX_VALUE_ARB:
        val = fmt->transparent_index_value;
        valid = !!fmt->transparent_index_value_valid;
        break;
    case WGL_SAMPLE_BUFFERS_ARB: val = fmt->sample_buffers; break;
    case WGL_SAMPLES_ARB: val = fmt->samples; break;
    case WGL_BIND_TO_TEXTURE_RGB_ARB: val = fmt->bind_to_texture_rgb; break;
    case WGL_BIND_TO_TEXTURE_RGBA_ARB: val = fmt->bind_to_texture_rgba; break;
    case WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV: val = fmt->bind_to_texture_rectangle_rgb; break;
    case WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV: val = fmt->bind_to_texture_rectangle_rgba; break;
    case WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB: val = fmt->framebuffer_srgb_capable; break;
    case WGL_FLOAT_COMPONENTS_NV: val = fmt->float_components; break;
    default:
        FIXME( "unsupported 0x%x WGL attribute\n", attrib );
        valid = 0;
        break;
    }

    /* If we haven't already determined validity, use the default check */
    if (valid == -1) valid = val != -1;
    if (valid) *value = val;

    return valid;
}

enum attrib_match
{
    ATTRIB_MATCH_INVALID = -1,
    ATTRIB_MATCH_IGNORE,
    ATTRIB_MATCH_EXACT,
    ATTRIB_MATCH_MINIMUM,
};

static enum attrib_match wgl_attrib_match_criteria( int attrib )
{
    switch (attrib)
    {
    case WGL_DRAW_TO_WINDOW_ARB:
    case WGL_DRAW_TO_BITMAP_ARB:
    case WGL_ACCELERATION_ARB:
    case WGL_NEED_PALETTE_ARB:
    case WGL_NEED_SYSTEM_PALETTE_ARB:
    case WGL_SWAP_LAYER_BUFFERS_ARB:
    case WGL_SHARE_DEPTH_ARB:
    case WGL_SHARE_STENCIL_ARB:
    case WGL_SHARE_ACCUM_ARB:
    case WGL_SUPPORT_GDI_ARB:
    case WGL_SUPPORT_OPENGL_ARB:
    case WGL_DOUBLE_BUFFER_ARB:
    case WGL_STEREO_ARB:
    case WGL_PIXEL_TYPE_ARB:
    case WGL_DRAW_TO_PBUFFER_ARB:
    case WGL_BIND_TO_TEXTURE_RGB_ARB:
    case WGL_BIND_TO_TEXTURE_RGBA_ARB:
    case WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV:
    case WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV:
    case WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB:
    case WGL_FLOAT_COMPONENTS_NV:
        return ATTRIB_MATCH_EXACT;
    case WGL_NUMBER_OVERLAYS_ARB:
    case WGL_NUMBER_UNDERLAYS_ARB:
    case WGL_COLOR_BITS_ARB:
    case WGL_RED_BITS_ARB:
    case WGL_GREEN_BITS_ARB:
    case WGL_BLUE_BITS_ARB:
    case WGL_ALPHA_BITS_ARB:
    case WGL_ACCUM_BITS_ARB:
    case WGL_ACCUM_RED_BITS_ARB:
    case WGL_ACCUM_GREEN_BITS_ARB:
    case WGL_ACCUM_BLUE_BITS_ARB:
    case WGL_ACCUM_ALPHA_BITS_ARB:
    case WGL_DEPTH_BITS_ARB:
    case WGL_STENCIL_BITS_ARB:
    case WGL_AUX_BUFFERS_ARB:
    case WGL_SAMPLE_BUFFERS_ARB:
    case WGL_SAMPLES_ARB:
        return ATTRIB_MATCH_MINIMUM;
    case WGL_NUMBER_PIXEL_FORMATS_ARB:
    case WGL_RED_SHIFT_ARB:
    case WGL_GREEN_SHIFT_ARB:
    case WGL_BLUE_SHIFT_ARB:
    case WGL_ALPHA_SHIFT_ARB:
    case WGL_TRANSPARENT_ARB:
    case WGL_TRANSPARENT_RED_VALUE_ARB:
    case WGL_TRANSPARENT_GREEN_VALUE_ARB:
    case WGL_TRANSPARENT_BLUE_VALUE_ARB:
    case WGL_TRANSPARENT_ALPHA_VALUE_ARB:
    case WGL_TRANSPARENT_INDEX_VALUE_ARB:
    case WGL_SWAP_METHOD_ARB:
        return ATTRIB_MATCH_IGNORE;
    default:
        return ATTRIB_MATCH_INVALID;
    }
}

static void filter_format_array( const struct wgl_pixel_format **array,
                                 UINT num_formats, int attrib, int value )
{
    enum attrib_match match = wgl_attrib_match_criteria( attrib );
    int fmt_value;
    UINT i;

    assert(match != ATTRIB_MATCH_INVALID);

    if (match == ATTRIB_MATCH_IGNORE && attrib != WGL_SWAP_METHOD_ARB) return;

    for (i = 0; i < num_formats; ++i)
    {
        if (!array[i]) continue;
        if (!wgl_pixel_format_get_attrib( array[i], attrib, &fmt_value ) ||
            (match == ATTRIB_MATCH_EXACT && fmt_value != value) ||
            (match == ATTRIB_MATCH_MINIMUM && fmt_value < value) ||
            (attrib == WGL_SWAP_METHOD_ARB && ((fmt_value == WGL_SWAP_COPY_ARB) ^ (value == WGL_SWAP_COPY_ARB))))
        {
            array[i] = NULL;
        }
    }
}

static int wgl_attrib_sort_priority( int attrib )
{
    switch (attrib)
    {
    case WGL_DRAW_TO_WINDOW_ARB: return 1;
    case WGL_DRAW_TO_BITMAP_ARB: return 2;
    case WGL_ACCELERATION_ARB: return 3;
    case WGL_COLOR_BITS_ARB: return 4;
    case WGL_ACCUM_BITS_ARB: return 5;
    case WGL_PIXEL_TYPE_ARB: return 6;
    case WGL_ALPHA_BITS_ARB: return 7;
    case WGL_AUX_BUFFERS_ARB: return 8;
    case WGL_DEPTH_BITS_ARB: return 9;
    case WGL_STENCIL_BITS_ARB: return 10;
    case WGL_DOUBLE_BUFFER_ARB: return 11;
    case WGL_SWAP_METHOD_ARB: return 12;
    default: return 100;
    }
}

static int compare_attribs( const void *a, const void *b )
{
    return wgl_attrib_sort_priority( *(int *)a ) - wgl_attrib_sort_priority( *(int *)b );
}

static int wgl_attrib_value_priority( int value )
{
    switch (value)
    {
    case WGL_SWAP_UNDEFINED_ARB: return 1;
    case WGL_SWAP_EXCHANGE_ARB: return 2;
    case WGL_SWAP_COPY_ARB: return 3;

    case WGL_FULL_ACCELERATION_ARB: return 1;
    case WGL_GENERIC_ACCELERATION_ARB: return 2;
    case WGL_NO_ACCELERATION_ARB: return 3;

    case WGL_TYPE_RGBA_ARB: return 1;
    case WGL_TYPE_RGBA_FLOAT_ATI: return 2;
    case WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT: return 3;
    case WGL_TYPE_COLORINDEX_ARB: return 4;

    default: return 100;
    }
}

struct compare_formats_ctx
{
    int attribs[256];
    UINT num_attribs;
};

static int compare_formats( void *arg, const void *a, const void *b )
{
    const struct wgl_pixel_format *fmt_a = *(void **)a, *fmt_b = *(void **)b;
    struct compare_formats_ctx *ctx = arg;
    int attrib, val_a, val_b;
    UINT i;

    if (!fmt_a) return 1;
    if (!fmt_b) return -1;

    for (i = 0; i < ctx->num_attribs; ++i)
    {
        attrib = ctx->attribs[2 * i];
        if (wgl_pixel_format_get_attrib( fmt_a, attrib, &val_a ) &&
            wgl_pixel_format_get_attrib( fmt_b, attrib, &val_b ) &&
            val_a != val_b)
        {
            switch (attrib)
            {
            case WGL_ACCELERATION_ARB:
            case WGL_SWAP_METHOD_ARB:
            case WGL_PIXEL_TYPE_ARB:
                return wgl_attrib_value_priority( val_a ) -
                       wgl_attrib_value_priority( val_b );
            case WGL_COLOR_BITS_ARB:
                /* Prefer 32bpp over other values */
                if (val_a >= 32 && val_b >= 32) return val_a - val_b;
                else return val_b - val_a;
            default:
                /* Smaller values first */
                return val_a - val_b;
            }
        }
    }

    /* Maintain pixel format id order */
    return fmt_a - fmt_b;
}

static void compare_formats_ctx_set_attrib( struct compare_formats_ctx *ctx,
                                            int attrib, int value )
{
    UINT i;

    /* Overwrite attribute if it exists already */
    for (i = 0; i < ctx->num_attribs; ++i)
        if (ctx->attribs[2 * i] == attrib) break;

    assert(i < ARRAY_SIZE(ctx->attribs) / 2);

    ctx->attribs[2 * i] = attrib;
    ctx->attribs[2 * i + 1] = value;
    if (i == ctx->num_attribs) ++ctx->num_attribs;
}

/***********************************************************************
 *		wglChoosePixelFormatARB (OPENGL32.@)
 */
BOOL WINAPI wglChoosePixelFormatARB( HDC hdc, const int *attribs_int, const FLOAT *attribs_float,
                                     UINT max_formats, int *formats, UINT *num_formats )
{
    struct wgl_pixel_format *wgl_formats;
    UINT i, num_wgl_formats, num_wgl_onscreen_formats;
    const struct wgl_pixel_format **format_array;
    struct compare_formats_ctx ctx = { 0 };
    DWORD is_memdc;

    TRACE( "hdc %p, attribs_int %p, attribs_float %p, max_formats %u, formats %p, num_formats %p\n",
           hdc, attribs_int, attribs_float, max_formats, formats, num_formats );

    if (!wmgl_is_memdc( hdc, &is_memdc )) is_memdc = 0;

    wgl_formats = get_pixel_formats( hdc, &num_wgl_formats, &num_wgl_onscreen_formats );


    /* Gather, validate and deduplicate all attributes */
    for (i = 0; attribs_int && attribs_int[i]; i += 2)
    {
        if (wgl_attrib_match_criteria( attribs_int[i] ) == ATTRIB_MATCH_INVALID) return FALSE;
        compare_formats_ctx_set_attrib( &ctx, attribs_int[i], attribs_int[i + 1] );
    }
    for (i = 0; attribs_float && attribs_float[i]; i += 2)
    {
        if (wgl_attrib_match_criteria( attribs_float[i] ) == ATTRIB_MATCH_INVALID) return FALSE;
        compare_formats_ctx_set_attrib( &ctx, attribs_float[i], attribs_float[i + 1] );
    }

    /* Initialize the format_array with (pointers to) all wgl formats */
    format_array = malloc( num_wgl_formats * sizeof(*format_array) );
    if (!format_array) return FALSE;
    for (i = 0; i < num_wgl_formats; ++i)
    {
        struct wgl_pixel_format *format = wgl_formats + i;
        if (is_memdc && format->pfd.cColorBits == 16 && format->pfd.cAlphaBits != 1) format_array[i] = NULL;
        else format_array[i] = &wgl_formats[i];
    }

    /* Remove formats that are not acceptable */
    for (i = 0; i < ctx.num_attribs; ++i)
        filter_format_array( format_array, num_wgl_formats, ctx.attribs[2 * i],
                             ctx.attribs[2 * i + 1] );

    /* Some attributes we always want to sort by (values don't matter for sorting) */
    compare_formats_ctx_set_attrib( &ctx, WGL_ACCELERATION_ARB, 0 );
    compare_formats_ctx_set_attrib( &ctx, WGL_COLOR_BITS_ARB, 0 );
    compare_formats_ctx_set_attrib( &ctx, WGL_ACCUM_BITS_ARB, 0 );

    /* Arrange attributes in the order which we want to check them */
    qsort( ctx.attribs, ctx.num_attribs, 2 * sizeof(*ctx.attribs), compare_attribs );

    /* Sort pixel formats based on the specified attributes */
    qsort_s( format_array, num_wgl_formats, sizeof(*format_array), compare_formats, &ctx );

    /* Return the best max_formats format ids */
    *num_formats = 0;
    for (i = 0; i < num_wgl_formats && i < max_formats && format_array[i]; ++i)
    {
        ++*num_formats;
        formats[i] = format_array[i] - wgl_formats + 1;
    }

    free( format_array );
    return TRUE;
}

INT WINAPI wglDescribePixelFormat( HDC hdc, int index, UINT size, PIXELFORMATDESCRIPTOR *ppfd )
{
    struct wgl_pixel_format *formats;
    UINT num_formats, num_onscreen_formats;

    TRACE( "hdc %p, index %d, size %u, ppfd %p\n", hdc, index, size, ppfd );

    if (!(formats = get_pixel_formats( hdc, &num_formats, &num_onscreen_formats ))) return 0;
    if (!ppfd) return num_onscreen_formats;
    if (size < sizeof(*ppfd)) return 0;
    if (index <= 0 || index > num_onscreen_formats) return 0;

    *ppfd = formats[index - 1].pfd;

    return num_onscreen_formats;
}

/***********************************************************************
 *		wglGetPixelFormatAttribivARB (OPENGL32.@)
 */
BOOL WINAPI wglGetPixelFormatAttribivARB( HDC hdc, int index, int plane, UINT count,
                                          const int *attributes, int *values )
{
    static const DWORD invalid_data_error = 0xC007000D;
    struct wgl_pixel_format *formats;
    UINT i, num_formats, num_onscreen_formats;

    TRACE( "hdc %p, index %d, plane %d, count %u, attributes %p, values %p\n",
           hdc, index, plane, count, attributes, values );

    formats = get_pixel_formats( hdc, &num_formats, &num_onscreen_formats );


    if (!count) return TRUE;
    if (count == 1 && attributes[0] == WGL_NUMBER_PIXEL_FORMATS_ARB)
    {
        values[0] = num_formats;
        return TRUE;
    }
    if (index <= 0 || index > num_formats)
    {
        SetLastError( invalid_data_error );
        return FALSE;
    }

    for (i = 0; i < count; ++i)
    {
        int attrib = attributes[i];

        if (attrib == WGL_NUMBER_PIXEL_FORMATS_ARB)
        {
            values[i] = num_formats;
        }
        else if ((plane != 0 && wgl_attrib_uses_layer( attrib )) ||
                 !wgl_pixel_format_get_attrib( &formats[index - 1], attrib, &values[i] ))
        {
            SetLastError( invalid_data_error );
            return FALSE;
        }
    }

    return TRUE;
}

/***********************************************************************
 *		wglGetPixelFormatAttribfvARB (OPENGL32.@)
 */
BOOL WINAPI wglGetPixelFormatAttribfvARB( HDC hdc, int index, int plane, UINT count,
                                          const int *attributes, FLOAT *values )
{
    int *ivalues;
    BOOL ret;
    UINT i;

    TRACE( "hdc %p, index %d, plane %d, count %u, attributes %p, values %p\n",
           hdc, index, plane, count, attributes, values );

    if (!(ivalues = malloc( count * sizeof(int) ))) return FALSE;

    /* For now we can piggy-back on wglGetPixelFormatAttribivARB, since we don't support
     * any non-integer attributes. */
    ret = wglGetPixelFormatAttribivARB( hdc, index, plane, count, attributes, ivalues );
    if (ret)
    {
        for (i = 0; i < count; i++)
            values[i] = ivalues[i];
    }

    free( ivalues );
    return ret;
}
