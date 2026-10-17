#!/usr/bin/env python3
"""Generate the GL thunks of the WineMetalGL Wine module (src/wine).

Every GL command of the core (tools/gen/gl_gen.py's selection) becomes:
  - a PE function in winemetalgl.dll that packs its arguments into a
    parameter struct and makes one unix call;
  - a unix call in winemetalgl.so that calls the core's entry point;
  - a wow64 unix call for 32-bit callers, which reads the 32-bit layout of
    the struct and widens pointers (zero-extended) and pointer-sized
    integers (sign-extended).
Commands whose arguments need more than that (pointer arrays, returned
pointers, sync objects, callbacks) are listed in MANUAL_UNIX and
MANUAL_WOW64 and implemented by hand in src/wine/unix.

Writes into OUTDIR:
  wine_gl_params.gen.h  unix call codes and parameter structs (PE and unix)
  wine_gl_pe.gen.c      PE functions and the wglGetProcAddress table
  wine_gl_unix.gen.c    unix and wow64 call tables
  winemetalgl.def       exports of winemetalgl.dll
  opengl32.def          exports of opengl32.dll, forwarded to winemetalgl.dll

    wine_gen.py --registry third_party/khronos/gl.xml --extensions tools/gen/extensions.txt --out DIR
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gl_gen  # noqa: E402

# Unix calls that are not GL commands (src/wine/include/wmgl_wine.h). Their
# codes come first; the GL commands follow in name order.
WGL_CALLS = """
init debug_message pixel_formats get_pixel_format set_pixel_format create_context delete_context
make_current share_lists copy_context swap_buffers swap_interval get_swap_interval create_pbuffer
destroy_pbuffer get_pbuffer_dc release_pbuffer_dc query_pbuffer bind_tex_image release_tex_image
set_pbuffer_attrib query_renderer_integer query_renderer_string extensions_string forget_window place_window batch
log
""".split()

# Pointer-sized types, opaque handles and callbacks (32-bit wide for wow64).
POINTER_SIZED = {"GLintptr", "GLintptrARB", "GLsizeiptr", "GLsizeiptrARB", "GLvdpauSurfaceNV"}
OPAQUE_POINTERS = {"GLeglImageOES", "GLeglClientBufferEXT", "GLsync", "GLDEBUGPROC", "GLDEBUGPROCARB",
                   "GLDEBUGPROCKHR", "GLDEBUGPROCAMD", "GLVULKANPROCNV"}

# Implemented by hand for 64-bit and 32-bit callers alike (src/wine/unix):
# the core calls these callbacks on the unix side, where PE code cannot run,
# and glGetPointerv reports them.
MANUAL_UNIX = {"glDebugMessageCallback", "glDebugMessageCallbackARB", "glDebugMessageCallbackAMD",
               "glDebugMessageCallbackKHR", "glGetPointerv", "glGetPointervEXT", "glGetPointervKHR"}

# Implemented by hand for 32-bit callers (src/wine/unix/unix_wow64.c):
# strings the core returns are copied below 4 GB, sync objects become
# 32-bit handles, arrays of pointers and of GLintptr/GLsizeiptr are widened
# and pointers written through pointers narrowed. (Buffer storage of 32-bit processes lies below
# 4 GB, wmgl_set_buffer_memory: mapped pointers are narrowed as they are.)
MANUAL_WOW64 = {
    "glGetString", "glGetStringi",
    "glFenceSync", "glDeleteSync", "glIsSync", "glClientWaitSync", "glWaitSync", "glGetSynciv",
    "glObjectPtrLabel", "glGetObjectPtrLabel", "glObjectPtrLabelKHR", "glGetObjectPtrLabelKHR",
    "glShaderSource", "glShaderSourceARB", "glGetVertexAttribPointerv", "glGetVertexAttribPointervARB",
    "glGetBufferPointerv", "glGetBufferPointervARB", "glGetBufferPointervOES", "glGetNamedBufferPointerv",
    "glMultiDrawElements", "glMultiDrawElementsEXT", "glMultiDrawElementsBaseVertex",
    "glMultiDrawElementsBaseVertexEXT",
    "glTransformFeedbackVaryings", "glTransformFeedbackVaryingsEXT", "glGetUniformIndices",
    "glCreateShaderProgramv", "glCreateShaderProgramvEXT",
    "glBindBuffersRange", "glBindVertexBuffers", "glVertexArrayVertexBuffers",
}


# Batched calls (src/wine/pe/winemetalgl.c wmgl_batch): a command that
# returns nothing and whose arguments are values is appended to the
# thread's batch instead of making a unix call of its own; the batch runs
# in one unix call before the thread's next unbatched call. Not batched:
# draws and dispatches (they read client arrays when called), glFlush and
# glFinish, and the hand-written commands.
NOT_BATCHED_PREFIXES = ("glDraw", "glMultiDraw", "glDispatch", "glArrayElement", "glFlush", "glFinish")
# Commands that only record a pointer (vertex array state; read by a later draw).
RECORDS_POINTER = {
    "glVertexAttribPointer", "glVertexAttribPointerARB", "glVertexAttribIPointer", "glVertexAttribIPointerEXT",
    "glVertexAttribLPointer", "glVertexAttribLPointerEXT", "glVertexPointer", "glNormalPointer", "glColorPointer",
    "glTexCoordPointer", "glSecondaryColorPointer", "glFogCoordPointer", "glEdgeFlagPointer", "glIndexPointer",
}
UNIFORM_VECTOR = re.compile(r"^gl(Program)?Uniform([1-4])(f|i|ui|d)v(ARB|EXT)?$")
UNIFORM_MATRIX = re.compile(r"^gl(Program)?UniformMatrix([2-4])(?:x([2-4]))?(f|d)v(ARB|EXT)?$")
VALUE_TYPES = {"GLenum", "GLboolean", "GLbitfield", "GLbyte", "GLshort", "GLint", "GLsizei", "GLubyte", "GLushort",
               "GLuint", "GLfloat", "GLclampf", "GLdouble", "GLclampd", "GLint64", "GLuint64", "GLint64EXT",
               "GLuint64EXT", "GLintptr", "GLsizeiptr", "GLintptrARB", "GLsizeiptrARB", "GLfixed", "GLhalfNV"}


def is_pointer(ptype):
    return "*" in ptype or ptype.replace("const", "").strip() in OPAQUE_POINTERS


def field64(ptype):
    return ptype


def field32(ptype):
    base = ptype.replace("const", "").strip()
    if is_pointer(ptype):
        return "UINT"
    if base in POINTER_SIZED:
        return "INT"
    return base


def widen(ptype, expr):
    """A 32-bit field's value as the 64-bit parameter type."""
    base = ptype.replace("const", "").strip()
    if is_pointer(ptype):
        return "(%s)ULongToPtr(%s)" % (ptype, expr)
    if base in POINTER_SIZED:
        return "(%s)%s" % (base, expr)
    return expr


def narrow(rtype, expr):
    base = rtype.replace("const", "").strip()
    if is_pointer(rtype):
        return "wmgl_wow64_pointer( (const void *)%s )" % expr
    if base in POINTER_SIZED:
        return "(INT)%s" % expr
    return expr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--extensions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ext_names = [l.split("#")[0].strip() for l in open(args.extensions)]
    ext_names = [l for l in ext_names if l]
    commands, selected, missing = gl_gen.load(args.registry, ext_names)
    if missing:
        raise SystemExit("extensions not in registry: " + ", ".join(sorted(missing)))
    names = sorted(selected)
    os.makedirs(args.out, exist_ok=True)
    unknown = (MANUAL_UNIX | MANUAL_WOW64) - set(names)
    if unknown:
        raise SystemExit("manual commands that are not commands: " + ", ".join(sorted(unknown)))

    # opengl32.dll's GL exports: the GL 1.0 and 1.1 commands.
    import xml.etree.ElementTree as ET
    root = ET.parse(args.registry).getroot()
    exported = set()
    for feature in root.findall("feature"):
        if feature.get("api") == "gl" and feature.get("number") in ("1.0", "1.1"):
            for require in feature.findall("require"):
                for cmd in require.findall("command"):
                    exported.add(cmd.get("name"))
    # Where each command comes from: a GL version (always offered by a 4.6
    # context) or an extension (offered when the context advertises it).
    version_of = {}
    for feature in root.findall("feature"):
        if feature.get("api") != "gl":
            continue
        for require in feature.findall("require"):
            for cmd in require.findall("command"):
                version_of.setdefault(cmd.get("name"), feature.get("number"))

    def params_of(cmd):
        return commands[cmd]["params"]

    def ret_of(cmd):
        return commands[cmd]["ret"]

    def batch_kind(cmd):
        """None, "values", or ("data", value parameter, bytes expression) for uniform arrays."""
        if ret_of(cmd) != "void" or cmd in MANUAL_UNIX or cmd in MANUAL_WOW64 or cmd.startswith(NOT_BATCHED_PREFIXES):
            return None
        params = params_of(cmd)
        if cmd in RECORDS_POINTER:
            return "values"
        vector, matrix = UNIFORM_VECTOR.match(cmd), UNIFORM_MATRIX.match(cmd)
        if vector or matrix:
            names = [n for _, n in params]
            if "count" not in names or "value" not in names:
                return None
            if vector:
                components = int(vector.group(2))
                base = {"f": "GLfloat", "i": "GLint", "ui": "GLuint", "d": "GLdouble"}[vector.group(3)]
            else:
                rows = int(matrix.group(2))
                components = rows * int(matrix.group(3) or rows)
                base = {"f": "GLfloat", "d": "GLdouble"}[matrix.group(4)]
            return ("data", "value", "(UINT64)count * %d * sizeof(%s)" % (components, base))
        if all(t.replace("const", "").strip() in VALUE_TYPES for t, _ in params):
            return "values"
        return None

    # ---------------------------------------------------------------- params
    with open(os.path.join(args.out, "wine_gl_params.gen.h"), "w") as out:
        out.write("/* Generated by tools/gen/wine_gen.py. Do not edit. */\n#pragma once\n\n")
        out.write("enum wmgl_unix_call\n{\n")
        for call in WGL_CALLS:
            out.write("    unix_wmgl_%s,\n" % call)
        for cmd in names:
            out.write("    unix_%s,\n" % cmd)
        out.write("    unix_wmgl_call_count\n};\n\n")
        for cmd in names:
            fields = ["    %s %s;\n" % (field64(t), n) for t, n in params_of(cmd)]
            if ret_of(cmd) != "void":
                fields.append("    %s ret;\n" % ret_of(cmd))
            if not fields:
                fields.append("    int unused;\n")
            out.write("struct %s_params\n{\n%s};\n\n" % (cmd, "".join(fields)))
        out.write("#ifdef WINE_UNIX_LIB\n\n/* The same structs as built by 32-bit (i386) callers. */\n")
        for cmd in names:
            fields = ["    %s %s;\n" % (field32(t), n) for t, n in params_of(cmd)]
            if ret_of(cmd) != "void":
                fields.append("    %s ret;\n" % field32(ret_of(cmd)))
            if not fields:
                fields.append("    int unused;\n")
            out.write("struct %s_params32\n{\n%s};\n\n" % (cmd, "".join(fields)))
        out.write("#endif /* WINE_UNIX_LIB */\n")

    # -------------------------------------------------------------------- PE
    with open(os.path.join(args.out, "wine_gl_pe.gen.c"), "w") as out:
        out.write("/* Generated by tools/gen/wine_gen.py. Do not edit. */\n\n#include \"wmgl_pe.h\"\n\n")
        for cmd in names:
            ret = ret_of(cmd)
            decl = ", ".join("%s %s" % (t, n) for t, n in params_of(cmd)) or "void"
            init = ", ".join(".%s = %s" % (n, n) for _, n in params_of(cmd))
            out.write("%s WINAPI %s(%s)\n{\n" % (ret, cmd, decl))
            out.write("    struct %s_params args = {%s};\n" % (cmd, (" " + init + " ") if init else ""))
            kind = batch_kind(cmd)
            if kind == "values":
                out.write("    if (wmgl_batch( unix_%s, &args, sizeof(args), NULL, 0, 0 )) return;\n" % cmd)
            elif kind:
                _, field, size = kind
                out.write("    if (count >= 0 && wmgl_batch( unix_%s, &args, sizeof(args), %s, %s, offsetof(struct %s_params, %s) ))"
                          " return;\n" % (cmd, field, size, cmd, field))
            out.write("    WMGL_CALL(%s, &args);\n" % cmd)
            if ret != "void":
                out.write("    return args.ret;\n")
            out.write("}\n\n")
        out.write("const struct wmgl_proc wmgl_procs[] =\n{\n")
        for cmd in names:
            source = version_of.get(cmd) or selected[cmd]
            out.write('    {"%s", (PROC)%s, "%s"},\n' % (cmd, cmd, source))
        out.write("};\nconst unsigned int wmgl_proc_count = %d;\n" % len(names))

    # ------------------------------------------------------------------ unix
    with open(os.path.join(args.out, "wine_gl_unix.gen.c"), "w") as out:
        out.write("/* Generated by tools/gen/wine_gen.py. Do not edit. */\n\n#include \"wmgl_unix.h\"\n\n")
        out.write("struct wmgl_core_procs core;\n\nconst char *const wmgl_core_names[] =\n{\n")
        for cmd in names:
            out.write('    "%s",\n' % cmd)
        out.write("};\n\n")
        for cmd in names:
            if cmd in MANUAL_UNIX:
                out.write("extern NTSTATUS manual_%s( void *args );\n" % cmd)
                continue
            call = ", ".join("params->%s" % n for _, n in params_of(cmd))
            out.write("static NTSTATUS gl_%s( void *args )\n{\n" % cmd)
            if params_of(cmd) or ret_of(cmd) != "void":
                out.write("    struct %s_params *params = args;\n" % cmd)
            out.write("    %score.p_%s(%s);\n" % ("params->ret = " if ret_of(cmd) != "void" else "", cmd, call))
            out.write("    return wmgl_after_call();\n}\n\n")
        out.write("#ifdef _WIN64\n\n")
        for cmd in names:
            if cmd in MANUAL_UNIX or cmd in MANUAL_WOW64:
                out.write("extern NTSTATUS wow64_manual_%s( void *args );\n" % cmd)
                continue
            call = ", ".join(widen(t, "params->%s" % n) for t, n in params_of(cmd))
            out.write("static NTSTATUS wow64_%s( void *args )\n{\n" % cmd)
            if params_of(cmd) or ret_of(cmd) != "void":
                out.write("    struct %s_params32 *params = args;\n" % cmd)
            if ret_of(cmd) != "void":
                out.write("    params->ret = %s;\n" % narrow(ret_of(cmd), "core.p_%s(%s)" % (cmd, call)))
            else:
                out.write("    core.p_%s(%s);\n" % (cmd, call))
            out.write("    return wmgl_after_call();\n}\n\n")
        out.write("#endif /* _WIN64 */\n\n")
        # The WGL calls (src/wine/unix) take the same struct layout from
        # 32-bit and 64-bit callers.
        for call in WGL_CALLS:
            out.write("extern NTSTATUS wmgl_unix_%s( void *args );\n" % call)
        out.write("\nconst unixlib_entry_t __wine_unix_call_funcs[] =\n{\n")
        for call in WGL_CALLS:
            out.write("    wmgl_unix_%s,\n" % call)
        for cmd in names:
            out.write("    %s_%s,\n" % ("manual" if cmd in MANUAL_UNIX else "gl", cmd))
        out.write("};\n\n#ifdef _WIN64\nextern NTSTATUS wmgl_unix_init_wow64( void *args );\n"
                  "extern NTSTATUS wmgl_unix_batch_wow64( void *args );\n")
        out.write("const unixlib_entry_t __wine_unix_call_wow64_funcs[] =\n{\n")
        for call in WGL_CALLS:
            out.write("    wmgl_unix_%s%s,\n" % (call, "_wow64" if call in ("init", "batch") else ""))
        for cmd in names:
            manual = cmd in MANUAL_UNIX or cmd in MANUAL_WOW64
            out.write("    %s_%s,\n" % ("wow64_manual" if manual else "wow64", cmd))
        out.write("};\n#endif\n\nC_ASSERT( ARRAYSIZE(__wine_unix_call_funcs) == unix_wmgl_call_count );\n")

    with open(os.path.join(args.out, "wine_gl_procs.gen.h"), "w") as out:
        out.write("/* Generated by tools/gen/wine_gen.py. Do not edit. */\n#pragma once\n\n")
        out.write("struct wmgl_core_procs\n{\n")
        for cmd in names:
            decl = ", ".join("%s %s" % (t, n) for t, n in params_of(cmd)) or "void"
            out.write("    %s (*p_%s)(%s);\n" % (ret_of(cmd), cmd, decl))
        out.write("};\n\nextern struct wmgl_core_procs core;\nextern const char *const wmgl_core_names[];\n")
        out.write("#define WMGL_CORE_PROC_COUNT %d\n" % len(names))

    # ------------------------------------------------------------------- def
    wgl_exports = """wglChoosePixelFormat wglCopyContext wglCreateContext wglCreateLayerContext wglDeleteContext
    wglDescribeLayerPlane wglDescribePixelFormat wglGetCurrentContext wglGetCurrentDC wglGetDefaultProcAddress
    wglGetLayerPaletteEntries wglGetPixelFormat wglGetProcAddress wglMakeCurrent wglRealizeLayerPalette
    wglSetLayerPaletteEntries wglSetPixelFormat wglShareLists wglSwapBuffers wglSwapLayerBuffers wglUseFontBitmapsA
    wglUseFontBitmapsW wglUseFontOutlinesA wglUseFontOutlinesW""".split()
    gl_exports = sorted(n for n in names if n in exported) + ["glDebugEntry"]
    with open(os.path.join(args.out, "winemetalgl.def"), "w") as out:
        out.write("; Generated by tools/gen/wine_gen.py. Do not edit.\nLIBRARY winemetalgl.dll\nEXPORTS\n")
        for name in gl_exports + wgl_exports:
            out.write("    %s\n" % name)
    with open(os.path.join(args.out, "opengl32.def"), "w") as out:
        out.write("; Generated by tools/gen/wine_gen.py. Do not edit.\nLIBRARY opengl32.dll\nEXPORTS\n")
        for name in gl_exports + wgl_exports:
            out.write("    %s = winemetalgl.%s\n" % (name, name))
    print("wine_gen: %d commands (%d exported), %d manual for wow64" % (
        len(names), len(gl_exports), len(MANUAL_WOW64 | MANUAL_UNIX)))


if __name__ == "__main__":
    main()
