// glGet* state queries (GL 3.3 chapter 6) with GL's type conversions.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/compat/compat_api.h"
#include "core/compat/fixed_function.h"
#include "core/context.h"
#include "core/draw.h"
#include "core/limits.h"
#include "core/metal/device.h"
#include "core/program.h"
#include "core/shader/spirv_input.h"
#include "core/api/read_format.h"
#include "gl_api.gen.h"

namespace wmgl::api {

namespace limits = wmgl::limits;

// Set for 32-bit (WoW64) programs, whose buffer storage the Wine module
// places below 4 GB (api/buffer.mm).
extern void* (*g_bufferAllocate)(size_t);

namespace {

// GL_NVX_gpu_memory_info / GL_ATI_meminfo memory, in KiB: Metal's
// recommended working set stands in for dedicated memory. A 32-bit program
// reaches 4 GB in all, its GL buffers included, so it is told at most
// 2047 MiB, which in bytes still fits the signed 32-bit integers such
// programs keep it in. On a 512 GB Mac the working set is several hundred
// GB; City of Heroes then failed a 2.2 GB allocation of its own (#94).
int64_t reportedMemoryKiB(uint64_t bytes) {
    const int64_t kib = static_cast<int64_t>(bytes / 1024);
    return g_bufferAllocate ? std::min<int64_t>(kib, 2047 * 1024) : kib;
}

}  // namespace

bool implementationReadFormat(Context& ctx, GLenum& format, GLenum& type) {
    if (CheckFramebufferStatus(ctx, GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return false;
    uint32_t level, slice;
    GLenum internalFormat = 0;
    int w, h;
    if (!ctx.draw().readColorTexture(level, slice, internalFormat, w, h)) return false;
    const FormatInfo* info = formatInfo(internalFormat);
    if (!info) return false;
    format = info->isInteger() ? GL_RGBA_INTEGER : GL_RGBA;
    type = info->isSignedInteger() ? GL_INT : info->isInteger() ? GL_UNSIGNED_INT
           : info->isFloat() ? GL_FLOAT : GL_UNSIGNED_BYTE;
    if (info->defaultTransferFormat && info->defaultTransferType && !info->isInteger() && !info->isFloat()) {
        format = info->defaultTransferFormat;
        type = info->defaultTransferType;
    }
    // OpenGL ES: floating-point and integer buffers read in their own
    // components (RED / FLOAT for R32F, RED_INTEGER / INT for R32I), which
    // EXT_clip_cull_distance's tests rely on.
    if (ctx.isEs() && (info->isFloat() || info->isInteger()) && info->defaultTransferFormat)
        format = info->defaultTransferFormat;
    return true;
}

namespace {

struct Value {
    // UInt: 32-bit unsigned state, returned bit-for-bit by GetIntegerv.
    enum Kind { Int, UInt, Int64, Bool, Float, Double, NormFloat } kind = Int;
    int count = 0;
    double v[64] = {};  // (COMPRESSED_TEXTURE_FORMATS lists ASTC's 28 in ES contexts)
};

Value ints(std::initializer_list<int64_t> list) {
    Value r;
    r.kind = Value::Int;
    for (int64_t x : list) r.v[r.count++] = static_cast<double>(x);
    return r;
}
Value uints(std::initializer_list<uint32_t> list) {
    Value r;
    r.kind = Value::UInt;
    for (uint32_t x : list) r.v[r.count++] = static_cast<double>(x);
    return r;
}
Value int64v(int64_t x) {
    Value r;
    r.kind = Value::Int64;
    r.v[r.count++] = static_cast<double>(x);
    return r;
}
Value floats(std::initializer_list<double> list, Value::Kind kind = Value::Float) {
    Value r;
    r.kind = kind;
    for (double x : list) r.v[r.count++] = x;
    return r;
}
Value bools(std::initializer_list<bool> list) {
    Value r;
    r.kind = Value::Bool;
    for (bool b : list) r.v[r.count++] = b ? 1.0 : 0.0;
    return r;
}

GLuint nameOf(const RefPtr<Buffer>& b) { return b ? b->name : 0; }

// The preferred (format, type) for reading the current read buffer.
GLint samples(Context& ctx) {
    Framebuffer* fb = ctx.state.drawFramebuffer.get();
    if (!fb) {
        Surface* s = ctx.drawSurface();
        return s && s->samples() > 1 ? static_cast<GLint>(s->samples()) : 0;
    }
    // A complete framebuffer's attachments all have the same sample count.
    auto attachmentSamples = [](const Attachment& a) -> GLint {
        if (a.type == GL_RENDERBUFFER && a.renderbuffer) return a.renderbuffer->samples > 1 ? a.renderbuffer->samples : 0;
        if (a.type == GL_TEXTURE && a.texture) {
            const GLsizei s = a.texture->image(0, a.level).samples;
            return s > 1 ? s : 0;
        }
        return -1;
    };
    for (const Attachment& a : fb->color)
        if (GLint s = attachmentSamples(a); s >= 0) return s;
    for (const Attachment* a : {&fb->depth, &fb->stencil})
        if (GLint s = attachmentSamples(*a); s >= 0) return s;
    return 0;
}

int textureBindingTarget(GLenum pname) {
    switch (pname) {
    case GL_TEXTURE_BINDING_1D: return kTex1D;
    case GL_TEXTURE_BINDING_2D: return kTex2D;
    case GL_TEXTURE_BINDING_3D: return kTex3D;
    case GL_TEXTURE_BINDING_1D_ARRAY: return kTex1DArray;
    case GL_TEXTURE_BINDING_2D_ARRAY: return kTex2DArray;
    case GL_TEXTURE_BINDING_RECTANGLE: return kTexRectangle;
    case GL_TEXTURE_BINDING_CUBE_MAP: return kTexCubeMap;
    case GL_TEXTURE_BINDING_BUFFER: return kTexBuffer;
    case GL_TEXTURE_BINDING_2D_MULTISAMPLE: return kTex2DMultisample;
    case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: return kTex2DMultisampleArray;
    case GL_TEXTURE_BINDING_CUBE_MAP_ARRAY: return kTexCubeMapArray;
    default: return -1;
    }
}

// Desktop state OpenGL ES does not have (queries of it are INVALID_ENUM).
bool desktopOnlyState(GLenum pname) {
    switch (pname) {
    case GL_LINE_SMOOTH_HINT: case GL_POLYGON_SMOOTH_HINT: case GL_TEXTURE_COMPRESSION_HINT: case GL_DOUBLEBUFFER:
    case GL_STEREO: case GL_DRAW_BUFFER: case GL_POINT_SIZE: case GL_POINT_SIZE_RANGE: case GL_POINT_SIZE_GRANULARITY:
    case GL_SMOOTH_LINE_WIDTH_RANGE: case GL_SMOOTH_LINE_WIDTH_GRANULARITY: case GL_POLYGON_MODE:
    case GL_PRIMITIVE_RESTART: case GL_PRIMITIVE_RESTART_INDEX: case GL_PROGRAM_POINT_SIZE: case GL_COLOR_LOGIC_OP:
    case GL_LOGIC_OP_MODE: case GL_LINE_SMOOTH: case GL_POLYGON_SMOOTH: case GL_POLYGON_OFFSET_LINE:
    case GL_POLYGON_OFFSET_POINT: case GL_MULTISAMPLE: case GL_SAMPLE_ALPHA_TO_ONE: case GL_PACK_SWAP_BYTES:
    case GL_PACK_LSB_FIRST: case GL_UNPACK_SWAP_BYTES: case GL_UNPACK_LSB_FIRST: case GL_PACK_IMAGE_HEIGHT:
    case GL_PACK_SKIP_IMAGES: case GL_TEXTURE_BINDING_1D: case GL_TEXTURE_BINDING_1D_ARRAY:
    case GL_TEXTURE_BINDING_RECTANGLE: case GL_MAX_RECTANGLE_TEXTURE_SIZE: case GL_CLAMP_READ_COLOR:
    case GL_TEXTURE_CUBE_MAP_SEAMLESS: case GL_FRAMEBUFFER_SRGB: case GL_MAX_DUAL_SOURCE_DRAW_BUFFERS:
    case GL_NUM_SHADING_LANGUAGE_VERSIONS: case GL_MIN_MAP_BUFFER_ALIGNMENT: case GL_MAX_VERTEX_STREAMS:
    case GL_MAX_SUBROUTINES: case GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS: case GL_PROVOKING_VERTEX:
    case GL_CONTEXT_PROFILE_MASK:
        return true;
    default: return false;
    }
}

bool getState(Context& ctx, GLenum pname, Value& out) {
    const ContextState& s = ctx.state;
    const DeviceCaps& caps = Device::get().caps();
    using namespace limits;
    if (ctx.isEs()) {
        if (desktopOnlyState(pname)) return false;
        if (pname == GL_GENERATE_MIPMAP_HINT) return out = ints({s.hintGenerateMipmap}), true;
    }
    switch (pname) {
    // --- version and context
    case GL_MAJOR_VERSION: out = ints({ctx.isEs() ? ctx.config().esMajor : ctx.versionMajor()}); return true;
    case GL_MINOR_VERSION: out = ints({ctx.isEs() ? ctx.config().esMinor : ctx.versionMinor()}); return true;
    case GL_CONTEXT_FLAGS:
        out = ints({(ctx.config().forwardCompatible ? GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT : 0) |
                    (ctx.config().debug ? GL_CONTEXT_FLAG_DEBUG_BIT : 0) |
                    (ctx.config().robustAccess ? GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT : 0) |
                    (ctx.config().noError ? GL_CONTEXT_FLAG_NO_ERROR_BIT : 0)});
        return true;
    case GL_CONTEXT_PROFILE_MASK:
        if (ctx.version() < 32 || ctx.isEs()) return false;
        out = ints({ctx.isCore() ? GL_CONTEXT_CORE_PROFILE_BIT : GL_CONTEXT_COMPATIBILITY_PROFILE_BIT});
        return true;
    case GL_NUM_EXTENSIONS: out = ints({static_cast<int64_t>(ctx.extensions().size())}); return true;
    case GL_NUM_SHADING_LANGUAGE_VERSIONS:
        // The glGetStringi(GL_SHADING_LANGUAGE_VERSION) list (api/state.mm).
        if (ctx.version() < 43) return false;
        out = ints({ctx.version() - 43 + 12 + (ctx.version() >= 45 ? 1 : 0)});
        return true;
    case GL_RESET_NOTIFICATION_STRATEGY: out = ints({ctx.config().resetNotification}); return true;
    case 0x90F3:  // GL_CONTEXT_ROBUST_ACCESS (GL_KHR_robustness)
        out = bools({ctx.config().robustAccess});
        return true;
    // --- limits
    case GL_MAX_TEXTURE_SIZE: case GL_MAX_CUBE_MAP_TEXTURE_SIZE: case GL_MAX_RECTANGLE_TEXTURE_SIZE:
    case GL_MAX_RENDERBUFFER_SIZE:
        out = ints({kMaxTextureSize});
        return true;
    case GL_MAX_3D_TEXTURE_SIZE: out = ints({kMax3DTextureSize}); return true;
    case GL_MAX_ARRAY_TEXTURE_LAYERS: out = ints({kMaxArrayLayers}); return true;
    case GL_MAX_TEXTURE_BUFFER_SIZE: out = ints({kMaxTextureBufferSize}); return true;
    case GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT:
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_texture_buffer_range")) return false;
        out = ints({kTextureBufferOffsetAlignment});
        return true;
    case GL_MAX_TEXTURE_LOD_BIAS: out = floats({kMaxTextureLodBias}); return true;
    case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: case GL_MAX_COMPUTE_UNIFORM_BLOCKS: case GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS:
    case GL_MAX_COMPUTE_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS:
    case GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: case GL_MAX_COMPUTE_ATOMIC_COUNTERS: case GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_COMPUTE_IMAGE_UNIFORMS: case GL_DISPATCH_INDIRECT_BUFFER_BINDING:
        if (!ctx.computeShaders()) return false;
        switch (pname) {
        case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: out = ints({kMaxComputeWorkGroupInvocations}); break;
        case GL_MAX_COMPUTE_UNIFORM_BLOCKS: out = ints({kMaxUniformBlocksPerStage}); break;
        case GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS: out = ints({kMaxTextureUnitsPerStage}); break;
        case GL_MAX_COMPUTE_UNIFORM_COMPONENTS: out = ints({kMaxDefaultUniformComponents}); break;
        case GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS:
            out = ints({kMaxDefaultUniformComponents + static_cast<int64_t>(kMaxUniformBlocksPerStage) * kMaxUniformBlockSize / 4});
            break;
        case GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: out = ints({kMaxComputeSharedMemorySize}); break;
        case GL_DISPATCH_INDIRECT_BUFFER_BINDING: out = ints({nameOf(s.dispatchIndirectBuffer)}); break;
        case GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS:
            out = ints({ctx.atomicCounters() ? kMaxStageAtomicCounterBuffers : 0});
            break;
        case GL_MAX_COMPUTE_ATOMIC_COUNTERS: out = ints({ctx.atomicCounters() ? kMaxStageAtomicCounters : 0}); break;
        default: out = ints({ctx.imageLoadStore() ? kMaxImageUniformsPerStage : 0}); break;  // GL_MAX_COMPUTE_IMAGE_UNIFORMS
        }
        return true;
    case GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: case GL_MAX_VERTEX_ATTRIB_BINDINGS:
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_vertex_attrib_binding")) return false;
        out = ints({pname == GL_MAX_VERTEX_ATTRIB_BINDINGS ? VertexArray::kMaxBindings : kMaxVertexAttribRelativeOffset});
        return true;
    case GL_MAX_VERTEX_ATTRIB_STRIDE:
        // GL 4.4; answered with the GL 4.3 binding queries too, as the limit
        // holds there (KHR-GL43.vertex_attrib_binding.basic-state3 reads it).
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_vertex_attrib_binding")) return false;
        out = ints({kMaxVertexAttribStride});
        return true;
    case GL_MAX_FRAMEBUFFER_WIDTH: case GL_MAX_FRAMEBUFFER_HEIGHT: case GL_MAX_FRAMEBUFFER_LAYERS:
    case GL_MAX_FRAMEBUFFER_SAMPLES:
        // GL 4.3 framebuffers without attachments.
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_framebuffer_no_attachments")) return false;
        out = ints({pname == GL_MAX_FRAMEBUFFER_LAYERS    ? kMaxArrayLayers
                    : pname == GL_MAX_FRAMEBUFFER_SAMPLES ? static_cast<int64_t>(caps.maxSamples)
                                                          : kMaxTextureSize});
        return true;
    case GL_MAX_UNIFORM_LOCATIONS:
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_explicit_uniform_location")) return false;
        out = ints({kMaxUniformLocations});
        return true;
    case GL_QUERY_BUFFER_BINDING:  // GL 4.4, GL_ARB_query_buffer_object
        if (ctx.version() < 44) return false;
        out = ints({nameOf(s.queryBuffer)});
        return true;
    case GL_MAX_SHADER_COMPILER_THREADS_ARB:
        if (!ctx.parallelShaderCompile()) return false;
        out = ints({static_cast<GLint>(s.maxShaderCompilerThreads)});
        return true;
    case GL_PARAMETER_BUFFER_BINDING:  // GL 4.6, GL_ARB_indirect_parameters
        if (!ctx.indirectParameters()) return false;
        out = ints({nameOf(s.parameterBuffer)});
        return true;
    // GL 4.3 shader storage blocks (limits.h): vertex, fragment and compute stages.
    case GL_SHADER_STORAGE_BUFFER_BINDING: case GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS: case GL_MAX_SHADER_STORAGE_BLOCK_SIZE:
    case GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: case GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS:
    case GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS: case GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS:
    case GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS: case GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS:
    case GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: case GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS:
        if (!ctx.storageBuffers()) return false;
        switch (pname) {
        case GL_SHADER_STORAGE_BUFFER_BINDING: out = ints({nameOf(s.shaderStorageBuffer)}); break;
        case GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS: out = ints({kMaxShaderStorageBufferBindings}); break;
        case GL_MAX_SHADER_STORAGE_BLOCK_SIZE: out = ints({kMaxShaderStorageBlockSize}); break;
        case GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: out = ints({kShaderStorageBufferOffsetAlignment}); break;
        case GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS: out = ints({kMaxCombinedStorageBlocks}); break;
        default: out = ints({kMaxStageStorageBlocks}); break;
        }
        return true;
    case GL_MAX_VIEWPORT_DIMS: out = ints({kMaxTextureSize, kMaxTextureSize}); return true;
    case GL_MAX_DRAW_BUFFERS: case GL_MAX_COLOR_ATTACHMENTS: out = ints({kMaxDrawBuffers}); return true;
    case GL_MAX_DUAL_SOURCE_DRAW_BUFFERS: out = ints({1}); return true;
    case GL_MAX_SAMPLES: case GL_MAX_COLOR_TEXTURE_SAMPLES: case GL_MAX_DEPTH_TEXTURE_SAMPLES: case GL_MAX_INTEGER_SAMPLES:
        out = ints({static_cast<int64_t>(caps.maxSamples)});
        return true;
    case GL_MAX_SAMPLE_MASK_WORDS: out = ints({1}); return true;
    case GL_MAX_ELEMENTS_INDICES: case GL_MAX_ELEMENTS_VERTICES: out = ints({1 << 24}); return true;
    case GL_MAX_VERTEX_ATTRIBS: out = ints({kMaxVertexAttribs}); return true;
    case GL_MAX_VERTEX_UNIFORM_COMPONENTS: case GL_MAX_FRAGMENT_UNIFORM_COMPONENTS: case GL_MAX_GEOMETRY_UNIFORM_COMPONENTS:
        out = ints({kMaxDefaultUniformComponents});
        return true;
    case GL_MAX_VERTEX_UNIFORM_VECTORS: case GL_MAX_FRAGMENT_UNIFORM_VECTORS:
        out = ints({kMaxDefaultUniformComponents / 4});
        return true;
    case GL_MAX_VARYING_COMPONENTS: out = ints({64}); return true;
    case GL_MAX_VARYING_VECTORS: out = ints({16}); return true;
    // GL_ARB_ES2_compatibility.
    case GL_SHADER_COMPILER: out = bools({true}); return true;
    // GL 4.6 / GL_ARB_gl_spirv: SPIR-V is the one shader binary format.
    case GL_NUM_SHADER_BINARY_FORMATS: out = ints({ctx.glSpirv() ? 1 : 0}); return true;
    case GL_NUM_SPIR_V_EXTENSIONS:
        if (!ctx.glSpirv()) return false;
        out = ints({static_cast<int64_t>(shader::spirvExtensions().size())});
        return true;
    case GL_NUM_PROGRAM_BINARY_FORMATS: case GL_PROGRAM_BINARY_FORMATS:
        if (ctx.version() < 41 && !ctx.hasExtension("GL_ARB_get_program_binary")) return false;
        out = pname == GL_NUM_PROGRAM_BINARY_FORMATS ? ints({1}) : ints({kProgramBinaryFormat});
        return true;
    case GL_SHADER_BINARY_FORMATS:
        out = ctx.glSpirv() ? ints({GL_SHADER_BINARY_FORMAT_SPIR_V}) : ints({});
        return true;
    case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
    case GL_IMPLEMENTATION_COLOR_READ_TYPE: {
        GLenum format = GL_RGBA, type = GL_UNSIGNED_BYTE;
        if (!implementationReadFormat(ctx, format, type)) {
            ctx.setError(GL_INVALID_OPERATION);
            out = ints({0});
            return true;
        }
        out = ints({pname == GL_IMPLEMENTATION_COLOR_READ_FORMAT ? format : type});
        return true;
    }
    case GL_MAX_VERTEX_OUTPUT_COMPONENTS: out = ints({kMaxVertexOutputComponents}); return true;
    case GL_MAX_GEOMETRY_INPUT_COMPONENTS: out = ints({kMaxGeometryInputComponents}); return true;
    case GL_MAX_GEOMETRY_OUTPUT_COMPONENTS: out = ints({kMaxInterfaceComponents}); return true;
    case GL_MAX_FRAGMENT_INPUT_COMPONENTS: out = ints({kMaxInterfaceComponents}); return true;
    case GL_MAX_GEOMETRY_OUTPUT_VERTICES: out = ints({256}); return true;
    // Primitive restart is treated as disabled for GL_PATCHES (GL 4.3 section 10.3.5).
    case GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED:
        if (!ctx.tessellationShaders()) return false;
        out = bools({false});
        return true;
    // GL 4.2 atomic counters (limits.h); tessellation stages have none.
    case GL_ATOMIC_COUNTER_BUFFER_BINDING: case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: case GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE:
    case GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS: case GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS: case GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS: case GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_VERTEX_ATOMIC_COUNTERS: case GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS: case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS:
    case GL_MAX_GEOMETRY_ATOMIC_COUNTERS: case GL_MAX_FRAGMENT_ATOMIC_COUNTERS: case GL_MAX_COMBINED_ATOMIC_COUNTERS:
        if (!ctx.atomicCounters()) return false;
        switch (pname) {
        case GL_ATOMIC_COUNTER_BUFFER_BINDING: out = ints({nameOf(s.atomicCounterBuffer)}); break;
        case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: out = ints({kMaxAtomicCounterBufferBindings}); break;
        case GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE: out = ints({kMaxAtomicCounterBufferSize}); break;
        case GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS: out = ints({kMaxVertexAtomicCounterBuffers}); break;
        case GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS: case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS: case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS:
            out = ints({0});
            break;
        case GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS: case GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS:
            out = ints({kMaxStageAtomicCounterBuffers});
            break;
        default: out = ints({kMaxStageAtomicCounters}); break;
        }
        return true;
    // GL 4.2 image load/store (limits.h): every stage has image uniforms.
    case GL_MAX_IMAGE_UNITS: case GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS: case GL_MAX_IMAGE_SAMPLES:
    case GL_MAX_VERTEX_IMAGE_UNIFORMS: case GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS: case GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS:
    case GL_MAX_GEOMETRY_IMAGE_UNIFORMS: case GL_MAX_FRAGMENT_IMAGE_UNIFORMS: case GL_MAX_COMBINED_IMAGE_UNIFORMS:
        if (!ctx.imageLoadStore()) return false;
        out = ints({pname == GL_MAX_IMAGE_UNITS ? kMaxImageUnits
                    // = GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES (GL 4.3), which counts storage blocks too
                    : pname == GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS ? kMaxCombinedShaderOutputResources  // = gl_MaxCombinedShaderOutputResources
                    : pname == GL_MAX_IMAGE_SAMPLES ? 0  // multisample images: Metal cannot write them
                    : pname == GL_MAX_COMBINED_IMAGE_UNIFORMS ? kMaxCombinedImageUniforms
                                                               : kMaxImageUniformsPerStage});
        return true;
    case GL_MAX_SUBROUTINES: case GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS:
        if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_shader_subroutine")) return false;
        out = ints({pname == GL_MAX_SUBROUTINES ? 256 : 1024});
        return true;
    case GL_MAX_GEOMETRY_SHADER_INVOCATIONS:
        if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_gpu_shader5")) return false;
        out = ints({32});
        return true;
    // --- tessellation (GL 4.0)
    case GL_PATCH_VERTICES: case GL_PATCH_DEFAULT_OUTER_LEVEL: case GL_PATCH_DEFAULT_INNER_LEVEL:
    case GL_MAX_PATCH_VERTICES: case GL_MAX_TESS_GEN_LEVEL: case GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS:
    case GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS: case GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS:
    case GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS: case GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS:
    case GL_MAX_TESS_PATCH_COMPONENTS: case GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS:
    case GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS: case GL_MAX_TESS_CONTROL_INPUT_COMPONENTS:
    case GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS: case GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS:
    case GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS: case GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS:
    case GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS:
        if (!ctx.tessellationShaders()) return false;
        switch (pname) {
        case GL_PATCH_VERTICES: out = ints({s.patchVertices}); break;
        case GL_PATCH_DEFAULT_OUTER_LEVEL:
            out = floats({s.patchDefaultOuterLevel[0], s.patchDefaultOuterLevel[1], s.patchDefaultOuterLevel[2],
                          s.patchDefaultOuterLevel[3]});
            break;
        case GL_PATCH_DEFAULT_INNER_LEVEL: out = floats({s.patchDefaultInnerLevel[0], s.patchDefaultInnerLevel[1]}); break;
        case GL_MAX_PATCH_VERTICES: out = ints({kMaxPatchVertices}); break;
        case GL_MAX_TESS_GEN_LEVEL: out = ints({kMaxTessGenLevel}); break;
        case GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS: case GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS:
            out = ints({kMaxDefaultUniformComponents});
            break;
        case GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS: case GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS:
            out = ints({kMaxTextureUnitsPerStage});
            break;
        case GL_MAX_TESS_PATCH_COMPONENTS: out = ints({120}); break;
        case GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS: out = ints({4096}); break;
        case GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS: case GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS:
            out = ints({kMaxUniformBlocksPerStage});
            break;
        case GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS:
            out = ints({kMaxDefaultUniformComponents + static_cast<int64_t>(kMaxUniformBlocksPerStage) * kMaxUniformBlockSize / 4});
            break;
        default: out = ints({kMaxInterfaceComponents}); break;  // input and output components
        }
        return true;
    case GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS: out = ints({1024}); return true;
    case GL_MAX_VERTEX_UNIFORM_BLOCKS: case GL_MAX_GEOMETRY_UNIFORM_BLOCKS: case GL_MAX_FRAGMENT_UNIFORM_BLOCKS:
        out = ints({kMaxUniformBlocksPerStage});
        return true;
    case GL_MAX_COMBINED_UNIFORM_BLOCKS: out = ints({kMaxCombinedUniformBlocks}); return true;
    case GL_MAX_UNIFORM_BUFFER_BINDINGS:
        static_assert(kMaxUniformBufferBindings == ContextState::kMaxUniformBufferBindings);
        out = ints({kMaxUniformBufferBindings});
        return true;
    case GL_MAX_UNIFORM_BLOCK_SIZE: out = ints({kMaxUniformBlockSize}); return true;
    case GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS:
    case GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS:
        out = ints({kMaxDefaultUniformComponents + static_cast<int64_t>(kMaxUniformBlocksPerStage) * kMaxUniformBlockSize / 4});
        return true;
    case GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT: out = ints({256}); return true;
    case GL_MIN_MAP_BUFFER_ALIGNMENT: out = ints({64}); return true;  // GL_ARB_map_buffer_alignment
    case GL_CLIP_ORIGIN: out = ints({s.clipOrigin}); return true;      // GL_ARB_clip_control
    // KHR_debug
    case GL_MAX_DEBUG_MESSAGE_LENGTH: out = ints({static_cast<int64_t>(Context::kMaxDebugMessageLength)}); return true;
    case GL_MAX_DEBUG_LOGGED_MESSAGES: out = ints({static_cast<int64_t>(Context::kMaxDebugLoggedMessages)}); return true;
    case GL_MAX_DEBUG_GROUP_STACK_DEPTH: out = ints({static_cast<int64_t>(Context::kMaxDebugGroupDepth)}); return true;
    case GL_MAX_LABEL_LENGTH: out = ints({Context::kMaxLabelLength}); return true;
    case GL_DEBUG_LOGGED_MESSAGES: out = ints({static_cast<int64_t>(ctx.debugLog.size())}); return true;
    case GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH:
        out = ints({ctx.debugLog.empty() ? 0 : static_cast<int64_t>(ctx.debugLog.front().message.size() + 1)});
        return true;
    case GL_DEBUG_GROUP_STACK_DEPTH: out = ints({static_cast<int64_t>(ctx.debugGroups.size())}); return true;
    case GL_CLIP_DEPTH_MODE: out = ints({s.clipDepthMode}); return true;
    case GL_MAX_TEXTURE_IMAGE_UNITS: case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: case GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS:
        out = ints({kMaxTextureUnitsPerStage});
        return true;
    case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: out = ints({kMaxCombinedTextureUnits}); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: out = ints({64}); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: out = ints({4}); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: out = ints({4}); return true;
    case GL_MAX_CLIP_DISTANCES: out = ints({8}); return true;
    case GL_CONTEXT_RELEASE_BEHAVIOR:  // GL 4.5, GL_KHR_context_flush_control: releasing a context flushes it
        if (ctx.version() < 45 && !ctx.hasExtension("GL_KHR_context_flush_control")) return false;
        out = ints({GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH});
        return true;
    case GL_MAX_CULL_DISTANCES: case GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES:  // GL 4.5, GL_ARB_cull_distance
        if (ctx.version() < 45 && !ctx.hasExtension("GL_ARB_cull_distance")) return false;
        out = ints({8});
        return true;
    case GL_MAX_SERVER_WAIT_TIMEOUT: out = int64v(INT64_MAX); return true;
    // GL 4.3 / GL_ARB_ES3_compatibility, which the KHR-GL32 API coverage test queries in
    // every core context. Element arrays take full 32-bit indices.
    case GL_MAX_ELEMENT_INDEX: out = int64v(UINT32_MAX); return true;
    case GL_SUBPIXEL_BITS: out = ints({8}); return true;
    case GL_MAX_PROGRAM_TEXEL_OFFSET: out = ints({7}); return true;
    case GL_MIN_PROGRAM_TEXEL_OFFSET: out = ints({-8}); return true;
    // GL 4.0 / GL_ARB_texture_gather: Metal's gather offsets span -8..7.
    case GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET: case GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET:
    case GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS_ARB:
        if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_texture_gather")) return false;
        out = ints({pname == GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET ? -8 : pname == GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET ? 7 : 4});
        return true;
    // Wide lines are drawn by the compatibility profile's fixed-function programs only.
    case GL_ALIASED_LINE_WIDTH_RANGE:
        out = floats({1.0, ctx.isCompat() ? static_cast<double>(compat::kMaxLineWidth) : 1.0});
        return true;
    case GL_SMOOTH_LINE_WIDTH_RANGE: out = floats({1.0, 1.0}); return true;
    case GL_SMOOTH_LINE_WIDTH_GRANULARITY: out = floats({0.125}); return true;
    case GL_POINT_SIZE_RANGE: out = floats({1.0, 511.0}); return true;
    case GL_ALIASED_POINT_SIZE_RANGE:  // OpenGL ES's name for it (the compatibility profile's: compat/fixed_get.mm)
        if (!ctx.isEs()) break;
        out = floats({1.0, 511.0});
        return true;
    case GL_POINT_SIZE_GRANULARITY: out = floats({0.125}); return true;
    case GL_NUM_COMPRESSED_TEXTURE_FORMATS: case GL_COMPRESSED_TEXTURE_FORMATS: {
        // Generic/RGTC formats are not listed (GL 3.3 excludes formats with restrictions); ETC2/EAC from GL 4.3.
        std::vector<int64_t> formats;
        if (caps.bcTextures)
            formats = {GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,
                       GL_COMPRESSED_RGBA_S3TC_DXT5_EXT};
        if (ctx.version() >= 43)
            for (int64_t f : {GL_COMPRESSED_RGB8_ETC2, GL_COMPRESSED_SRGB8_ETC2, GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
                              GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, GL_COMPRESSED_RGBA8_ETC2_EAC,
                              GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC, GL_COMPRESSED_R11_EAC, GL_COMPRESSED_SIGNED_R11_EAC,
                              GL_COMPRESSED_RG11_EAC, GL_COMPRESSED_SIGNED_RG11_EAC})
                formats.push_back(f);
        if (ctx.isEs()) {
            formats.push_back(0x8D64);  // GL_ETC1_RGB8_OES
            if (caps.astcTextures)
                for (int64_t i = 0; i < 14; ++i) {
                    formats.push_back(0x93B0 + i);  // GL_COMPRESSED_RGBA_ASTC_*_KHR
                    formats.push_back(0x93D0 + i);  // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_*_KHR
                }
        }
        if (pname == GL_NUM_COMPRESSED_TEXTURE_FORMATS) out = ints({static_cast<int64_t>(formats.size())});
        else {
            out.kind = Value::Int;
            out.count = static_cast<int>(formats.size());
            for (size_t i = 0; i < formats.size(); ++i) out.v[i] = static_cast<double>(formats[i]);
        }
        return true;
    }
    case GL_TIMESTAMP: {
        uint64_t cpu = 0, gpu = 0;
        [Device::get().mtl() sampleTimestamps:&cpu gpuTimestamp:&gpu];
        out = int64v(static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)));
        return true;
    }
    case GL_MAX_TEXTURE_MAX_ANISOTROPY: out = floats({16.0}); return true;
    // --- rasterization and fragment state
    case GL_VIEWPORT: out = floats({s.viewports[0][0], s.viewports[0][1], s.viewports[0][2], s.viewports[0][3]}); return true;
    // GL_EXT_fragment_shading_rate (api/es.mm): the 1x1 rate only.
    case GL_SHADING_RATE_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = ints({static_cast<int64_t>(s.shadingRate)});
        return true;
    case GL_MIN_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_WIDTH_EXT: case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_WIDTH_EXT:
    case GL_MIN_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_HEIGHT_EXT: case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_HEIGHT_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = ints({16});
        return true;
    case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_ASPECT_RATIO_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = ints({1});
        return true;
    case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_LAYERS_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = ints({limits::kMaxArrayLayers});
        return true;
    case GL_FRAGMENT_SHADING_RATE_WITH_SHADER_DEPTH_STENCIL_WRITES_SUPPORTED_EXT:
    case GL_FRAGMENT_SHADING_RATE_WITH_SAMPLE_MASK_SUPPORTED_EXT:
    case GL_FRAGMENT_SHADING_RATE_NON_TRIVIAL_COMBINERS_SUPPORTED_EXT:
    case GL_FRAGMENT_SHADING_RATE_PRIMITIVE_RATE_WITH_MULTI_VIEWPORT_SUPPORTED_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = bools({true});
        return true;
    case GL_FRAGMENT_SHADING_RATE_ATTACHMENT_WITH_DEFAULT_FRAMEBUFFER_SUPPORTED_EXT:
        if (!ctx.advertises("GL_EXT_fragment_shading_rate")) return false;
        out = bools({false});
        return true;
    case GL_PRIMITIVE_BOUNDING_BOX:
        if (!ctx.isEs() || ctx.esVersion() < 31) return false;
        out = floats({s.primitiveBoundingBox[0], s.primitiveBoundingBox[1], s.primitiveBoundingBox[2], s.primitiveBoundingBox[3],
                      s.primitiveBoundingBox[4], s.primitiveBoundingBox[5], s.primitiveBoundingBox[6], s.primitiveBoundingBox[7]});
        return true;
    // GL 4.1 viewport arrays.
    case GL_MAX_VIEWPORTS: case GL_VIEWPORT_SUBPIXEL_BITS: case GL_VIEWPORT_BOUNDS_RANGE: case GL_LAYER_PROVOKING_VERTEX:
    case GL_VIEWPORT_INDEX_PROVOKING_VERTEX:
        if (ctx.version() < 41 && !ctx.hasExtension("GL_ARB_viewport_array")) return false;
        switch (pname) {
        case GL_MAX_VIEWPORTS: out = ints({ContextState::kMaxViewports}); break;
        case GL_VIEWPORT_SUBPIXEL_BITS: out = ints({8}); break;
        case GL_VIEWPORT_BOUNDS_RANGE: out = floats({-32768.0, 32767.0}); break;
        // Metal takes the layer and viewport of a primitive from one of its
        // vertices without specifying which.
        default: out = ints({GL_UNDEFINED_VERTEX}); break;
        }
        return true;
    case GL_DEPTH_RANGE: out = floats({s.depthRanges[0][0], s.depthRanges[0][1]}, Value::NormFloat); return true;
    case GL_SCISSOR_BOX: out = ints({s.scissors[0][0], s.scissors[0][1], s.scissors[0][2], s.scissors[0][3]}); return true;
    case GL_COLOR_CLEAR_VALUE:
        out = floats({s.clearColor[0], s.clearColor[1], s.clearColor[2], s.clearColor[3]}, Value::NormFloat);
        return true;
    case GL_DEPTH_CLEAR_VALUE: out = floats({s.clearDepth}, Value::NormFloat); return true;
    case GL_STENCIL_CLEAR_VALUE: out = ints({s.clearStencil}); return true;
    case GL_COLOR_WRITEMASK:
        out = bools({s.colorMask[0][0], s.colorMask[0][1], s.colorMask[0][2], s.colorMask[0][3]});
        return true;
    case GL_DEPTH_WRITEMASK: out = bools({s.depthMask}); return true;
    case GL_STENCIL_WRITEMASK: out = uints({s.stencilFront.writeMask}); return true;
    case GL_STENCIL_BACK_WRITEMASK: out = uints({s.stencilBack.writeMask}); return true;
    case GL_DEPTH_FUNC: out = ints({s.depthFunc}); return true;
    case GL_STENCIL_FUNC: out = ints({s.stencilFront.func}); return true;
    case GL_STENCIL_REF: out = ints({s.stencilFront.ref}); return true;
    case GL_STENCIL_VALUE_MASK: out = uints({s.stencilFront.valueMask}); return true;
    case GL_STENCIL_FAIL: out = ints({s.stencilFront.sfail}); return true;
    case GL_STENCIL_PASS_DEPTH_FAIL: out = ints({s.stencilFront.dpfail}); return true;
    case GL_STENCIL_PASS_DEPTH_PASS: out = ints({s.stencilFront.dppass}); return true;
    case GL_STENCIL_BACK_FUNC: out = ints({s.stencilBack.func}); return true;
    case GL_STENCIL_BACK_REF: out = ints({s.stencilBack.ref}); return true;
    case GL_STENCIL_BACK_VALUE_MASK: out = uints({s.stencilBack.valueMask}); return true;
    case GL_STENCIL_BACK_FAIL: out = ints({s.stencilBack.sfail}); return true;
    case GL_STENCIL_BACK_PASS_DEPTH_FAIL: out = ints({s.stencilBack.dpfail}); return true;
    case GL_STENCIL_BACK_PASS_DEPTH_PASS: out = ints({s.stencilBack.dppass}); return true;
    case GL_BLEND_SRC_RGB: out = ints({s.blend[0].srcRGB}); return true;
    case GL_BLEND_DST_RGB: out = ints({s.blend[0].dstRGB}); return true;
    case GL_BLEND_SRC_ALPHA: out = ints({s.blend[0].srcAlpha}); return true;
    case GL_BLEND_DST_ALPHA: out = ints({s.blend[0].dstAlpha}); return true;
    case GL_BLEND_EQUATION_RGB: out = ints({s.blend[0].equationRGB}); return true;
    case GL_BLEND_EQUATION_ALPHA: out = ints({s.blend[0].equationAlpha}); return true;
    case GL_BLEND_COLOR: out = floats({s.blendColor[0], s.blendColor[1], s.blendColor[2], s.blendColor[3]}, Value::NormFloat); return true;
    case GL_CULL_FACE_MODE: out = ints({s.cullFaceMode}); return true;
    case GL_FRONT_FACE: out = ints({s.frontFace}); return true;
    case GL_POLYGON_MODE: out = ints({s.polygonMode, s.polygonModeBack}); return true;
    case GL_POLYGON_OFFSET_FACTOR: out = floats({s.polygonOffsetFactor}); return true;
    case GL_POLYGON_OFFSET_UNITS: out = floats({s.polygonOffsetUnits}); return true;
    case GL_POLYGON_OFFSET_CLAMP:  // GL_ARB/EXT_polygon_offset_clamp
        if (!ctx.hasExtension("GL_ARB_polygon_offset_clamp")) return false;
        out = floats({s.polygonOffsetClamp});
        return true;
    case GL_LINE_WIDTH: out = floats({s.lineWidth}); return true;
    case GL_POINT_SIZE: out = floats({s.pointSize}); return true;
    case GL_POINT_FADE_THRESHOLD_SIZE: out = floats({s.pointFadeThresholdSize}); return true;
    case GL_POINT_SPRITE_COORD_ORIGIN: out = ints({s.pointSpriteCoordOrigin}); return true;
    case GL_PROVOKING_VERTEX: out = ints({s.provokingVertex}); return true;
    case GL_PRIMITIVE_RESTART_INDEX: out = uints({s.primitiveRestartIndex}); return true;
    case GL_LOGIC_OP_MODE: out = ints({s.logicOp}); return true;
    case GL_SAMPLE_COVERAGE_VALUE: out = floats({s.sampleCoverageValue}); return true;
    case GL_SAMPLE_COVERAGE_INVERT: out = bools({s.sampleCoverageInvert}); return true;
    case GL_SAMPLE_BUFFERS: out = ints({samples(ctx) > 0 ? 1 : 0}); return true;
    case GL_SAMPLES: out = ints({samples(ctx)}); return true;
    case GL_CLAMP_READ_COLOR: out = ints({s.clampReadColorMode}); return true;
    // --- pixel store
    case GL_PACK_ALIGNMENT: out = ints({s.pack.alignment}); return true;
    case GL_UNPACK_ALIGNMENT: out = ints({s.unpack.alignment}); return true;
    case GL_PACK_ROW_LENGTH: out = ints({s.pack.rowLength}); return true;
    case GL_UNPACK_ROW_LENGTH: out = ints({s.unpack.rowLength}); return true;
    case GL_PACK_IMAGE_HEIGHT: out = ints({s.pack.imageHeight}); return true;
    case GL_UNPACK_IMAGE_HEIGHT: out = ints({s.unpack.imageHeight}); return true;
    case GL_PACK_SKIP_PIXELS: out = ints({s.pack.skipPixels}); return true;
    case GL_UNPACK_SKIP_PIXELS: out = ints({s.unpack.skipPixels}); return true;
    case GL_PACK_SKIP_ROWS: out = ints({s.pack.skipRows}); return true;
    case GL_UNPACK_SKIP_ROWS: out = ints({s.unpack.skipRows}); return true;
    case GL_PACK_SKIP_IMAGES: out = ints({s.pack.skipImages}); return true;
    case GL_UNPACK_SKIP_IMAGES: out = ints({s.unpack.skipImages}); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_WIDTH: out = ints({s.unpack.blockWidth}); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT: out = ints({s.unpack.blockHeight}); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_DEPTH: out = ints({s.unpack.blockDepth}); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_SIZE: out = ints({s.unpack.blockSize}); return true;
    case GL_PACK_COMPRESSED_BLOCK_WIDTH: out = ints({s.pack.blockWidth}); return true;
    case GL_PACK_COMPRESSED_BLOCK_HEIGHT: out = ints({s.pack.blockHeight}); return true;
    case GL_PACK_COMPRESSED_BLOCK_DEPTH: out = ints({s.pack.blockDepth}); return true;
    case GL_PACK_COMPRESSED_BLOCK_SIZE: out = ints({s.pack.blockSize}); return true;
    case GL_PACK_SWAP_BYTES: out = bools({s.pack.swapBytes}); return true;
    case GL_UNPACK_SWAP_BYTES: out = bools({s.unpack.swapBytes}); return true;
    case GL_PACK_LSB_FIRST: out = bools({s.pack.lsbFirst}); return true;
    case GL_UNPACK_LSB_FIRST: out = bools({s.unpack.lsbFirst}); return true;
    // --- bindings
    case GL_ARRAY_BUFFER_BINDING: out = ints({nameOf(s.arrayBuffer)}); return true;
    case GL_ELEMENT_ARRAY_BUFFER_BINDING: out = ints({nameOf(s.vertexArray->elementBuffer)}); return true;
    case GL_COPY_READ_BUFFER_BINDING: out = ints({nameOf(s.copyReadBuffer)}); return true;
    case GL_DRAW_INDIRECT_BUFFER_BINDING:
        if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_draw_indirect")) return false;
        out = ints({nameOf(s.drawIndirectBuffer)});
        return true;
    case GL_COPY_WRITE_BUFFER_BINDING: out = ints({nameOf(s.copyWriteBuffer)}); return true;
    case GL_PIXEL_PACK_BUFFER_BINDING: out = ints({nameOf(s.pixelPackBuffer)}); return true;
    case GL_PIXEL_UNPACK_BUFFER_BINDING: out = ints({nameOf(s.pixelUnpackBuffer)}); return true;
    case GL_UNIFORM_BUFFER_BINDING: out = ints({nameOf(s.uniformBuffer)}); return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: out = ints({nameOf(s.transformFeedbackBuffer)}); return true;
    case GL_TEXTURE_BUFFER: out = ints({nameOf(s.textureBuffer)}); return true;
    case GL_VERTEX_ARRAY_BINDING: out = ints({s.vertexArray->name}); return true;
    case GL_DRAW_FRAMEBUFFER_BINDING: out = ints({s.drawFramebuffer ? s.drawFramebuffer->name : 0}); return true;
    case GL_READ_FRAMEBUFFER_BINDING: out = ints({s.readFramebuffer ? s.readFramebuffer->name : 0}); return true;
    case GL_RENDERBUFFER_BINDING: out = ints({s.renderbuffer ? s.renderbuffer->name : 0}); return true;
    case GL_CURRENT_PROGRAM: out = ints({s.program ? s.program->name : 0}); return true;
    case GL_PROGRAM_PIPELINE_BINDING:
        if (ctx.version() < 41 && !ctx.hasExtension("GL_ARB_separate_shader_objects")) return false;
        out = ints({s.programPipeline ? s.programPipeline->name : 0});
        return true;
    case GL_ACTIVE_TEXTURE: out = ints({GL_TEXTURE0 + static_cast<int64_t>(s.activeTexture)}); return true;
    case GL_SAMPLER_BINDING: {
        const auto& unit = s.textureUnits[s.activeTexture];
        out = ints({unit.sampler ? unit.sampler->name : 0});
        return true;
    }
    case GL_TRANSFORM_FEEDBACK_BINDING:
        if (ctx.version() < 40) return false;
        out = ints({s.transformFeedback->name});
        return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE: case GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED:
        if (ctx.version() < 40) return false;
        out = bools({pname == GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE ? s.transformFeedback->active : s.transformFeedback->paused});
        return true;
    case GL_MIN_SAMPLE_SHADING_VALUE:
        if (!ctx.sampleShading()) return false;
        out = floats({s.minSampleShading});
        return true;
    // GL 4.0 interpolateAtOffset: offsets in [-0.5, 0.5 - 1/16], in sixteenths.
    case GL_MIN_FRAGMENT_INTERPOLATION_OFFSET: case GL_MAX_FRAGMENT_INTERPOLATION_OFFSET:
    case GL_FRAGMENT_INTERPOLATION_OFFSET_BITS:
        if (ctx.version() < 40) return false;
        if (pname == GL_FRAGMENT_INTERPOLATION_OFFSET_BITS) out = ints({4});
        else out = floats({pname == GL_MIN_FRAGMENT_INTERPOLATION_OFFSET ? -0.5f : 0.4375f});
        return true;
    case GL_MAX_TRANSFORM_FEEDBACK_BUFFERS: case GL_MAX_VERTEX_STREAMS:
        if (ctx.version() < 40) return false;
        out = ints({4});
        return true;
    case GL_DRAW_BUFFER: case GL_DRAW_BUFFER0:
        out = ints({s.drawFramebuffer ? s.drawFramebuffer->drawBuffers[0] : s.defaultDrawBuffer});
        return true;
    case GL_READ_BUFFER: out = ints({s.readFramebuffer ? s.readFramebuffer->readBuffer : s.defaultReadBuffer}); return true;
    case GL_DOUBLEBUFFER: out = bools({ctx.drawSurface() && ctx.drawSurface()->pixelFormat().double_buffer}); return true;
    case GL_STEREO: out = bools({false}); return true;
    // --- hints
    case GL_LINE_SMOOTH_HINT: out = ints({s.hintLineSmooth}); return true;
    case GL_POLYGON_SMOOTH_HINT: out = ints({s.hintPolygonSmooth}); return true;
    case GL_TEXTURE_COMPRESSION_HINT: out = ints({s.hintTextureCompression}); return true;
    case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: out = ints({s.hintFragmentShaderDerivative}); return true;
    default: break;
    }
    if (pname > GL_DRAW_BUFFER0 && pname < GL_DRAW_BUFFER0 + 8) {
        const int i = static_cast<int>(pname - GL_DRAW_BUFFER0);
        out = ints({s.drawFramebuffer ? s.drawFramebuffer->drawBuffers[i] : GL_NONE});
        return true;
    }
    // GL_NVX_gpu_memory_info / GL_ATI_meminfo (reportedMemoryKiB).
    const auto freeMemory = [] {
        const uint64_t workingSet = Device::get().caps().workingSetBytes;
        return reportedMemoryKiB(workingSet - std::min<uint64_t>(workingSet, Device::get().mtl().currentAllocatedSize));
    };
    if (pname >= GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX && pname <= GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX) {
        const int64_t total = reportedMemoryKiB(Device::get().caps().workingSetBytes);
        const int64_t free = freeMemory();
        switch (pname) {
        case GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX: case GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX: out = ints({total}); break;
        case GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX: out = ints({free}); break;
        default: out = ints({0}); break;  // eviction count / evicted memory
        }
        return true;
    }
    if (pname == GL_VBO_FREE_MEMORY_ATI || pname == GL_TEXTURE_FREE_MEMORY_ATI || pname == GL_RENDERBUFFER_FREE_MEMORY_ATI) {
        const int64_t free = freeMemory();
        out = ints({free, free, 0, 0});  // total free, largest block, auxiliary free, largest auxiliary block
        return true;
    }
    // Compatibility profile state (api/fixed_get.mm).
    {
        double values[16];
        char kind = 'f';
        if (const int n = getFixedState(ctx, pname, values, kind)) {
            if (n < 0) return true;  // error set; nothing is written
            out.kind = kind == 'c' ? Value::NormFloat : kind == 'i' ? Value::Int : kind == 'b' ? Value::Bool : Value::Float;
            out.count = n;
            std::copy(values, values + n, out.v);
            return true;
        }
    }
    const int target = textureBindingTarget(pname);
    if (target >= 0 && (target != kTexCubeMapArray || ctx.version() >= 40)) {
        const RefPtr<Texture>& t = s.textureUnits[s.activeTexture].bound[target];
        out = ints({t ? t->name : 0});
        return true;
    }
    // Capabilities can be queried with glGet as well.
    const GLenum before = ctx.takeError();
    const GLboolean enabled = IsEnabled(ctx, pname);
    const GLenum error = ctx.takeError();
    if (before) ctx.setError(before);
    if (error) return false;
    out = bools({enabled == GL_TRUE});
    return true;
}

bool getIndexed(Context& ctx, GLenum pname, GLuint index, Value& out, bool& badIndex) {
    const ContextState& s = ctx.state;
    badIndex = false;
    switch (pname) {
    // The texture and sampler bindings of a unit (GL 4.4, GL_ARB_multi_bind).
    case GL_TEXTURE_BINDING_1D: case GL_TEXTURE_BINDING_2D: case GL_TEXTURE_BINDING_3D: case GL_TEXTURE_BINDING_1D_ARRAY:
    case GL_TEXTURE_BINDING_2D_ARRAY: case GL_TEXTURE_BINDING_RECTANGLE: case GL_TEXTURE_BINDING_CUBE_MAP:
    case GL_TEXTURE_BINDING_BUFFER: case GL_TEXTURE_BINDING_CUBE_MAP_ARRAY: case GL_TEXTURE_BINDING_2D_MULTISAMPLE:
    case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: case GL_SAMPLER_BINDING: {
        if (ctx.version() < 44 && !ctx.hasExtension("GL_ARB_multi_bind")) return false;
        if (index >= static_cast<GLuint>(limits::kMaxCombinedTextureUnits)) { badIndex = true; return true; }
        const TextureUnit& unit = s.textureUnits[index];
        if (pname == GL_SAMPLER_BINDING) {
            out = ints({unit.sampler ? static_cast<int64_t>(unit.sampler->name) : 0});
            return true;
        }
        static const std::pair<GLenum, GLenum> kTargets[] = {
            {GL_TEXTURE_BINDING_1D, GL_TEXTURE_1D}, {GL_TEXTURE_BINDING_2D, GL_TEXTURE_2D},
            {GL_TEXTURE_BINDING_3D, GL_TEXTURE_3D}, {GL_TEXTURE_BINDING_1D_ARRAY, GL_TEXTURE_1D_ARRAY},
            {GL_TEXTURE_BINDING_2D_ARRAY, GL_TEXTURE_2D_ARRAY}, {GL_TEXTURE_BINDING_RECTANGLE, GL_TEXTURE_RECTANGLE},
            {GL_TEXTURE_BINDING_CUBE_MAP, GL_TEXTURE_CUBE_MAP}, {GL_TEXTURE_BINDING_BUFFER, GL_TEXTURE_BUFFER},
            {GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, GL_TEXTURE_CUBE_MAP_ARRAY},
            {GL_TEXTURE_BINDING_2D_MULTISAMPLE, GL_TEXTURE_2D_MULTISAMPLE},
            {GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_2D_MULTISAMPLE_ARRAY}};
        for (const auto& [binding, target] : kTargets) {
            if (binding != pname) continue;
            const int t = textureTargetIndex(target);
            const Texture* tex = t >= 0 ? unit.bound[static_cast<size_t>(t)].get() : nullptr;
            out = ints({tex ? static_cast<int64_t>(tex->name) : 0});
        }
        return true;
    }
    case GL_MAX_COMPUTE_WORK_GROUP_COUNT: case GL_MAX_COMPUTE_WORK_GROUP_SIZE:
        if (!ctx.computeShaders()) return false;
        if (index >= 3) { badIndex = true; return true; }
        out = ints({pname == GL_MAX_COMPUTE_WORK_GROUP_COUNT ? limits::kMaxComputeWorkGroupCount
                                                             : limits::kMaxComputeWorkGroupSize[index]});
        return true;
    case GL_UNIFORM_BUFFER_BINDING: case GL_UNIFORM_BUFFER_START: case GL_UNIFORM_BUFFER_SIZE: {
        if (index >= static_cast<GLuint>(limits::kMaxUniformBufferBindings)) { badIndex = true; return true; }
        const IndexedBufferBinding& b = s.uniformBufferBindings[index];
        out = pname == GL_UNIFORM_BUFFER_BINDING ? ints({nameOf(b.buffer)})
              : pname == GL_UNIFORM_BUFFER_START ? int64v(b.offset)
                                                 : int64v(b.size);
        return true;
    }
    case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: case GL_TRANSFORM_FEEDBACK_BUFFER_START:
    case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE: {
        if (index >= TransformFeedback::kMaxBuffers) { badIndex = true; return true; }
        const TransformFeedback::Binding& b = s.transformFeedback->buffers[index];
        out = pname == GL_TRANSFORM_FEEDBACK_BUFFER_BINDING ? ints({nameOf(b.buffer)})
              : pname == GL_TRANSFORM_FEEDBACK_BUFFER_START ? int64v(b.offset)
                                                            : int64v(b.size);
        return true;
    }
    case GL_COLOR_WRITEMASK:
        if (index >= static_cast<GLuint>(ContextState::kMaxDrawBuffers)) { badIndex = true; return true; }
        out = bools({s.colorMask[index][0], s.colorMask[index][1], s.colorMask[index][2], s.colorMask[index][3]});
        return true;
    case GL_BLEND:
        if (index >= static_cast<GLuint>(ContextState::kMaxDrawBuffers)) { badIndex = true; return true; }
        out = bools({((s.blendEnable >> index) & 1) != 0});
        return true;
    case GL_BLEND_SRC_RGB: case GL_BLEND_DST_RGB: case GL_BLEND_SRC_ALPHA: case GL_BLEND_DST_ALPHA:
    case GL_BLEND_EQUATION_RGB: case GL_BLEND_EQUATION_ALPHA: {  // GL_ARB_draw_buffers_blend
        if (index >= static_cast<GLuint>(ContextState::kMaxDrawBuffers)) { badIndex = true; return true; }
        const ContextState::Blend& b = s.blend[index];
        out = ints({pname == GL_BLEND_SRC_RGB ? b.srcRGB : pname == GL_BLEND_DST_RGB ? b.dstRGB
                    : pname == GL_BLEND_SRC_ALPHA ? b.srcAlpha : pname == GL_BLEND_DST_ALPHA ? b.dstAlpha
                    : pname == GL_BLEND_EQUATION_RGB ? b.equationRGB : b.equationAlpha});
        return true;
    }
    case GL_VIEWPORT: case GL_DEPTH_RANGE: case GL_SCISSOR_BOX: case GL_SCISSOR_TEST:
        if (ctx.version() < 41 && !ctx.hasExtension("GL_ARB_viewport_array")) return false;
        if (index >= static_cast<GLuint>(ContextState::kMaxViewports)) { badIndex = true; return true; }
        if (pname == GL_VIEWPORT) {
            const GLfloat* v = s.viewports[index];
            out = floats({v[0], v[1], v[2], v[3]});
        } else if (pname == GL_DEPTH_RANGE) {
            out = floats({s.depthRanges[index][0], s.depthRanges[index][1]}, Value::NormFloat);
        } else if (pname == GL_SCISSOR_BOX) {
            const GLint* b = s.scissors[index];
            out = ints({b[0], b[1], b[2], b[3]});
        } else {
            out = bools({((s.scissorTests >> index) & 1) != 0});
        }
        return true;
    case GL_IMAGE_BINDING_NAME: case GL_IMAGE_BINDING_LEVEL: case GL_IMAGE_BINDING_LAYERED: case GL_IMAGE_BINDING_LAYER:
    case GL_IMAGE_BINDING_ACCESS: case GL_IMAGE_BINDING_FORMAT: {
        if (!ctx.imageLoadStore()) return false;
        if (index >= static_cast<GLuint>(ContextState::kMaxImageUnits)) { badIndex = true; return true; }
        const ImageUnit& u = s.imageUnits[index];
        switch (pname) {
        case GL_IMAGE_BINDING_NAME: out = ints({u.texture ? u.texture->name : 0}); break;
        case GL_IMAGE_BINDING_LEVEL: out = ints({u.level}); break;
        case GL_IMAGE_BINDING_LAYERED: out = bools({u.layered}); break;
        case GL_IMAGE_BINDING_LAYER: out = ints({u.layer}); break;
        case GL_IMAGE_BINDING_ACCESS: out = ints({u.access}); break;
        default: out = ints({u.format}); break;
        }
        return true;
    }
    case GL_ATOMIC_COUNTER_BUFFER_BINDING: case GL_ATOMIC_COUNTER_BUFFER_START: case GL_ATOMIC_COUNTER_BUFFER_SIZE: {
        if (!ctx.atomicCounters()) return false;
        if (index >= static_cast<GLuint>(ContextState::kMaxAtomicCounterBufferBindings)) { badIndex = true; return true; }
        const IndexedBufferBinding& b = s.atomicCounterBufferBindings[index];
        out = pname == GL_ATOMIC_COUNTER_BUFFER_BINDING ? ints({nameOf(b.buffer)})
              : pname == GL_ATOMIC_COUNTER_BUFFER_START ? int64v(b.offset)
                                                        : int64v(b.size);
        return true;
    }
    case GL_VERTEX_BINDING_BUFFER: case GL_VERTEX_BINDING_OFFSET: case GL_VERTEX_BINDING_STRIDE: case GL_VERTEX_BINDING_DIVISOR: {
        if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_vertex_attrib_binding")) return false;
        if (index >= static_cast<GLuint>(VertexArray::kMaxBindings)) { badIndex = true; return true; }
        const VertexBinding& b = s.vertexArray->bindings[index];
        out = pname == GL_VERTEX_BINDING_BUFFER   ? ints({nameOf(b.buffer)})
              : pname == GL_VERTEX_BINDING_OFFSET ? int64v(b.offset)
              : pname == GL_VERTEX_BINDING_STRIDE ? ints({b.stride})
                                                  : ints({static_cast<int64_t>(b.divisor)});
        return true;
    }
    case GL_SHADER_STORAGE_BUFFER_BINDING: case GL_SHADER_STORAGE_BUFFER_START: case GL_SHADER_STORAGE_BUFFER_SIZE: {
        if (!ctx.storageBuffers()) return false;
        if (index >= static_cast<GLuint>(ContextState::kMaxShaderStorageBufferBindings)) { badIndex = true; return true; }
        const IndexedBufferBinding& b = s.shaderStorageBufferBindings[index];
        out = pname == GL_SHADER_STORAGE_BUFFER_BINDING ? ints({nameOf(b.buffer)})
              : pname == GL_SHADER_STORAGE_BUFFER_START ? int64v(b.offset)
                                                        : int64v(b.size);
        return true;
    }
    case GL_SAMPLE_MASK_VALUE:
        if (index >= 1) { badIndex = true; return true; }
        out = uints({s.sampleMaskValue});
        return true;
    default: return false;
    }
}

// `es`: OpenGL ES maps normalized values to the 32-bit range for 64-bit
// queries too (OpenGL ES 3.2 section 2.2.2).
template <class T>
void convert(const Value& v, T* params, bool es = false) {
    for (int i = 0; i < v.count; ++i) {
        const double x = v.v[i];
        if constexpr (std::is_same_v<T, GLboolean>) {
            params[i] = x != 0.0 ? GL_TRUE : GL_FALSE;
        } else if constexpr (std::is_floating_point_v<T>) {
            params[i] = static_cast<T>(x);
        } else {
            if (v.kind == Value::NormFloat) {
                // Normalized values map [-1, 1] linearly to the integer range (GL 3.3 section 6.1.2).
                const double scaled = std::clamp(x, -1.0, 1.0) * (sizeof(T) == 8 && !es ? 9223372036854775807.0 : 2147483647.0);
                params[i] = static_cast<T>(std::llround(scaled));
            } else if (v.kind == Value::Float || v.kind == Value::Double) {
                params[i] = static_cast<T>(std::llround(x));
            } else if (v.kind == Value::UInt && sizeof(T) == 4) {
                params[i] = static_cast<T>(static_cast<uint32_t>(x));
            } else if (v.kind == Value::UInt && es) {
                // OpenGL ES: 32-bit unsigned state reads as its signed value in 64 bits.
                params[i] = static_cast<T>(static_cast<int32_t>(static_cast<uint32_t>(x)));
            } else if constexpr (sizeof(T) == 4) {
                params[i] = static_cast<T>(std::clamp(x, -2147483648.0, 2147483647.0));
            } else {
                params[i] = static_cast<T>(x);
            }
        }
    }
}

template <class T>
void get(Context& ctx, GLenum pname, T* params) {
    Value v;
    if (!getState(ctx, pname, v)) return ctx.setError(GL_INVALID_ENUM);
    if (params) convert(v, params, ctx.isEs());
}

template <class T>
void getIndexedT(Context& ctx, GLenum pname, GLuint index, T* params) {
    Value v;
    bool badIndex = false;
    if (!getIndexed(ctx, pname, index, v, badIndex)) return ctx.setError(GL_INVALID_ENUM);
    if (badIndex) return ctx.setError(GL_INVALID_VALUE);
    if (params) convert(v, params, ctx.isEs());
}

}  // namespace

void GetBooleanv(Context& ctx, GLenum pname, GLboolean* data) { get(ctx, pname, data); }
void GetIntegerv(Context& ctx, GLenum pname, GLint* data) { get(ctx, pname, data); }
void GetInteger64v(Context& ctx, GLenum pname, GLint64* data) { get(ctx, pname, data); }
void GetFloatv(Context& ctx, GLenum pname, GLfloat* data) { get(ctx, pname, data); }
void GetDoublev(Context& ctx, GLenum pname, GLdouble* data) { get(ctx, pname, data); }
void GetBooleani_v(Context& ctx, GLenum target, GLuint index, GLboolean* data) { getIndexedT(ctx, target, index, data); }
void GetIntegeri_v(Context& ctx, GLenum target, GLuint index, GLint* data) { getIndexedT(ctx, target, index, data); }
void GetInteger64i_v(Context& ctx, GLenum target, GLuint index, GLint64* data) { getIndexedT(ctx, target, index, data); }
void GetFloati_v(Context& ctx, GLenum target, GLuint index, GLfloat* data) { getIndexedT(ctx, target, index, data); }
void GetDoublei_v(Context& ctx, GLenum target, GLuint index, GLdouble* data) { getIndexedT(ctx, target, index, data); }

}  // namespace wmgl::api
