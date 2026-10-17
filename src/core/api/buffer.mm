// Buffer objects (GL 3.3 section 2.9).

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <utility>
#include <vector>

#include "core/context.h"
#include "core/draw.h"
#include "core/format/formats.h"
#include "core/format/pixel_transfer.h"
#include "core/limits.h"
#include "core/metal/device.h"
#include "gl_api.gen.h"

namespace wmgl::api {
namespace {

bool validUsage(GLenum usage) {
    switch (usage) {
    case GL_STREAM_DRAW: case GL_STREAM_READ: case GL_STREAM_COPY: case GL_STATIC_DRAW: case GL_STATIC_READ:
    case GL_STATIC_COPY: case GL_DYNAMIC_DRAW: case GL_DYNAMIC_READ: case GL_DYNAMIC_COPY: return true;
    default: return false;
    }
}

}  // namespace

// The host's memory for buffer storage (wmgl_set_buffer_memory), or none.
void* (*g_bufferAllocate)(size_t) = nullptr;
void (*g_bufferRelease)(void*, size_t) = nullptr;

namespace {

// Host memory that Metal has let go of. Metal can drop its last reference on
// its own completion thread, where the host (Wine) must not be called, so
// the memory goes back on an application thread, before the next allocation.
struct ReleasedMemory {
    std::mutex lock;
    std::vector<std::pair<void*, size_t>> memory;
};

ReleasedMemory& releasedMemory() {
    static ReleasedMemory* released = new ReleasedMemory;  // never destroyed: Metal may release at exit
    return *released;
}

void returnReleasedMemory() {
    std::vector<std::pair<void*, size_t>> memory;
    {
        std::lock_guard<std::mutex> lock(releasedMemory().lock);
        memory.swap(releasedMemory().memory);
    }
    for (const auto& [pointer, bytes] : memory) g_bufferRelease(pointer, bytes);
}

id<MTLBuffer> newStorage(GLsizeiptr size) {
    if (size <= 0) return nil;
    // Padded to 256 bytes, so texture views of the buffer (texture buffers,
    // buffer images) have room for Metal's aligned rows at any size.
    const NSUInteger length = (static_cast<NSUInteger>(size) + 255) & ~NSUInteger(255);
    if (!g_bufferAllocate) return [Device::get().mtl() newBufferWithLength:length options:MTLResourceStorageModeShared];
    // In the host's memory, in whole 16 KiB (kernel) pages as Metal needs,
    // zero-filled like Metal's.
    returnReleasedMemory();
    const NSUInteger pages = (length + 0x3fff) & ~NSUInteger(0x3fff);
    void* memory = g_bufferAllocate(pages);
    if (!memory) return nil;
    std::memset(memory, 0, pages);
    return [Device::get().mtl() newBufferWithBytesNoCopy:memory
                                                  length:pages
                                                 options:MTLResourceStorageModeShared
                                             deallocator:^(void* pointer, NSUInteger bytes) {
                                                 std::lock_guard<std::mutex> lock(releasedMemory().lock);
                                                 releasedMemory().memory.emplace_back(pointer, bytes);
                                             }];
}

bool gpuBusy(Context& ctx, const Buffer& buffer) {
    return buffer.lastUseSerial && !ctx.share().queue.isComplete(buffer.lastUseSerial);
}

// Replaces the storage so in-flight work keeps the old contents. Without
// memory for new storage it waits for the GPU and keeps the old one.
void rename(Context& ctx, Buffer& buffer, bool preserveContents) {
    id<MTLBuffer> fresh = newStorage(buffer.size);
    if (!fresh) {
        ctx.stream().waitFor(buffer.lastUseSerial);
        ++buffer.version;
        return;
    }
    if (preserveContents && buffer.mtl) {
        // Queued GPU writes (copies, transform feedback) must land first.
        if (buffer.lastWriteSerial) ctx.stream().waitFor(buffer.lastWriteSerial);
        std::memcpy(fresh.contents, buffer.mtl.contents, static_cast<size_t>(buffer.size));
    }
    buffer.mtl = fresh;
    buffer.lastUseSerial = buffer.lastWriteSerial = 0;
    ++buffer.version;
}

Buffer* boundBuffer(Context& ctx, GLenum target) {
    RefPtr<Buffer>* binding = ctx.bufferBinding(target);
    if (!binding) {
        ctx.setError(GL_INVALID_ENUM);
        return nullptr;
    }
    if (!*binding) {
        ctx.setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    return binding->get();
}

RefPtr<Buffer> bufferForBind(Context& ctx, GLuint name, bool& ok) {
    ok = true;
    if (!name) return nullptr;
    ShareLock lock(ctx);
    RefPtr<Buffer> buffer = ctx.share().buffers.getRef(name);
    if (buffer) return buffer;
    if (!ctx.share().buffers.isGenerated(name)) {
        if (ctx.isDesktopCore()) {  // OpenGL ES creates the object, as the compatibility profile
            ctx.setError(GL_INVALID_OPERATION);
            ok = false;
            return nullptr;
        }
        ctx.share().buffers.reserve(name);
    }
    buffer = makeRef<Buffer>(name);
    ctx.share().buffers.set(name, buffer);
    return buffer;
}

void unmapBuffer(Buffer& buffer) {
    buffer.mapped = false;
    buffer.mapAccess = 0;
    buffer.mapOffset = 0;
    buffer.mapLength = 0;
    buffer.mapPointer = nullptr;
}

}  // namespace

// glInvalidateBuffer(Sub)Data: storage the GPU still uses is replaced by fresh
// storage (the contents become undefined), so the writes that typically
// follow need neither an ordered GPU copy nor a wait. A mapped or
// client-memory buffer keeps its storage. Called by robustness.mm.
void orphanBuffer(Context& ctx, Buffer& buffer);
void orphanBuffer(Context& ctx, Buffer& buffer) {
    if (!buffer.mtl || buffer.mapped || buffer.pinned || !gpuBusy(ctx, buffer)) return;
    rename(ctx, buffer, false);
    ctx.markDirty(kDirtyVertexInput | kDirtyUniformBuffers);
}

void GenBuffers(Context& ctx, GLsizei n, GLuint* buffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!buffers) return;
    ShareLock lock(ctx);
    ctx.share().buffers.generate(n, buffers);
}

void DeleteBuffers(Context& ctx, GLsizei n, const GLuint* buffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!buffers) return;
    ContextState& s = ctx.state;
    for (GLsizei i = 0; i < n; ++i) {
        if (!buffers[i]) continue;
        RefPtr<Buffer> buffer;
        {
            ShareLock lock(ctx);
            buffer = ctx.share().buffers.remove(buffers[i]);
        }
        if (!buffer) continue;
        if (buffer->mapped) unmapBuffer(*buffer);
        // Deleting a bound buffer resets the bindings of the current context.
        for (RefPtr<Buffer>* b : {&s.arrayBuffer, &s.copyReadBuffer, &s.copyWriteBuffer, &s.pixelPackBuffer,
                                  &s.pixelUnpackBuffer, &s.textureBuffer, &s.uniformBuffer, &s.transformFeedbackBuffer,
                                  &s.drawIndirectBuffer, &s.dispatchIndirectBuffer, &s.shaderStorageBuffer,
                                  &s.atomicCounterBuffer, &s.queryBuffer, &s.parameterBuffer, &s.externalVirtualMemoryBuffer})
            if (*b == buffer) b->reset();
        if (s.vertexArray->elementBuffer == buffer) s.vertexArray->elementBuffer.reset();
        for (VertexAttrib& a : s.vertexArray->attribs)
            if (a.buffer == buffer) a.buffer.reset();
        for (VertexBinding& b : s.vertexArray->bindings)
            if (b.buffer == buffer) b.buffer.reset();
        for (IndexedBufferBinding& b : s.uniformBufferBindings)
            if (b.buffer == buffer) b = IndexedBufferBinding();
        for (IndexedBufferBinding& b : s.atomicCounterBufferBindings)
            if (b.buffer == buffer) b = IndexedBufferBinding();
        for (IndexedBufferBinding& b : s.shaderStorageBufferBindings)
            if (b.buffer == buffer) b = IndexedBufferBinding();
        for (auto& b : s.transformFeedback->buffers)
            if (b.buffer == buffer) b = TransformFeedback::Binding();
        ctx.markDirty(kDirtyVertexInput | kDirtyUniformBuffers);
    }
}

GLboolean IsBuffer(Context& ctx, GLuint buffer) {
    ShareLock lock(ctx);
    return ctx.share().buffers.exists(buffer) ? GL_TRUE : GL_FALSE;
}

void BindBuffer(Context& ctx, GLenum target, GLuint name) {
    RefPtr<Buffer>* binding = ctx.bufferBinding(target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    bool ok;
    RefPtr<Buffer> buffer = bufferForBind(ctx, name, ok);
    if (!ok) return;
    *binding = buffer;
    if (target == GL_ELEMENT_ARRAY_BUFFER) ++ctx.state.vertexArray->version;
    ctx.markDirty(kDirtyVertexInput);
}

namespace {

bool indexedTarget(Context& ctx, GLenum target) {
    return target == GL_UNIFORM_BUFFER || target == GL_TRANSFORM_FEEDBACK_BUFFER ||
           (target == GL_ATOMIC_COUNTER_BUFFER && ctx.atomicCounters()) ||
           (target == GL_SHADER_STORAGE_BUFFER && ctx.storageBuffers());
}

}  // namespace

void BindBufferRange(Context& ctx, GLenum target, GLuint index, GLuint name, GLintptr offset, GLsizeiptr size) {
    if (!indexedTarget(ctx, target)) return ctx.setError(GL_INVALID_ENUM);
    if (name) {
        if (size <= 0 || offset < 0) return ctx.setError(GL_INVALID_VALUE);
        if (target == GL_UNIFORM_BUFFER && offset % 256) return ctx.setError(GL_INVALID_VALUE);
        if (target == GL_TRANSFORM_FEEDBACK_BUFFER && (offset % 4 || size % 4)) return ctx.setError(GL_INVALID_VALUE);
        if (target == GL_ATOMIC_COUNTER_BUFFER && offset % 4) return ctx.setError(GL_INVALID_VALUE);
    }
    // Checked with buffer 0 too (KHR-GL43.shader_storage_buffer_object.negative-api-bind).
    if (target == GL_SHADER_STORAGE_BUFFER && offset % limits::kShaderStorageBufferOffsetAlignment)
        return ctx.setError(GL_INVALID_VALUE);
    // Binding buffer 0 clears the range (GL_*_START and GL_*_SIZE read 0).
    if (!name) offset = size = 0;
    bool ok;
    RefPtr<Buffer> buffer = bufferForBind(ctx, name, ok);
    if (!ok) return;
    if (target == GL_SHADER_STORAGE_BUFFER) {
        if (index >= ContextState::kMaxShaderStorageBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.shaderStorageBufferBindings[index] = {buffer, offset, size};
        ctx.state.shaderStorageBuffer = buffer;
    } else if (target == GL_ATOMIC_COUNTER_BUFFER) {
        if (index >= ContextState::kMaxAtomicCounterBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.atomicCounterBufferBindings[index] = {buffer, offset, size};
        ctx.state.atomicCounterBuffer = buffer;
    } else if (target == GL_UNIFORM_BUFFER) {
        if (index >= ContextState::kMaxUniformBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.uniformBufferBindings[index] = {buffer, offset, size};
        ctx.state.uniformBuffer = buffer;
        ctx.markDirty(kDirtyUniformBuffers);
    } else {
        if (index >= TransformFeedback::kMaxBuffers) return ctx.setError(GL_INVALID_VALUE);
        TransformFeedback& xfb = *ctx.state.transformFeedback;
        if (xfb.active) return ctx.setError(GL_INVALID_OPERATION);
        xfb.buffers[index] = {buffer, offset, size};
        ctx.state.transformFeedbackBuffer = buffer;
    }
}

void BindBufferBase(Context& ctx, GLenum target, GLuint index, GLuint name) {
    if (!indexedTarget(ctx, target)) return ctx.setError(GL_INVALID_ENUM);
    bool ok;
    RefPtr<Buffer> buffer = bufferForBind(ctx, name, ok);
    if (!ok) return;
    if (target == GL_SHADER_STORAGE_BUFFER) {
        if (index >= ContextState::kMaxShaderStorageBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.shaderStorageBufferBindings[index] = {buffer, 0, 0};
        ctx.state.shaderStorageBuffer = buffer;
    } else if (target == GL_ATOMIC_COUNTER_BUFFER) {
        if (index >= ContextState::kMaxAtomicCounterBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.atomicCounterBufferBindings[index] = {buffer, 0, 0};
        ctx.state.atomicCounterBuffer = buffer;
    } else if (target == GL_UNIFORM_BUFFER) {
        if (index >= ContextState::kMaxUniformBufferBindings) return ctx.setError(GL_INVALID_VALUE);
        ctx.state.uniformBufferBindings[index] = {buffer, 0, 0};
        ctx.state.uniformBuffer = buffer;
        ctx.markDirty(kDirtyUniformBuffers);
    } else {
        if (index >= TransformFeedback::kMaxBuffers) return ctx.setError(GL_INVALID_VALUE);
        TransformFeedback& xfb = *ctx.state.transformFeedback;
        if (xfb.active) return ctx.setError(GL_INVALID_OPERATION);
        xfb.buffers[index] = {buffer, 0, 0};
        ctx.state.transformFeedbackBuffer = buffer;
    }
}

// ------------------------------------------------------------ multi-bind (GL 4.4)

namespace {

GLuint indexedBindingCount(GLenum target) {
    switch (target) {
    case GL_SHADER_STORAGE_BUFFER: return ContextState::kMaxShaderStorageBufferBindings;
    case GL_ATOMIC_COUNTER_BUFFER: return ContextState::kMaxAtomicCounterBufferBindings;
    case GL_UNIFORM_BUFFER: return ContextState::kMaxUniformBufferBindings;
    default: return TransformFeedback::kMaxBuffers;
    }
}

RefPtr<Buffer>& genericBinding(Context& ctx, GLenum target) {
    switch (target) {
    case GL_SHADER_STORAGE_BUFFER: return ctx.state.shaderStorageBuffer;
    case GL_ATOMIC_COUNTER_BUFFER: return ctx.state.atomicCounterBuffer;
    case GL_UNIFORM_BUFFER: return ctx.state.uniformBuffer;
    default: return ctx.state.transformFeedbackBuffer;
    }
}

// glBindBuffersBase/Range (GL 4.4 section 6.1.1): a range of indexed bindings
// at once. A range past the last binding changes nothing; otherwise an error
// in one binding leaves that binding alone and the others are still made. The
// generic binding point is not changed.
void bindBuffers(Context& ctx, const char* command, GLenum target, GLuint first, GLsizei count, const GLuint* buffers,
                 const GLintptr* offsets, const GLsizeiptr* sizes) {
    if (ctx.version() < 44 && !ctx.hasExtension("GL_ARB_multi_bind")) return ctx.unimplemented(command);
    if (!indexedTarget(ctx, target)) return ctx.setError(GL_INVALID_ENUM);
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    if (static_cast<uint64_t>(first) + static_cast<uint64_t>(count) > indexedBindingCount(target))
        return ctx.setError(GL_INVALID_OPERATION);
    if (target == GL_TRANSFORM_FEEDBACK_BUFFER && ctx.state.transformFeedback->active) return ctx.setError(GL_INVALID_OPERATION);
    const RefPtr<Buffer> generic = genericBinding(ctx, target);
    for (GLsizei i = 0; i < count; ++i) {
        const GLuint name = buffers ? buffers[i] : 0;
        if (offsets && sizes && name) BindBufferRange(ctx, target, first + static_cast<GLuint>(i), name, offsets[i], sizes[i]);
        else BindBufferBase(ctx, target, first + static_cast<GLuint>(i), name);
    }
    genericBinding(ctx, target) = generic;
}

}  // namespace

void BindBuffersBase(Context& ctx, GLenum target, GLuint first, GLsizei count, const GLuint* buffers) {
    bindBuffers(ctx, "glBindBuffersBase", target, first, count, buffers, nullptr, nullptr);
}

void BindBuffersRange(Context& ctx, GLenum target, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets,
                      const GLsizeiptr* sizes) {
    bindBuffers(ctx, "glBindBuffersRange", target, first, count, buffers, buffers ? offsets : nullptr, buffers ? sizes : nullptr);
}

void BufferData(Context& ctx, GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    if (size < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!validUsage(usage)) return ctx.setError(GL_INVALID_ENUM);
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    if (buffer->immutable) return ctx.setError(GL_INVALID_OPERATION);
    if (buffer->mapped) unmapBuffer(*buffer);
    if (target == GL_EXTERNAL_VIRTUAL_MEMORY_BUFFER_AMD) {
        // GL_AMD_pinned_memory: `data` (page aligned, owned by the
        // application while the buffer uses it) is the storage itself.
        const uintptr_t page = static_cast<uintptr_t>(getpagesize());
        if (!data || reinterpret_cast<uintptr_t>(data) % page) return ctx.setError(GL_INVALID_OPERATION);
        const NSUInteger length = (static_cast<NSUInteger>(size) + page - 1) & ~static_cast<NSUInteger>(page - 1);
        id<MTLBuffer> storage = size ? [Device::get().mtl() newBufferWithBytesNoCopy:const_cast<void*>(data)
                                                                               length:length
                                                                              options:MTLResourceStorageModeShared
                                                                          deallocator:nil]
                                     : nil;
        if (size && !storage) return ctx.setError(GL_INVALID_OPERATION);
        buffer->mtl = storage;
        buffer->size = size;
        buffer->usage = usage;
        buffer->pinned = true;
        buffer->lastUseSerial = buffer->lastWriteSerial = 0;
        ++buffer->version;
        ctx.markDirty(kDirtyVertexInput | kDirtyUniformBuffers);
        return;
    }
    // Reuse the storage only when no queued work references it (orphaning),
    // and never the application's memory of a pinned buffer.
    const bool wasPinned = buffer->pinned;
    buffer->pinned = false;
    if (buffer->mtl && !wasPinned && buffer->size == size && !gpuBusy(ctx, *buffer)) {
        ++buffer->version;
    } else {
        buffer->mtl = newStorage(size);
        buffer->lastUseSerial = buffer->lastWriteSerial = 0;
        ++buffer->version;
        if (size && !buffer->mtl) {
            buffer->size = 0;
            return ctx.setError(GL_OUT_OF_MEMORY);
        }
    }
    buffer->size = size;
    buffer->usage = usage;
    if (data && size) std::memcpy(buffer->mtl.contents, data, static_cast<size_t>(size));
    ctx.markDirty(kDirtyVertexInput | kDirtyUniformBuffers);
}

// GL_ARB_buffer_storage: immutable storage. Shared Metal memory is coherent
// with the GPU, so persistent and coherent mappings need no extra copies.
void BufferStorage(Context& ctx, GLenum target, GLsizeiptr size, const void* data, GLbitfield flags) {
    const GLbitfield known = GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT |
                             GL_MAP_COHERENT_BIT | GL_CLIENT_STORAGE_BIT;
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    if (size <= 0 || (flags & ~known)) return ctx.setError(GL_INVALID_VALUE);
    if ((flags & GL_MAP_PERSISTENT_BIT) && !(flags & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT))) return ctx.setError(GL_INVALID_VALUE);
    if ((flags & GL_MAP_COHERENT_BIT) && !(flags & GL_MAP_PERSISTENT_BIT)) return ctx.setError(GL_INVALID_VALUE);
    if (buffer->immutable) return ctx.setError(GL_INVALID_OPERATION);
    if (buffer->mapped) unmapBuffer(*buffer);
    id<MTLBuffer> storage = newStorage(size);
    if (!storage) return ctx.setError(GL_OUT_OF_MEMORY);
    if (data) std::memcpy(storage.contents, data, static_cast<size_t>(size));
    buffer->mtl = storage;
    buffer->pinned = false;
    buffer->size = size;
    buffer->usage = GL_DYNAMIC_DRAW;
    buffer->immutable = true;
    buffer->storageFlags = flags;
    buffer->lastUseSerial = buffer->lastWriteSerial = 0;
    ++buffer->version;
    ctx.markDirty(kDirtyVertexInput | kDirtyUniformBuffers);
}

void BufferSubData(Context& ctx, GLenum target, GLintptr offset, GLsizeiptr size, const void* data) {
    if (offset < 0 || size < 0) return ctx.setError(GL_INVALID_VALUE);
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    if (size > buffer->size - offset) return ctx.setError(GL_INVALID_VALUE);
    if (buffer->rangeMapped(offset, size)) return ctx.setError(GL_INVALID_OPERATION);
    if (buffer->immutable && !(buffer->storageFlags & GL_DYNAMIC_STORAGE_BIT)) return ctx.setError(GL_INVALID_OPERATION);
    if (!size || !data || !buffer->mtl) return;
    if (gpuBusy(ctx, *buffer)) {
        // Immutable storage keeps its memory (it may be persistently mapped),
        // and so does a mapped buffer.
        if (!buffer->immutable && !buffer->pinned && !buffer->mapped && (size == buffer->size || buffer->size <= (256 << 10))) {
            // Small or whole-buffer updates: copy-on-write.
            rename(ctx, *buffer, size != buffer->size);
        } else {
            // Large partial update: ordered GPU copy from staging memory.
            TransientAllocation staging = ctx.draw().transient().allocate(static_cast<size_t>(size), ctx.stream().serial(), 16);
            std::memcpy(staging.data, data, static_cast<size_t>(size));
            [ctx.stream().blit(true) copyFromBuffer:staging.buffer sourceOffset:staging.offset toBuffer:buffer->mtl
                                destinationOffset:static_cast<NSUInteger>(offset) size:static_cast<NSUInteger>(size)];
            buffer->lastUseSerial = buffer->lastWriteSerial = ctx.stream().serial();
            ++buffer->version;
            return;
        }
    }
    std::memcpy(static_cast<uint8_t*>(buffer->mtl.contents) + offset, data, static_cast<size_t>(size));
    ++buffer->version;
}

// GL 4.3 / GL_ARB_clear_buffer_object: fills a range with one element of
// `internalformat` (a texture buffer format, GL 4.3 table 8.18) converted
// from one pixel of `data` (zero if null), through an ordered GPU copy.
void ClearBufferSubData(Context& ctx, GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format,
                        GLenum type, const void* data) {
    if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_clear_buffer_object")) return ctx.unimplemented("glClearBufferSubData");
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    size_t element = 0;
    switch (internalformat) {
    case GL_R8: case GL_R8I: case GL_R8UI: element = 1; break;
    case GL_R16: case GL_R16F: case GL_R16I: case GL_R16UI: case GL_RG8: case GL_RG8I: case GL_RG8UI: element = 2; break;
    case GL_R32F: case GL_R32I: case GL_R32UI: case GL_RG16: case GL_RG16F: case GL_RG16I: case GL_RG16UI: case GL_RGBA8:
    case GL_RGBA8I: case GL_RGBA8UI: element = 4; break;
    case GL_RG32F: case GL_RG32I: case GL_RG32UI: case GL_RGBA16: case GL_RGBA16F: case GL_RGBA16I: case GL_RGBA16UI:
        element = 8;
        break;
    case GL_RGB32F: case GL_RGB32I: case GL_RGB32UI: element = 12; break;
    case GL_RGBA32F: case GL_RGBA32I: case GL_RGBA32UI: element = 16; break;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
    const FormatInfo* info = formatInfo(internalformat);
    if (!info || !validTransferFormatType(format, type, false)) return ctx.setError(GL_INVALID_VALUE);
    if (!transferCompatible(format, type, *info)) return ctx.setError(GL_INVALID_OPERATION);
    if (offset < 0 || size < 0 || size > buffer->size - offset) return ctx.setError(GL_INVALID_VALUE);
    if (offset % static_cast<GLintptr>(element) || size % static_cast<GLsizeiptr>(element)) return ctx.setError(GL_INVALID_VALUE);
    if (buffer->rangeMapped(offset, size)) return ctx.setError(GL_INVALID_OPERATION);
    if (!size || !buffer->mtl) return;
    // The element, as stored (RGB32 texels are stored with a fourth component).
    uint8_t texel[16] = {};
    if (data) {
        PixelStoreParams ps;
        ps.alignment = 1;
        unpackPixels(data, format, type, 1, 1, 1, ps, *info, texel);
    }
    TransientAllocation staging = ctx.draw().transient().allocate(static_cast<size_t>(size), ctx.stream().serial(), 16);
    for (size_t at = 0; at < static_cast<size_t>(size); at += element)
        std::memcpy(static_cast<uint8_t*>(staging.data) + at, texel, element);
    [ctx.stream().blit(true) copyFromBuffer:staging.buffer sourceOffset:staging.offset toBuffer:buffer->mtl
                        destinationOffset:static_cast<NSUInteger>(offset) size:static_cast<NSUInteger>(size)];
    buffer->lastUseSerial = buffer->lastWriteSerial = ctx.stream().serial();
    ++buffer->version;
}

void ClearBufferData(Context& ctx, GLenum target, GLenum internalformat, GLenum format, GLenum type, const void* data) {
    if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_clear_buffer_object")) return ctx.unimplemented("glClearBufferData");
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    ClearBufferSubData(ctx, target, internalformat, 0, buffer->size, format, type, data);
}

void GetBufferSubData(Context& ctx, GLenum target, GLintptr offset, GLsizeiptr size, void* data) {
    if (offset < 0 || size < 0) return ctx.setError(GL_INVALID_VALUE);
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    if (size > buffer->size - offset) return ctx.setError(GL_INVALID_VALUE);
    // Any non-persistent mapping, not just an overlapping one (GL 4.6 section 6.6).
    if (buffer->mappedNonPersistent()) return ctx.setError(GL_INVALID_OPERATION);
    if (!size || !data || !buffer->mtl) return;
    waitForGpuUnlocked(ctx, buffer->lastWriteSerial);  // a CPU read waits for GPU writes only
    // Another context may have respecified the buffer during the wait.
    if (!buffer->mtl || size > buffer->size - offset) return ctx.setError(GL_INVALID_VALUE);
    std::memcpy(data, static_cast<uint8_t*>(buffer->mtl.contents) + offset, static_cast<size_t>(size));
}

void* MapBufferRange(Context& ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return nullptr;
    const GLbitfield known = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
                             GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_UNSYNCHRONIZED_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    if (offset < 0 || length < 0 || length > buffer->size - offset || (access & ~known)) {
        ctx.setError(GL_INVALID_VALUE);
        return nullptr;
    }
    // A zero length: INVALID_VALUE in GL 4.6, INVALID_OPERATION in OpenGL ES 3.2 (section 6.3).
    if (length == 0) {
        ctx.setError(ctx.isEs() ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return nullptr;
    }
    const bool read = access & GL_MAP_READ_BIT, write = access & GL_MAP_WRITE_BIT;
    if (buffer->mapped || (!read && !write) ||
        (read && (access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT))) ||
        ((access & GL_MAP_FLUSH_EXPLICIT_BIT) && !write)) {
        ctx.setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    if (buffer->immutable) {
        const GLbitfield flags = buffer->storageFlags;
        if ((read && !(flags & GL_MAP_READ_BIT)) || (write && !(flags & GL_MAP_WRITE_BIT)) ||
            ((access & GL_MAP_PERSISTENT_BIT) && !(flags & GL_MAP_PERSISTENT_BIT)) ||
            ((access & GL_MAP_COHERENT_BIT) && !(flags & GL_MAP_COHERENT_BIT))) {
            ctx.setError(GL_INVALID_OPERATION);
            return nullptr;
        }
    } else if (access & (GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)) {
        ctx.setError(GL_INVALID_OPERATION);  // only immutable storage can be mapped persistently
        return nullptr;
    }
    if (!buffer->mtl) {
        ctx.setError(GL_OUT_OF_MEMORY);
        return nullptr;
    }
    if (!(access & GL_MAP_UNSYNCHRONIZED_BIT) && gpuBusy(ctx, *buffer)) {
        // Immutable storage is never renamed: an invalidating map just waits.
        if ((access & GL_MAP_INVALIDATE_BUFFER_BIT) && !buffer->immutable && !buffer->pinned) rename(ctx, *buffer, false);
        else if ((access & GL_MAP_INVALIDATE_RANGE_BIT) && !read && !buffer->immutable && !buffer->pinned) rename(ctx, *buffer, true);
        // A read-only mapping waits for GPU writes only; a writable one also for GPU reads.
        else waitForGpuUnlocked(ctx, (access & GL_MAP_WRITE_BIT) ? buffer->lastUseSerial : buffer->lastWriteSerial);
    }
    // Another context may have respecified or mapped the buffer during a wait.
    if (buffer->mapped || !buffer->mtl || length > buffer->size - offset) {
        ctx.setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    buffer->mapped = true;
    buffer->mapAccess = access;
    buffer->mapOffset = offset;
    buffer->mapLength = length;
    buffer->mapPointer = static_cast<uint8_t*>(buffer->mtl.contents) + offset;
    return buffer->mapPointer;
}

void* MapBuffer(Context& ctx, GLenum target, GLenum access) {
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return nullptr;
    GLbitfield bits;
    switch (access) {
    case GL_READ_ONLY: bits = GL_MAP_READ_BIT; break;
    case GL_WRITE_ONLY: bits = GL_MAP_WRITE_BIT; break;
    case GL_READ_WRITE: bits = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT; break;
    default: ctx.setError(GL_INVALID_ENUM); return nullptr;
    }
    if (buffer->size == 0) {
        // Mapping an empty buffer succeeds with a null pointer in practice; GL says INVALID_VALUE for MapBufferRange only.
        if (buffer->mapped) { ctx.setError(GL_INVALID_OPERATION); return nullptr; }
        buffer->mapped = true;
        buffer->mapAccess = bits;
        return nullptr;
    }
    return MapBufferRange(ctx, target, 0, buffer->size, bits);
}

GLboolean UnmapBuffer(Context& ctx, GLenum target) {
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return GL_FALSE;
    if (!buffer->mapped) {
        ctx.setError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    unmapBuffer(*buffer);
    ++buffer->version;
    return GL_TRUE;
}

void FlushMappedBufferRange(Context& ctx, GLenum target, GLintptr offset, GLsizeiptr length) {
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer) return;
    if (!buffer->mapped || !(buffer->mapAccess & GL_MAP_FLUSH_EXPLICIT_BIT)) return ctx.setError(GL_INVALID_OPERATION);
    if (offset < 0 || length < 0 || length > buffer->mapLength - offset) return ctx.setError(GL_INVALID_VALUE);
    ++buffer->version;  // shared storage: the CPU writes are already visible
}

void CopyBufferSubData(Context& ctx, GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset,
                       GLsizeiptr size) {
    Buffer* src = boundBuffer(ctx, readTarget);
    if (!src) return;
    Buffer* dst = boundBuffer(ctx, writeTarget);
    if (!dst) return;
    if (readOffset < 0 || writeOffset < 0 || size < 0 || size > src->size - readOffset || size > dst->size - writeOffset)
        return ctx.setError(GL_INVALID_VALUE);
    if (src == dst && std::max(readOffset, writeOffset) < std::min(readOffset, writeOffset) + size)
        return ctx.setError(GL_INVALID_VALUE);
    // Any non-persistent mapping of either buffer (GL 4.6 section 6.7).
    if (src->mappedNonPersistent() || dst->mappedNonPersistent()) return ctx.setError(GL_INVALID_OPERATION);
    if (!size) return;
    [ctx.stream().blit(true) copyFromBuffer:src->mtl sourceOffset:static_cast<NSUInteger>(readOffset) toBuffer:dst->mtl
                        destinationOffset:static_cast<NSUInteger>(writeOffset) size:static_cast<NSUInteger>(size)];
    src->lastUseSerial = dst->lastUseSerial = dst->lastWriteSerial = ctx.stream().serial();
    ++dst->version;
}

void GetBufferParameteri64v(Context& ctx, GLenum target, GLenum pname, GLint64* params) {
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer || !params) return;
    switch (pname) {
    case GL_BUFFER_SIZE: *params = buffer->size; return;
    case GL_BUFFER_USAGE: *params = buffer->usage; return;
    case GL_BUFFER_ACCESS:
        *params = (buffer->mapAccess & GL_MAP_READ_BIT) && (buffer->mapAccess & GL_MAP_WRITE_BIT) ? GL_READ_WRITE
                  : (buffer->mapAccess & GL_MAP_READ_BIT)                                          ? GL_READ_ONLY
                                                                                                   : GL_WRITE_ONLY;
        // OpenGL ES (GL_OES_mapbuffer) maps write-only, its initial value.
        if (!buffer->mapped) *params = ctx.isEs() ? GL_WRITE_ONLY : GL_READ_WRITE;
        return;
    case GL_BUFFER_ACCESS_FLAGS: *params = buffer->mapped ? buffer->mapAccess : 0; return;
    case GL_BUFFER_MAPPED: *params = buffer->mapped ? GL_TRUE : GL_FALSE; return;
    case GL_BUFFER_MAP_OFFSET: *params = buffer->mapped ? buffer->mapOffset : 0; return;
    case GL_BUFFER_MAP_LENGTH: *params = buffer->mapped ? buffer->mapLength : 0; return;
    default: break;
    }
    if (ctx.version() >= 44 || ctx.hasExtension("GL_ARB_buffer_storage")) {
        if (pname == GL_BUFFER_IMMUTABLE_STORAGE) { *params = buffer->immutable; return; }
        if (pname == GL_BUFFER_STORAGE_FLAGS) { *params = buffer->storageFlags; return; }
    }
    ctx.setError(GL_INVALID_ENUM);
}

void GetBufferParameteriv(Context& ctx, GLenum target, GLenum pname, GLint* params) {
    GLint64 value = 0;
    const GLenum before = ctx.takeError();
    GetBufferParameteri64v(ctx, target, pname, &value);
    const GLenum error = ctx.takeError();
    if (before) ctx.setError(before);
    if (error) return ctx.setError(error);
    if (params) *params = static_cast<GLint>(std::clamp<GLint64>(value, INT32_MIN, INT32_MAX));
}

void GetBufferPointerv(Context& ctx, GLenum target, GLenum pname, void** params) {
    if (pname != GL_BUFFER_MAP_POINTER) return ctx.setError(GL_INVALID_ENUM);
    Buffer* buffer = boundBuffer(ctx, target);
    if (!buffer || !params) return;
    *params = buffer->mapped ? buffer->mapPointer : nullptr;
}

// GL 4.2 / GL_ARB_shader_image_load_store.
// GL 4.5 / GL_ARB_texture_barrier: texels written by earlier draws become
// readable by later ones. Ending the render pass orders them (Apple GPUs
// cannot order fragment-stage writes inside one; see CommandStream).
void TextureBarrier(Context& ctx) {
    if (ctx.version() < 45 && !ctx.hasExtension("GL_ARB_texture_barrier")) return ctx.unimplemented("glTextureBarrier");
    ctx.stream().memoryBarrier(true);
}

void MemoryBarrier(Context& ctx, GLbitfield barriers) {
    if (ctx.version() < 42 && !ctx.hasExtension("GL_ARB_shader_image_load_store")) return ctx.unimplemented("glMemoryBarrier");
    if (barriers != GL_ALL_BARRIER_BITS && (barriers & ~0xffefu)) return ctx.setError(GL_INVALID_VALUE);
    ctx.stream().memoryBarrier(false, (barriers & GL_COMMAND_BARRIER_BIT) != 0);
}

}  // namespace wmgl::api
