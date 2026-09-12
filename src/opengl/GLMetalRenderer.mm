/// @file GLMetalRenderer.mm
/// @brief Objective-C++ implementation of the GL→Metal draw emitter.
///
/// Compile with -fobjc-arc (set globally for metalsharp_opengl32). We
/// store Objective-C object pointers in fields declared @c id<Protocol>
/// inside the .mm; the @c Impl pimpl also uses @c id directly so we can
/// keep the public header C++-portable via #ifdef __OBJC__.
///
/// The Metal object lifetime is managed by ARC. The Impl struct inherits
/// from Objective-C's root object model only by virtue of holding strong
/// references; we do not need an Obj-C class for Impl because we never
/// allocate it from Obj-C code.
///
/// Threading: every public method acquires @c Impl::mutex before touching
/// shared state. The MTLCommandQueue itself is thread-safe but we serialize
/// Metal command-encoder creation so concurrent begin/end pairs stay
/// well-defined.
///
/// Phase 3d-3k extensions:
///   * setVertexLayout()  → stash stride/offsets/formats; applied to the
///                          MTLRenderPipelineDescriptor inside createPipeline.
///   * updateUniformBuffer() → per-binding shared MTLBuffer; bound via
///                              setFragmentBuffer:offset:atIndex:.
///   * createTexture()/bindTexture() → fragment texture handle table.
///   * setViewport()/setScissor()  → MTLRenderCommandEncoder state.
///   * flush()    → commit current command buffer (no wait).
///   * finish()   → commit + waitUntilCompleted (glFinish equivalent).

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <algorithm>
#include <metalsharp/GLMetalRenderer.h>
#include <metalsharp/GLShaderTracker.h>
#include <metalsharp/OpenGLBridge.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace metalsharp {

struct GLMetalRenderer::Impl {
    // Last created pipeline state
    id<MTLRenderPipelineState> currentPipeline = nil;

    // Buffer tracking (vertex buffers and uniform buffers).
    // Uniform buffers live alongside vertex buffers under a separate handle
    // space because we drive fragment-shader uniform writes through
    // updateUniformBuffer() at a different binding index than the vertex
    // buffers used for drawPrimitives().
    std::unordered_map<uint64_t, id<MTLBuffer>> buffers;
    uint64_t nextBufferHandle = 1;

    // Per-binding fragment uniform buffers. Keyed by `binding` (the argument
    // index passed to updateUniformBuffer); the buffer's contents are
    // updated in place via didModifyRange:.
    std::unordered_map<uint32_t, id<MTLBuffer>> uniformBuffers;

    // Texture tracking. Maps handle → MTLTexture.
    std::unordered_map<uint64_t, id<MTLTexture>> textures;
    uint64_t nextTextureHandle = 1;

    // Active render command encoder
    id<MTLRenderCommandEncoder> currentEncoder = nil;

    // Current command buffer. Created in beginRenderPass() and retained
    // here until endRenderPass() (or flush()/finish()) commits it.
    id<MTLCommandBuffer> currentCommandBuffer = nil;

    // Current render pass descriptor
    MTLRenderPassDescriptor* currentPassDescriptor = nil;
    id<MTLTexture> colorTarget = nil;
    CAMetalLayer* metalLayer = nil;
    id<CAMetalDrawable> currentDrawable = nil;
    bool drawableBacked = false;
    id<MTLBuffer> currentIndexBuffer = nil;
    size_t currentIndexOffset = 0;

    struct VertexAttribute {
        uint32_t index = 0;
        MTLVertexFormat format = MTLVertexFormatInvalid;
        uint32_t stride = 0;
        uint64_t bufferHandle = 0;
        size_t offset = 0;
    };
    std::vector<VertexAttribute> vertexAttributes;

    // Pending vertex layout (Phase 3d). stride==0 means "no layout set";
    // createPipeline copies these into MTLRenderPipelineDescriptor.
    uint32_t vertexStride = 0;
    std::vector<uint32_t> vertexAttributeOffsets;
    std::vector<uint32_t> vertexAttributeFormats;

    mutable std::mutex mutex;
};

GLMetalRenderer::GLMetalRenderer() : m_impl(new Impl()) {}

GLMetalRenderer::~GLMetalRenderer() {
    delete m_impl;
    m_device = nil;
    m_commandQueue = nil;
}

bool GLMetalRenderer::init() {
    m_device = MTLCreateSystemDefaultDevice();
    if (!m_device)
        return false;
    m_commandQueue = [m_device newCommandQueue];
    return m_commandQueue != nil;
}

void GLMetalRenderer::setMetalLayer(void* layer) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->metalLayer = (__bridge CAMetalLayer*)layer;
    if (!m_impl->metalLayer) {
        m_impl->currentDrawable = nil;
        m_impl->drawableBacked = false;
    }
}

bool GLMetalRenderer::isDrawableBacked() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->drawableBacked;
}

bool GLMetalRenderer::createPipeline(const GLShaderState& vertexShader, const GLShaderState& fragmentShader,
                                     const GLState& glState) {
    if (!m_device)
        return false;
    if (vertexShader.msl.empty() || fragmentShader.msl.empty())
        return false;

    NSError* error = nil;

    // Compile vertex MSL
    NSString* vSrc = [NSString stringWithUTF8String:vertexShader.msl.c_str()];
    id<MTLLibrary> vLib = [m_device newLibraryWithSource:vSrc options:nil error:&error];
    if (!vLib) {
        if (error)
            NSLog(@"Vertex shader compile: %@", error);
        return false;
    }

    // Compile fragment MSL
    NSString* fSrc = [NSString stringWithUTF8String:fragmentShader.msl.c_str()];
    id<MTLLibrary> fLib = [m_device newLibraryWithSource:fSrc options:nil error:&error];
    if (!fLib) {
        if (error)
            NSLog(@"Fragment shader compile: %@", error);
        return false;
    }

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = [vLib newFunctionWithName:@"vertex_main"];
    desc.fragmentFunction = [fLib newFunctionWithName:@"fragment_main"];

    // Default color attachment
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    // Blend state from GL state
    if (glState.blendEnabled) {
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorZero;
        // Phase 3c: proper GL→Metal blend mapping in later refinement
    }

    // Phase 3d: apply pending vertex layout (if any) to the descriptor.
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->vertexAttributes.empty()) {
        for (const auto& attribute : m_impl->vertexAttributes) {
            if (attribute.index >= 31 || attribute.format == MTLVertexFormatInvalid) continue;
            desc.vertexDescriptor.attributes[attribute.index].format = attribute.format;
            desc.vertexDescriptor.attributes[attribute.index].offset = attribute.offset;
            desc.vertexDescriptor.attributes[attribute.index].bufferIndex = 0;
            desc.vertexDescriptor.layouts[0].stride = attribute.stride;
        }
        desc.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    } else if (m_impl->vertexStride != 0 && !m_impl->vertexAttributeFormats.empty()) {
        // Compatibility API for callers that provide a complete interleaved
        // layout in one shot.
        desc.vertexDescriptor.layouts[0].stride = m_impl->vertexStride;
        desc.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

        const uint32_t count = static_cast<uint32_t>(m_impl->vertexAttributeFormats.size());
        for (uint32_t i = 0; i < count; ++i) {
            desc.vertexDescriptor.attributes[i].format =
                static_cast<MTLVertexFormat>(m_impl->vertexAttributeFormats[i]);
            desc.vertexDescriptor.attributes[i].offset = m_impl->vertexAttributeOffsets[i];
            desc.vertexDescriptor.attributes[i].bufferIndex = 0;
        }
    }

    m_impl->currentPipeline = [m_device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!m_impl->currentPipeline) {
        if (error)
            NSLog(@"Pipeline create: %@", error);
        return false;
    }

    return true;
}

void GLMetalRenderer::usePipeline() {
    if (m_impl->currentEncoder && m_impl->currentPipeline) {
        [m_impl->currentEncoder setRenderPipelineState:m_impl->currentPipeline];
    }
}

uint64_t GLMetalRenderer::createBuffer(const void* data, size_t size) {
    if (!m_device || size == 0)
        return 0;
    id<MTLBuffer> buf;
    if (data)
        buf = [m_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    else
        buf = [m_device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buf)
        return 0;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    uint64_t handle = m_impl->nextBufferHandle++;
    m_impl->buffers[handle] = buf;
    return handle;
}

void GLMetalRenderer::bindVertexBuffer(uint64_t bufferHandle, size_t offset, uint32_t index) {
    if (!m_impl->currentEncoder)
        return;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it != m_impl->buffers.end()) {
        [m_impl->currentEncoder setVertexBuffer:it->second offset:offset atIndex:index];
    }
}

void GLMetalRenderer::drawArrays(uint32_t primitiveType, uint32_t first, uint32_t count) {
    if (!m_impl->currentEncoder)
        return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        for (const auto& attribute : m_impl->vertexAttributes) {
            auto it = m_impl->buffers.find(attribute.bufferHandle);
            if (it != m_impl->buffers.end())
                [m_impl->currentEncoder setVertexBuffer:it->second offset:attribute.offset atIndex:0];
        }
    }
    // Map GL primitive type to Metal
    MTLPrimitiveType mtlType = MTLPrimitiveTypeTriangle;
    switch (primitiveType) {
    case 0x0000:
        mtlType = MTLPrimitiveTypePoint;
        break; // GL_POINTS
    case 0x0001:
        mtlType = MTLPrimitiveTypeLine;
        break; // GL_LINES
    case 0x0003:
        mtlType = MTLPrimitiveTypeLineStrip;
        break; // GL_LINE_STRIP
    case 0x0004:
        mtlType = MTLPrimitiveTypeTriangle;
        break; // GL_TRIANGLES
    case 0x0005:
        mtlType = MTLPrimitiveTypeTriangleStrip;
        break; // GL_TRIANGLE_STRIP
    default:
        break;
    }
    [m_impl->currentEncoder drawPrimitives:mtlType vertexStart:first vertexCount:count];
}

static MTLPrimitiveType metalPrimitiveType(uint32_t primitiveType)
{
    switch (primitiveType) {
    case 0x0000: return MTLPrimitiveTypePoint;
    case 0x0001: return MTLPrimitiveTypeLine;
    case 0x0003: return MTLPrimitiveTypeLineStrip;
    case 0x0004: return MTLPrimitiveTypeTriangle;
    case 0x0005: return MTLPrimitiveTypeTriangleStrip;
    case 0x0006: return MTLPrimitiveTypeTriangleStrip;
    default: return MTLPrimitiveTypeTriangle;
    }
}

void GLMetalRenderer::bindIndexBuffer(uint64_t bufferHandle, size_t offset)
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it != m_impl->buffers.end()) {
        m_impl->currentIndexBuffer = it->second;
        m_impl->currentIndexOffset = offset;
    } else {
        m_impl->currentIndexBuffer = nil;
        m_impl->currentIndexOffset = 0;
    }
}

void GLMetalRenderer::drawElements(uint32_t primitiveType, uint32_t count, uint32_t indexType, size_t offset)
{
    if (!m_impl->currentEncoder || !m_impl->currentIndexBuffer || !count)
        return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        for (const auto& attribute : m_impl->vertexAttributes) {
            auto it = m_impl->buffers.find(attribute.bufferHandle);
            if (it != m_impl->buffers.end())
                [m_impl->currentEncoder setVertexBuffer:it->second offset:attribute.offset atIndex:0];
        }
    }
    MTLIndexType mtlIndexType;
    switch (indexType) {
    case 0x1403: mtlIndexType = MTLIndexTypeUInt16; break;
    case 0x1405: mtlIndexType = MTLIndexTypeUInt32; break;
    default: return;
    }
    [m_impl->currentEncoder drawIndexedPrimitives:metalPrimitiveType(primitiveType)
                                       indexCount:count
                                        indexType:mtlIndexType
                                      indexBuffer:m_impl->currentIndexBuffer
                                indexBufferOffset:m_impl->currentIndexOffset + offset];
}

void GLMetalRenderer::beginRenderPass(uint32_t width, uint32_t height) {
    if (!m_device)
        return;

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    id<MTLTexture> texture = nil;
    id<CAMetalDrawable> drawable = nil;

    if (m_impl->metalLayer) {
        /* CAMetalLayer drawableSize is maintained by the Wine view. A
         * zero-size/minimized layer has no drawable; retrying on the next
         * frame is preferable to encoding into a stale surface. */
        if (width && height)
            m_impl->metalLayer.drawableSize = CGSizeMake(width, height);
        drawable = [m_impl->metalLayer nextDrawable];
        texture = drawable.texture;
    }

    if (!texture) {
        MTLTextureDescriptor* texDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                 width:width
                                                                height:height
                                                             mipmapped:NO];
        texDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        texture = [m_device newTextureWithDescriptor:texDesc];
    }
    if (!texture)
        return;

    MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = texture;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLCommandBuffer> cmdBuf = [m_commandQueue commandBuffer];
    m_impl->colorTarget = texture;
    m_impl->currentDrawable = drawable;
    m_impl->drawableBacked = drawable != nil;
    m_impl->currentCommandBuffer = cmdBuf;
    m_impl->currentPassDescriptor = passDesc;
    m_impl->currentEncoder = [cmdBuf renderCommandEncoderWithDescriptor:passDesc];
}

void GLMetalRenderer::endRenderPass() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->currentEncoder)
        [m_impl->currentEncoder endEncoding];
    if (m_impl->currentCommandBuffer && m_impl->currentDrawable)
        [m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable];
    m_impl->currentEncoder = nil;
    // The command buffer stays live so flush()/finish() can commit it.
}

void GLMetalRenderer::flush() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->currentCommandBuffer) {
        [m_impl->currentCommandBuffer commit];
        // After commit the buffer is read-only; clear so the next beginRenderPass
        // gets a fresh one.
        m_impl->currentCommandBuffer = nil;
        m_impl->currentDrawable = nil;
    }
}

void GLMetalRenderer::finish() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->currentCommandBuffer) {
        [m_impl->currentCommandBuffer commit];
        [m_impl->currentCommandBuffer waitUntilCompleted];
        if (m_impl->currentCommandBuffer.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"MetalSharp OpenGL command buffer failed: %@",
                  m_impl->currentCommandBuffer.error);
        }
        m_impl->currentCommandBuffer = nil;
        m_impl->currentDrawable = nil;
    }
}

bool GLMetalRenderer::readPixelsRGBA8(uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* data) {
    if (!data || !width || !height)
        return false;

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    id<MTLTexture> texture = m_impl->colorTarget;
    if (!texture || x + width > texture.width || y + height > texture.height)
        return false;

    // Render targets are normally private Metal resources. Copy the requested
    // rectangle into a shared, 256-byte-row-aligned staging buffer before
    // exposing it to the guest. This works for both private and shared
    // textures and is the Metal equivalent of a synchronous glReadPixels.
    const NSUInteger bytesPerRow = (static_cast<NSUInteger>(width) * 4 + 255) & ~static_cast<NSUInteger>(255);
    const NSUInteger bufferSize = bytesPerRow * static_cast<NSUInteger>(height);
    id<MTLBuffer> staging = [m_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer = [m_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!staging || !commandBuffer || !blit)
        return false;
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(x, y, 0)
               sourceSize:MTLSizeMake(width, height, 1)
                 toBuffer:staging
        destinationOffset:0
   destinationBytesPerRow:bytesPerRow
 destinationBytesPerImage:bufferSize];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted)
        return false;

    uint8_t* rgba = static_cast<uint8_t*>(data);
    const uint8_t* bgra = static_cast<const uint8_t*>(staging.contents);
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t column = 0; column < width; ++column) {
            const size_t source = static_cast<size_t>(row) * bytesPerRow + column * 4;
            const size_t destination = (static_cast<size_t>(row) * width + column) * 4;
            rgba[destination + 0] = bgra[source + 2];
            rgba[destination + 1] = bgra[source + 1];
            rgba[destination + 2] = bgra[source + 0];
            rgba[destination + 3] = bgra[source + 3];
        }
    }
    return true;
}

void GLMetalRenderer::setVertexLayout(uint32_t stride, const uint32_t* offsets, const uint32_t* formats,
                                      uint32_t count) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->vertexStride = stride;
    m_impl->vertexAttributeOffsets.clear();
    m_impl->vertexAttributeFormats.clear();
    if (offsets == nullptr || formats == nullptr || count == 0) {
        return;
    }
    m_impl->vertexAttributeOffsets.assign(offsets, offsets + count);
    m_impl->vertexAttributeFormats.assign(formats, formats + count);
}

void GLMetalRenderer::setVertexAttribute(uint32_t index, int32_t size, uint32_t type, bool normalized,
                                          uint32_t stride, uint64_t bufferHandle, size_t offset)
{
    MTLVertexFormat format = MTLVertexFormatInvalid;
    if (type == 0x1406) {
        switch (size) {
        case 1: format = MTLVertexFormatFloat; break;
        case 2: format = MTLVertexFormatFloat2; break;
        case 3: format = MTLVertexFormatFloat3; break;
        case 4: format = MTLVertexFormatFloat4; break;
        }
    } else if (type == 0x1401 && size == 4) {
        format = normalized ? MTLVertexFormatUChar4Normalized : MTLVertexFormatUChar4;
    } else if (type == 0x1403 && size == 4) {
        format = normalized ? MTLVertexFormatUShort4Normalized : MTLVertexFormatUShort4;
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = std::find_if(m_impl->vertexAttributes.begin(), m_impl->vertexAttributes.end(),
                           [index](const auto& attribute) { return attribute.index == index; });
    Impl::VertexAttribute attribute{index, format, stride, bufferHandle, offset};
    if (it == m_impl->vertexAttributes.end()) m_impl->vertexAttributes.push_back(attribute);
    else *it = attribute;
}

void GLMetalRenderer::updateUniformBuffer(uint32_t binding, const void* data, size_t size) {
    if (!m_device)
        return;

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    // Find-or-create the backing buffer at this binding. We grow monotonically
    // — never shrink — to keep the binding handle stable across frames. Using
    // shared storage lets the CPU write directly without a staging copy.
    id<MTLBuffer> buf = nil;
    auto it = m_impl->uniformBuffers.find(binding);
    if (it != m_impl->uniformBuffers.end()) {
        buf = it->second;
    }

    if (size == 0 || data == nullptr) {
        // Caller wants to drop the binding — release the buffer.
        if (buf) {
            m_impl->uniformBuffers.erase(binding);
        }
        return;
    }

    if (buf == nullptr || buf.length < size) {
        // Allocate (or grow) the backing buffer. We round up to a 256-byte
        // alignment so a slightly larger payload next frame doesn't force a
        // reallocation.
        size_t allocSize = (size + 255) & ~static_cast<size_t>(255);
        if (allocSize < size)
            allocSize = size; // overflow guard
        buf = [m_device newBufferWithLength:allocSize options:MTLResourceStorageModeShared];
        if (!buf)
            return;
        m_impl->uniformBuffers[binding] = buf;
    }

    memcpy(buf.contents, data, size);
    // If a render pass is currently active, push the updated buffer into the
    // encoder so the next fragment-shader invocation sees the new contents.
    if (m_impl->currentEncoder) {
        [m_impl->currentEncoder setVertexBuffer:buf offset:0 atIndex:binding];
        [m_impl->currentEncoder setFragmentBuffer:buf offset:0 atIndex:binding];
    }
}

uint64_t GLMetalRenderer::createTexture(uint32_t width, uint32_t height, const void* data) {
    if (!m_device || width == 0 || height == 0)
        return 0;

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                       width:width
                                                                                      height:height
                                                                                   mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:texDesc];
    if (!texture)
        return 0;

    if (data) {
        const size_t bytesPerRow = static_cast<size_t>(width) * 4u;
        const MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:bytesPerRow];
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const uint64_t handle = m_impl->nextTextureHandle++;
    m_impl->textures[handle] = texture;
    return handle;
}

void GLMetalRenderer::bindTexture(uint64_t textureHandle, uint32_t index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->textures.find(textureHandle);
    if (it == m_impl->textures.end())
        return;
    if (m_impl->currentEncoder) {
        [m_impl->currentEncoder setFragmentTexture:it->second atIndex:index];
    }
}

void GLMetalRenderer::setViewport(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    if (!m_impl->currentEncoder)
        return;
    MTLViewport vp;
    vp.originX = static_cast<double>(x);
    vp.originY = static_cast<double>(y);
    vp.width = static_cast<double>(width);
    vp.height = static_cast<double>(height);
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [m_impl->currentEncoder setViewport:vp];
}

void GLMetalRenderer::setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    if (!m_impl->currentEncoder)
        return;
    MTLScissorRect rect;
    rect.x = x;
    rect.y = y;
    rect.width = width;
    rect.height = height;
    [m_impl->currentEncoder setScissorRect:rect];
}

} // namespace metalsharp
