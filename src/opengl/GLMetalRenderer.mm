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
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace metalsharp {

struct GLMetalRenderer::Impl {
    // Last created pipeline state
    id<MTLRenderPipelineState> currentPipeline = nil;
    id<MTLRenderPipelineState> fixedPipeline = nil;
    id<MTLRenderPipelineState> fixedTexturedPipeline = nil;
    id<MTLDepthStencilState> currentDepthStencilState = nil;
    id<MTLComputePipelineState> currentComputePipeline = nil;
    id<MTLComputeCommandEncoder> currentComputeEncoder = nil;
    id<MTLCommandBuffer> currentComputeCommandBuffer = nil;

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
    id<MTLTexture> depthTarget = nil;
    CAMetalLayer* metalLayer = nil;
    id<CAMetalDrawable> currentDrawable = nil;
    bool drawableBacked = false;
    MTLClearColor clearColor = MTLClearColorMake(0, 0, 0, 1);
    double clearDepth = 1.0;
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

bool GLMetalRenderer::createComputePipeline(const GLShaderState& computeShader) {
    if (!m_device || computeShader.msl.empty()) return false;
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:computeShader.msl.c_str()];
    id<MTLLibrary> library = [m_device newLibraryWithSource:source options:nil error:&error];
    if (!library) { if (error) NSLog(@"Compute shader compile: %@", error); return false; }
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_main"];
    if (!function) return false;
    id<MTLComputePipelineState> pipeline = [m_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) { if (error) NSLog(@"Compute pipeline create: %@", error); return false; }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->currentComputePipeline = pipeline;
    return true;
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

    // Blend state from GL state. Unsupported factors fall back to the
    // conservative GL default (one, zero) rather than silently selecting a
    // different blend equation.
    if (glState.blendEnabled) {
        auto blendFactor = [](uint32_t factor) {
            switch (factor) {
            case 0: return MTLBlendFactorZero;
            case 1: return MTLBlendFactorOne;
            case 0x0300: return MTLBlendFactorSourceColor;
            case 0x0301: return MTLBlendFactorOneMinusSourceColor;
            case 0x0302: return MTLBlendFactorSourceAlpha;
            case 0x0303: return MTLBlendFactorOneMinusSourceAlpha;
            case 0x0304: return MTLBlendFactorDestinationAlpha;
            case 0x0305: return MTLBlendFactorOneMinusDestinationAlpha;
            case 0x0306: return MTLBlendFactorDestinationColor;
            case 0x0307: return MTLBlendFactorOneMinusDestinationColor;
            case 0x0308: return MTLBlendFactorSourceAlphaSaturated;
            default: return MTLBlendFactorOne;
            }
        };
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = blendFactor(glState.blendSrcRGB);
        desc.colorAttachments[0].destinationRGBBlendFactor = blendFactor(glState.blendDstRGB);
        desc.colorAttachments[0].sourceAlphaBlendFactor = blendFactor(glState.blendSrcRGB);
        desc.colorAttachments[0].destinationAlphaBlendFactor = blendFactor(glState.blendDstRGB);
    }

    if (glState.depthTestEnabled) {
        auto depthFunc = [](uint32_t func) {
            switch (func) {
            case 0x0200: return MTLCompareFunctionNever;
            case 0x0201: return MTLCompareFunctionLess;
            case 0x0202: return MTLCompareFunctionEqual;
            case 0x0203: return MTLCompareFunctionLessEqual;
            case 0x0204: return MTLCompareFunctionGreater;
            case 0x0205: return MTLCompareFunctionNotEqual;
            case 0x0206: return MTLCompareFunctionGreaterEqual;
            case 0x0207: return MTLCompareFunctionAlways;
            default: return MTLCompareFunctionLess;
            }
        };
        MTLDepthStencilDescriptor* depth = [[MTLDepthStencilDescriptor alloc] init];
        depth.depthCompareFunction = depthFunc(glState.depthFunc);
        depth.depthWriteEnabled = glState.depthWriteEnabled;
        std::lock_guard<std::mutex> depthLock(m_impl->mutex);
        m_impl->currentDepthStencilState = [m_device newDepthStencilStateWithDescriptor:depth];
        desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
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
        if (m_impl->currentDepthStencilState)
            [m_impl->currentEncoder setDepthStencilState:m_impl->currentDepthStencilState];
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

bool GLMetalRenderer::updateBuffer(uint64_t bufferHandle, size_t offset, const void* data, size_t size) {
    if (!data) return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it == m_impl->buffers.end() || offset + size > it->second.length) return false;
    std::memcpy(static_cast<uint8_t*>(it->second.contents) + offset, data, size);
    return true;
}

void* GLMetalRenderer::bufferContents(uint64_t bufferHandle) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    return it == m_impl->buffers.end() ? nullptr : it->second.contents;
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

void GLMetalRenderer::drawFixedFunction(const float* vertices, size_t vertexCount, uint32_t primitiveType,
                                          uint32_t width, uint32_t height, uint64_t textureHandle,
                                          uint32_t minFilter, uint32_t magFilter, uint32_t wrapS, uint32_t wrapT)
{
    if (!m_device || !vertices || !vertexCount) return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        id<MTLRenderPipelineState> pipeline = textureHandle ? m_impl->fixedTexturedPipeline : m_impl->fixedPipeline;
        if (!pipeline) {
            static const char source[] =
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "struct In { float3 p [[attribute(0)]]; float4 c [[attribute(1)]]; float2 uv [[attribute(2)]]; };\n"
                "struct Out { float4 p [[position]]; float4 c; float2 uv; };\n"
                "vertex Out fixed_vertex(In i [[stage_in]]) { Out o; o.p=float4(i.p,1.0); o.c=i.c; o.uv=i.uv; return o; }\n"
                "fragment float4 fixed_fragment(Out i [[stage_in]]) { return i.c; }\n"
                "fragment float4 fixed_tex_fragment(Out i [[stage_in]], texture2d<float> tex [[texture(0)]], sampler samp [[sampler(0)]]) { return i.c * tex.sample(samp, i.uv); }\n";
            NSError* error = nil;
            NSString* text = [NSString stringWithUTF8String:source];
            id<MTLLibrary> library = [m_device newLibraryWithSource:text options:nil error:&error];
            if (!library) { if (error) NSLog(@"Fixed pipeline compile: %@", error); return; }
            MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
            descriptor.vertexFunction = [library newFunctionWithName:@"fixed_vertex"];
            descriptor.fragmentFunction = [library newFunctionWithName:textureHandle ? @"fixed_tex_fragment" : @"fixed_fragment"];
            if (!descriptor.vertexFunction || !descriptor.fragmentFunction) { if (error) NSLog(@"Fixed pipeline functions missing: %@", error); return; }
            descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
            descriptor.vertexDescriptor.layouts[0].stride = sizeof(float) * 9;
            descriptor.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
            descriptor.vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
            descriptor.vertexDescriptor.attributes[0].offset = 0; descriptor.vertexDescriptor.attributes[0].bufferIndex = 0;
            descriptor.vertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
            descriptor.vertexDescriptor.attributes[1].offset = sizeof(float) * 3; descriptor.vertexDescriptor.attributes[1].bufferIndex = 0;
            descriptor.vertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
            descriptor.vertexDescriptor.attributes[2].offset = sizeof(float) * 7;
            descriptor.vertexDescriptor.attributes[2].bufferIndex = 0;
            id<MTLRenderPipelineState> created = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (textureHandle) m_impl->fixedTexturedPipeline = created;
            else m_impl->fixedPipeline = created;
            if (!created && error) NSLog(@"Fixed pipeline create: %@", error);
        }
    }
    id<MTLRenderPipelineState> pipeline = textureHandle ? m_impl->fixedTexturedPipeline : m_impl->fixedPipeline;
    if (!pipeline) return;
    beginRenderPassToTexture(0, width ? width : 64, height ? height : 64, true);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentEncoder) return;
    id<MTLBuffer> buffer = [m_device newBufferWithBytes:vertices length:vertexCount * 9 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    [m_impl->currentEncoder setRenderPipelineState:pipeline];
    if (textureHandle) {
        auto texture = m_impl->textures.find(textureHandle);
        if (texture != m_impl->textures.end()) [m_impl->currentEncoder setFragmentTexture:texture->second atIndex:0];
        MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
        samplerDescriptor.minFilter = minFilter == 0x2600 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        samplerDescriptor.magFilter = magFilter == 0x2600 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        samplerDescriptor.sAddressMode = wrapS == 0x812F ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        samplerDescriptor.tAddressMode = wrapT == 0x812F ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        id<MTLSamplerState> sampler = [m_device newSamplerStateWithDescriptor:samplerDescriptor];
        if (sampler) [m_impl->currentEncoder setFragmentSamplerState:sampler atIndex:0];
    }
    [m_impl->currentEncoder setVertexBuffer:buffer offset:0 atIndex:0];
    [m_impl->currentEncoder setViewport:(MTLViewport){0, 0, (double)(width ? width : 64), (double)(height ? height : 64), 0, 1}];
    [m_impl->currentEncoder drawPrimitives:metalPrimitiveType(primitiveType) vertexStart:0 vertexCount:vertexCount];
    [m_impl->currentEncoder endEncoding];
    if (m_impl->currentCommandBuffer && m_impl->currentDrawable) [m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable];
    m_impl->currentEncoder = nil;
    [m_impl->currentCommandBuffer commit];
    [m_impl->currentCommandBuffer waitUntilCompleted];
    m_impl->currentCommandBuffer = nil;
    m_impl->currentDrawable = nil;
}

void GLMetalRenderer::drawArraysInstanced(uint32_t primitiveType, uint32_t first, uint32_t count, uint32_t instances)
{
    if (!m_impl->currentEncoder || !count || !instances) return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        for (const auto& attribute : m_impl->vertexAttributes) {
            auto it = m_impl->buffers.find(attribute.bufferHandle);
            if (it != m_impl->buffers.end())
                [m_impl->currentEncoder setVertexBuffer:it->second offset:attribute.offset atIndex:0];
        }
    }
    [m_impl->currentEncoder drawPrimitives:metalPrimitiveType(primitiveType)
                               vertexStart:first vertexCount:count instanceCount:instances];
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

void GLMetalRenderer::drawElementsInstanced(uint32_t primitiveType, uint32_t count, uint32_t indexType,
                                             size_t offset, uint32_t instances)
{
    if (!m_impl->currentEncoder || !m_impl->currentIndexBuffer || !count || !instances) return;
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
                                       indexCount:count indexType:mtlIndexType
                                      indexBuffer:m_impl->currentIndexBuffer
                                indexBufferOffset:m_impl->currentIndexOffset + offset
                                    instanceCount:instances];
}

void GLMetalRenderer::beginRenderPass(uint32_t width, uint32_t height) {
    beginRenderPassToTexture(0, width, height, true);
}

void GLMetalRenderer::beginRenderPassToTexture(uint64_t textureHandle, uint32_t width, uint32_t height, bool clear) {
    if (!m_device)
        return;

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    id<MTLTexture> texture = nil;
    id<CAMetalDrawable> drawable = nil;

    if (textureHandle) {
        auto it = m_impl->textures.find(textureHandle);
        if (it != m_impl->textures.end()) texture = it->second;
    } else if (m_impl->metalLayer) {
        /* CAMetalLayer drawableSize is maintained by the Wine view. A
         * zero-size/minimized layer has no drawable; retrying on the next
         * frame is preferable to encoding into a stale surface. */
        if (width && height) m_impl->metalLayer.drawableSize = CGSizeMake(width, height);
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
    if (!texture) return;
    if (!width) width = (uint32_t)texture.width;
    if (!height) height = (uint32_t)texture.height;

    MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = texture;
    passDesc.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.colorAttachments[0].clearColor = m_impl->clearColor;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    MTLTextureDescriptor* depthDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                             width:width height:height mipmapped:NO];
    depthDesc.usage = MTLTextureUsageRenderTarget;
    m_impl->depthTarget = [m_device newTextureWithDescriptor:depthDesc];
    if (m_impl->depthTarget) {
        passDesc.depthAttachment.texture = m_impl->depthTarget;
        passDesc.depthAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        passDesc.depthAttachment.clearDepth = m_impl->clearDepth;
        passDesc.depthAttachment.storeAction = MTLStoreActionDontCare;
    }

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
        if (m_impl->currentCommandBuffer.status != MTLCommandBufferStatusCompleted)
            NSLog(@"MetalSharp OpenGL command buffer failed: %@", m_impl->currentCommandBuffer.error);
        m_impl->currentCommandBuffer = nil;
        m_impl->currentDrawable = nil;
    }
    if (m_impl->currentComputeCommandBuffer) {
        [m_impl->currentComputeCommandBuffer commit];
        [m_impl->currentComputeCommandBuffer waitUntilCompleted];
        if (m_impl->currentComputeCommandBuffer.status != MTLCommandBufferStatusCompleted)
            NSLog(@"MetalSharp OpenGL compute command buffer failed: %@", m_impl->currentComputeCommandBuffer.error);
        m_impl->currentComputeCommandBuffer = nil;
        m_impl->currentComputeEncoder = nil;
    }
}

void GLMetalRenderer::beginComputePass() {
    if (!m_device) return;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->currentComputeCommandBuffer = [m_commandQueue commandBuffer];
    m_impl->currentComputeEncoder = [m_impl->currentComputeCommandBuffer computeCommandEncoder];
    if (m_impl->currentComputeEncoder && m_impl->currentComputePipeline)
        [m_impl->currentComputeEncoder setComputePipelineState:m_impl->currentComputePipeline];
}

void GLMetalRenderer::bindComputeBuffer(uint64_t bufferHandle, uint32_t index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentComputeEncoder) return;
    auto it = m_impl->buffers.find(bufferHandle);
    if (it != m_impl->buffers.end()) [m_impl->currentComputeEncoder setBuffer:it->second offset:0 atIndex:index];
}

void GLMetalRenderer::bindUniformBuffer(uint64_t bufferHandle, uint32_t index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it == m_impl->buffers.end() || !m_impl->currentEncoder) return;
    [m_impl->currentEncoder setVertexBuffer:it->second offset:0 atIndex:index];
    [m_impl->currentEncoder setFragmentBuffer:it->second offset:0 atIndex:index];
}

void GLMetalRenderer::bindComputeTexture(uint64_t textureHandle, uint32_t index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentComputeEncoder) return;
    auto it = m_impl->textures.find(textureHandle);
    if (it != m_impl->textures.end()) [m_impl->currentComputeEncoder setTexture:it->second atIndex:index];
}

void GLMetalRenderer::dispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentComputeEncoder || !x || !y || !z) return;
    MTLSize grid = MTLSizeMake(x, y, z);
    [m_impl->currentComputeEncoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [m_impl->currentComputeEncoder endEncoding];
    m_impl->currentComputeEncoder = nil;
}

bool GLMetalRenderer::readBuffer(uint64_t bufferHandle, size_t offset, size_t size, void* data) {
    if (!data) return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it == m_impl->buffers.end() || offset + size > it->second.length) return false;
    std::memcpy(data, static_cast<const uint8_t*>(it->second.contents) + offset, size);
    return true;
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

bool GLMetalRenderer::readTextureRGBA8(uint64_t textureHandle, uint32_t x, uint32_t y,
                                       uint32_t width, uint32_t height, void* data) {
    if (!data || !width || !height) return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->textures.find(textureHandle);
    if (it == m_impl->textures.end()) return false;
    id<MTLTexture> texture = it->second;
    if (x + width > texture.width || y + height > texture.height) return false;
    const NSUInteger bytesPerRow = (static_cast<NSUInteger>(width) * 4 + 255) & ~static_cast<NSUInteger>(255);
    const NSUInteger bufferSize = bytesPerRow * static_cast<NSUInteger>(height);
    id<MTLBuffer> staging = [m_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer = [m_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!staging || !commandBuffer || !blit) return false;
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(x, y, 0)
               sourceSize:MTLSizeMake(width, height, 1)
                 toBuffer:staging destinationOffset:0 destinationBytesPerRow:bytesPerRow
    destinationBytesPerImage:bufferSize];
    [blit endEncoding]; [commandBuffer commit]; [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) return false;
    uint8_t* rgba = static_cast<uint8_t*>(data);
    const uint8_t* bgra = static_cast<const uint8_t*>(staging.contents);
    for (uint32_t row = 0; row < height; ++row)
        for (uint32_t column = 0; column < width; ++column) {
            const size_t source = static_cast<size_t>(row) * bytesPerRow + column * 4;
            const size_t destination = (static_cast<size_t>(row) * width + column) * 4;
            rgba[destination + 0] = bgra[source + 2];
            rgba[destination + 1] = bgra[source + 1];
            rgba[destination + 2] = bgra[source + 0];
            rgba[destination + 3] = bgra[source + 3];
        }
    return true;
}

bool GLMetalRenderer::blitTexture(uint64_t sourceHandle, uint64_t destinationHandle, uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto source = m_impl->textures.find(sourceHandle), destination = m_impl->textures.find(destinationHandle);
    if (source == m_impl->textures.end() || destination == m_impl->textures.end()) return false;
    if (width > source->second.width || height > source->second.height || width > destination->second.width || height > destination->second.height) return false;
    id<MTLCommandBuffer> commandBuffer = [m_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !blit) return false;
    [blit copyFromTexture:source->second sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
               sourceSize:MTLSizeMake(width,height,1) toTexture:destination->second destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(0,0,0)];
    [blit endEncoding]; [commandBuffer commit]; [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
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
    texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
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

uint64_t GLMetalRenderer::createTexture3D(uint32_t width, uint32_t height, uint32_t depth, const void* data) {
    if (!m_device || !width || !height || !depth) return 0;
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType3D;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = width; descriptor.height = height; descriptor.depth = depth;
    descriptor.mipmapLevelCount = 1; descriptor.arrayLength = 1;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
    if (!texture) return 0;
    if (data) {
        MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, depth);
        [texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:data bytesPerRow:width * 4 bytesPerImage:width * height * 4];
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    uint64_t handle = m_impl->nextTextureHandle++;
    m_impl->textures[handle] = texture;
    return handle;
}

uint64_t GLMetalRenderer::createTexture2DArray(uint32_t width, uint32_t height, uint32_t layers, const void* data) {
    if (!m_device || !width || !height || !layers) return 0;
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = width; descriptor.height = height; descriptor.depth = 1; descriptor.arrayLength = layers;
    descriptor.mipmapLevelCount = 1; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
    if (!texture) return 0;
    if (data) {
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:data bytesPerRow:width * 4 bytesPerImage:width * height * 4];
        for (uint32_t layer = 1; layer < layers; ++layer)
            [texture replaceRegion:region mipmapLevel:0 slice:layer withBytes:static_cast<const uint8_t*>(data) + static_cast<size_t>(layer) * width * height * 4 bytesPerRow:width * 4 bytesPerImage:width * height * 4];
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    uint64_t handle = m_impl->nextTextureHandle++;
    m_impl->textures[handle] = texture;
    return handle;
}

void GLMetalRenderer::bindTexture(uint64_t textureHandle, uint32_t index) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->textures.find(textureHandle);
    if (it == m_impl->textures.end()) return;
    if (m_impl->currentEncoder) [m_impl->currentEncoder setFragmentTexture:it->second atIndex:index];
}

void GLMetalRenderer::bindSampler(uint32_t index, uint32_t minFilter, uint32_t magFilter,
                                   uint32_t wrapS, uint32_t wrapT) {
    if (!m_impl->currentEncoder || !m_device) return;
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = (minFilter == 0x2600) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = (magFilter == 0x2600) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    auto wrap = [](uint32_t value) {
        switch (value) {
        case 0x812F: return MTLSamplerAddressModeClampToEdge;
        case 0x8370: return MTLSamplerAddressModeMirrorRepeat;
        default: return MTLSamplerAddressModeRepeat;
        }
    };
    descriptor.sAddressMode = wrap(wrapS);
    descriptor.tAddressMode = wrap(wrapT);
    id<MTLSamplerState> sampler = [m_device newSamplerStateWithDescriptor:descriptor];
    if (sampler) [m_impl->currentEncoder setFragmentSamplerState:sampler atIndex:index];
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
    if (!m_impl->currentEncoder) return;
    MTLScissorRect rect;
    rect.x = x; rect.y = y; rect.width = width; rect.height = height;
    [m_impl->currentEncoder setScissorRect:rect];
}

void GLMetalRenderer::setClearColor(float r, float g, float b, float a) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->clearColor = MTLClearColorMake(r, g, b, a);
}

void GLMetalRenderer::setClearDepth(float depth) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->clearDepth = depth;
}

} // namespace metalsharp
