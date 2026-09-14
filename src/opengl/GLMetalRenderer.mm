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
#include <cmath>
#include <cstdio>
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
    uint32_t currentStencilReference = 0;
    id<MTLComputePipelineState> currentComputePipeline = nil;
    id<MTLRenderPipelineState> tessellationPipeline = nil;
    id<MTLBuffer> tessellationFactors = nil;
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
    std::unordered_map<uint64_t, uint64_t> resolveTextures;
    std::unordered_map<uint64_t, uint32_t> textureSampleCounts;
    uint64_t nextTextureHandle = 1;

    // Active render command encoder
    id<MTLRenderCommandEncoder> currentEncoder = nil;

    // Current command buffer. Created in beginRenderPass() and retained
    // here until endRenderPass() (or flush()/finish()) commits it.
    id<MTLCommandBuffer> currentCommandBuffer = nil;

    // Current render pass descriptor
    MTLRenderPassDescriptor* currentPassDescriptor = nil;
    id<MTLTexture> colorTarget = nil;
    id<MTLTexture> defaultColorTarget = nil;
    uint32_t defaultColorWidth = 0, defaultColorHeight = 0;
    uint64_t defaultColorHandle = 0;
    id<MTLTexture> depthTarget = nil;
    CAMetalLayer* metalLayer = nil;
    id<CAMetalDrawable> currentDrawable = nil;
    bool drawableBacked = false;
    int swapInterval = 1;
    MTLClearColor clearColor = MTLClearColorMake(0, 0, 0, 1);
    double clearDepth = 1.0;
    uint32_t clearStencil = 0;
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

void GLMetalRenderer::setSwapInterval(int interval) { std::lock_guard<std::mutex> lock(m_impl->mutex);m_impl->swapInterval=std::max(0,std::min(4,interval)); }

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

bool GLMetalRenderer::createTessellationPipeline(const GLShaderState& evaluationShader,
                                                  const GLShaderState& fragmentShader, const GLState& glState, bool quad) {
    if (!m_device || evaluationShader.msl.empty() || fragmentShader.msl.empty()) return false;
    NSError* error = nil;
    id<MTLLibrary> evalLibrary = [m_device newLibraryWithSource:[NSString stringWithUTF8String:evaluationShader.msl.c_str()] options:nil error:&error];
    if (!evalLibrary) return false;
    id<MTLLibrary> fragmentLibrary = [m_device newLibraryWithSource:[NSString stringWithUTF8String:fragmentShader.msl.c_str()] options:nil error:&error];
    if (!fragmentLibrary) return false;
    MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = [evalLibrary newFunctionWithName:@"tess_eval_main"];
    descriptor.fragmentFunction = [fragmentLibrary newFunctionWithName:@"fragment_main"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
    descriptor.maxTessellationFactor = 64;
    descriptor.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    descriptor.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
    descriptor.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
    descriptor.tessellationOutputWindingOrder = MTLWindingClockwise;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    for (const auto& attribute : m_impl->vertexAttributes) {
        if (attribute.index >= 31 || attribute.format == MTLVertexFormatInvalid) continue;
        descriptor.vertexDescriptor.attributes[attribute.index].format = attribute.format;
        descriptor.vertexDescriptor.attributes[attribute.index].offset = attribute.offset;
        descriptor.vertexDescriptor.attributes[attribute.index].bufferIndex = 30;
        descriptor.vertexDescriptor.layouts[30].stride = attribute.stride;
    }
    descriptor.vertexDescriptor.layouts[30].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
    m_impl->tessellationPipeline = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    return m_impl->tessellationPipeline != nil;
}

bool GLMetalRenderer::createPipeline(const GLShaderState& vertexShader, const GLShaderState& fragmentShader,
                                     const GLState& glState, uint32_t rasterSampleCount) {
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
    desc.rasterSampleCount = std::max<NSUInteger>(1, rasterSampleCount);

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
            case 0x8001: return MTLBlendFactorBlendColor;
            case 0x8002: return MTLBlendFactorOneMinusBlendColor;
            case 0x8003: return MTLBlendFactorBlendAlpha;
            case 0x8004: return MTLBlendFactorOneMinusBlendAlpha;
            default: return MTLBlendFactorOne;
            }
        };
        auto blendOperation = [](uint32_t operation) {
            switch (operation) {
            case 0x800A: return MTLBlendOperationSubtract;
            case 0x800B: return MTLBlendOperationReverseSubtract;
            case 0x8007: return MTLBlendOperationMin;
            case 0x8008: return MTLBlendOperationMax;
            default: return MTLBlendOperationAdd;
            }
        };
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = blendFactor(glState.blendSrcRGB);
        desc.colorAttachments[0].destinationRGBBlendFactor = blendFactor(glState.blendDstRGB);
        desc.colorAttachments[0].sourceAlphaBlendFactor = blendFactor(glState.blendSrcAlpha);
        desc.colorAttachments[0].destinationAlphaBlendFactor = blendFactor(glState.blendDstAlpha);
        desc.colorAttachments[0].rgbBlendOperation = blendOperation(glState.blendEquationRGB);
        desc.colorAttachments[0].alphaBlendOperation = blendOperation(glState.blendEquationAlpha);
    }
    desc.colorAttachments[0].writeMask = (glState.colorMask[0] ? MTLColorWriteMaskRed : 0) |
                                         (glState.colorMask[1] ? MTLColorWriteMaskGreen : 0) |
                                         (glState.colorMask[2] ? MTLColorWriteMaskBlue : 0) |
                                         (glState.colorMask[3] ? MTLColorWriteMaskAlpha : 0);

    if (glState.depthTestEnabled || glState.stencilTestEnabled) {
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
        depth.depthCompareFunction = glState.depthTestEnabled ? depthFunc(glState.depthFunc) : MTLCompareFunctionAlways;
        depth.depthWriteEnabled = glState.depthTestEnabled && glState.depthWriteEnabled;
        {
            std::lock_guard<std::mutex> refLock(m_impl->mutex);
            m_impl->currentStencilReference = static_cast<uint32_t>(glState.stencilRef);
        }
        if (glState.stencilTestEnabled) {
            auto stencilOperation = [](uint32_t operation) {
                switch (operation) {
                case 0x1500: return MTLStencilOperationZero;
                case 0x1E01: return MTLStencilOperationReplace;
                case 0x1E02: return MTLStencilOperationIncrementClamp;
                case 0x1E03: return MTLStencilOperationDecrementClamp;
                case 0x150A: return MTLStencilOperationInvert;
                case 0x8507: return MTLStencilOperationIncrementWrap;
                case 0x8508: return MTLStencilOperationDecrementWrap;
                default: return MTLStencilOperationKeep;
                }
            };
            auto makeStencil = [&](uint32_t func, uint32_t fail, uint32_t depthFail, uint32_t pass, uint32_t readMask, uint32_t writeMask) {
                MTLStencilDescriptor* stencil = [[MTLStencilDescriptor alloc] init];
                stencil.stencilCompareFunction = depthFunc(func);
                stencil.stencilFailureOperation = stencilOperation(fail);
                stencil.depthFailureOperation = stencilOperation(depthFail);
                stencil.depthStencilPassOperation = stencilOperation(pass);
                stencil.readMask = readMask; stencil.writeMask = writeMask;
                return stencil;
            };
            depth.frontFaceStencil = makeStencil(glState.stencilFunc,glState.stencilFail,glState.stencilDepthFail,glState.stencilDepthPass,glState.stencilValueMask,glState.stencilWriteMask);
            depth.backFaceStencil = makeStencil(glState.stencilFuncBack,glState.stencilFailBack,glState.stencilDepthFailBack,glState.stencilDepthPassBack,glState.stencilValueMaskBack,glState.stencilWriteMaskBack);
        }
        std::lock_guard<std::mutex> depthLock(m_impl->mutex);
        m_impl->currentDepthStencilState = [m_device newDepthStencilStateWithDescriptor:depth];
        desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    }

    // Phase 3d: apply pending vertex layout (if any) to the descriptor.
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->vertexAttributes.empty()) {
        for (const auto& attribute : m_impl->vertexAttributes) {
            if (attribute.index >= 31 || attribute.format == MTLVertexFormatInvalid) continue;
            desc.vertexDescriptor.attributes[attribute.index].format = attribute.format;
            desc.vertexDescriptor.attributes[attribute.index].offset = attribute.offset;
            desc.vertexDescriptor.attributes[attribute.index].bufferIndex = 30;
            desc.vertexDescriptor.layouts[30].stride = attribute.stride;
        }
        desc.vertexDescriptor.layouts[30].stepFunction = MTLVertexStepFunctionPerVertex;
    } else if (m_impl->vertexStride != 0 && !m_impl->vertexAttributeFormats.empty()) {
        // Compatibility API for callers that provide a complete interleaved
        // layout in one shot.
        desc.vertexDescriptor.layouts[30].stride = m_impl->vertexStride;
        desc.vertexDescriptor.layouts[30].stepFunction = MTLVertexStepFunctionPerVertex;

        const uint32_t count = static_cast<uint32_t>(m_impl->vertexAttributeFormats.size());
        for (uint32_t i = 0; i < count; ++i) {
            desc.vertexDescriptor.attributes[i].format =
                static_cast<MTLVertexFormat>(m_impl->vertexAttributeFormats[i]);
            desc.vertexDescriptor.attributes[i].offset = m_impl->vertexAttributeOffsets[i];
            desc.vertexDescriptor.attributes[i].bufferIndex = 30;
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
        if (m_impl->currentDepthStencilState) {
            [m_impl->currentEncoder setDepthStencilState:m_impl->currentDepthStencilState];
            [m_impl->currentEncoder setStencilReferenceValue:m_impl->currentStencilReference];
        }
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
        bindVertexAttributes();
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

void GLMetalRenderer::setRasterState(const GLState& glState) {
    if (!m_impl->currentEncoder) return;
    [m_impl->currentEncoder setCullMode:glState.cullEnabled ? (glState.cullFace == 0x0404 ? MTLCullModeFront : MTLCullModeBack) : MTLCullModeNone];
    [m_impl->currentEncoder setFrontFacingWinding:glState.frontFace == 0x0900 ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [m_impl->currentEncoder setDepthClipMode:glState.depthClampEnabled ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
    if (glState.polygonOffsetFill || glState.polygonOffsetLine || glState.polygonOffsetPoint) [m_impl->currentEncoder setDepthBias:glState.polygonOffsetUnits slopeScale:glState.polygonOffsetFactor clamp:0.0f];
    else [m_impl->currentEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
}

void GLMetalRenderer::setBlendColor(const GLState& glState) {
    if (!m_impl->currentEncoder) return;
    [m_impl->currentEncoder setBlendColorRed:glState.blendColor[0] green:glState.blendColor[1] blue:glState.blendColor[2] alpha:glState.blendColor[3]];
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
                                          uint32_t width, uint32_t height, uint64_t textureHandle, uint64_t textureHandle1,
                                          uint32_t minFilter, uint32_t magFilter, uint32_t wrapS, uint32_t wrapT,
                                          bool alphaTest, uint32_t alphaFunc, float alphaRef, const FixedTextureEnvironment& textureEnv, const GLState& glState)
{
    if (!m_device || !vertices || !vertexCount) return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        id<MTLRenderPipelineState> pipeline = nil;
        if (!pipeline) {
            static const char source[] =
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "struct In { float3 p [[attribute(0)]]; float4 c [[attribute(1)]]; float2 uv [[attribute(2)]]; float2 uv1 [[attribute(3)]]; };\n"
                "struct Out { float4 p [[position]]; float4 c; float2 uv; float2 uv1; float pointSize [[point_size]]; };\n"
                "vertex Out fixed_vertex(In i [[stage_in]], constant float& pointSize [[buffer(3)]]) { Out o; o.p=float4(i.p,1.0); o.c=i.c; o.uv=i.uv; o.uv1=i.uv1; o.pointSize=pointSize; return o; }\n"
                "struct Alpha { float ref; uint func; };\n"
                "struct TextureEnv { uint mode; uint combineRGB; uint combineAlpha; uint sourceRGB[3]; uint operandRGB[3]; uint sourceAlpha[3]; uint operandAlpha[3]; uint padding; float4 constantColor; float rgbScale; float alphaScale; };\n"
                "bool alpha_pass(float a, constant Alpha& x) { if(x.func==0x0200) return false; if(x.func==0x0201) return a<x.ref; if(x.func==0x0202) return a==x.ref; if(x.func==0x0203) return a<=x.ref; if(x.func==0x0204) return a>x.ref; if(x.func==0x0205) return a!=x.ref; if(x.func==0x0206) return a>=x.ref; return true; }\n"
                "float4 env_source(uint source,float4 tex,float4 primary,float4 constantValue) { if(source==0x8576)return constantValue; if(source==0x8577)return primary; if(source==0x8578)return primary; return tex; }\n"
                "float3 env_rgb(float4 value,uint operand) { return operand==0x0301?1.0-value.rgb:(operand==0x0302?float3(value.a):(operand==0x0303?float3(1.0-value.a):value.rgb)); }\n"
                "float env_alpha(float4 value,uint operand) { return operand==0x0300?value.r:(operand==0x0301?1.0-value.r:(operand==0x0303?1.0-value.a:value.a)); }\n"
                "float3 combine_rgb(constant TextureEnv& e,float4 tex,float4 primary) { float4 a=env_source(e.sourceRGB[0],tex,primary,e.constantColor),b=env_source(e.sourceRGB[1],tex,primary,e.constantColor),d=env_source(e.sourceRGB[2],tex,primary,e.constantColor); float3 x=env_rgb(a,e.operandRGB[0]),y=env_rgb(b,e.operandRGB[1]),z=env_rgb(d,e.operandRGB[2]); float3 result; if(e.combineRGB==0x1e01)result=x; else if(e.combineRGB==0x0104)result=x+y; else if(e.combineRGB==0x84E7)result=x-y; else if(e.combineRGB==0x8577)result=x+y-0.5; else if(e.combineRGB==0x8574)result=x*y+(1.0-x)*z; else if(e.combineRGB==0x86AE||e.combineRGB==0x86AF)result=float3(dot(x*2.0-1.0,y*2.0-1.0)); else result=x*y; return clamp(result*e.rgbScale,0.0,1.0); }\n"
                "float combine_alpha(constant TextureEnv& e,float4 tex,float4 primary) { float4 a=env_source(e.sourceAlpha[0],tex,primary,e.constantColor),b=env_source(e.sourceAlpha[1],tex,primary,e.constantColor),d=env_source(e.sourceAlpha[2],tex,primary,e.constantColor); float x=env_alpha(a,e.operandAlpha[0]),y=env_alpha(b,e.operandAlpha[1]),z=env_alpha(d,e.operandAlpha[2]); float result=e.combineAlpha==0x1e01?x:e.combineAlpha==0x0104?x+y:e.combineAlpha==0x84E7?x-y:e.combineAlpha==0x8577?x+y-0.5:e.combineAlpha==0x8574?x*y+(1.0-x)*z:x*y; return clamp(result*e.alphaScale,0.0,1.0); }\n"
                "fragment float4 fixed_fragment(Out i [[stage_in]], constant Alpha& a [[buffer(1)]]) { if(!alpha_pass(i.c.a,a)) discard_fragment(); return i.c; }\n"
                "fragment float4 fixed_tex_fragment(Out i [[stage_in]], texture2d<float> tex [[texture(0)]], sampler samp [[sampler(0)]], texture2d<float> tex1 [[texture(1)]], sampler samp1 [[sampler(1)]], constant Alpha& a [[buffer(1)]], constant TextureEnv& env [[buffer(2)]], constant uint& textureCount [[buffer(3)]]) { float4 t=tex.sample(samp,i.uv); float4 c=i.c; if(env.mode==0x1e01)c=t; else if(env.mode==0x2101)c=float4(mix(i.c.rgb,t.rgb,t.a),i.c.a); else if(env.mode==0x0be2)c=float4(mix(i.c.rgb,env.constantColor.rgb,t.rgb),i.c.a*t.a); else if(env.mode==0x0104)c=i.c+t; else if(env.mode==0x8570)c=float4(combine_rgb(env,t,i.c),combine_alpha(env,t,i.c)); else c=i.c*t; if(textureCount>1)c*=tex1.sample(samp1,i.uv1); if(!alpha_pass(c.a,a)) discard_fragment(); return c; }\n";
            NSError* error = nil;
            NSString* text = [NSString stringWithUTF8String:source];
            id<MTLLibrary> library = [m_device newLibraryWithSource:text options:nil error:&error];
            if (!library) { if (error) NSLog(@"Fixed pipeline compile: %@", error); return; }
            MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
            descriptor.vertexFunction = [library newFunctionWithName:@"fixed_vertex"];
            descriptor.fragmentFunction = [library newFunctionWithName:textureHandle ? @"fixed_tex_fragment" : @"fixed_fragment"];
            if (!descriptor.vertexFunction || !descriptor.fragmentFunction) { if (error) NSLog(@"Fixed pipeline functions missing: %@", error); return; }
            descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
            descriptor.vertexDescriptor.layouts[0].stride = sizeof(float) * 11;
            descriptor.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
            descriptor.vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
            descriptor.vertexDescriptor.attributes[0].offset = 0; descriptor.vertexDescriptor.attributes[0].bufferIndex = 0;
            descriptor.vertexDescriptor.attributes[1].format = MTLVertexFormatFloat4;
            descriptor.vertexDescriptor.attributes[1].offset = sizeof(float) * 3; descriptor.vertexDescriptor.attributes[1].bufferIndex = 0;
            descriptor.vertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
            descriptor.vertexDescriptor.attributes[2].offset = sizeof(float) * 7;
            descriptor.vertexDescriptor.attributes[2].bufferIndex = 0;
            descriptor.vertexDescriptor.attributes[3].format = MTLVertexFormatFloat2;
            descriptor.vertexDescriptor.attributes[3].offset = sizeof(float) * 9;
            descriptor.vertexDescriptor.attributes[3].bufferIndex = 0;
            if (glState.blendEnabled) {
                auto blendFactor = [](uint32_t factor) {
                    switch (factor) {
                    case 0: return MTLBlendFactorZero;
                    case 1: return MTLBlendFactorOne;
                    case 0x0302: return MTLBlendFactorSourceAlpha;
                    case 0x0303: return MTLBlendFactorOneMinusSourceAlpha;
                    case 0x0304: return MTLBlendFactorDestinationAlpha;
                    case 0x0305: return MTLBlendFactorOneMinusDestinationAlpha;
                    case 0x0306: return MTLBlendFactorDestinationColor;
                    case 0x0307: return MTLBlendFactorOneMinusDestinationColor;
                    case 0x8001: return MTLBlendFactorBlendColor;
                    case 0x8002: return MTLBlendFactorOneMinusBlendColor;
                    case 0x8003: return MTLBlendFactorBlendAlpha;
                    case 0x8004: return MTLBlendFactorOneMinusBlendAlpha;
                    default: return MTLBlendFactorOne;
                    }
                };
                descriptor.colorAttachments[0].blendingEnabled = YES;
                descriptor.colorAttachments[0].sourceRGBBlendFactor = blendFactor(glState.blendSrcRGB);
                descriptor.colorAttachments[0].destinationRGBBlendFactor = blendFactor(glState.blendDstRGB);
                descriptor.colorAttachments[0].sourceAlphaBlendFactor = blendFactor(glState.blendSrcAlpha);
                descriptor.colorAttachments[0].destinationAlphaBlendFactor = blendFactor(glState.blendDstAlpha);
                auto blendOperation = [](uint32_t operation) {
                    switch (operation) {
                    case 0x800A: return MTLBlendOperationSubtract;
                    case 0x800B: return MTLBlendOperationReverseSubtract;
                    case 0x8007: return MTLBlendOperationMin;
                    case 0x8008: return MTLBlendOperationMax;
                    default: return MTLBlendOperationAdd;
                    }
                };
                descriptor.colorAttachments[0].rgbBlendOperation = blendOperation(glState.blendEquationRGB);
                descriptor.colorAttachments[0].alphaBlendOperation = blendOperation(glState.blendEquationAlpha);
            }
            descriptor.colorAttachments[0].writeMask = (glState.colorMask[0] ? MTLColorWriteMaskRed : 0) |
                                                         (glState.colorMask[1] ? MTLColorWriteMaskGreen : 0) |
                                                         (glState.colorMask[2] ? MTLColorWriteMaskBlue : 0) |
                                                         (glState.colorMask[3] ? MTLColorWriteMaskAlpha : 0);
            id<MTLRenderPipelineState> created = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (textureHandle) m_impl->fixedTexturedPipeline = created;
            else m_impl->fixedPipeline = created;
            if (!created && error) NSLog(@"Fixed pipeline create: %@", error);
        }
    }
    id<MTLRenderPipelineState> pipeline = textureHandle ? m_impl->fixedTexturedPipeline : m_impl->fixedPipeline;
    if (!pipeline) return;
    beginRenderPassToTexture(0, width ? width : 64, height ? height : 64, true);
    if (glState.scissorEnabled) setScissor(glState.scissorX, glState.scissorY, static_cast<uint32_t>(glState.scissorWidth), static_cast<uint32_t>(glState.scissorHeight));
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentEncoder) return;
    std::vector<float> expandedLineVertices;const float* drawVertices=vertices;size_t drawVertexCount=vertexCount;uint32_t drawPrimitive=primitiveType;
    if(primitiveType==0x0001&&glState.lineWidth>1.0f&&vertexCount>=2){float half=glState.lineWidth*0.5f;for(size_t i=0;i+1<vertexCount;i+=2){const float* a=vertices+i*11;const float* b=vertices+(i+1)*11;float dx=b[0]-a[0],dy=b[1]-a[1],length=std::sqrt(dx*dx+dy*dy);if(length<1e-6f)continue;float nx=-dy/length*(half/(width?width:64))*2.0f,ny=dx/length*(half/(height?height:64))*2.0f;float quad[4][11];std::memcpy(quad[0],a,sizeof(quad[0]));std::memcpy(quad[1],a,sizeof(quad[1]));std::memcpy(quad[2],b,sizeof(quad[2]));std::memcpy(quad[3],b,sizeof(quad[3]));quad[0][0]+=nx;quad[0][1]+=ny;quad[1][0]-=nx;quad[1][1]-=ny;quad[2][0]+=nx;quad[2][1]+=ny;quad[3][0]-=nx;quad[3][1]-=ny;for(int index:{0,1,2,2,1,3})expandedLineVertices.insert(expandedLineVertices.end(),quad[index],quad[index]+11);}if(!expandedLineVertices.empty()){drawVertices=expandedLineVertices.data();drawVertexCount=expandedLineVertices.size()/11;drawPrimitive=0x0004;}}
    id<MTLBuffer> buffer = [m_device newBufferWithBytes:drawVertices length:drawVertexCount * 11 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    [m_impl->currentEncoder setRenderPipelineState:pipeline];
    [m_impl->currentEncoder setCullMode:glState.cullEnabled ? (glState.cullFace == 0x0404 ? MTLCullModeFront : MTLCullModeBack) : MTLCullModeNone];
    [m_impl->currentEncoder setFrontFacingWinding:glState.frontFace == 0x0900 ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [m_impl->currentEncoder setDepthClipMode:glState.depthClampEnabled ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
    if (glState.polygonOffsetFill || glState.polygonOffsetLine || glState.polygonOffsetPoint) [m_impl->currentEncoder setDepthBias:glState.polygonOffsetUnits slopeScale:glState.polygonOffsetFactor clamp:0.0f];
    else [m_impl->currentEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    [m_impl->currentEncoder setBlendColorRed:glState.blendColor[0] green:glState.blendColor[1] blue:glState.blendColor[2] alpha:glState.blendColor[3]];
    struct AlphaState { float ref; uint32_t func; } alpha = { alphaRef, alphaTest ? alphaFunc : 0x0207 };
    id<MTLBuffer> alphaBuffer = [m_device newBufferWithBytes:&alpha length:sizeof(alpha) options:MTLResourceStorageModeShared];
    if (alphaBuffer) [m_impl->currentEncoder setFragmentBuffer:alphaBuffer offset:0 atIndex:1];
    float pointSize=std::max(1.0f,glState.pointSize);id<MTLBuffer> pointSizeBuffer=[m_device newBufferWithBytes:&pointSize length:sizeof(pointSize) options:MTLResourceStorageModeShared];if(pointSizeBuffer)[m_impl->currentEncoder setVertexBuffer:pointSizeBuffer offset:0 atIndex:3];
    if (textureHandle) {
        id<MTLBuffer> textureEnvBuffer = [m_device newBufferWithBytes:&textureEnv length:sizeof(textureEnv) options:MTLResourceStorageModeShared];
        if (textureEnvBuffer) [m_impl->currentEncoder setFragmentBuffer:textureEnvBuffer offset:0 atIndex:2];
        auto texture = m_impl->textures.find(textureHandle);
        if (texture != m_impl->textures.end()) [m_impl->currentEncoder setFragmentTexture:texture->second atIndex:0];
        auto texture1 = m_impl->textures.find(textureHandle1 ? textureHandle1 : textureHandle);
        if (texture1 != m_impl->textures.end()) [m_impl->currentEncoder setFragmentTexture:texture1->second atIndex:1];
        uint32_t textureCount=textureHandle1?2:1;id<MTLBuffer> textureCountBuffer=[m_device newBufferWithBytes:&textureCount length:sizeof(textureCount) options:MTLResourceStorageModeShared];if(textureCountBuffer)[m_impl->currentEncoder setFragmentBuffer:textureCountBuffer offset:0 atIndex:3];
        MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
        samplerDescriptor.minFilter = minFilter == 0x2600 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        samplerDescriptor.magFilter = magFilter == 0x2600 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        samplerDescriptor.sAddressMode = wrapS == 0x812F ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        samplerDescriptor.tAddressMode = wrapT == 0x812F ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        id<MTLSamplerState> sampler = [m_device newSamplerStateWithDescriptor:samplerDescriptor];
        if (sampler) {[m_impl->currentEncoder setFragmentSamplerState:sampler atIndex:0];[m_impl->currentEncoder setFragmentSamplerState:sampler atIndex:1];}
    }
    [m_impl->currentEncoder setVertexBuffer:buffer offset:0 atIndex:0];
    [m_impl->currentEncoder setViewport:(MTLViewport){0, 0, (double)(width ? width : 64), (double)(height ? height : 64), 0, 1}];
    [m_impl->currentEncoder drawPrimitives:metalPrimitiveType(drawPrimitive) vertexStart:0 vertexCount:drawVertexCount];
    [m_impl->currentEncoder endEncoding];
    if (m_impl->currentCommandBuffer && m_impl->currentDrawable) { if(m_impl->swapInterval>0)[m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable afterMinimumDuration:static_cast<CFTimeInterval>(m_impl->swapInterval)/60.0];else [m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable]; }
    m_impl->currentEncoder = nil;
    [m_impl->currentCommandBuffer commit];
    [m_impl->currentCommandBuffer waitUntilCompleted];
    m_impl->currentCommandBuffer = nil;
    m_impl->currentDrawable = nil;
}

static uint16_t halfFromFloat(float value) {
    union { float f; uint32_t u; } bits = {value};
    uint32_t sign = (bits.u >> 16) & 0x8000u, exponent = (bits.u >> 23) & 0xffu, mantissa = bits.u & 0x7fffffu;
    if (exponent == 0) return static_cast<uint16_t>(sign);
    int32_t e = static_cast<int32_t>(exponent) - 127 + 15;
    if (e <= 0) return static_cast<uint16_t>(sign);
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mantissa >> 13));
}

void GLMetalRenderer::drawPatches(uint32_t patchControlPoints, uint32_t patchCount, float tessellationFactor, bool quad, const float* outerFactors, const float* innerFactors) {
    if (!m_impl->currentEncoder || !m_impl->tessellationPipeline || !patchCount) return;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    bindVertexAttributes();
    [m_impl->currentEncoder setRenderPipelineState:m_impl->tessellationPipeline];
    auto factorAt=[&](const float* values,size_t index){return halfFromFloat(std::max(1.0f,std::min(64.0f,values?values[index]:tessellationFactor)));};
    uint16_t outer0=factorAt(outerFactors,0),outer1=factorAt(outerFactors,1),outer2=factorAt(outerFactors,2),outer3=factorAt(outerFactors,3),inner0=factorAt(innerFactors,0),inner1=factorAt(innerFactors,1);
    MTLTriangleTessellationFactorsHalf triangleFactors = {{outer0,outer1,outer2},inner0};
    MTLQuadTessellationFactorsHalf quadFactors = {{outer0,outer1,outer2,outer3},{inner0,inner1}};
    const void* factorData = quad ? static_cast<const void*>(&quadFactors) : static_cast<const void*>(&triangleFactors);
    size_t factorSize = quad ? sizeof(quadFactors) : sizeof(triangleFactors);
    m_impl->tessellationFactors = [m_device newBufferWithBytes:factorData length:factorSize options:MTLResourceStorageModeShared];
    [m_impl->currentEncoder setTessellationFactorBuffer:m_impl->tessellationFactors offset:0 instanceStride:factorSize];
    [m_impl->currentEncoder drawPatches:patchControlPoints patchStart:0 patchCount:patchCount patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
}

void GLMetalRenderer::drawArraysInstanced(uint32_t primitiveType, uint32_t first, uint32_t count, uint32_t instances, uint32_t baseInstance)
{
    if (!m_impl->currentEncoder || !count || !instances) return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        bindVertexAttributes();
    }
    [m_impl->currentEncoder drawPrimitives:metalPrimitiveType(primitiveType)
                               vertexStart:first vertexCount:count instanceCount:instances baseInstance:baseInstance];
}

void GLMetalRenderer::drawElements(uint32_t primitiveType, uint32_t count, uint32_t indexType, size_t offset, int32_t baseVertex, uint32_t baseInstance)
{
    if (!m_impl->currentEncoder || !m_impl->currentIndexBuffer || !count)
        return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        bindVertexAttributes();
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
                                indexBufferOffset:m_impl->currentIndexOffset + offset
                                  instanceCount:1 baseVertex:baseVertex baseInstance:baseInstance];
}

void GLMetalRenderer::drawElementsInstanced(uint32_t primitiveType, uint32_t count, uint32_t indexType,
                                             size_t offset, uint32_t instances, int32_t baseVertex, uint32_t baseInstance)
{
    if (!m_impl->currentEncoder || !m_impl->currentIndexBuffer || !count || !instances) return;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        bindVertexAttributes();
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
                                    instanceCount:instances baseVertex:baseVertex baseInstance:baseInstance];
}

void GLMetalRenderer::beginRenderPass(uint32_t width, uint32_t height) {
    beginRenderPassToTexture(0, width, height, true);
}

void GLMetalRenderer::beginRenderPassToTexture(uint64_t textureHandle, uint32_t width, uint32_t height, bool clear, uint64_t depthTextureHandle, uint32_t colorSlice, uint64_t stencilTextureHandle) {
    if (!m_device)
        return;

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    id<MTLTexture> texture = nil;
    id<MTLTexture> resolveTexture = nil;
    uint32_t colorSampleCount = 1;
    id<CAMetalDrawable> drawable = nil;

    if (textureHandle) {
        auto it = m_impl->textures.find(textureHandle);
        if (it != m_impl->textures.end()) texture = it->second;
        auto samples = m_impl->textureSampleCounts.find(textureHandle);if(samples!=m_impl->textureSampleCounts.end())colorSampleCount=samples->second;
        auto resolve = m_impl->resolveTextures.find(textureHandle);
        if (resolve != m_impl->resolveTextures.end()) { auto resolved = m_impl->textures.find(resolve->second); if(resolved!=m_impl->textures.end())resolveTexture=resolved->second; }
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
    passDesc.colorAttachments[0].slice = colorSlice;
    passDesc.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
    passDesc.colorAttachments[0].clearColor = m_impl->clearColor;
    if (resolveTexture) { passDesc.colorAttachments[0].resolveTexture=resolveTexture; passDesc.colorAttachments[0].resolveSlice=colorSlice; passDesc.colorAttachments[0].storeAction=MTLStoreActionStoreAndMultisampleResolve; }
    else passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    m_impl->depthTarget = nil;
    id<MTLTexture> stencilTarget = nil;
    if (depthTextureHandle) {
        auto depth = m_impl->textures.find(depthTextureHandle); if (depth != m_impl->textures.end()) m_impl->depthTarget = depth->second;
    }
    if (stencilTextureHandle) { auto stencil = m_impl->textures.find(stencilTextureHandle); if (stencil != m_impl->textures.end()) stencilTarget = stencil->second; }
    if (!m_impl->depthTarget) {
        MTLTextureDescriptor* depthDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:width height:height mipmapped:NO];
        if(colorSampleCount>1){depthDesc.textureType=MTLTextureType2DMultisample;depthDesc.sampleCount=colorSampleCount;}
        depthDesc.usage = MTLTextureUsageRenderTarget;
        m_impl->depthTarget = [m_device newTextureWithDescriptor:depthDesc];
    }
    if (m_impl->depthTarget) {
        MTLPixelFormat depthFormat = m_impl->depthTarget.pixelFormat;
        if (depthFormat == MTLPixelFormatDepth32Float || depthFormat == MTLPixelFormatDepth32Float_Stencil8) {
            passDesc.depthAttachment.texture = m_impl->depthTarget;
            passDesc.depthAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
            passDesc.depthAttachment.clearDepth = m_impl->clearDepth;
            passDesc.depthAttachment.storeAction = depthTextureHandle ? MTLStoreActionStore : MTLStoreActionDontCare;
        }
        if (!stencilTarget) stencilTarget = m_impl->depthTarget;
        MTLPixelFormat stencilFormat = stencilTarget.pixelFormat;
        if (stencilFormat == MTLPixelFormatStencil8 || stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
            passDesc.stencilAttachment.texture = stencilTarget;
            passDesc.stencilAttachment.slice = colorSlice;
            passDesc.stencilAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
            passDesc.stencilAttachment.clearStencil = m_impl->clearStencil;
            passDesc.stencilAttachment.storeAction = (depthTextureHandle || stencilTextureHandle) ? MTLStoreActionStore : MTLStoreActionDontCare;
        }
    }

    id<MTLCommandBuffer> cmdBuf = [m_commandQueue commandBuffer];
    m_impl->colorTarget = resolveTexture ? resolveTexture : texture;
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
        if(m_impl->swapInterval>0)[m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable afterMinimumDuration:static_cast<CFTimeInterval>(m_impl->swapInterval)/60.0];else [m_impl->currentCommandBuffer presentDrawable:m_impl->currentDrawable];
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

void GLMetalRenderer::bindComputeBuffer(uint64_t bufferHandle, uint32_t index, size_t offset) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->currentComputeEncoder) return;
    auto it = m_impl->buffers.find(bufferHandle);
    if (it != m_impl->buffers.end()) [m_impl->currentComputeEncoder setBuffer:it->second offset:offset atIndex:index];
}

void GLMetalRenderer::bindUniformBuffer(uint64_t bufferHandle, uint32_t index, size_t offset) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->buffers.find(bufferHandle);
    if (it == m_impl->buffers.end() || !m_impl->currentEncoder) return;
    [m_impl->currentEncoder setVertexBuffer:it->second offset:offset atIndex:index];
    [m_impl->currentEncoder setFragmentBuffer:it->second offset:offset atIndex:index];
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

bool GLMetalRenderer::readStencil8(uint64_t stencilTextureHandle, uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* data) {
    if(!data||!width||!height)return false; std::lock_guard<std::mutex> lock(m_impl->mutex); auto it=m_impl->textures.find(stencilTextureHandle); if(it==m_impl->textures.end()||it->second.pixelFormat!=MTLPixelFormatDepth32Float_Stencil8)return false; if(x+width>it->second.width||y+height>it->second.height)return false; const size_t bytesPerPixel=8; std::vector<uint8_t> raw(static_cast<size_t>(width)*height*bytesPerPixel); [it->second getBytes:raw.data() bytesPerRow:width*bytesPerPixel fromRegion:MTLRegionMake2D(x,y,width,height) mipmapLevel:0]; for(size_t i=0;i<static_cast<size_t>(width)*height;++i)static_cast<uint8_t*>(data)[i]=raw[i*bytesPerPixel+4]; return true;
}

bool GLMetalRenderer::readDepth32(uint64_t depthTextureHandle, uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* data) {
    if(!data||!width||!height)return false; std::lock_guard<std::mutex> lock(m_impl->mutex); auto it=m_impl->textures.find(depthTextureHandle); if(it==m_impl->textures.end()||it->second.pixelFormat!=MTLPixelFormatDepth32Float)return false; if(x+width>it->second.width||y+height>it->second.height)return false; [it->second getBytes:data bytesPerRow:width*sizeof(float) fromRegion:MTLRegionMake2D(x,y,width,height) mipmapLevel:0]; return true;
}

static float metalHalfToFloat(uint16_t value) { uint32_t sign=(value>>15)&1,e=(value>>10)&0x1f,m=value&0x3ff,bits;if(!e)bits=sign<<31;else if(e==0x1f)bits=(sign<<31)|0x7f800000|(m<<13);else bits=(sign<<31)|((static_cast<int32_t>(e)-15+127)<<23)|(m<<13);float result;std::memcpy(&result,&bits,4);return result; }
bool GLMetalRenderer::readTextureRGBA8(uint64_t textureHandle, uint32_t x, uint32_t y,
                                       uint32_t width, uint32_t height, void* data, uint32_t slice) {
    if (!data || !width || !height) return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex); auto resolvedHandle=m_impl->resolveTextures.find(textureHandle);if(resolvedHandle!=m_impl->resolveTextures.end())textureHandle=resolvedHandle->second;auto it=m_impl->textures.find(textureHandle); if(it==m_impl->textures.end())return false; id<MTLTexture> texture=it->second; if(x+width>texture.width||y+height>texture.height||slice>=texture.arrayLength)return false;
    const bool r8=texture.pixelFormat==MTLPixelFormatR8Unorm,rg8=texture.pixelFormat==MTLPixelFormatRG8Unorm,rgba16=texture.pixelFormat==MTLPixelFormatRGBA16Float,rgba32=texture.pixelFormat==MTLPixelFormatRGBA32Float; const size_t bpp=r8?1:rg8?2:rgba16?8:rgba32?16:4; const NSUInteger bytesPerRow=(static_cast<NSUInteger>(width)*bpp+255)&~static_cast<NSUInteger>(255),bufferSize=bytesPerRow*static_cast<NSUInteger>(height); id<MTLBuffer> staging=[m_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];id<MTLCommandBuffer> commandBuffer=[m_commandQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[commandBuffer blitCommandEncoder];if(!staging||!commandBuffer||!blit)return false;[blit copyFromTexture:texture sourceSlice:slice sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0) sourceSize:MTLSizeMake(width,height,1) toBuffer:staging destinationOffset:0 destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bufferSize];[blit endEncoding];[commandBuffer commit];[commandBuffer waitUntilCompleted];if(commandBuffer.status!=MTLCommandBufferStatusCompleted)return false;
    uint8_t* rgbaOut=static_cast<uint8_t*>(data);const uint8_t* bytes=static_cast<const uint8_t*>(staging.contents);const bool bgra=texture.pixelFormat==MTLPixelFormatBGRA8Unorm||texture.pixelFormat==MTLPixelFormatBGRA8Unorm_sRGB;for(uint32_t row=0;row<height;++row)for(uint32_t column=0;column<width;++column){size_t src=static_cast<size_t>(height-1-row)*bytesPerRow+static_cast<size_t>(column)*bpp,dst=(static_cast<size_t>(row)*width+column)*4;float value[4]={0,0,0,1};if(r8)value[0]=bytes[src]/255.0f;else if(rg8){value[0]=bytes[src]/255.0f;value[1]=bytes[src+1]/255.0f;}else if(rgba16){const uint16_t* v=reinterpret_cast<const uint16_t*>(bytes+src);value[0]=metalHalfToFloat(v[0]);value[1]=metalHalfToFloat(v[1]);value[2]=metalHalfToFloat(v[2]);value[3]=metalHalfToFloat(v[3]);}else if(rgba32)std::memcpy(value,bytes+src,sizeof(value));else if(bgra){value[0]=bytes[src+2]/255.0f;value[1]=bytes[src+1]/255.0f;value[2]=bytes[src]/255.0f;value[3]=bytes[src+3]/255.0f;}else{for(int c=0;c<4;++c)value[c]=bytes[src+c]/255.0f;}for(int c=0;c<4;++c)rgbaOut[dst+c]=static_cast<uint8_t>(std::clamp(value[c],0.0f,1.0f)*255.0f+0.5f);}
    return true;
}

bool GLMetalRenderer::readTextureScalar32(uint64_t textureHandle,uint32_t x,uint32_t y,uint32_t width,uint32_t height,void* data) { if(!data||!width||!height)return false;std::lock_guard<std::mutex> lock(m_impl->mutex);auto resolved=m_impl->resolveTextures.find(textureHandle);if(resolved!=m_impl->resolveTextures.end())textureHandle=resolved->second;auto it=m_impl->textures.find(textureHandle);if(it==m_impl->textures.end()||it->second.pixelFormat!=MTLPixelFormatR32Uint&&it->second.pixelFormat!=MTLPixelFormatR32Float)return false;id<MTLTexture> texture=it->second;if(x+width>texture.width||y+height>texture.height)return false;NSUInteger rowBytes=(static_cast<NSUInteger>(width)*4+255)&~static_cast<NSUInteger>(255);id<MTLBuffer> staging=[m_device newBufferWithLength:rowBytes*height options:MTLResourceStorageModeShared];id<MTLCommandBuffer> command=[m_commandQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];if(!staging||!command||!blit)return false;[blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0) sourceSize:MTLSizeMake(width,height,1) toBuffer:staging destinationOffset:0 destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes*height];[blit endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)return false;for(uint32_t row=0;row<height;++row)std::memcpy(static_cast<uint8_t*>(data)+static_cast<size_t>(row)*width*4,static_cast<const uint8_t*>(staging.contents)+static_cast<size_t>(row)*rowBytes,static_cast<size_t>(width)*4);return true; }

bool GLMetalRenderer::blitTexture(uint64_t sourceHandle, uint64_t destinationHandle, uint32_t width, uint32_t height, uint32_t sourceX, uint32_t sourceY, uint32_t destinationX, uint32_t destinationY) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto sourceResolve=m_impl->resolveTextures.find(sourceHandle);if(sourceResolve!=m_impl->resolveTextures.end())sourceHandle=sourceResolve->second;auto destinationResolve=m_impl->resolveTextures.find(destinationHandle);if(destinationResolve!=m_impl->resolveTextures.end())destinationHandle=destinationResolve->second;
    auto source = m_impl->textures.find(sourceHandle), destination = m_impl->textures.find(destinationHandle);
    if (source == m_impl->textures.end() || destination == m_impl->textures.end()) return false;
    if (sourceX + width > source->second.width || sourceY + height > source->second.height || destinationX + width > destination->second.width || destinationY + height > destination->second.height) return false;
    if (source->second.textureType == MTLTextureType2D && destination->second.textureType == MTLTextureType2D) {
        auto bytesPerPixel = [](MTLPixelFormat format) -> size_t {
            if (format == MTLPixelFormatR8Unorm || format == MTLPixelFormatR8Uint) return 1;
            if (format == MTLPixelFormatRG8Unorm || format == MTLPixelFormatRG8Uint) return 2;
            if (format == MTLPixelFormatR32Float) return 4;
            if (format == MTLPixelFormatRGBA16Float) return 8;
            return format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB || format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB || format == MTLPixelFormatRGBA8Uint ? 4 : 0;
        };
        const size_t sourceBpp = bytesPerPixel(source->second.pixelFormat), destinationBpp = bytesPerPixel(destination->second.pixelFormat);
        if (sourceBpp && destinationBpp) {
            std::vector<uint8_t> sourcePixels(static_cast<size_t>(width) * height * sourceBpp), destinationPixels(static_cast<size_t>(width) * height * destinationBpp);
            [source->second getBytes:sourcePixels.data() bytesPerRow:static_cast<size_t>(width) * sourceBpp fromRegion:MTLRegionMake2D(sourceX,sourceY,width,height) mipmapLevel:0];

            auto read = [&](const uint8_t* pixel, MTLPixelFormat format) { std::array<uint8_t,4> value = {0,0,0,255}; if(format == MTLPixelFormatR8Unorm || format == MTLPixelFormatR8Uint) value[0]=pixel[0]; else if(format == MTLPixelFormatRG8Unorm || format == MTLPixelFormatRG8Uint) { value[0]=pixel[0]; value[1]=pixel[1]; } else if(format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB) { value={pixel[2],pixel[1],pixel[0],pixel[3]}; } else if(format == MTLPixelFormatR32Float) { float f; std::memcpy(&f,pixel,sizeof(f)); value[0]=static_cast<uint8_t>(std::clamp(f,0.0f,1.0f)*255.0f+0.5f); } else { value={pixel[0],pixel[1],pixel[2],pixel[3]}; } return value; };
            auto write = [&](uint8_t* pixel, MTLPixelFormat format, const std::array<uint8_t,4>& value) { if(format == MTLPixelFormatR8Unorm || format == MTLPixelFormatR8Uint) pixel[0]=value[0]; else if(format == MTLPixelFormatRG8Unorm || format == MTLPixelFormatRG8Uint) { pixel[0]=value[0]; pixel[1]=value[1]; } else if(format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB) { pixel[0]=value[2]; pixel[1]=value[1]; pixel[2]=value[0]; pixel[3]=value[3]; } else { std::memcpy(pixel,value.data(),std::min<size_t>(destinationBpp,4)); } };
            for (uint32_t row = 0; row < height; ++row) for (uint32_t column = 0; column < width; ++column) write(destinationPixels.data() + (static_cast<size_t>(row) * width + column) * destinationBpp, destination->second.pixelFormat, read(sourcePixels.data() + (static_cast<size_t>(row) * width + column) * sourceBpp, source->second.pixelFormat));
            [destination->second replaceRegion:MTLRegionMake2D(destinationX,destinationY,width,height) mipmapLevel:0 withBytes:destinationPixels.data() bytesPerRow:static_cast<size_t>(width) * destinationBpp];
            return true;
        }
    }
    id<MTLCommandBuffer> commandBuffer = [m_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !blit) return false;
    [blit copyFromTexture:source->second sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(sourceX,sourceY,0)
               sourceSize:MTLSizeMake(width,height,1) toTexture:destination->second destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(destinationX,destinationY,0)];
    [blit endEncoding]; [commandBuffer commit]; [commandBuffer waitUntilCompleted];
    return commandBuffer.status == MTLCommandBufferStatusCompleted;
}

bool GLMetalRenderer::clearColorTexture(uint64_t textureHandle, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                         float red, float green, float blue, float alpha) {
    if (!width || !height) return true;
    MTLPixelFormat targetFormat = MTLPixelFormatInvalid;
    uint32_t sampleCount = 1;
    bool multisample = false;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        auto it = m_impl->textures.find(textureHandle);
        if (it == m_impl->textures.end() || x + width > it->second.width || y + height > it->second.height) return false;
        targetFormat = it->second.pixelFormat;
        multisample = it->second.textureType == MTLTextureType2DMultisample;
        auto samples = m_impl->textureSampleCounts.find(textureHandle);
        if (samples != m_impl->textureSampleCounts.end()) sampleCount = samples->second;
    }
    if (multisample) {
        static const char source[] = "#include <metal_stdlib>\nusing namespace metal;\n"
                                      "vertex float4 clear_vertex(uint id [[vertex_id]]) { float4 p[4] = {float4(-1,-1,0,1),float4(-1,1,0,1),float4(1,-1,0,1),float4(1,1,0,1)}; return p[id]; }\n"
                                      "fragment float4 clear_fragment(constant float4& color [[buffer(0)]]) { return color; }\n";
        NSError* error = nil;
        id<MTLLibrary> library = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
        id<MTLFunction> vertex = [library newFunctionWithName:@"clear_vertex"];
        id<MTLFunction> fragment = [library newFunctionWithName:@"clear_fragment"];
        if (!library || !vertex || !fragment) return false;
        MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.vertexFunction = vertex;
        descriptor.fragmentFunction = fragment;
        descriptor.rasterSampleCount = sampleCount;
        descriptor.colorAttachments[0].pixelFormat = targetFormat;
        descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        descriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        id<MTLRenderPipelineState> pipeline = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!pipeline) return false;
        beginRenderPassToTexture(textureHandle, 0, 0, false);
        setViewport(0, 0, static_cast<uint32_t>(m_impl->textures[textureHandle].width), static_cast<uint32_t>(m_impl->textures[textureHandle].height), 0.0, 1.0);
        setScissor(static_cast<int32_t>(x), static_cast<int32_t>(y), width, height);
        float color[4] = {red, green, blue, alpha};
        id<MTLBuffer> colorBuffer = [m_device newBufferWithBytes:color length:sizeof(color) options:MTLResourceStorageModeShared];
        {
            std::lock_guard<std::mutex> lock(m_impl->mutex);
            if (!m_impl->currentEncoder) return false;
            [m_impl->currentEncoder setRenderPipelineState:pipeline];
            if (colorBuffer) [m_impl->currentEncoder setFragmentBuffer:colorBuffer offset:0 atIndex:0];
            [m_impl->currentEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
        endRenderPass();
        finish();
        return true;
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->textures.find(textureHandle);
    if (it == m_impl->textures.end()) return false;
    const auto clampByte = [](float value) { return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f); };
    const uint8_t r = clampByte(red), g = clampByte(green), b = clampByte(blue), a = clampByte(alpha);
    const MTLPixelFormat format = it->second.pixelFormat;
    size_t bytesPerPixel = 0;
    enum class Layout { R8, RG8, RGBA8, BGRA8, R32F, RG16F, RGBA16F } layout;
    if (format == MTLPixelFormatR8Unorm || format == MTLPixelFormatR8Uint) { bytesPerPixel = 1; layout = Layout::R8; }
    else if (format == MTLPixelFormatRG8Unorm || format == MTLPixelFormatRG8Uint) { bytesPerPixel = 2; layout = Layout::RG8; }
    else if (format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB) { bytesPerPixel = 4; layout = Layout::BGRA8; }
    else if (format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB || format == MTLPixelFormatRGBA8Uint) { bytesPerPixel = 4; layout = Layout::RGBA8; }
    else if (format == MTLPixelFormatR32Float) { bytesPerPixel = sizeof(float); layout = Layout::R32F; }
    else if (format == MTLPixelFormatRG16Float) { bytesPerPixel = sizeof(uint16_t) * 2; layout = Layout::RG16F; }
    else if (format == MTLPixelFormatRGBA16Float) { bytesPerPixel = sizeof(uint16_t) * 4; layout = Layout::RGBA16F; }
    else return false;
    std::vector<uint8_t> row(static_cast<size_t>(width) * bytesPerPixel);
    for (uint32_t column = 0; column < width; ++column) {
        uint8_t* pixel = row.data() + static_cast<size_t>(column) * bytesPerPixel;
        if (layout == Layout::R8) pixel[0] = r;
        else if (layout == Layout::RG8) { pixel[0] = r; pixel[1] = g; }
        else if (layout == Layout::RGBA8) { pixel[0] = r; pixel[1] = g; pixel[2] = b; pixel[3] = a; }
        else if (layout == Layout::BGRA8) { pixel[0] = b; pixel[1] = g; pixel[2] = r; pixel[3] = a; }
        else if (layout == Layout::R32F) { float value = std::clamp(red, 0.0f, 1.0f); std::memcpy(pixel, &value, sizeof(value)); }
        else if (layout == Layout::RG16F) { uint16_t values[2] = {halfFromFloat(red), halfFromFloat(green)}; std::memcpy(pixel, values, sizeof(values)); }
        else { uint16_t values[4] = {halfFromFloat(red), halfFromFloat(green), halfFromFloat(blue), halfFromFloat(alpha)}; std::memcpy(pixel, values, sizeof(values)); }
    }
    for (uint32_t rowIndex = 0; rowIndex < height; ++rowIndex)
        [it->second replaceRegion:MTLRegionMake2D(x, y + rowIndex, width, 1) mipmapLevel:0 withBytes:row.data() bytesPerRow:row.size()];
    return true;
}

bool GLMetalRenderer::clearDefaultColorRegion(uint32_t width, uint32_t height, uint32_t x, uint32_t y,
                                               uint32_t regionWidth, uint32_t regionHeight, float red, float green,
                                               float blue, float alpha) {
    if (!width || !height || x + regionWidth > width || y + regionHeight > height) return false;
    uint64_t handle = 0;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (!m_impl->defaultColorTarget || m_impl->defaultColorWidth < width || m_impl->defaultColorHeight < height) {
            const uint32_t newWidth = std::max(width, m_impl->defaultColorWidth), newHeight = std::max(height, m_impl->defaultColorHeight);
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:newWidth height:newHeight mipmapped:NO];
            descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
            if (!texture) return false;
            if (m_impl->defaultColorTarget) {
                std::vector<uint8_t> oldPixels(static_cast<size_t>(m_impl->defaultColorWidth) * m_impl->defaultColorHeight * 4);
                [m_impl->defaultColorTarget getBytes:oldPixels.data() bytesPerRow:static_cast<size_t>(m_impl->defaultColorWidth) * 4 fromRegion:MTLRegionMake2D(0,0,m_impl->defaultColorWidth,m_impl->defaultColorHeight) mipmapLevel:0];
                [texture replaceRegion:MTLRegionMake2D(0,0,m_impl->defaultColorWidth,m_impl->defaultColorHeight) mipmapLevel:0 withBytes:oldPixels.data() bytesPerRow:static_cast<size_t>(m_impl->defaultColorWidth) * 4];
            }
            m_impl->defaultColorTarget = texture;
            m_impl->defaultColorWidth = newWidth;
            m_impl->defaultColorHeight = newHeight;
            m_impl->defaultColorHandle = m_impl->nextTextureHandle++;
            m_impl->textures[m_impl->defaultColorHandle] = texture;
        }
        handle = m_impl->defaultColorHandle;
    }
    const bool result = clearColorTexture(handle, x, y, regionWidth, regionHeight, red, green, blue, alpha);
    if (result) { std::lock_guard<std::mutex> lock(m_impl->mutex); m_impl->colorTarget = m_impl->defaultColorTarget; }
    return result;
}

uint64_t GLMetalRenderer::defaultColorTextureHandle() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->defaultColorHandle;
}

bool GLMetalRenderer::clearDepthTexture(uint64_t textureHandle, uint32_t x, uint32_t y, uint32_t width, uint32_t height, float depth) {
    if (!width || !height) return true;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it=m_impl->textures.find(textureHandle);
    if (it==m_impl->textures.end() || it->second.pixelFormat!=MTLPixelFormatR32Float || it->second.textureType==MTLTextureType2DMultisample || x+width>it->second.width || y+height>it->second.height) return false;
    std::vector<float> row(width,std::clamp(depth,0.0f,1.0f));
    for(uint32_t rowIndex=0;rowIndex<height;++rowIndex)[it->second replaceRegion:MTLRegionMake2D(x,y+rowIndex,width,1) mipmapLevel:0 withBytes:row.data() bytesPerRow:static_cast<size_t>(width)*sizeof(float)];
    return true;
}

bool GLMetalRenderer::readDepthTexture(uint64_t textureHandle, uint32_t x, uint32_t y, uint32_t width, uint32_t height, float* data) {
    if (!data || !width || !height) return false;
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it=m_impl->textures.find(textureHandle);
    if(it==m_impl->textures.end() || it->second.pixelFormat!=MTLPixelFormatR32Float || x+width>it->second.width || y+height>it->second.height) return false;
    const NSUInteger bytesPerRow=(static_cast<NSUInteger>(width)*sizeof(float)+255)&~static_cast<NSUInteger>(255), bufferSize=bytesPerRow*height;
    id<MTLBuffer> staging=[m_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];id<MTLCommandBuffer> command=[m_commandQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];if(!staging||!command||!blit)return false;
    [blit copyFromTexture:it->second sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0) sourceSize:MTLSizeMake(width,height,1) toBuffer:staging destinationOffset:0 destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bufferSize];[blit endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)return false;
    for(uint32_t row=0;row<height;++row)std::memcpy(data+static_cast<size_t>(row)*width,static_cast<const uint8_t*>(staging.contents)+static_cast<size_t>(row)*bytesPerRow,static_cast<size_t>(width)*sizeof(float));
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
    } else if (type == 0x1401) {
        switch (size) { case 1: format = normalized ? MTLVertexFormatUCharNormalized : MTLVertexFormatUChar; break; case 2: format = normalized ? MTLVertexFormatUChar2Normalized : MTLVertexFormatUChar2; break; case 3: format = normalized ? MTLVertexFormatUChar3Normalized : MTLVertexFormatUChar3; break; case 4: format = normalized ? MTLVertexFormatUChar4Normalized : MTLVertexFormatUChar4; break; }
    } else if (type == 0x1400) {
        switch (size) { case 1: format = normalized ? MTLVertexFormatCharNormalized : MTLVertexFormatChar; break; case 2: format = normalized ? MTLVertexFormatChar2Normalized : MTLVertexFormatChar2; break; case 3: format = normalized ? MTLVertexFormatChar3Normalized : MTLVertexFormatChar3; break; case 4: format = normalized ? MTLVertexFormatChar4Normalized : MTLVertexFormatChar4; break; }
    } else if (type == 0x1403) {
        switch (size) { case 1: format = normalized ? MTLVertexFormatUShortNormalized : MTLVertexFormatUShort; break; case 2: format = normalized ? MTLVertexFormatUShort2Normalized : MTLVertexFormatUShort2; break; case 3: format = normalized ? MTLVertexFormatUShort3Normalized : MTLVertexFormatUShort3; break; case 4: format = normalized ? MTLVertexFormatUShort4Normalized : MTLVertexFormatUShort4; break; }
    } else if (type == 0x1402) {
        switch (size) { case 1: format = normalized ? MTLVertexFormatShortNormalized : MTLVertexFormatShort; break; case 2: format = normalized ? MTLVertexFormatShort2Normalized : MTLVertexFormatShort2; break; case 3: format = normalized ? MTLVertexFormatShort3Normalized : MTLVertexFormatShort3; break; case 4: format = normalized ? MTLVertexFormatShort4Normalized : MTLVertexFormatShort4; break; }
    } else if (type == 0x1405) {
        switch (size) { case 1: format = MTLVertexFormatUInt; break; case 2: format = MTLVertexFormatUInt2; break; case 3: format = MTLVertexFormatUInt3; break; case 4: format = MTLVertexFormatUInt4; break; }
    } else if (type == 0x1404) {
        switch (size) { case 1: format = MTLVertexFormatInt; break; case 2: format = MTLVertexFormatInt2; break; case 3: format = MTLVertexFormatInt3; break; case 4: format = MTLVertexFormatInt4; break; }
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = std::find_if(m_impl->vertexAttributes.begin(), m_impl->vertexAttributes.end(),
                           [index](const auto& attribute) { return attribute.index == index; });
    Impl::VertexAttribute attribute{index, format, stride, bufferHandle, offset};
    if (it == m_impl->vertexAttributes.end()) m_impl->vertexAttributes.push_back(attribute);
    else *it = attribute;
}

void GLMetalRenderer::bindVertexAttributes()
{
    if (!m_impl->currentEncoder || m_impl->vertexAttributes.empty()) return;

    // All shader attributes use one Metal vertex-buffer slot.  A draw may
    // have reached here through the VBO interleaving path, in which case
    // every attribute points at the same staging buffer.  Binding that
    // buffer once at offset zero preserves each descriptor attribute offset;
    // repeatedly binding it with each attribute's offset would leave only
    // the last attribute visible and double-apply the descriptor offsets.
    uint64_t commonHandle = m_impl->vertexAttributes.front().bufferHandle;
    bool common = commonHandle != 0;
    for (const auto& attribute : m_impl->vertexAttributes)
        common = common && attribute.bufferHandle == commonHandle;
    if (common) {
        auto it = m_impl->buffers.find(commonHandle);
        if (it != m_impl->buffers.end())
            [m_impl->currentEncoder setVertexBuffer:it->second offset:0 atIndex:30];
        return;
    }
    for (const auto& attribute : m_impl->vertexAttributes) {
        auto it = m_impl->buffers.find(attribute.bufferHandle);
        if (it != m_impl->buffers.end())
            [m_impl->currentEncoder setVertexBuffer:it->second offset:attribute.offset atIndex:30];
    }
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

uint64_t GLMetalRenderer::createTexture1D(uint32_t width,uint32_t glInternalFormat,const void* data,bool mipmapped) { if(!m_device||!width)return 0;MTLPixelFormat format=glInternalFormat==0x8229?MTLPixelFormatR8Unorm:glInternalFormat==0x822B?MTLPixelFormatRG8Unorm:glInternalFormat==0x881A?MTLPixelFormatRGBA16Float:glInternalFormat==0x8814?MTLPixelFormatRGBA32Float:glInternalFormat==0x8C43?MTLPixelFormatBGRA8Unorm_sRGB:MTLPixelFormatBGRA8Unorm;size_t bpp=format==MTLPixelFormatR8Unorm?1:format==MTLPixelFormatRG8Unorm?2:format==MTLPixelFormatRGBA16Float?8:format==MTLPixelFormatRGBA32Float?16:4;MTLTextureDescriptor* descriptor=[[MTLTextureDescriptor alloc] init];descriptor.textureType=MTLTextureType1D;descriptor.pixelFormat=format;descriptor.width=width;descriptor.height=1;descriptor.depth=1;descriptor.mipmapLevelCount=1;descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;id<MTLTexture> texture=[m_device newTextureWithDescriptor:descriptor];if(!texture)return 0;if(data)[texture replaceRegion:MTLRegionMake1D(0,width) mipmapLevel:0 withBytes:data bytesPerRow:static_cast<size_t>(width)*bpp];std::lock_guard<std::mutex> lock(m_impl->mutex);uint64_t handle=m_impl->nextTextureHandle++;m_impl->textures[handle]=texture;return handle; }
bool GLMetalRenderer::updateTexture1DLevel(uint64_t textureHandle,uint32_t level,uint32_t width,const void* data,size_t bytesPerRow) { if(!m_device||!data||!width)return false;std::lock_guard<std::mutex> lock(m_impl->mutex);auto it=m_impl->textures.find(textureHandle);if(it==m_impl->textures.end()||level>=it->second.mipmapLevelCount||width>(it->second.width>>level))return false;[it->second replaceRegion:MTLRegionMake1D(0,width) mipmapLevel:level withBytes:data bytesPerRow:bytesPerRow];return true; }

uint64_t GLMetalRenderer::createTextureFormat(uint32_t width, uint32_t height, uint32_t glInternalFormat, const void* data, bool mipmapped) {
    if(!m_device||!width||!height)return 0; MTLPixelFormat format=glInternalFormat==0x8231?MTLPixelFormatR8Sint:glInternalFormat==0x8232?MTLPixelFormatR8Uint:glInternalFormat==0x8229?MTLPixelFormatR8Unorm:glInternalFormat==0x8237?MTLPixelFormatRG8Sint:glInternalFormat==0x8238?MTLPixelFormatRG8Uint:glInternalFormat==0x822B?MTLPixelFormatRG8Unorm:glInternalFormat==0x8234?MTLPixelFormatR16Uint:glInternalFormat==0x8233?MTLPixelFormatR16Sint:glInternalFormat==0x823A?MTLPixelFormatRG16Uint:glInternalFormat==0x8239?MTLPixelFormatRG16Sint:glInternalFormat==0x8235?MTLPixelFormatR32Sint:glInternalFormat==0x8236?MTLPixelFormatR32Uint:glInternalFormat==0x823B?MTLPixelFormatRG32Sint:glInternalFormat==0x823C?MTLPixelFormatRG32Uint:glInternalFormat==0x822E?MTLPixelFormatR32Float:glInternalFormat==0x1902||glInternalFormat==0x81A5||glInternalFormat==0x81A6||glInternalFormat==0x8CAC||glInternalFormat==0x88F0||glInternalFormat==0x8D48?MTLPixelFormatR32Float:glInternalFormat==0x8D8E?MTLPixelFormatRGBA8Sint:glInternalFormat==0x8D7C?MTLPixelFormatRGBA8Uint:glInternalFormat==0x8D88?MTLPixelFormatRGBA16Sint:glInternalFormat==0x8D76?MTLPixelFormatRGBA16Uint:glInternalFormat==0x8D82?MTLPixelFormatRGBA32Sint:glInternalFormat==0x8D70?MTLPixelFormatRGBA32Uint:glInternalFormat==0x881A?MTLPixelFormatRGBA16Float:glInternalFormat==0x8814?MTLPixelFormatRGBA32Float:glInternalFormat==0x8C43?MTLPixelFormatBGRA8Unorm_sRGB:MTLPixelFormatBGRA8Unorm; size_t bytesPerPixel=format==MTLPixelFormatR8Unorm||format==MTLPixelFormatR8Sint||format==MTLPixelFormatR8Uint?1:format==MTLPixelFormatRG8Unorm||format==MTLPixelFormatRG8Sint||format==MTLPixelFormatRG8Uint?2:format==MTLPixelFormatR16Sint||format==MTLPixelFormatR16Uint?2:format==MTLPixelFormatRG16Sint||format==MTLPixelFormatRG16Uint?4:format==MTLPixelFormatR32Float||format==MTLPixelFormatR32Sint||format==MTLPixelFormatR32Uint?4:format==MTLPixelFormatRG32Sint||format==MTLPixelFormatRG32Uint?8:format==MTLPixelFormatRGBA8Uint||format==MTLPixelFormatRGBA8Sint?4:format==MTLPixelFormatRGBA16Uint||format==MTLPixelFormatRGBA16Sint||format==MTLPixelFormatRGBA16Float?8:format==MTLPixelFormatRGBA32Uint||format==MTLPixelFormatRGBA32Sint||format==MTLPixelFormatRGBA32Float?16:4;
    MTLTextureDescriptor* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:mipmapped]; descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget; id<MTLTexture> texture=[m_device newTextureWithDescriptor:descriptor];if(!texture)return 0;if(data){std::vector<uint8_t> upload(static_cast<size_t>(width)*height*bytesPerPixel);for(uint32_t row=0;row<height;++row)std::memcpy(upload.data()+static_cast<size_t>(row)*width*bytesPerPixel,static_cast<const uint8_t*>(data)+static_cast<size_t>(height-1-row)*width*bytesPerPixel,static_cast<size_t>(width)*bytesPerPixel);[texture replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:upload.data() bytesPerRow:static_cast<size_t>(width)*bytesPerPixel];}if(mipmapped && format != MTLPixelFormatR8Sint && format != MTLPixelFormatR8Uint && format != MTLPixelFormatRG8Sint && format != MTLPixelFormatRG8Uint && format != MTLPixelFormatR16Sint && format != MTLPixelFormatR16Uint && format != MTLPixelFormatRG16Sint && format != MTLPixelFormatRG16Uint && format != MTLPixelFormatRGBA8Sint && format != MTLPixelFormatRGBA8Uint && format != MTLPixelFormatRGBA16Sint && format != MTLPixelFormatRGBA16Uint && format != MTLPixelFormatRGBA32Sint && format != MTLPixelFormatRGBA32Uint){id<MTLCommandBuffer> commandBuffer=[m_commandQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[commandBuffer blitCommandEncoder];if(blit){[blit generateMipmapsForTexture:texture];[blit endEncoding];[commandBuffer commit];[commandBuffer waitUntilCompleted];}}
    std::lock_guard<std::mutex> lock(m_impl->mutex);uint64_t handle=m_impl->nextTextureHandle++;m_impl->textures[handle]=texture;return handle;
}
bool GLMetalRenderer::updateTextureLevel(uint64_t textureHandle,uint32_t level,uint32_t width,uint32_t height,const void* data,size_t bytesPerRow) { if(!m_device||!data||!width||!height)return false;std::lock_guard<std::mutex> lock(m_impl->mutex);auto resolved=m_impl->resolveTextures.find(textureHandle);if(resolved!=m_impl->resolveTextures.end())textureHandle=resolved->second;auto it=m_impl->textures.find(textureHandle);if(it==m_impl->textures.end()||level>=it->second.mipmapLevelCount||width>(it->second.width>>level)||height>(it->second.height>>level))return false;std::vector<uint8_t> upload(bytesPerRow * static_cast<size_t>(height));for(uint32_t row=0;row<height;++row)std::memcpy(upload.data()+static_cast<size_t>(row)*bytesPerRow,static_cast<const uint8_t*>(data)+static_cast<size_t>(height-1-row)*bytesPerRow,bytesPerRow);[it->second replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:level withBytes:upload.data() bytesPerRow:bytesPerRow];return true;}
uint64_t GLMetalRenderer::createTextureCube(uint32_t width,uint32_t height,uint32_t glInternalFormat,const void* const* faces) { if(!m_device||!width||!height)return 0;MTLPixelFormat format=glInternalFormat==0x8C43?MTLPixelFormatBGRA8Unorm_sRGB:MTLPixelFormatBGRA8Unorm;MTLTextureDescriptor* descriptor=[MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:width mipmapped:YES];descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;id<MTLTexture> texture=[m_device newTextureWithDescriptor:descriptor];if(!texture)return 0;for(uint32_t face=0;face<6;++face)if(faces&&faces[face])[texture replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 slice:face withBytes:faces[face] bytesPerRow:static_cast<size_t>(width)*4 bytesPerImage:static_cast<size_t>(width)*height*4];if(faces){id<MTLCommandBuffer> command=[m_commandQueue commandBuffer];id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];if(command&&blit){[blit generateMipmapsForTexture:texture];[blit endEncoding];[command commit];[command waitUntilCompleted];}}std::lock_guard<std::mutex> lock(m_impl->mutex);uint64_t handle=m_impl->nextTextureHandle++;m_impl->textures[handle]=texture;return handle; }

uint64_t GLMetalRenderer::createTexture(uint32_t width, uint32_t height, const void* data, bool mipmapped, bool srgb) { return createTextureFormat(width,height,srgb?0x8C43:0x8058,data,mipmapped); }

uint64_t GLMetalRenderer::createTexture3D(uint32_t width, uint32_t height, uint32_t depth, const void* data) {
    if (!m_device || !width || !height || !depth) return 0;
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType3D;
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.width = width; descriptor.height = height; descriptor.depth = depth;
    descriptor.mipmapLevelCount = 1; descriptor.arrayLength = 1;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
    if (!texture) return 0;
    if (data) {
        std::vector<uint8_t> upload(static_cast<size_t>(width) * height * depth * 4);
        for (size_t i=0;i<upload.size();i+=4) { upload[i]=static_cast<const uint8_t*>(data)[i+2]; upload[i+1]=static_cast<const uint8_t*>(data)[i+1]; upload[i+2]=static_cast<const uint8_t*>(data)[i]; upload[i+3]=static_cast<const uint8_t*>(data)[i+3]; }
        MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, depth);
        [texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:upload.data() bytesPerRow:width * 4 bytesPerImage:width * height * 4];
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
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.width = width; descriptor.height = height; descriptor.depth = 1; descriptor.arrayLength = layers;
    descriptor.mipmapLevelCount = 1; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
    if (!texture) return 0;
    if (data) {
        const size_t layerBytes = static_cast<size_t>(width) * height * 4;
        std::vector<uint8_t> upload(layerBytes * layers);
        for (size_t i=0;i<upload.size();i+=4) { upload[i]=static_cast<const uint8_t*>(data)[i+2]; upload[i+1]=static_cast<const uint8_t*>(data)[i+1]; upload[i+2]=static_cast<const uint8_t*>(data)[i]; upload[i+3]=static_cast<const uint8_t*>(data)[i+3]; }
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:upload.data() bytesPerRow:width * 4 bytesPerImage:layerBytes];
        for (uint32_t layer = 1; layer < layers; ++layer)
            [texture replaceRegion:region mipmapLevel:0 slice:layer withBytes:upload.data() + static_cast<size_t>(layer) * layerBytes bytesPerRow:width * 4 bytesPerImage:layerBytes];
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    uint64_t handle = m_impl->nextTextureHandle++;
    m_impl->textures[handle] = texture;
    return handle;
}

uint64_t GLMetalRenderer::createMultisampleTexture2D(uint32_t width,uint32_t height,uint32_t glInternalFormat,uint32_t samples) {
    if(!m_device||!width||!height||samples<2)return createTextureFormat(width,height,glInternalFormat,nullptr,false);
    MTLPixelFormat format=(glInternalFormat==0x8C43)?MTLPixelFormatBGRA8Unorm_sRGB:MTLPixelFormatBGRA8Unorm;uint32_t selected=samples;if(![m_device supportsTextureSampleCount:selected]){selected=samples>=4&&[m_device supportsTextureSampleCount:4]?4:[m_device supportsTextureSampleCount:2]?2:1;}if(selected<2)return createTextureFormat(width,height,glInternalFormat,nullptr,false);
    MTLTextureDescriptor* multisample=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];multisample.textureType=MTLTextureType2DMultisample;multisample.sampleCount=selected;multisample.usage=MTLTextureUsageRenderTarget;id<MTLTexture> msaa=[m_device newTextureWithDescriptor:multisample];if(!msaa)return 0;
    MTLTextureDescriptor* resolve=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];resolve.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;id<MTLTexture> resolved=[m_device newTextureWithDescriptor:resolve];if(!resolved)return 0;
    std::lock_guard<std::mutex> lock(m_impl->mutex);uint64_t handle=m_impl->nextTextureHandle++,resolveHandle=m_impl->nextTextureHandle++;m_impl->textures[handle]=msaa;m_impl->textures[resolveHandle]=resolved;m_impl->resolveTextures[handle]=resolveHandle;m_impl->textureSampleCounts[handle]=selected;return handle;
}

uint64_t GLMetalRenderer::createMultisampleDepthStencilTarget(uint32_t width,uint32_t height,uint32_t glInternalFormat,uint32_t samples) { if(!m_device||!width||!height||samples<2)return createDepthStencilTarget(width,height,glInternalFormat);uint32_t selected=samples;if(![m_device supportsTextureSampleCount:selected])selected=samples>=4&&[m_device supportsTextureSampleCount:4]?4:[m_device supportsTextureSampleCount:2]?2:1;if(selected<2)return createDepthStencilTarget(width,height,glInternalFormat);MTLPixelFormat format=glInternalFormat==0x8D48?MTLPixelFormatStencil8:(glInternalFormat==0x1902||glInternalFormat==0x81A5||glInternalFormat==0x81A6||glInternalFormat==0x8CAC)?MTLPixelFormatDepth32Float:MTLPixelFormatDepth32Float_Stencil8;MTLTextureDescriptor* descriptor=[[MTLTextureDescriptor alloc]init];descriptor.textureType=MTLTextureType2DMultisample;descriptor.pixelFormat=format;descriptor.width=width;descriptor.height=height;descriptor.sampleCount=selected;descriptor.usage=MTLTextureUsageRenderTarget;id<MTLTexture> texture=[m_device newTextureWithDescriptor:descriptor];if(!texture)return 0;std::lock_guard<std::mutex> lock(m_impl->mutex);uint64_t handle=m_impl->nextTextureHandle++;m_impl->textures[handle]=texture;m_impl->textureSampleCounts[handle]=selected;return handle; }

void GLMetalRenderer::bindTexture(uint64_t textureHandle, uint32_t index, uint32_t baseLevel, uint32_t maxLevel) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto it = m_impl->textures.find(textureHandle);
    if (it == m_impl->textures.end() || !m_impl->currentEncoder) return;
    id<MTLTexture> texture = it->second;
    const NSUInteger levelCount = texture.mipmapLevelCount;
    if (baseLevel < levelCount && (baseLevel > 0 || maxLevel != UINT32_MAX)) {
        const NSUInteger lastLevel = maxLevel == UINT32_MAX ? levelCount - 1 : std::min<NSUInteger>(maxLevel, levelCount - 1);
        if (lastLevel >= baseLevel) {
            id<MTLTexture> view = [texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:texture.textureType levels:NSMakeRange(baseLevel, lastLevel - baseLevel + 1) slices:NSMakeRange(0, texture.arrayLength)];
            if (view) texture = view;
        }
    }
    [m_impl->currentEncoder setFragmentTexture:texture atIndex:index];
    [m_impl->currentEncoder setVertexTexture:texture atIndex:index];
}

void GLMetalRenderer::setTextureSwizzle(uint64_t textureHandle,uint32_t red,uint32_t green,uint32_t blue,uint32_t alpha) { std::lock_guard<std::mutex> lock(m_impl->mutex);auto it=m_impl->textures.find(textureHandle);if(it==m_impl->textures.end())return;auto map=[](uint32_t value){switch(value){case 0x1903:return MTLTextureSwizzleRed;case 0x1904:return MTLTextureSwizzleGreen;case 0x1905:return MTLTextureSwizzleBlue;case 0x1906:return MTLTextureSwizzleAlpha;case 1:return MTLTextureSwizzleOne;case 0:return MTLTextureSwizzleZero;default:return MTLTextureSwizzleRed;}};MTLTextureSwizzleChannels swizzle={map(red),map(green),map(blue),map(alpha)};id<MTLTexture> view=[it->second newTextureViewWithPixelFormat:it->second.pixelFormat textureType:it->second.textureType levels:NSMakeRange(0,it->second.mipmapLevelCount) slices:NSMakeRange(0,it->second.arrayLength) swizzle:swizzle];if(view)it->second=view; }

void GLMetalRenderer::bindSampler(uint32_t index, uint32_t minFilter, uint32_t magFilter,
                                   uint32_t wrapS, uint32_t wrapT, uint32_t maxAnisotropy,
                                   float minLod, float maxLod, uint32_t compareFunc,
                                   bool compare, bool normalizedCoordinates, const float* borderColor) {
    if (!m_impl->currentEncoder || !m_device) return;
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = (minFilter == 0x2600 || minFilter == 0x2700 || minFilter == 0x2702) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = (magFilter == 0x2600) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    descriptor.mipFilter = (minFilter == 0x2700 || minFilter == 0x2701) ? MTLSamplerMipFilterNearest :
                            (minFilter == 0x2702 || minFilter == 0x2703) ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNotMipmapped;
    auto wrap = [](uint32_t value) {
        switch (value) {
        case 0x812F: return MTLSamplerAddressModeClampToEdge;
        case 0x8370: return MTLSamplerAddressModeMirrorRepeat;
        default: return MTLSamplerAddressModeRepeat;
        }
    };
    descriptor.sAddressMode = wrap(wrapS);
    descriptor.tAddressMode = wrap(wrapT);
    descriptor.normalizedCoordinates = normalizedCoordinates;
    descriptor.maxAnisotropy = std::max<NSUInteger>(1, std::min<NSUInteger>(16, maxAnisotropy));
    descriptor.lodMinClamp = std::max(0.0f, minLod);
    descriptor.lodMaxClamp = std::max(descriptor.lodMinClamp, maxLod);
    if (compare) {
        switch (compareFunc) { case 0x0201: descriptor.compareFunction=MTLCompareFunctionLess; break; case 0x0202: descriptor.compareFunction=MTLCompareFunctionEqual; break; case 0x0203: descriptor.compareFunction=MTLCompareFunctionLessEqual; break; case 0x0204: descriptor.compareFunction=MTLCompareFunctionGreater; break; case 0x0205: descriptor.compareFunction=MTLCompareFunctionNotEqual; break; case 0x0206: descriptor.compareFunction=MTLCompareFunctionGreaterEqual; break; case 0x0207: descriptor.compareFunction=MTLCompareFunctionAlways; break; default: descriptor.compareFunction=MTLCompareFunctionNever; break; }
    }
    if (borderColor) { if(borderColor[3]<=0.0f) descriptor.borderColor=MTLSamplerBorderColorTransparentBlack; else if(borderColor[0]>=1.0f&&borderColor[1]>=1.0f&&borderColor[2]>=1.0f) descriptor.borderColor=MTLSamplerBorderColorOpaqueWhite; else if(borderColor[0]<=0.0f&&borderColor[1]<=0.0f&&borderColor[2]<=0.0f) descriptor.borderColor=MTLSamplerBorderColorOpaqueBlack; }
    id<MTLSamplerState> sampler = [m_device newSamplerStateWithDescriptor:descriptor];
    if (sampler) {
        [m_impl->currentEncoder setFragmentSamplerState:sampler atIndex:index];
        [m_impl->currentEncoder setVertexSamplerState:sampler atIndex:index];
    }
}

uint64_t GLMetalRenderer::createDepthStencilTarget(uint32_t width, uint32_t height, uint32_t internalFormat) {
    if (!m_device || !width || !height) return 0;
    MTLPixelFormat format = internalFormat == 0x8D48 ? MTLPixelFormatStencil8 : (internalFormat == 0x1902 || internalFormat == 0x81A5 || internalFormat == 0x81A6 || internalFormat == 0x8CAC) ? MTLPixelFormatDepth32Float : MTLPixelFormatDepth32Float_Stencil8;
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor]; if (!texture) return 0;
    std::lock_guard<std::mutex> lock(m_impl->mutex); uint64_t handle=m_impl->nextTextureHandle++; m_impl->textures[handle]=texture; return handle;
}

void GLMetalRenderer::setViewport(int32_t x, int32_t y, uint32_t width, uint32_t height, double znear, double zfar) {
    if (!m_impl->currentEncoder)
        return;
    MTLViewport vp;
    vp.originX = static_cast<double>(x);
    vp.originY = static_cast<double>(y);
    vp.width = static_cast<double>(width);
    vp.height = static_cast<double>(height);
    vp.znear = znear;
    vp.zfar = zfar;
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

void GLMetalRenderer::setClearStencil(uint32_t stencil) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->clearStencil = stencil;
}

} // namespace metalsharp
