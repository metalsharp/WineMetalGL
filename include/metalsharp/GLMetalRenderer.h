/// @file GLMetalRenderer.h
/// @brief Bridges OpenGL draw calls to Metal.
///
/// Phase 3a/3b/3c: this is the GL→Metal draw emitter. Each GL context owns
/// one @ref GLMetalRenderer which owns its own MTLDevice / MTLCommandQueue
/// and provides the minimum surface needed to translate GL stateful draw
/// calls (glBindBuffer / glDrawArrays) into stateless Metal render passes
/// (setVertexBuffer / drawPrimitives).
///
/// The shader-translation pipeline (GLSL→SPIR-V→MSL) lives upstream in
/// @ref GLShaderTracker and @ref GLSLCompiler. By the time the renderer sees
/// a shader, its MSL source is already cached in the GLShaderState's `msl`
/// field; createPipeline() compiles that MSL via the renderer's MTLDevice
/// and caches an MTLRenderPipelineState handle.
///
/// Refinement notes (Phase 3g/3j/3k):
///   * The MSL entry-point names are fixed to @c vertex_main /
///     @c fragment_main, matching GLSLCompiler's translation contract.
///   * The render pass is currently backed by an offscreen MTLTexture
///     because we don't yet integrate with the WGL swap chain — Phase 3g
///     will plug in CAMetalLayer-backed drawables.
///   * Blend / depth-stencil state translation is intentionally minimal in
///     this phase; later refinement fills in proper GL→Metal blend factor
///     mapping using glBlendFunc / glBlendEquation.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLRenderPipelineState;
@protocol MTLBuffer;
@protocol MTLTexture;
@class MTLRenderPipelineDescriptor;
#else
typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLRenderPipelineStateRef;
typedef void* MTLBufferRef;
typedef void* MTLTextureRef;
typedef void* MTLRenderPipelineDescriptorRef;
#endif

namespace metalsharp {

struct GLState;
struct GLShaderState;

/// Bridges OpenGL draw calls to Metal. Each GL context owns one renderer.
///
/// Lifetime is independent of any particular GL context — the renderer
/// itself owns an MTLDevice reference. Callers should create one renderer
/// per GL context they intend to drive and call @ref init exactly once
/// before any other method.
/// @note Performance: pipeline states are cached via GLShaderCache;
/// command encoders are reused across consecutive draws in the same
/// render pass; buffer/uniform updates are coalesced between draw calls.
class GLMetalRenderer {
  public:
    GLMetalRenderer();
    ~GLMetalRenderer();

    /// Initialize with the default Metal device. Returns true on success.
    bool init();

    /// Check if Metal rendering is available.
    bool isAvailable() const { return m_device != nullptr; }

    /// Create a render pipeline state from a compiled shader and GL state.
    /// The shader must have valid MSL in its GLShaderState::msl field.
    /// @return true on success; false if shader compilation or pipeline
    /// creation failed.
    bool createPipeline(const GLShaderState& vertexShader, const GLShaderState& fragmentShader, const GLState& glState);
    bool createComputePipeline(const GLShaderState& computeShader);

    /// Bind the current pipeline state for drawing.
    void usePipeline();

    /// Create a Metal buffer from raw vertex data.
    /// @return buffer handle (non-zero on success)
    uint64_t createBuffer(const void* data, size_t size);

    /// Bind a vertex buffer at the given index.
    void bindVertexBuffer(uint64_t bufferHandle, size_t offset, uint32_t index);

    /// Encode a draw call (glDrawArrays equivalent).
    void drawArrays(uint32_t primitiveType, uint32_t first, uint32_t count);

    /// Encode an indexed draw using an index buffer previously bound with
    /// bindIndexBuffer(). `indexType` is GL_UNSIGNED_BYTE/SHORT/INT.
    void bindIndexBuffer(uint64_t bufferHandle, size_t offset);
    void drawElements(uint32_t primitiveType, uint32_t count, uint32_t indexType, size_t offset);
    void drawArraysInstanced(uint32_t primitiveType, uint32_t first, uint32_t count, uint32_t instances);
    void drawElementsInstanced(uint32_t primitiveType, uint32_t count, uint32_t indexType,
                               size_t offset, uint32_t instances);

    /// Begin a render pass on the default framebuffer (FBO 0).
    /// If setMetalLayer() was called, the pass targets the current
    /// CAMetalDrawable. Otherwise it uses a private BGRA8 texture for
    /// off-screen/FBO operation.
    /// @param width,height fallback framebuffer dimensions
    void beginRenderPass(uint32_t width, uint32_t height);

    /// Begin a pass targeting a Metal texture owned by an OpenGL FBO.
    /// `textureHandle == 0` selects the current CAMetalDrawable/offscreen
    /// default target. The texture must have render-target usage.
    void beginRenderPassToTexture(uint64_t textureHandle, uint32_t width, uint32_t height, bool clear);

    /// End the current render pass and present.
    void endRenderPass();

    /// Flush/commit pending Metal commands.
    void flush();

    /// Commit and wait for GPU to finish (glFinish equivalent).
    void finish();

    /// Begin/dispatch a compute workload for a compute-only OpenGL program.
    void beginComputePass();
    void bindComputeBuffer(uint64_t bufferHandle, uint32_t index);
    void dispatchCompute(uint32_t x, uint32_t y, uint32_t z);
    bool readBuffer(uint64_t bufferHandle, size_t offset, size_t size, void* data);

    /// Attach the native CAMetalLayer owned by the Wine window. The layer is
    /// retained by the renderer until it is replaced or cleared. Passing
    /// nullptr returns the renderer to its deterministic off-screen target.
    void setMetalLayer(void* layer);

    /// Return whether the last render pass used a CAMetalDrawable.
    bool isDrawableBacked() const;

    /// Copy RGBA8 pixels from the most recent offscreen render target.
    /// The renderer's Metal target is BGRA8; this method performs the channel
    /// conversion required by glReadPixels(..., GL_RGBA, GL_UNSIGNED_BYTE).
    bool readPixelsRGBA8(uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* data);

    /// Set the vertex descriptor stride and per-attribute layout.
    /// @param stride   byte stride of a single vertex
    /// @param offsets  per-attribute byte offsets (length = count)
    /// @param formats  per-attribute Metal vertex format enum (length = count)
    /// @param count    number of vertex attributes
    void setVertexLayout(uint32_t stride, const uint32_t* offsets, const uint32_t* formats, uint32_t count);

    /// Configure one interleaved vertex attribute. This is the GL-facing
    /// form used by the state tracker; the renderer maps the GL scalar type
    /// and component count to an MTLVertexFormat.
    void setVertexAttribute(uint32_t index, int32_t size, uint32_t type, bool normalized,
                            uint32_t stride, uint64_t bufferHandle, size_t offset);

    /// Allocate/update a uniform buffer at the given binding index.
    /// @param binding  fragment-shader uniform buffer binding slot
    /// @param data     pointer to uniform data (nullptr and size==0 deletes)
    /// @param size     size of uniform data in bytes
    void updateUniformBuffer(uint32_t binding, const void* data, size_t size);

    /// Create a Metal texture from raw pixel data.
    /// @param width,height  texture dimensions in pixels
    /// @param data         BGRA8 pixel data, tightly packed
    /// @return non-zero texture handle on success
    uint64_t createTexture(uint32_t width, uint32_t height, const void* data);

    /// Bind a texture at the given fragment shader index.
    /// @param textureHandle  handle returned by createTexture
    /// @param index          fragment texture slot index
    void bindTexture(uint64_t textureHandle, uint32_t index);

    /// Bind a sampler state using OpenGL enum values for min/mag filters and
    /// S/T wrap modes.
    void bindSampler(uint32_t index, uint32_t minFilter, uint32_t magFilter,
                     uint32_t wrapS, uint32_t wrapT);

    /// Set the encoder viewport (glViewport equivalent).
    void setViewport(int32_t x, int32_t y, uint32_t width, uint32_t height);

    /// Set the encoder scissor rectangle (glScissor equivalent).
    void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);

    /// Set the clear color/depth used by the next render pass.
    void setClearColor(float r, float g, float b, float a);
    void setClearDepth(float depth);

  private:
    struct Impl;
    Impl* m_impl;

#ifdef __OBJC__
    id<MTLDevice> m_device;
    id<MTLCommandQueue> m_commandQueue;
#else
    void* m_device = nullptr;
    void* m_commandQueue = nullptr;
#endif
};

} // namespace metalsharp
