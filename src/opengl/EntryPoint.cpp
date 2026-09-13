/// @file EntryPoint.cpp
/// @brief opengl32.dll shim exports for MetalSharp GL→Metal bridge.
///
/// Exports all commonly-used OpenGL 1.0-2.1 entry points. The shim is thin:
/// each function delegates through OpenGLBridge::getGLProcAddress() to the
/// macOS native OpenGL framework. Phase 4b focuses on correctness over
/// completeness — games that stick to GL 1.x/2.x immediate mode and fixed
/// pipeline will run unmodified.
///
/// Extension functions (GL 3.x/4.x, e.g. glCreateShader, glGenBuffers,
/// glBindBuffer) are resolved at runtime by Wine/wglGetProcAddress from
/// the framework. Full GL 3.x/4.x shader translation via SPIRV-Cross →
/// MSL is scaffolded for Phase 4c; the OpenGLBridge::GLState tracker
/// already captures program/blend/depth/attrib/viewport so future
/// instrumentation can be added without breaking this shim.

#include <algorithm>
#include <array>
#include <deque>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <chrono>
#include <metalsharp/GLErrorTracker.h>
#include <metalsharp/GLMetalRenderer.h>
#include <metalsharp/GLShaderCache.h>
#include <metalsharp/GLShaderTracker.h>
#include <metalsharp/GLSLCompiler.h>
#include <metalsharp/GLSLVersion.h>
#include <metalsharp/OpenGLBridge.h>
#include <metalsharp/ShaderStage.h>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __has_include(<spirv_cross_c.h>)
#define METALSHARP_HAS_SPIRV_CROSS 1
#else
#define METALSHARP_HAS_SPIRV_CROSS 0
#endif

namespace {

thread_local metalsharp::OpenGLBridge g_glBridge;
metalsharp::GLMetalRenderer g_metalRenderer;
thread_local bool g_glInitialized = false;
std::once_flag g_metalInitFlag;
bool g_metalAvailable = false;
bool g_modernContextReady = false;
thread_local void* g_currentContextKey = nullptr;
std::mutex g_contextStateMapMutex;
std::unordered_map<void*, metalsharp::GLState> g_contextStates;

struct ExperimentalProgram {
    bool linked = false;
    bool separable = false;
    uint32_t vertexShader = 0;
    uint32_t fragmentShader = 0;
    bool linkSuccess = false;
    std::string infoLog;
    std::unordered_map<std::string, int32_t> uniformLocations;
    std::unordered_map<std::string, uint32_t> uniformTypes;
    std::unordered_map<std::string, int32_t> uniformSizes;
    std::vector<std::string> uniformOrder;
    std::unordered_map<std::string, int32_t> attributeLocations;
    std::unordered_map<std::string, uint32_t> attributeTypes;
    std::vector<std::string> attributeOrder;
    std::unordered_map<std::string, std::string> vertexOutputs;
    std::unordered_map<std::string, std::string> geometryOutputs;
    std::unordered_map<std::string, std::string> fragmentInputs;
    std::unordered_map<int32_t, std::vector<uint8_t>> uniformValues;
    std::unordered_map<std::string, uint32_t> uniformBlockIndices;
    std::unordered_map<uint32_t, uint32_t> uniformBlockBindings;
    std::unordered_map<std::string, uint32_t> storageBlockIndices;
    std::unordered_map<uint32_t, uint32_t> storageBlockBindings;
    std::unordered_map<std::string, uint32_t> fragDataLocations;
    bool binaryRetrievable = false;
    std::vector<uint8_t> binary;
};

std::mutex g_programMutex;
std::unordered_map<uint32_t, ExperimentalProgram> g_programs;
struct ExperimentalPipeline { uint32_t vertexProgram=0; uint32_t fragmentProgram=0; };
std::unordered_map<uint32_t, ExperimentalPipeline> g_pipelines;
std::unordered_map<uint32_t, uint32_t> g_pipelineProxies;
std::mutex g_pipelineMutex;
std::mutex g_syncMutex;
std::unordered_set<void*> g_syncs;
struct ExperimentalQuery { uint32_t target = 0; bool active = false; uint64_t value = 0; };
std::mutex g_queryMutex;
std::unordered_map<uint32_t, ExperimentalQuery> g_queries;
std::unordered_map<uint64_t, uint8_t> g_stencilClearShadow;
struct ExperimentalDebugMessage { uint32_t source=0,type=0,id=0,severity=0; std::string text; };
using DebugCallback = void (*)(uint32_t,uint32_t,uint32_t,uint32_t,int32_t,const char*,const void*);
std::mutex g_debugMutex;
std::deque<ExperimentalDebugMessage> g_debugMessages;
DebugCallback g_debugCallback = nullptr;
const void* g_debugUserParam = nullptr;
bool g_debugOutputEnabled = true;
std::unordered_map<uint64_t,std::string> g_objectLabels;
uint32_t g_nextQuery = 1;
uint32_t g_activeQuery = 0;

struct ExperimentalBuffer {
    uint64_t metalHandle = 0;
    size_t size = 0;
};
std::mutex g_bufferMutex;
std::unordered_map<uint32_t, ExperimentalBuffer> g_buffers;

struct ExperimentalTexture {
    uint64_t metalHandle = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t target = 0x0DE1;
    uint32_t sampleCount = 1;
    uint8_t cubeFaceMask = 0;
    std::array<std::vector<uint8_t>,6> cubePixels;
    uint32_t swizzle[4] = {0x1903,0x1904,0x1905,0x1906};
    int32_t internalFormat = 0x8058;
    uint32_t minFilter = 0x2601; /* GL_LINEAR */
    uint32_t magFilter = 0x2601;
    uint32_t wrapS = 0x2901; /* GL_REPEAT */
    uint32_t wrapT = 0x2901;
    float minLod=0.0f,maxLod=1000.0f,lodBias=0.0f; uint32_t maxAnisotropy=1,compareFunc=0x0207; bool compare=false; float borderColor[4]={0,0,0,0};
    std::vector<uint8_t> pixels;
};
struct ExperimentalFramebuffer {
    uint32_t colorTexture = 0;
    uint32_t renderbuffer = 0;
    uint32_t depthRenderbuffer = 0;
    uint32_t stencilRenderbuffer = 0;
    uint32_t depthTexture = 0;
    uint32_t stencilTexture = 0;
    uint64_t colorHandle = 0;
    uint64_t depthHandle = 0;
    uint64_t stencilHandle = 0;
    uint32_t colorLayer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sampleCount = 1;
    uint32_t depthSampleCount = 1, stencilSampleCount = 1;
};
struct ExperimentalRenderbuffer { uint64_t metalHandle = 0; uint32_t width = 0, height = 0; uint32_t internalFormat = 0, sampleCount = 1; };
std::unordered_map<uint32_t, ExperimentalRenderbuffer> g_renderbuffers;
uint32_t g_boundRenderbuffer = 0;
std::mutex g_resourceMutex;
std::unordered_map<uint32_t, ExperimentalTexture> g_textures;
std::unordered_map<uint32_t, ExperimentalFramebuffer> g_framebuffers;
std::array<uint32_t, metalsharp::kMaxTextureUnits> g_textureUnits{};
std::array<uint32_t, 16> g_imageUnits{};
std::array<uint32_t, metalsharp::kMaxVertexAttribs> g_attributeDivisors{};
std::unordered_map<uint32_t, bool> g_vertexArrays;
struct ExperimentalSampler { uint32_t minFilter = 0x2601, magFilter = 0x2601, wrapS = 0x2901, wrapT = 0x2901; float minLod=0.0f,maxLod=1000.0f,lodBias=0.0f; uint32_t maxAnisotropy=1,compareFunc=0x0207; bool compare=false; float borderColor[4]={0,0,0,0}; };
std::unordered_map<uint32_t, ExperimentalSampler> g_samplers;
std::array<uint32_t, metalsharp::kMaxTextureUnits> g_samplerUnits{};
uint32_t g_boundStorageBuffer = 0;
uint32_t g_boundTransformFeedbackBuffer = 0;
size_t g_transformFeedbackBufferOffset = 0, g_transformFeedbackBufferSize = 0;
uint32_t g_boundUniformBuffer = 0;
bool g_transformFeedbackActive = false;
uint32_t g_transformFeedbackProgram = 0;
bool g_transformFeedbackPositionVarying = false;
std::vector<std::string> g_transformFeedbackVaryings;
struct ExperimentalVertexAttribute { bool set=false; int32_t size=0; uint32_t type=0; uint32_t stride=0; uint32_t buffer=0; size_t offset=0; bool normalized=false; };
std::array<ExperimentalVertexAttribute, metalsharp::kMaxVertexAttribs> g_experimentalVertexAttributes{};
struct ExperimentalVertexBinding { uint32_t buffer=0; size_t offset=0; uint32_t stride=0; };
std::array<ExperimentalVertexBinding, metalsharp::kMaxVertexAttribs> g_vertexBindings{};
std::array<uint32_t, metalsharp::kMaxVertexAttribs> g_attribBindings{};
std::array<uint32_t, 16> g_uniformBufferUnits{};
std::array<size_t, 16> g_uniformBufferOffsets{};
std::array<size_t, 16> g_uniformBufferSizes{};
size_t g_storageBufferOffset = 0;
size_t g_storageBufferSize = 0;
uint32_t g_boundQueryBuffer = 0;
uint32_t g_boundIndirectBuffer = 0;
uint32_t g_boundDispatchIndirectBuffer = 0;
uint32_t g_boundPixelPackBuffer = 0;
uint32_t g_boundPixelUnpackBuffer = 0;
uint32_t g_currentVertexArray = 0;
uint32_t g_activeTextureUnit = 0;
bool g_fixedRecording = false;
uint32_t g_fixedPrimitive = 0;
std::vector<float> g_fixedVertices;
float g_fixedColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
bool g_fixedLighting = false;
bool g_fixedLight0 = false;
bool g_fixedLights[8] = {};
float g_fixedLightPositions[8][4] = {{0,0,1,0}};
float g_fixedLightAmbients[8][4] = {{0.2f,0.2f,0.2f,1.0f}};
float g_fixedLightDiffuses[8][4] = {{1.0f,1.0f,1.0f,1.0f}};
float g_fixedLightSpeculars[8][4] = {{1.0f,1.0f,1.0f,1.0f}};
bool g_fixedTextureEnabled = false;
metalsharp::FixedTextureEnvironment g_fixedTextureEnv;
bool g_fixedFogEnabled = false;
uint32_t g_fixedFogMode = 0x2601; /* GL_LINEAR */
float g_fixedFogColor[4] = {0,0,0,1};
float g_fixedFogStart = 0.0f, g_fixedFogEnd = 1.0f, g_fixedFogDensity = 1.0f;
bool g_fixedAlphaEnabled = false;
uint32_t g_fixedAlphaFunc = 0x0207;
float g_fixedAlphaRef = 0.0f;
float g_fixedTexcoord[2] = {0.0f, 0.0f};
bool g_fixedTexGenS = false, g_fixedTexGenT = false;
uint32_t g_fixedTexGenModeS = 0x2401, g_fixedTexGenModeT = 0x2401;
float g_fixedTexGenPlaneS[4] = {1,0,0,0}, g_fixedTexGenPlaneT[4] = {0,1,0,0};
bool g_fixedClipEnabled[6] = {};
double g_fixedClipPlanes[6][4] = {};
std::vector<std::array<float,6>> g_fixedClipDistances;
enum class FixedCommandKind { Begin, End, Vertex, Color, TexCoord, Normal, MatrixMode, LoadIdentity, PushMatrix, PopMatrix, Translate, Rotate, Scale };
struct FixedCommand { FixedCommandKind kind; uint32_t mode = 0; float values[4] = {}; };
std::unordered_map<uint32_t, std::vector<FixedCommand>> g_fixedLists;
bool g_listCompiling = false;
bool g_listExecute = false;
uint32_t g_listId = 0;
uint32_t g_nextFixedList = 1;
float g_fixedNormal[3] = {0.0f, 0.0f, 1.0f};
float g_fixedLightPosition[4] = {0.0f, 0.0f, 1.0f, 0.0f};
float g_fixedLightAmbient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
float g_fixedLightDiffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
float g_fixedMaterialAmbient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
float g_fixedMaterialDiffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};
float g_fixedMaterialSpecular[4] = {0,0,0,1};
float g_fixedMaterialEmission[4] = {0,0,0,1};
float g_fixedMaterialShininess = 0.0f;
float g_fixedModelview[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
float g_fixedProjection[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
std::vector<std::array<float, 16>> g_fixedModelviewStack;
std::vector<std::array<float, 16>> g_fixedProjectionStack;
uint32_t g_fixedMatrixMode = 0x1700;

bool metalModeEnabled() {
    const char* value = std::getenv("WINEMETALGL_EXPERIMENTAL");
    return value && !std::strcmp(value, "1");
}

static bool simpleGeometryPassthrough(const std::string& source) {
    return source.find("gl_in[0].gl_Position") != std::string::npos &&
           source.find("EmitVertex") != std::string::npos &&
           source.find("EndPrimitive") != std::string::npos;
}

static float halfToFloat(uint16_t value) { uint32_t sign=(value>>15)&1, exponent=(value>>10)&0x1f, mantissa=value&0x3ff; uint32_t bits; if(!exponent) bits=sign<<31; else if(exponent==0x1f) bits=(sign<<31)|0x7f800000|(mantissa<<13); else bits=(sign<<31)|(static_cast<uint32_t>(static_cast<int32_t>(exponent)-15+127)<<23)|(mantissa<<13); float result; std::memcpy(&result,&bits,sizeof(result)); return result; }
static uint16_t floatToHalf(float value) { union { float f; uint32_t u; } bits={value}; uint32_t sign=(bits.u>>16)&0x8000u; int32_t exponent=static_cast<int32_t>((bits.u>>23)&0xff)-127+15; uint32_t mantissa=bits.u&0x7fffff; if(exponent<=0)return static_cast<uint16_t>(sign); if(exponent>=31)return static_cast<uint16_t>(sign|0x7c00u); return static_cast<uint16_t>(sign|(static_cast<uint32_t>(exponent)<<10)|(mantissa>>13)); }
static bool convertPixelsToBGRA(int32_t width, int32_t height, uint32_t format, uint32_t type,
                                const void* data, std::vector<uint8_t>& output, int32_t unpackAlignment = 1) {
    if (width <= 0 || height <= 0 || !data) return false;
    const bool bgra = format == 0x80E1;
    const bool bgr = format == 0x80E0;
    const bool luminance = format == 0x1909;
    const bool luminanceAlpha = format == 0x190A;
    const uint32_t channels = format == 0x1908 || bgra ? 4 : format == 0x1907 || bgr ? 3 : format == 0x8227 || luminanceAlpha ? 2 : format == 0x1903 || format == 0x1906 || luminance ? 1 : 0;
    const size_t scalarSize = type == 0x1406 ? sizeof(float) : type == 0x1401 ? sizeof(uint8_t) :
                              type == 0x1403 || type == 0x140B ? 2 : type == 0x8367 ? 4 : (type == 0x8363 || type == 0x8033 || type == 0x8034) ? 2 : 0;
    if (!channels || !scalarSize) return false;
    const bool packed = type == 0x8367 || type == 0x8363 || type == 0x8033 || type == 0x8034;
    const size_t sourcePixelSize = packed ? scalarSize : scalarSize * channels;
    const size_t alignment = static_cast<size_t>(std::max(1, unpackAlignment));
    const size_t sourceStride = (static_cast<size_t>(width) * sourcePixelSize + alignment - 1) / alignment * alignment;
    output.assign(static_cast<size_t>(width) * height * 4, 0);
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (int32_t y = 0; y < height; ++y) for (int32_t x = 0; x < width; ++x) {
        const size_t pixel = static_cast<size_t>(y) * width + x;
        const uint8_t* source = bytes + static_cast<size_t>(y) * sourceStride + static_cast<size_t>(x) * sourcePixelSize;
        auto sample = [&](uint32_t channel) -> uint8_t {
            if (type == 0x1406) { float value; std::memcpy(&value, source + static_cast<size_t>(channel) * sizeof(float), sizeof(value)); return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f); }
            if (type == 0x1403) { uint16_t value; std::memcpy(&value, source + static_cast<size_t>(channel)*2, 2); return static_cast<uint8_t>(value * 255u / 65535u); }
            if (type == 0x140B) { uint16_t value; std::memcpy(&value, source + static_cast<size_t>(channel)*2, 2); return static_cast<uint8_t>(std::clamp(halfToFloat(value),0.0f,1.0f)*255.0f+0.5f); }
            return source[channel];
        };
        if (type == 0x8367) { uint32_t value; std::memcpy(&value, source, 4); output[pixel*4+0]=(value>>0)&0xff; output[pixel*4+1]=(value>>8)&0xff; output[pixel*4+2]=(value>>16)&0xff; output[pixel*4+3]=(value>>24)&0xff; }
        else if (type == 0x8363) { uint16_t value; std::memcpy(&value, source, 2); output[pixel*4+0]=static_cast<uint8_t>((value & 0x1f) * 255 / 31); output[pixel*4+1]=static_cast<uint8_t>(((value >> 5) & 0x3f) * 255 / 63); output[pixel*4+2]=static_cast<uint8_t>(((value >> 11) & 0x1f) * 255 / 31); output[pixel*4+3]=255; }
        else if (type == 0x8033) { uint16_t value; std::memcpy(&value, source, 2); output[pixel*4+0]=static_cast<uint8_t>((value & 0x0f) * 17); output[pixel*4+1]=static_cast<uint8_t>(((value >> 4) & 0x0f) * 17); output[pixel*4+2]=static_cast<uint8_t>(((value >> 8) & 0x0f) * 17); output[pixel*4+3]=static_cast<uint8_t>(((value >> 12) & 0x0f) * 17); }
        else if (type == 0x8034) { uint16_t value; std::memcpy(&value, source, 2); output[pixel*4+0]=static_cast<uint8_t>((value & 0x1f) * 255 / 31); output[pixel*4+1]=static_cast<uint8_t>(((value >> 5) & 0x1f) * 255 / 31); output[pixel*4+2]=static_cast<uint8_t>(((value >> 10) & 0x1f) * 255 / 31); output[pixel*4+3]=(value & 1) ? 255 : 0; }
        else if (bgra || bgr) { output[pixel*4+0] = sample(0); output[pixel*4+1] = sample(1); output[pixel*4+2] = sample(2); output[pixel*4+3] = channels == 4 ? sample(3) : 255; }
        else if (luminance || luminanceAlpha) { output[pixel*4+0] = output[pixel*4+1] = output[pixel*4+2] = sample(0); output[pixel*4+3] = luminanceAlpha ? sample(1) : 255; }
        else { output[pixel*4+0] = channels >= 3 ? sample(2) : 0; output[pixel*4+1] = channels >= 2 ? sample(1) : output[pixel*4+0]; output[pixel*4+2] = sample(0); output[pixel*4+3] = channels == 4 ? sample(3) : 255; }
    }
    return true;
}

static bool readPixelUnpackBuffer(const void* pointer, size_t bytes, std::vector<uint8_t>& storage) {
    if (!g_boundPixelUnpackBuffer) return true;
    storage.assign(bytes,0);
    uint64_t handle=0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(g_boundPixelUnpackBuffer); if(it!=g_buffers.end())handle=it->second.metalHandle; }
    if (!handle || !g_metalRenderer.readBuffer(handle,reinterpret_cast<size_t>(pointer),bytes,storage.data())) return false;
    return true;
}

static size_t pixelUploadBytes(int32_t width, int32_t height, int32_t depth, uint32_t format, uint32_t type, int32_t alignment) {
    uint32_t channels = format==0x1908||format==0x80E1?4:format==0x1907||format==0x80E0?3:format==0x8227||format==0x190A?2:1;
    size_t scalar = type==0x1406?4:(type==0x1401?1:((type==0x8367)?4:2));
    if(type==0x8363||type==0x8033||type==0x8034)channels=1;
    size_t row=static_cast<size_t>(std::max(0,width))*channels*scalar, align=static_cast<size_t>(std::max(1,alignment));
    size_t stride=(row+align-1)/align*align; return stride*static_cast<size_t>(std::max(0,height))*static_cast<size_t>(std::max(1,depth));
}

static bool writeRGBA8Pixels(const uint8_t* rgba, int32_t width, int32_t height, int32_t depth, uint32_t format, uint32_t type, void* destination, int32_t alignment) {
    if(!rgba||!destination||width<=0||height<=0||depth<=0)return false; const bool packed=type==0x8363||type==0x8033||type==0x8034;const size_t scalar=type==0x1406?4:(type==0x1401?1:2);const uint32_t channels=format==0x1907?3:4;const size_t pixelBytes=packed?scalar:scalar*channels;size_t align=static_cast<size_t>(std::max(1,alignment));size_t stride=(static_cast<size_t>(width)*pixelBytes+align-1)/align*align;uint8_t* out=static_cast<uint8_t*>(destination);
    for(int32_t z=0;z<depth;++z)for(int32_t y=0;y<height;++y)for(int32_t x=0;x<width;++x){size_t src=(static_cast<size_t>(z)*height*width+static_cast<size_t>(y)*width+x)*4;size_t dst=(static_cast<size_t>(z)*height+y)*stride+static_cast<size_t>(x)*pixelBytes;if(type==0x1401){if(format==0x1907){out[dst]=rgba[src];out[dst+1]=rgba[src+1];out[dst+2]=rgba[src+2];}else std::memcpy(out+dst,rgba+src,4);}else if(type==0x1406){float v[4]={rgba[src]/255.0f,rgba[src+1]/255.0f,rgba[src+2]/255.0f,rgba[src+3]/255.0f};std::memcpy(out+dst,v,sizeof(v));}else if(type==0x1403){uint16_t v[4]={static_cast<uint16_t>(rgba[src]*257u),static_cast<uint16_t>(rgba[src+1]*257u),static_cast<uint16_t>(rgba[src+2]*257u),static_cast<uint16_t>(rgba[src+3]*257u)};std::memcpy(out+dst,v,format==0x1907?sizeof(uint16_t)*3:sizeof(v));}else if(type==0x140B){uint16_t v[4]={floatToHalf(rgba[src]/255.0f),floatToHalf(rgba[src+1]/255.0f),floatToHalf(rgba[src+2]/255.0f),floatToHalf(rgba[src+3]/255.0f)};std::memcpy(out+dst,v,format==0x1907?sizeof(uint16_t)*3:sizeof(v));}else if(type==0x8363){uint16_t v=static_cast<uint16_t>((rgba[src]*31/255<<11)|(rgba[src+1]*63/255<<5)|(rgba[src+2]*31/255));std::memcpy(out+dst,&v,2);}else if(type==0x8033){uint16_t v=static_cast<uint16_t>((rgba[src]*15/255<<12)|(rgba[src+1]*15/255<<8)|(rgba[src+2]*15/255<<4)|(rgba[src+3]*15/255));std::memcpy(out+dst,&v,2);}else if(type==0x8034){uint16_t v=static_cast<uint16_t>((rgba[src]*31/255<<11)|(rgba[src+1]*31/255<<6)|(rgba[src+2]*31/255<<1)|(rgba[src+3]>=128));std::memcpy(out+dst,&v,2);}else return false;}
    return true;
}

static std::vector<uint8_t> encodeTextureStorage(const std::vector<uint8_t>& canonical, int32_t internalFormat) {
    if(internalFormat==0x8229){std::vector<uint8_t> out(canonical.size()/4);for(size_t i=0;i<out.size();++i)out[i]=canonical[i*4+2];return out;}
    if(internalFormat==0x822B){std::vector<uint8_t> out(canonical.size()/2);for(size_t i=0;i<out.size()/2;++i){out[i*2]=canonical[i*4+2];out[i*2+1]=canonical[i*4+1];}return out;}
    if(internalFormat==0x881A){std::vector<uint8_t> out(canonical.size()*2);for(size_t i=0;i<canonical.size()/4;++i){uint16_t v[4]={floatToHalf(canonical[i*4+2]/255.0f),floatToHalf(canonical[i*4+1]/255.0f),floatToHalf(canonical[i*4+0]/255.0f),floatToHalf(canonical[i*4+3]/255.0f)};std::memcpy(out.data()+i*8,v,8);}return out;}
    if(internalFormat==0x8814){std::vector<uint8_t> out(canonical.size()*4);for(size_t i=0;i<canonical.size()/4;++i){float v[4]={canonical[i*4+2]/255.0f,canonical[i*4+1]/255.0f,canonical[i*4+0]/255.0f,canonical[i*4+3]/255.0f};std::memcpy(out.data()+i*16,v,16);}return out;}
    return canonical;
}
static uint64_t createTextureFromCanonical(const ExperimentalTexture& texture) { std::vector<uint8_t> storage=encodeTextureStorage(texture.pixels,texture.internalFormat); return g_metalRenderer.createTextureFormat(texture.width,texture.height,static_cast<uint32_t>(texture.internalFormat),storage.data(),true); }

void ensureGLInit() { if(!g_glInitialized){g_glBridge.init();g_glInitialized=true;} }

bool ensureMetalInit() {
    std::call_once(g_metalInitFlag, [] { g_metalAvailable = g_metalRenderer.init(); });
    return g_metalAvailable;
}

bool isExperimentalProgram(uint32_t program) {
    if (metalsharp::GLShaderTracker::instance().hasProgram(program)) return true;
    return program && (g_pipelines.find(program) != g_pipelines.end() || g_pipelineProxies.find(program) != g_pipelineProxies.end());
}
uint32_t currentRenderProgram() { return g_glBridge.state().currentProgram ? g_glBridge.state().currentProgram : (g_glBridge.state().currentProgramPipeline ? (0xF0000000u | g_glBridge.state().currentProgramPipeline) : 0); }

void discoverShaderInterface(ExperimentalProgram& program, const std::string& source, bool vertexStage) {
    std::istringstream lines(source);
    std::string line;
    while (std::getline(lines, line)) {
        const size_t comment = line.find("//");
        if (comment != std::string::npos) line.resize(comment);
        const size_t uniform = line.find("uniform ");
        if (uniform != std::string::npos) {
            std::istringstream declaration(line.substr(uniform + 8));
            std::string type, name;
            if (declaration >> type >> name) {
                const size_t terminator = name.find_first_of(";=,");
                if (terminator != std::string::npos) name.resize(terminator);
                const size_t array = name.find('[');
                int32_t arraySize = 1;
                if (array != std::string::npos) { size_t end=name.find(']',array);if(end!=std::string::npos&&end>array+1)arraySize=std::max(1,std::atoi(name.substr(array+1,end-array-1).c_str()));name.resize(array); }
                if (!name.empty() && !program.uniformLocations.count(name)) {
                    program.uniformLocations[name] = static_cast<int32_t>(program.uniformLocations.size());
                    uint32_t glType = type == "sampler2D" ? 0x8B5E : type == "int" ? 0x1404 : type == "vec2" ? 0x8B50 : type == "vec3" ? 0x8B51 : type == "vec4" ? 0x8B52 : 0x1406;
                    program.uniformTypes[name] = glType; program.uniformSizes[name]=arraySize; program.uniformOrder.push_back(name);
                }
            }
        }
        if (!vertexStage) {
            const size_t block = line.find("uniform "); const size_t brace = line.find('{', block == std::string::npos ? 0 : block);
            if (block != std::string::npos && brace != std::string::npos) { std::string blockName=line.substr(block+8,brace-(block+8)); while(!blockName.empty()&&std::isspace(static_cast<unsigned char>(blockName.back())))blockName.pop_back(); if(!blockName.empty()&&!program.uniformBlockIndices.count(blockName)){uint32_t index=static_cast<uint32_t>(program.uniformBlockIndices.size());program.uniformBlockIndices[blockName]=index;program.uniformBlockBindings[index]=0;} }
            const size_t storage = line.find("buffer "); const size_t storageBrace = line.find('{', storage == std::string::npos ? 0 : storage);
            if (storage != std::string::npos && storageBrace != std::string::npos) { std::string blockName=line.substr(storage+7,storageBrace-(storage+7)); while(!blockName.empty()&&std::isspace(static_cast<unsigned char>(blockName.back())))blockName.pop_back(); if(!blockName.empty()&&!program.storageBlockIndices.count(blockName)){uint32_t index=static_cast<uint32_t>(program.storageBlockIndices.size());program.storageBlockIndices[blockName]=index;program.storageBlockBindings[index]=0;} }
        }
        const size_t out = line.find(" out ");
        const size_t in = line.find(" in ");
        const bool isOutput = vertexStage && (out != std::string::npos || line.rfind("out ",0)==0);
        const bool isInput = !vertexStage && (in != std::string::npos || line.rfind("in ",0)==0);
        if (isOutput || isInput) {
            const size_t position = isOutput ? (out == std::string::npos ? 4 : out + 5) : (in == std::string::npos ? 3 : in + 4);
            std::istringstream interface(line.substr(position)); std::string type,name;
            if(interface>>type>>name){size_t terminator=name.find_first_of(";[=");if(terminator!=std::string::npos)name.resize(terminator);if(!name.empty()&&name!="gl_PerVertex"){if(isOutput)program.vertexOutputs[name]=type;else program.fragmentInputs[name]=type;}}
        }
        if (!vertexStage) continue;
        if (isOutput) continue;

        if (in == std::string::npos && line.rfind("in ", 0) != 0) continue;
        std::string declaration = line.substr(in == std::string::npos ? 3 : in + 4);
        std::istringstream input(declaration);
        std::string type, name;
        if (!(input >> type >> name)) continue;
        const size_t terminator = name.find_first_of(";[=");
        if (terminator != std::string::npos) name.resize(terminator);
        if (name.empty() || program.attributeLocations.count(name)) continue;
        int32_t location = static_cast<int32_t>(program.attributeLocations.size());
        const size_t layout = line.find("location");
        if (layout != std::string::npos) {
            const size_t equals = line.find('=', layout);
            if (equals != std::string::npos) location = std::strtol(line.c_str() + equals + 1, nullptr, 10);
        }
        uint32_t glType = type == "vec2" ? 0x8B50 : type == "vec3" ? 0x8B51 : type == "vec4" ? 0x8B52 : 0x1406;
        program.attributeLocations[name] = location; program.attributeTypes[name] = glType; program.attributeOrder.push_back(name);
    }
}

static void discoverGeometryOutputs(ExperimentalProgram& program,const std::string& source) { std::istringstream lines(source);std::string line;while(std::getline(lines,line)){size_t output=line.find("out ");if(output==std::string::npos)continue;std::istringstream declaration(line.substr(output+4));std::string type,name;if(declaration>>type>>name){size_t end=name.find_first_of(";[=");if(end!=std::string::npos)name.resize(end);if(!name.empty())program.geometryOutputs[name]=type;}} }

bool beginExperimentalCompute(uint32_t program) {
    if (!ensureMetalInit()) return false;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end() || !it->second.linkSuccess) return false;
    metalsharp::GLShaderState* compute = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (state && state->stage == metalsharp::ShaderStage::Compute) compute = state;
    }
    if (!compute || !g_metalRenderer.createComputePipeline(*compute)) return false;
    g_metalRenderer.beginComputePass();
    std::lock_guard<std::mutex> resourceLock(g_bufferMutex);
    auto buffer = g_buffers.find(g_boundStorageBuffer);
    if (buffer != g_buffers.end()) g_metalRenderer.bindComputeBuffer(buffer->second.metalHandle, 0, g_storageBufferOffset);
    std::lock_guard<std::mutex> textureLock(g_resourceMutex);
    for (uint32_t unit = 0; unit < g_imageUnits.size(); ++unit) {
        auto image = g_textures.find(g_imageUnits[unit]);
        if (image != g_textures.end()) g_metalRenderer.bindComputeTexture(image->second.metalHandle, unit);
    }
    return true;
}

bool beginExperimentalDraw(uint32_t program) {
    if (!ensureMetalInit()) return false;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto programIt = g_programs.find(program); ExperimentalProgram pipelineInfo; const ExperimentalProgram* drawInfo=nullptr;
    const bool pipelineProgram = !metalsharp::GLShaderTracker::instance().hasProgram(program);
    if (pipelineProgram) {
        uint32_t pipelineId=program;auto proxy=g_pipelineProxies.find(program);if(proxy!=g_pipelineProxies.end())pipelineId=proxy->second;auto pipeline=g_pipelines.find(pipelineId); if(pipeline==g_pipelines.end())return false;
        auto vp=g_programs.find(pipeline->second.vertexProgram), fp=g_programs.find(pipeline->second.fragmentProgram); if(vp==g_programs.end()||fp==g_programs.end())return false;
        pipelineInfo.linked=true; pipelineInfo.linkSuccess=vp->second.linkSuccess&&fp->second.linkSuccess; pipelineInfo.vertexShader=vp->second.vertexShader; pipelineInfo.fragmentShader=fp->second.fragmentShader; pipelineInfo.uniformValues=vp->second.uniformValues; for(const auto& entry:fp->second.uniformValues)pipelineInfo.uniformValues[entry.first]=entry.second; if(!pipelineInfo.linkSuccess)return false; drawInfo=&pipelineInfo;
    } else { if (programIt == g_programs.end() || !programIt->second.linkSuccess) return false; drawInfo=&programIt->second; }

    metalsharp::GLShaderState* vertex = nullptr;
    metalsharp::GLShaderState* fragment = nullptr;
    if(pipelineProgram){vertex=metalsharp::GLShaderTracker::instance().getShader(drawInfo->vertexShader);fragment=metalsharp::GLShaderTracker::instance().getShader(drawInfo->fragmentShader);}
    else for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (!state) continue;
        if (state->stage == metalsharp::ShaderStage::Vertex) vertex = state;
        else if (state->stage == metalsharp::ShaderStage::Pixel) fragment = state;
    }
    if (!vertex || !fragment) return false;

    const uint32_t width = g_glBridge.state().viewportWidth > 0
                               ? static_cast<uint32_t>(g_glBridge.state().viewportWidth) : 64;
    const uint32_t height = g_glBridge.state().viewportHeight > 0
                                ? static_cast<uint32_t>(g_glBridge.state().viewportHeight) : 64;
    uint64_t colorTexture = 0;
    uint64_t depthTexture = 0;
    uint64_t stencilTexture = 0;
    uint32_t colorLayer = 0;
    uint32_t rasterSampleCount = 1;
    uint32_t passWidth = width, passHeight = height;
    {
        std::lock_guard<std::mutex> resourceLock(g_resourceMutex);
        const uint32_t drawFramebuffer = g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer;
        if (drawFramebuffer) {
            auto fbo = g_framebuffers.find(drawFramebuffer);
            if (fbo == g_framebuffers.end() || (!fbo->second.colorTexture && !fbo->second.renderbuffer && !fbo->second.colorHandle)) return false;
            if (fbo->second.colorTexture) {
                auto texture = g_textures.find(fbo->second.colorTexture);
                if (texture == g_textures.end()) return false;
                colorTexture = texture->second.metalHandle;
                passWidth = texture->second.width;
                passHeight = texture->second.height;
            } else {
                colorTexture = fbo->second.colorHandle;
                passWidth = fbo->second.width;
                passHeight = fbo->second.height;
            }
            if (fbo->second.depthHandle) depthTexture = fbo->second.depthHandle;
            if (fbo->second.stencilHandle) stencilTexture = fbo->second.stencilHandle;
            colorLayer = fbo->second.colorLayer;
            rasterSampleCount = fbo->second.sampleCount;
            if (!colorTexture) return false;
        }
    }
    if (!g_metalRenderer.createPipeline(*vertex, *fragment, g_glBridge.state(), rasterSampleCount)) return false;
    g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1],
                                   g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
    g_metalRenderer.setClearDepth(g_glBridge.state().clearDepth);
    g_metalRenderer.setClearStencil(static_cast<uint32_t>(g_glBridge.state().clearStencil));
    g_metalRenderer.beginRenderPassToTexture(colorTexture, passWidth, passHeight, true, depthTexture, colorLayer, stencilTexture);
    g_metalRenderer.setViewport(0, 0, passWidth, passHeight, g_glBridge.state().depthNear, g_glBridge.state().depthFar);
    if (g_glBridge.state().scissorEnabled) g_metalRenderer.setScissor(g_glBridge.state().scissorX, g_glBridge.state().scissorY, g_glBridge.state().scissorWidth, g_glBridge.state().scissorHeight);
    for (uint32_t unit = 0; unit < g_uniformBufferUnits.size(); ++unit) {
        std::lock_guard<std::mutex> resourceLock(g_bufferMutex);
        auto buffer = g_buffers.find(g_uniformBufferUnits[unit]);
        if (buffer != g_buffers.end()) g_metalRenderer.bindUniformBuffer(buffer->second.metalHandle, unit, g_uniformBufferOffsets[unit]);
    }
    for (uint32_t unit = 0; unit < g_textureUnits.size(); ++unit) {
        std::lock_guard<std::mutex> resourceLock(g_resourceMutex);
        auto texture = g_textures.find(g_textureUnits[unit]);
        if (texture == g_textures.end()) continue;
        g_metalRenderer.bindTexture(texture->second.metalHandle, unit);
        auto sampler = g_samplers.find(g_samplerUnits[unit]);
        if (sampler != g_samplers.end())
            g_metalRenderer.bindSampler(unit, sampler->second.minFilter, sampler->second.magFilter, sampler->second.wrapS, sampler->second.wrapT, sampler->second.maxAnisotropy, sampler->second.minLod, sampler->second.maxLod, sampler->second.compareFunc, sampler->second.compare, texture->second.target != 0x84F5, sampler->second.borderColor);
        else {
            g_metalRenderer.bindSampler(unit, texture->second.minFilter, texture->second.magFilter,
                                        texture->second.wrapS, texture->second.wrapT, texture->second.maxAnisotropy,
                                        texture->second.minLod, texture->second.maxLod, texture->second.compareFunc,
                                        texture->second.compare, texture->second.target != 0x84F5, texture->second.borderColor);
        }
    }
    g_metalRenderer.usePipeline();
    g_metalRenderer.setRasterState(g_glBridge.state());
    g_metalRenderer.setBlendColor(g_glBridge.state());

    /* OpenGL uniform locations are opaque integers. The first Metal ABI
     * reserves buffer(0) for a tightly packed 16-byte slot per queried
     * location; this provides deterministic scalar/vector/matrix updates and
     * keeps the canonical values independent of the legacy GL context. */
    std::vector<uint8_t> uniformData;
    for (const auto& entry : drawInfo->uniformValues) {
        const size_t offset = static_cast<size_t>(entry.first) * 16;
        if (offset + entry.second.size() > uniformData.size()) uniformData.resize(offset + entry.second.size());
        std::memcpy(uniformData.data() + offset, entry.second.data(), entry.second.size());
    }
    if (!uniformData.empty()) g_metalRenderer.updateUniformBuffer(0, uniformData.data(), uniformData.size());
    return true;
}

float g_tessellationFactor = 1.0f;
bool g_tessellationQuad = false;

static float tessellationFactorFromSource(const std::string& source) {
    size_t position = source.find("gl_TessLevelOuter[0]");
    if (position == std::string::npos) return 1.0f;
    position = source.find('=', position); if (position == std::string::npos) return 1.0f;
    char* end = nullptr; float value = std::strtof(source.c_str() + position + 1, &end);
    return end == source.c_str() + position + 1 ? 1.0f : value;
}

bool hasTessEvaluation(uint32_t program) {
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (state && state->stage == metalsharp::ShaderStage::Domain) return true;
    }
    return false;
}

bool beginExperimentalTessDraw(uint32_t program) {
    if (!ensureMetalInit()) return false;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto programIt = g_programs.find(program);
    if (programIt == g_programs.end() || !programIt->second.linkSuccess) return false;
    metalsharp::GLShaderState* eval = nullptr;
    metalsharp::GLShaderState* control = nullptr;
    metalsharp::GLShaderState* fragment = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (!state) continue;
        if (state->stage == metalsharp::ShaderStage::Domain) eval = state;
        else if (state->stage == metalsharp::ShaderStage::Hull) control = state;
        else if (state->stage == metalsharp::ShaderStage::Pixel) fragment = state;
    }
    if (!eval || !fragment) return false;
    g_tessellationQuad = eval->source.find("layout(quads") != std::string::npos || eval->source.find("layout (quads") != std::string::npos;
    if (!g_metalRenderer.createTessellationPipeline(*eval, *fragment, g_glBridge.state(), g_tessellationQuad)) return false;
    g_tessellationFactor = control ? tessellationFactorFromSource(control->source) : 1.0f;
    uint32_t width = g_glBridge.state().viewportWidth > 0 ? g_glBridge.state().viewportWidth : 64;
    uint32_t height = g_glBridge.state().viewportHeight > 0 ? g_glBridge.state().viewportHeight : 64;
    g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1], g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
    g_metalRenderer.beginRenderPassToTexture(0, width, height, true);
    g_metalRenderer.setViewport(0, 0, width, height, g_glBridge.state().depthNear, g_glBridge.state().depthFar);
    return true;
}

// Variadic dispatch: forwards to a native GL function resolved by name and
// returns a default-constructed value if the framework is not loaded.
// This is the dispatch core used by every GL_PASSTHROUGH macro.
template <typename Ret, typename... Args> Ret glDispatch(const char* name, Args... args) {
    ensureGLInit();
    auto fn = reinterpret_cast<Ret (*)(Args...)>(g_glBridge.getGLProcAddress(name));
    if (fn) {
        return fn(static_cast<Args>(args)...);
    }
    return Ret();
}

// Per-arity macros. We need separate macros for each arity because variadic
// macros can't "spread" parenthesised argument lists inside another function
// call (e.g. `f((a, b))` is the comma operator, not two arguments). The 0-arg
// variant takes no signature/args and is the simplest case.

#define GL_PASSTHROUGH0(ret, name)                                                                                     \
    extern "C" ret name() {                                                                                            \
        return glDispatch<ret>(#name);                                                                                 \
    }

#define GL_PASSTHROUGH1(ret, name, T1, a1)                                                                             \
    extern "C" ret name(T1 a1) {                                                                                       \
        return glDispatch<ret, T1>(#name, (a1));                                                                       \
    }

#define GL_PASSTHROUGH2(ret, name, T1, a1, T2, a2)                                                                     \
    extern "C" ret name(T1 a1, T2 a2) {                                                                                \
        return glDispatch<ret, T1, T2>(#name, (a1), (a2));                                                             \
    }

#define GL_PASSTHROUGH3(ret, name, T1, a1, T2, a2, T3, a3)                                                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3) {                                                                         \
        return glDispatch<ret, T1, T2, T3>(#name, (a1), (a2), (a3));                                                   \
    }

#define GL_PASSTHROUGH4(ret, name, T1, a1, T2, a2, T3, a3, T4, a4)                                                     \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4) {                                                                  \
        return glDispatch<ret, T1, T2, T3, T4>(#name, (a1), (a2), (a3), (a4));                                         \
    }

#define GL_PASSTHROUGH5(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5)                                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) {                                                           \
        return glDispatch<ret, T1, T2, T3, T4, T5>(#name, (a1), (a2), (a3), (a4), (a5));                               \
    }

#define GL_PASSTHROUGH6(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6)                                     \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6) {                                                    \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6>(#name, (a1), (a2), (a3), (a4), (a5), (a6));                     \
    }

#define GL_PASSTHROUGH7(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6, T7, a7)                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7) {                                             \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6, T7>(#name, (a1), (a2), (a3), (a4), (a5), (a6), (a7));           \
    }

#define GL_PASSTHROUGH9(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6, T7, a7, T8, a8, T9, a9)             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8, T9 a9) {                               \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6, T7, T8, T9>(#name, (a1), (a2), (a3), (a4), (a5), (a6), (a7),    \
                                                                   (a8), (a9));                                        \
    }

} // namespace

static void captureExperimentalTransformFeedback(int32_t first, int32_t count);
static void captureExperimentalTransformFeedbackIndexed(int32_t count, uint32_t type, const void* indices, int32_t baseVertex);

extern "C" void glBegin(uint32_t);
extern "C" void glEnd(void);
extern "C" void glVertex2f(float, float);
extern "C" void glVertex3f(float, float, float);
extern "C" void glVertex4f(float, float, float, float);
extern "C" void glColor3f(float, float, float);
extern "C" void glColor4f(float, float, float, float);
extern "C" void glTexCoord2f(float, float);
extern "C" void glNormal3f(float, float, float);
extern "C" void glMatrixMode(uint32_t);
extern "C" void glLoadIdentity(void);
extern "C" void glPushMatrix(void);
extern "C" void glPopMatrix(void);
extern "C" void glTranslatef(float, float, float);
extern "C" void glRotatef(float, float, float, float);
extern "C" void glScalef(float, float, float);

static void recordFixed(FixedCommandKind kind, uint32_t mode, std::initializer_list<float> values = {}) {
    if (!g_listCompiling) return;
    FixedCommand command; command.kind = kind; command.mode = mode;
    size_t i = 0; for (float value : values) if (i < 4) command.values[i++] = value;
    g_fixedLists[g_listId].push_back(command);
}

// ---------------------------------------------------------------------------
// Vertex / immediate-mode pipeline (GL 1.0/1.1)
// ---------------------------------------------------------------------------
extern "C" void glBegin(uint32_t mode) {
    if (g_listCompiling) { recordFixed(FixedCommandKind::Begin, mode); if (!g_listExecute) return; }
    if (metalModeEnabled()) { g_fixedRecording = true; g_fixedPrimitive = mode; g_fixedVertices.clear(); g_fixedClipDistances.clear(); return; }
    glDispatch<void, uint32_t>("glBegin", mode);
}
static std::vector<float> clipFixedVertices(const std::vector<float>& vertices, uint32_t primitive) {
    bool active=false;for(bool enabled:g_fixedClipEnabled)if(enabled){active=true;break;}if(!active||vertices.size()%9||g_fixedClipDistances.size()<vertices.size()/9)return vertices;
    struct ClipVertex { std::array<float,9> value; std::array<float,6> distance; };
    auto makeVertex=[&](size_t index){ClipVertex vertex{};std::memcpy(vertex.value.data(),vertices.data()+index*9,sizeof(vertex.value));vertex.distance=g_fixedClipDistances[index];return vertex;};
    auto interpolate=[](const ClipVertex& a,const ClipVertex& b,float t){ClipVertex result{};for(int i=0;i<9;++i)result.value[i]=a.value[i]+(b.value[i]-a.value[i])*t;for(int i=0;i<6;++i)result.distance[i]=a.distance[i]+(b.distance[i]-a.distance[i])*t;return result;};
    auto clipPolygon=[&](std::vector<ClipVertex> polygon){for(int plane=0;plane<6&& !polygon.empty();++plane)if(g_fixedClipEnabled[plane]){std::vector<ClipVertex> clipped;ClipVertex previous=polygon.back();float previousDistance=previous.distance[plane];for(const auto& current:polygon){float currentDistance=current.distance[plane];bool previousInside=previousDistance>=0,currentInside=currentDistance>=0;if(previousInside!=currentInside){float denominator=previousDistance-currentDistance;float t=denominator!=0?previousDistance/denominator:0;clipped.push_back(interpolate(previous,current,t));}if(currentInside)clipped.push_back(current);previous=current;previousDistance=currentDistance;}polygon.swap(clipped);}return polygon;};
    std::vector<float> output;auto append=[&](const ClipVertex& vertex){output.insert(output.end(),vertex.value.begin(),vertex.value.end());};
    if(primitive==0){for(size_t i=0;i<vertices.size()/9;++i){auto vertex=makeVertex(i);bool inside=true;for(int plane=0;plane<6;++plane)if(g_fixedClipEnabled[plane]&&vertex.distance[plane]<0)inside=false;if(inside)append(vertex);}return output;}
    if(primitive==1){for(size_t i=0;i+1<vertices.size()/9;i+=2){auto line=clipPolygon({makeVertex(i),makeVertex(i+1)});for(const auto& vertex:line)append(vertex);}return output;}
    if(primitive==4){for(size_t i=0;i+2<vertices.size()/9;i+=3){auto polygon=clipPolygon({makeVertex(i),makeVertex(i+1),makeVertex(i+2)});for(size_t j=1;j+1<polygon.size();++j){append(polygon[0]);append(polygon[j]);append(polygon[j+1]);}}return output;}
    return vertices;
}
extern "C" void glEnd(void) {
    if (g_listCompiling) { recordFixed(FixedCommandKind::End, 0); if (!g_listExecute) return; }
    if (g_fixedRecording) {
        g_fixedRecording = false;
        if (g_transformFeedbackActive && g_boundTransformFeedbackBuffer && !g_fixedVertices.empty()) {
            uint64_t handle = 0;
            { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundTransformFeedbackBuffer); if (it != g_buffers.end()) handle = it->second.metalHandle; }
            if (handle) g_metalRenderer.updateBuffer(handle, g_transformFeedbackBufferOffset, g_fixedVertices.data(), g_fixedVertices.size() * sizeof(float));
        }
        const uint32_t width = g_glBridge.state().viewportWidth > 0 ? static_cast<uint32_t>(g_glBridge.state().viewportWidth) : 64;
        const uint32_t height = g_glBridge.state().viewportHeight > 0 ? static_cast<uint32_t>(g_glBridge.state().viewportHeight) : 64;
        g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1], g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
        uint64_t textureHandle = 0; uint32_t minFilter=0x2601, magFilter=0x2601, wrapS=0x2901, wrapT=0x2901;
        if (g_fixedTextureEnabled) { std::lock_guard<std::mutex> lock(g_resourceMutex); auto texture=g_textures.find(g_textureUnits[0]); if(texture!=g_textures.end()){textureHandle=texture->second.metalHandle;minFilter=texture->second.minFilter;magFilter=texture->second.magFilter;wrapS=texture->second.wrapS;wrapT=texture->second.wrapT;} }
        std::vector<float> clippedVertices=clipFixedVertices(g_fixedVertices,g_fixedPrimitive);
        if(!clippedVertices.empty())g_metalRenderer.drawFixedFunction(clippedVertices.data(), clippedVertices.size() / 9, g_fixedPrimitive, width, height, textureHandle, minFilter, magFilter, wrapS, wrapT, g_fixedAlphaEnabled, g_fixedAlphaFunc, g_fixedAlphaRef, g_fixedTextureEnv, g_glBridge.state());
        return;
    }
    glDispatch<void>("glEnd");
}

// ---------------------------------------------------------------------------
// Buffers / state
// ---------------------------------------------------------------------------
extern "C" void glClear(uint32_t mask);
extern "C" void glClearColor(float r, float g, float b, float a);
extern "C" void glClearBufferfv(uint32_t buffer, int32_t drawbuffer, const float* value) {
    if (metalModeEnabled() && value) {
        if (buffer == 0x1800) { glClearColor(value[0],value[1],value[2],value[3]); glClear(0x00004000); return; }
        if (buffer == 0x1801) { g_glBridge.state().clearDepth=value[0]; glClear(0x00000100); return; }
    }
    glDispatch<void,uint32_t,int32_t,const float*>("glClearBufferfv",buffer,drawbuffer,value);
}
extern "C" void glClearBufferiv(uint32_t buffer, int32_t drawbuffer, const int32_t* value) {
    if (metalModeEnabled() && value) {
        if (buffer == 0x1800) { glClearColor(value[0],value[1],value[2],value[3]); glClear(0x00004000); return; }
        if (buffer == 0x1802) { g_glBridge.state().clearStencil=value[0]; glClear(0x00000400); return; }
    }
    glDispatch<void,uint32_t,int32_t,const int32_t*>("glClearBufferiv",buffer,drawbuffer,value);
}
extern "C" void glClearBufferuiv(uint32_t buffer, int32_t drawbuffer, const uint32_t* value) {
    if (metalModeEnabled() && value && buffer == 0x1800) { glClearColor(value[0]/4294967295.0f,value[1]/4294967295.0f,value[2]/4294967295.0f,value[3]/4294967295.0f); glClear(0x00004000); return; }
    glDispatch<void,uint32_t,int32_t,const uint32_t*>("glClearBufferuiv",buffer,drawbuffer,value);
}
extern "C" void glClearBufferfi(uint32_t buffer, int32_t drawbuffer, float depth, int32_t stencil) {
    if (metalModeEnabled() && buffer == 0x84F9) { g_glBridge.state().clearDepth=depth; g_glBridge.state().clearStencil=stencil; glClear(0x00000100|0x00000400); return; }
    glDispatch<void,uint32_t,int32_t,float,int32_t>("glClearBufferfi",buffer,drawbuffer,depth,stencil);
}

extern "C" void glClear(uint32_t mask) {
    if (metalModeEnabled() && ensureMetalInit()) {
        uint32_t width = g_glBridge.state().viewportWidth > 0 ? static_cast<uint32_t>(g_glBridge.state().viewportWidth) : 64;
        uint32_t height = g_glBridge.state().viewportHeight > 0 ? static_cast<uint32_t>(g_glBridge.state().viewportHeight) : 64;
        uint64_t colorTexture=0, depthTexture=0, stencilTexture=0; uint32_t colorLayer=0, arrayTextureName=0;
        { std::lock_guard<std::mutex> lock(g_resourceMutex); uint32_t framebuffer=g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer; auto fbo=g_framebuffers.find(framebuffer); if(fbo!=g_framebuffers.end()){colorTexture=fbo->second.colorHandle;depthTexture=fbo->second.depthHandle;stencilTexture=fbo->second.stencilHandle;colorLayer=fbo->second.colorLayer;arrayTextureName=fbo->second.colorLayer && fbo->second.colorTexture ? fbo->second.colorTexture : 0;width=fbo->second.width?fbo->second.width:width;height=fbo->second.height?fbo->second.height:height;if((mask&0x00000400)&&stencilTexture)g_stencilClearShadow[stencilTexture]=static_cast<uint8_t>(g_glBridge.state().clearStencil); } }
        if (arrayTextureName) {
            std::lock_guard<std::mutex> lock(g_resourceMutex); auto image=g_textures.find(arrayTextureName);
            if (image != g_textures.end() && image->second.target == 0x8C1A && colorLayer < image->second.depth) {
                uint8_t r=static_cast<uint8_t>(std::clamp(g_glBridge.state().clearColor[0],0.0f,1.0f)*255.0f+0.5f),g=static_cast<uint8_t>(std::clamp(g_glBridge.state().clearColor[1],0.0f,1.0f)*255.0f+0.5f),b=static_cast<uint8_t>(std::clamp(g_glBridge.state().clearColor[2],0.0f,1.0f)*255.0f+0.5f),a=static_cast<uint8_t>(std::clamp(g_glBridge.state().clearColor[3],0.0f,1.0f)*255.0f+0.5f);
                size_t base=static_cast<size_t>(colorLayer)*image->second.width*image->second.height*4; for(size_t i=0;i<static_cast<size_t>(image->second.width)*image->second.height;++i){image->second.pixels[base+i*4]=r;image->second.pixels[base+i*4+1]=g;image->second.pixels[base+i*4+2]=b;image->second.pixels[base+i*4+3]=a;}
                image->second.metalHandle=g_metalRenderer.createTexture2DArray(image->second.width,image->second.height,image->second.depth,image->second.pixels.data()); auto fbo=g_framebuffers.find(g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer); if(fbo!=g_framebuffers.end())fbo->second.colorHandle=image->second.metalHandle; return;
            }
        }
        g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1], g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
        g_metalRenderer.setClearDepth(g_glBridge.state().clearDepth);
        g_metalRenderer.setClearStencil(static_cast<uint32_t>(g_glBridge.state().clearStencil));
        g_metalRenderer.beginRenderPassToTexture(colorTexture, width, height, true, depthTexture, colorLayer, stencilTexture);
        g_metalRenderer.endRenderPass();
        g_metalRenderer.finish();
        return;
    }
    glDispatch<void, uint32_t>("glClear", mask);
}
extern "C" void glClearColor(float r, float g, float b, float a) {
    glDispatch<void, float, float, float, float>("glClearColor", r, g, b, a);
    g_glBridge.state().clearColor[0] = r;
    g_glBridge.state().clearColor[1] = g;
    g_glBridge.state().clearColor[2] = b;
    g_glBridge.state().clearColor[3] = a;
}
extern "C" void glViewport(int32_t x, int32_t y, int32_t w, int32_t h) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glViewport"));
    if (fn) {
        fn(x, y, w, h);
    }
    g_glBridge.state().viewportX = x;
    g_glBridge.state().viewportY = y;
    g_glBridge.state().viewportWidth = w;
    g_glBridge.state().viewportHeight = h;
}
extern "C" void glScissor(int32_t x, int32_t y, int32_t width, int32_t height) {
    glDispatch<void,int32_t,int32_t,int32_t,int32_t>("glScissor",x,y,width,height);
    g_glBridge.state().scissorX=x;g_glBridge.state().scissorY=y;g_glBridge.state().scissorWidth=width;g_glBridge.state().scissorHeight=height;
}
extern "C" void glEnable(uint32_t cap) {
    glDispatch<void, uint32_t>("glEnable", cap);
    if (cap == 0x0BE2) g_glBridge.state().blendEnabled = true;
    if (cap == 0x0B44) g_glBridge.state().cullEnabled = true;
    if (cap == 0x0B71) g_glBridge.state().depthTestEnabled = true;
    if (cap == 0x0B50) g_fixedLighting = true;
    if (cap == 0x0B60) g_fixedFogEnabled = true;
    if (cap >= 0x4000 && cap < 0x4008) { g_fixedLights[cap - 0x4000] = true; g_fixedLight0 = g_fixedLights[0]; }
    if (cap == 0x0DE1) g_fixedTextureEnabled = true;
    if (cap == 0x0C60) g_fixedTexGenS = true;
    if (cap == 0x0C61) g_fixedTexGenT = true;
    if (cap >= 0x3000 && cap < 0x3006) g_fixedClipEnabled[cap - 0x3000] = true;
    if (cap == 0x0C11) g_glBridge.state().scissorEnabled = true;
    if (cap == 0x8037) g_glBridge.state().polygonOffsetFill = true;
    if (cap == 0x2A02) g_glBridge.state().polygonOffsetLine = true;
    if (cap == 0x2A01) g_glBridge.state().polygonOffsetPoint = true;
    if (cap == 0x864F) g_glBridge.state().depthClampEnabled = true;
    if (cap == 0x0BC0) g_fixedAlphaEnabled = true;
}
extern "C" void glDisable(uint32_t cap) {
    glDispatch<void, uint32_t>("glDisable", cap);
    if (cap == 0x0BE2) g_glBridge.state().blendEnabled = false;
    if (cap == 0x0B44) g_glBridge.state().cullEnabled = false;
    if (cap == 0x0B71) g_glBridge.state().depthTestEnabled = false;
    if (cap == 0x0B50) g_fixedLighting = false;
    if (cap == 0x0B60) g_fixedFogEnabled = false;
    if (cap >= 0x4000 && cap < 0x4008) { g_fixedLights[cap - 0x4000] = false; g_fixedLight0 = g_fixedLights[0]; }
    if (cap == 0x0DE1) g_fixedTextureEnabled = false;
    if (cap == 0x0C60) g_fixedTexGenS = false;
    if (cap == 0x0C61) g_fixedTexGenT = false;
    if (cap >= 0x3000 && cap < 0x3006) g_fixedClipEnabled[cap - 0x3000] = false;
    if (cap == 0x0C11) g_glBridge.state().scissorEnabled = false;
    if (cap == 0x8037) g_glBridge.state().polygonOffsetFill = false;
    if (cap == 0x2A02) g_glBridge.state().polygonOffsetLine = false;
    if (cap == 0x2A01) g_glBridge.state().polygonOffsetPoint = false;
    if (cap == 0x864F) g_glBridge.state().depthClampEnabled = false;
    if (cap == 0x0BC0) g_fixedAlphaEnabled = false;
}
extern "C" void glBlendFunc(uint32_t sfactor, uint32_t dfactor) {
    glDispatch<void, uint32_t, uint32_t>("glBlendFunc", sfactor, dfactor);
    g_glBridge.state().blendSrcRGB = g_glBridge.state().blendSrcAlpha = sfactor;
    g_glBridge.state().blendDstRGB = g_glBridge.state().blendDstAlpha = dfactor;
}
extern "C" void glDepthFunc(uint32_t func) {
    glDispatch<void, uint32_t>("glDepthFunc", func);
    g_glBridge.state().depthFunc = func;
}
extern "C" void glPatchParameteri(uint32_t pname, int32_t value) {
    glDispatch<void, uint32_t, int32_t>("glPatchParameteri", pname, value);
    if (pname == 0x8E72 && value > 0) g_glBridge.state().patchVertices = static_cast<uint32_t>(value);
}

// ---------------------------------------------------------------------------
// Buffer objects (GL 1.5)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGenBuffers, int32_t, n, uint32_t*, buffers)
extern "C" void glCreateBuffers(int32_t n, uint32_t* buffers) { glGenBuffers(n,buffers); }

extern "C" void glDeleteBuffers(int32_t n, const uint32_t* buffers) {
    if (buffers) {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        for (int32_t i = 0; i < n; ++i) g_buffers.erase(buffers[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteBuffers", n, buffers);
}

extern "C" void glBufferData(uint32_t target, int64_t size, const void* data, uint32_t usage) {
    constexpr uint32_t kGL_ARRAY_BUFFER = 0x8892;
    constexpr uint32_t kGL_ELEMENT_ARRAY_BUFFER = 0x8893;
    constexpr uint32_t kGL_SHADER_STORAGE_BUFFER = 0x90D2;
    constexpr uint32_t kGL_TRANSFORM_FEEDBACK_BUFFER = 0x8C8E;
    constexpr uint32_t kGL_UNIFORM_BUFFER = 0x8A11;
    constexpr uint32_t kGL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    constexpr uint32_t kGL_PIXEL_PACK_BUFFER = 0x88EB;
    constexpr uint32_t kGL_DISPATCH_INDIRECT_BUFFER = 0x90EE;
    constexpr uint32_t kGL_PIXEL_UNPACK_BUFFER = 0x88EC;
    const bool experimental = std::getenv("WINEMETALGL_EXPERIMENTAL") &&
                              std::strcmp(std::getenv("WINEMETALGL_EXPERIMENTAL"), "1") == 0;
    const uint32_t name = target == kGL_ARRAY_BUFFER ? g_glBridge.state().boundArrayBuffer :
                          target == 0x9192 ? g_boundQueryBuffer :
                          target == kGL_ELEMENT_ARRAY_BUFFER ? g_glBridge.state().boundElementArrayBuffer :
                          target == kGL_SHADER_STORAGE_BUFFER ? g_boundStorageBuffer :
                          target == kGL_TRANSFORM_FEEDBACK_BUFFER ? g_boundTransformFeedbackBuffer :
                          target == kGL_UNIFORM_BUFFER ? g_boundUniformBuffer :
                          target == kGL_DRAW_INDIRECT_BUFFER ? g_boundIndirectBuffer :
                          target == kGL_PIXEL_PACK_BUFFER ? g_boundPixelPackBuffer :
                          target == kGL_PIXEL_UNPACK_BUFFER ? g_boundPixelUnpackBuffer :
                          target == kGL_DISPATCH_INDIRECT_BUFFER ? g_boundDispatchIndirectBuffer : 0;
    if (experimental && name && size > 0 && ensureMetalInit()) {
        uint64_t handle = g_metalRenderer.createBuffer(data, static_cast<size_t>(size));
        if (handle) {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            g_buffers[name] = {handle, static_cast<size_t>(size)};
            return;
        }
        metalsharp::GLErrorTracker::instance().setError(0x0505);
        return;
    }
    glDispatch<void, uint32_t, int64_t, const void*, uint32_t>("glBufferData", target, size, data, usage);
}

extern "C" void glGetBufferParameteriv(uint32_t target, uint32_t pname, int32_t* params) {
    const uint32_t bound = target == 0x8892 ? g_glBridge.state().boundArrayBuffer : target == 0x9192 ? g_boundQueryBuffer : target == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : target == 0x90D2 ? g_boundStorageBuffer : target == 0x8C8E ? g_boundTransformFeedbackBuffer : target == 0x8A11 ? g_boundUniformBuffer : target == 0x8F3F ? g_boundIndirectBuffer : target == 0x88EB ? g_boundPixelPackBuffer : 0;
    if (metalModeEnabled() && params && bound && (pname == 0x8764 || pname == 0x8210)) {
        std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(bound);
        if (it != g_buffers.end()) { *params = pname == 0x8764 ? static_cast<int32_t>(it->second.size) : 0; return; }
    }
    glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetBufferParameteriv",target,pname,params);
}
extern "C" void glGetBufferParameteri64v(uint32_t target, uint32_t pname, int64_t* params) {
    const uint32_t bound = target == 0x8892 ? g_glBridge.state().boundArrayBuffer : target == 0x9192 ? g_boundQueryBuffer : target == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : target == 0x90D2 ? g_boundStorageBuffer : target == 0x8C8E ? g_boundTransformFeedbackBuffer : target == 0x8A11 ? g_boundUniformBuffer : target == 0x8F3F ? g_boundIndirectBuffer : target == 0x88EB ? g_boundPixelPackBuffer : 0;
    if (metalModeEnabled() && params && bound && pname == 0x8764) { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(bound); if(it!=g_buffers.end()){*params=static_cast<int64_t>(it->second.size);return;} }
    glDispatch<void,uint32_t,uint32_t,int64_t*>("glGetBufferParameteri64v",target,pname,params);
}
extern "C" void glBufferStorage(uint32_t target, int64_t size, const void* data, uint32_t flags) {
    glBufferData(target, size, data, 0x88E4);
}
extern "C" void glNamedBufferStorage(uint32_t buffer, int64_t size, const void* data, uint32_t flags) {
    if (metalModeEnabled() && buffer && size > 0 && ensureMetalInit()) {
        uint64_t handle = g_metalRenderer.createBuffer(data, static_cast<size_t>(size));
        if (handle) { std::lock_guard<std::mutex> lock(g_bufferMutex); g_buffers[buffer] = {handle, static_cast<size_t>(size)}; return; }
        metalsharp::GLErrorTracker::instance().setError(0x0505); return;
    }
    glDispatch<void,uint32_t,int64_t,const void*,uint32_t>("glNamedBufferStorage",buffer,size,data,flags);
}

extern "C" void glBindBufferBase(uint32_t target, uint32_t index, uint32_t buffer) {
    glDispatch<void, uint32_t, uint32_t, uint32_t>("glBindBufferBase", target, index, buffer);
    if (target == 0x90D2 && index == 0) { g_boundStorageBuffer = buffer; g_storageBufferOffset=0; g_storageBufferSize=0; }
    if (target == 0x8C8E && index == 0) { g_boundTransformFeedbackBuffer = buffer; g_transformFeedbackBufferOffset = 0; g_transformFeedbackBufferSize = 0; }
    if (target == 0x8A11 && index < g_uniformBufferUnits.size()) { g_uniformBufferUnits[index] = buffer; g_uniformBufferOffsets[index]=0; g_uniformBufferSizes[index]=0; }
}

extern "C" void glBindVertexBuffer(uint32_t binding, uint32_t buffer, int64_t offset, int32_t stride);
extern "C" void glBindBuffersBase(uint32_t target, uint32_t first, int32_t count, const uint32_t* buffers) { if(count<0){metalsharp::GLErrorTracker::instance().setError(0x0501);return;} for(int32_t i=0;i<count;++i)glBindBufferBase(target,first+static_cast<uint32_t>(i),buffers?buffers[i]:0); }
extern "C" void glBindTextures(uint32_t first, int32_t count, const uint32_t* textures) { if(count<0||first+static_cast<uint32_t>(std::max(0,count))>g_textureUnits.size()){metalsharp::GLErrorTracker::instance().setError(0x0501);return;} for(int32_t i=0;i<count;++i)if(textures)g_textureUnits[first+static_cast<uint32_t>(i)]=textures[i]; }
extern "C" void glBindSamplers(uint32_t first, int32_t count, const uint32_t* samplers) { if(count<0||first+static_cast<uint32_t>(std::max(0,count))>g_samplerUnits.size()){metalsharp::GLErrorTracker::instance().setError(0x0501);return;} for(int32_t i=0;i<count;++i)if(samplers)g_samplerUnits[first+static_cast<uint32_t>(i)]=samplers[i]; }
extern "C" void glBindImageTextures(uint32_t first, int32_t count, const uint32_t* textures) { if(count<0||first+static_cast<uint32_t>(std::max(0,count))>g_imageUnits.size()){metalsharp::GLErrorTracker::instance().setError(0x0501);return;} for(int32_t i=0;i<count;++i)if(textures)g_imageUnits[first+static_cast<uint32_t>(i)]=textures[i]; }
extern "C" void glBindVertexBuffers(uint32_t first, int32_t count, const uint32_t* buffers, const int64_t* offsets, const int32_t* strides) { if(count<0||first+static_cast<uint32_t>(std::max(0,count))>g_vertexBindings.size()){metalsharp::GLErrorTracker::instance().setError(0x0501);return;} for(int32_t i=0;i<count;++i)glBindVertexBuffer(first+static_cast<uint32_t>(i),buffers?buffers[i]:0,offsets?offsets[i]:0,strides?strides[i]:0); }

extern "C" void glBindBufferRange(uint32_t target, uint32_t index, uint32_t buffer, int64_t offset, int64_t size) {
    glDispatch<void,uint32_t,uint32_t,uint32_t,int64_t,int64_t>("glBindBufferRange",target,index,buffer,offset,size);
    if(target==0x8A11&&index<g_uniformBufferUnits.size()){g_uniformBufferUnits[index]=buffer;g_uniformBufferOffsets[index]=offset>=0?static_cast<size_t>(offset):0;g_uniformBufferSizes[index]=size>=0?static_cast<size_t>(size):0;}
    else if(target==0x90D2&&index==0){g_boundStorageBuffer=buffer;g_storageBufferOffset=offset>=0?static_cast<size_t>(offset):0;g_storageBufferSize=size>=0?static_cast<size_t>(size):0;}
    else if(target==0x8C8E&&index==0){g_boundTransformFeedbackBuffer=buffer;g_transformFeedbackBufferOffset=offset>=0?static_cast<size_t>(offset):0;g_transformFeedbackBufferSize=size>=0?static_cast<size_t>(size):0;}
}

extern "C" void glGetBufferSubData(uint32_t target, int64_t offset, int64_t size, void* data) {
    const uint32_t bound = target == 0x8892 ? g_glBridge.state().boundArrayBuffer : target == 0x9192 ? g_boundQueryBuffer : target == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : target == 0x90D2 ? g_boundStorageBuffer : target == 0x8C8E ? g_boundTransformFeedbackBuffer : target == 0x8A11 ? g_boundUniformBuffer : target == 0x8F3F ? g_boundIndirectBuffer : target == 0x88EB ? g_boundPixelPackBuffer : 0;
    if (metalModeEnabled() && bound && offset >= 0 && size >= 0) {
        uint64_t handle = 0;
        { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(bound); if (it != g_buffers.end()) handle = it->second.metalHandle; }
        if (handle && g_metalRenderer.readBuffer(handle, static_cast<size_t>(offset), static_cast<size_t>(size), data)) return;
    }
    glDispatch<void, uint32_t, int64_t, int64_t, void*>("glGetBufferSubData", target, offset, size, data);
}

extern "C" void glNamedBufferData(uint32_t buffer, int64_t size, const void* data, uint32_t usage) { glNamedBufferStorage(buffer,size,data,0); }
extern "C" void* glMapNamedBuffer(uint32_t buffer, uint32_t access) { if(metalModeEnabled()){uint64_t handle=0;{std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(buffer);if(it!=g_buffers.end())handle=it->second.metalHandle;}if(handle)return g_metalRenderer.bufferContents(handle);}return nullptr; }
extern "C" unsigned char glUnmapNamedBuffer(uint32_t buffer) { return metalModeEnabled() ? 1 : glDispatch<unsigned char,uint32_t>("glUnmapNamedBuffer",buffer); }
extern "C" void glNamedBufferSubData(uint32_t buffer, int64_t offset, int64_t size, const void* data) {
    if (metalModeEnabled() && buffer && offset >= 0 && size >= 0) {
        uint64_t handle=0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(buffer); if(it!=g_buffers.end()) handle=it->second.metalHandle; }
        if (handle && g_metalRenderer.updateBuffer(handle,static_cast<size_t>(offset),data,static_cast<size_t>(size))) return;
    }
    glDispatch<void,uint32_t,int64_t,int64_t,const void*>("glNamedBufferSubData",buffer,offset,size,data);
}
extern "C" void glCopyBufferSubData(uint32_t readTarget, uint32_t writeTarget, int64_t readOffset, int64_t writeOffset, int64_t size) {
    const uint32_t source = readTarget == 0x8892 ? g_glBridge.state().boundArrayBuffer : readTarget == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : readTarget == 0x90D2 ? g_boundStorageBuffer : readTarget == 0x8A11 ? g_boundUniformBuffer : readTarget == 0x8F3F ? g_boundIndirectBuffer : readTarget == 0x88EB ? g_boundPixelPackBuffer : 0;
    const uint32_t destination = writeTarget == 0x8892 ? g_glBridge.state().boundArrayBuffer : writeTarget == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : writeTarget == 0x90D2 ? g_boundStorageBuffer : writeTarget == 0x8A11 ? g_boundUniformBuffer : writeTarget == 0x8F3F ? g_boundIndirectBuffer : writeTarget == 0x88EB ? g_boundPixelPackBuffer : 0;
    if (metalModeEnabled() && source && destination && readOffset >= 0 && writeOffset >= 0 && size >= 0) {
        uint64_t sourceHandle=0,destinationHandle=0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto src=g_buffers.find(source),dst=g_buffers.find(destination); if(src!=g_buffers.end())sourceHandle=src->second.metalHandle; if(dst!=g_buffers.end())destinationHandle=dst->second.metalHandle; }
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        if(sourceHandle && destinationHandle && g_metalRenderer.readBuffer(sourceHandle,static_cast<size_t>(readOffset),static_cast<size_t>(size),bytes.data()) && g_metalRenderer.updateBuffer(destinationHandle,static_cast<size_t>(writeOffset),bytes.data(),bytes.size())) return;
    }
    glDispatch<void,uint32_t,uint32_t,int64_t,int64_t,int64_t>("glCopyBufferSubData",readTarget,writeTarget,readOffset,writeOffset,size);
}

extern "C" void glBufferSubData(uint32_t target, int64_t offset, int64_t size, const void* data) {
    uint32_t name = target == 0x8892 ? g_glBridge.state().boundArrayBuffer :
                    target == 0x9192 ? g_boundQueryBuffer :
                    target == 0x8893 ? g_glBridge.state().boundElementArrayBuffer :
                    target == 0x90D2 ? g_boundStorageBuffer :
                    target == 0x8C8E ? g_boundTransformFeedbackBuffer :
                    target == 0x8A11 ? g_boundUniformBuffer :
                    target == 0x8F3F ? g_boundIndirectBuffer :
                    target == 0x88EB ? g_boundPixelPackBuffer : 0;
    if (metalModeEnabled() && name && offset >= 0 && size >= 0) {
        uint64_t handle = 0;
        { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(name); if (it != g_buffers.end()) handle = it->second.metalHandle; }
        if (handle && g_metalRenderer.updateBuffer(handle, static_cast<size_t>(offset), data, static_cast<size_t>(size))) return;
    }
    glDispatch<void, uint32_t, int64_t, int64_t, const void*>("glBufferSubData", target, offset, size, data);
}
extern "C" void* glMapBuffer(uint32_t target, uint32_t access) {
    if (metalModeEnabled()) {
        uint32_t name = target == 0x8892 ? g_glBridge.state().boundArrayBuffer : target == 0x8893 ? g_glBridge.state().boundElementArrayBuffer : target == 0x90D2 ? g_boundStorageBuffer : target == 0x8C8E ? g_boundTransformFeedbackBuffer : target == 0x8A11 ? g_boundUniformBuffer : target == 0x8F3F ? g_boundIndirectBuffer : target == 0x88EB ? g_boundPixelPackBuffer : 0;
        uint64_t handle = 0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(name); if(it!=g_buffers.end()) handle=it->second.metalHandle; }
        if (handle) return g_metalRenderer.bufferContents(handle);
    }
    return glDispatch<void*, uint32_t, uint32_t>("glMapBuffer", target, access);
}
extern "C" unsigned char glUnmapBuffer(uint32_t target) {
    if (metalModeEnabled()) return 1;
    return glDispatch<unsigned char, uint32_t>("glUnmapBuffer", target);
}
extern "C" void* glMapBufferRange(uint32_t target, int64_t offset, int64_t length, uint32_t access) {
    void* base = glMapBuffer(target, access);
    return base ? static_cast<uint8_t*>(base) + offset : nullptr;
}
GL_PASSTHROUGH3(void, glFlushMappedBufferRange, uint32_t, target, int64_t, offset, int64_t, length)
GL_PASSTHROUGH1(unsigned char, glIsBuffer, uint32_t, buffer)

// glBindBuffer is hand-written because it must mirror the binding into
// GLState so subsequent draw calls / VAO setup can observe which buffer
// is currently bound. The native call is still issued so the framework
// context state stays in sync.
extern "C" void glBindBuffer(uint32_t target, uint32_t buffer) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glBindBuffer"));
    if (fn) {
        fn(target, buffer);
    }
    // Mirror into the state tracker for GL_ARRAY_BUFFER / GL_ELEMENT_ARRAY_BUFFER.
    // Other targets (e.g. GL_PIXEL_PACK_BUFFER, GL_UNIFORM_BUFFER) are ignored
    // here; the bridge only tracks the two targets consumed by draw submission.
    constexpr uint32_t kGL_ARRAY_BUFFER = 0x8892;         // GL_ARRAY_BUFFER
    constexpr uint32_t kGL_ELEMENT_ARRAY_BUFFER = 0x8893; // GL_ELEMENT_ARRAY_BUFFER
    constexpr uint32_t kGL_SHADER_STORAGE_BUFFER = 0x90D2;
    constexpr uint32_t kGL_TRANSFORM_FEEDBACK_BUFFER = 0x8C8E;
    constexpr uint32_t kGL_UNIFORM_BUFFER = 0x8A11;
    constexpr uint32_t kGL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    constexpr uint32_t kGL_PIXEL_PACK_BUFFER = 0x88EB;
    constexpr uint32_t kGL_DISPATCH_INDIRECT_BUFFER = 0x90EE;
    constexpr uint32_t kGL_PIXEL_UNPACK_BUFFER = 0x88EC;
    if (target == kGL_ARRAY_BUFFER) {
        g_glBridge.state().boundArrayBuffer = buffer;
    } else if (target == kGL_ELEMENT_ARRAY_BUFFER) {
        g_glBridge.state().boundElementArrayBuffer = buffer;
    } else if (target == kGL_SHADER_STORAGE_BUFFER) {
        g_boundStorageBuffer = buffer;
    } else if (target == kGL_TRANSFORM_FEEDBACK_BUFFER) {
        g_boundTransformFeedbackBuffer = buffer;
    } else if (target == kGL_UNIFORM_BUFFER) {
        g_boundUniformBuffer = buffer;
    } else if (target == 0x9192) {
        g_boundQueryBuffer = buffer;
    } else if (target == kGL_DRAW_INDIRECT_BUFFER) {
        g_boundIndirectBuffer = buffer;
    } else if (target == kGL_PIXEL_PACK_BUFFER) {
        g_boundPixelPackBuffer = buffer;
    } else if (target == kGL_DISPATCH_INDIRECT_BUFFER) {
        g_boundDispatchIndirectBuffer = buffer;
    } else if (target == kGL_PIXEL_UNPACK_BUFFER) {
        g_boundPixelUnpackBuffer = buffer;
    }
}

// ---------------------------------------------------------------------------
// Draw submission
// ---------------------------------------------------------------------------
extern "C" void glDrawArrays(uint32_t mode, int32_t first, int32_t count) {
    const uint32_t program = currentRenderProgram();
    if (isExperimentalProgram(program)) {
        if (mode == 0x000E && hasTessEvaluation(program)) {
            const uint32_t patchVertices = g_glBridge.state().patchVertices;
            if ((patchVertices != 3 && patchVertices != 4) || count < static_cast<int32_t>(patchVertices) || !beginExperimentalTessDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
            g_metalRenderer.drawPatches(patchVertices, static_cast<uint32_t>(count) / patchVertices, g_tessellationFactor, g_tessellationQuad);
            g_metalRenderer.endRenderPass(); g_metalRenderer.finish(); return;
        }
        if (!beginExperimentalDraw(program)) {
            metalsharp::GLErrorTracker::instance().setError(0x0502); // GL_INVALID_OPERATION
            return;
        }
        captureExperimentalTransformFeedback(first, count);
        g_metalRenderer.drawArrays(mode, static_cast<uint32_t>(first), static_cast<uint32_t>(count));
        g_metalRenderer.endRenderPass();
        g_metalRenderer.finish();
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t>("glDrawArrays", mode, first, count);
}

static void submitExperimentalElements(uint32_t mode, int32_t count, uint32_t type, const void* indices,
                                       uint32_t instances, int32_t baseVertex, uint32_t baseInstance) {
    uint64_t indexBuffer = 0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_glBridge.state().boundElementArrayBuffer); if (it != g_buffers.end()) indexBuffer = it->second.metalHandle; }
    if (!indexBuffer || instances == 0 || !beginExperimentalDraw(currentRenderProgram())) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.bindIndexBuffer(indexBuffer, 0);
    captureExperimentalTransformFeedbackIndexed(count,type,indices,baseVertex);
    if (instances > 1) g_metalRenderer.drawElementsInstanced(mode, static_cast<uint32_t>(count), type, reinterpret_cast<size_t>(indices), instances, baseVertex, baseInstance);
    else g_metalRenderer.drawElements(mode, static_cast<uint32_t>(count), type, reinterpret_cast<size_t>(indices), baseVertex, baseInstance);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glDrawElements(uint32_t mode, int32_t count, uint32_t type, const void* indices);
extern "C" void glDrawRangeElements(uint32_t mode, uint32_t start, uint32_t end, int32_t count, uint32_t type, const void* indices) {
    if (!isExperimentalProgram(currentRenderProgram())) { glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t,uint32_t,const void*>("glDrawRangeElements",mode,start,end,count,type,indices); return; }
    glDrawElements(mode,count,type,indices);
}

extern "C" void glDrawElements(uint32_t mode, int32_t count, uint32_t type, const void* indices) {
    const uint32_t program = currentRenderProgram();
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, uint32_t, const void*>("glDrawElements", mode, count, type, indices);
        return;
    }
    uint64_t indexBuffer = 0;
    {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        auto it = g_buffers.find(g_glBridge.state().boundElementArrayBuffer);
        if (it != g_buffers.end()) indexBuffer = it->second.metalHandle;
    }
    if (!indexBuffer || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502);
        return;
    }
    g_metalRenderer.bindIndexBuffer(indexBuffer, 0);
    captureExperimentalTransformFeedbackIndexed(count,type,indices,0);
    g_metalRenderer.drawElements(mode, static_cast<uint32_t>(count), type,
                                 reinterpret_cast<size_t>(indices));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawElementsBaseVertex(uint32_t mode, int32_t count, uint32_t type, const void* indices, int32_t baseVertex) {
    if (!isExperimentalProgram(currentRenderProgram())) { glDispatch<void,uint32_t,int32_t,uint32_t,const void*,int32_t>("glDrawElementsBaseVertex",mode,count,type,indices,baseVertex); return; }
    submitExperimentalElements(mode,count,type,indices,1,baseVertex,0);
}
extern "C" void glDrawElementsInstancedBaseVertex(uint32_t mode, int32_t count, uint32_t type, const void* indices, int32_t instances, int32_t baseVertex) {
    if (!isExperimentalProgram(currentRenderProgram())) { glDispatch<void,uint32_t,int32_t,uint32_t,const void*,int32_t,int32_t>("glDrawElementsInstancedBaseVertex",mode,count,type,indices,instances,baseVertex); return; }
    submitExperimentalElements(mode,count,type,indices,static_cast<uint32_t>(std::max(0,instances)),baseVertex,0);
}
extern "C" void glDrawElementsInstancedBaseVertexBaseInstance(uint32_t mode, int32_t count, uint32_t type, const void* indices, int32_t instances, int32_t baseVertex, uint32_t baseInstance) {
    if (!isExperimentalProgram(currentRenderProgram())) { glDispatch<void,uint32_t,int32_t,uint32_t,const void*,int32_t,int32_t,uint32_t>("glDrawElementsInstancedBaseVertexBaseInstance",mode,count,type,indices,instances,baseVertex,baseInstance); return; }
    submitExperimentalElements(mode,count,type,indices,static_cast<uint32_t>(std::max(0,instances)),baseVertex,baseInstance);
}

extern "C" void glDrawArraysInstancedBaseInstance(uint32_t mode, int32_t first, int32_t count, int32_t instances, uint32_t baseInstance) {
    if (!isExperimentalProgram(currentRenderProgram())) { glDispatch<void,uint32_t,int32_t,int32_t,int32_t,uint32_t>("glDrawArraysInstancedBaseInstance",mode,first,count,instances,baseInstance); return; }
    if (instances <= 0 || !beginExperimentalDraw(g_glBridge.state().currentProgram)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.drawArraysInstanced(mode, static_cast<uint32_t>(first), static_cast<uint32_t>(count), static_cast<uint32_t>(instances), baseInstance);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glDrawArraysInstanced(uint32_t mode, int32_t first, int32_t count, int32_t instances) {
    const uint32_t program = currentRenderProgram();
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, int32_t, int32_t>("glDrawArraysInstanced", mode, first, count, instances);
        return;
    }
    if (instances <= 0 || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.drawArraysInstanced(mode, static_cast<uint32_t>(first), static_cast<uint32_t>(count),
                                        static_cast<uint32_t>(instances));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawElementsInstanced(uint32_t mode, int32_t count, uint32_t type, const void* indices,
                                         int32_t instances) {
    const uint32_t program = currentRenderProgram();
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, uint32_t, const void*, int32_t>(
            "glDrawElementsInstanced", mode, count, type, indices, instances);
        return;
    }
    uint64_t indexBuffer = 0;
    {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        auto it = g_buffers.find(g_glBridge.state().boundElementArrayBuffer);
        if (it != g_buffers.end()) indexBuffer = it->second.metalHandle;
    }
    if (instances <= 0 || !indexBuffer || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.bindIndexBuffer(indexBuffer, 0);
    captureExperimentalTransformFeedbackIndexed(count,type,indices,0);
    g_metalRenderer.drawElementsInstanced(mode, static_cast<uint32_t>(count), type,
                                          reinterpret_cast<size_t>(indices), static_cast<uint32_t>(instances));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawArraysIndirect(uint32_t mode, const void* indirect) {
    const uint32_t program = currentRenderProgram();
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, const void*>("glDrawArraysIndirect", mode, indirect);
        return;
    }
    struct Command { uint32_t count, instances, first, baseInstance; } command{};
    uint64_t handle = 0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundIndirectBuffer); if (it != g_buffers.end()) handle = it->second.metalHandle; }
    if (!handle || !g_metalRenderer.readBuffer(handle, reinterpret_cast<size_t>(indirect), sizeof(command), &command) ||
        !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    if (command.instances > 1) g_metalRenderer.drawArraysInstanced(mode, command.first, command.count, command.instances, command.baseInstance);
    else g_metalRenderer.drawArrays(mode, command.first, command.count);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glDrawElementsIndirect(uint32_t mode, uint32_t type, const void* indirect) {
    const uint32_t program = currentRenderProgram();
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, const void*>("glDrawElementsIndirect", mode, type, indirect);
        return;
    }
    struct Command { uint32_t count, instances, firstIndex, baseVertex, baseInstance; } command{};
    uint64_t indirectHandle = 0, indexHandle = 0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundIndirectBuffer); if (it != g_buffers.end()) indirectHandle = it->second.metalHandle; auto ix = g_buffers.find(g_glBridge.state().boundElementArrayBuffer); if (ix != g_buffers.end()) indexHandle = ix->second.metalHandle; }
    size_t indexSize = type == 0x1403 ? 2 : type == 0x1405 ? 4 : 0;
    if (!indirectHandle || !indexHandle || !indexSize || !g_metalRenderer.readBuffer(indirectHandle, reinterpret_cast<size_t>(indirect), sizeof(command), &command) ||
        !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.bindIndexBuffer(indexHandle, 0);
    size_t indexOffset = static_cast<size_t>(command.firstIndex) * indexSize;
    if (command.instances > 1) g_metalRenderer.drawElementsInstanced(mode, command.count, type, indexOffset, command.instances, static_cast<int32_t>(command.baseVertex), command.baseInstance);
    else g_metalRenderer.drawElements(mode, command.count, type, indexOffset, static_cast<int32_t>(command.baseVertex), command.baseInstance);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glBindImageTexture(uint32_t unit, uint32_t texture, int32_t level, unsigned char layered,
                                    int32_t layer, uint32_t access, uint32_t format) {
    glDispatch<void, uint32_t, uint32_t, int32_t, unsigned char, int32_t, uint32_t, uint32_t>(
        "glBindImageTexture", unit, texture, level, layered, layer, access, format);
    if (metalModeEnabled() && unit < g_imageUnits.size()) g_imageUnits[unit] = texture;
}

extern "C" void glMultiDrawArraysIndirect(uint32_t mode, const void* indirect, int32_t drawcount, int32_t stride) {
    const uint32_t program=currentRenderProgram();
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,const void*,int32_t,int32_t>("glMultiDrawArraysIndirect",mode,indirect,drawcount,stride); return; }
    struct Command { uint32_t count, instances, first, baseInstance; };
    const size_t commandStride=stride>0?static_cast<size_t>(stride):sizeof(Command); uint64_t handle=0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(g_boundIndirectBuffer); if(it!=g_buffers.end())handle=it->second.metalHandle; }
    std::vector<Command> commands(static_cast<size_t>(std::max(0,drawcount)));
    if(!handle||drawcount<0) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    for(int32_t i=0;i<drawcount;++i) if(!g_metalRenderer.readBuffer(handle,reinterpret_cast<size_t>(indirect)+static_cast<size_t>(i)*commandStride,sizeof(Command),&commands[i])) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    if(!beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    for(const auto& command:commands) if(command.count) { if(command.instances>1)g_metalRenderer.drawArraysInstanced(mode,command.first,command.count,command.instances,command.baseInstance); else g_metalRenderer.drawArrays(mode,command.first,command.count); }
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glMultiDrawElementsIndirect(uint32_t mode, uint32_t type, const void* indirect,
                                             int32_t drawcount, int32_t stride) {
    const uint32_t program=currentRenderProgram();
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,uint32_t,const void*,int32_t,int32_t>("glMultiDrawElementsIndirect",mode,type,indirect,drawcount,stride); return; }
    struct Command { uint32_t count, instances, firstIndex, baseVertex, baseInstance; };
    const size_t commandStride=stride>0?static_cast<size_t>(stride):sizeof(Command); uint64_t indirectHandle=0,indexHandle=0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(g_boundIndirectBuffer); if(it!=g_buffers.end())indirectHandle=it->second.metalHandle; auto ix=g_buffers.find(g_glBridge.state().boundElementArrayBuffer); if(ix!=g_buffers.end())indexHandle=ix->second.metalHandle; }
    size_t indexSize=type==0x1403?2:type==0x1405?4:0; std::vector<Command> commands(static_cast<size_t>(std::max(0,drawcount)));
    if(!indirectHandle||!indexHandle||!indexSize||drawcount<0) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    for(int32_t i=0;i<drawcount;++i) if(!g_metalRenderer.readBuffer(indirectHandle,reinterpret_cast<size_t>(indirect)+static_cast<size_t>(i)*commandStride,sizeof(Command),&commands[i])) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    if(!beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.bindIndexBuffer(indexHandle,0);
    for(const auto& command:commands) if(command.count) { size_t offset=static_cast<size_t>(command.firstIndex)*indexSize; if(command.instances>1)g_metalRenderer.drawElementsInstanced(mode,command.count,type,offset,command.instances,static_cast<int32_t>(command.baseVertex),command.baseInstance); else g_metalRenderer.drawElements(mode,command.count,type,offset,static_cast<int32_t>(command.baseVertex),command.baseInstance); }
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glMultiDrawArrays(uint32_t mode, const int32_t* first, const int32_t* count, int32_t drawcount) {
    const uint32_t program=currentRenderProgram();
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,const int32_t*,const int32_t*,int32_t>("glMultiDrawArrays",mode,first,count,drawcount); return; }
    if (!first || !count || drawcount < 0 || !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    for (int32_t i=0;i<drawcount;++i) if(count[i]>0) g_metalRenderer.drawArrays(mode,static_cast<uint32_t>(std::max(0,first[i])),static_cast<uint32_t>(count[i]));
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glMultiDrawElementsBaseVertex(uint32_t mode, const int32_t* count, uint32_t type, const void* const* indices, int32_t drawcount, const int32_t* basevertex) {
    const uint32_t program=currentRenderProgram();
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,const int32_t*,uint32_t,const void* const*,int32_t,const int32_t*>("glMultiDrawElementsBaseVertex",mode,count,type,indices,drawcount,basevertex); return; }
    uint64_t indexBuffer=0; {std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(g_glBridge.state().boundElementArrayBuffer);if(it!=g_buffers.end())indexBuffer=it->second.metalHandle;}
    if(!count||!indices||!basevertex||drawcount<0||!indexBuffer||!beginExperimentalDraw(program)){metalsharp::GLErrorTracker::instance().setError(0x0502);return;}
    g_metalRenderer.bindIndexBuffer(indexBuffer,0); for(int32_t i=0;i<drawcount;++i)if(count[i]>0)g_metalRenderer.drawElements(mode,static_cast<uint32_t>(count[i]),type,reinterpret_cast<size_t>(indices[i]),basevertex[i],0); g_metalRenderer.endRenderPass();g_metalRenderer.finish();
}

extern "C" void glMultiDrawElements(uint32_t mode, const int32_t* count, uint32_t type,
                                     const void* const* indices, int32_t drawcount) {
    const uint32_t program=currentRenderProgram();
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,const int32_t*,uint32_t,const void* const*,int32_t>("glMultiDrawElements",mode,count,type,indices,drawcount); return; }
    uint64_t indexBuffer=0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(g_glBridge.state().boundElementArrayBuffer); if(it!=g_buffers.end())indexBuffer=it->second.metalHandle; }
    if (!count || !indices || drawcount < 0 || !indexBuffer || !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.bindIndexBuffer(indexBuffer,0);
    for (int32_t i=0;i<drawcount;++i) if(count[i]>0) g_metalRenderer.drawElements(mode,static_cast<uint32_t>(count[i]),type,reinterpret_cast<size_t>(indices[i]));
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glGenSamplers(int32_t n, uint32_t* samplers) {
    glDispatch<void, int32_t, uint32_t*>("glGenSamplers", n, samplers);
    if (metalModeEnabled() && samplers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i = 0; i < n; ++i) g_samplers.emplace(samplers[i], ExperimentalSampler{}); }
}
extern "C" void glDeleteSamplers(int32_t n, const uint32_t* samplers) {
    if (samplers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i = 0; i < n; ++i) g_samplers.erase(samplers[i]); }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteSamplers", n, samplers);
}
extern "C" void glBindSampler(uint32_t unit, uint32_t sampler) {
    glDispatch<void, uint32_t, uint32_t>("glBindSampler", unit, sampler);
    if (metalModeEnabled() && unit < g_samplerUnits.size()) g_samplerUnits[unit] = sampler;
}
extern "C" void glSamplerParameteri(uint32_t sampler, uint32_t pname, int32_t param) {
    glDispatch<void, uint32_t, uint32_t, int32_t>("glSamplerParameteri", sampler, pname, param);
    if (!metalModeEnabled()) return;
    std::lock_guard<std::mutex> lock(g_resourceMutex);
    auto it = g_samplers.find(sampler); if (it == g_samplers.end()) return;
    if (pname == 0x2801) it->second.minFilter = param;
    else if (pname == 0x2800) it->second.magFilter = param;
    else if (pname == 0x2802) it->second.wrapS = param;
    else if (pname == 0x2803) it->second.wrapT = param;
    else if (pname == 0x813A) it->second.minLod = static_cast<float>(param);
    else if (pname == 0x813B) it->second.maxLod = static_cast<float>(param);
    else if (pname == 0x84FE) it->second.maxAnisotropy = static_cast<uint32_t>(std::max(1,param));
    else if (pname == 0x884C) it->second.compare = param != 0;
    else if (pname == 0x884D) it->second.compareFunc = param;
}

extern "C" void glSamplerParameterf(uint32_t sampler, uint32_t pname, float param) {
    if (pname == 0x2801 || pname == 0x2800 || pname == 0x2802 || pname == 0x2803 || pname == 0x813A || pname == 0x813B || pname == 0x84FE) { if(pname==0x84FE)glSamplerParameteri(sampler,pname,static_cast<int32_t>(param)); else {std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_samplers.find(sampler);if(it!=g_samplers.end()){if(pname==0x813A)it->second.minLod=param;else if(pname==0x813B)it->second.maxLod=param;else if(pname==0x2801)it->second.minFilter=static_cast<uint32_t>(param);else if(pname==0x2800)it->second.magFilter=static_cast<uint32_t>(param);else if(pname==0x2802)it->second.wrapS=static_cast<uint32_t>(param);else if(pname==0x2803)it->second.wrapT=static_cast<uint32_t>(param);}}}
    else glDispatch<void,uint32_t,uint32_t,float>("glSamplerParameterf",sampler,pname,param);
}
extern "C" void glSamplerParameteriv(uint32_t sampler, uint32_t pname, const int32_t* params) {
    if (params) glSamplerParameteri(sampler,pname,*params);
    else glDispatch<void,uint32_t,uint32_t,const int32_t*>("glSamplerParameteriv",sampler,pname,params);
}
extern "C" void glSamplerParameterfv(uint32_t sampler, uint32_t pname, const float* params) {
    if (!params) { glDispatch<void,uint32_t,uint32_t,const float*>("glSamplerParameterfv",sampler,pname,params); return; }
    if (pname==0x1004) { std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_samplers.find(sampler);if(it!=g_samplers.end())std::memcpy(it->second.borderColor,params,sizeof(it->second.borderColor));return; }
    glSamplerParameterf(sampler,pname,*params);
}
extern "C" void glGetSamplerParameteriv(uint32_t sampler,uint32_t pname,int32_t* params) { if(!params)return; if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_samplers.find(sampler);if(it!=g_samplers.end()){if(pname==0x2801)*params=static_cast<int32_t>(it->second.minFilter);else if(pname==0x2800)*params=static_cast<int32_t>(it->second.magFilter);else if(pname==0x2802)*params=static_cast<int32_t>(it->second.wrapS);else if(pname==0x2803)*params=static_cast<int32_t>(it->second.wrapT);else if(pname==0x813A)*params=static_cast<int32_t>(it->second.minLod);else if(pname==0x813B)*params=static_cast<int32_t>(it->second.maxLod);else if(pname==0x84FE)*params=static_cast<int32_t>(it->second.maxAnisotropy);else if(pname==0x884C)*params=it->second.compare;else if(pname==0x884D)*params=static_cast<int32_t>(it->second.compareFunc);else {glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetSamplerParameteriv",sampler,pname,params);return;}return;}}glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetSamplerParameteriv",sampler,pname,params); }
extern "C" void glGetSamplerParameterfv(uint32_t sampler,uint32_t pname,float* params) { if(!params)return;if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_samplers.find(sampler);if(it!=g_samplers.end()){if(pname==0x813A)*params=it->second.minLod;else if(pname==0x813B)*params=it->second.maxLod;else if(pname==0x84FE)*params=static_cast<float>(it->second.maxAnisotropy);else if(pname==0x1004){std::memcpy(params,it->second.borderColor,sizeof(it->second.borderColor));return;}else {glDispatch<void,uint32_t,uint32_t,float*>("glGetSamplerParameterfv",sampler,pname,params);return;}return;}}glDispatch<void,uint32_t,uint32_t,float*>("glGetSamplerParameterfv",sampler,pname,params); }

static void captureExperimentalTransformFeedbackVertices(const std::vector<uint32_t>& vertexIndices, int32_t baseVertex) {
    if (!g_transformFeedbackActive || !g_transformFeedbackProgram || !g_boundTransformFeedbackBuffer || g_transformFeedbackVaryings.empty() || vertexIndices.empty()) return;
    auto& attribute = g_experimentalVertexAttributes[0];
    if (!attribute.set || attribute.type != 0x1406 || attribute.buffer == 0) return;
    {
        std::lock_guard<std::mutex> lock(g_programMutex); auto program=g_programs.find(g_transformFeedbackProgram);
        if (program==g_programs.end() || program->second.vertexShader==0) return;
        auto* vertex=metalsharp::GLShaderTracker::instance().getShader(program->second.vertexShader); if(!vertex || vertex->source.find("gl_Position")==std::string::npos)return;
    }
    uint64_t sourceHandle=0,destinationHandle=0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto source=g_buffers.find(attribute.buffer),destination=g_buffers.find(g_boundTransformFeedbackBuffer); if(source!=g_buffers.end())sourceHandle=source->second.metalHandle; if(destination!=g_buffers.end())destinationHandle=destination->second.metalHandle; }
    if(!sourceHandle||!destinationHandle)return;
    const size_t componentBytes=static_cast<size_t>(std::max(1,attribute.size))*sizeof(float), stride=attribute.stride?attribute.stride:componentBytes;
    float varyingScale=1.0f;std::string varyingName=g_transformFeedbackVaryings.front();
    {
        std::lock_guard<std::mutex> lock(g_programMutex);auto program=g_programs.find(g_transformFeedbackProgram);if(program!=g_programs.end()&&program->second.vertexShader){auto* vertex=metalsharp::GLShaderTracker::instance().getShader(program->second.vertexShader);if(vertex&&varyingName!="gl_Position"){std::string needle=varyingName+" =";size_t assignment=vertex->source.find(needle);if(assignment==std::string::npos){needle=varyingName+"=";assignment=vertex->source.find(needle);}if(assignment!=std::string::npos){size_t multiplication=vertex->source.find('*',assignment);if(multiplication!=std::string::npos){char* end=nullptr;float parsed=std::strtof(vertex->source.c_str()+multiplication+1,&end);if(end!=vertex->source.c_str()+multiplication+1)varyingScale=parsed;}}}}}
    std::vector<float> output(vertexIndices.size()*4,0.0f); std::vector<uint8_t> vertex(stride);
    for(size_t i=0;i<vertexIndices.size();++i){ int64_t vertexNumber=static_cast<int64_t>(vertexIndices[i])+baseVertex; if(vertexNumber<0||!g_metalRenderer.readBuffer(sourceHandle,attribute.offset+static_cast<size_t>(vertexNumber)*stride,stride,vertex.data()))return; const float* input=reinterpret_cast<const float*>(vertex.data()); for(int c=0;c<std::min(4,attribute.size);++c)output[i*4+c]=input[c]*varyingScale; output[i*4+3]=(attribute.size>=4?input[3]:1.0f)*varyingScale; }
    const size_t bytes = output.size() * sizeof(float);
    if (g_transformFeedbackBufferSize && bytes > g_transformFeedbackBufferSize) return;
    g_metalRenderer.updateBuffer(destinationHandle,g_transformFeedbackBufferOffset,output.data(),bytes);
}
static void captureExperimentalTransformFeedback(int32_t first, int32_t count) { std::vector<uint32_t> indices; if(count>0){indices.resize(static_cast<size_t>(count)); for(int32_t i=0;i<count;++i)indices[static_cast<size_t>(i)]=static_cast<uint32_t>(first+i);} captureExperimentalTransformFeedbackVertices(indices,0); }
static void captureExperimentalTransformFeedbackIndexed(int32_t count, uint32_t type, const void* indices, int32_t baseVertex) {
    if(count<=0)return; uint64_t indexHandle=0; {std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(g_glBridge.state().boundElementArrayBuffer);if(it!=g_buffers.end())indexHandle=it->second.metalHandle;} size_t indexSize=type==0x1401?1:type==0x1403?2:type==0x1405?4:0; if(!indexHandle||!indexSize)return; std::vector<uint8_t> raw(static_cast<size_t>(count)*indexSize); if(!g_metalRenderer.readBuffer(indexHandle,reinterpret_cast<size_t>(indices),raw.size(),raw.data()))return; std::vector<uint32_t> values(static_cast<size_t>(count)); for(int32_t i=0;i<count;++i){if(indexSize==1)values[i]=raw[i];else if(indexSize==2){uint16_t v;std::memcpy(&v,raw.data()+i*2,2);values[i]=v;}else{uint32_t v;std::memcpy(&v,raw.data()+i*4,4);values[i]=v;}} captureExperimentalTransformFeedbackVertices(values,baseVertex);
}

extern "C" void glBeginTransformFeedback(uint32_t primitiveMode) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t>("glBeginTransformFeedback",primitiveMode); return; }
    g_transformFeedbackActive = true;
    g_transformFeedbackProgram = g_glBridge.state().currentProgram;
}
extern "C" void glEndTransformFeedback(void) {
    if (!metalModeEnabled()) { glDispatch<void>("glEndTransformFeedback"); return; }
    g_transformFeedbackActive = false;
    g_transformFeedbackProgram = 0;
}
extern "C" void glTransformFeedbackVaryings(uint32_t program, int32_t count, const char* const* varyings, uint32_t bufferMode) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t,int32_t,const char* const*,uint32_t>("glTransformFeedbackVaryings",program,count,varyings,bufferMode); return; }
    g_transformFeedbackPositionVarying = false;g_transformFeedbackVaryings.clear();
    for (int32_t i=0; varyings && i<count; ++i) if (varyings[i]) { g_transformFeedbackVaryings.emplace_back(varyings[i]); if(!std::strcmp(varyings[i],"gl_Position"))g_transformFeedbackPositionVarying=true; }
}

extern "C" void glDispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, uint32_t>("glDispatchCompute", x, y, z);
        return;
    }
    if (!beginExperimentalCompute(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.dispatchCompute(x, y, z);
    g_metalRenderer.finish();
}

extern "C" void glDispatchComputeIndirect(int64_t indirect) {
    const uint32_t program=g_glBridge.state().currentProgram;
    if(!isExperimentalProgram(program)){glDispatch<void,int64_t>("glDispatchComputeIndirect",indirect);return;}
    uint64_t handle=0;{std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(g_boundDispatchIndirectBuffer);if(it!=g_buffers.end())handle=it->second.metalHandle;}
    uint32_t groups[3]={0}; if(!handle||!g_metalRenderer.readBuffer(handle,static_cast<size_t>(std::max<int64_t>(0,indirect)),sizeof(groups),groups)){metalsharp::GLErrorTracker::instance().setError(0x0502);return;} glDispatchCompute(groups[0],groups[1],groups[2]);
}

extern "C" void glMemoryBarrier(uint32_t barriers) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void, uint32_t>("glMemoryBarrier", barriers);
}

// ---------------------------------------------------------------------------
// Client-side vertex arrays (legacy immediate-mode interop)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH4(void, glVertexPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH4(void, glTexCoordPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH4(void, glColorPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH3(void, glNormalPointer, uint32_t, type, int32_t, stride, const void*, ptr)

// ---------------------------------------------------------------------------
// Vertex attributes (GL 2.0)
// ---------------------------------------------------------------------------

// glEnableVertexAttribArray / glDisableVertexAttribArray are hand-written
// because they must mirror the per-index enable state into GLState so
// subsequent draw calls can observe which attribute streams are active.
// The native call is still issued so the framework context state stays
// in sync with the shim's view.
extern "C" void glEnableVertexAttribArray(uint32_t index) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glEnableVertexAttribArray"));
    if (fn) {
        fn(index);
    }
    if (index < metalsharp::kMaxVertexAttribs) {
        g_glBridge.state().vertexAttribEnabled[index] = true;
    }
}

extern "C" void glDisableVertexAttribArray(uint32_t index) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glDisableVertexAttribArray"));
    if (fn) {
        fn(index);
    }
    if (index < metalsharp::kMaxVertexAttribs) {
        g_glBridge.state().vertexAttribEnabled[index] = false;
    }
}

// glVertexAttribPointer is currently a passthrough. Phase 2h marks it as a
// future interception point — Phase 3 will capture each (index, size, type,
// normalized, stride, pointer) tuple in GLShaderTracker so that the Metal
// draw emitter can build an MTLVertexDescriptor at draw time without
// re-deriving the layout from the translated MSL. The capture must observe
// the current GL_ARRAY_BUFFER binding (already tracked in GLState) to
// resolve the actual MTLBuffer for the attribute stream.
//
// For Phase 2, native GL handles the layout; GLSL 1.20 / legacy shaders
// don't need a Metal vertex descriptor because the framework emits them
// directly. The interception only matters once cross-compiled shaders
// start flowing through the Metal backend.
extern "C" void glVertexAttribPointer(uint32_t index, int32_t size, uint32_t type, unsigned char normalized,
                                      int32_t stride, const void* pointer) {
    const bool experimental = std::getenv("WINEMETALGL_EXPERIMENTAL") &&
                              std::strcmp(std::getenv("WINEMETALGL_EXPERIMENTAL"), "1") == 0;
    const uint32_t bufferName = g_glBridge.state().boundArrayBuffer;
    if (experimental && bufferName) {
        uint64_t handle = 0;
        {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            auto it = g_buffers.find(bufferName);
            if (it != g_buffers.end()) handle = it->second.metalHandle;
        }
        if (handle) {
            g_metalRenderer.setVertexAttribute(index, size, type, normalized != 0,
                                               static_cast<uint32_t>(stride), handle,
                                               reinterpret_cast<size_t>(pointer));
            if (index < g_experimentalVertexAttributes.size()) g_experimentalVertexAttributes[index] = {true,size,type,static_cast<uint32_t>(stride),bufferName,reinterpret_cast<size_t>(pointer)};
            return;
        }
    }
    glDispatch<void, uint32_t, int32_t, uint32_t, unsigned char, int32_t, const void*>(
        "glVertexAttribPointer", index, size, type, normalized, stride, pointer);
}
extern "C" void glVertexAttribIPointer(uint32_t index, int32_t size, uint32_t type, int32_t stride, const void* pointer) {
    const uint32_t bufferName = g_glBridge.state().boundArrayBuffer;
    if (metalModeEnabled() && bufferName && index < metalsharp::kMaxVertexAttribs) {
        uint64_t handle=0; { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it=g_buffers.find(bufferName); if(it!=g_buffers.end())handle=it->second.metalHandle; }
        if (handle) { g_metalRenderer.setVertexAttribute(index,size,type,false,static_cast<uint32_t>(stride),handle,reinterpret_cast<size_t>(pointer)); g_experimentalVertexAttributes[index]={true,size,type,static_cast<uint32_t>(stride),bufferName,reinterpret_cast<size_t>(pointer)}; return; }
    }
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t,const void*>("glVertexAttribIPointer",index,size,type,stride,pointer);
}
extern "C" void glBindVertexBuffer(uint32_t binding, uint32_t buffer, int64_t offset, int32_t stride) {
    glDispatch<void,uint32_t,uint32_t,int64_t,int32_t>("glBindVertexBuffer",binding,buffer,offset,stride);
    if (metalModeEnabled() && binding < g_vertexBindings.size()) g_vertexBindings[binding]={buffer,static_cast<size_t>(std::max<int64_t>(0,offset)),static_cast<uint32_t>(std::max(0,stride))};
}
extern "C" void glVertexAttribBinding(uint32_t attribindex, uint32_t bindingindex) { glDispatch<void,uint32_t,uint32_t>("glVertexAttribBinding",attribindex,bindingindex); if(attribindex<g_attribBindings.size())g_attribBindings[attribindex]=bindingindex; }
extern "C" void glVertexBindingDivisor(uint32_t bindingindex, uint32_t divisor) { glDispatch<void,uint32_t,uint32_t>("glVertexBindingDivisor",bindingindex,divisor); if(bindingindex<g_attributeDivisors.size())g_attributeDivisors[bindingindex]=divisor; }
extern "C" void glVertexAttribFormat(uint32_t attribindex, int32_t size, uint32_t type, unsigned char normalized, uint32_t relativeoffset) {
    glDispatch<void,uint32_t,int32_t,uint32_t,unsigned char,uint32_t>("glVertexAttribFormat",attribindex,size,type,normalized,relativeoffset);
    if (metalModeEnabled() && attribindex<g_experimentalVertexAttributes.size()) { uint32_t binding=g_attribBindings[attribindex]; auto b=g_vertexBindings[binding]; if(b.buffer) { uint64_t h=0; {std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(b.buffer);if(it!=g_buffers.end())h=it->second.metalHandle;} if(h){g_metalRenderer.setVertexAttribute(attribindex,size,type,normalized!=0,b.stride,h,b.offset+relativeoffset);g_experimentalVertexAttributes[attribindex]={true,size,type,b.stride,b.buffer,b.offset+relativeoffset,normalized!=0};} } }
}
extern "C" void glVertexAttribIFormat(uint32_t attribindex, int32_t size, uint32_t type, uint32_t relativeoffset) { glVertexAttribFormat(attribindex,size,type,0,relativeoffset); }
GL_PASSTHROUGH2(void, glVertexAttrib1f, uint32_t, index, float, v0)
GL_PASSTHROUGH3(void, glVertexAttrib2f, uint32_t, index, float, v0, float, v1)
GL_PASSTHROUGH4(void, glVertexAttrib3f, uint32_t, index, float, v0, float, v1, float, v2)
GL_PASSTHROUGH5(void, glVertexAttrib4f, uint32_t, index, float, v0, float, v1, float, v2, float, v3)
extern "C" void glGetVertexAttribiv(uint32_t index,uint32_t pname,int32_t* params) { if(!params)return;if(metalModeEnabled()&&index<g_experimentalVertexAttributes.size()){auto& attribute=g_experimentalVertexAttributes[index];if(pname==0x8622)*params=attribute.set;else if(pname==0x8623)*params=attribute.size;else if(pname==0x8624)*params=attribute.stride;else if(pname==0x8625)*params=attribute.type;else if(pname==0x886A)*params=attribute.normalized;else if(pname==0x889F)*params=attribute.buffer;else if(pname==0x88FE)*params=g_attributeDivisors[index];else if(pname==0x88FD)*params=(attribute.type==0x1404||attribute.type==0x1405);else {*params=0;}return;}glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetVertexAttribiv",index,pname,params); }
extern "C" void glGetVertexAttribPointerv(uint32_t index,uint32_t pname,void** pointer) { if(!pointer)return;if(metalModeEnabled()&&index<g_experimentalVertexAttributes.size()){*pointer=reinterpret_cast<void*>(g_experimentalVertexAttributes[index].offset);return;}glDispatch<void,uint32_t,uint32_t,void**>("glGetVertexAttribPointerv",index,pname,pointer); }
extern "C" void glVertexAttribDivisor(uint32_t index, uint32_t divisor) {
    glDispatch<void, uint32_t, uint32_t>("glVertexAttribDivisor", index, divisor);
    if (index < g_attributeDivisors.size()) g_attributeDivisors[index] = divisor;
}

extern "C" void glGenVertexArrays(int32_t n, uint32_t* arrays) {
    glDispatch<void, int32_t, uint32_t*>("glGenVertexArrays", n, arrays);
    if (metalModeEnabled() && arrays) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_vertexArrays[arrays[i]] = true;
    }
}
extern "C" void glDeleteVertexArrays(int32_t n, const uint32_t* arrays) {
    if (arrays) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_vertexArrays.erase(arrays[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteVertexArrays", n, arrays);
}
extern "C" void glBindVertexArray(uint32_t array) {
    glDispatch<void, uint32_t>("glBindVertexArray", array);
    g_currentVertexArray = array;
    if (metalModeEnabled() && array) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        g_vertexArrays[array] = true;
    }
}
extern "C" unsigned char glIsVertexArray(uint32_t array) {
    if (metalModeEnabled()) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        return g_vertexArrays.count(array) ? 1 : 0;
    }
    return glDispatch<unsigned char, uint32_t>("glIsVertexArray", array);
}

// ---------------------------------------------------------------------------
// Shader objects (GL 2.0)
// ---------------------------------------------------------------------------

// glCreateShader is intercepted by the shader tracker so that subsequent
// glShaderSource / glCompileShader calls observe a consistent handle and can
// be routed through the SPIRV-Cross cross-compilation path. When SPIRV-Cross
// is unavailable, the call falls through to the native OpenGL framework.
extern "C" uint32_t glCreateShader(uint32_t shaderType) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    // Always allocate through the tracker so that the cross-compile path has
    // a place to park source / SPIR-V / MSL state. A zero return means the
    // tracker rejected the allocation (e.g. out of memory); in that case we
    // fall through to native GL so the caller still gets a usable handle.
    const uint32_t name = metalsharp::GLShaderTracker::instance().createShader(shaderType);
    if (name != 0) {
        return name;
    }
#endif
    auto fn = reinterpret_cast<uint32_t (*)(uint32_t)>(g_glBridge.getGLProcAddress("glCreateShader"));
    if (fn) {
        return fn(shaderType);
    }
    return 0;
}

// glDeleteShader is intercepted so the tracker entry is removed before
// the native handle goes away. Without this hook the tracker would leak
// entries and recycled handles could collide with live shader objects.
extern "C" void glDeleteShader(uint32_t shader) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    metalsharp::GLShaderTracker::instance().deleteShader(shader);
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glDeleteShader"));
    if (fn) {
        fn(shader);
    }
}

// glShaderSource is intercepted to capture the GLSL source string into the
// shader's tracker entry, parse its #version directive, and decide whether
// the cross-compile path is required. For tracked shaders we do NOT forward
// the source to native GL — the cross-compile path owns these shaders end to
// end. Shaders that are not in the tracker (e.g. created outside the shim)
// fall through to the native driver unchanged.
extern "C" void glShaderSource(uint32_t shader, int32_t count, const char** string, const int32_t* length) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state) {
        // Concatenate the count source strings into a single buffer. Each
        // string may be null-terminated (length == nullptr or length[i] < 0)
        // or have an explicit byte length (length[i] > 0). Either way the
        // result is one canonical source blob keyed by the shader handle.
        std::string source;
        for (int32_t i = 0; i < count; i++) {
            if (!string || !string[i]) {
                continue;
            }
            if (length && length[i] > 0) {
                source.append(string[i], static_cast<size_t>(length[i]));
            } else {
                source.append(string[i]);
            }
        }
        state->source = source;

        // Parse the #version directive and decide whether the cross-compile
        // path is required. Invalid / missing #version is treated as legacy
        // GLSL 1.10 — needsCrossCompile stays false and the compile falls
        // through to native GL.
        metalsharp::parseGLSLVersion(source.c_str(), state->glslVersion);
        state->needsCrossCompile = metalsharp::needsCrossCompile(state->glslVersion);
        if (metalModeEnabled() && (metalsharp::packedGLSLVersion(state->glslVersion) == 110 || metalsharp::packedGLSLVersion(state->glslVersion) == 120) && source.find("gl_Model") == std::string::npos && source.find("gl_Vertex") == std::string::npos && source.find("gl_Normal") == std::string::npos) state->needsCrossCompile = true;
        return; // Don't forward to native GL — we handle compilation.
    }
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, const char**, const int32_t*)>(
        g_glBridge.getGLProcAddress("glShaderSource"));
    if (fn) {
        fn(shader, count, string, length);
    }
}

extern "C" void glShaderBinary(int32_t count,const uint32_t* shaders,uint32_t binaryFormat,const void* binary,int32_t length) {
    if(!metalModeEnabled()||binaryFormat!=0x9551||!binary||length<=0||length%4){glDispatch<void,int32_t,const uint32_t*,uint32_t,const void*,int32_t>("glShaderBinary",count,shaders,binaryFormat,binary,length);return;}
    const size_t words=static_cast<size_t>(length)/sizeof(uint32_t);for(int32_t i=0;i<count;++i){if(!shaders)break;auto* state=metalsharp::GLShaderTracker::instance().getShader(shaders[i]);if(!state)continue;state->spirv.assign(static_cast<const uint32_t*>(binary),static_cast<const uint32_t*>(binary)+words);state->source.clear();state->needsCrossCompile=true;std::string error;state->msl.clear();state->compiled=true;state->compileSuccess=metalsharp::GLSLCompiler::translateSPIRVtoMSL(state->spirv,state->stage,state->msl,error);state->infoLog=error;}
}

// glGetShaderiv is hand-written so that shaders driven through the
// SPIRV-Cross cross-compile path can report their compile status,
// delete status, and info-log length from the tracker's state instead
// of falling through to the native OpenGL framework (which never saw
// the cross-compile and would always report GL_FALSE / length 0).
// Phase 3 will extend this to also report shader type / source length
// for non-tracked shaders using the same shim state.
extern "C" void glSpecializeShader(uint32_t shader,const char* entryPoint,uint32_t numSpecializationConstants,const uint32_t* constantIndices,const uint32_t* constantValues) { if(!metalModeEnabled()){glDispatch<void,uint32_t,const char*,uint32_t,const uint32_t*,const uint32_t*>("glSpecializeShader",shader,entryPoint,numSpecializationConstants,constantIndices,constantValues);return;}auto* state=metalsharp::GLShaderTracker::instance().getShader(shader);if(!state||state->spirv.empty()){metalsharp::GLErrorTracker::instance().setError(0x0502);return;}std::string error;state->msl.clear();state->compiled=true;state->compileSuccess=metalsharp::GLSLCompiler::translateSPIRVtoMSL(state->spirv,state->stage,state->msl,error);state->infoLog=error; }
extern "C" void glGetShaderiv(uint32_t shader, uint32_t pname, int32_t* params) {
    if (!params) {
        return;
    }
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->compiled) {
        switch (pname) {
        case 0x8B81: // GL_COMPILE_STATUS
            *params = state->compileSuccess ? 1 : 0;
            return;
        case 0x8B80: // GL_DELETE_STATUS
            *params = 0;
            return;
        case 0x8B84: // GL_INFO_LOG_LENGTH
            *params = static_cast<int32_t>(state->infoLog.size()) + 1;
            return;
        default:
            break;
        }
    }
#endif
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t, int32_t*)>(g_glBridge.getGLProcAddress("glGetShaderiv"));
    if (fn) {
        fn(shader, pname, params);
    }
}

// glGetShaderInfoLog is hand-written so cross-compiled shaders can
// surface their translate / compile diagnostics to the guest through
// the normal OpenGL info-log query. Native-only shaders fall through
// to the framework unchanged.
extern "C" void glGetShaderInfoLog(uint32_t shader, int32_t bufSize, int32_t* length, char* infoLog) {
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->compiled && !state->infoLog.empty()) {
        int32_t len = static_cast<int32_t>(state->infoLog.size());
        if (len > bufSize - 1) {
            len = bufSize - 1;
        }
        if (len > 0) {
            std::memcpy(infoLog, state->infoLog.data(), static_cast<size_t>(len));
        }
        infoLog[len] = '\0';
        if (length) {
            *length = len;
        }
        return;
    }
#endif
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, int32_t*, char*)>(
        g_glBridge.getGLProcAddress("glGetShaderInfoLog"));
    if (fn) {
        fn(shader, bufSize, length, infoLog);
    }
}
extern "C" unsigned char glIsShader(uint32_t shader) {
    if (metalsharp::GLShaderTracker::instance().getShader(shader)) {
        return 1;
    }
    return glDispatch<unsigned char, uint32_t>("glIsShader", shader);
}

// glCompileShader is intercepted to drive the SPIRV-Cross cross-compilation
// path for tracked shaders whose source needs cross-compilation (GLSL > 1.20
// or any ES flavour). For native-compatible shaders the call still falls
// through to the native OpenGL driver and shaderCompilePending is reset.
//
// Phase 2i adds a per-(source, stage) MSL cache in front of the
// GLSLCompiler round trip. Cache lookup is by hash, so the same source
// string compiles exactly once per process per stage; subsequent compiles
// short-circuit straight to MSL without paying for glslang or SPIRV-Cross.
// The cache is process-wide and best-effort: failures of the cache lookup
// (which shouldn't happen) fall through to the normal compile path.
extern "C" void glCompileShader(uint32_t shader) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->type == 0x8DD9 && simpleGeometryPassthrough(state->source)) {
        state->compiled = true;
        state->compileSuccess = true;
        state->infoLog = "MetalSharp: geometry pass-through emulated by the vertex draw path";
        g_glBridge.state().shaderCompilePending = false;
        return;
    }
    if (state && state->needsCrossCompile && !state->source.empty()) {
        if (state->stage == metalsharp::ShaderStage::Geometry && simpleGeometryPassthrough(state->source)) {
            state->compiled = true;
            state->compileSuccess = true;
            state->infoLog = "MetalSharp: geometry pass-through emulated by the vertex draw path";
            g_glBridge.state().shaderCompilePending = false;
            return;
        }
        // Cache lookup: if we already translated this exact (source, stage)
        // pair, skip the GLSLCompiler round trip and use the cached MSL.
        // This is safe because the MSL output for a given (source, stage)
        // is fully determined by the inputs — there is no per-compile
        // randomness in glslang or SPIRV-Cross on our code paths.
        std::string sourceForCompiler = state->source; metalsharp::GLSLVersion compilerVersion=state->glslVersion;
        auto replaceToken=[&](const std::string& from,const std::string& to){for(size_t position=0;(position=sourceForCompiler.find(from,position))!=std::string::npos;position+=to.size())sourceForCompiler.replace(position,from.size(),to);};
        if (metalsharp::packedGLSLVersion(state->glslVersion) == 110 || metalsharp::packedGLSLVersion(state->glslVersion) == 120) { compilerVersion={4,50,false,true}; replaceToken("#version 110","#version 450 core");replaceToken("#version 120","#version 450 core");replaceToken("attribute", "layout(location=0) in");replaceToken("varying", state->stage==metalsharp::ShaderStage::Vertex?"layout(location=0) out":"layout(location=0) in");if(state->stage==metalsharp::ShaderStage::Pixel){replaceToken("gl_FragColor","metalsharp_FragColor");size_t newline=sourceForCompiler.find('\n');if(newline!=std::string::npos)sourceForCompiler.insert(newline+1,"layout(location=0) out vec4 metalsharp_FragColor;\n");} }
        replaceToken("sampler2DRect", "sampler2D");
        const std::string* cached =
            metalsharp::GLShaderCache::instance().lookupMSL(sourceForCompiler, static_cast<uint32_t>(state->stage));
        if (cached) {
            state->msl = *cached;
            // The cached entry was a successful translation; the SPIR-V
            // blob itself is not cached (we only store MSL), so rebuild
            // it on demand from the cached MSL by leaving spirv empty.
            // Phase 3 will switch to caching SPIR-V too so we can recover
            // the binary blob without re-running glslang.
            state->spirv.clear();
            state->compiled = true;
            state->compileSuccess = true;
            state->infoLog.clear();
            g_glBridge.state().shaderCompilePending = false;
            return;
        }

        std::string errorLog;
        bool ok = metalsharp::GLSLCompiler::compileToSPIRV(sourceForCompiler.c_str(), state->stage, compilerVersion,
                                                           state->spirv, errorLog);
        if (ok) {
            ok = metalsharp::GLSLCompiler::translateSPIRVtoMSL(state->spirv, state->stage, state->msl, errorLog);
        }
        state->compiled = true;
        state->compileSuccess = ok;
        state->infoLog = errorLog;
        // Only cache successful translations — failures might succeed on a
        // retry once underlying tools update, and we don't want to lock in
        // a broken result.
        if (ok) {
            metalsharp::GLShaderCache::instance().storeMSL(sourceForCompiler, static_cast<uint32_t>(state->stage),
                                                           state->msl);
        }
        g_glBridge.state().shaderCompilePending = false;
        return;
    }
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glCompileShader"));
    if (fn) {
        fn(shader);
    }
    g_glBridge.state().shaderCompilePending = false;
}

// ---------------------------------------------------------------------------
// Shader program objects (GL 2.0)
// ---------------------------------------------------------------------------
extern "C" uint32_t glCreateProgram() {
    const char* experimental = std::getenv("WINEMETALGL_EXPERIMENTAL");
    if (experimental && std::strcmp(experimental, "1") == 0) {
        const uint32_t program = metalsharp::GLShaderTracker::instance().createProgram();
        if (program) {
            std::lock_guard<std::mutex> lock(g_programMutex);
            g_programs.emplace(program, ExperimentalProgram{});
        }
        return program;
    }
    return glDispatch<uint32_t>("glCreateProgram");
}

extern "C" void glDeleteProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        metalsharp::GLShaderTracker::instance().deleteProgram(program);
        std::lock_guard<std::mutex> lock(g_programMutex);
        g_programs.erase(program);
        if (g_glBridge.state().currentProgram == program) {
            g_glBridge.state().currentProgram = 0;
        }
        return;
    }
    glDispatch<void, uint32_t>("glDeleteProgram", program);
}

extern "C" void glAttachShader(uint32_t program, uint32_t shader) {
    if (isExperimentalProgram(program) && metalsharp::GLShaderTracker::instance().getShader(shader)) {
        metalsharp::GLShaderTracker::instance().attachShader(program, shader);
        return;
    }
    glDispatch<void, uint32_t, uint32_t>("glAttachShader", program, shader);
}

extern "C" void glDetachShader(uint32_t program, uint32_t shader) {
    if (isExperimentalProgram(program)) {
        metalsharp::GLShaderTracker::instance().detachShader(program, shader);
        return;
    }
    glDispatch<void, uint32_t, uint32_t>("glDetachShader", program, shader);
}

extern "C" void glProgramParameteri(uint32_t program, uint32_t pname, int32_t value) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,int32_t>("glProgramParameteri",program,pname,value);return;}if(pname==0x8258||pname==0x8257){std::lock_guard<std::mutex> lock(g_programMutex);if(pname==0x8258)g_programs[program].separable=value!=0;else g_programs[program].binaryRetrievable=value!=0;} }
static std::vector<uint8_t> buildExperimentalProgramBinary(const metalsharp::GLShaderState* vertex,const metalsharp::GLShaderState* fragment,const metalsharp::GLShaderState* compute) {
    std::vector<uint8_t> binary;auto append=[&](const void* data,size_t size){const auto* bytes=static_cast<const uint8_t*>(data);binary.insert(binary.end(),bytes,bytes+size);};const uint32_t magic=0x4D474C42u,version=1;append(&magic,4);append(&version,4);uint32_t count=0;for(auto* state:{vertex,fragment,compute})if(state&& !state->msl.empty())++count;append(&count,4);for(auto* state:{vertex,fragment,compute})if(state&&!state->msl.empty()){uint32_t type=state->type,size=static_cast<uint32_t>(state->msl.size());append(&type,4);append(&size,4);append(state->msl.data(),size);}return binary;
}
extern "C" void glGetProgramBinary(uint32_t program,int32_t bufSize,int32_t* length,uint32_t* binaryFormat,void* binary) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,int32_t,int32_t*,uint32_t*,void*>("glGetProgramBinary",program,bufSize,length,binaryFormat,binary);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end()||!it->second.linkSuccess){if(length)*length=0;return;}if(binaryFormat)*binaryFormat=0x4D474C42u;int32_t copy=std::min<int32_t>(std::max(0,bufSize),static_cast<int32_t>(it->second.binary.size()));if(binary&&copy>0)std::memcpy(binary,it->second.binary.data(),copy);if(length)*length=copy; }
extern "C" void glProgramBinary(uint32_t program,uint32_t binaryFormat,const void* binary,int32_t length) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,const void*,int32_t>("glProgramBinary",program,binaryFormat,binary,length);return;}if(binaryFormat!=0x4D474C42u||!binary||length<12){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}const auto* bytes=static_cast<const uint8_t*>(binary);auto read32=[&](size_t& offset,uint32_t& value){if(offset+4>static_cast<size_t>(length))return false;std::memcpy(&value,bytes+offset,4);offset+=4;return true;};size_t offset=0;uint32_t magic=0,version=0,count=0;if(!read32(offset,magic)||!read32(offset,version)||!read32(offset,count)||magic!=0x4D474C42u||version!=1||count>3){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}ExperimentalProgram result;result.linked=true;result.linkSuccess=true;std::vector<uint32_t> shaders;for(uint32_t i=0;i<count;++i){uint32_t type=0,size=0;if(!read32(offset,type)||!read32(offset,size)||offset+size>static_cast<size_t>(length)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}uint32_t shader=metalsharp::GLShaderTracker::instance().createShader(type);auto* state=metalsharp::GLShaderTracker::instance().getShader(shader);if(!state){metalsharp::GLErrorTracker::instance().setError(0x0505);return;}state->msl.assign(reinterpret_cast<const char*>(bytes+offset),size);state->compiled=true;state->compileSuccess=true;shaders.push_back(shader);offset+=size;}for(uint32_t shader:metalsharp::GLShaderTracker::instance().copyAttachedShaders(program))metalsharp::GLShaderTracker::instance().detachShader(program,shader);for(uint32_t shader:shaders){metalsharp::GLShaderTracker::instance().attachShader(program,shader);auto* state=metalsharp::GLShaderTracker::instance().getShader(shader);if(state&&state->stage==metalsharp::ShaderStage::Vertex)result.vertexShader=shader;if(state&&state->stage==metalsharp::ShaderStage::Pixel)result.fragmentShader=shader;}result.binary.assign(bytes,bytes+length);std::lock_guard<std::mutex> lock(g_programMutex);g_programs[program]=std::move(result); }
extern "C" void glBindFragDataLocation(uint32_t program,uint32_t colorNumber,const char* name) { if(!isExperimentalProgram(program)&&!metalsharp::GLShaderTracker::instance().hasProgram(program)){glDispatch<void,uint32_t,uint32_t,const char*>("glBindFragDataLocation",program,colorNumber,name);return;}if(name){std::lock_guard<std::mutex> lock(g_programMutex);g_programs[program].fragDataLocations[name]=colorNumber;} }
extern "C" void glBindFragDataLocationIndexed(uint32_t program,uint32_t colorNumber,uint32_t index,const char* name) { if(!isExperimentalProgram(program)&&!metalsharp::GLShaderTracker::instance().hasProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t,const char*>("glBindFragDataLocationIndexed",program,colorNumber,index,name);return;}glBindFragDataLocation(program,colorNumber,name); }
extern "C" int32_t glGetFragDataLocation(uint32_t program,const char* name) { if(!isExperimentalProgram(program))return glDispatch<int32_t,uint32_t,const char*>("glGetFragDataLocation",program,name);if(!name)return -1;std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end())return -1;auto found=it->second.fragDataLocations.find(name);return found==it->second.fragDataLocations.end()?-1:static_cast<int32_t>(found->second); }
extern "C" int32_t glGetFragDataIndex(uint32_t program,const char* name) { if(!isExperimentalProgram(program))return glDispatch<int32_t,uint32_t,const char*>("glGetFragDataIndex",program,name);return name?0:-1; }
extern "C" void glLinkProgram(uint32_t program) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t>("glLinkProgram", program);
        return;
    }

    metalsharp::GLShaderState* vertex = nullptr;
    metalsharp::GLShaderState* fragment = nullptr;
    metalsharp::GLShaderState* compute = nullptr;
    metalsharp::GLShaderState* geometry = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (!state) {
            continue;
        }
        if (state->stage == metalsharp::ShaderStage::Vertex) {
            vertex = state;
            // Preserve the shader handle for limited transform-feedback capture.
            // The tracker state itself intentionally does not own its GL name.
            // The loop's shader variable is the authoritative identifier.
        } else if (state->stage == metalsharp::ShaderStage::Pixel) {
            fragment = state;
        } else if (state->stage == metalsharp::ShaderStage::Compute) {
            compute = state;
        } else if (state->stage == metalsharp::ShaderStage::Geometry) {
            geometry = state;
        }
    }

    ExperimentalProgram result;
    result.linked = true;
    { std::lock_guard<std::mutex> lock(g_programMutex); auto old=g_programs.find(program); if(old!=g_programs.end()){result.separable=old->second.separable;result.fragDataLocations=old->second.fragDataLocations;} }
    if (compute && !vertex && !fragment) {
        if (!compute->compiled || !compute->compileSuccess) result.infoLog = "MetalSharp: compute shader did not compile";
        else if (!ensureMetalInit()) result.infoLog = "MetalSharp: Metal device initialization failed";
        else { discoverShaderInterface(result, compute->source, false); result.linkSuccess = true; }
    } else if ((!vertex || !fragment) && !result.separable) {
        result.infoLog = "MetalSharp: a vertex and fragment shader are required";
    } else if (!vertex && !fragment) {
        result.infoLog = "MetalSharp: separable program has no render stage";
    } else if ((vertex && (!vertex->compiled || !vertex->compileSuccess)) || (fragment && (!fragment->compiled || !fragment->compileSuccess)) ||
               (geometry && (!geometry->compiled || !geometry->compileSuccess))) {
        result.infoLog = "MetalSharp: attached shaders did not compile";
    } else if (!ensureMetalInit()) {
        result.infoLog = "MetalSharp: Metal device initialization failed";
    } else {
        if (vertex) discoverShaderInterface(result, vertex->source, true);
        if (geometry) discoverGeometryOutputs(result, geometry->source);
        if (fragment) discoverShaderInterface(result, fragment->source, false);
        result.linkSuccess = true;
        if (vertex && fragment) for (const auto& input : result.fragmentInputs) { std::string outputType; if(geometry){auto output=result.geometryOutputs.find(input.first);if(output!=result.geometryOutputs.end())outputType=output->second;else{auto vertexOutput=result.vertexOutputs.find(input.first);if(vertexOutput!=result.vertexOutputs.end())outputType=vertexOutput->second;}}else{auto output=result.vertexOutputs.find(input.first);if(output!=result.vertexOutputs.end())outputType=output->second;} if(!outputType.empty()&&outputType!=input.second){result.linkSuccess=false;result.infoLog="MetalSharp: vertex/fragment interface type mismatch for "+input.first;break;} }
        if (vertex && fragment) for (const auto& input : result.fragmentInputs) if(!result.vertexOutputs.count(input.first)&&!(geometry&&result.geometryOutputs.count(input.first))){result.linkSuccess=false;result.infoLog="MetalSharp: fragment input has no matching vertex output: "+input.first;break;}
    }

    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state=metalsharp::GLShaderTracker::instance().getShader(shader);
        if(state==vertex)result.vertexShader=shader;
        if(state==fragment)result.fragmentShader=shader;
    }
    { std::lock_guard<std::mutex> lock(g_programMutex); auto old=g_programs.find(program); if(old!=g_programs.end()&&result.binaryRetrievable==false)result.binaryRetrievable=old->second.binaryRetrievable; }
    if(result.linkSuccess) result.binary=buildExperimentalProgramBinary(vertex,fragment,compute);
    std::lock_guard<std::mutex> lock(g_programMutex);
    g_programs[program] = std::move(result);
}

extern "C" void glValidateProgram(uint32_t program) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t>("glValidateProgram", program);
    }
}

extern "C" void glGetProgramiv(uint32_t program, uint32_t pname, int32_t* params) {
    if (!params) {
        return;
    }
    if (isExperimentalProgram(program)) {
        std::lock_guard<std::mutex> lock(g_programMutex);
        auto it = g_programs.find(program);
        if (it == g_programs.end()) {
            *params = 0;
            return;
        }
        switch (pname) {
        case 0x8B82: // GL_LINK_STATUS
        case 0x8B83: // GL_VALIDATE_STATUS
            *params = it->second.linkSuccess ? 1 : 0;
            return;
        case 0x8B84: // GL_INFO_LOG_LENGTH
            *params = static_cast<int32_t>(it->second.infoLog.size()) + 1;
            return;
        case 0x8B86: // GL_ACTIVE_UNIFORMS
            *params = static_cast<int32_t>(it->second.uniformLocations.size());
            return;
        case 0x8B89: // GL_ACTIVE_ATTRIBUTES
            *params = static_cast<int32_t>(it->second.attributeLocations.size());
            return;
        default:
            *params = 0;
            return;
        }
    }
    glDispatch<void, uint32_t, uint32_t, int32_t*>("glGetProgramiv", program, pname, params);
}

extern "C" void glGetProgramInfoLog(uint32_t program, int32_t bufSize, int32_t* length, char* infoLog) {
    if (isExperimentalProgram(program)) {
        std::lock_guard<std::mutex> lock(g_programMutex);
        auto it = g_programs.find(program);
        if (it == g_programs.end() || !infoLog || bufSize <= 0) {
            if (length) {
                *length = 0;
            }
            return;
        }
        const int32_t count =
            std::min<int32_t>(bufSize - 1, static_cast<int32_t>(it->second.infoLog.size()));
        if (count > 0) {
            std::memcpy(infoLog, it->second.infoLog.data(), static_cast<size_t>(count));
        }
        infoLog[count] = '\0';
        if (length) {
            *length = count;
        }
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t*, char*>("glGetProgramInfoLog", program, bufSize, length, infoLog);
}

extern "C" unsigned char glIsProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        return 1;
    }
    return glDispatch<unsigned char, uint32_t>("glIsProgram", program);
}

// glUseProgram is hand-written because it must mirror the active program
// into GLState::currentProgram so subsequent draw calls can observe which
// program is bound. The native call is still issued so the framework
// context state stays in sync with the shim's view.
extern "C" void glUseProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        g_glBridge.state().currentProgram = program;
        return;
    }
    if (!program && isExperimentalProgram(g_glBridge.state().currentProgram)) {
        g_glBridge.state().currentProgram = 0;
        return;
    }
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glUseProgram"));
    if (fn) {
        fn(program);
    }
    g_glBridge.state().currentProgram = program;
}

extern "C" void glGenProgramPipelines(int32_t n, uint32_t* pipelines) { glDispatch<void,int32_t,uint32_t*>("glGenProgramPipelines",n,pipelines); if(metalModeEnabled()&&pipelines){std::lock_guard<std::mutex> lock(g_pipelineMutex);for(int32_t i=0;i<n;++i){g_pipelines.emplace(pipelines[i],ExperimentalPipeline{});g_pipelineProxies[0xF0000000u|pipelines[i]]=pipelines[i];}} }
extern "C" void glDeleteProgramPipelines(int32_t n, const uint32_t* pipelines) { if(pipelines){std::lock_guard<std::mutex> lock(g_pipelineMutex);for(int32_t i=0;i<n;++i){g_pipelines.erase(pipelines[i]);g_pipelineProxies.erase(0xF0000000u|pipelines[i]);}} glDispatch<void,int32_t,const uint32_t*>("glDeleteProgramPipelines",n,pipelines); }
extern "C" void glBindProgramPipeline(uint32_t pipeline) { glDispatch<void,uint32_t>("glBindProgramPipeline",pipeline); g_glBridge.state().currentProgramPipeline=pipeline; }
extern "C" unsigned char glIsProgramPipeline(uint32_t pipeline) { if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_pipelineMutex);return g_pipelines.count(pipeline)!=0;}return glDispatch<unsigned char,uint32_t>("glIsProgramPipeline",pipeline); }
extern "C" void glUseProgramStages(uint32_t pipeline, uint32_t stages, uint32_t program) { if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_pipelineMutex);auto& p=g_pipelines[pipeline];if(stages==0xFFFFFFFFu){p.vertexProgram=program;p.fragmentProgram=program;}else{if(stages&0x00000001)p.vertexProgram=program;if(stages&0x00000002)p.fragmentProgram=program;}return;} glDispatch<void,uint32_t,uint32_t,uint32_t>("glUseProgramStages",pipeline,stages,program); }
extern "C" void glValidateProgramPipeline(uint32_t pipeline) { if(!metalModeEnabled())glDispatch<void,uint32_t>("glValidateProgramPipeline",pipeline); }
extern "C" void glGetProgramPipelineiv(uint32_t pipeline, uint32_t pname, int32_t* params) { if(!params)return;if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_pipelineMutex);auto it=g_pipelines.find(pipeline);if(it==g_pipelines.end()){*params=0;return;}if(pname==0x8B82)*params=(it->second.vertexProgram&&it->second.fragmentProgram)?1:0;else *params=0;return;}glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetProgramPipelineiv",pipeline,pname,params); }

// ---------------------------------------------------------------------------
// Uniforms (GL 2.0)
// ---------------------------------------------------------------------------
//
// Phase 2g: glGetUniformLocation and the glUniform* family are currently
// pass-through to the native OpenGL framework. Phase 3 will intercept them
// so that:
//   * glGetUniformLocation returns a Metal buffer-binding index that maps
//     back to the guest's location (a 1:1 mapping is a reasonable Phase 3
//     default since Metal buffer bindings also use small integers).
//   * glUniform* writes are captured by GLShaderTracker and replayed against
//     the Metal argument buffer at glDrawArrays / glDrawElements time.
//
// For Phase 2 we just forward to native GL so that GLSL 1.20 / legacy
// shaders (which never reach the cross-compile path) keep working without
// any uniform-mapping shim.

extern "C" int32_t glGetUniformLocation(uint32_t program, const char* name) {
    if (!isExperimentalProgram(program))
        return glDispatch<int32_t, uint32_t, const char*>("glGetUniformLocation", program, name);
    if (!name || !*name) return -1;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return -1;
    std::string lookup=name;int32_t element=0;size_t bracket=lookup.find('[');if(bracket!=std::string::npos){size_t end=lookup.find(']',bracket);if(end==lookup.size()-1){element=std::max(0,std::atoi(lookup.substr(bracket+1,end-bracket-1).c_str()));lookup.resize(bracket);}}
    auto found = it->second.uniformLocations.find(lookup);
    if (found != it->second.uniformLocations.end()) return found->second + element;
    const int32_t location = static_cast<int32_t>(it->second.uniformLocations.size());
    it->second.uniformLocations.emplace(lookup, location);
    return location + element;
}

template <typename T>
void setExperimentalUniform(int32_t location, const T* values, size_t count) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program) || location < 0 || !values) return;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return;
    it->second.uniformValues[location] = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(values),
                                                              reinterpret_cast<const uint8_t*>(values) + count);
}
template <typename T>
void setExperimentalProgramUniform(uint32_t program, int32_t location, const T* values, size_t count) {
    if (!isExperimentalProgram(program) || location < 0 || !values) return;
    std::lock_guard<std::mutex> lock(g_programMutex); auto it=g_programs.find(program); if(it==g_programs.end())return;
    it->second.uniformValues[location]=std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(values),reinterpret_cast<const uint8_t*>(values)+count);
}

extern "C" void glUniform1f(int32_t location, float v0) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, 0, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float>("glUniform1f", location, v0); }
extern "C" void glUniform2f(int32_t location, float v0, float v1) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float>("glUniform2f", location, v0, v1); }
extern "C" void glUniform3f(int32_t location, float v0, float v1, float v2) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, v2, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float, float>("glUniform3f", location, v0, v1, v2); }
extern "C" void glUniform4f(int32_t location, float v0, float v1, float v2, float v3) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, v2, v3}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float, float, float>("glUniform4f", location, v0, v1, v2, v3); }
extern "C" void glUniform1i(int32_t location, int32_t v0) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, 0, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t>("glUniform1i", location, v0); }
extern "C" void glUniform2i(int32_t location, int32_t v0, int32_t v1) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t>("glUniform2i", location, v0, v1); }
extern "C" void glUniform3i(int32_t location, int32_t v0, int32_t v1, int32_t v2) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, v2, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t, int32_t>("glUniform3i", location, v0, v1, v2); }
extern "C" void glUniform4i(int32_t location, int32_t v0, int32_t v1, int32_t v2, int32_t v3) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, v2, v3}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t, int32_t, int32_t>("glUniform4i", location, v0, v1, v2, v3); }
extern "C" void glUniform1ui(int32_t location, uint32_t v0) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) setExperimentalUniform(location,&v0,sizeof(v0)); else glDispatch<void,int32_t,uint32_t>("glUniform1ui",location,v0); }
extern "C" void glUniform2ui(int32_t location, uint32_t v0, uint32_t v1) { uint32_t v[4]={v0,v1,0,0}; if (isExperimentalProgram(g_glBridge.state().currentProgram)) setExperimentalUniform(location,v,sizeof(v)); else glDispatch<void,int32_t,uint32_t,uint32_t>("glUniform2ui",location,v0,v1); }
extern "C" void glUniform3ui(int32_t location, uint32_t v0, uint32_t v1, uint32_t v2) { uint32_t v[4]={v0,v1,v2,0}; if (isExperimentalProgram(g_glBridge.state().currentProgram)) setExperimentalUniform(location,v,sizeof(v)); else glDispatch<void,int32_t,uint32_t,uint32_t,uint32_t>("glUniform3ui",location,v0,v1,v2); }
extern "C" void glUniform4ui(int32_t location, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) { uint32_t v[4]={v0,v1,v2,v3}; if (isExperimentalProgram(g_glBridge.state().currentProgram)) setExperimentalUniform(location,v,sizeof(v)); else glDispatch<void,int32_t,uint32_t,uint32_t,uint32_t,uint32_t>("glUniform4ui",location,v0,v1,v2,v3); }
#define WINEMETALGL_PROGRAM_UNIFORM_SCALAR(name,type) \
extern "C" void name(uint32_t program,int32_t location,type v0) { if(isExperimentalProgram(program))setExperimentalProgramUniform(program,location,&v0,sizeof(v0)); else glDispatch<void,uint32_t,int32_t,type>(#name,program,location,v0); }
WINEMETALGL_PROGRAM_UNIFORM_SCALAR(glProgramUniform1f,float)
WINEMETALGL_PROGRAM_UNIFORM_SCALAR(glProgramUniform1i,int32_t)
WINEMETALGL_PROGRAM_UNIFORM_SCALAR(glProgramUniform1ui,uint32_t)
#undef WINEMETALGL_PROGRAM_UNIFORM_SCALAR
extern "C" void glProgramUniform2f(uint32_t p,int32_t l,float a,float b){float v[4]={a,b,0,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,float,float>("glProgramUniform2f",p,l,a,b);}
extern "C" void glProgramUniform3f(uint32_t p,int32_t l,float a,float b,float c){float v[4]={a,b,c,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,float,float,float>("glProgramUniform3f",p,l,a,b,c);}
extern "C" void glProgramUniform4f(uint32_t p,int32_t l,float a,float b,float c,float d){float v[4]={a,b,c,d};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,float,float,float,float>("glProgramUniform4f",p,l,a,b,c,d);}
extern "C" void glProgramUniform2i(uint32_t p,int32_t l,int32_t a,int32_t b){int32_t v[4]={a,b,0,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,int32_t,int32_t>("glProgramUniform2i",p,l,a,b);}
extern "C" void glProgramUniform3i(uint32_t p,int32_t l,int32_t a,int32_t b,int32_t c){int32_t v[4]={a,b,c,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t>("glProgramUniform3i",p,l,a,b,c);}
extern "C" void glProgramUniform4i(uint32_t p,int32_t l,int32_t a,int32_t b,int32_t c,int32_t d){int32_t v[4]={a,b,c,d};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t>("glProgramUniform4i",p,l,a,b,c,d);}
extern "C" void glProgramUniform2ui(uint32_t p,int32_t l,uint32_t a,uint32_t b){uint32_t v[4]={a,b,0,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t>("glProgramUniform2ui",p,l,a,b);}
extern "C" void glProgramUniform3ui(uint32_t p,int32_t l,uint32_t a,uint32_t b,uint32_t c){uint32_t v[4]={a,b,c,0};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,uint32_t>("glProgramUniform3ui",p,l,a,b,c);}
extern "C" void glProgramUniform4ui(uint32_t p,int32_t l,uint32_t a,uint32_t b,uint32_t c,uint32_t d){uint32_t v[4]={a,b,c,d};if(isExperimentalProgram(p))setExperimentalProgramUniform(p,l,v,sizeof(v));else glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,uint32_t,uint32_t>("glProgramUniform4ui",p,l,a,b,c,d);}
#define WINEMETALGL_PROGRAM_UNIFORM_ARRAY(name,type) \
extern "C" void name(uint32_t program,int32_t location,int32_t count,const type* value) { if(isExperimentalProgram(program)){if(count>0&&value)setExperimentalProgramUniform(program,location,value,sizeof(type)*static_cast<size_t>(count));}else glDispatch<void,uint32_t,int32_t,int32_t,const type*>(#name,program,location,count,value); }
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform1fv,float)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform2fv,float)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform3fv,float)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform4fv,float)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform1iv,int32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform2iv,int32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform3iv,int32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform4iv,int32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform1uiv,uint32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform2uiv,uint32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform3uiv,uint32_t)
WINEMETALGL_PROGRAM_UNIFORM_ARRAY(glProgramUniform4uiv,uint32_t)
#undef WINEMETALGL_PROGRAM_UNIFORM_ARRAY
#define WINEMETALGL_PROGRAM_UNIFORM_MATRIX(name,components) \
extern "C" void name(uint32_t program,int32_t location,int32_t count,unsigned char transpose,const float* value) { if(isExperimentalProgram(program)){if(count>0&&value)setExperimentalProgramUniform(program,location,value,sizeof(float)*components*static_cast<size_t>(count));}else glDispatch<void,uint32_t,int32_t,int32_t,unsigned char,const float*>(#name,program,location,count,transpose,value); }
WINEMETALGL_PROGRAM_UNIFORM_MATRIX(glProgramUniformMatrix2fv,4)
WINEMETALGL_PROGRAM_UNIFORM_MATRIX(glProgramUniformMatrix3fv,9)
WINEMETALGL_PROGRAM_UNIFORM_MATRIX(glProgramUniformMatrix4fv,16)
#undef WINEMETALGL_PROGRAM_UNIFORM_MATRIX

#define WINEMETALGL_UNIFORM_ARRAY(name, scalar) \
extern "C" void name(int32_t location, int32_t count, const scalar* values) { \
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { \
        if (count > 0 && values) setExperimentalUniform(location, values, sizeof(scalar) * static_cast<size_t>(count)); \
    } else glDispatch<void, int32_t, int32_t, const scalar*>(#name, location, count, values); \
}
WINEMETALGL_UNIFORM_ARRAY(glUniform1fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform2fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform3fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform4fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform1iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform2iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform3iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform4iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform1uiv, uint32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform2uiv, uint32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform3uiv, uint32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform4uiv, uint32_t)
#undef WINEMETALGL_UNIFORM_ARRAY
#define WINEMETALGL_UNIFORM_MATRIX_ARRAY(name, components) \
extern "C" void name(int32_t location, int32_t count, unsigned char transpose, const float* value) { \
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if(count>0&&value)setExperimentalUniform(location,value,sizeof(float)*components*static_cast<size_t>(count)); } \
    else glDispatch<void,int32_t,int32_t,unsigned char,const float*>(#name,location,count,transpose,value); \
}
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix2x3fv,6)
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix2x4fv,8)
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix3x2fv,6)
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix3x4fv,12)
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix4x2fv,8)
WINEMETALGL_UNIFORM_MATRIX_ARRAY(glUniformMatrix4x3fv,12)
#undef WINEMETALGL_UNIFORM_MATRIX_ARRAY

extern "C" void glUniformMatrix2fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 4 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix2fv", location, count, transpose, value);
}
extern "C" void glUniformMatrix3fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 9 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix3fv", location, count, transpose, value);
}
extern "C" void glUniformMatrix4fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 16 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix4fv", location, count, transpose, value);
}
extern "C" void glGetUniformfv(uint32_t program, int32_t location, float* params) {
    if (isExperimentalProgram(program)) { std::lock_guard<std::mutex> lock(g_programMutex); auto it=g_programs.find(program); if(it!=g_programs.end()){auto v=it->second.uniformValues.find(location);if(v!=it->second.uniformValues.end()&&params)std::memcpy(params,v->second.data(),v->second.size());} return; }
    glDispatch<void,uint32_t,int32_t,float*>("glGetUniformfv",program,location,params);
}
extern "C" void glGetUniformiv(uint32_t program, int32_t location, int32_t* params) {
    if (isExperimentalProgram(program)) { std::lock_guard<std::mutex> lock(g_programMutex); auto it=g_programs.find(program); if(it!=g_programs.end()){auto v=it->second.uniformValues.find(location);if(v!=it->second.uniformValues.end()&&params)std::memcpy(params,v->second.data(),v->second.size());} return; }
    glDispatch<void,uint32_t,int32_t,int32_t*>("glGetUniformiv",program,location,params);
}
extern "C" uint32_t glGetUniformBlockIndex(uint32_t program, const char* name) {
    if(!isExperimentalProgram(program))return glDispatch<uint32_t,uint32_t,const char*>("glGetUniformBlockIndex",program,name); if(!name)return 0xFFFFFFFFu;std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end())return 0xFFFFFFFFu;auto block=it->second.uniformBlockIndices.find(name);return block==it->second.uniformBlockIndices.end()?0xFFFFFFFFu:block->second;
}
extern "C" void glUniformBlockBinding(uint32_t program, uint32_t uniformBlockIndex, uint32_t uniformBlockBinding) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t>("glUniformBlockBinding",program,uniformBlockIndex,uniformBlockBinding);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it!=g_programs.end()&&uniformBlockIndex<it->second.uniformBlockIndices.size())it->second.uniformBlockBindings[uniformBlockIndex]=uniformBlockBinding; }
extern "C" void glGetActiveUniformBlockiv(uint32_t program, uint32_t uniformBlockIndex, uint32_t pname, int32_t* params) { if(!params)return;if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t*>("glGetActiveUniformBlockiv",program,uniformBlockIndex,pname,params);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end()||uniformBlockIndex>=it->second.uniformBlockIndices.size()){*params=0;return;}if(pname==0x8A40)*params=16;else if(pname==0x8A3F)*params=it->second.uniformBlockBindings[uniformBlockIndex];else if(pname==0x8A41)*params=1;else *params=0; }
static std::string experimentalResourceName(const ExperimentalProgram& program,uint32_t programInterface,uint32_t index) { const std::unordered_map<std::string,uint32_t>* map=nullptr;const std::vector<std::string>* order=nullptr;if(programInterface==0x92E2)map=&program.uniformBlockIndices;else if(programInterface==0x92E6)map=&program.storageBlockIndices;else if(programInterface==0x92E1)order=&program.uniformOrder;else if(programInterface==0x92E3)order=&program.attributeOrder;if(order)return index<order->size()?(*order)[index]:std::string();if(map)for(const auto& entry:*map)if(entry.second==index)return entry.first;return {}; }
extern "C" uint32_t glGetProgramResourceIndex(uint32_t program, uint32_t programInterface, const char* name) { if(!isExperimentalProgram(program))return glDispatch<uint32_t,uint32_t,uint32_t,const char*>("glGetProgramResourceIndex",program,programInterface,name);if(!name)return 0xFFFFFFFFu;std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end())return 0xFFFFFFFFu;if(programInterface==0x92E2){auto block=it->second.uniformBlockIndices.find(name);return block==it->second.uniformBlockIndices.end()?0xFFFFFFFFu:block->second;}if(programInterface==0x92E6){auto block=it->second.storageBlockIndices.find(name);return block==it->second.storageBlockIndices.end()?0xFFFFFFFFu:block->second;}if(programInterface==0x92E1){auto found=it->second.uniformLocations.find(name);return found==it->second.uniformLocations.end()?0xFFFFFFFFu:static_cast<uint32_t>(found->second);}if(programInterface==0x92E3){auto found=it->second.attributeLocations.find(name);return found==it->second.attributeLocations.end()?0xFFFFFFFFu:static_cast<uint32_t>(found->second);}return 0xFFFFFFFFu; }
extern "C" void glGetProgramInterfaceiv(uint32_t program,uint32_t programInterface,uint32_t pname,int32_t* params) { if(!params)return;if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t*>("glGetProgramInterfaceiv",program,programInterface,pname,params);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end()){*params=0;return;}if(pname!=0x92F5&&pname!=0x92F6){*params=0;return;}size_t count=programInterface==0x92E1?it->second.uniformOrder.size():programInterface==0x92E2?it->second.uniformBlockIndices.size():programInterface==0x92E3?it->second.attributeOrder.size():programInterface==0x92E6?it->second.storageBlockIndices.size():0;if(pname==0x92F5)*params=static_cast<int32_t>(count);else{size_t maxLength=1;for(size_t i=0;i<count;++i)maxLength=std::max(maxLength,experimentalResourceName(it->second,programInterface,static_cast<uint32_t>(i)).size()+1);*params=static_cast<int32_t>(maxLength);} }
extern "C" void glGetProgramResourceName(uint32_t program,uint32_t programInterface,uint32_t index,int32_t bufSize,int32_t* length,char* name) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t,int32_t*,char*>("glGetProgramResourceName",program,programInterface,index,bufSize,length,name);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);std::string resource=it==g_programs.end()?std::string():experimentalResourceName(it->second,programInterface,index);int32_t copied=bufSize>0?std::min<int32_t>(bufSize-1,static_cast<int32_t>(resource.size())):0;if(name&&bufSize>0){if(copied)std::memcpy(name,resource.data(),copied);name[copied]=0;}if(length)*length=copied; }
extern "C" int32_t glGetProgramResourceLocation(uint32_t program,uint32_t programInterface,const char* name) { if(!isExperimentalProgram(program))return glDispatch<int32_t,uint32_t,uint32_t,const char*>("glGetProgramResourceLocation",program,programInterface,name);if(!name)return -1;std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end())return -1;if(programInterface==0x92E1){auto found=it->second.uniformLocations.find(name);return found==it->second.uniformLocations.end()?-1:found->second;}if(programInterface==0x92E3){auto found=it->second.attributeLocations.find(name);return found==it->second.attributeLocations.end()?-1:found->second;}return -1; }
extern "C" void glGetProgramResourceiv(uint32_t program,uint32_t programInterface,uint32_t index,int32_t propertyCount,const uint32_t* properties,int32_t bufSize,int32_t* length,int32_t* params) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t,const uint32_t*,int32_t,int32_t*,int32_t*>("glGetProgramResourceiv",program,programInterface,index,propertyCount,properties,bufSize,length,params);return;}if(!params||bufSize<=0||propertyCount<=0)return;std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end())return;std::string name=experimentalResourceName(it->second,programInterface,index);int32_t written=0;for(int32_t i=0;i<propertyCount&&written<bufSize;++i){uint32_t property=properties?properties[i]:0;int32_t value=0;if(property==0x92F9)value=static_cast<int32_t>(name.size()+1);else if(property==0x92FA&&programInterface==0x92E1){auto type=it->second.uniformTypes.find(name);value=type==it->second.uniformTypes.end()?0:static_cast<int32_t>(type->second);}else if(property==0x92FB)value=it->second.uniformSizes.count(name)?it->second.uniformSizes[name]:1;else if(property==0x930E){if(programInterface==0x92E1){auto found=it->second.uniformLocations.find(name);value=found==it->second.uniformLocations.end()?-1:found->second;}else if(programInterface==0x92E3){auto found=it->second.attributeLocations.find(name);value=found==it->second.attributeLocations.end()?-1:found->second;}else value=-1;}else if(property==0x9302){if(programInterface==0x92E2){auto found=it->second.uniformBlockIndices.find(name);value=found==it->second.uniformBlockIndices.end()?0:static_cast<int32_t>(it->second.uniformBlockBindings[found->second]);}else if(programInterface==0x92E6){auto found=it->second.storageBlockIndices.find(name);value=found==it->second.storageBlockIndices.end()?0:static_cast<int32_t>(it->second.storageBlockBindings[found->second]);}}params[written++]=value;}if(length)*length=written; }
extern "C" void glShaderStorageBlockBinding(uint32_t program, uint32_t storageBlockIndex, uint32_t storageBlockBinding) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,uint32_t>("glShaderStorageBlockBinding",program,storageBlockIndex,storageBlockBinding);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it!=g_programs.end()&&storageBlockIndex<it->second.storageBlockIndices.size())it->second.storageBlockBindings[storageBlockIndex]=storageBlockBinding; }
extern "C" void glGetActiveUniformBlockName(uint32_t program, uint32_t uniformBlockIndex, int32_t bufSize, int32_t* length, char* name) { if(!isExperimentalProgram(program)){glDispatch<void,uint32_t,uint32_t,int32_t,int32_t*,char*>("glGetActiveUniformBlockName",program,uniformBlockIndex,bufSize,length,name);return;}std::lock_guard<std::mutex> lock(g_programMutex);auto it=g_programs.find(program);if(it==g_programs.end()||uniformBlockIndex>=it->second.uniformBlockIndices.size()){if(length)*length=0;if(name&&bufSize>0)*name=0;return;}std::string block;for(const auto& entry:it->second.uniformBlockIndices)if(entry.second==uniformBlockIndex){block=entry.first;break;}int32_t copied=bufSize>0?std::min<int32_t>(bufSize-1,static_cast<int32_t>(block.size())):0;if(name&&bufSize>0){std::memcpy(name,block.data(),copied);name[copied]=0;}if(length)*length=copied;}
extern "C" void glGetActiveUniform(uint32_t program, uint32_t index, int32_t bufSize, int32_t* length, int32_t* size, uint32_t* type, char* name) {
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,uint32_t,int32_t,int32_t*,int32_t*,uint32_t*,char*>("glGetActiveUniform",program,index,bufSize,length,size,type,name); return; }
    std::lock_guard<std::mutex> lock(g_programMutex); auto it=g_programs.find(program); if(it==g_programs.end()||index>=it->second.uniformOrder.size()){if(length)*length=0;return;} const auto& n=it->second.uniformOrder[index]; int32_t copied=bufSize>0?std::min<int32_t>(bufSize-1,n.size()):0; if(name&&bufSize>0){std::memcpy(name,n.data(),copied);name[copied]=0;} if(length)*length=copied;if(size)*size=it->second.uniformSizes.count(n)?it->second.uniformSizes[n]:1;if(type)*type=it->second.uniformTypes[n];
}

// ---------------------------------------------------------------------------
// Vertex attribute location queries (GL 2.0)
// ---------------------------------------------------------------------------
extern "C" int32_t glGetAttribLocation(uint32_t program, const char* name) {
    if (!isExperimentalProgram(program))
        return glDispatch<int32_t, uint32_t, const char*>("glGetAttribLocation", program, name);
    if (!name) return -1;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return -1;
    auto found = it->second.attributeLocations.find(name);
    return found == it->second.attributeLocations.end() ? -1 : found->second;
}
extern "C" void glBindAttribLocation(uint32_t program, uint32_t index, const char* name) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, const char*>("glBindAttribLocation", program, index, name);
        return;
    }
    if (!name) return;
    std::lock_guard<std::mutex> lock(g_programMutex);
    g_programs[program].attributeLocations[name] = static_cast<int32_t>(index);
}
extern "C" void glGetActiveAttrib(uint32_t program, uint32_t index, int32_t bufSize, int32_t* length, int32_t* size, uint32_t* type, char* name) {
    if (!isExperimentalProgram(program)) { glDispatch<void,uint32_t,uint32_t,int32_t,int32_t*,int32_t*,uint32_t*,char*>("glGetActiveAttrib",program,index,bufSize,length,size,type,name); return; }
    std::lock_guard<std::mutex> lock(g_programMutex); auto it=g_programs.find(program); if(it==g_programs.end()||index>=it->second.attributeOrder.size()){if(length)*length=0;return;} const auto& n=it->second.attributeOrder[index]; int32_t copied=bufSize>0?std::min<int32_t>(bufSize-1,n.size()):0; if(name&&bufSize>0){std::memcpy(name,n.data(),copied);name[copied]=0;} if(length)*length=copied;if(size)*size=1;if(type)*type=it->second.attributeTypes[n];
}

// ---------------------------------------------------------------------------
// Rasterization state (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glCullFace(uint32_t mode) { glDispatch<void,uint32_t>("glCullFace",mode); g_glBridge.state().cullFace=mode; }
extern "C" void glFrontFace(uint32_t mode) { glDispatch<void,uint32_t>("glFrontFace",mode); g_glBridge.state().frontFace=mode; }
extern "C" void glLineWidth(float width) { glDispatch<void,float>("glLineWidth",width);if(metalModeEnabled()&&width>0)g_glBridge.state().lineWidth=width; }
extern "C" void glPointSize(float size) { glDispatch<void,float>("glPointSize",size);if(metalModeEnabled()&&size>0)g_glBridge.state().pointSize=size; }
GL_PASSTHROUGH2(void, glPolygonMode, uint32_t, face, uint32_t, mode)
extern "C" void glPolygonOffset(float factor,float units) { glDispatch<void,float,float>("glPolygonOffset",factor,units);g_glBridge.state().polygonOffsetFactor=factor;g_glBridge.state().polygonOffsetUnits=units; }
extern "C" void glClipControl(uint32_t origin,uint32_t depth) { if(origin!=0x8CA1&&origin!=0x8CA2){metalsharp::GLErrorTracker::instance().setError(0x0500);return;}if(depth!=0x935E&&depth!=0x935F){metalsharp::GLErrorTracker::instance().setError(0x0500);return;}if(metalModeEnabled()){g_glBridge.state().clipOrigin=origin;g_glBridge.state().clipDepthMode=depth;return;}glDispatch<void,uint32_t,uint32_t>("glClipControl",origin,depth); }

// ---------------------------------------------------------------------------
// Stencil state (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glStencilFunc(uint32_t func, int32_t ref, uint32_t mask) { glDispatch<void,uint32_t,int32_t,uint32_t>("glStencilFunc",func,ref,mask); auto& s=g_glBridge.state();s.stencilFunc=s.stencilFuncBack=func;s.stencilRef=s.stencilRefBack=ref;s.stencilValueMask=s.stencilValueMaskBack=mask; }
extern "C" void glStencilFuncSeparate(uint32_t face,uint32_t func,int32_t ref,uint32_t mask) { glDispatch<void,uint32_t,uint32_t,int32_t,uint32_t>("glStencilFuncSeparate",face,func,ref,mask); auto& s=g_glBridge.state();if(face==0x0404||face==0x0408){s.stencilFunc=func;s.stencilRef=ref;s.stencilValueMask=mask;}if(face==0x0405||face==0x0408){s.stencilFuncBack=func;s.stencilRefBack=ref;s.stencilValueMaskBack=mask;} }
extern "C" void glStencilOp(uint32_t sfail,uint32_t dpfail,uint32_t dppass) { glDispatch<void,uint32_t,uint32_t,uint32_t>("glStencilOp",sfail,dpfail,dppass); auto& s=g_glBridge.state();s.stencilFail=s.stencilFailBack=sfail;s.stencilDepthFail=s.stencilDepthFailBack=dpfail;s.stencilDepthPass=s.stencilDepthPassBack=dppass; }
extern "C" void glStencilOpSeparate(uint32_t face,uint32_t sfail,uint32_t dpfail,uint32_t dppass) { glDispatch<void,uint32_t,uint32_t,uint32_t,uint32_t>("glStencilOpSeparate",face,sfail,dpfail,dppass); auto& s=g_glBridge.state();if(face==0x0404||face==0x0408){s.stencilFail=sfail;s.stencilDepthFail=dpfail;s.stencilDepthPass=dppass;}if(face==0x0405||face==0x0408){s.stencilFailBack=sfail;s.stencilDepthFailBack=dpfail;s.stencilDepthPassBack=dppass;} }
extern "C" void glStencilMask(uint32_t mask) { glDispatch<void,uint32_t>("glStencilMask",mask); g_glBridge.state().stencilWriteMask=g_glBridge.state().stencilWriteMaskBack=mask; }
extern "C" void glStencilMaskSeparate(uint32_t face,uint32_t mask) { glDispatch<void,uint32_t,uint32_t>("glStencilMaskSeparate",face,mask); auto& s=g_glBridge.state();if(face==0x0404||face==0x0408)s.stencilWriteMask=mask;if(face==0x0405||face==0x0408)s.stencilWriteMaskBack=mask; }
extern "C" void glClearStencil(int32_t s) { glDispatch<void,int32_t>("glClearStencil",s); g_glBridge.state().clearStencil=s; }

// ---------------------------------------------------------------------------
// Color / blend state (GL 1.0-1.4)
// ---------------------------------------------------------------------------
extern "C" void glColorMask(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    glDispatch<void,unsigned char,unsigned char,unsigned char,unsigned char>("glColorMask",r,g,b,a);
    g_glBridge.state().colorMask[0]=r!=0; g_glBridge.state().colorMask[1]=g!=0; g_glBridge.state().colorMask[2]=b!=0; g_glBridge.state().colorMask[3]=a!=0;
}
extern "C" void glBlendEquation(uint32_t mode) { glDispatch<void,uint32_t>("glBlendEquation",mode); g_glBridge.state().blendEquationRGB=g_glBridge.state().blendEquationAlpha=mode; }
extern "C" void glBlendEquationSeparate(uint32_t modeRGB,uint32_t modeAlpha) { glDispatch<void,uint32_t,uint32_t>("glBlendEquationSeparate",modeRGB,modeAlpha); g_glBridge.state().blendEquationRGB=modeRGB;g_glBridge.state().blendEquationAlpha=modeAlpha; }
extern "C" void glBlendFuncSeparate(uint32_t srcRGB,uint32_t dstRGB,uint32_t srcAlpha,uint32_t dstAlpha) { glDispatch<void,uint32_t,uint32_t,uint32_t,uint32_t>("glBlendFuncSeparate",srcRGB,dstRGB,srcAlpha,dstAlpha); g_glBridge.state().blendSrcRGB=srcRGB;g_glBridge.state().blendDstRGB=dstRGB;g_glBridge.state().blendSrcAlpha=srcAlpha;g_glBridge.state().blendDstAlpha=dstAlpha; }
extern "C" void glBlendColor(float r,float g,float b,float a) { glDispatch<void,float,float,float,float>("glBlendColor",r,g,b,a); g_glBridge.state().blendColor[0]=r;g_glBridge.state().blendColor[1]=g;g_glBridge.state().blendColor[2]=b;g_glBridge.state().blendColor[3]=a; }
extern "C" void glBlendFunci(uint32_t buf,uint32_t src,uint32_t dst) { if(buf==0)glBlendFunc(src,dst);else metalsharp::GLErrorTracker::instance().setError(0x0501); }
extern "C" void glBlendFuncSeparatei(uint32_t buf,uint32_t srcRGB,uint32_t dstRGB,uint32_t srcAlpha,uint32_t dstAlpha) { if(buf==0)glBlendFuncSeparate(srcRGB,dstRGB,srcAlpha,dstAlpha);else metalsharp::GLErrorTracker::instance().setError(0x0501); }
extern "C" void glBlendEquationi(uint32_t buf,uint32_t mode) { if(buf==0)glBlendEquation(mode);else metalsharp::GLErrorTracker::instance().setError(0x0501); }
extern "C" void glBlendEquationSeparatei(uint32_t buf,uint32_t modeRGB,uint32_t modeAlpha) { if(buf==0)glBlendEquationSeparate(modeRGB,modeAlpha);else metalsharp::GLErrorTracker::instance().setError(0x0501); }
extern "C" void glColorMaski(uint32_t buf,unsigned char r,unsigned char g,unsigned char b,unsigned char a) { if(buf==0)glColorMask(r,g,b,a);else metalsharp::GLErrorTracker::instance().setError(0x0501); }
extern "C" void glEnablei(uint32_t target,uint32_t index) { if(index==0&&target==0x0BE2)glEnable(target);else if(index!=0)metalsharp::GLErrorTracker::instance().setError(0x0501);else glDispatch<void,uint32_t,uint32_t>("glEnablei",target,index); }
extern "C" void glDisablei(uint32_t target,uint32_t index) { if(index==0&&target==0x0BE2)glDisable(target);else if(index!=0)metalsharp::GLErrorTracker::instance().setError(0x0501);else glDispatch<void,uint32_t,uint32_t>("glDisablei",target,index); }
GL_PASSTHROUGH1(void, glLogicOp, uint32_t, opcode)

// ---------------------------------------------------------------------------
// Depth state (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glDepthMask(unsigned char flag) {
    glDispatch<void, unsigned char>("glDepthMask", flag);
    g_glBridge.state().depthWriteEnabled = flag != 0;
}
extern "C" void glDepthRange(double nearVal, double farVal) { glDispatch<void,double,double>("glDepthRange",nearVal,farVal); g_glBridge.state().depthNear=nearVal; g_glBridge.state().depthFar=farVal; }
extern "C" void glDepthRangef(float nearVal, float farVal) { glDepthRange(nearVal,farVal); }
extern "C" void glClearDepth(double depth) {
    glDispatch<void, double>("glClearDepth", depth);
    g_glBridge.state().clearDepth = static_cast<float>(depth);
}

// ---------------------------------------------------------------------------
// Pixel storage / transfer (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glPixelStorei(uint32_t pname, int32_t param) {
    glDispatch<void,uint32_t,int32_t>("glPixelStorei", pname, param);
    if (pname == 0x0D05 && (param == 1 || param == 2 || param == 4 || param == 8)) g_glBridge.state().packAlignment = param;
    else if (pname == 0x0CF5 && (param == 1 || param == 2 || param == 4 || param == 8)) g_glBridge.state().unpackAlignment = param;
}
GL_PASSTHROUGH2(void, glPixelStoref, uint32_t, pname, float, param)
extern "C" void glReadBuffer(uint32_t mode) { glDispatch<void,uint32_t>("glReadBuffer",mode); g_glBridge.state().readBuffer=mode; }
extern "C" void glDrawBuffer(uint32_t mode) { glDispatch<void,uint32_t>("glDrawBuffer",mode); g_glBridge.state().drawBuffer=mode; }
extern "C" void glDrawBuffers(int32_t n, const uint32_t* buffers) { glDispatch<void,int32_t,const uint32_t*>("glDrawBuffers",n,buffers); if(n>0&&buffers)g_glBridge.state().drawBuffer=buffers[0]; }

// ---------------------------------------------------------------------------
// State queries (GL 1.0-1.1)
// ---------------------------------------------------------------------------
static bool experimentalGetFloatValues(uint32_t pname,float* params) {
    if(!params||!metalModeEnabled())return false; auto& state=g_glBridge.state();
    switch(pname){
    case 0x0BA2: params[0]=static_cast<float>(state.viewportX);params[1]=static_cast<float>(state.viewportY);params[2]=static_cast<float>(state.viewportWidth);params[3]=static_cast<float>(state.viewportHeight);return true;
    case 0x0C10: params[0]=static_cast<float>(state.scissorX);params[1]=static_cast<float>(state.scissorY);params[2]=static_cast<float>(state.scissorWidth);params[3]=static_cast<float>(state.scissorHeight);return true;
    case 0x0B70: params[0]=static_cast<float>(state.depthNear);params[1]=static_cast<float>(state.depthFar);return true;
    case 0x0C22: std::memcpy(params,state.clearColor,sizeof(state.clearColor));return true;
    case 0x8005: std::memcpy(params,state.blendColor,sizeof(state.blendColor));return true;
    case 0x8038: params[0]=state.polygonOffsetFactor;return true;
    case 0x2A00: params[0]=state.polygonOffsetUnits;return true;
    case 0x0BE2: params[0]=state.blendEnabled;return true;
    case 0x0B44: params[0]=state.cullEnabled;return true;
    case 0x0B71: params[0]=state.depthTestEnabled;return true;
    case 0x0C11: params[0]=state.scissorEnabled;return true;
    case 0x864F: params[0]=state.depthClampEnabled;return true;
    case 0x0B00: std::memcpy(params,g_fixedColor,sizeof(g_fixedColor));return true;
    case 0x0B03: params[0]=g_fixedTexcoord[0];params[1]=g_fixedTexcoord[1];params[2]=0;params[3]=1;return true;
    case 0x0BA6: std::memcpy(params,g_fixedModelview,sizeof(g_fixedModelview));return true;
    case 0x0BA7: std::memcpy(params,g_fixedProjection,sizeof(g_fixedProjection));return true;
    default:return false;
    }
}
extern "C" void glGetFloatv(uint32_t pname,float* params) { if(experimentalGetFloatValues(pname,params))return;glDispatch<void,uint32_t,float*>("glGetFloatv",pname,params); }
extern "C" void glGetDoublev(uint32_t pname,double* params) { if(params&&metalModeEnabled()){float values[16];if(experimentalGetFloatValues(pname,values)){int count=(pname==0x0BA6||pname==0x0BA7)?16:(pname==0x0BA2||pname==0x0C10)?4:(pname==0x0B70)?2:(pname==0x0C22||pname==0x8005||pname==0x0B00)?4:1;for(int i=0;i<count;++i)params[i]=values[i];return;}}glDispatch<void,uint32_t,double*>("glGetDoublev",pname,params); }
extern "C" void glGetBooleanv(uint32_t pname,unsigned char* params) { if(params&&metalModeEnabled()){float values[16];if(experimentalGetFloatValues(pname,values)){int count=(pname==0x0BA6||pname==0x0BA7)?16:(pname==0x0BA2||pname==0x0C10)?4:(pname==0x0B70)?2:(pname==0x0C22||pname==0x8005||pname==0x0B00)?4:1;for(int i=0;i<count;++i)params[i]=values[i]!=0;return;}}glDispatch<void,uint32_t,unsigned char*>("glGetBooleanv",pname,params); }
GL_PASSTHROUGH3(void, glGetTexEnviv, uint32_t, target, uint32_t, pname, int32_t*, params)
GL_PASSTHROUGH3(void, glGetTexEnvfv, uint32_t, target, uint32_t, pname, float*, params)
// glGetError is hand-written: checks the shim error tracker first
// (Phase 5a), then falls through to native GL.
extern "C" uint32_t glGetError() {
    uint32_t err = metalsharp::GLErrorTracker::instance().getError();
    if (err != 0)
        return err;
    // Note: we do NOT forward to native GL's glGetError here because
    // dlsym on the framework handle may resolve to our own shim symbol
    // on macOS, creating infinite recursion. The native GL error state
    // is irrelevant when we're managing our own context via Metal.
    return 0;
}

// Phase 5b — intercept glGetString for GL_EXTENSIONS.
// The old passthrough at the end of the file is kept for non-extension queries.
static const char* const g_experimentalExtensions[] = {
    "GL_ARB_vertex_buffer_object", "GL_ARB_framebuffer_object", "GL_EXT_framebuffer_object",
    "GL_ARB_shader_objects", "GL_ARB_vertex_shader", "GL_ARB_fragment_shader",
    "GL_ARB_multitexture", "METALSHARP_opengl_bridge"
};
static constexpr size_t g_experimentalExtensionCount = sizeof(g_experimentalExtensions) / sizeof(g_experimentalExtensions[0]);
extern "C" const uint8_t* glGetString_EXTENSIONS_override(uint32_t name) {
    if (name != 0x1F03) return nullptr;
    static const char kExts[] = "GL_ARB_vertex_buffer_object GL_ARB_framebuffer_object GL_EXT_framebuffer_object GL_ARB_shader_objects GL_ARB_vertex_shader GL_ARB_fragment_shader GL_ARB_multitexture METALSHARP_opengl_bridge";
    return reinterpret_cast<const uint8_t*>(kExts);
}

GL_PASSTHROUGH1(unsigned char, glIsEnabled, uint32_t, cap)

// glGetStringi is hand-written following the glGetString pattern.
extern "C" const uint8_t* glGetStringi(uint32_t name, uint32_t index) {
    if (name == 0x1F03 && index < g_experimentalExtensionCount) return reinterpret_cast<const uint8_t*>(g_experimentalExtensions[index]);
    ensureGLInit(); auto fn = reinterpret_cast<const uint8_t* (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glGetStringi"));
    return fn ? fn(name,index) : reinterpret_cast<const uint8_t*>("");
}

// ---------------------------------------------------------------------------
// Misc commands (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glFlush(void) {
    if (metalModeEnabled()) { g_metalRenderer.flush(); return; }
    glDispatch<void>("glFlush");
}
extern "C" void glFinish(void) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void>("glFinish");
}
extern "C" void* glFenceSync(uint32_t condition, uint32_t flags) {
    if (!metalModeEnabled()) return glDispatch<void*, uint32_t, uint32_t>("glFenceSync", condition, flags);
    void* sync = new uint64_t(1);
    std::lock_guard<std::mutex> lock(g_syncMutex); g_syncs.insert(sync); return sync;
}
extern "C" uint32_t glClientWaitSync(void* sync, uint32_t flags, uint64_t timeout) {
    if (!metalModeEnabled()) return glDispatch<uint32_t, void*, uint32_t, uint64_t>("glClientWaitSync", sync, flags, timeout);
    std::lock_guard<std::mutex> lock(g_syncMutex); if (!g_syncs.count(sync)) return 0x0501;
    g_metalRenderer.finish(); return 0x911C; /* GL_CONDITION_SATISFIED */
}
extern "C" void glWaitSync(void* sync, uint32_t flags, uint64_t timeout) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void, void*, uint32_t, uint64_t>("glWaitSync", sync, flags, timeout);
}
extern "C" void glDeleteSync(void* sync) {
    if (metalModeEnabled()) { std::lock_guard<std::mutex> lock(g_syncMutex); if (g_syncs.erase(sync)) delete static_cast<uint64_t*>(sync); return; }
    glDispatch<void, void*>("glDeleteSync", sync);
}
extern "C" unsigned char glIsSync(void* sync) {
    if (metalModeEnabled()) { std::lock_guard<std::mutex> lock(g_syncMutex); return g_syncs.count(sync) != 0; }
    return glDispatch<unsigned char, void*>("glIsSync", sync);
}
GL_PASSTHROUGH2(void, glHint, uint32_t, target, uint32_t, mode)

extern "C" void glGenQueries(int32_t n, uint32_t* ids) {
    if (!metalModeEnabled()) { glDispatch<void,int32_t,uint32_t*>("glGenQueries",n,ids); return; }
    if (!ids) return; std::lock_guard<std::mutex> lock(g_queryMutex); for (int32_t i=0;i<n;++i) { ids[i]=g_nextQuery++; g_queries.emplace(ids[i],ExperimentalQuery{}); }
}
extern "C" void glDeleteQueries(int32_t n, const uint32_t* ids) {
    if (!metalModeEnabled()) { glDispatch<void,int32_t,const uint32_t*>("glDeleteQueries",n,ids); return; }
    if (!ids) return; std::lock_guard<std::mutex> lock(g_queryMutex); for (int32_t i=0;i<n;++i) g_queries.erase(ids[i]);
}
extern "C" unsigned char glIsQuery(uint32_t id) {
    if (!metalModeEnabled()) return glDispatch<unsigned char,uint32_t>("glIsQuery",id);
    std::lock_guard<std::mutex> lock(g_queryMutex); return g_queries.count(id) != 0;
}
extern "C" void glBeginQuery(uint32_t target, uint32_t id) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t,uint32_t>("glBeginQuery",target,id); return; }
    std::lock_guard<std::mutex> lock(g_queryMutex); auto it=g_queries.find(id); if(it==g_queries.end()||g_activeQuery){metalsharp::GLErrorTracker::instance().setError(0x0502);return;} it->second.target=target;it->second.active=true;it->second.value=0;g_activeQuery=id;
}
extern "C" void glEndQuery(uint32_t target) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t>("glEndQuery",target); return; }
    std::lock_guard<std::mutex> lock(g_queryMutex); if(!g_activeQuery){metalsharp::GLErrorTracker::instance().setError(0x0502);return;} auto it=g_queries.find(g_activeQuery); if(it!=g_queries.end()){it->second.active=false;it->second.value=target==0x8914?1:static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());} g_activeQuery=0;
}
extern "C" void glGetQueryObjectuiv(uint32_t id, uint32_t pname, uint32_t* params) {
    if (!params) return; if (!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint32_t*>("glGetQueryObjectuiv",id,pname,params);return;} std::lock_guard<std::mutex> lock(g_queryMutex); auto it=g_queries.find(id); *params=it==g_queries.end()?0:static_cast<uint32_t>(it->second.value);
}
extern "C" void glGetQueryObjectui64v(uint32_t id, uint32_t pname, uint64_t* params) {
    if (!params) return; if (!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint64_t*>("glGetQueryObjectui64v",id,pname,params);return;} std::lock_guard<std::mutex> lock(g_queryMutex); auto it=g_queries.find(id); *params=it==g_queries.end()?0:it->second.value;
}
static bool queryResult(uint32_t id,uint64_t& value){std::lock_guard<std::mutex> lock(g_queryMutex);auto it=g_queries.find(id);if(it==g_queries.end())return false;value=it->second.value;return true;}
extern "C" void glGetQueryBufferObjectuiv(uint32_t id,uint32_t buffer,uint32_t pname,int64_t offset) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint32_t,int64_t>("glGetQueryBufferObjectuiv",id,buffer,pname,offset);return;}uint64_t value=0;uint32_t result=0;if(queryResult(id,value))result=static_cast<uint32_t>(value);uint64_t handle=0;{std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(buffer);if(it!=g_buffers.end())handle=it->second.metalHandle;}if(!handle||offset<0||!g_metalRenderer.updateBuffer(handle,static_cast<size_t>(offset),&result,sizeof(result)))metalsharp::GLErrorTracker::instance().setError(0x0502); }
extern "C" void glGetQueryBufferObjectiv(uint32_t id,uint32_t buffer,uint32_t pname,int64_t offset) { glGetQueryBufferObjectuiv(id,buffer,pname,offset); }
extern "C" void glGetQueryBufferObjectui64v(uint32_t id,uint32_t buffer,uint32_t pname,int64_t offset) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint32_t,int64_t>("glGetQueryBufferObjectui64v",id,buffer,pname,offset);return;}uint64_t value=0;uint64_t handle=0;queryResult(id,value);{std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(buffer);if(it!=g_buffers.end())handle=it->second.metalHandle;}if(!handle||offset<0||!g_metalRenderer.updateBuffer(handle,static_cast<size_t>(offset),&value,sizeof(value)))metalsharp::GLErrorTracker::instance().setError(0x0502); }
extern "C" void glGetQueryBufferObjecti64v(uint32_t id,uint32_t buffer,uint32_t pname,int64_t offset) { glGetQueryBufferObjectui64v(id,buffer,pname,offset); }

// ---------------------------------------------------------------------------
// Display lists (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" uint32_t glGenLists(int32_t range) {
    if (!metalModeEnabled()) return glDispatch<uint32_t, int32_t>("glGenLists", range);
    if (range <= 0) return 0; uint32_t first = g_nextFixedList; g_nextFixedList += static_cast<uint32_t>(range); return first;
}
extern "C" void glNewList(uint32_t list, uint32_t mode) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t,uint32_t>("glNewList",list,mode); return; }
    g_listId=list; g_fixedLists[list].clear(); g_listCompiling=true; g_listExecute=(mode==0x1301);
}
extern "C" void glEndList(void) {
    if (!metalModeEnabled()) { glDispatch<void>("glEndList"); return; }
    g_listCompiling=false; g_listExecute=false; g_listId=0;
}
extern "C" void glCallList(uint32_t list) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t>("glCallList",list); return; }
    auto it=g_fixedLists.find(list); if(it==g_fixedLists.end()) return;
    bool wasCompiling=g_listCompiling; g_listCompiling=false;
    for(const auto& c:it->second) switch(c.kind) {
    case FixedCommandKind::Begin: glBegin(c.mode); break; case FixedCommandKind::End: glEnd(); break;
    case FixedCommandKind::Vertex: glVertex4f(c.values[0],c.values[1],c.values[2],c.values[3]); break;
    case FixedCommandKind::Color: glColor4f(c.values[0],c.values[1],c.values[2],c.values[3]); break;
    case FixedCommandKind::TexCoord: glTexCoord2f(c.values[0],c.values[1]); break;
    case FixedCommandKind::Normal: glNormal3f(c.values[0],c.values[1],c.values[2]); break;
    case FixedCommandKind::MatrixMode: glMatrixMode(static_cast<uint32_t>(c.mode)); break; case FixedCommandKind::LoadIdentity: glLoadIdentity(); break; case FixedCommandKind::PushMatrix: glPushMatrix(); break; case FixedCommandKind::PopMatrix: glPopMatrix(); break; case FixedCommandKind::Translate: glTranslatef(c.values[0],c.values[1],c.values[2]); break; case FixedCommandKind::Scale: glScalef(c.values[0],c.values[1],c.values[2]); break; case FixedCommandKind::Rotate: glRotatef(c.values[0],c.values[1],c.values[2],c.values[3]); break;
    default: break;
    }
    g_listCompiling=wasCompiling;
}
GL_PASSTHROUGH3(void, glCallLists, int32_t, n, uint32_t, type, const void*, lists)
extern "C" void glDeleteLists(uint32_t list, int32_t range) {
    if (!metalModeEnabled()) { glDispatch<void,uint32_t,int32_t>("glDeleteLists",list,range); return; }
    for(int32_t i=0;i<range;++i) g_fixedLists.erase(list+static_cast<uint32_t>(i));
}
extern "C" unsigned char glIsList(uint32_t list) { if (metalModeEnabled()) return g_fixedLists.count(list)!=0; return glDispatch<unsigned char,uint32_t>("glIsList",list); }

// ---------------------------------------------------------------------------
// Immediate-mode vertex data (GL 1.0)
// ---------------------------------------------------------------------------
static float* fixedCurrentMatrix() { return g_fixedMatrixMode == 0x1701 ? g_fixedProjection : g_fixedModelview; }
static void fixedMultiply(float* out, const float* a, const float* b) {
    float result[16] = {};
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k) result[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    std::memcpy(out, result, sizeof(result));
}
static float fixedTexGenCoordinate(uint32_t mode, const float* plane, float x, float y, float z, const float* eye, bool second) {
    if (mode == 0x2400) return plane[0]*eye[0] + plane[1]*eye[1] + plane[2]*eye[2] + plane[3]*eye[3];
    if (mode == 0x2402) {
        float ex=-eye[0], ey=-eye[1], ez=-eye[2], el=std::sqrt(ex*ex+ey*ey+ez*ez); if(el>0){ex/=el;ey/=el;ez/=el;}
        float nx=g_fixedNormal[0],ny=g_fixedNormal[1],nz=g_fixedNormal[2],nl=std::sqrt(nx*nx+ny*ny+nz*nz);if(nl>0){nx/=nl;ny/=nl;nz/=nl;}
        float dot=ex*nx+ey*ny+ez*nz,rx=2.0f*dot*nx-ex,ry=2.0f*dot*ny-ey,rz=2.0f*dot*nz-ez,m=2.0f*std::sqrt(rx*rx+ry*ry+(rz+1.0f)*(rz+1.0f));return m>0?((second?ry:rx)/m+0.5f):0.5f;
    }
    if (mode == 0x8511) return (plane[0]*g_fixedNormal[0]+plane[1]*g_fixedNormal[1]+plane[2]*g_fixedNormal[2]);
    return plane[0]*x + plane[1]*y + plane[2]*z + plane[3];
}
static void fixedVertex(float x, float y, float z, float w) {
    if (g_fixedRecording) {
        float mvp[16], input[4] = {x, y, z, w}, output[4] = {};
        fixedMultiply(mvp, g_fixedProjection, g_fixedModelview);
        for (int r = 0; r < 4; ++r) for (int k = 0; k < 4; ++k) output[r] += mvp[k * 4 + r] * input[k];
        float color[4] = {g_fixedColor[0], g_fixedColor[1], g_fixedColor[2], g_fixedColor[3]};
        if (g_fixedLighting) {
            float lit[3] = {0,0,0};
            for (int light = 0; light < 8; ++light) if (g_fixedLights[light]) {
                float lx = g_fixedLightPositions[light][0], ly = g_fixedLightPositions[light][1], lz = g_fixedLightPositions[light][2];
                if (g_fixedLightPositions[light][3] != 0.0f) { lx -= x; ly -= y; lz -= z; }
                float length = std::sqrt(lx * lx + ly * ly + lz * lz);
                if (length > 0.0f) { lx /= length; ly /= length; lz /= length; }
                float diffuse = std::max(0.0f, g_fixedNormal[0] * lx + g_fixedNormal[1] * ly + g_fixedNormal[2] * lz);
                for (int channel = 0; channel < 3; ++channel) lit[channel] += g_fixedLightAmbients[light][channel] * g_fixedMaterialAmbient[channel] + g_fixedMaterialDiffuse[channel] * g_fixedLightDiffuses[light][channel] * diffuse + g_fixedMaterialEmission[channel];
                if (g_fixedMaterialShininess > 0.0f) { float halfDot=std::max(0.0f,(g_fixedNormal[0]*(lx+0.0f)+g_fixedNormal[1]*(ly+0.0f)+g_fixedNormal[2]*(lz+1.0f))); float halfLen=std::sqrt((lx)*(lx)+(ly)*(ly)+(lz+1.0f)*(lz+1.0f)); if(halfLen>0)halfDot/=halfLen; float spec=std::pow(std::max(0.0f,halfDot),g_fixedMaterialShininess); for(int channel=0;channel<3;++channel)lit[channel]+=g_fixedMaterialSpecular[channel]*g_fixedLightSpeculars[light][channel]*spec; }
            }
            for (int channel = 0; channel < 3; ++channel) color[channel] = std::min(1.0f, lit[channel]);
            color[3] = g_fixedMaterialDiffuse[3];
        }
        if (g_fixedFogEnabled) {
            float depth=std::fabs(z), factor=1.0f;
            if (g_fixedFogMode==0x2601) factor=std::clamp((g_fixedFogEnd-depth)/(g_fixedFogEnd-g_fixedFogStart),0.0f,1.0f);
            else if (g_fixedFogMode==0x0801) factor=std::exp(-g_fixedFogDensity*depth);
            else if (g_fixedFogMode==0x0802) factor=std::exp(-std::pow(g_fixedFogDensity*depth,2.0f));
            for(int channel=0;channel<3;++channel)color[channel]=color[channel]*factor+g_fixedFogColor[channel]*(1.0f-factor);
        }
        float eye[4] = {};
        for (int r=0;r<4;++r) for (int k=0;k<4;++k) eye[r] += g_fixedModelview[k*4+r]*input[k];
        std::array<float,6> clipDistances{};
        for (int plane=0;plane<6;++plane) if(g_fixedClipEnabled[plane]) clipDistances[plane]=static_cast<float>(g_fixedClipPlanes[plane][0]*eye[0]+g_fixedClipPlanes[plane][1]*eye[1]+g_fixedClipPlanes[plane][2]*eye[2]+g_fixedClipPlanes[plane][3]*eye[3]);
        g_fixedClipDistances.push_back(clipDistances);
        float tex[2] = {g_fixedTexcoord[0],g_fixedTexcoord[1]};
        if (g_fixedTexGenS) tex[0]=fixedTexGenCoordinate(g_fixedTexGenModeS,g_fixedTexGenPlaneS,x,y,z,eye,false);
        if (g_fixedTexGenT) tex[1]=fixedTexGenCoordinate(g_fixedTexGenModeT,g_fixedTexGenPlaneT,x,y,z,eye,true);
        g_fixedVertices.insert(g_fixedVertices.end(), {output[0], output[1], output[2], color[0], color[1], color[2], color[3], tex[0], tex[1]});
    }
}
extern "C" void glVertex2f(float x, float y) { if (g_listCompiling) { recordFixed(FixedCommandKind::Vertex, 0, {x,y,0,1}); if (!g_listExecute) return; } if (g_fixedRecording) fixedVertex(x, y, 0.0f, 1.0f); else glDispatch<void, float, float>("glVertex2f", x, y); }
extern "C" void glVertex3f(float x, float y, float z) { if (g_listCompiling) { recordFixed(FixedCommandKind::Vertex, 0, {x,y,z,1}); if (!g_listExecute) return; } if (g_fixedRecording) fixedVertex(x, y, z, 1.0f); else glDispatch<void, float, float, float>("glVertex3f", x, y, z); }
extern "C" void glVertex4f(float x, float y, float z, float w) { if (g_listCompiling) { recordFixed(FixedCommandKind::Vertex, 0, {x,y,z,w}); if (!g_listExecute) return; } if (g_fixedRecording) fixedVertex(x, y, z, w); else glDispatch<void, float, float, float, float>("glVertex4f", x, y, z, w); }
extern "C" void glTexCoord1f(float s) { if (g_listCompiling) { recordFixed(FixedCommandKind::TexCoord,0,{s,0}); if (!g_listExecute) return; } if (metalModeEnabled()) { g_fixedTexcoord[0]=s; g_fixedTexcoord[1]=0; return; } glDispatch<void,float>("glTexCoord1f",s); }
extern "C" void glTexCoord2f(float s,float t) { if (g_listCompiling) { recordFixed(FixedCommandKind::TexCoord,0,{s,t}); if (!g_listExecute) return; } if (metalModeEnabled()) { g_fixedTexcoord[0]=s; g_fixedTexcoord[1]=t; return; } glDispatch<void,float,float>("glTexCoord2f",s,t); }
extern "C" void glTexCoord3f(float s,float t,float r) { if (metalModeEnabled()) { g_fixedTexcoord[0]=s; g_fixedTexcoord[1]=t; return; } glDispatch<void,float,float,float>("glTexCoord3f",s,t,r); }
extern "C" void glTexCoord4f(float s,float t,float r,float q) { if (metalModeEnabled()) { g_fixedTexcoord[0]=s; g_fixedTexcoord[1]=t; return; } glDispatch<void,float,float,float,float>("glTexCoord4f",s,t,r,q); }
extern "C" void glTexGeni(uint32_t coord,uint32_t pname,int32_t param) { glDispatch<void,uint32_t,uint32_t,int32_t>("glTexGeni",coord,pname,param);if(!metalModeEnabled())return;if(coord==0x2000&&pname==0x2500)g_fixedTexGenModeS=static_cast<uint32_t>(param);else if(coord==0x2001&&pname==0x2500)g_fixedTexGenModeT=static_cast<uint32_t>(param); }
extern "C" void glTexGenf(uint32_t coord,uint32_t pname,float param) { glDispatch<void,uint32_t,uint32_t,float>("glTexGenf",coord,pname,param);if(!metalModeEnabled())return;if(coord==0x2000&&pname==0x2500)g_fixedTexGenModeS=static_cast<uint32_t>(param);else if(coord==0x2001&&pname==0x2500)g_fixedTexGenModeT=static_cast<uint32_t>(param); }
extern "C" void glTexGenfv(uint32_t coord,uint32_t pname,const float* params) { glDispatch<void,uint32_t,uint32_t,const float*>("glTexGenfv",coord,pname,params);if(!metalModeEnabled()||!params)return;if(coord==0x2000&&pname==0x2501)std::memcpy(g_fixedTexGenPlaneS,params,sizeof(g_fixedTexGenPlaneS));else if(coord==0x2001&&pname==0x2501)std::memcpy(g_fixedTexGenPlaneT,params,sizeof(g_fixedTexGenPlaneT));else if(coord==0x2000&&pname==0x2502)std::memcpy(g_fixedTexGenPlaneS,params,sizeof(g_fixedTexGenPlaneS));else if(coord==0x2001&&pname==0x2502)std::memcpy(g_fixedTexGenPlaneT,params,sizeof(g_fixedTexGenPlaneT)); }
extern "C" void glTexGeniv(uint32_t coord,uint32_t pname,const int32_t* params) { if(params){float values[4]={(float)params[0],(float)params[1],(float)params[2],(float)params[3]};glTexGenfv(coord,pname,values);}else glDispatch<void,uint32_t,uint32_t,const int32_t*>("glTexGeniv",coord,pname,params); }
extern "C" void glColor3ub(unsigned char r, unsigned char g, unsigned char b) {
    if (metalModeEnabled()) { g_fixedColor[0]=r/255.0f; g_fixedColor[1]=g/255.0f; g_fixedColor[2]=b/255.0f; g_fixedColor[3]=1.0f; return; }
    glDispatch<void, unsigned char, unsigned char, unsigned char>("glColor3ub", r, g, b);
}
extern "C" void glColor4ub(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    if (metalModeEnabled()) { g_fixedColor[0]=r/255.0f; g_fixedColor[1]=g/255.0f; g_fixedColor[2]=b/255.0f; g_fixedColor[3]=a/255.0f; return; }
    glDispatch<void, unsigned char, unsigned char, unsigned char, unsigned char>("glColor4ub", r, g, b, a);
}
GL_PASSTHROUGH2(void, glColorMaterial, uint32_t, face, uint32_t, mode)

// ---------------------------------------------------------------------------
// Lighting / material (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glNormal3f(float x, float y, float z) {
    if (g_listCompiling) { recordFixed(FixedCommandKind::Normal,0,{x,y,z}); if (!g_listExecute) return; }
    if (metalModeEnabled()) { g_fixedNormal[0]=x; g_fixedNormal[1]=y; g_fixedNormal[2]=z; return; }
    glDispatch<void, float, float, float>("glNormal3f", x, y, z);
}
extern "C" void glLightfv(uint32_t light, uint32_t pname, const float* params) {
    glDispatch<void, uint32_t, uint32_t, const float*>("glLightfv", light, pname, params);
    if (!metalModeEnabled() || !params || light < 0x4000 || light >= 0x4008) return;
    const int index = static_cast<int>(light - 0x4000);
    if (pname == 0x1203) { std::memcpy(g_fixedLightPositions[index], params, sizeof(g_fixedLightPositions[index])); if(index==0)std::memcpy(g_fixedLightPosition,params,sizeof(g_fixedLightPosition)); }
    else if (pname == 0x1200) { std::memcpy(g_fixedLightAmbients[index], params, sizeof(g_fixedLightAmbients[index])); if(index==0)std::memcpy(g_fixedLightAmbient,params,sizeof(g_fixedLightAmbient)); }
    else if (pname == 0x1201) { std::memcpy(g_fixedLightDiffuses[index], params, sizeof(g_fixedLightDiffuses[index])); if(index==0)std::memcpy(g_fixedLightDiffuse,params,sizeof(g_fixedLightDiffuse)); }
    else if (pname == 0x1202) std::memcpy(g_fixedLightSpeculars[index], params, sizeof(g_fixedLightSpeculars[index]));
}
GL_PASSTHROUGH2(void, glLightModelfv, uint32_t, pname, const float*, params)
extern "C" void glMaterialf(uint32_t face, uint32_t pname, float param) { glDispatch<void,uint32_t,uint32_t,float>("glMaterialf",face,pname,param); if(metalModeEnabled()&&pname==0x1601)g_fixedMaterialShininess=param; }
extern "C" void glMaterialfv(uint32_t face, uint32_t pname, const float* params) {
    glDispatch<void, uint32_t, uint32_t, const float*>("glMaterialfv", face, pname, params);
    if (metalModeEnabled() && params) {
        if (pname == 0x1200) std::memcpy(g_fixedMaterialAmbient,params,sizeof(g_fixedMaterialAmbient));
        else if (pname == 0x1201) std::memcpy(g_fixedMaterialDiffuse, params, sizeof(g_fixedMaterialDiffuse));
        else if (pname == 0x1202) std::memcpy(g_fixedMaterialSpecular,params,sizeof(g_fixedMaterialSpecular));
        else if (pname == 0x1600) std::memcpy(g_fixedMaterialEmission,params,sizeof(g_fixedMaterialEmission));
        else if (pname == 0x1601) g_fixedMaterialShininess=params[0];
    }
}
GL_PASSTHROUGH1(void, glShadeModel, uint32_t, mode)

// ---------------------------------------------------------------------------
// Fog (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glFogfv(uint32_t pname, const float* params) { glDispatch<void,uint32_t,const float*>("glFogfv",pname,params); if(metalModeEnabled()&&params){if(pname==0x0B66)std::memcpy(g_fixedFogColor,params,sizeof(g_fixedFogColor));else if(pname==0x0B63)g_fixedFogStart=params[0];else if(pname==0x0B64)g_fixedFogEnd=params[0];else if(pname==0x0B62)g_fixedFogDensity=params[0];} }
extern "C" void glFogi(uint32_t pname, int32_t param) { glDispatch<void,uint32_t,int32_t>("glFogi",pname,param); if(metalModeEnabled()&&pname==0x0B65)g_fixedFogMode=static_cast<uint32_t>(param); }
extern "C" void glFogf(uint32_t pname, float param) { glDispatch<void,uint32_t,float>("glFogf",pname,param); if(metalModeEnabled()){if(pname==0x0B63)g_fixedFogStart=param;else if(pname==0x0B64)g_fixedFogEnd=param;else if(pname==0x0B62)g_fixedFogDensity=param;} }
GL_PASSTHROUGH2(void, glFogiv, uint32_t, pname, const int32_t*, params)

// ---------------------------------------------------------------------------
// Alpha test (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glAlphaFunc(uint32_t func, float ref) { glDispatch<void,uint32_t,float>("glAlphaFunc",func,ref); g_fixedAlphaFunc=func; g_fixedAlphaRef=ref; }

// ---------------------------------------------------------------------------
// Clip planes (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glClipPlane(uint32_t plane,const double* equation) { glDispatch<void,uint32_t,const double*>("glClipPlane",plane,equation);if(!metalModeEnabled()||!equation||plane<0x3000||plane>=0x3006)return;std::memcpy(g_fixedClipPlanes[plane-0x3000],equation,sizeof(g_fixedClipPlanes[plane-0x3000]));}

// ---------------------------------------------------------------------------
// Matrix stack (fixed pipeline, GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glMatrixMode(uint32_t mode) { if(g_listCompiling){recordFixed(FixedCommandKind::MatrixMode,mode,{static_cast<float>(mode)});if(!g_listExecute)return;} if (metalModeEnabled() && (mode == 0x1700 || mode == 0x1701)) g_fixedMatrixMode = mode; else glDispatch<void, uint32_t>("glMatrixMode", mode); }
extern "C" void glLoadIdentity(void) { if(g_listCompiling){recordFixed(FixedCommandKind::LoadIdentity,0,{});if(!g_listExecute)return;} if (metalModeEnabled()) { float* m=fixedCurrentMatrix(); std::fill(m,m+16,0); m[0]=m[5]=m[10]=m[15]=1; } else glDispatch<void>("glLoadIdentity"); }
extern "C" void glPushMatrix(void) { if(g_listCompiling){recordFixed(FixedCommandKind::PushMatrix,0,{});if(!g_listExecute)return;} if (metalModeEnabled()) { auto& stack=g_fixedMatrixMode==0x1701?g_fixedProjectionStack:g_fixedModelviewStack; auto* m=fixedCurrentMatrix(); stack.emplace_back(); std::memcpy(stack.back().data(),m,sizeof(float)*16); } else glDispatch<void>("glPushMatrix"); }
extern "C" void glPopMatrix(void) { if(g_listCompiling){recordFixed(FixedCommandKind::PopMatrix,0,{});if(!g_listExecute)return;} if (metalModeEnabled()) { auto& stack=g_fixedMatrixMode==0x1701?g_fixedProjectionStack:g_fixedModelviewStack; if(!stack.empty()){std::memcpy(fixedCurrentMatrix(),stack.back().data(),sizeof(float)*16);stack.pop_back();} } else glDispatch<void>("glPopMatrix"); }
extern "C" void glTranslatef(float x,float y,float z) { if(g_listCompiling){recordFixed(FixedCommandKind::Translate,0,{x,y,z});if(!g_listExecute)return;} if (metalModeEnabled()){float t[16]={1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1};float r[16];fixedMultiply(r,fixedCurrentMatrix(),t);std::memcpy(fixedCurrentMatrix(),r,sizeof(r));} else glDispatch<void,float,float,float>("glTranslatef",x,y,z); }
extern "C" void glScalef(float x,float y,float z) { if(g_listCompiling){recordFixed(FixedCommandKind::Scale,0,{x,y,z});if(!g_listExecute)return;} if (metalModeEnabled()){float t[16]={x,0,0,0,0,y,0,0,0,0,z,0,0,0,0,1};float r[16];fixedMultiply(r,fixedCurrentMatrix(),t);std::memcpy(fixedCurrentMatrix(),r,sizeof(r));} else glDispatch<void,float,float,float>("glScalef",x,y,z); }
extern "C" void glRotatef(float angle,float x,float y,float z) { if(g_listCompiling){recordFixed(FixedCommandKind::Rotate,0,{angle,x,y,z});if(!g_listExecute)return;} if (metalModeEnabled()){float rad=angle*0.0174532925199433f,c=std::cos(rad),s=std::sin(rad),len=std::sqrt(x*x+y*y+z*z);if(len){x/=len;y/=len;z/=len;}float t[16]={x*x*(1-c)+c,x*y*(1-c)+z*s,x*z*(1-c)-y*s,0,x*y*(1-c)-z*s,y*y*(1-c)+c,y*z*(1-c)+x*s,0,x*z*(1-c)+y*s,y*z*(1-c)-x*s,z*z*(1-c)+c,0,0,0,0,1};float r[16];fixedMultiply(r,fixedCurrentMatrix(),t);std::memcpy(fixedCurrentMatrix(),r,sizeof(r));} else glDispatch<void,float,float,float,float>("glRotatef",angle,x,y,z); }
extern "C" void glOrtho(double l,double r,double b,double t,double n,double f) { if (metalModeEnabled()){float m[16]={float(2/(r-l)),0,0,0,0,float(2/(t-b)),0,0,0,0,float(-2/(f-n)),0,float(-(r+l)/(r-l)),float(-(t+b)/(t-b)),float(-(f+n)/(f-n)),1};float out[16];fixedMultiply(out,fixedCurrentMatrix(),m);std::memcpy(fixedCurrentMatrix(),out,sizeof(out));} else glDispatch<void,double,double,double,double,double,double>("glOrtho",l,r,b,t,n,f); }
GL_PASSTHROUGH6(void, glFrustum, double, l, double, r, double, b, double, t, double, n, double, f)

// ---------------------------------------------------------------------------
// Color (legacy)
// ---------------------------------------------------------------------------
extern "C" void glColor3f(float r, float g, float b) {
    if (g_listCompiling) { recordFixed(FixedCommandKind::Color,0,{r,g,b,1}); if (!g_listExecute) return; }
    if (metalModeEnabled()) { g_fixedColor[0] = r; g_fixedColor[1] = g; g_fixedColor[2] = b; g_fixedColor[3] = 1.0f; return; }
    glDispatch<void, float, float, float>("glColor3f", r, g, b);
}
extern "C" void glColor4f(float r, float g, float b, float a) {
    if (g_listCompiling) { recordFixed(FixedCommandKind::Color,0,{r,g,b,a}); if (!g_listExecute) return; }
    if (metalModeEnabled()) { g_fixedColor[0] = r; g_fixedColor[1] = g; g_fixedColor[2] = b; g_fixedColor[3] = a; return; }
    glDispatch<void, float, float, float, float>("glColor4f", r, g, b, a);
}

// ---------------------------------------------------------------------------
// Texture upload / readback
// ---------------------------------------------------------------------------
extern "C" void glTexImage1D(uint32_t target,int32_t level,int32_t internalFormat,int32_t width,int32_t border,uint32_t format,uint32_t type,const void* data) { const uint32_t textureName=g_activeTextureUnit<g_textureUnits.size()?g_textureUnits[g_activeTextureUnit]:0;if(metalModeEnabled()&&target==0x0DE0&&level==0&&width>0&&textureName){std::vector<uint8_t> pixels,unpacked;const void* source=data;if(g_boundPixelUnpackBuffer){size_t bytes=pixelUploadBytes(width,1,1,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(data,bytes,unpacked)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}source=unpacked.data();}if(!source)pixels.assign(static_cast<size_t>(width)*4,0);else if(!convertPixelsToBGRA(width,1,format,type,source,pixels,g_glBridge.state().unpackAlignment)){metalsharp::GLErrorTracker::instance().setError(0x0500);return;}std::vector<uint8_t> storage=encodeTextureStorage(pixels,internalFormat);uint64_t handle=g_metalRenderer.createTexture1D(width,static_cast<uint32_t>(internalFormat),storage.data(),true);if(!handle){metalsharp::GLErrorTracker::instance().setError(0x0505);return;}std::lock_guard<std::mutex> lock(g_resourceMutex);auto& texture=g_textures[textureName];texture.metalHandle=handle;texture.width=width;texture.height=1;texture.depth=1;texture.target=target;texture.internalFormat=internalFormat;texture.pixels=std::move(pixels);return;}glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTexImage1D",target,level,internalFormat,width,border,format,type,data); }

extern "C" void glTexImage2D(uint32_t target, int32_t level, int32_t internalFormat, int32_t w, int32_t h,
                              int32_t border, uint32_t format, uint32_t type, const void* data) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && (target == 0x0DE1 || target == 0x84F5 || (target >= 0x8515 && target <= 0x851A)) && level == 0 && w > 0 && h > 0 && textureName) {
        const bool cubeFace = target >= 0x8515 && target <= 0x851A;
        const uint32_t cubeFaceIndex = cubeFace ? target - 0x8515 : 0;
        const bool scalarUint = internalFormat == 0x8236 && format == 0x8D94 && type == 0x1405;
        const bool scalarFloat = internalFormat == 0x822E && format == 0x1903 && type == 0x1406;
        if (scalarUint || scalarFloat) { std::vector<uint8_t> raw(static_cast<size_t>(w)*h*4,0),unpacked;const void* source=data;if(g_boundPixelUnpackBuffer){size_t bytes=static_cast<size_t>(w)*h*4;if(!readPixelUnpackBuffer(data,bytes,unpacked)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}source=unpacked.data();}if(source)std::memcpy(raw.data(),source,raw.size());ExperimentalTexture uploadTexture;uploadTexture.width=w;uploadTexture.height=h;uploadTexture.internalFormat=internalFormat;uploadTexture.pixels=raw;uint64_t handle=createTextureFromCanonical(uploadTexture);if(!handle){metalsharp::GLErrorTracker::instance().setError(0x0505);return;}std::lock_guard<std::mutex> lock(g_resourceMutex);auto& texture=g_textures[textureName];texture.metalHandle=handle;texture.width=w;texture.height=h;texture.internalFormat=internalFormat;texture.pixels=std::move(raw);return; }
        const bool depthTexture = internalFormat == 0x1902 || internalFormat == 0x81A5 || internalFormat == 0x81A6 || internalFormat == 0x8CAC || internalFormat == 0x88F0 || internalFormat == 0x8D48;
        if (depthTexture) {
            uint64_t handle = g_metalRenderer.createDepthStencilTarget(static_cast<uint32_t>(w), static_cast<uint32_t>(h), static_cast<uint32_t>(internalFormat));
            if (!handle) { metalsharp::GLErrorTracker::instance().setError(0x0505); return; }
            std::lock_guard<std::mutex> lock(g_resourceMutex); auto& texture=g_textures[textureName]; texture.metalHandle=handle; texture.width=w; texture.height=h; texture.internalFormat=internalFormat; return;
        }
        std::vector<uint8_t> pixels, unpacked;
        const void* uploadData=data;
        if (g_boundPixelUnpackBuffer) { size_t bytes=pixelUploadBytes(w,h,1,format,type,g_glBridge.state().unpackAlignment); if(!readPixelUnpackBuffer(data,bytes,unpacked)){ metalsharp::GLErrorTracker::instance().setError(0x0501);return;} uploadData=unpacked.data(); }
        if (!uploadData) pixels.assign(static_cast<size_t>(w) * h * 4, 0);
        else if (!convertPixelsToBGRA(w, h, format, type, uploadData, pixels, g_glBridge.state().unpackAlignment)) { metalsharp::GLErrorTracker::instance().setError(0x0500); return; }
        if (cubeFace) { std::lock_guard<std::mutex> lock(g_resourceMutex);auto& texture=g_textures[textureName];texture.target=0x8513;texture.width=w;texture.height=h;texture.internalFormat=internalFormat;texture.cubeFaceMask|=static_cast<uint8_t>(1u<<cubeFaceIndex);texture.cubePixels[cubeFaceIndex]=pixels;const void* faces[6]={};for(uint32_t face=0;face<6;++face)if(!texture.cubePixels[face].empty())faces[face]=texture.cubePixels[face].data();texture.metalHandle=g_metalRenderer.createTextureCube(w,h,static_cast<uint32_t>(internalFormat),faces);texture.pixels=pixels;return; }
        ExperimentalTexture uploadTexture; uploadTexture.width=static_cast<uint32_t>(w); uploadTexture.height=static_cast<uint32_t>(h); uploadTexture.internalFormat=internalFormat; uploadTexture.pixels=pixels;
        uint64_t handle = createTextureFromCanonical(uploadTexture);
        if (!handle) { metalsharp::GLErrorTracker::instance().setError(0x0505); return; }
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto& texture = g_textures[textureName];
        texture.metalHandle = handle;
        texture.width = static_cast<uint32_t>(w);
        texture.height = static_cast<uint32_t>(h);
        texture.internalFormat = internalFormat;
        texture.target = target;
        texture.pixels = std::move(pixels);
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>(
        "glTexImage2D", target, level, internalFormat, w, h, border, format, type, data);
}
extern "C" void glGetTexImage(uint32_t target, int32_t level, uint32_t format, uint32_t type, void* pixels) {
    if (metalModeEnabled() && (target == 0x806F || target == 0x8C1A) && level == 0 && pixels && g_activeTextureUnit < g_textureUnits.size()) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto it = g_textures.find(g_textureUnits[g_activeTextureUnit]);
        if (it != g_textures.end() && !it->second.pixels.empty()) {
            if(writeRGBA8Pixels(it->second.pixels.data(),static_cast<int32_t>(it->second.width),static_cast<int32_t>(it->second.height),static_cast<int32_t>(it->second.depth),format,type,pixels,g_glBridge.state().packAlignment))return;
            glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,void*>("glGetTexImage",target,level,format,type,pixels); return;
        }
    }
    if (metalModeEnabled() && target >= 0x8515 && target <= 0x851A && level == 0 && pixels && g_activeTextureUnit < g_textureUnits.size()) { uint64_t handle=0;uint32_t width=0,height=0;{std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(it!=g_textures.end()){handle=it->second.metalHandle;width=it->second.width;height=it->second.height;}}std::vector<uint8_t> rgba(static_cast<size_t>(width)*height*4);if(handle&&g_metalRenderer.readTextureRGBA8(handle,0,0,width,height,rgba.data(),target-0x8515)&&writeRGBA8Pixels(rgba.data(),width,height,1,format,type,pixels,g_glBridge.state().packAlignment))return; }
    if (metalModeEnabled() && (target == 0x0DE0 || target == 0x0DE1 || target == 0x84F5) && level == 0 && pixels && g_activeTextureUnit < g_textureUnits.size()) {
        uint64_t scalarHandle=0;int32_t scalarFormat=0;uint32_t scalarWidth=0,scalarHeight=0;{std::lock_guard<std::mutex> lock(g_resourceMutex);auto scalar=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(scalar!=g_textures.end()){scalarHandle=scalar->second.metalHandle;scalarFormat=scalar->second.internalFormat;scalarWidth=scalar->second.width;scalarHeight=scalar->second.height;}}
        if(scalarHandle&&((scalarFormat==0x8236&&format==0x8D94&&type==0x1405)||(scalarFormat==0x822E&&format==0x1903&&type==0x1406))&&g_metalRenderer.readTextureScalar32(scalarHandle,0,0,scalarWidth,scalarHeight,pixels))return;
        std::lock_guard<std::mutex> lock(g_resourceMutex); auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);
        if (it != g_textures.end()) {
            std::vector<uint8_t> rgba(static_cast<size_t>(it->second.width) * it->second.height * 4);
            if (!g_metalRenderer.readTextureRGBA8(it->second.metalHandle,0,0,it->second.width,it->second.height,rgba.data())) return;
            if(!writeRGBA8Pixels(rgba.data(),static_cast<int32_t>(it->second.width),static_cast<int32_t>(it->second.height),1,format,type,pixels,g_glBridge.state().packAlignment))glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,void*>("glGetTexImage",target,level,format,type,pixels);
            return;
        }
    }
    glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,void*>("glGetTexImage",target,level,format,type,pixels);
}

extern "C" void glTexImage3D(uint32_t target, int32_t level, int32_t internalFormat, int32_t width, int32_t height,
                              int32_t depth, int32_t border, uint32_t format, uint32_t type, const void* data) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && (target == 0x806F || target == 0x8C1A) && level == 0 && width > 0 && height > 0 && depth > 0 && format == 0x1908 && type == 0x1401 && textureName) {
        size_t bytes = static_cast<size_t>(width) * height * depth * 4u;
        std::vector<uint8_t> rgba(bytes, 0), unpacked;
        const void* uploadData=data;
        if(g_boundPixelUnpackBuffer){size_t uploadBytes=pixelUploadBytes(width,height,depth,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(data,uploadBytes,unpacked)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}uploadData=unpacked.data();}
        if (uploadData) std::memcpy(rgba.data(), uploadData, bytes);
        uint64_t handle = target == 0x806F ? g_metalRenderer.createTexture3D(width, height, depth, rgba.data()) : g_metalRenderer.createTexture2DArray(width, height, depth, rgba.data());
        if (!handle) { metalsharp::GLErrorTracker::instance().setError(0x0505); return; }
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto& texture = g_textures[textureName]; texture.metalHandle = handle; texture.width = width; texture.height = height; texture.depth = depth; texture.target = target; texture.internalFormat = internalFormat; texture.pixels = std::move(rgba);
        return;
    }
    glDispatch<void, uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTexImage3D", target,level,internalFormat,width,height,depth,border,format,type,data);
}
extern "C" void glTexSubImage3D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset, int32_t zoffset,
                                  int32_t width, int32_t height, int32_t depth, uint32_t format, uint32_t type, const void* data) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && (target == 0x806F || target == 0x8C1A) && level == 0 && width > 0 && height > 0 && depth > 0 &&
        xoffset >= 0 && yoffset >= 0 && zoffset >= 0 && (data || g_boundPixelUnpackBuffer) && textureName && format == 0x1908 && type == 0x1401) {
        std::vector<uint8_t> unpacked; const void* uploadData=data;
        if(g_boundPixelUnpackBuffer){size_t uploadBytes=pixelUploadBytes(width,height,depth,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(data,uploadBytes,unpacked)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}uploadData=unpacked.data();}
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto it = g_textures.find(textureName);
        if (it != g_textures.end() && static_cast<uint32_t>(xoffset + width) <= it->second.width &&
            static_cast<uint32_t>(yoffset + height) <= it->second.height && static_cast<uint32_t>(zoffset + depth) <= it->second.depth) {
            const uint8_t* source = static_cast<const uint8_t*>(uploadData);
            for (int32_t z = 0; z < depth; ++z) {
                const uint8_t* slice = source + static_cast<size_t>(z) * width * height * 4;
                for (int32_t row = 0; row < height; ++row) {
                    size_t dst = ((static_cast<size_t>(zoffset + z) * it->second.height + yoffset + row) * it->second.width + xoffset) * 4;
                    size_t src = static_cast<size_t>(row) * width * 4;
                    std::memcpy(it->second.pixels.data() + dst, slice + src, static_cast<size_t>(width) * 4);
                }
            }
            it->second.metalHandle = target == 0x806F
                ? g_metalRenderer.createTexture3D(it->second.width, it->second.height, it->second.depth, it->second.pixels.data())
                : g_metalRenderer.createTexture2DArray(it->second.width, it->second.height, it->second.depth, it->second.pixels.data());
            if (it->second.metalHandle) return;
        }
    }
    glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>(
        "glTexSubImage3D", target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data);
}

extern "C" void glCopyImageSubData(uint32_t srcName, uint32_t srcTarget, int32_t srcLevel, int32_t srcX, int32_t srcY, int32_t srcZ,
                                    uint32_t dstName, uint32_t dstTarget, int32_t dstLevel, int32_t dstX, int32_t dstY, int32_t dstZ,
                                    int32_t width, int32_t height, int32_t depth) {
    if (!metalModeEnabled()) {
        glDispatch<void,uint32_t,uint32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t>(
            "glCopyImageSubData", srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, width, height, depth);
        return;
    }
    const bool supportedTarget = (srcTarget == 0x0DE1 || srcTarget == 0x806F || srcTarget == 0x8C1A) &&
                                 (dstTarget == 0x0DE1 || dstTarget == 0x806F || dstTarget == 0x8C1A);
    if (!supportedTarget || srcLevel != 0 || dstLevel != 0 || srcX < 0 || srcY < 0 || srcZ < 0 || dstX < 0 || dstY < 0 || dstZ < 0 || width <= 0 || height <= 0 || depth <= 0) {
        metalsharp::GLErrorTracker::instance().setError(0x0500); return;
    }
    std::lock_guard<std::mutex> lock(g_resourceMutex);
    auto source = g_textures.find(srcName), destination = g_textures.find(dstName);
    if (source == g_textures.end() || destination == g_textures.end() ||
        static_cast<uint32_t>(srcX + width) > source->second.width || static_cast<uint32_t>(srcY + height) > source->second.height || static_cast<uint32_t>(srcZ + depth) > source->second.depth ||
        static_cast<uint32_t>(dstX + width) > destination->second.width || static_cast<uint32_t>(dstY + height) > destination->second.height || static_cast<uint32_t>(dstZ + depth) > destination->second.depth) {
        metalsharp::GLErrorTracker::instance().setError(0x0501); return;
    }
    std::vector<uint8_t> copied(static_cast<size_t>(width) * height * depth * 4);
    for (int32_t z = 0; z < depth; ++z) for (int32_t row = 0; row < height; ++row) {
        size_t sourceOffset = ((static_cast<size_t>(srcZ + z) * source->second.height + srcY + row) * source->second.width + srcX) * 4;
        size_t copiedOffset = (static_cast<size_t>(z) * height + row) * width * 4;
        std::memcpy(copied.data() + copiedOffset, source->second.pixels.data() + sourceOffset, static_cast<size_t>(width) * 4);
    }
    for (int32_t z = 0; z < depth; ++z) for (int32_t row = 0; row < height; ++row) {
        size_t destinationOffset = ((static_cast<size_t>(dstZ + z) * destination->second.height + dstY + row) * destination->second.width + dstX) * 4;
        size_t copiedOffset = (static_cast<size_t>(z) * height + row) * width * 4;
        std::memcpy(destination->second.pixels.data() + destinationOffset, copied.data() + copiedOffset, static_cast<size_t>(width) * 4);
    }
    destination->second.metalHandle = dstTarget == 0x0DE1
        ? createTextureFromCanonical(destination->second)
        : dstTarget == 0x806F
            ? g_metalRenderer.createTexture3D(destination->second.width, destination->second.height, destination->second.depth, destination->second.pixels.data())
            : g_metalRenderer.createTexture2DArray(destination->second.width, destination->second.height, destination->second.depth, destination->second.pixels.data());
    if (!destination->second.metalHandle) metalsharp::GLErrorTracker::instance().setError(0x0505);
}

extern "C" void glReadPixels(int32_t x, int32_t y, int32_t w, int32_t h, uint32_t format, uint32_t type, void* data) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (metalModeEnabled() && (isExperimentalProgram(program) || !program)) {
        bool read = false;
        const bool pixelPack = g_boundPixelPackBuffer != 0;
        const size_t pixelPackOffset = pixelPack ? reinterpret_cast<size_t>(data) : 0;
        std::vector<uint8_t> pixelPackScratch;
        if (pixelPack) { size_t scalar=type==0x1406?4:(type==0x1401?1:2); size_t channels=format==0x1907?3:4; if(type==0x8363||type==0x8033||type==0x8034)channels=1; size_t rowBytes=static_cast<size_t>(std::max(0,w))*scalar*channels; size_t alignment=static_cast<size_t>(std::max(1,g_glBridge.state().packAlignment)); size_t stride=(rowBytes+alignment-1)/alignment*alignment; pixelPackScratch.assign(stride*static_cast<size_t>(std::max(0,h)),0); data=pixelPackScratch.data(); }
        if (x >= 0 && y >= 0 && w > 0 && h > 0 && format == 0x1901 && type == 0x1401) {
            uint64_t stencilHandle=0; {std::lock_guard<std::mutex> lock(g_resourceMutex);uint32_t framebuffer=g_glBridge.state().boundReadFramebuffer?g_glBridge.state().boundReadFramebuffer:g_glBridge.state().boundFramebuffer;auto fbo=g_framebuffers.find(framebuffer);if(fbo!=g_framebuffers.end())stencilHandle=fbo->second.stencilHandle;}
            if(stencilHandle){uint8_t value=0;bool shadow=false;{std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_stencilClearShadow.find(stencilHandle);if(it!=g_stencilClearShadow.end()){value=it->second;shadow=true;}}if(shadow){size_t alignment=static_cast<size_t>(std::max(1,g_glBridge.state().packAlignment));size_t stride=(static_cast<size_t>(w)+alignment-1)/alignment*alignment;auto* out=static_cast<uint8_t*>(data);for(int32_t row=0;row<h;++row)std::memset(out+static_cast<size_t>(row)*stride,value,static_cast<size_t>(w));if(pixelPack){uint64_t handle=0;{std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(g_boundPixelPackBuffer);if(it!=g_buffers.end())handle=it->second.metalHandle;}if(handle)g_metalRenderer.updateBuffer(handle,pixelPackOffset,pixelPackScratch.data(),pixelPackScratch.size());}return;}std::vector<uint8_t> stencil(static_cast<size_t>(w)*h);if(g_metalRenderer.readStencil8(stencilHandle,static_cast<uint32_t>(x),static_cast<uint32_t>(y),static_cast<uint32_t>(w),static_cast<uint32_t>(h),stencil.data())){size_t alignment=static_cast<size_t>(std::max(1,g_glBridge.state().packAlignment));size_t stride=(static_cast<size_t>(w)+alignment-1)/alignment*alignment;auto* out=static_cast<uint8_t*>(data);for(int32_t row=0;row<h;++row)std::memcpy(out+static_cast<size_t>(row)*stride,stencil.data()+static_cast<size_t>(row)*w,static_cast<size_t>(w));return;}}
        }
        if (x >= 0 && y >= 0 && w > 0 && h > 0 && format == 0x1902 && type == 0x1406) {
            uint64_t depthHandle=0; { std::lock_guard<std::mutex> lock(g_resourceMutex); uint32_t framebuffer=g_glBridge.state().boundReadFramebuffer?g_glBridge.state().boundReadFramebuffer:g_glBridge.state().boundFramebuffer; auto fbo=g_framebuffers.find(framebuffer); if(fbo!=g_framebuffers.end())depthHandle=fbo->second.depthHandle; }
            if(depthHandle){std::vector<float> depth(static_cast<size_t>(w)*h);if(g_metalRenderer.readDepth32(depthHandle,static_cast<uint32_t>(x),static_cast<uint32_t>(y),static_cast<uint32_t>(w),static_cast<uint32_t>(h),depth.data())){size_t alignment=static_cast<size_t>(std::max(1,g_glBridge.state().packAlignment));size_t stride=(static_cast<size_t>(w)*sizeof(float)+alignment-1)/alignment*alignment;auto* out=static_cast<uint8_t*>(data);for(int32_t row=0;row<h;++row)std::memcpy(out+static_cast<size_t>(row)*stride,depth.data()+static_cast<size_t>(row)*w,static_cast<size_t>(w)*sizeof(float));return;}}
        }
        if (x >= 0 && y >= 0 && w > 0 && h > 0 && ((format == 0x1908 || format == 0x1907) && (type == 0x1401 || type == 0x1403 || type == 0x1406 || type == 0x140B || type == 0x8363 || type == 0x8033 || type == 0x8034))) {
            uint64_t textureHandle = 0;
            uint32_t textureSlice = 0;
            { std::lock_guard<std::mutex> lock(g_resourceMutex);
              const uint32_t readFramebuffer = g_glBridge.state().boundReadFramebuffer ? g_glBridge.state().boundReadFramebuffer : g_glBridge.state().boundFramebuffer;
              auto fbo = g_framebuffers.find(readFramebuffer);
              if (fbo != g_framebuffers.end()) {
                  if (fbo->second.colorTexture) { auto texture = g_textures.find(fbo->second.colorTexture); if (texture != g_textures.end()) textureHandle = texture->second.metalHandle; textureSlice = fbo->second.colorLayer; }
                  else textureHandle = fbo->second.colorHandle;
              } }
            std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
            bool copied = textureHandle ? g_metalRenderer.readTextureRGBA8(textureHandle, static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h), rgba.data(), textureSlice) :
                g_metalRenderer.readPixelsRGBA8(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h), rgba.data());
            const size_t pack = static_cast<size_t>(std::max(1, g_glBridge.state().packAlignment));
            auto packedStride = [pack](size_t rowBytes) { return (rowBytes + pack - 1) / pack * pack; };
            if (copied && type == 0x1401 && format == 0x1908) { auto* out=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*4); for(int32_t row=0;row<h;++row) std::memcpy(out+static_cast<size_t>(row)*stride,rgba.data()+static_cast<size_t>(row)*w*4,static_cast<size_t>(w)*4); read = true; }
            else if (copied && type == 0x1401 && format == 0x1907) { auto* rgb=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*3); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t out=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*3;size_t in=(static_cast<size_t>(row)*w+column)*4;rgb[out]=rgba[in];rgb[out+1]=rgba[in+1];rgb[out+2]=rgba[in+2];} read=true; }
            else if (copied && type == 0x1406 && format == 0x1908) { auto* floats=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*sizeof(float)*4); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t out=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*sizeof(float)*4;size_t in=(static_cast<size_t>(row)*w+column)*4;float values[4]={rgba[in]/255.0f,rgba[in+1]/255.0f,rgba[in+2]/255.0f,rgba[in+3]/255.0f};std::memcpy(floats+out,values,sizeof(values));} read=true; }
            else if (copied && type == 0x1403 && format == 0x1908) { auto* out=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*8); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t dst=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*8,src=(static_cast<size_t>(row)*w+column)*4;uint16_t values[4]={static_cast<uint16_t>(rgba[src]*257u),static_cast<uint16_t>(rgba[src+1]*257u),static_cast<uint16_t>(rgba[src+2]*257u),static_cast<uint16_t>(rgba[src+3]*257u)};std::memcpy(out+dst,values,sizeof(values));} read=true; }
            else if (copied && type == 0x140B && format == 0x1908) { auto* out=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*8); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t dst=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*8,src=(static_cast<size_t>(row)*w+column)*4;uint16_t values[4]={floatToHalf(rgba[src]/255.0f),floatToHalf(rgba[src+1]/255.0f),floatToHalf(rgba[src+2]/255.0f),floatToHalf(rgba[src+3]/255.0f)};std::memcpy(out+dst,values,sizeof(values));} read=true; }
            else if (copied && type == 0x8363 && format == 0x1907) { auto* packed=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*2); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t out=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*2;size_t in=(static_cast<size_t>(row)*w+column)*4;uint16_t value=static_cast<uint16_t>((rgba[in+0]*31/255<<11)|(rgba[in+1]*63/255<<5)|(rgba[in+2]*31/255));std::memcpy(packed+out,&value,2);} read=true; }
            else if (copied && type == 0x8033 && format == 0x1908) { auto* packed=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*2); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t out=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*2;size_t in=(static_cast<size_t>(row)*w+column)*4;uint16_t value=static_cast<uint16_t>((rgba[in+0]*15/255<<12)|(rgba[in+1]*15/255<<8)|(rgba[in+2]*15/255<<4)|(rgba[in+3]*15/255));std::memcpy(packed+out,&value,2);} read=true; }
            else if (copied && type == 0x8034 && format == 0x1908) { auto* packed=static_cast<uint8_t*>(data); const size_t stride=packedStride(static_cast<size_t>(w)*2); for(int32_t row=0;row<h;++row) for(int32_t column=0;column<w;++column){size_t out=static_cast<size_t>(row)*stride+static_cast<size_t>(column)*2;size_t in=(static_cast<size_t>(row)*w+column)*4;uint16_t value=static_cast<uint16_t>((rgba[in+0]*31/255<<11)|(rgba[in+1]*31/255<<6)|(rgba[in+2]*31/255<<1)|(rgba[in+3]>=128));std::memcpy(packed+out,&value,2);} read=true; }
        }
        if (read) {
            if (pixelPack) { uint64_t handle=0; {std::lock_guard<std::mutex> lock(g_bufferMutex);auto it=g_buffers.find(g_boundPixelPackBuffer);if(it!=g_buffers.end())handle=it->second.metalHandle;} if(!handle||!g_metalRenderer.updateBuffer(handle,pixelPackOffset,pixelPackScratch.data(),pixelPackScratch.size()))metalsharp::GLErrorTracker::instance().setError(0x0501); }
            return;
        }
        metalsharp::GLErrorTracker::instance().setError(0x0502);
        return;
    }
    glDispatch<void, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, void*>(
        "glReadPixels", x, y, w, h, format, type, data);
}

// ---------------------------------------------------------------------------
// Texture objects (GL 1.1-1.3)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGenTextures, int32_t, n, uint32_t*, textures)
extern "C" void glCreateTextures(uint32_t target, int32_t n, uint32_t* textures) { glGenTextures(n,textures); if(metalModeEnabled()&&textures){std::lock_guard<std::mutex> lock(g_resourceMutex);for(int32_t i=0;i<n;++i)g_textures[textures[i]].target=target;} }
extern "C" void glTextureView(uint32_t texture, uint32_t target, uint32_t originalTexture, uint32_t internalFormat, uint32_t minLevel, uint32_t numLevels, uint32_t minLayer, uint32_t numLayers) {
    if(metalModeEnabled()&&minLevel==0&&numLevels>0&&minLayer==0&&numLayers>0){std::lock_guard<std::mutex> lock(g_resourceMutex);auto source=g_textures.find(originalTexture);if(source!=g_textures.end()){auto view=source->second;view.target=target;view.internalFormat=static_cast<int32_t>(internalFormat);if(target==0x0DE1){view.depth=1;}else if(target==0x8C1A)view.depth=std::min(view.depth,numLayers);g_textures[texture]=std::move(view);return;}}
    glDispatch<void,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>("glTextureView",texture,target,originalTexture,internalFormat,minLevel,numLevels,minLayer,numLayers);
}
extern "C" void glBindTextureUnit(uint32_t unit, uint32_t texture) { if(metalModeEnabled()&&unit<g_textureUnits.size())g_textureUnits[unit]=texture; else glDispatch<void,uint32_t,uint32_t>("glBindTextureUnit",unit,texture); }
extern "C" void glTextureStorage2D(uint32_t texture, int32_t levels, uint32_t internalFormat, int32_t width, int32_t height) {
    if(metalModeEnabled()&&levels>0&&width>0&&height>0){std::lock_guard<std::mutex> lock(g_resourceMutex);auto& t=g_textures[texture];t.target=0x0DE1;t.width=width;t.height=height;t.depth=1;t.internalFormat=internalFormat;t.pixels.assign(static_cast<size_t>(width)*height*4,0);t.metalHandle=createTextureFromCanonical(t);if(t.metalHandle)return;}
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t,int32_t>("glTextureStorage2D",texture,levels,internalFormat,width,height);
}
extern "C" void glTextureStorage3D(uint32_t texture, int32_t levels, uint32_t internalFormat, int32_t width, int32_t height, int32_t depth) {
    if(metalModeEnabled()&&levels>0&&width>0&&height>0&&depth>0){std::lock_guard<std::mutex> lock(g_resourceMutex);auto& t=g_textures[texture];t.target=(t.target==0x8C1A)?0x8C1A:0x806F;t.width=width;t.height=height;t.depth=depth;t.internalFormat=internalFormat;t.pixels.assign(static_cast<size_t>(width)*height*depth*4,0);t.metalHandle=t.target==0x8C1A?g_metalRenderer.createTexture2DArray(width,height,depth,t.pixels.data()):g_metalRenderer.createTexture3D(width,height,depth,t.pixels.data());if(t.metalHandle)return;}
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t,int32_t,int32_t>("glTextureStorage3D",texture,levels,internalFormat,width,height,depth);
}
extern "C" void glTextureParameteri(uint32_t texture, uint32_t pname, int32_t param) { if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(texture);if(it!=g_textures.end()){if(pname==0x2801)it->second.minFilter=param;else if(pname==0x2800)it->second.magFilter=param;else if(pname==0x2802)it->second.wrapS=param;else if(pname==0x2803)it->second.wrapT=param;else if(pname>=0x8E42&&pname<=0x8E45){it->second.swizzle[pname-0x8E42]=static_cast<uint32_t>(param);g_metalRenderer.setTextureSwizzle(it->second.metalHandle,it->second.swizzle[0],it->second.swizzle[1],it->second.swizzle[2],it->second.swizzle[3]);}return;}} glDispatch<void,uint32_t,uint32_t,int32_t>("glTextureParameteri",texture,pname,param);}
extern "C" void glTextureSubImage2D(uint32_t texture, int32_t level, int32_t xoffset, int32_t yoffset, int32_t width, int32_t height, uint32_t format, uint32_t type, const void* pixels) {
    if(metalModeEnabled()&&level==0&&width>0&&height>0&&(pixels||g_boundPixelUnpackBuffer)){std::vector<uint8_t> unpacked,converted;const void* uploadPixels=pixels;if(g_boundPixelUnpackBuffer){size_t bytes=pixelUploadBytes(width,height,1,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(pixels,bytes,unpacked)){glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTextureSubImage2D",texture,level,xoffset,yoffset,width,height,format,type,pixels);return;}uploadPixels=unpacked.data();}if(convertPixelsToBGRA(width,height,format,type,uploadPixels,converted,g_glBridge.state().unpackAlignment)){ std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(texture);if(it!=g_textures.end()&&xoffset>=0&&yoffset>=0&&static_cast<uint32_t>(xoffset+width)<=it->second.width&&static_cast<uint32_t>(yoffset+height)<=it->second.height){for(int32_t row=0;row<height;++row)std::memcpy(it->second.pixels.data()+((static_cast<size_t>(yoffset+row)*it->second.width+xoffset)*4),converted.data()+static_cast<size_t>(row)*width*4,static_cast<size_t>(width)*4);it->second.metalHandle=createTextureFromCanonical(it->second);if(it->second.metalHandle)return;}}}
    glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTextureSubImage2D",texture,level,xoffset,yoffset,width,height,format,type,pixels);
}
extern "C" void glTextureSubImage3D(uint32_t texture, int32_t level, int32_t xoffset, int32_t yoffset, int32_t zoffset, int32_t width, int32_t height, int32_t depth, uint32_t format, uint32_t type, const void* pixels) {
    if(metalModeEnabled()&&level==0&&width>0&&height>0&&depth>0&&(pixels||g_boundPixelUnpackBuffer)&&format==0x1908&&type==0x1401){std::vector<uint8_t> unpacked;const void* uploadPixels=pixels;if(g_boundPixelUnpackBuffer){size_t bytes=pixelUploadBytes(width,height,depth,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(pixels,bytes,unpacked)){glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTextureSubImage3D",texture,level,xoffset,yoffset,zoffset,width,height,depth,format,type,pixels);return;}uploadPixels=unpacked.data();}std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(texture);if(it!=g_textures.end()&&it->second.target!=0x0DE1&&xoffset>=0&&yoffset>=0&&zoffset>=0&&static_cast<uint32_t>(xoffset+width)<=it->second.width&&static_cast<uint32_t>(yoffset+height)<=it->second.height&&static_cast<uint32_t>(zoffset+depth)<=it->second.depth){const uint8_t* src=static_cast<const uint8_t*>(uploadPixels);for(int32_t z=0;z<depth;++z)for(int32_t row=0;row<height;++row){size_t dst=((static_cast<size_t>(zoffset+z)*it->second.height+yoffset+row)*it->second.width+xoffset)*4;size_t off=(static_cast<size_t>(z)*height+row)*width*4;std::memcpy(it->second.pixels.data()+dst,src+off,static_cast<size_t>(width)*4);}it->second.metalHandle=it->second.target==0x8C1A?g_metalRenderer.createTexture2DArray(it->second.width,it->second.height,it->second.depth,it->second.pixels.data()):g_metalRenderer.createTexture3D(it->second.width,it->second.height,it->second.depth,it->second.pixels.data());if(it->second.metalHandle)return;}}
    glDispatch<void,uint32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTextureSubImage3D",texture,level,xoffset,yoffset,zoffset,width,height,depth,format,type,pixels);
}
extern "C" void glGetTextureImage(uint32_t texture, int32_t level, uint32_t format, uint32_t type, int32_t bufSize, void* pixels) {
    if (metalModeEnabled() && pixels && bufSize > 0 && g_activeTextureUnit < g_textureUnits.size()) {
        uint32_t target=0x0DE1; { std::lock_guard<std::mutex> lock(g_resourceMutex); auto it=g_textures.find(texture); if(it!=g_textures.end())target=it->second.target; }
        uint32_t saved=g_textureUnits[g_activeTextureUnit]; g_textureUnits[g_activeTextureUnit]=texture; glGetTexImage(target,level,format,type,pixels); g_textureUnits[g_activeTextureUnit]=saved; return;
    }
    glDispatch<void,uint32_t,int32_t,uint32_t,uint32_t,int32_t,void*>("glGetTextureImage",texture,level,format,type,bufSize,pixels);
}
extern "C" void glGenerateTextureMipmap(uint32_t texture) { if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(texture);if(it!=g_textures.end()&&!it->second.pixels.empty()&&it->second.target==0x0DE1){it->second.metalHandle=createTextureFromCanonical(it->second);if(it->second.metalHandle)return;}} glDispatch<void,uint32_t>("glGenerateTextureMipmap",texture); }
extern "C" void glGetTextureLevelParameteriv(uint32_t texture, int32_t level, uint32_t pname, int32_t* params) { if(metalModeEnabled()&&params&&level==0){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(texture);if(it!=g_textures.end()){if(pname==0x1000)*params=it->second.width;else if(pname==0x1001)*params=it->second.height;else if(pname==0x8071)*params=it->second.depth;else if(pname==0x1003)*params=it->second.internalFormat;else *params=0;return;}} glDispatch<void,uint32_t,int32_t,uint32_t,int32_t*>("glGetTextureLevelParameteriv",texture,level,pname,params);}

extern "C" void glDeleteTextures(int32_t n, const uint32_t* textures) {
    if (textures) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_textures.erase(textures[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteTextures", n, textures);
}

extern "C" void glBindTexture(uint32_t target, uint32_t texture) {
    glDispatch<void, uint32_t, uint32_t>("glBindTexture", target, texture);
    if ((target == 0x0DE0 || target == 0x0DE1 || target == 0x84F5 || target == 0x9100 || target == 0x806F || target == 0x8C1A || target == 0x8513 || (target >= 0x8515 && target <= 0x851A)) && g_activeTextureUnit < g_textureUnits.size()) {
        g_textureUnits[g_activeTextureUnit] = texture;
        g_glBridge.state().boundTexture2D = texture;
    }
}

extern "C" void glActiveTexture(uint32_t texture) {
    glDispatch<void, uint32_t>("glActiveTexture", texture);
    if (texture >= 0x84C0 && texture < 0x84C0 + g_textureUnits.size())
        g_activeTextureUnit = texture - 0x84C0;
}

extern "C" void glTexParameteri(uint32_t target, uint32_t pname, int32_t param) {
    glDispatch<void, uint32_t, uint32_t, int32_t>("glTexParameteri", target, pname, param);
    if ((target != 0x0DE0 && target != 0x0DE1 && target != 0x84F5 && target != 0x8513 && (target < 0x8515 || target > 0x851A)) || g_activeTextureUnit >= g_textureUnits.size()) return;
    std::lock_guard<std::mutex> lock(g_resourceMutex);
    auto it = g_textures.find(g_textureUnits[g_activeTextureUnit]);
    if (it == g_textures.end()) return;
    if (pname == 0x2801) it->second.minFilter = static_cast<uint32_t>(param);
    else if (pname == 0x2800) it->second.magFilter = static_cast<uint32_t>(param);
    else if (pname == 0x2802) it->second.wrapS = static_cast<uint32_t>(param);
    else if (pname == 0x2803) it->second.wrapT = static_cast<uint32_t>(param);
    else if (pname == 0x813A) it->second.minLod = static_cast<float>(param);
    else if (pname == 0x813B) it->second.maxLod = static_cast<float>(param);
    else if (pname == 0x84FE) it->second.maxAnisotropy = static_cast<uint32_t>(std::max(1,param));
    else if (pname == 0x884C) it->second.compare = param != 0;
    else if (pname == 0x884D) it->second.compareFunc = static_cast<uint32_t>(param);
    else if (pname >= 0x8E42 && pname <= 0x8E45) it->second.swizzle[pname - 0x8E42] = static_cast<uint32_t>(param);
    if (pname >= 0x8E42 && pname <= 0x8E45) g_metalRenderer.setTextureSwizzle(it->second.metalHandle,it->second.swizzle[0],it->second.swizzle[1],it->second.swizzle[2],it->second.swizzle[3]);
}
extern "C" void glTexParameterf(uint32_t target,uint32_t pname,float param) { glDispatch<void,uint32_t,uint32_t,float>("glTexParameterf",target,pname,param); if(!metalModeEnabled()||(target!=0x0DE1&&target!=0x8513&&(target<0x8515||target>0x851A))||g_activeTextureUnit>=g_textureUnits.size())return;std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(it==g_textures.end())return;if(pname==0x813A)it->second.minLod=param;else if(pname==0x813B)it->second.maxLod=param;else if(pname==0x84FE)it->second.maxAnisotropy=static_cast<uint32_t>(std::max(1.0f,param)); }
extern "C" void glTexParameteriv(uint32_t target,uint32_t pname,const int32_t* params) { if(params)glTexParameteri(target,pname,*params);else glDispatch<void,uint32_t,uint32_t,const int32_t*>("glTexParameteriv",target,pname,params); }
extern "C" void glTexParameterfv(uint32_t target,uint32_t pname,const float* params) { if(!params){glDispatch<void,uint32_t,uint32_t,const float*>("glTexParameterfv",target,pname,params);return;}if(pname==0x1004&&metalModeEnabled()&&(target==0x0DE1||target==0x8513||(target>=0x8515&&target<=0x851A))&&g_activeTextureUnit<g_textureUnits.size()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(it!=g_textures.end())std::memcpy(it->second.borderColor,params,sizeof(it->second.borderColor));return;}glTexParameterf(target,pname,*params); }
extern "C" void glTexSubImage1D(uint32_t target,int32_t level,int32_t xoffset,int32_t width,uint32_t format,uint32_t type,const void* data) { const uint32_t textureName=g_activeTextureUnit<g_textureUnits.size()?g_textureUnits[g_activeTextureUnit]:0;if(metalModeEnabled()&&target==0x0DE0&&level==0&&width>0&&textureName){std::vector<uint8_t> converted,unpacked;const void* source=data;if(g_boundPixelUnpackBuffer){size_t bytes=pixelUploadBytes(width,1,1,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(data,bytes,unpacked)){metalsharp::GLErrorTracker::instance().setError(0x0501);return;}source=unpacked.data();}if(!convertPixelsToBGRA(width,1,format,type,source,converted,g_glBridge.state().unpackAlignment)){metalsharp::GLErrorTracker::instance().setError(0x0500);return;}std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(textureName);if(it!=g_textures.end()&&xoffset>=0&&static_cast<uint32_t>(xoffset+width)<=it->second.width){std::memcpy(it->second.pixels.data()+static_cast<size_t>(xoffset)*4,converted.data(),static_cast<size_t>(width)*4);std::vector<uint8_t> storage=encodeTextureStorage(it->second.pixels,it->second.internalFormat);it->second.metalHandle=g_metalRenderer.createTexture1D(it->second.width,static_cast<uint32_t>(it->second.internalFormat),storage.data(),true);return;}}glDispatch<void,uint32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t,const void*>("glTexSubImage1D",target,level,xoffset,width,format,type,data); }
extern "C" void glTexSubImage2D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset,
                                int32_t width, int32_t height, uint32_t format, uint32_t type, const void* pixels) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && (target == 0x0DE1 || target == 0x84F5 || (target >= 0x8515 && target <= 0x851A)) && level == 0 && width > 0 && height > 0 && (pixels || g_boundPixelUnpackBuffer) && textureName) {
        std::vector<uint8_t> converted, unpacked; const void* uploadPixels=pixels;
        if(g_boundPixelUnpackBuffer){size_t bytes=pixelUploadBytes(width,height,1,format,type,g_glBridge.state().unpackAlignment);if(!readPixelUnpackBuffer(pixels,bytes,unpacked)){glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>("glTexSubImage2D", target, level, xoffset, yoffset, width, height, format, type, pixels);return;}uploadPixels=unpacked.data();}
        if (!convertPixelsToBGRA(width, height, format, type, uploadPixels, converted, g_glBridge.state().unpackAlignment)) { glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>("glTexSubImage2D", target, level, xoffset, yoffset, width, height, format, type, pixels); return; }
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto it = g_textures.find(textureName);
        if (it != g_textures.end() && xoffset >= 0 && yoffset >= 0 &&
            static_cast<uint32_t>(xoffset + width) <= it->second.width &&
            static_cast<uint32_t>(yoffset + height) <= it->second.height) {
            if(it->second.target==0x8513&&target>=0x8515&&target<=0x851A){uint32_t face=target-0x8515;if(it->second.cubePixels[face].empty())it->second.cubePixels[face].assign(static_cast<size_t>(it->second.width)*it->second.height*4,0);for(int32_t row=0;row<height;++row)std::memcpy(it->second.cubePixels[face].data()+((static_cast<size_t>(yoffset+row)*it->second.width+xoffset)*4),converted.data()+static_cast<size_t>(row)*width*4,static_cast<size_t>(width)*4);const void* faces[6]={};for(uint32_t i=0;i<6;++i)if(!it->second.cubePixels[i].empty())faces[i]=it->second.cubePixels[i].data();it->second.metalHandle=g_metalRenderer.createTextureCube(it->second.width,it->second.height,static_cast<uint32_t>(it->second.internalFormat),faces);return;}
            for (int32_t row = 0; row < height; ++row) {
                size_t dst = (static_cast<size_t>(yoffset + row) * it->second.width + xoffset) * 4;
                size_t src = static_cast<size_t>(row) * width * 4;
                std::memcpy(it->second.pixels.data() + dst, converted.data() + src, static_cast<size_t>(width) * 4);
            }
            it->second.metalHandle = createTextureFromCanonical(it->second);
            return;
        }
    }
    glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>(
        "glTexSubImage2D", target, level, xoffset, yoffset, width, height, format, type, pixels);
}
extern "C" void glTexStorage1D(uint32_t target,int32_t levels,uint32_t internalFormat,int32_t width) { if(metalModeEnabled()&&target==0x0DE0&&levels>0&&width>0){glTexImage1D(target,0,static_cast<int32_t>(internalFormat),width,0,0x1908,0x1401,nullptr);return;}glDispatch<void,uint32_t,int32_t,uint32_t,int32_t>("glTexStorage1D",target,levels,internalFormat,width); }
extern "C" void glTexStorage2D(uint32_t target, int32_t levels, uint32_t internalFormat, int32_t width, int32_t height) {
    if (metalModeEnabled() && target == 0x0DE1 && levels > 0 && width > 0 && height > 0) { glTexImage2D(target,0,static_cast<int32_t>(internalFormat),width,height,0,0x1908,0x1401,nullptr); return; }
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t,int32_t>("glTexStorage2D",target,levels,internalFormat,width,height);
}
extern "C" void glTexStorage3D(uint32_t target, int32_t levels, uint32_t internalFormat, int32_t width, int32_t height, int32_t depth) {
    if (metalModeEnabled() && (target == 0x806F || target == 0x8C1A) && levels > 0 && width > 0 && height > 0 && depth > 0) { std::vector<uint8_t> zeros(static_cast<size_t>(width)*height*depth*4,0); glTexImage3D(target,0,static_cast<int32_t>(internalFormat),width,height,depth,0,0x1908,0x1401,zeros.data()); return; }
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t,int32_t,int32_t>("glTexStorage3D",target,levels,internalFormat,width,height,depth);
}
extern "C" void glGenerateMipmap(uint32_t target) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if(metalModeEnabled()&&target==0x8513&&textureName){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(textureName);if(it!=g_textures.end()&&it->second.target==0x8513){const void* faces[6]={};for(uint32_t face=0;face<6;++face)if(!it->second.cubePixels[face].empty())faces[face]=it->second.cubePixels[face].data();it->second.metalHandle=g_metalRenderer.createTextureCube(it->second.width,it->second.height,static_cast<uint32_t>(it->second.internalFormat),faces);if(it->second.metalHandle)return;}}
    if (metalModeEnabled() && target == 0x0DE0 && textureName) { std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(textureName);if(it!=g_textures.end()&&!it->second.pixels.empty()){std::vector<uint8_t> storage=encodeTextureStorage(it->second.pixels,it->second.internalFormat);it->second.metalHandle=g_metalRenderer.createTexture1D(it->second.width,static_cast<uint32_t>(it->second.internalFormat),storage.data(),true);if(it->second.metalHandle)return;} }
    if (metalModeEnabled() && target == 0x0DE1 && textureName) {
        std::lock_guard<std::mutex> lock(g_resourceMutex); auto it=g_textures.find(textureName);
        if (it != g_textures.end() && !it->second.pixels.empty()) { it->second.metalHandle=g_metalRenderer.createTexture(it->second.width,it->second.height,it->second.pixels.data(),true); if(it->second.metalHandle)return; }
    }
    glDispatch<void,uint32_t>("glGenerateMipmap",target);
}
extern "C" void glTexEnvi(uint32_t target, uint32_t pname, int32_t param) { glDispatch<void,uint32_t,uint32_t,int32_t>("glTexEnvi",target,pname,param); if(!metalModeEnabled()||target!=0x2300)return; if(pname==0x2200)g_fixedTextureEnv.mode=static_cast<uint32_t>(param); else if(pname==0x8571)g_fixedTextureEnv.combineRGB=static_cast<uint32_t>(param); else if(pname==0x8572)g_fixedTextureEnv.combineAlpha=static_cast<uint32_t>(param); else if(pname>=0x8580&&pname<=0x8582)g_fixedTextureEnv.sourceRGB[pname-0x8580]=static_cast<uint32_t>(param); else if(pname>=0x8590&&pname<=0x8592)g_fixedTextureEnv.operandRGB[pname-0x8590]=static_cast<uint32_t>(param); else if(pname>=0x8588&&pname<=0x858A)g_fixedTextureEnv.sourceAlpha[pname-0x8588]=static_cast<uint32_t>(param); else if(pname>=0x8598&&pname<=0x859A)g_fixedTextureEnv.operandAlpha[pname-0x8598]=static_cast<uint32_t>(param); }
extern "C" void glTexEnvf(uint32_t target, uint32_t pname, float param) { glDispatch<void,uint32_t,uint32_t,float>("glTexEnvf",target,pname,param); if(metalModeEnabled()&&target==0x2300&&pname==0x2200)g_fixedTextureEnv.mode=static_cast<uint32_t>(param); }
extern "C" void glTexEnvfv(uint32_t target, uint32_t pname, const float* params) { glDispatch<void,uint32_t,uint32_t,const float*>("glTexEnvfv",target,pname,params); if(metalModeEnabled()&&target==0x2300&&pname==0x2201&&params)std::memcpy(g_fixedTextureEnv.constantColor,params,sizeof(g_fixedTextureEnv.constantColor)); }
extern "C" void glTexEnviv(uint32_t target, uint32_t pname, const int32_t* params) { if(params)glTexEnvi(target,pname,*params);else glDispatch<void,uint32_t,uint32_t,const int32_t*>("glTexEnviv",target,pname,params); }
extern "C" void glGetTexLevelParameteriv(uint32_t target, int32_t level, uint32_t pname, int32_t* params) {
    if (metalModeEnabled() && params && level == 0 && g_activeTextureUnit < g_textureUnits.size()) {
        std::lock_guard<std::mutex> lock(g_resourceMutex); auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);
        if (it != g_textures.end() && (target == 0x0DE0 || target == 0x0DE1 || target == 0x806F || target == 0x8C1A || target == 0x9100)) {
            if (pname == 0x1000) *params=static_cast<int32_t>(it->second.width); else if (pname == 0x1001) *params=static_cast<int32_t>(it->second.height); else if (pname == 0x8071) *params=static_cast<int32_t>(it->second.depth); else if (pname == 0x9106) *params=static_cast<int32_t>(it->second.sampleCount); else if (pname == 0x9107) *params=it->second.sampleCount>1; else if (pname == 0x1003) *params=it->second.internalFormat; else { glDispatch<void,uint32_t,int32_t,uint32_t,int32_t*>("glGetTexLevelParameteriv",target,level,pname,params); return; } return;
        }
    }
    glDispatch<void,uint32_t,int32_t,uint32_t,int32_t*>("glGetTexLevelParameteriv",target,level,pname,params);
}
extern "C" void glGetTexParameteriv(uint32_t target,uint32_t pname,int32_t* params) { if(params&&metalModeEnabled()&&(target==0x0DE0||target==0x0DE1||target==0x84F5||target==0x8513||(target>=0x8515&&target<=0x851A))&&g_activeTextureUnit<g_textureUnits.size()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(it!=g_textures.end()){if(pname==0x2801)*params=static_cast<int32_t>(it->second.minFilter);else if(pname==0x2800)*params=static_cast<int32_t>(it->second.magFilter);else if(pname==0x2802)*params=static_cast<int32_t>(it->second.wrapS);else if(pname==0x2803)*params=static_cast<int32_t>(it->second.wrapT);else if(pname==0x813A)*params=static_cast<int32_t>(it->second.minLod);else if(pname==0x813B)*params=static_cast<int32_t>(it->second.maxLod);else if(pname==0x84FE)*params=static_cast<int32_t>(it->second.maxAnisotropy);else if(pname==0x884C)*params=it->second.compare;else if(pname==0x884D)*params=static_cast<int32_t>(it->second.compareFunc);else if(pname>=0x8E42&&pname<=0x8E45)*params=static_cast<int32_t>(it->second.swizzle[pname-0x8E42]);else {glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetTexParameteriv",target,pname,params);return;}return;}}glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetTexParameteriv",target,pname,params); }
extern "C" void glGetTexParameterfv(uint32_t target,uint32_t pname,float* params) { if(params&&metalModeEnabled()&&(target==0x0DE0||target==0x0DE1||target==0x84F5||target==0x8513||(target>=0x8515&&target<=0x851A))&&g_activeTextureUnit<g_textureUnits.size()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_textures.find(g_textureUnits[g_activeTextureUnit]);if(it!=g_textures.end()){if(pname==0x813A)*params=it->second.minLod;else if(pname==0x813B)*params=it->second.maxLod;else if(pname==0x84FE)*params=static_cast<float>(it->second.maxAnisotropy);else if(pname==0x1004){std::memcpy(params,it->second.borderColor,sizeof(it->second.borderColor));return;}else {glDispatch<void,uint32_t,uint32_t,float*>("glGetTexParameterfv",target,pname,params);return;}return;}}glDispatch<void,uint32_t,uint32_t,float*>("glGetTexParameterfv",target,pname,params); }
GL_PASSTHROUGH1(unsigned char, glIsTexture, uint32_t, texture)

// glCopyTexImage2D and glCopyTexSubImage2D update tracked Metal textures
// from the same readback source used by glReadPixels.
extern "C" void glCopyTexSubImage2D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset, int32_t x,
                                    int32_t y, int32_t width, int32_t height);
extern "C" void glCopyTexImage2D(uint32_t target, int32_t level, uint32_t internalformat, int32_t x, int32_t y,
                                 int32_t width, int32_t height, int32_t border) {
    if (metalModeEnabled() && target == 0x0DE1 && level == 0 && width > 0 && height > 0) {
        glTexImage2D(target, level, static_cast<int32_t>(internalformat), width, height, border, 0x1908, 0x1401, nullptr);
        glCopyTexSubImage2D(target, level, 0, 0, x, y, width, height);
        return;
    }
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glCopyTexImage2D"));
    if (fn) fn(target, level, internalformat, x, y, width, height, border);
}

extern "C" void glCopyTexSubImage2D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset, int32_t x,
                                    int32_t y, int32_t width, int32_t height) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && target == 0x0DE1 && level == 0 && textureName && width > 0 && height > 0 && xoffset >= 0 && yoffset >= 0) {
        std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
        glReadPixels(x, y, width, height, 0x1908, 0x1401, rgba.data());
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto it = g_textures.find(textureName);
        if (it != g_textures.end() && static_cast<uint32_t>(xoffset + width) <= it->second.width && static_cast<uint32_t>(yoffset + height) <= it->second.height) {
            for (int32_t row=0; row<height; ++row) for (int32_t column=0; column<width; ++column) {
                size_t source=(static_cast<size_t>(row)*width+column)*4;
                size_t destination=((static_cast<size_t>(yoffset+row)*it->second.width+xoffset+column)*4);
                it->second.pixels[destination+0]=rgba[source+2]; it->second.pixels[destination+1]=rgba[source+1]; it->second.pixels[destination+2]=rgba[source+0]; it->second.pixels[destination+3]=rgba[source+3];
            }
            it->second.metalHandle=createTextureFromCanonical(it->second);
            if (it->second.metalHandle) return;
        }
    }
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glCopyTexSubImage2D"));
    if (fn) fn(target, level, xoffset, yoffset, x, y, width, height);
}

// ---------------------------------------------------------------------------
// Renderbuffer objects (GL 3.0 / EXT)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH1(unsigned char, glIsRenderbuffer, uint32_t, renderbuffer)
extern "C" void glGetRenderbufferParameteriv(uint32_t target,uint32_t pname,int32_t* params) { if(params&&metalModeEnabled()&&target==0x8D41){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_renderbuffers.find(g_boundRenderbuffer);if(it!=g_renderbuffers.end()){if(pname==0x8D42)*params=it->second.width;else if(pname==0x8D43)*params=it->second.height;else if(pname==0x8D44)*params=it->second.internalFormat;else if(pname==0x8D57)*params=it->second.sampleCount;else {*params=0;}return;}}glDispatch<void,uint32_t,uint32_t,int32_t*>("glGetRenderbufferParameteriv",target,pname,params); }

// ---------------------------------------------------------------------------
// Framebuffer objects (GL 3.0 / EXT_framebuffer_object)
// ---------------------------------------------------------------------------
extern "C" void glGenFramebuffers(int32_t n, uint32_t* framebuffers) {
    glDispatch<void, int32_t, uint32_t*>("glGenFramebuffers", n, framebuffers);
    if (metalModeEnabled() && framebuffers) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_framebuffers.emplace(framebuffers[i], ExperimentalFramebuffer{});
    }
}
extern "C" void glCreateFramebuffers(int32_t n, uint32_t* framebuffers) { glGenFramebuffers(n,framebuffers); }
extern "C" void glNamedFramebufferTexture(uint32_t framebuffer, uint32_t attachment, uint32_t texture, int32_t level) {
    if(metalModeEnabled()&&level==0){std::lock_guard<std::mutex> lock(g_resourceMutex);auto& fbo=g_framebuffers[framebuffer];auto image=g_textures.find(texture);if(attachment==0x8CE0){fbo.colorTexture=texture;fbo.colorHandle=image==g_textures.end()?0:image->second.metalHandle;if(image!=g_textures.end()){fbo.width=image->second.width;fbo.height=image->second.height;fbo.sampleCount=image->second.sampleCount;}}else if(attachment==0x8D00||attachment==0x821A){fbo.depthHandle=image==g_textures.end()?0:image->second.metalHandle;if(image!=g_textures.end())fbo.depthSampleCount=image->second.sampleCount;if(attachment==0x8D00)fbo.depthTexture=texture;if(attachment==0x821A){fbo.stencilHandle=fbo.depthHandle;fbo.stencilTexture=texture;}}return;}
    glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t>("glNamedFramebufferTexture",framebuffer,attachment,texture,level);
}
extern "C" void glGetNamedFramebufferAttachmentParameteriv(uint32_t framebuffer,uint32_t attachment,uint32_t pname,int32_t* params) { if(!params)return;if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_framebuffers.find(framebuffer);if(it!=g_framebuffers.end()){uint32_t texture=attachment==0x8CE0?it->second.colorTexture:attachment==0x8D00?it->second.depthTexture:attachment==0x8D20?it->second.stencilTexture:0;uint32_t renderbuffer=attachment==0x8CE0?it->second.renderbuffer:attachment==0x8D00?it->second.depthRenderbuffer:attachment==0x8D20?it->second.stencilRenderbuffer:0;if(pname==0x8CD0)*params=texture?0x1702:renderbuffer?0x8D41:0;else if(pname==0x8CD1)*params=texture?static_cast<int32_t>(texture):static_cast<int32_t>(renderbuffer);else if(pname==0x8CD2||pname==0x8CD3)*params=0;else if(pname==0x8CD4)*params=static_cast<int32_t>(it->second.colorLayer);else *params=0;return;}}glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t*>("glGetNamedFramebufferAttachmentParameteriv",framebuffer,attachment,pname,params); }
extern "C" void glGetFramebufferAttachmentParameteriv(uint32_t target,uint32_t attachment,uint32_t pname,int32_t* params) { uint32_t framebuffer=g_glBridge.state().boundFramebuffer;glGetNamedFramebufferAttachmentParameteriv(framebuffer,attachment,pname,params); }
extern "C" void glNamedFramebufferRenderbuffer(uint32_t framebuffer, uint32_t attachment, uint32_t renderbufferTarget, uint32_t renderbuffer) {
    if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto& fbo=g_framebuffers[framebuffer];auto rb=g_renderbuffers.find(renderbuffer);if(attachment==0x8CE0){fbo.renderbuffer=renderbuffer;fbo.colorHandle=rb==g_renderbuffers.end()?0:rb->second.metalHandle;if(rb!=g_renderbuffers.end()){fbo.width=rb->second.width;fbo.height=rb->second.height;fbo.sampleCount=rb->second.sampleCount;}}else if(attachment==0x8D00||attachment==0x821A){fbo.depthHandle=rb==g_renderbuffers.end()?0:rb->second.metalHandle;if(rb!=g_renderbuffers.end())fbo.depthSampleCount=rb->second.sampleCount;if(attachment==0x8D00)fbo.depthRenderbuffer=renderbuffer;if(attachment==0x821A){fbo.stencilHandle=fbo.depthHandle;fbo.stencilRenderbuffer=renderbuffer;}}return;}
    glDispatch<void,uint32_t,uint32_t,uint32_t,uint32_t>("glNamedFramebufferRenderbuffer",framebuffer,attachment,renderbufferTarget,renderbuffer);
}
extern "C" uint32_t glCheckNamedFramebufferStatus(uint32_t framebuffer, uint32_t target) { if(metalModeEnabled()){std::lock_guard<std::mutex> lock(g_resourceMutex);auto it=g_framebuffers.find(framebuffer);return it!=g_framebuffers.end()&&it->second.colorHandle&&(!it->second.depthHandle||it->second.sampleCount==it->second.depthSampleCount)?0x8CD5:0x8CD7;} return glDispatch<uint32_t,uint32_t,uint32_t>("glCheckNamedFramebufferStatus",framebuffer,target); }

extern "C" void glDeleteFramebuffers(int32_t n, const uint32_t* framebuffers) {
    if (framebuffers) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_framebuffers.erase(framebuffers[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteFramebuffers", n, framebuffers);
}
extern "C" void glFramebufferTexture2D(uint32_t target, uint32_t attachment, uint32_t textarget,
                                        uint32_t texture, int32_t level) {
    glDispatch<void, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(
        "glFramebufferTexture2D", target, attachment, textarget, texture, level);
    if (metalModeEnabled() && target == 0x8D40 && (textarget == 0x0DE1 || textarget == 0x84F5 || textarget == 0x9100)) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto& fbo = g_framebuffers[g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer];
        auto image = g_textures.find(texture);
        if (attachment == 0x8CE0) { fbo.colorTexture=texture; if (image != g_textures.end()) { fbo.colorHandle=image->second.metalHandle; fbo.width=image->second.width; fbo.height=image->second.height; fbo.sampleCount=image->second.sampleCount; } }
        else if (attachment == 0x8D00 || attachment == 0x821A) { fbo.depthHandle=image == g_textures.end() ? 0 : image->second.metalHandle; if(image!=g_textures.end())fbo.depthSampleCount=image->second.sampleCount; if(attachment==0x8D00)fbo.depthTexture=texture; if(attachment==0x821A){fbo.stencilHandle=fbo.depthHandle;fbo.stencilTexture=texture;} }
    }
}
extern "C" void glFramebufferTextureLayer(uint32_t target, uint32_t attachment, uint32_t texture, int32_t level, int32_t layer) {
    glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t,int32_t>("glFramebufferTextureLayer",target,attachment,texture,level,layer);
    if (metalModeEnabled() && target == 0x8D40 && attachment == 0x8CE0 && level == 0 && layer >= 0) {
        std::lock_guard<std::mutex> lock(g_resourceMutex); auto& fbo=g_framebuffers[g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer]; auto image=g_textures.find(texture);
        fbo.colorTexture=texture; fbo.colorLayer=static_cast<uint32_t>(layer); if(image!=g_textures.end()){fbo.colorHandle=image->second.metalHandle;fbo.width=image->second.width;fbo.height=image->second.height;fbo.sampleCount=image->second.sampleCount;}
    }
}
extern "C" void glGenRenderbuffers(int32_t n, uint32_t* renderbuffers) {
    glDispatch<void, int32_t, uint32_t*>("glGenRenderbuffers", n, renderbuffers);
    if (metalModeEnabled() && renderbuffers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i=0;i<n;++i) g_renderbuffers.emplace(renderbuffers[i], ExperimentalRenderbuffer{}); }
}
extern "C" void glDeleteRenderbuffers(int32_t n, const uint32_t* renderbuffers) {
    if (renderbuffers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i=0;i<n;++i) g_renderbuffers.erase(renderbuffers[i]); }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteRenderbuffers", n, renderbuffers);
}
extern "C" void glBindRenderbuffer(uint32_t target, uint32_t renderbuffer) {
    glDispatch<void, uint32_t, uint32_t>("glBindRenderbuffer", target, renderbuffer);
    if (target == 0x8D41) g_boundRenderbuffer = renderbuffer;
}
extern "C" void glRenderbufferStorage(uint32_t target, uint32_t internalformat, int32_t width, int32_t height) {
    glDispatch<void, uint32_t, uint32_t, int32_t, int32_t>("glRenderbufferStorage", target, internalformat, width, height);
    if (metalModeEnabled() && target == 0x8D41 && g_boundRenderbuffer && width > 0 && height > 0) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto& rb = g_renderbuffers[g_boundRenderbuffer]; rb.width=width; rb.height=height; rb.internalFormat=internalformat; rb.sampleCount=1;
        if (internalformat == 0x81A5 || internalformat == 0x81A6 || internalformat == 0x88F0 || internalformat == 0x8D48) rb.metalHandle = g_metalRenderer.createDepthStencilTarget(width,height,internalformat);
        else { std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4u, 0); rb.metalHandle = g_metalRenderer.createTexture(width, height, pixels.data(), false); }
    }
}
extern "C" void glRenderbufferStorageMultisample(uint32_t target, int32_t samples, uint32_t internalformat, int32_t width, int32_t height) {
    glDispatch<void, uint32_t, int32_t, uint32_t, int32_t, int32_t>("glRenderbufferStorageMultisample", target, samples, internalformat, width, height);
    if (metalModeEnabled() && target == 0x8D41 && g_boundRenderbuffer && width > 0 && height > 0) {
        if (internalformat == 0x81A5 || internalformat == 0x81A6 || internalformat == 0x88F0 || internalformat == 0x8D48) { std::lock_guard<std::mutex> lock(g_resourceMutex);auto& rb=g_renderbuffers[g_boundRenderbuffer];rb.width=width;rb.height=height;rb.internalFormat=internalformat;rb.sampleCount=static_cast<uint32_t>(std::max(1,samples));rb.metalHandle=g_metalRenderer.createMultisampleDepthStencilTarget(width,height,internalformat,rb.sampleCount);return; }
        std::lock_guard<std::mutex> lock(g_resourceMutex);auto& rb=g_renderbuffers[g_boundRenderbuffer];rb.width=width;rb.height=height;rb.internalFormat=internalformat;rb.sampleCount=static_cast<uint32_t>(std::max(1,samples));rb.metalHandle=g_metalRenderer.createMultisampleTexture2D(width,height,internalformat,rb.sampleCount);return;
    }
}
extern "C" void glTexImage2DMultisample(uint32_t target, int32_t samples, uint32_t internalformat,
                                         int32_t width, int32_t height, unsigned char fixedsamplelocations) {
    glDispatch<void, uint32_t, int32_t, uint32_t, int32_t, int32_t, unsigned char>("glTexImage2DMultisample", target, samples, internalformat, width, height, fixedsamplelocations);
    const uint32_t textureName=g_activeTextureUnit<g_textureUnits.size()?g_textureUnits[g_activeTextureUnit]:0;
    if(metalModeEnabled()&&target==0x9100&&width>0&&height>0&&textureName){std::lock_guard<std::mutex> lock(g_resourceMutex);auto& texture=g_textures[textureName];texture.target=target;texture.width=width;texture.height=height;texture.depth=1;texture.internalFormat=internalformat;texture.sampleCount=static_cast<uint32_t>(std::max(1,samples));texture.pixels.assign(static_cast<size_t>(width)*height*4,0);bool depth=(internalformat==0x1902||internalformat==0x81A5||internalformat==0x81A6||internalformat==0x8CAC||internalformat==0x8D48);texture.metalHandle=depth?g_metalRenderer.createMultisampleDepthStencilTarget(width,height,internalformat,texture.sampleCount):g_metalRenderer.createMultisampleTexture2D(width,height,internalformat,texture.sampleCount);}
}
extern "C" void glFramebufferRenderbuffer(uint32_t target, uint32_t attachment, uint32_t renderbuffertarget, uint32_t renderbuffer) {
    glDispatch<void, uint32_t, uint32_t, uint32_t, uint32_t>("glFramebufferRenderbuffer", target, attachment, renderbuffertarget, renderbuffer);
    if (metalModeEnabled() && target == 0x8D40 && renderbuffertarget == 0x8D41) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto rb = g_renderbuffers.find(renderbuffer); auto& fbo = g_framebuffers[g_glBridge.state().boundDrawFramebuffer ? g_glBridge.state().boundDrawFramebuffer : g_glBridge.state().boundFramebuffer];
        if (attachment == 0x8CE0) { fbo.renderbuffer=renderbuffer; fbo.colorHandle = rb == g_renderbuffers.end() ? 0 : rb->second.metalHandle; if (rb != g_renderbuffers.end()) { fbo.width=rb->second.width; fbo.height=rb->second.height; fbo.sampleCount=rb->second.sampleCount; } }
        else if (attachment == 0x8D00 || attachment == 0x821A) { fbo.depthRenderbuffer=renderbuffer; fbo.depthHandle=rb == g_renderbuffers.end() ? 0 : rb->second.metalHandle; if(rb!=g_renderbuffers.end())fbo.depthSampleCount=rb->second.sampleCount; }
        if (attachment == 0x8D20 || attachment == 0x821A) { fbo.stencilRenderbuffer=renderbuffer; fbo.stencilHandle=rb == g_renderbuffers.end() ? 0 : rb->second.metalHandle; if (!fbo.depthHandle) fbo.depthHandle=fbo.stencilHandle; }
    }
}
extern "C" uint32_t glCheckFramebufferStatus(uint32_t target) {
    if (metalModeEnabled() && target == 0x8D40) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        const uint32_t framebuffer = target == 0x8CA8 ? g_glBridge.state().boundReadFramebuffer :
                                     target == 0x8CA9 ? g_glBridge.state().boundDrawFramebuffer :
                                     g_glBridge.state().boundFramebuffer;
        if (!framebuffer) return 0x8CD5;
        auto fbo = g_framebuffers.find(framebuffer);
        if (fbo != g_framebuffers.end() && fbo->second.colorHandle && (!fbo->second.depthHandle || fbo->second.sampleCount == fbo->second.depthSampleCount)) return 0x8CD5;
        return 0x8CD7;
    }
    return glDispatch<uint32_t, uint32_t>("glCheckFramebufferStatus", target);
}
GL_PASSTHROUGH1(unsigned char, glIsFramebuffer, uint32_t, framebuffer)
extern "C" void glBlitFramebuffer(int32_t srcX0, int32_t srcY0, int32_t srcX1, int32_t srcY1,
                                   int32_t dstX0, int32_t dstY0, int32_t dstX1, int32_t dstY1,
                                   uint32_t mask, uint32_t filter) {
    if (metalModeEnabled() && srcX0 == 0 && srcY0 == 0 && dstX0 == 0 && dstY0 == 0 &&
        srcX1 == dstX1 && srcY1 == dstY1 && srcX1 > 0 && srcY1 > 0) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto src = g_framebuffers.find(g_glBridge.state().boundReadFramebuffer);
        auto dst = g_framebuffers.find(g_glBridge.state().boundDrawFramebuffer);
        if (src != g_framebuffers.end() && dst != g_framebuffers.end() && src->second.colorHandle && dst->second.colorHandle &&
            g_metalRenderer.blitTexture(src->second.colorHandle, dst->second.colorHandle, srcX1, srcY1)) return;
    }
    glDispatch<void, int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,uint32_t,uint32_t>(
        "glBlitFramebuffer", srcX0,srcY0,srcX1,srcY1,dstX0,dstY0,dstX1,dstY1,mask,filter);
}

// glBindFramebuffer is hand-written because it must mirror the binding into
// GLState so subsequent framebuffer attachment calls can observe which
// framebuffer is currently bound. The native call is still issued so the
// framework context state stays in sync with the shim's view.
extern "C" void glBindFramebuffer(uint32_t target, uint32_t framebuffer) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glBindFramebuffer"));
    if (fn) fn(target, framebuffer);
    if (target == 0x8D40) {
        g_glBridge.state().boundFramebuffer = framebuffer;
        g_glBridge.state().boundReadFramebuffer = framebuffer;
        g_glBridge.state().boundDrawFramebuffer = framebuffer;
    } else if (target == 0x8CA8) g_glBridge.state().boundReadFramebuffer = framebuffer;
    else if (target == 0x8CA9) g_glBridge.state().boundDrawFramebuffer = framebuffer;
    if (metalModeEnabled() && framebuffer) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        g_framebuffers.try_emplace(framebuffer, ExperimentalFramebuffer{});
    }
}

// ---------------------------------------------------------------------------
// Info queries (return non-default values — declared by hand instead of
// via GL_PASSTHROUGH).
// glGetString is hand-written for GL_EXTENSIONS passthrough (Phase 5b).
// ---------------------------------------------------------------------------
extern "C" void glDebugMessageControl(uint32_t source,uint32_t type,uint32_t severity,int32_t count,const uint32_t* ids,unsigned char enabled) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint32_t,int32_t,const uint32_t*,unsigned char>("glDebugMessageControl",source,type,severity,count,ids,enabled);return;}std::lock_guard<std::mutex> lock(g_debugMutex);g_debugOutputEnabled=enabled!=0; }
extern "C" void glDebugMessageCallback(void(*callback)(uint32_t,uint32_t,uint32_t,uint32_t,int32_t,const char*,const void*),const void* userParam) { if(!metalModeEnabled()){glDispatch<void,void(*)(uint32_t,uint32_t,uint32_t,uint32_t,int32_t,const char*,const void*),const void*>("glDebugMessageCallback",callback,userParam);return;}std::lock_guard<std::mutex> lock(g_debugMutex);g_debugCallback=callback;g_debugUserParam=userParam; }
extern "C" void glDebugMessageInsert(uint32_t source,uint32_t type,uint32_t id,uint32_t severity,int32_t length,const char* message) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,uint32_t,uint32_t,int32_t,const char*>("glDebugMessageInsert",source,type,id,severity,length,message);return;}if(!message)return;ExperimentalDebugMessage item;item.source=source;item.type=type;item.id=id;item.severity=severity;item.text.assign(message,length<0?std::strlen(message):static_cast<size_t>(length));DebugCallback callback;const void* userParam;{std::lock_guard<std::mutex> lock(g_debugMutex);if(!g_debugOutputEnabled)return;g_debugMessages.push_back(item);callback=g_debugCallback;userParam=g_debugUserParam;}if(callback)callback(source,type,id,severity,static_cast<int32_t>(item.text.size()),item.text.c_str(),userParam); }
extern "C" uint32_t glGetDebugMessageLog(uint32_t count,int32_t bufSize,uint32_t* sources,uint32_t* types,uint32_t* ids,uint32_t* severities,int32_t* lengths,char* messageLog) { if(!metalModeEnabled())return glDispatch<uint32_t,uint32_t,int32_t,uint32_t*,uint32_t*,uint32_t*,uint32_t*,int32_t*,char*>("glGetDebugMessageLog",count,bufSize,sources,types,ids,severities,lengths,messageLog);std::lock_guard<std::mutex> lock(g_debugMutex);uint32_t returned=0;int32_t used=0;while(returned<count&&!g_debugMessages.empty()){auto& item=g_debugMessages.front();int32_t available=static_cast<int32_t>(item.text.size());if(messageLog&&bufSize>used){int32_t copy=std::min(available,bufSize-used-1);if(copy>0)std::memcpy(messageLog+used,item.text.data(),copy);messageLog[used+std::max(0,copy)]=0;if(lengths)lengths[returned]=copy;used+=copy+1;}else if(lengths)lengths[returned]=available;if(sources)sources[returned]=item.source;if(types)types[returned]=item.type;if(ids)ids[returned]=item.id;if(severities)severities[returned]=item.severity;g_debugMessages.pop_front();++returned;}return returned; }
extern "C" void glPushDebugGroup(uint32_t source,uint32_t id,int32_t length,const char* message) { glDebugMessageInsert(source,0x824C,id,0x826B,length,message); }
extern "C" void glPopDebugGroup(void) { }
extern "C" void glObjectLabel(uint32_t identifier,uint32_t name,int32_t length,const char* label) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,int32_t,const char*>("glObjectLabel",identifier,name,length,label);return;}std::lock_guard<std::mutex> lock(g_debugMutex);uint64_t key=(static_cast<uint64_t>(identifier)<<32)|name;if(!label){g_objectLabels.erase(key);return;}g_objectLabels[key]=std::string(label,length<0?std::strlen(label):static_cast<size_t>(length)); }
extern "C" void glGetObjectLabel(uint32_t identifier,uint32_t name,int32_t bufSize,int32_t* length,char* label) { if(!metalModeEnabled()){glDispatch<void,uint32_t,uint32_t,int32_t,int32_t*,char*>("glGetObjectLabel",identifier,name,bufSize,length,label);return;}std::lock_guard<std::mutex> lock(g_debugMutex);uint64_t key=(static_cast<uint64_t>(identifier)<<32)|name;auto it=g_objectLabels.find(key);if(bufSize>0&&label){int32_t copy=it==g_objectLabels.end()?0:std::min<int32_t>(bufSize-1,static_cast<int32_t>(it->second.size()));if(copy)std::memcpy(label,it->second.data(),copy);label[copy]=0;if(length)*length=copy;}else if(length)*length=0; }
extern "C" int metalsharp_opengl_modern_context_ready(void);
extern "C" const uint8_t* glGetString(uint32_t name) {
    if (metalsharp_opengl_modern_context_ready() && name == 0x1F00) return (const uint8_t*)"MetalSharp";
    if (metalsharp_opengl_modern_context_ready() && name == 0x1F01) return (const uint8_t*)"Apple M4 Metal (WineMetalGL)";
    if (metalsharp_opengl_modern_context_ready() && name == 0x1F02) return (const uint8_t*)"3.3 WineMetalGL";
    if (name == 0x1F03) return glGetString_EXTENSIONS_override(name);
    return reinterpret_cast<const uint8_t*>("");
}

extern "C" void glGetIntegerv(uint32_t pname, int32_t* params) {
    ensureGLInit();
    if (!params) return;
    if (metalModeEnabled()) {
        switch (pname) {
        case 0x821B: *params=3; return; /* GL_MAJOR_VERSION */
        case 0x821C: *params=3; return; /* GL_MINOR_VERSION */
        case 0x821D: *params=static_cast<int32_t>(g_experimentalExtensionCount); return; /* GL_NUM_EXTENSIONS */
        case 0x8869: *params=16; return; /* GL_MAX_VERTEX_ATTRIBS */
        case 0x8872: case 0x8B4D: *params=16; return; /* texture units */
        case 0x8A2F: case 0x90DD: *params=16; return; /* UBO/SSBO bindings */
        case 0x8824: *params=1; return; /* GL_MAX_DRAW_BUFFERS */
        case 0x8D57: *params=4; return; /* GL_MAX_SAMPLES */
        case 0x0BA0: *params=static_cast<int32_t>(g_fixedMatrixMode); return; /* GL_MATRIX_MODE */
        case 0x0BA2: {float values[4];experimentalGetFloatValues(pname,values);for(int i=0;i<4;++i)params[i]=static_cast<int32_t>(values[i]);return;}
        case 0x0C10: {float values[4];experimentalGetFloatValues(pname,values);for(int i=0;i<4;++i)params[i]=static_cast<int32_t>(values[i]);return;}
        case 0x0B70: {float values[2];experimentalGetFloatValues(pname,values);for(int i=0;i<2;++i)params[i]=static_cast<int32_t>(values[i]);return;}
        case 0x8B8D: *params=static_cast<int32_t>(g_glBridge.state().currentProgram); return;
        case 0x85B5: *params=static_cast<int32_t>(g_glBridge.state().currentProgramPipeline); return;
        case 0x8894: *params=static_cast<int32_t>(g_glBridge.state().boundArrayBuffer); return;
        case 0x8895: *params=static_cast<int32_t>(g_glBridge.state().boundElementArrayBuffer); return;
        case 0x8069: *params=static_cast<int32_t>(g_glBridge.state().boundTexture2D); return;
        case 0x8CA6: *params=static_cast<int32_t>(g_glBridge.state().boundFramebuffer); return;
        case 0x0B44: *params=g_glBridge.state().cullEnabled; return;
        case 0x0BE2: *params=g_glBridge.state().blendEnabled; return;
        case 0x0B71: *params=g_glBridge.state().depthTestEnabled; return;
        case 0x0C11: *params=g_glBridge.state().scissorEnabled; return;
        case 0x0BC0: *params=g_fixedAlphaEnabled; return;
        case 0x8038: *params=static_cast<int32_t>(g_glBridge.state().polygonOffsetFactor); return;
        case 0x2A00: *params=static_cast<int32_t>(g_glBridge.state().polygonOffsetUnits); return;
        case 0x935C: *params=static_cast<int32_t>(g_glBridge.state().clipOrigin); return;
        case 0x935D: *params=static_cast<int32_t>(g_glBridge.state().clipDepthMode); return;
        default: break;
        }
    }
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t*)>(g_glBridge.getGLProcAddress("glGetIntegerv"));
    if (fn) {
        fn(pname, params);
    } else {
        // Best-effort default. Return zero so callers don't read garbage.
        *params = 0;
    }
}

/* Private ABI used by the Wine macOS driver.  The driver owns the Cocoa
 * view and creates the CAMetalLayer through its normal view helpers; the
 * sidecar only receives the opaque layer pointer and therefore does not
 * depend on Wine's private Objective-C classes. */
extern "C" void metalsharp_opengl_set_metal_layer(void* layer) {
    g_modernContextReady = true;
    if (ensureMetalInit()) g_metalRenderer.setMetalLayer(layer);
}

extern "C" int metalsharp_opengl_modern_context_ready(void) {
    return g_modernContextReady && metalModeEnabled();
}
extern "C" void metalsharp_opengl_set_current_context(void* context) {
    if(!metalModeEnabled())return;
    std::lock_guard<std::mutex> lock(g_contextStateMapMutex);
    if(context==g_currentContextKey)return;
    if(g_currentContextKey)g_contextStates[g_currentContextKey]=g_glBridge.state();
    g_currentContextKey=context;
    if(context){auto it=g_contextStates.find(context);if(it!=g_contextStates.end())g_glBridge.state()=it->second;else {g_glBridge.state()=metalsharp::GLState{};g_contextStates.emplace(context,g_glBridge.state());}}
}

extern "C" int metalsharp_opengl_is_drawable_backed(void) {
    return ensureMetalInit() && g_metalRenderer.isDrawableBacked();
}
