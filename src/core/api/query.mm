// Query objects, conditional rendering and sync objects (GL 3.3 sections
// 2.14, 2.15, 4.1.6, 5.2 and 6.1.7-6.1.8).
//
// Occlusion queries count samples with Metal visibility results; timer
// queries use command-buffer GPU times, so beginning or ending one commits
// the current command buffer. PRIMITIVES_GENERATED is counted on the CPU
// from the draw parameters.

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <tuple>
#include <unistd.h>

#include "core/context.h"
#include "core/draw.h"
#include "core/metal/device.h"
#include "core/query.h"
#include "core/stats.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl {

static_assert(std::tuple_size<decltype(ContextState::activeQueries)>::value == kQuerySlotCount);

int queryTargetIndex(Context& ctx, GLenum target) {
    if (ctx.isEs()) {
        // OpenGL ES's (and EXT_disjoint_timer_query's TIME_ELAPSED).
        switch (target) {
        case GL_ANY_SAMPLES_PASSED: return kQueryAnySamplesPassed;
        case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: return kQueryAnySamplesPassedConservative;
        case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: return kQueryXfbPrimitivesWritten;
        case GL_TIME_ELAPSED: return kQueryTimeElapsed;
        case GL_PRIMITIVES_GENERATED: return ctx.esVersion() >= 32 ? kQueryPrimitivesGenerated : -1;
        default: return -1;
        }
    }
    const bool overflow = ctx.version() >= 46 || ctx.hasExtension("GL_ARB_transform_feedback_overflow_query");
    const int statistics = ctx.version() >= 46 || ctx.hasExtension("GL_ARB_pipeline_statistics_query") ? kQueryStatisticsFirst : -1;
    auto statistic = [&](PipelineStatistic s) { return statistics < 0 ? -1 : statistics + s; };
    switch (target) {
    case GL_SAMPLES_PASSED: return kQuerySamplesPassed;
    case GL_ANY_SAMPLES_PASSED: return kQueryAnySamplesPassed;
    case GL_PRIMITIVES_GENERATED: return kQueryPrimitivesGenerated;
    case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: return kQueryXfbPrimitivesWritten;
    case GL_TIME_ELAPSED: return kQueryTimeElapsed;
    case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: return ctx.version() >= 43 ? kQueryAnySamplesPassedConservative : -1;
    case GL_TRANSFORM_FEEDBACK_OVERFLOW: return overflow ? kQueryXfbOverflow : -1;
    case GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW: return overflow ? kQueryXfbStreamOverflow : -1;
    case GL_VERTICES_SUBMITTED: return statistic(kStatVerticesSubmitted);
    case GL_PRIMITIVES_SUBMITTED: return statistic(kStatPrimitivesSubmitted);
    case GL_VERTEX_SHADER_INVOCATIONS: return statistic(kStatVertexShaderInvocations);
    case GL_TESS_CONTROL_SHADER_PATCHES: return statistic(kStatTessControlShaderPatches);
    case GL_TESS_EVALUATION_SHADER_INVOCATIONS: return statistic(kStatTessEvaluationShaderInvocations);
    case GL_GEOMETRY_SHADER_INVOCATIONS: return statistic(kStatGeometryShaderInvocations);
    case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED: return statistic(kStatGeometryShaderPrimitivesEmitted);
    case GL_FRAGMENT_SHADER_INVOCATIONS: return statistic(kStatFragmentShaderInvocations);
    case GL_COMPUTE_SHADER_INVOCATIONS: return statistic(kStatComputeShaderInvocations);
    case GL_CLIPPING_INPUT_PRIMITIVES: return statistic(kStatClippingInputPrimitives);
    case GL_CLIPPING_OUTPUT_PRIMITIVES: return statistic(kStatClippingOutputPrimitives);
    default: return -1;
    }
}

int queryTargetIndex(Context& ctx, GLenum target, GLuint stream) {
    const int index = queryTargetIndex(ctx, target);
    if (index < 0 || stream == 0) return index;
    if (stream >= static_cast<GLuint>(kMaxVertexStreams)) return -2;
    if (index == kQueryPrimitivesGenerated) return kQueryStreamFirst + static_cast<int>(stream) - 1;
    if (index == kQueryXfbPrimitivesWritten) return kQueryStreamFirst + 3 + static_cast<int>(stream) - 1;
    if (index == kQueryXfbStreamOverflow) return kQueryStreamFirst + 6 + static_cast<int>(stream) - 1;
    return -2;
}

namespace {

bool isOcclusion(int index) {
    return index == kQuerySamplesPassed || index == kQueryAnySamplesPassed || index == kQueryAnySamplesPassedConservative;
}

bool anyOcclusionActive(Context& ctx) {
    for (int i : {kQuerySamplesPassed, kQueryAnySamplesPassed, kQueryAnySamplesPassedConservative})
        if (ctx.state.activeQueries[i]) return true;
    return false;
}

void finalize(Query& q) {
    switch (q.target) {
    case GL_SAMPLES_PASSED: case GL_ANY_SAMPLES_PASSED: case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: {
        uint64_t samples = 0;
        for (const auto& [buffer, offset] : q.visibilitySlots) {
            uint64_t value;
            std::memcpy(&value, static_cast<const uint8_t*>(buffer.contents) + offset, sizeof(value));
            samples += value;
        }
        q.result = q.target == GL_SAMPLES_PASSED ? samples : (samples ? 1 : 0);
        if (logEnabled(LogLevel::Trace, LogCategory::Draw)) {
            WMGL_TRACE(Draw, "occlusion query: %zu slots, %llu samples", q.visibilitySlots.size(),
                       static_cast<unsigned long long>(samples));
            for (size_t i = 0; i < q.visibilitySlots.size(); ++i) {
                uint64_t value;
                std::memcpy(&value, static_cast<const uint8_t*>(q.visibilitySlots[i].first.contents) + q.visibilitySlots[i].second,
                            sizeof(value));
                if (!value) WMGL_TRACE(Draw, "occlusion query: slot %zu (offset %u) counted nothing", i,
                                       q.visibilitySlots[i].second);
            }
        }
        q.visibilitySlots.clear();
        break;
    }
    case GL_FRAGMENT_SHADER_INVOCATIONS: {
        uint64_t invocations = 0;
        for (const auto& [buffer, offset] : q.visibilitySlots) {
            uint32_t value;
            std::memcpy(&value, static_cast<const uint8_t*>(buffer.contents) + offset, sizeof(value));
            invocations += value;
        }
        q.result = invocations;
        q.visibilitySlots.clear();
        break;
    }
    case GL_TIME_ELAPSED:
        if (q.timer && q.timer->startSet && q.timer->endSet)
            q.result = static_cast<uint64_t>(std::max(0.0, (q.timer->end - q.timer->start) * 1e9));
        break;
    case GL_TIMESTAMP:
        if (q.timer && q.timer->endSet) q.result = static_cast<uint64_t>(q.timer->end * 1e9);
        break;
    default: break;
    }
    q.resultFinal = true;
}

}  // namespace

bool queryResultReady(Context& ctx, Query& q, bool wait) {
    if (q.resultFinal) return true;
    Queue& queue = ctx.stream().queue();
    if (q.pendingSerial) {
        if (wait) {
            ctx.stream().waitFor(q.pendingSerial);
        } else if (!queue.isComplete(q.pendingSerial)) {
            // Availability must eventually become true: make sure the work is submitted.
            if (!queue.isCommitted(q.pendingSerial)) ctx.stream().flush();
            return false;
        }
    }
    // Completion handlers for timers run beside the queue's, after it: a
    // wait waits for them too (they always run, so the bound is only a guard).
    if (q.timer) {
        const bool needStart = q.target == GL_TIME_ELAPSED;
        for (int spin = 0; !q.timer->endSet || (needStart && !q.timer->startSet); ++spin) {
            if (!wait && spin > 0) return false;
            if (spin < 100000) {
                sched_yield();
            } else if (spin < 100000 + 50000) {
                usleep(100);
            } else {
                logOnce(LogLevel::Warn, LogCategory::Gpu, "timer-handler",
                        "a timer query's completion handler did not run; its result is 0");
                break;
            }
        }
    }
    finalize(q);
    return true;
}

uint64_t primitiveCount(GLenum mode, GLsizei count) {
    const uint64_t n = static_cast<uint64_t>(std::max(0, count));
    switch (mode) {
    case GL_POINTS: return n;
    case GL_LINES: return n / 2;
    case GL_LINE_STRIP: return n >= 2 ? n - 1 : 0;
    case GL_LINE_LOOP: return n >= 2 ? n : 0;
    case GL_TRIANGLES: return n / 3;
    case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: return n >= 3 ? n - 2 : 0;
    case GL_LINES_ADJACENCY: return n / 4;
    case GL_LINE_STRIP_ADJACENCY: return n >= 4 ? n - 3 : 0;
    case GL_TRIANGLES_ADJACENCY: return n / 6;
    case GL_TRIANGLE_STRIP_ADJACENCY: return n >= 6 ? (n - 4) / 2 : 0;
    case GL_QUADS: return n / 4;
    case GL_QUAD_STRIP: return n >= 4 ? (n - 2) / 2 : 0;
    case GL_POLYGON: return n >= 3 ? 1 : 0;
    default: return 0;
    }
}

void queriesCountPrimitives(Context& ctx, GLenum mode, GLsizei count, GLsizei instances) {
    Query* q = ctx.state.activeQueries[kQueryPrimitivesGenerated].get();
    if (q) q->result += primitiveCount(mode, count) * static_cast<uint64_t>(std::max(0, instances));
}

void queriesNoteXfbOverflow(Context& ctx, GLuint stream) {
    for (int index : {static_cast<int>(kQueryXfbOverflow), queryTargetIndex(ctx, GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW, stream)})
        if (Query* q = index >= 0 ? ctx.state.activeQueries[index].get() : nullptr) q->result = 1;
}

bool queriesCountingStatistics(Context& ctx) {
    for (int s = 0; s < kStatCount; ++s)
        if (ctx.state.activeQueries[kQueryStatisticsFirst + s]) return true;
    return false;
}

bool queriesNeedDrawSizes(Context& ctx) {
    return ctx.state.activeQueries[kQueryPrimitivesGenerated] || queriesCountingStatistics(ctx) || anyOcclusionActive(ctx);
}

bool queriesCounting(Context& ctx, PipelineStatistic statistic) {
    return ctx.state.activeQueries[kQueryStatisticsFirst + statistic] != nullptr;
}

void queriesCountStatistic(Context& ctx, PipelineStatistic statistic, uint64_t n) {
    if (Query* q = ctx.state.activeQueries[kQueryStatisticsFirst + statistic].get()) q->result += n;
}

void queriesCountSubmitted(Context& ctx, GLenum mode, GLsizei vertices, GLsizei instances, bool clipped) {
    if (!queriesCountingStatistics(ctx)) return;
    const uint64_t n = static_cast<uint64_t>(std::max(0, vertices)), copies = static_cast<uint64_t>(std::max(0, instances));
    // A line loop of two vertices is one primitive here, as the CTS expects.
    const uint64_t primitives = mode == GL_LINE_LOOP ? (n > 2 ? n : n == 2 ? 1 : 0)
                                : mode == GL_PATCHES ? n / static_cast<uint64_t>(std::max(1, ctx.state.patchVertices))
                                                     : primitiveCount(mode, vertices);
    queriesCountStatistic(ctx, kStatVerticesSubmitted, n * copies);
    queriesCountStatistic(ctx, kStatPrimitivesSubmitted, primitives * copies);
    queriesCountStatistic(ctx, kStatVertexShaderInvocations, n * copies);
    if (!clipped) return;
    queriesCountStatistic(ctx, kStatClippingInputPrimitives, primitives * copies);
    queriesCountStatistic(ctx, kStatClippingOutputPrimitives, primitives * copies);
}

bool queriesFragmentCounter(Context& ctx, __strong id<MTLBuffer>& buffer, NSUInteger& offset) {
    Query* q = ctx.state.activeQueries[kQueryStatisticsFirst + kStatFragmentShaderInvocations].get();
    if (!q) return false;
    const uint64_t serial = ctx.stream().serial();
    if (q->visibilitySlots.empty() || q->counterSerial != serial || q->counterBound + ctx.drawSampleBound >= (1ull << 31)) {
        // A new zeroed counter per command buffer (or when it could wrap), from pages of 1024.
        constexpr uint32_t kPage = 4096;
        id<MTLBuffer> page = q->visibilitySlots.empty() ? nil : q->visibilitySlots.back().first;
        uint32_t next = q->visibilitySlots.empty() ? kPage : q->visibilitySlots.back().second + 4;
        if (next >= kPage) {
            page = [Device::get().mtl() newBufferWithLength:kPage options:MTLResourceStorageModeShared];
            next = 0;
        }
        q->visibilitySlots.emplace_back(page, next);
        q->counterSerial = serial;
        q->counterBound = 0;
        q->pendingSerial = std::max(q->pendingSerial, serial);
    }
    q->counterBound += ctx.drawSampleBound;
    buffer = q->visibilitySlots.back().first;
    offset = q->visibilitySlots.back().second;
    return true;
}

void queriesPrepareVisibility(Context& ctx) {
    if (anyOcclusionActive(ctx) && !ctx.stream().visibilitySlotAvailable()) ctx.stream().flush();
}

void queriesBeforeDraw(Context& ctx, id<MTLRenderCommandEncoder> encoder, GLsizei instances) {
    // Apple GPUs count a slot's samples in 32 bits: a slot takes draws while
    // the samples they can cover (the whole target once per instance) fit in
    // 31 bits, then the pass moves on to a new slot.
    const RenderTargetDesc& target = ctx.stream().currentTarget();
    const uint64_t drawBound = static_cast<uint64_t>(target.width) * target.height * std::max(1u, target.samples) *
                               std::max(1u, target.layers) * static_cast<uint64_t>(std::max(1, instances));
    ctx.drawSampleBound = drawBound;
    if (!anyOcclusionActive(ctx)) return;
    if (drawBound >= (1ull << 32))
        logOnce(LogLevel::Warn, LogCategory::Draw, "visibility-wrap",
                "a draw can cover 2^32 samples or more: occlusion counts may wrap");
    auto& slot = ctx.occlusionSlot;
    const uint64_t pass = ctx.stream().passGeneration();
    const bool current = slot.valid && slot.pass == pass && slot.epoch == ctx.occlusionEpoch;
    if (current && slot.samplesBound + drawBound < (1ull << 31)) {
        slot.samplesBound += drawBound;
        return;
    }
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    if (!ctx.stream().allocateVisibilitySlot(buffer, offset)) {
        if (current) return;  // the pass's last slot keeps counting
        logOnce(LogLevel::Warn, LogCategory::Draw, "visibility-slots",
                "occlusion query slots exhausted; samples not counted");
        return;
    }
    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting offset:offset];
    slot.samplesBound = drawBound;
    slot.valid = true;
    slot.pass = pass;
    slot.epoch = ctx.occlusionEpoch;
    const uint64_t serial = ctx.stream().serial();
    for (int i : {kQuerySamplesPassed, kQueryAnySamplesPassed, kQueryAnySamplesPassedConservative}) {
        Query* q = ctx.state.activeQueries[i].get();
        if (!q) continue;
        q->visibilitySlots.emplace_back(buffer, static_cast<uint32_t>(offset));
        q->pendingSerial = std::max(q->pendingSerial, serial);
    }
}

void queriesSuspendVisibility(Context& ctx, id<MTLRenderCommandEncoder> encoder) {
    auto& slot = ctx.occlusionSlot;
    if (!slot.valid || slot.pass != ctx.stream().passGeneration()) return;
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    slot.valid = false;
}

bool conditionalRenderAllows(Context& ctx) {
    auto& cr = ctx.state.conditionalRender;
    if (!cr.active || !cr.query) return true;
    // GL 4.5 / ARB_conditional_render_inverted: the *_INVERTED modes render when no samples passed.
    const bool inverted = cr.mode >= GL_QUERY_WAIT_INVERTED && cr.mode <= GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
    const bool wait = cr.mode == GL_QUERY_WAIT || cr.mode == GL_QUERY_BY_REGION_WAIT || cr.mode == GL_QUERY_WAIT_INVERTED ||
                      cr.mode == GL_QUERY_BY_REGION_WAIT_INVERTED;
    if (!queryResultReady(ctx, *cr.query, wait)) return true;  // NO_WAIT: render while unknown
    return (cr.query->result != 0) != inverted;
}

}  // namespace wmgl

namespace wmgl::api {
namespace {

void endVisibility(Context& ctx) {
    ++ctx.occlusionEpoch;
    if (id<MTLRenderCommandEncoder> encoder = ctx.stream().currentRender()) queriesSuspendVisibility(ctx, encoder);
    ctx.occlusionSlot.valid = false;
    ctx.stream().setVisibilityWanted(anyOcclusionActive(ctx));
}

// GPU times recorded so far never decrease: an empty command buffer can
// report a later GPUEndTime than the busy buffer submitted after it, and GL
// timestamps are monotonic.
double monotonicGpuTime(double t) {
    static std::mutex mutex;
    static double last = 0.0;
    std::lock_guard<std::mutex> lock(mutex);
    last = std::max(last, t);
    return last;
}

// Records the completion time of the current command buffer into the timer:
// monotonic for timestamps; as reported for elapsed time, which its start
// (also as reported) is subtracted from.
void markTimerEnd(Context& ctx, Query& q) {
    std::shared_ptr<Query::Timer> timer = q.timer;
    const bool timestamp = q.target == GL_TIMESTAMP;
    [ctx.stream().commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> done) {
        timer->end = timestamp ? monotonicGpuTime(done.GPUEndTime) : done.GPUEndTime;
        timer->endSet = true;
    }];
    stats::count(&stats::Frame::queryCommits);
    q.pendingSerial = ctx.stream().flush();
}

}  // namespace

void GenQueries(Context& ctx, GLsizei n, GLuint* ids) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    ctx.queries.generate(n, ids);
}

void DeleteQueries(Context& ctx, GLsizei n, const GLuint* ids) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        RefPtr<Query> q = ctx.queries.remove(ids[i]);
        if (!q) continue;
        // Deleting an active query ends it.
        for (size_t t = 0; t < ctx.state.activeQueries.size(); ++t) {
            if (ctx.state.activeQueries[t] != q) continue;
            ctx.state.activeQueries[t] = nullptr;
            if (isOcclusion(static_cast<int>(t))) endVisibility(ctx);
        }
        if (ctx.state.conditionalRender.query == q) ctx.state.conditionalRender = {};
    }
}

GLboolean IsQuery(Context& ctx, GLuint name) { return ctx.queries.exists(name) ? GL_TRUE : GL_FALSE; }

namespace {

// index: queryTargetIndex(ctx, target, stream); -1 is an invalid target, -2
// an invalid stream for it.
bool checkQuerySlot(Context& ctx, int index) {
    if (index == -1) ctx.setError(GL_INVALID_ENUM);
    else if (index < 0) ctx.setError(GL_INVALID_VALUE);
    return index >= 0;
}

void beginQuery(Context& ctx, GLenum target, int index, GLuint name);
void endQuery(Context& ctx, int index);

}  // namespace

void BeginQuery(Context& ctx, GLenum target, GLuint name) {
    const int index = queryTargetIndex(ctx, target);
    if (!checkQuerySlot(ctx, index)) return;
    beginQuery(ctx, target, index, name);
}

void BeginQueryIndexed(Context& ctx, GLenum target, GLuint stream, GLuint name) {
    if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_transform_feedback3")) return ctx.unimplemented("glBeginQueryIndexed");
    const int index = queryTargetIndex(ctx, target, stream);
    if (!checkQuerySlot(ctx, index)) return;
    beginQuery(ctx, target, index, name);
}

void EndQuery(Context& ctx, GLenum target) {
    const int index = queryTargetIndex(ctx, target);
    if (!checkQuerySlot(ctx, index)) return;
    endQuery(ctx, index);
}

void EndQueryIndexed(Context& ctx, GLenum target, GLuint stream) {
    if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_transform_feedback3")) return ctx.unimplemented("glEndQueryIndexed");
    const int index = queryTargetIndex(ctx, target, stream);
    if (!checkQuerySlot(ctx, index)) return;
    endQuery(ctx, index);
}

namespace {

void beginQuery(Context& ctx, GLenum target, int index, GLuint name) {
    if (ctx.state.activeQueries[index]) return ctx.setError(GL_INVALID_OPERATION);
    // OpenGL ES: the two occlusion query targets share one active query.
    if (ctx.isEs() && ((index == kQueryAnySamplesPassed && ctx.state.activeQueries[kQueryAnySamplesPassedConservative]) ||
                       (index == kQueryAnySamplesPassedConservative && ctx.state.activeQueries[kQueryAnySamplesPassed])))
        return ctx.setError(GL_INVALID_OPERATION);
    if (name == 0 || !ctx.queries.isGenerated(name)) return ctx.setError(GL_INVALID_OPERATION);
    RefPtr<Query> q = ctx.queries.getRef(name);
    if (!q) {
        q = makeRef<Query>(name);
        ctx.queries.set(name, q);
    } else if ((q->everActive || q->target) && q->target != target) {  // a created query has its target (GL 4.5)
        return ctx.setError(GL_INVALID_OPERATION);
    }
    for (const RefPtr<Query>& active : ctx.state.activeQueries)
        if (active == q) return ctx.setError(GL_INVALID_OPERATION);
    if (ctx.state.conditionalRender.active && ctx.state.conditionalRender.query == q)
        return ctx.setError(GL_INVALID_OPERATION);
    q->target = target;
    q->active = q->everActive = true;
    q->result = 0;
    q->resultFinal = false;
    q->pendingSerial = 0;
    q->visibilitySlots.clear();
    q->counterSerial = 0;
    q->timer.reset();
    ctx.state.activeQueries[index] = q;
    if (isOcclusion(index)) {
        ++ctx.occlusionEpoch;
        ctx.stream().setVisibilityWanted(true);
    } else if (index == kQueryTimeElapsed) {
        // Time from the start of the next command buffer.
        stats::count(&stats::Frame::queryCommits);
        ctx.stream().flush();
        q->timer = std::make_shared<Query::Timer>();
        std::shared_ptr<Query::Timer> timer = q->timer;
        [ctx.stream().commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> done) {
            timer->start = done.GPUStartTime;
            timer->startSet = true;
        }];
    }
}

void endQuery(Context& ctx, int index) {
    RefPtr<Query> q = ctx.state.activeQueries[index];
    if (!q) return ctx.setError(GL_INVALID_OPERATION);
    ctx.state.activeQueries[index] = nullptr;
    q->active = false;
    if (isOcclusion(index)) {
        endVisibility(ctx);
        if (q->visibilitySlots.empty()) q->resultFinal = true;
    } else if (index == kQueryTimeElapsed) {
        markTimerEnd(ctx, *q);
    } else if (index == kQueryStatisticsFirst + kStatFragmentShaderInvocations) {
        if (q->visibilitySlots.empty()) q->resultFinal = true;  // otherwise summed once its command buffers complete
    } else {
        q->resultFinal = true;  // counted on the CPU
    }
}

}  // namespace

void QueryCounter(Context& ctx, GLuint name, GLenum target) {
    if (target != GL_TIMESTAMP) return ctx.setError(GL_INVALID_ENUM);
    if (name == 0 || !ctx.queries.isGenerated(name)) return ctx.setError(GL_INVALID_OPERATION);
    RefPtr<Query> q = ctx.queries.getRef(name);
    if (!q) {
        q = makeRef<Query>(name);
        ctx.queries.set(name, q);
    } else if (q->active || (q->target && q->target != GL_TIMESTAMP)) {  // used, or made by glCreateQueries
        return ctx.setError(GL_INVALID_OPERATION);
    }
    q->target = GL_TIMESTAMP;
    q->everActive = true;
    q->result = 0;
    q->resultFinal = false;
    q->timer = std::make_shared<Query::Timer>();
    markTimerEnd(ctx, *q);
}

namespace {

void getQuery(Context& ctx, int index, GLenum pname, GLint* params) {
    switch (pname) {
    case GL_CURRENT_QUERY: *params = ctx.state.activeQueries[index] ? static_cast<GLint>(ctx.state.activeQueries[index]->name) : 0; return;
    case GL_QUERY_COUNTER_BITS: {
        const bool boolean = index == kQueryAnySamplesPassed || index == kQueryAnySamplesPassedConservative ||
                             index == kQueryXfbOverflow || index == kQueryXfbStreamOverflow ||
                             index >= kQueryStreamFirst + 6;
        *params = boolean ? 1 : 64;
        return;
    }
    default: return ctx.setError(GL_INVALID_ENUM);
    }
}

}  // namespace

void GetQueryiv(Context& ctx, GLenum target, GLenum pname, GLint* params) {
    if (target == GL_TIMESTAMP) {
        if (pname != GL_QUERY_COUNTER_BITS) return ctx.setError(GL_INVALID_ENUM);
        *params = 64;
        return;
    }
    const int index = queryTargetIndex(ctx, target);
    if (!checkQuerySlot(ctx, index)) return;
    getQuery(ctx, index, pname, params);
}

void GetQueryIndexediv(Context& ctx, GLenum target, GLuint stream, GLenum pname, GLint* params) {
    if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_transform_feedback3")) return ctx.unimplemented("glGetQueryIndexediv");
    if (target == GL_TIMESTAMP && stream == 0) return GetQueryiv(ctx, target, pname, params);
    const int index = queryTargetIndex(ctx, target, stream);
    if (!checkQuerySlot(ctx, index)) return;
    getQuery(ctx, index, pname, params);
}

namespace {

template <class T>
void getQueryObject(Context& ctx, GLuint name, GLenum pname, T* params) {
    Query* q = ctx.queries.get(name);
    // A query created by glCreateQueries (GL 4.5) has its target before it is first used.
    if (!q || (!q->everActive && !q->target) || q->active) return ctx.setError(GL_INVALID_OPERATION);
    if (pname != GL_QUERY_RESULT && pname != GL_QUERY_RESULT_AVAILABLE && pname != GL_QUERY_RESULT_NO_WAIT &&
        !(pname == GL_QUERY_TARGET && ctx.version() >= 45))
        return ctx.setError(GL_INVALID_ENUM);
    // GL 4.4 (GL_ARB_query_buffer_object): with a buffer bound to
    // GL_QUERY_BUFFER, `params` is an offset into it.
    Buffer* queryBuffer = ctx.version() >= 44 ? ctx.state.queryBuffer.get() : nullptr;
    if (queryBuffer) {
        const intptr_t offset = reinterpret_cast<intptr_t>(params);
        if (offset < 0) return ctx.setError(GL_INVALID_VALUE);
        if (static_cast<intptr_t>(sizeof(T)) > queryBuffer->size - offset) return ctx.setError(GL_INVALID_OPERATION);
    }
    uint64_t value = 0;
    switch (pname) {
    case GL_QUERY_RESULT:
        // Wait without the share-group lock (queries belong to one context;
        // the wait touches only the queue), then collect the result.
        if (q->pendingSerial && !q->resultFinal) {
            if (!ctx.stream().queue().isCommitted(q->pendingSerial)) ctx.stream().flush();
            ShareUnlock unlock(ctx);
            ctx.stream().waitFor(q->pendingSerial);
        }
        queryResultReady(ctx, *q, true);
        value = q->result;
        break;
    case GL_QUERY_RESULT_AVAILABLE: value = queryResultReady(ctx, *q, false) ? 1 : 0; break;
    case GL_QUERY_RESULT_NO_WAIT:
        if (ctx.version() < 44) return ctx.setError(GL_INVALID_ENUM);
        if (!queryResultReady(ctx, *q, false)) return;
        value = q->result;
        break;
    case GL_QUERY_TARGET: value = q->target; break;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
    // Results that do not fit the requested type saturate.
    if (value > static_cast<uint64_t>(std::numeric_limits<T>::max())) value = static_cast<uint64_t>(std::numeric_limits<T>::max());
    const T result = static_cast<T>(value);
    if (!queryBuffer) {
        *params = result;
        return;
    }
    // The result goes into the query buffer through an ordered GPU copy, so
    // draws before and after see the buffer in command order.
    if (!queryBuffer->mtl) return;
    const intptr_t offset = reinterpret_cast<intptr_t>(params);
    if (offset % 4) {
        // Metal copies whole words: an unaligned result is written once
        // the queued work using the buffer is done.
        ctx.stream().waitFor(queryBuffer->lastUseSerial);
        std::memcpy(static_cast<uint8_t*>(queryBuffer->mtl.contents) + offset, &result, sizeof(T));
        ++queryBuffer->version;
        return;
    }
    TransientAllocation staging = ctx.draw().transient().allocate(sizeof(T), ctx.stream().serial(), 16);
    std::memcpy(staging.data, &result, sizeof(T));
    [ctx.stream().blit(true) copyFromBuffer:staging.buffer sourceOffset:staging.offset toBuffer:queryBuffer->mtl
                        destinationOffset:static_cast<NSUInteger>(offset) size:sizeof(T)];
    queryBuffer->lastUseSerial = queryBuffer->lastWriteSerial = ctx.stream().serial();
    ++queryBuffer->version;
}

}  // namespace

void GetQueryObjectiv(Context& ctx, GLuint name, GLenum pname, GLint* params) { getQueryObject(ctx, name, pname, params); }
void GetQueryObjectuiv(Context& ctx, GLuint name, GLenum pname, GLuint* params) { getQueryObject(ctx, name, pname, params); }
void GetQueryObjecti64v(Context& ctx, GLuint name, GLenum pname, GLint64* params) { getQueryObject(ctx, name, pname, params); }
void GetQueryObjectui64v(Context& ctx, GLuint name, GLenum pname, GLuint64* params) { getQueryObject(ctx, name, pname, params); }

// ------------------------------------------------------------ conditional rendering

void BeginConditionalRender(Context& ctx, GLuint name, GLenum mode) {
    const bool inverted = (ctx.version() >= 45 || ctx.hasExtension("GL_ARB_conditional_render_inverted")) &&
                          mode >= GL_QUERY_WAIT_INVERTED && mode <= GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
    if (mode != GL_QUERY_WAIT && mode != GL_QUERY_NO_WAIT && mode != GL_QUERY_BY_REGION_WAIT &&
        mode != GL_QUERY_BY_REGION_NO_WAIT && !inverted)
        return ctx.setError(GL_INVALID_ENUM);
    auto& cr = ctx.state.conditionalRender;
    if (cr.active) return ctx.setError(GL_INVALID_OPERATION);
    Query* q = ctx.queries.get(name);
    if (!q || !q->everActive || q->active) return ctx.setError(GL_INVALID_OPERATION);
    // GL 4.6 / ARB_transform_feedback_overflow_query: overflow queries render when they overflowed.
    if (q->target != GL_SAMPLES_PASSED && q->target != GL_ANY_SAMPLES_PASSED &&
        q->target != GL_ANY_SAMPLES_PASSED_CONSERVATIVE && q->target != GL_TRANSFORM_FEEDBACK_OVERFLOW &&
        q->target != GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW)
        return ctx.setError(GL_INVALID_OPERATION);
    cr.query = q;
    cr.mode = mode;
    cr.active = true;
}

void EndConditionalRender(Context& ctx) {
    auto& cr = ctx.state.conditionalRender;
    if (!cr.active) return ctx.setError(GL_INVALID_OPERATION);
    cr = {};
}

// ------------------------------------------------------------ syncs

namespace {

Sync* syncObject(Context& ctx, GLsync handle) {
    Sync* sync = reinterpret_cast<Sync*>(handle);
    ShareLock lock(ctx);
    return sync && ctx.share().syncs.count(sync) ? sync : nullptr;
}

}  // namespace

GLsync FenceSync(Context& ctx, GLenum condition, GLbitfield flags) {
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE) return ctx.setError(GL_INVALID_ENUM), nullptr;
    if (flags != 0) return ctx.setError(GL_INVALID_VALUE), nullptr;
    Sync* sync = new Sync();
    sync->retain();
    // The fence is the current command buffer: it signals once that buffer is
    // flushed (glFlush, SYNC_FLUSH_COMMANDS_BIT, SwapBuffers, ...) and done, as
    // on other drivers. Committing here would split the stream into a command
    // buffer per fence for fence-heavy renderers. With nothing recorded, the
    // fence covers the work already committed and needs no flush.
    sync->serial = ctx.stream().fenceSerial();
    sync->shareGroup = &ctx.share();
    ShareLock lock(ctx);
    ctx.share().syncs.insert(sync);
    return reinterpret_cast<GLsync>(sync);
}

GLboolean IsSync(Context& ctx, GLsync sync) { return syncObject(ctx, sync) ? GL_TRUE : GL_FALSE; }

void DeleteSync(Context& ctx, GLsync handle) {
    if (!handle) return;
    Sync* sync = syncObject(ctx, handle);
    if (!sync) return ctx.setError(GL_INVALID_VALUE);
    {
        ShareLock lock(ctx);
        ctx.share().syncs.erase(sync);
    }
    sync->release();
}

GLenum ClientWaitSync(Context& ctx, GLsync handle, GLbitfield flags, GLuint64 timeout) {
    Sync* sync = syncObject(ctx, handle);
    if (!sync) return ctx.setError(GL_INVALID_VALUE), GL_WAIT_FAILED;
    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) return ctx.setError(GL_INVALID_VALUE), GL_WAIT_FAILED;
    RefPtr<Sync> hold(sync);
    Queue& queue = ctx.share().queue;
    stats::count(&stats::Frame::clientWaits);
    if (!sync->serial || queue.isComplete(sync->serial)) {
        stats::count(&stats::Frame::clientWaitsSignaled);
        return GL_ALREADY_SIGNALED;
    }
    if (flags & GL_SYNC_FLUSH_COMMANDS_BIT) ctx.stream().flush();
    if (timeout == 0) return GL_TIMEOUT_EXPIRED;
    stats::Timer timer(&stats::Frame::clientWaitNanos);
    ShareUnlock unlock(ctx);  // `hold` keeps the sync alive
    return queue.waitSignaled(sync->serial, timeout) ? GL_CONDITION_SATISFIED : GL_TIMEOUT_EXPIRED;
}

void WaitSync(Context& ctx, GLsync handle, GLbitfield flags, GLuint64 timeout) {
    Sync* sync = syncObject(ctx, handle);
    if (!sync) return ctx.setError(GL_INVALID_VALUE);
    if (flags != 0 || timeout != GL_TIMEOUT_IGNORED) return ctx.setError(GL_INVALID_VALUE);
    // Contexts in a share group submit to one in-order Metal queue. A fence of
    // this context is in its stream already, and a fence of another context
    // must have been flushed by it (GL 3.3 section 5.2), so later work from
    // this context is committed, and runs, after it.
}

void GetSynciv(Context& ctx, GLsync handle, GLenum pname, GLsizei count, GLsizei* length, GLint* values) {
    Sync* sync = syncObject(ctx, handle);
    if (!sync) return ctx.setError(GL_INVALID_VALUE);
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    GLint value = 0;
    switch (pname) {
    case GL_OBJECT_TYPE: value = GL_SYNC_FENCE; break;
    case GL_SYNC_STATUS:
        value = !sync->serial || ctx.share().queue.isComplete(sync->serial) ? GL_SIGNALED : GL_UNSIGNALED;
        break;
    case GL_SYNC_CONDITION: value = GL_SYNC_GPU_COMMANDS_COMPLETE; break;
    case GL_SYNC_FLAGS: value = 0; break;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
    if (count >= 1) {
        values[0] = value;
        if (length) *length = 1;
    } else if (length) {
        *length = 0;
    }
}

}  // namespace wmgl::api
