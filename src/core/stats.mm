#include "core/stats.h"

#include "core/metal/device.h"

#include <mach/mach_time.h>
#include <os/signpost.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace wmgl::stats {
namespace {

bool enabledFromEnvironment() {
    const char* v = std::getenv("WINEMETALGL_STATS");
    return v && *v && *v != '0';
}

os_log_t signpostLog() {
    static os_log_t log = os_log_create("org.winehq.winemetalgl", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    return log;
}

struct ThreadState {
    Frame frame;
    uint64_t index = 0, lastPresent = 0;
    os_signpost_id_t interval = OS_SIGNPOST_ID_NULL;
};

ThreadState& thread() {
    static thread_local ThreadState state;
    return state;
}

double ms(uint64_t nanos) { return static_cast<double>(nanos) / 1e6; }

}  // namespace

const bool g_reporting = enabledFromEnvironment();
bool g_enabled = g_reporting;

Frame& frame() { return thread().frame; }

uint64_t nowNanos() {
    static mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t t;
        mach_timebase_info(&t);
        return t;
    }();
    return mach_absolute_time() * timebase.numer / timebase.denom;
}

namespace {

// The process-wide profile: per command (by its name's address) and per
// thread, since the 60 frames last printed.
struct CommandProfile {
    uint64_t calls = 0, nanos = 0, lockNanos = 0;
};
struct Profile {
    std::mutex mutex;
    std::unordered_map<const char*, CommandProfile> commands;
    std::map<uint64_t, CommandProfile> threads;  // by thread number
};
Profile& profile() {
    static Profile* p = new Profile;
    return *p;
}
uint64_t threadNumber() {
    static std::atomic<uint64_t> next{1};
    static thread_local uint64_t number = next++;
    return number;
}

}  // namespace

void recordCall(const char* command, uint64_t nanos, uint64_t lockNanos) {
    Profile& p = profile();
    const uint64_t thread = threadNumber();
    std::lock_guard<std::mutex> lock(p.mutex);
    CommandProfile& c = p.commands[command];
    ++c.calls;
    c.nanos += nanos;
    c.lockNanos += lockNanos;
    CommandProfile& t = p.threads[thread];
    ++t.calls;
    t.nanos += nanos;
    t.lockNanos += lockNanos;
}

namespace {

// Every 60 presented frames: the commands that took the most time (per
// frame, all threads) and each thread's share.
void printProfile(uint64_t frames) {
    Profile& p = profile();
    std::vector<std::pair<const char*, CommandProfile>> top;
    std::map<uint64_t, CommandProfile> threads;
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        top.assign(p.commands.begin(), p.commands.end());
        threads.swap(p.threads);
        p.commands.clear();
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.nanos > b.second.nanos; });
    const double n = static_cast<double>(frames);
    for (const auto& [number, t] : threads)
        std::fprintf(stderr, "winemetalgl:stats: thread %llu: %.2f ms/frame in %.0f calls/frame (lock %.2f ms)\n",
                     static_cast<unsigned long long>(number), ms(t.nanos) / n, static_cast<double>(t.calls) / n,
                     ms(t.lockNanos) / n);
    for (size_t i = 0; i < top.size() && i < 15; ++i)
        std::fprintf(stderr, "winemetalgl:stats:   %-36s %7.2f ms/frame %7.1f calls/frame (lock %.2f ms)\n", top[i].first,
                     ms(top[i].second.nanos) / n, static_cast<double>(top[i].second.calls) / n,
                     ms(top[i].second.lockNanos) / n);
}

}  // namespace

namespace {
std::atomic<uint64_t> g_gpuNanos{0};
}  // namespace

void addGpuNanos(uint64_t nanos) { g_gpuNanos.fetch_add(nanos, std::memory_order_relaxed); }

namespace {
std::mutex g_busyMutex;
double g_busyEnd = 0;
uint64_t g_busyNanos = 0;
}  // namespace

void addGpuInterval(double start, double end) {
    std::lock_guard<std::mutex> lock(g_busyMutex);
    const double from = std::max(start, g_busyEnd);
    if (end > from) g_busyNanos += static_cast<uint64_t>((end - from) * 1e9);
    g_busyEnd = std::max(g_busyEnd, end);
}

static uint64_t takeGpuBusyNanos() {
    std::lock_guard<std::mutex> lock(g_busyMutex);
    const uint64_t n = g_busyNanos;
    g_busyNanos = 0;
    return n;
}

bool counter(const char* name, uint64_t& value) {
#define WMGL_COUNTER(field) {#field, &Frame::field}
    static const std::pair<const char*, uint64_t Frame::*> fields[] = {
        WMGL_COUNTER(calls), WMGL_COUNTER(draws), WMGL_COUNTER(passes), WMGL_COUNTER(pipelineMisses),
        WMGL_COUNTER(programLinks), WMGL_COUNTER(waits), WMGL_COUNTER(convertedVertices), WMGL_COUNTER(cpuIndexDraws),
        WMGL_COUNTER(tessCpu), WMGL_COUNTER(tessMetal), WMGL_COUNTER(tessMetalFlat), WMGL_COUNTER(tessMdiGpu), WMGL_COUNTER(positionOnlyPipelines), WMGL_COUNTER(indirectGpu),
        WMGL_COUNTER(indirectGpuBatches),
        WMGL_COUNTER(indirectCpu), WMGL_COUNTER(indirectCpuEmpty), WMGL_COUNTER(indirectWaits), WMGL_COUNTER(commits),
        WMGL_COUNTER(queryCommits), WMGL_COUNTER(readsAsync), WMGL_COUNTER(readsWaiting)};
#undef WMGL_COUNTER
    // CommandStream::PassEnd order.
    static const char* const passEnds[8] = {"other", "target", "blit", "compute", "visibility", "commit", "barrier", "clear"};
    const Frame& f = frame();
    const std::string n = name ? name : "";
    // Not a frame counter: the bytes the Metal device has allocated (all of
    // the process's buffers, textures and heaps), for leak checks.
    if (n == "deviceAllocatedBytes") return value = Device::get().mtl().currentAllocatedSize, true;
    for (const auto& [field, member] : fields)
        if (n == field) return value = f.*member, true;
    for (int i = 0; i < 8; ++i)
        if (n == std::string("passEnds.") + passEnds[i]) return value = f.passEnds[i], true;
    for (int i = 0; i < kIndirectCpuReasons; ++i)
        if (n == "indirectCpuReasons." + std::to_string(i)) return value = f.indirectCpuReasons[i], true;
    return false;
}

void describeOnce(const std::string& key, const char* format, ...) {
    if (!g_reporting) return;
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!seen.insert(key).second) return;
    }
    char message[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    std::fprintf(stderr, "winemetalgl:stats: %s\n", message);
}

namespace {
thread_local IndirectCpu t_lastIndirectCpu = kIndirectCpuReasons;
}  // namespace

void indirectCpu(IndirectCpu reason, const std::string& key, const char* detail) {
    if (!g_enabled) return;
    t_lastIndirectCpu = reason;
    ++frame().indirectCpuReasons[reason];
    describeOnce("indirect:" + std::to_string(reason) + ':' + key, "indirect commands read on the CPU: %s", detail);
}

void repeatIndirectCpu() {
    if (g_enabled && t_lastIndirectCpu < kIndirectCpuReasons) ++frame().indirectCpuReasons[t_lastIndirectCpu];
}

void endFrame() {
    ThreadState& t = thread();
    os_log_t log = signpostLog();
    if (os_signpost_enabled(log)) {
        if (t.interval != OS_SIGNPOST_ID_NULL) os_signpost_interval_end(log, t.interval, "frame");
        t.interval = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, t.interval, "frame");
    }
    if (!g_reporting) return;
    const uint64_t now = nowNanos();
    const Frame& f = t.frame;
    std::fprintf(stderr,
                 "winemetalgl:stats: frame %llu %.2f ms | gl %.2f ms in %llu calls | present %.2f ms (drawable %.2f ms) | "
                 "draws %llu passes %llu | pipelines %llu (%.2f ms) links %llu (%.2f ms) | transient %llu KiB | "
                 "waits %llu (%.2f ms) | lock %.2f ms | converted %llu vertices, %llu reused (client %llu format %llu "
                 "[0x%llx] other %llu) | cpu indices %llu | tessellation cpu %llu metal %llu (flat %llu, prelude %llu, mdi gpu %llu commands %llu) | indirect gpu %llu (batches %llu) cpu %llu (empty %llu waits %llu; commands %llu "
                 "stage %llu capture %llu query %llu primitive %llu parameters %llu converted %llu index type %llu "
                 "restart %llu elements mapped %llu writing %llu max index %llu) | uniforms %llu KiB | commits %llu (query %llu) | "
                 "pass ends: target %llu blit %llu compute %llu visibility %llu commit %llu barrier %llu clear %llu "
                 "other %llu | reads async %llu waiting %llu | client waits %llu (signaled %llu, %.2f ms) | gpu %.2f ms busy %.2f ms | metal %.1f MiB | draw cpu: samplers %.2f vertex %.2f "
                 "prepare %.2f pipeline %.2f bind %.2f encode %.2f index %.2f ms\n",
                 static_cast<unsigned long long>(t.index), t.lastPresent ? ms(now - t.lastPresent) : 0.0, ms(f.glNanos),
                 static_cast<unsigned long long>(f.calls), ms(f.presentNanos), ms(f.drawableNanos),
                 static_cast<unsigned long long>(f.draws),
                 static_cast<unsigned long long>(f.passes), static_cast<unsigned long long>(f.pipelineMisses),
                 ms(f.pipelineNanos), static_cast<unsigned long long>(f.programLinks), ms(f.linkNanos),
                 static_cast<unsigned long long>(f.transientBytes / 1024), static_cast<unsigned long long>(f.waits),
                 ms(f.waitNanos), ms(f.lockNanos), static_cast<unsigned long long>(f.convertedVertices),
                 static_cast<unsigned long long>(f.convertedReused), static_cast<unsigned long long>(f.convertedClient),
                 static_cast<unsigned long long>(f.convertedFormat), static_cast<unsigned long long>(f.convertedFormatType),
                 static_cast<unsigned long long>(f.convertedOther), static_cast<unsigned long long>(f.cpuIndexDraws),
                 static_cast<unsigned long long>(f.tessCpu), static_cast<unsigned long long>(f.tessMetal),
                 static_cast<unsigned long long>(f.tessMetalFlat), static_cast<unsigned long long>(f.tessPrelude),
                 static_cast<unsigned long long>(f.tessMdiGpu), static_cast<unsigned long long>(f.tessMdiGpuCommands),
                 static_cast<unsigned long long>(f.indirectGpu), static_cast<unsigned long long>(f.indirectGpuBatches),
                 static_cast<unsigned long long>(f.indirectCpu),
                 static_cast<unsigned long long>(f.indirectCpuEmpty), static_cast<unsigned long long>(f.indirectWaits),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectCommands]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectStage]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectCapture]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectQuery]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectPrimitive]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectParameters]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectConverted]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectIndexType]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectRestart]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectElementsMapped]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectElementsWriting]),
                 static_cast<unsigned long long>(f.indirectCpuReasons[kIndirectMaxIndex]),
                 static_cast<unsigned long long>(f.uniformBytes / 1024), static_cast<unsigned long long>(f.commits),
                 static_cast<unsigned long long>(f.queryCommits), static_cast<unsigned long long>(f.passEnds[1]),
                 static_cast<unsigned long long>(f.passEnds[2]), static_cast<unsigned long long>(f.passEnds[3]),
                 static_cast<unsigned long long>(f.passEnds[4]), static_cast<unsigned long long>(f.passEnds[5]),
                 static_cast<unsigned long long>(f.passEnds[6]), static_cast<unsigned long long>(f.passEnds[7]),
                 static_cast<unsigned long long>(f.passEnds[0]), static_cast<unsigned long long>(f.readsAsync),
                 static_cast<unsigned long long>(f.readsWaiting), static_cast<unsigned long long>(f.clientWaits),
                 static_cast<unsigned long long>(f.clientWaitsSignaled), ms(f.clientWaitNanos),
                 ms(g_gpuNanos.exchange(0, std::memory_order_relaxed)), ms(takeGpuBusyNanos()),
                 static_cast<double>(Device::get().mtl().currentAllocatedSize) / 1048576.0,
                 ms(f.drawSamplersNanos), ms(f.drawVertexNanos), ms(f.drawPrepareNanos), ms(f.drawPipelineNanos),
                 ms(f.drawBindNanos), ms(f.drawEncodeNanos), ms(f.drawIndexNanos));
    if (t.index % 60 == 59) printProfile(60);
    t.frame = Frame();
    t.lastPresent = now;
    ++t.index;
}

void passStarted(uint32_t width, uint32_t height) {
    count(&Frame::passes);
    os_log_t log = signpostLog();
    if (os_signpost_enabled(log)) os_signpost_event_emit(log, OS_SIGNPOST_ID_EXCLUSIVE, "render pass", "%ux%u", width, height);
}

}  // namespace wmgl::stats
