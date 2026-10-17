#include "core/util/log.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

namespace wmgl {
namespace {

int configuredLevel() {
    static const int level = [] {
        const char* value = std::getenv("WINEMETALGL_LOG");
        return value ? std::atoi(value) : 0;
    }();
    return level;
}

void vlog(LogLevel level, const char* format, va_list args) {
    static const char* const names[] = {"err", "warn", "info", "trace"};
    char buffer[2048];
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    std::fprintf(stderr, "winemetalgl:%s: %s\n", names[static_cast<int>(level)], buffer);
}

}  // namespace

bool logEnabled(LogLevel level) { return static_cast<int>(level) <= configuredLevel(); }

void logMessage(LogLevel level, const char* format, ...) {
    if (!logEnabled(level)) return;
    va_list args;
    va_start(args, format);
    vlog(level, format, args);
    va_end(args);
}

bool logOnce(LogLevel level, const char* key, const char* format, ...) {
    if (!logEnabled(level)) return false;
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!seen.insert(key).second) return false;
    }
    va_list args;
    va_start(args, format);
    vlog(level, format, args);
    va_end(args);
    return true;
}

}  // namespace wmgl
