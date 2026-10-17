// Logging for the WineMetalGL core. WINEMETALGL_LOG selects the level:
// 0 = errors (default), 1 = warnings, 2 = info, 3 = trace.
#pragma once

#include <cstdarg>

namespace wmgl {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Trace = 3 };

bool logEnabled(LogLevel level);
void logMessage(LogLevel level, const char* format, ...) __attribute__((format(printf, 2, 3)));
// Logs only the first time this exact key is seen (per process).
// True when it logged.
bool logOnce(LogLevel level, const char* key, const char* format, ...) __attribute__((format(printf, 3, 4)));

}  // namespace wmgl

#define WMGL_ERR(...) ::wmgl::logMessage(::wmgl::LogLevel::Error, __VA_ARGS__)
#define WMGL_WARN(...) ::wmgl::logMessage(::wmgl::LogLevel::Warn, __VA_ARGS__)
#define WMGL_INFO(...) ::wmgl::logMessage(::wmgl::LogLevel::Info, __VA_ARGS__)
#define WMGL_TRACE(...) \
    do { if (::wmgl::logEnabled(::wmgl::LogLevel::Trace)) ::wmgl::logMessage(::wmgl::LogLevel::Trace, __VA_ARGS__); } while (0)
