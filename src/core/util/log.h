// Logging for WineMetalGL: the core's, and through the driver ABI
// (wmgl_log_*) the Wine module's, unix and PE sides.
//
// WINEMETALGL_LOG chooses the levels, globally and per category:
//   WINEMETALGL_LOG=2                 everything at info
//   WINEMETALGL_LOG=shader:3,gpu:2    trace shaders, info for the GPU, defaults elsewhere
//   WINEMETALGL_LOG=warn,wine:trace   levels by name, a global level with exceptions
// Levels are 0 = errors, 1 = warnings, 2 = info, 3 = trace (or error, warn,
// info, trace); categories are listed in logCategoryName.
//
// Messages go to stderr (errors by default, as before) and to a log file
// (warnings by default): WINEMETALGL_LOG_FILE names it, a directory to put it
// in, or 0 for none; by default it is ~/Library/Logs/WineMetalGL/<program>-
// <date>-<pid>.log. The file is created when its first message is written, so a
// program that logs nothing leaves none, and starts with the recent history
// (below): the startup banner and what led up to the message.
//
// Every message at info or above is kept in a ring of recent history whatever
// the levels, which the crash handler (crash.h) writes out with a fault.
#pragma once

#include <cstdarg>
#include <cstddef>

namespace wmgl {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Trace = 3 };

enum class LogCategory {
    General,   // startup, configuration, anything else
    Driver,    // the driver ABI: initialization, contexts and surfaces as the module asks for them
    Context,   // GL contexts, current state, GL errors
    Surface,   // windows, pbuffers, presentation, MetalFX
    Draw,      // draws, framebuffers, passes
    Shader,    // GLSL, SPIR-V and ARB programs, their translation to MSL
    Pipeline,  // Metal libraries and pipeline states
    Texture,   // textures, samplers, images
    Buffer,    // buffer objects and their memory
    Gpu,       // the Metal device, command buffers and their errors
    Api,       // unimplemented or unsupported GL usage
    Perf,      // WINEMETALGL_STATS output, slow paths
    Wine,      // the Wine module's unix side (winemetalgl.so)
    Pe,        // the Wine module's PE side (winemetalgl.dll)
    Crash,     // faults and uncaught exceptions
    Count
};

constexpr int kLogCategoryCount = static_cast<int>(LogCategory::Count);

const char* logCategoryName(LogCategory category);
// The category with this name, or false.
bool logCategoryFromName(const char* name, LogCategory& category);

// What WINEMETALGL_LOG asks for: a level per category for stderr and for the
// file. Parsed from the environment once; exposed for tests.
struct LogLevels {
    int stderrLevel[kLogCategoryCount];
    int fileLevel[kLogCategoryCount];
    bool valid = true;  // false: the specification had something not understood (reported once a log opens)
};
LogLevels parseLogLevels(const char* spec, bool wineDebug);

// True when a message at this level reaches stderr or the file: for messages
// that cost something to build (WMGL_TRACE checks it).
bool logEnabled(LogLevel level, LogCategory category);
// True when a message is at least kept in the recent history.
bool logCaptured(LogLevel level, LogCategory category);

void logMessage(LogLevel level, LogCategory category, const char* format, ...) __attribute__((format(printf, 3, 4)));
void logMessageV(LogLevel level, LogCategory category, const char* format, va_list args);
// Logs only the first time this exact key is seen (per process). True when it logged.
bool logOnce(LogLevel level, LogCategory category, const char* key, const char* format, ...)
    __attribute__((format(printf, 4, 5)));
// A finished message (the Wine module's): a trailing newline is dropped.
void logText(LogLevel level, LogCategory category, const char* text);
// A line written to stderr as it is (WINEMETALGL_STATS output, which tools
// parse) and to the file as a message of the category.
void logRaw(LogCategory category, const char* line);
void logRawFormat(LogCategory category, const char* format, ...) __attribute__((format(printf, 2, 3)));

// The process as the host knows it (the Windows program's name, Wine's
// version), for the banner and the log file's name. Call before anything is
// logged to name the file after the program.
void logSetHost(const char* program, const char* hostVersion);
// Writes the banner (version, system, GPU, environment), once: kept for the
// log file's head and the crash report. The core calls it when it initializes.
void logBanner(const char* version, const char* commit, const char* gpu);

// The log file's path, chosen once (empty: no file).
const char* logFilePath();
// Opens the log file now (it is otherwise opened by its first message) and
// returns its descriptor, or -1. For the crash handler, which also needs
// writeHistory to be safe in a signal handler.
int logFileDescriptor();
// True when the log file is open.
bool logFileOpen();
// Writes the recent history to fd, oldest first, with write(2) only.
void logWriteHistory(int fd);

// Tests: replace the configuration, as if from these settings (null: unset),
// and forget the file and history.
void logResetForTesting(const char* spec, const char* file);

}  // namespace wmgl

#define WMGL_LOG_CAT(cat) ::wmgl::LogCategory::cat
#define WMGL_ERR(cat, ...) ::wmgl::logMessage(::wmgl::LogLevel::Error, WMGL_LOG_CAT(cat), __VA_ARGS__)
#define WMGL_WARN(cat, ...) ::wmgl::logMessage(::wmgl::LogLevel::Warn, WMGL_LOG_CAT(cat), __VA_ARGS__)
#define WMGL_INFO(cat, ...) ::wmgl::logMessage(::wmgl::LogLevel::Info, WMGL_LOG_CAT(cat), __VA_ARGS__)
#define WMGL_TRACE(cat, ...)                                                          \
    do {                                                                              \
        if (::wmgl::logEnabled(::wmgl::LogLevel::Trace, WMGL_LOG_CAT(cat)))           \
            ::wmgl::logMessage(::wmgl::LogLevel::Trace, WMGL_LOG_CAT(cat), __VA_ARGS__); \
    } while (0)
