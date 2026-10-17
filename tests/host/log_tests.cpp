// Logging (core/util/log.h) and crash reports (core/util/crash.h).

#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "core/util/crash.h"
#include "core/util/log.h"
#include "test.h"

using namespace wmgl;

namespace {

int level(const LogLevels& levels, LogCategory c, bool file) {
    return file ? levels.fileLevel[static_cast<int>(c)] : levels.stderrLevel[static_cast<int>(c)];
}

std::string tempDir() {
    char path[] = "/tmp/wmgl-log-test-XXXXXX";
    return mkdtemp(path) ? path : "/tmp";
}

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

size_t count(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
}

// Faults in this test program, which the forked children count as WineMetalGL's.
volatile int* volatile g_null = nullptr;
__attribute__((noinline)) void faultHere() { *g_null = 1; }

// Runs body in a child process with the crash handler; returns its wait status.
template <typename F>
int inChild(const std::string& file, F body) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        logResetForTesting("0", file.c_str());
        registerCrashImage(reinterpret_cast<const void*>(&faultHere));
        installCrashHandlers();
        WMGL_INFO(General, "breadcrumb before the crash");
        body();
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return status;
}

void exitFromPreviousHandler(int) { _exit(42); }

}  // namespace

TEST(log_levels_default) {
    const LogLevels l = parseLogLevels(nullptr, false);
    CHECK(l.valid);
    CHECK_EQ(level(l, LogCategory::Shader, false), 0);  // stderr: errors, as before
    CHECK_EQ(level(l, LogCategory::Shader, true), 1);   // the file: warnings
    CHECK_EQ(level(l, LogCategory::Wine, false), 0);
}

TEST(log_levels_global_and_categories) {
    LogLevels l = parseLogLevels("2", false);
    CHECK_EQ(level(l, LogCategory::Gpu, false), 2);
    CHECK_EQ(level(l, LogCategory::Gpu, true), 2);
    l = parseLogLevels("shader:3,gpu:2", false);
    CHECK(l.valid);
    CHECK_EQ(level(l, LogCategory::Shader, false), 3);
    CHECK_EQ(level(l, LogCategory::Gpu, false), 2);
    CHECK_EQ(level(l, LogCategory::Draw, false), 0);
    CHECK_EQ(level(l, LogCategory::Draw, true), 1);
    // Names, '=', spaces; the global level applies first whatever the order.
    l = parseLogLevels("wine=trace warn", false);
    CHECK(l.valid);
    CHECK_EQ(level(l, LogCategory::Wine, false), 3);
    CHECK_EQ(level(l, LogCategory::Texture, false), 1);
    // WINEMETALGL_LOG=0 keeps the file's warnings; off silences a category.
    l = parseLogLevels("0,api:off", false);
    CHECK_EQ(level(l, LogCategory::Draw, true), 1);
    CHECK_EQ(level(l, LogCategory::Api, false), -1);
    CHECK_EQ(level(l, LogCategory::Api, true), -1);
    // Large numbers have always meant everything.
    CHECK_EQ(level(parseLogLevels("9", false), LogCategory::Pe, false), 3);
}

TEST(log_levels_wine_debug_and_errors) {
    LogLevels l = parseLogLevels(nullptr, true);  // WINEMETALGL_WINE_DEBUG
    CHECK_EQ(level(l, LogCategory::Wine, false), 3);
    CHECK_EQ(level(l, LogCategory::Pe, false), 3);
    CHECK_EQ(level(l, LogCategory::Shader, false), 0);
    l = parseLogLevels("wine:1", true);  // explicit wins
    CHECK_EQ(level(l, LogCategory::Wine, false), 1);
    CHECK(!parseLogLevels("shaders:3", false).valid);
    CHECK(!parseLogLevels("gpu:loud", false).valid);
    LogCategory c;
    CHECK(logCategoryFromName("GPU", c) && c == LogCategory::Gpu);
    CHECK(!logCategoryFromName("gpus", c));
}

TEST(log_file_created_by_first_message) {
    const std::string dir = tempDir(), path = dir + "/game.log";
    logResetForTesting("0", path.c_str());
    WMGL_INFO(Shader, "info before anything goes wrong");
    CHECK(!exists(path));  // info is history only: no file yet
    WMGL_WARN(Gpu, "first warning");
    WMGL_WARN(Gpu, "second warning");
    WMGL_TRACE(Gpu, "trace is not kept");
    const std::string text = readFile(path);
    CHECK(text.find("---- recent history ----") != std::string::npos);
    CHECK(text.find("shader: info before anything goes wrong") != std::string::npos);
    CHECK_EQ(count(text, "first warning"), size_t(1));  // in the history, not again
    CHECK_EQ(count(text, "second warning"), size_t(1));
    CHECK(text.find("trace is not kept") == std::string::npos);
    CHECK(text.find(" warn  gpu: first warning") != std::string::npos);
    logResetForTesting(nullptr, "0");
}

TEST(log_file_settings) {
    const std::string dir = tempDir();
    // A directory: a file named after the program in it.
    logResetForTesting("0", (dir + "/").c_str());
    WMGL_ERR(General, "into a directory");
    const std::string chosen = logFilePath();
    CHECK(chosen.rfind(dir + "/", 0) == 0);
    CHECK(chosen.find("wmgl_host_tests-") != std::string::npos);
    CHECK(readFile(chosen).find("into a directory") != std::string::npos);
    // %p: the process ID.
    logResetForTesting("0", (dir + "/run-%p.log").c_str());
    WMGL_ERR(General, "with a pid");
    CHECK(exists(dir + "/run-" + std::to_string(getpid()) + ".log"));
    // 0: no file.
    logResetForTesting("0", "0");
    WMGL_ERR(General, "nowhere");
    CHECK(std::string(logFilePath()).empty());
    logResetForTesting(nullptr, "0");
}

TEST(log_long_messages) {
    const std::string dir = tempDir(), path = dir + "/long.log";
    logResetForTesting("0", path.c_str());
    const std::string big(5000, 'x');
    WMGL_WARN(Shader, "%s end", big.c_str());
    WMGL_TRACE(Shader, "unused %s", big.c_str());
    const std::string text = readFile(path);
    // The history keeps a slot's worth; the file gets the message whole once it is open.
    WMGL_WARN(Shader, "%s tail", big.c_str());
    CHECK(readFile(path).find(big + " tail") != std::string::npos);
    CHECK(!text.empty());
    logResetForTesting(nullptr, "0");
}

TEST(log_text_from_the_module) {
    const std::string dir = tempDir(), path = dir + "/module.log";
    logResetForTesting("wine:2", path.c_str());
    logText(LogLevel::Info, LogCategory::Wine, "wglCreateContextAttribsARB: 3.3 core\n");
    logText(LogLevel::Warn, LogCategory::Pe, "a PE warning\n");
    const std::string text = readFile(path);
    CHECK(text.find("wine: wglCreateContextAttribsARB: 3.3 core\n") != std::string::npos);
    CHECK(text.find("pe: a PE warning\n") != std::string::npos);
    CHECK(text.find("core\n\n") == std::string::npos);  // the newline is not doubled
    logResetForTesting(nullptr, "0");
}

TEST(crash_is_ours) {
    registerCrashImage(reinterpret_cast<const void*>(&faultHere));
    const uintptr_t here = reinterpret_cast<uintptr_t>(&faultHere);
    CHECK(crashIsOurs(here, nullptr, 0));
    CHECK(!crashIsOurs(0x1000, nullptr, 0));
    const uintptr_t frames[] = {0x2000, here + 4};
    CHECK(crashIsOurs(0x1000, frames, 2));  // a system function called from ours
    const uintptr_t elsewhere[] = {0x2000, 0x3000};
    CHECK(!crashIsOurs(0x1000, elsewhere, 2));
}

TEST(crash_report_on_fault) {
    const std::string dir = tempDir(), path = dir + "/crash.log";
    const int status = inChild(path, [] { faultHere(); });
    CHECK(WIFSIGNALED(status));
    CHECK(WIFSIGNALED(status) && (WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
    const std::string text = readFile(path);
    CHECK(text.find("==== WineMetalGL crash: SIG") != std::string::npos);
    CHECK(text.find("breadcrumb before the crash") != std::string::npos);
    CHECK(text.find("faultHere") != std::string::npos);  // the faulting function, by name
    CHECK(text.find("==== end of crash report") != std::string::npos);
}

TEST(crash_report_chains_to_previous_handler) {
    const std::string dir = tempDir(), path = dir + "/chain.log";
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        // The process's own handler (Wine's) is there first.
        signal(SIGSEGV, exitFromPreviousHandler);
        signal(SIGBUS, exitFromPreviousHandler);
        logResetForTesting("0", path.c_str());
        registerCrashImage(reinterpret_cast<const void*>(&faultHere));
        installCrashHandlers();
        faultHere();
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 42);
    CHECK(readFile(path).find("WineMetalGL crash") != std::string::npos);
}

TEST(crash_report_on_abort) {
    const std::string dir = tempDir(), path = dir + "/abort.log";
    const int status = inChild(path, [] { std::abort(); });
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    CHECK(readFile(path).find("WineMetalGL crash: SIGABRT") != std::string::npos);
}

TEST(crash_report_on_uncaught_exception) {
    const std::string dir = tempDir(), path = dir + "/terminate.log";
    const int status = inChild(path, [] { throw std::runtime_error("out of patience"); });
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    const std::string text = readFile(path);
    CHECK(text.find("uncaught C++ exception") != std::string::npos);
    CHECK(text.find("out of patience") != std::string::npos);
}
