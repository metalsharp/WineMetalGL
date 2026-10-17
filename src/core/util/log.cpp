#include "core/util/log.h"

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

extern char** environ;

namespace wmgl {
namespace {

const char* const kCategoryNames[kLogCategoryCount] = {
    "general", "driver", "context", "surface", "draw", "shader", "pipeline", "texture",
    "buffer",  "gpu",    "api",     "perf",    "wine", "pe",     "crash"};
const char* const kLevelNames[] = {"err", "warn", "info", "trace"};

constexpr int kDefaultStderrLevel = static_cast<int>(LogLevel::Error);
constexpr int kDefaultFileLevel = static_cast<int>(LogLevel::Warn);
constexpr int kCaptureLevel = static_cast<int>(LogLevel::Info);
// Log files kept in the default directory (the oldest go when a new one opens).
constexpr int kKeptFiles = 30;
constexpr unsigned long long kDefaultMaxFileBytes = 64ull << 20;

// The recent history: the last kSlots messages at info or above, as the
// file has them. Written by any thread, read by the crash handler, which
// must not lock: a message being written as it reads may come out torn.
constexpr size_t kSlots = 512, kSlotBytes = 320;
struct Slot {
    std::atomic<uint32_t> length{0};
    char text[kSlotBytes];
};
Slot g_history[kSlots];
std::atomic<uint64_t> g_historyHead{0};

// The banner, kept apart from the history so that a busy log does not push
// it out. Written once, before it is published.
char g_banner[8192];
std::atomic<size_t> g_bannerLength{0};

struct State {
    LogLevels levels;
    std::string spec;           // WINEMETALGL_LOG as given
    std::string fileSetting;    // WINEMETALGL_LOG_FILE as given ("" unset)
    bool fileSettingGiven = false;
    unsigned long long maxFileBytes = kDefaultMaxFileBytes;
    std::string program, hostVersion;
    bool pathChosen = false;
    bool defaultDirectory = false;
    char path[1024] = {};       // the file's path, "" for none; fixed once chosen (read by the crash handler)
    std::atomic<int> fd{-1};
    bool openFailed = false;
    bool limitReached = false;
    unsigned long long written = 0;
    std::mutex mutex;           // the file, path and host
};

State& state() {
    static State* s = [] {
        auto* s = new State();  // never destroyed: messages may come from other threads at exit
        const char* spec = std::getenv("WINEMETALGL_LOG");
        s->spec = spec ? spec : "";
        s->levels = parseLogLevels(spec, std::getenv("WINEMETALGL_WINE_DEBUG") != nullptr);
        if (const char* file = std::getenv("WINEMETALGL_LOG_FILE")) {
            s->fileSetting = file;
            s->fileSettingGiven = true;
        }
        if (const char* mb = std::getenv("WINEMETALGL_LOG_MAX_MB")) {
            const long long v = std::atoll(mb);
            if (v > 0) s->maxFileBytes = static_cast<unsigned long long>(v) << 20;
        }
        return s;
    }();
    return *s;
}

int parseLevel(const std::string& text, bool& ok) {
    ok = true;
    if (text == "off" || text == "none" || text == "-1") return -1;
    if (text == "0" || text == "err" || text == "error" || text == "errors") return 0;
    if (text == "1" || text == "warn" || text == "warning" || text == "warnings") return 1;
    if (text == "2" || text == "info") return 2;
    if (text == "3" || text == "trace" || text == "all") return 3;
    // Higher numbers are trace (WINEMETALGL_LOG=9 has always meant everything).
    if (!text.empty() && std::all_of(text.begin(), text.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
        return 3;
    ok = false;
    return 0;
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

uint64_t threadId() {
    uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    return id;
}

// "HH:MM:SS.mmm pid:tid lvl  category: " (the file's and history's prefix).
std::string prefix(const char* level, LogCategory category) {
    timeval now;
    gettimeofday(&now, nullptr);
    tm local;
    const time_t seconds = now.tv_sec;
    localtime_r(&seconds, &local);
    char text[96];
    std::snprintf(text, sizeof(text), "%02d:%02d:%02d.%03d %d:%llu %-5s %s: ", local.tm_hour, local.tm_min, local.tm_sec,
                  static_cast<int>(now.tv_usec / 1000), static_cast<int>(getpid()),
                  static_cast<unsigned long long>(threadId()), level, logCategoryName(category));
    return text;
}

void remember(const std::string& line) {
    Slot& slot = g_history[g_historyHead.fetch_add(1, std::memory_order_relaxed) % kSlots];
    slot.length.store(0, std::memory_order_relaxed);
    size_t n = std::min(line.size(), kSlotBytes);
    std::memcpy(slot.text, line.data(), n);
    if (n == kSlotBytes) slot.text[n - 1] = '\n';  // cut short, still a line
    slot.length.store(static_cast<uint32_t>(n), std::memory_order_release);
}

void writeAll(int fd, const char* data, size_t size) {
    while (size) {
        const ssize_t n = ::write(fd, data, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return;
        data += n;
        size -= static_cast<size_t>(n);
    }
}

void writeString(int fd, const char* text) { writeAll(fd, text, std::strlen(text)); }

std::string homeDirectory() {
    if (const char* home = std::getenv("HOME"); home && *home) return home;
    if (const passwd* pw = getpwuid(getuid()); pw && pw->pw_dir) return pw->pw_dir;
    return "/tmp";
}

std::string sanitize(const std::string& name) {
    std::string out;
    for (char c : name) out += std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_' ? c : '_';
    return out.empty() ? "process" : out;
}

std::string programName(const State& s) {
    std::string name = s.program;
    if (name.empty()) name = getprogname() ? getprogname() : "process";
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    return sanitize(name);
}

void makeDirectories(const std::string& path) {
    for (size_t at = 1; at <= path.size(); ++at) {
        if (at == path.size() || path[at] == '/') ::mkdir(path.substr(0, at).c_str(), 0755);
    }
}

bool isDirectory(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Chooses the file's path (once; under the mutex). Creates its directory, so
// that the crash handler can create the file without doing so.
void choosePath(State& s) {
    if (s.pathChosen) return;
    s.pathChosen = true;
    std::string setting = s.fileSetting;
    if (s.fileSettingGiven && (setting.empty() || setting == "0" || lower(setting) == "off" || lower(setting) == "none")) return;
    char stamp[32];
    const time_t now = time(nullptr);
    tm local;
    localtime_r(&now, &local);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const std::string name = programName(s) + "-" + stamp + "-" + std::to_string(getpid()) + ".log";
    std::string dir, path;
    if (!s.fileSettingGiven) {
        dir = homeDirectory() + "/Library/Logs/WineMetalGL";
        s.defaultDirectory = true;
    } else if (setting.back() == '/' || isDirectory(setting)) {
        dir = setting;
        while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    } else {
        // A file; %p is the process ID.
        for (size_t at; (at = setting.find("%p")) != std::string::npos;) setting.replace(at, 2, std::to_string(getpid()));
        path = setting;
        const size_t slash = path.find_last_of('/');
        if (slash != std::string::npos && slash) makeDirectories(path.substr(0, slash));
    }
    if (path.empty()) {
        makeDirectories(dir);
        path = dir + "/" + name;
    }
    if (path.size() < sizeof(s.path)) std::memcpy(s.path, path.c_str(), path.size() + 1);
}

// The oldest log files in the default directory go, so that it keeps kKeptFiles.
void pruneOldFiles(const State& s) {
    std::string dir = s.path;
    dir = dir.substr(0, dir.find_last_of('/'));
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::pair<time_t, std::string>> files;
    while (const dirent* e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < 5 || name.compare(name.size() - 4, 4, ".log")) continue;
        struct stat st;
        const std::string full = dir + "/" + name;
        if (::stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) files.push_back({st.st_mtime, full});
    }
    closedir(d);
    if (files.size() < static_cast<size_t>(kKeptFiles)) return;
    std::sort(files.begin(), files.end());
    for (size_t i = 0; i + kKeptFiles <= files.size(); ++i) ::unlink(files[i].second.c_str());
}

// Opens the file with O_APPEND and writes its head; async-signal-safe when
// the path is chosen (the crash handler's way in). Returns the descriptor or -1.
int openFile(State& s) {
    int fd = s.fd.load(std::memory_order_acquire);
    if (fd >= 0 || !s.path[0]) return fd;
    const int opened = ::open(s.path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (opened < 0) return -1;
    int expected = -1;
    if (!s.fd.compare_exchange_strong(expected, opened)) {
        ::close(opened);
        return expected;
    }
    writeString(opened, "==== WineMetalGL log ====\n");
    if (const size_t n = g_bannerLength.load(std::memory_order_acquire)) writeAll(opened, g_banner, n);
    writeString(opened, "---- recent history ----\n");
    logWriteHistory(opened);
    writeString(opened, "---- live ----\n");
    return opened;
}

// The file, opened if need be (under the mutex). The history it starts with
// has the message being written: true when the caller still has to write it.
int fileForWriting(State& s, bool& opened) {
    opened = false;
    choosePath(s);
    if (s.limitReached || s.openFailed || !s.path[0]) return -1;
    int fd = s.fd.load(std::memory_order_acquire);
    if (fd >= 0) return fd;
    if (s.defaultDirectory) pruneOldFiles(s);
    fd = openFile(s);
    if (fd < 0) {
        s.openFailed = true;
        std::fprintf(stderr, "winemetalgl:err: cannot open the log file %s: %s\n", s.path, std::strerror(errno));
        return -1;
    }
    opened = true;
    if (!s.levels.valid)
        writeString(fd, "(WINEMETALGL_LOG has a part that is not a level or category:level; it is ignored)\n");
    return fd;
}

void writeToFile(State& s, const std::string& line, bool inHistory) {
    std::lock_guard<std::mutex> lock(s.mutex);
    bool opened;
    const int fd = fileForWriting(s, opened);
    if (fd < 0) return;
    if (!(opened && inHistory)) writeAll(fd, line.data(), line.size());
    s.written += line.size();
    if (s.written > s.maxFileBytes) {
        s.limitReached = true;
        writeString(fd, "==== the log file reached its limit (WINEMETALGL_LOG_MAX_MB); nothing more is written ====\n");
    }
}

void emit(LogLevel level, LogCategory category, const char* message) {
    State& s = state();
    const int l = static_cast<int>(level), c = static_cast<int>(category);
    std::string text = message;
    while (!text.empty() && text.back() == '\n') text.pop_back();
    const bool toStderr = l <= s.levels.stderrLevel[c];
    const bool toFile = l <= s.levels.fileLevel[c];
    const bool toHistory = l <= kCaptureLevel;
    if (toStderr) {
        const std::string line = std::string("winemetalgl:") + kLevelNames[l] + ": " + text + "\n";
        std::fwrite(line.data(), 1, line.size(), stderr);
    }
    if (!toFile && !toHistory) return;
    const std::string line = prefix(kLevelNames[l], category) + text + "\n";
    if (toHistory) remember(line);
    if (toFile) writeToFile(s, line, toHistory);
}

std::string format(const char* fmt, va_list args) {
    char stack[1024];
    va_list copy;
    va_copy(copy, args);
    const int n = std::vsnprintf(stack, sizeof(stack), fmt, copy);
    va_end(copy);
    if (n < 0) return fmt;
    if (static_cast<size_t>(n) < sizeof(stack)) return std::string(stack, static_cast<size_t>(n));
    std::string big(static_cast<size_t>(n) + 1, '\0');
    std::vsnprintf(big.data(), big.size(), fmt, args);
    big.resize(static_cast<size_t>(n));
    return big;
}

std::string sysctlString(const char* name) {
    char value[256];
    size_t size = sizeof(value);
    if (sysctlbyname(name, value, &size, nullptr, 0) != 0 || !size) return "";
    return std::string(value, strnlen(value, size));
}

long long sysctlInt(const char* name) {
    long long value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return -1;
    if (size == sizeof(int)) return *reinterpret_cast<int*>(&value);
    return value;
}

void addBannerLine(std::string& banner, const std::string& text) {
    banner += prefix("info", LogCategory::General) + text + "\n";
}

}  // namespace

const char* logCategoryName(LogCategory category) {
    const int c = static_cast<int>(category);
    return c >= 0 && c < kLogCategoryCount ? kCategoryNames[c] : "general";
}

bool logCategoryFromName(const char* name, LogCategory& category) {
    if (!name) return false;
    const std::string n = lower(name);
    for (int c = 0; c < kLogCategoryCount; ++c) {
        if (n == kCategoryNames[c]) {
            category = static_cast<LogCategory>(c);
            return true;
        }
    }
    return false;
}

LogLevels parseLogLevels(const char* spec, bool wineDebug) {
    LogLevels levels;
    std::fill(std::begin(levels.stderrLevel), std::end(levels.stderrLevel), kDefaultStderrLevel);
    std::fill(std::begin(levels.fileLevel), std::end(levels.fileLevel), kDefaultFileLevel);
    std::vector<std::string> items;
    if (spec) {
        std::string item;
        for (const char* p = spec;; ++p) {
            if (!*p || *p == ',' || *p == ';' || std::isspace(static_cast<unsigned char>(*p))) {
                if (!item.empty()) items.push_back(lower(item));
                item.clear();
                if (!*p) break;
            } else {
                item += *p;
            }
        }
    }
    // A level alone (or all:level) first, whatever the order, then categories over it.
    for (const auto& item : items) {
        std::string name, value = item;
        if (const size_t sep = item.find_first_of(":="); sep != std::string::npos) {
            name = trim(item.substr(0, sep));
            value = trim(item.substr(sep + 1));
        }
        if (!name.empty() && name != "all" && name != "*") continue;
        bool ok;
        const int level = parseLevel(value, ok);
        if (!ok) {
            levels.valid = false;
            continue;
        }
        std::fill(std::begin(levels.stderrLevel), std::end(levels.stderrLevel), level);
        std::fill(std::begin(levels.fileLevel), std::end(levels.fileLevel), level < 0 ? -1 : std::max(level, kDefaultFileLevel));
    }
    // WINEMETALGL_WINE_DEBUG: the Wine module's trace, as before.
    if (wineDebug) {
        for (LogCategory c : {LogCategory::Wine, LogCategory::Pe}) {
            levels.stderrLevel[static_cast<int>(c)] = levels.fileLevel[static_cast<int>(c)] = 3;
        }
    }
    for (const auto& item : items) {
        const size_t sep = item.find_first_of(":=");
        if (sep == std::string::npos) continue;
        const std::string name = trim(item.substr(0, sep));
        if (name.empty() || name == "all" || name == "*") continue;
        LogCategory category;
        bool ok;
        const int level = parseLevel(trim(item.substr(sep + 1)), ok);
        if (!ok || !logCategoryFromName(name.c_str(), category)) {
            levels.valid = false;
            continue;
        }
        levels.stderrLevel[static_cast<int>(category)] = level;
        levels.fileLevel[static_cast<int>(category)] = level < 0 ? -1 : std::max(level, kDefaultFileLevel);
    }
    return levels;
}

bool logEnabled(LogLevel level, LogCategory category) {
    const State& s = state();
    const int l = static_cast<int>(level), c = static_cast<int>(category);
    return l <= s.levels.stderrLevel[c] || l <= s.levels.fileLevel[c];
}

bool logCaptured(LogLevel level, LogCategory category) {
    return static_cast<int>(level) <= kCaptureLevel || logEnabled(level, category);
}

void logMessageV(LogLevel level, LogCategory category, const char* fmt, va_list args) {
    if (!logCaptured(level, category)) return;
    emit(level, category, format(fmt, args).c_str());
}

void logMessage(LogLevel level, LogCategory category, const char* fmt, ...) {
    if (!logCaptured(level, category)) return;
    va_list args;
    va_start(args, fmt);
    logMessageV(level, category, fmt, args);
    va_end(args);
}

bool logOnce(LogLevel level, LogCategory category, const char* key, const char* fmt, ...) {
    if (!logCaptured(level, category)) return false;
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!seen.insert(std::string(logCategoryName(category)) + "/" + key).second) return false;
    }
    va_list args;
    va_start(args, fmt);
    logMessageV(level, category, fmt, args);
    va_end(args);
    return logEnabled(level, category);
}

void logText(LogLevel level, LogCategory category, const char* text) {
    if (!text || !logCaptured(level, category)) return;
    emit(level, category, text);
}

void logRaw(LogCategory category, const char* line) {
    std::string text = line;
    while (!text.empty() && text.back() == '\n') text.pop_back();
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fputc('\n', stderr);
    // Not into the history: per-frame statistics would push out what matters in a crash.
    writeToFile(state(), prefix("info", category) + text + "\n", false);
}

void logRawFormat(LogCategory category, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const std::string text = format(fmt, args);
    va_end(args);
    logRaw(category, text.c_str());
}

void logSetHost(const char* program, const char* hostVersion) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (program && *program) s.program = program;
    if (hostVersion && *hostVersion) s.hostVersion = hostVersion;
}

void logBanner(const char* version, const char* commit, const char* gpu) {
    static std::once_flag once;
    std::call_once(once, [&] {
        State& s = state();
        std::string program, host, path;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            choosePath(s);
            program = s.program.empty() ? (getprogname() ? getprogname() : "?") : s.program;
            host = s.hostVersion;
            path = s.path;
        }
        std::string banner;
        addBannerLine(banner, std::string("WineMetalGL ") + (version ? version : "?") + " (" + (commit && *commit ? commit : "unknown commit") +
                                  ", built " __DATE__ ")");
        const bool translated = sysctlInt("sysctl.proc_translated") == 1;
#if defined(__x86_64__)
        const char* arch = translated ? "x86_64 under Rosetta" : "x86_64";
#elif defined(__aarch64__)
        const char* arch = "arm64";
#else
        const char* arch = "?";
#endif
        addBannerLine(banner, "process " + program + " (pid " + std::to_string(getpid()) + ", " + arch + ")");
        if (!host.empty()) addBannerLine(banner, "host " + host);
        const long long memory = sysctlInt("hw.memsize");
        addBannerLine(banner, "macOS " + sysctlString("kern.osproductversion") + " (" + sysctlString("kern.osversion") + "), " +
                                  sysctlString("hw.model") + ", " + std::to_string(sysctlInt("hw.logicalcpu")) + " CPUs, " +
                                  std::to_string(memory > 0 ? memory >> 30 : 0) + " GB");
        addBannerLine(banner, std::string("GPU ") + (gpu ? gpu : "none"));
        addBannerLine(banner, "log WINEMETALGL_LOG=" + (s.spec.empty() ? std::string("(unset)") : s.spec) + ", file " +
                                  (path.empty() ? std::string("none") : path));
        // Settings that change what WineMetalGL, Metal or Wine do.
        static const char* const prefixes[] = {"WINEMETALGL_", "DXMT_", "MTL_", "METAL_", "WINEDLLOVERRIDES=", "WINEDEBUG=",
                                               "WINEESYNC=",   "WINEMSYNC=", "WINE_LARGE_ADDRESS_AWARE=", "ROSETTA_"};
        for (char** e = environ; e && *e; ++e) {
            for (const char* p : prefixes) {
                if (!std::strncmp(*e, p, std::strlen(p))) {
                    addBannerLine(banner, std::string("env ") + *e);
                    break;
                }
            }
        }
        const size_t n = std::min(banner.size(), sizeof(g_banner));
        std::memcpy(g_banner, banner.data(), n);
        g_bannerLength.store(n, std::memory_order_release);
        // Also where info messages go (WINEMETALGL_LOG=2): stderr, and the
        // file, which opening writes the banner into.
        const int general = static_cast<int>(LogCategory::General);
        if (s.levels.stderrLevel[general] >= static_cast<int>(LogLevel::Info)) {
            for (size_t start = 0; start < banner.size();) {
                const size_t end = banner.find('\n', start);
                const std::string line = banner.substr(start, end - start);
                const size_t at = line.find("general: ");  // the text after the file's prefix
                const std::string out = "winemetalgl:info: " + (at == std::string::npos ? line : line.substr(at + 9)) + "\n";
                std::fwrite(out.data(), 1, out.size(), stderr);
                start = end == std::string::npos ? banner.size() : end + 1;
            }
        }
        std::lock_guard<std::mutex> lock(s.mutex);
        if (const int fd = s.fd.load(std::memory_order_acquire); fd >= 0) {
            // Opened before the banner was ready (a message as the core starts).
            writeString(fd, "---- banner ----\n");
            writeAll(fd, g_banner, n);
        } else if (s.levels.fileLevel[general] >= static_cast<int>(LogLevel::Info)) {
            bool opened;
            fileForWriting(s, opened);
        }
    });
}

const char* logFilePath() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    choosePath(s);
    return s.path;
}

int logFileDescriptor() { return openFile(state()); }

bool logFileOpen() { return state().fd.load(std::memory_order_acquire) >= 0; }

void logWriteHistory(int fd) {
    const uint64_t head = g_historyHead.load(std::memory_order_acquire);
    for (uint64_t i = head > kSlots ? head - kSlots : 0; i < head; ++i) {
        const Slot& slot = g_history[i % kSlots];
        const uint32_t n = slot.length.load(std::memory_order_acquire);
        if (n) writeAll(fd, slot.text, n);
    }
}

void logResetForTesting(const char* spec, const char* file) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.spec = spec ? spec : "";
    s.levels = parseLogLevels(spec, false);
    s.fileSetting = file ? file : "";
    s.fileSettingGiven = file != nullptr;
    s.pathChosen = s.defaultDirectory = s.openFailed = s.limitReached = false;
    s.written = 0;
    s.path[0] = 0;
    if (const int fd = s.fd.exchange(-1); fd >= 0) ::close(fd);
    for (Slot& slot : g_history) slot.length.store(0);
    g_historyHead.store(0);
    g_bannerLength.store(0);
}

}  // namespace wmgl
