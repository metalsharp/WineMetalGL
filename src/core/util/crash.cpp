#include "core/util/crash.h"

#include <dlfcn.h>
#include <execinfo.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <typeinfo>

#include "core/util/log.h"

namespace wmgl {
namespace {

// WineMetalGL's images' __TEXT ranges.
struct Range {
    uintptr_t start, end;
};
constexpr int kMaxRanges = 8;
Range g_ranges[kMaxRanges];
std::atomic<int> g_rangeCount{0};

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT};
struct sigaction g_previous[NSIG];
std::terminate_handler g_previousTerminate;
// Reports written (a fault the program survives may come back: the first few are enough).
std::atomic<int> g_reports{0};
constexpr int kMaxReports = 4;
std::atomic<bool> g_reporting{false};
// An uncaught exception was reported: the abort that follows is the same crash.
std::atomic<bool> g_exceptionReported{false};

void addRange(const mach_header* header) {
    if (!header) return;
    unsigned long size = 0;
    const uint8_t* text = getsegmentdata(reinterpret_cast<const mach_header_64*>(header), "__TEXT", &size);
    if (!text || !size) return;
    const uintptr_t start = reinterpret_cast<uintptr_t>(text);
    const int count = g_rangeCount.load();
    for (int i = 0; i < count; ++i)
        if (g_ranges[i].start == start) return;
    if (count >= kMaxRanges) return;
    g_ranges[count] = {start, start + size};
    g_rangeCount.store(count + 1, std::memory_order_release);
}

bool ours(uintptr_t address) {
    const int count = g_rangeCount.load(std::memory_order_acquire);
    for (int i = 0; i < count; ++i)
        if (address >= g_ranges[i].start && address < g_ranges[i].end) return true;
    return false;
}

// A line built without allocating (the handler cannot).
struct Text {
    char data[1024] = {};  // always terminated
    size_t size = 0;
    Text& add(const char* s) {
        while (s && *s && size < sizeof(data) - 1) data[size++] = *s++;
        return *this;
    }
    Text& hex(uintptr_t v) {
        char digits[2 + 16];
        int n = 0;
        do {
            digits[n++] = "0123456789abcdef"[v & 15];
            v >>= 4;
        } while (v);
        add("0x");
        while (n && size < sizeof(data) - 1) data[size++] = digits[--n];
        data[size] = 0;
        return *this;
    }
    Text& dec(long long v) {
        char digits[24];
        int n = 0;
        const bool negative = v < 0;
        unsigned long long u = negative ? 0ull - static_cast<unsigned long long>(v) : static_cast<unsigned long long>(v);
        do {
            digits[n++] = static_cast<char>('0' + u % 10);
            u /= 10;
        } while (u);
        if (negative) add("-");
        while (n && size < sizeof(data) - 1) data[size++] = digits[--n];
        data[size] = 0;
        return *this;
    }
    void write(int fd) const {
        const char* p = data;
        size_t left = size;
        while (left) {
            const ssize_t n = ::write(fd, p, left);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return;
            p += n;
            left -= static_cast<size_t>(n);
        }
    }
};

const char* signalName(int sig) {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    case SIGTRAP: return "SIGTRAP";
    case SIGABRT: return "SIGABRT";
    default: return "signal";
    }
}

// "image+0xoffset (symbol+0xoffset)" for an address.
void describe(Text& t, uintptr_t address) {
    t.hex(address);
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(address), &info) || !info.dli_fname) return;
    const char* image = std::strrchr(info.dli_fname, '/');
    t.add(" ").add(image ? image + 1 : info.dli_fname).add("+").hex(address - reinterpret_cast<uintptr_t>(info.dli_fbase));
    if (info.dli_sname) t.add(" (").add(info.dli_sname).add("+").hex(address - reinterpret_cast<uintptr_t>(info.dli_saddr)).add(")");
}

// The interrupted code's frames, from its frame pointer: return addresses
// read within the stack it ran on only. That is this thread's own stack, or
// (Wine runs unix calls on a stack of its own) the mapped region holding sp.
int walkFrames(uintptr_t fp, uintptr_t sp, uintptr_t* frames, int max) {
    pthread_t self = pthread_self();
    uintptr_t top = reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(self));
    uintptr_t bottom = top - pthread_get_stacksize_np(self);
    if (sp && (sp < bottom || sp >= top)) {
        mach_vm_address_t address = sp;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t region;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
                           reinterpret_cast<vm_region_info_t>(&region), &count, &object) != KERN_SUCCESS ||
            address > sp || !(region.protection & VM_PROT_READ))
            return 0;
        bottom = static_cast<uintptr_t>(address);
        top = static_cast<uintptr_t>(address + size);
    }
    int n = 0;
    while (n < max && fp >= bottom && fp + 2 * sizeof(uintptr_t) <= top && !(fp & (sizeof(uintptr_t) - 1))) {
        const uintptr_t* frame = reinterpret_cast<const uintptr_t*>(fp);
        const uintptr_t ret = frame[1];
        if (!ret) break;
        frames[n++] = ret;
        if (frame[0] <= fp) break;  // frames go up the stack
        fp = frame[0];
    }
    return n;
}

// The faulting pc, frame and stack pointers and (arm64) link register: a
// leaf function keeps its return address there, not in a frame.
void registersOf(void* context, uintptr_t& pc, uintptr_t& fp, uintptr_t& sp, uintptr_t& lr) {
    pc = fp = sp = lr = 0;
    if (!context) return;
    const ucontext_t* uc = static_cast<const ucontext_t*>(context);
    if (!uc->uc_mcontext) return;
#if defined(__x86_64__)
    pc = uc->uc_mcontext->__ss.__rip;
    fp = uc->uc_mcontext->__ss.__rbp;
    sp = uc->uc_mcontext->__ss.__rsp;
#elif defined(__aarch64__)
    pc = __darwin_arm_thread_state64_get_pc(uc->uc_mcontext->__ss);
    fp = __darwin_arm_thread_state64_get_fp(uc->uc_mcontext->__ss);
    lr = __darwin_arm_thread_state64_get_lr(uc->uc_mcontext->__ss);
    sp = __darwin_arm_thread_state64_get_sp(uc->uc_mcontext->__ss);
#endif
}

// A protection fault on a mapped page whose protection Wine narrowed (write
// watches, guard pages, its own bookkeeping): Wine resolves those and the
// code goes on, so they are not reported.
bool wineMayResolve(int sig, const siginfo_t* info) {
    if ((sig != SIGSEGV && sig != SIGBUS) || !info) return false;
    const uintptr_t target = reinterpret_cast<uintptr_t>(info->si_addr);
    if (target < 0x10000) return false;  // the null page
    mach_vm_address_t address = target;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t region;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&region),
                       &count, &object) != KERN_SUCCESS)
        return false;
    if (address > target) return false;  // unmapped: the region found is the next one
    return (region.protection & region.max_protection) != region.max_protection;
}

// The report: to the log file (with the history) and, shorter, to stderr.
void writeReport(const Text& title, bool ours, const uintptr_t* frames, int frameCount, uintptr_t pc) {
    const bool wasOpen = logFileOpen();
    const int fd = logFileDescriptor();  // opening it writes the banner and history
    Text head;
    head.add(ours ? "==== WineMetalGL crash: " : "==== process crash, outside WineMetalGL's code: ").add(title.data).add(" (thread ");
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    head.dec(static_cast<long long>(tid)).add(") ====\n");
    for (int out : {fd, STDERR_FILENO}) {
        if (out < 0) continue;
        head.write(out);
        if (pc) {
            Text line;
            line.add("at ");
            describe(line, pc);
            line.add("\n");
            line.write(out);
        }
        for (int i = 0; i < frameCount; ++i) {
            Text line;
            line.add("  #").dec(i).add(" ");
            describe(line, frames[i]);
            line.add("\n");
            line.write(out);
        }
    }
    if (fd >= 0) {
        if (wasOpen) {
            Text t;
            t.add("---- recent history ----\n").write(fd);
            logWriteHistory(fd);
        }
        Text t;
        t.add("==== end of crash report; the process's own handling follows ====\n").write(fd);
        Text where;
        where.add("winemetalgl:err: crash report and recent history in ").add(logFilePath()).add("\n").write(STDERR_FILENO);
    }
}

void chain(int sig, siginfo_t* info, void* context) {
    const struct sigaction& previous = g_previous[sig];
    if (previous.sa_flags & SA_SIGINFO) {
        if (previous.sa_sigaction) previous.sa_sigaction(sig, info, context);
        return;
    }
    if (previous.sa_handler == SIG_IGN) return;
    if (previous.sa_handler == SIG_DFL || !previous.sa_handler) {
        // The default action: the faulting instruction runs again and kills
        // the process; a signal sent (abort, raise) is sent again.
        struct sigaction dfl = {};
        dfl.sa_handler = SIG_DFL;
        sigemptyset(&dfl.sa_mask);
        sigaction(sig, &dfl, nullptr);
        if (!info || info->si_code == SI_USER || sig == SIGABRT) raise(sig);
        return;
    }
    previous.sa_handler(sig);
}

void onSignal(int sig, siginfo_t* info, void* context) {
    const int savedErrno = errno;
    uintptr_t pc, fp, sp, lr;
    registersOf(context, pc, fp, sp, lr);
    uintptr_t frames[48];
    int frameCount = 0;
    if (lr) frames[frameCount++] = lr;
    // Cheap (a region lookup at most): Wine takes many faults of its own
    // (guest code, write watches), which pass through here too.
    frameCount += walkFrames(fp, sp, frames + frameCount, 48 - frameCount);
    if (frameCount > 1 && frames[0] == frames[1]) {  // not a leaf: its frame has the same return address
        for (int i = 1; i < frameCount; ++i) frames[i - 1] = frames[i];
        --frameCount;
    }
    const bool ours = crashIsOurs(pc, frames, frameCount);
    // An abort is the process ending, wherever it comes from; faults only in WineMetalGL's code.
    const bool report = sig == SIGABRT ? !g_exceptionReported.load() : ours && !wineMayResolve(sig, info);
    if (report && g_reports.fetch_add(1) < kMaxReports && !g_reporting.exchange(true)) {
        Text title;
        title.add(signalName(sig));
        if (info) {
            title.add(" code ").dec(info->si_code);
            if (sig != SIGABRT) title.add(" address ").hex(reinterpret_cast<uintptr_t>(info->si_addr));
        }
        writeReport(title, ours, frames, frameCount, pc);
        g_reporting.store(false);
    }
    errno = savedErrno;
    chain(sig, info, context);
}

void onTerminate() {
    const char* what = "no exception";
    std::string text;
    if (std::exception_ptr current = std::current_exception()) {
        try {
            std::rethrow_exception(current);
        } catch (const std::exception& e) {
            text = std::string(typeid(e).name()) + ": " + e.what();
            what = text.c_str();
        } catch (...) {
            what = "an exception that is not a std::exception";
        }
    }
    crashReport("uncaught C++ exception (std::terminate)", what);
    if (g_previousTerminate) g_previousTerminate();
    std::abort();
}

}  // namespace

void registerCrashImage(const void* addressInImage) {
    Dl_info info;
    if (dladdr(addressInImage, &info) && info.dli_fbase) addRange(static_cast<const mach_header*>(info.dli_fbase));
}

bool crashIsOurs(uintptr_t pc, const uintptr_t* frames, int frameCount) {
    if (ours(pc)) return true;
    for (int i = 0; i < frameCount; ++i)
        if (ours(frames[i])) return true;
    return false;
}

void crashReport(const char* title, const char* detail) {
    if (g_reports.fetch_add(1) >= kMaxReports || g_reporting.exchange(true)) return;
    void* raw[48];
    const int n = backtrace(raw, 48);
    uintptr_t frames[48];
    // The first frames are this function and its caller's handler.
    int count = 0;
    for (int i = 1; i < n; ++i) frames[count++] = reinterpret_cast<uintptr_t>(raw[i]);
    Text t;
    t.add(title);
    if (detail && *detail) t.add(": ").add(detail);
    writeReport(t, true, frames, count, 0);
    g_exceptionReported.store(true);
    g_reporting.store(false);
}

void installCrashHandlers() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (const char* v = std::getenv("WINEMETALGL_CRASH_HANDLER"); v && !std::strcmp(v, "0")) return;
        registerCrashImage(reinterpret_cast<const void*>(&installCrashHandlers));
        for (uint32_t i = 0, n = _dyld_image_count(); i < n; ++i) {
            const char* name = _dyld_get_image_name(i);
            if (name && strcasestr(name, "winemetalgl")) addRange(_dyld_get_image_header(i));
        }
        logFilePath();  // chosen (and its directory made) now: the handler cannot
        struct sigaction action = {};
        action.sa_sigaction = onSignal;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        for (int sig : kSignals) sigaction(sig, &action, &g_previous[sig]);
        g_previousTerminate = std::set_terminate(onTerminate);
        WMGL_INFO(Crash, "crash handler installed (WINEMETALGL_CRASH_HANDLER=0 turns it off)");
    });
}

}  // namespace wmgl
