// Crash reports: faults in WineMetalGL's code, aborts and uncaught
// exceptions are written to the log file (log.h) and stderr, with a backtrace
// and the recent history, before the process's own handling (Wine's) goes on.
//
// Wine handles SIGSEGV and its kin itself (Windows exceptions, write watches,
// guard pages), so a fault is reported only when it is WineMetalGL's: its
// program counter, or a frame of its backtrace, in an image of WineMetalGL's
// (libwinemetalgl.dylib, winemetalgl.so) and its address not a mapped page
// that Wine may be watching. The previous handler always runs afterwards.
// WINEMETALGL_CRASH_HANDLER=0 installs nothing.
#pragma once

#include <cstdint>

namespace wmgl {

// Installs the handlers, once (later calls do nothing).
void installCrashHandlers();

// Counts the image holding this address as WineMetalGL's (the core and every
// loaded image named *winemetalgl* are counted at install). For tests.
void registerCrashImage(const void* addressInImage);

// Whether a fault at pc with this backtrace is in WineMetalGL's code (the
// handler's first test, exposed for tests).
bool crashIsOurs(uintptr_t pc, const uintptr_t* frames, int frameCount);

// Writes a crash report (title: detail, the caller's backtrace, the recent
// history) from ordinary code: uncaught exceptions. Not for signal handlers.
void crashReport(const char* title, const char* detail);

}  // namespace wmgl
