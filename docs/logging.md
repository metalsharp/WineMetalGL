# Logging and crash reports

WineMetalGL logs what it does, what goes wrong and how it crashed: the core
(`libwinemetalgl.dylib`), the Wine module's unix side (`winemetalgl.so`) and
its PE side (`winemetalgl.dll`) share one log, with the same levels, the same
file and the same crash history. The code is
[`src/core/util/log.h`](../src/core/util/log.h) and
[`src/core/util/crash.h`](../src/core/util/crash.h).

## Where messages go

| | By default | Chosen by |
|---|---|---|
| stderr | errors | `WINEMETALGL_LOG` |
| the log file | warnings and errors | `WINEMETALGL_LOG`, `WINEMETALGL_LOG_FILE` |
| the recent history | everything at info and above | (always) |

The log file is `~/Library/Logs/WineMetalGL/<program>-<date>-<pid>.log`, named
after the Windows program. It is created by its first message, so a program
that runs cleanly leaves no file. When it is created it starts with:

1. the **banner**: WineMetalGL's version and commit, the program, Wine's
   version, macOS and the Mac, the GPU, the log settings, and every
   `WINEMETALGL_*`, `DXMT_*`, `MTL_*`, `WINEDLLOVERRIDES` and `WINEDEBUG`
   setting;
2. the **recent history**: the last 512 messages at info and above, whatever
   the levels: the contexts and surfaces created, shaders that failed, and
   what else led up to the first warning;
3. then each message as it comes.

The directory keeps the 30 newest log files; older ones are deleted when a new
one is created. A file stops growing at 64 MiB (`WINEMETALGL_LOG_MAX_MB`).

Lines look like this (time, process:thread, level, category):

```
11:59:05.123 4242:91234 warn  shader: shader 7 failed to compile: ...
```

stderr keeps the old format (`winemetalgl:warn: ...`), which scripts parse.

## Levels and categories

`WINEMETALGL_LOG` sets a level for everything, for single categories, or both
(later categories win over the global level whatever the order):

```sh
WINEMETALGL_LOG=2                    # info everywhere
WINEMETALGL_LOG=shader:3,gpu:2       # trace shaders, GPU info, defaults elsewhere
WINEMETALGL_LOG=warn,wine:trace      # names work too
WINEMETALGL_LOG=api:off              # silence a category, errors included
```

Levels are `0`/`error`, `1`/`warn`, `2`/`info`, `3`/`trace`, and `off`. A
level applies to stderr and the file; the file never gets less than warnings.

| Category | What |
|---|---|
| `general` | startup, the banner, configuration |
| `driver` | the driver ABI: contexts, surfaces, make-current, refused context requests and why |
| `context` | GL contexts, GL errors (each command and error once, with its arguments), calls without a context |
| `surface` | windows, pbuffers, presentation, MetalFX |
| `draw` | draws, framebuffers (incomplete ones and why), queries |
| `shader` | GLSL, SPIR-V and ARB programs that fail to compile or link, their translation to MSL |
| `pipeline` | Metal libraries and pipeline states that fail to build |
| `texture` | textures, views, allocation failures |
| `buffer` | buffer objects, allocation failures |
| `gpu` | the Metal device, command buffers that fail (and, at info, which encoder faulted) |
| `api` | unimplemented or unsupported GL usage |
| `perf` | `WINEMETALGL_STATS` output (also written to the log file) |
| `wine` | the Wine module's unix side: SetPixelFormat, wglCreateContext, wglMakeCurrent, pbuffers, windows |
| `pe` | the Wine module's PE side |
| `crash` | the crash handler |

`WINEMETALGL_WINE_DEBUG=1` is `wine:trace,pe:trace`.

At `gpu:2` (info) and above, command buffers record each encoder's state, so
a failed command buffer's report says which encoder faulted. This costs a
little GPU time, so it is off by default.

## Crash reports

When the core starts, it installs handlers for SIGSEGV, SIGBUS, SIGILL,
SIGFPE, SIGTRAP and SIGABRT, and for uncaught C++ and Objective-C exceptions.
A crash report goes to the log file and, shorter, to stderr:

```
==== WineMetalGL crash: SIGSEGV code 1 address 0x18 (thread 91234) ====
at 0x1234abcd libwinemetalgl.dylib+0x5abcd (_ZN4wmgl7Surface4swapERNS_13CommandStreamE+0x4c)
  #0 0x1234a000 libwinemetalgl.dylib+0x5a000 (wmgl_surface_swap+0x80)
  #1 ...
---- recent history ----
...
==== end of crash report; the process's own handling follows ====
```

Wine uses these signals itself (Windows exceptions, write watches, guard
pages), so the handler reports a fault only when:

- the faulting instruction, or a return address on the stack, is in
  WineMetalGL's code (`libwinemetalgl.dylib`, `winemetalgl.so`), and
- the address is not a mapped page whose protection Wine has narrowed (which
  Wine resolves, and the program goes on).

Aborts are always reported. At most four reports are written per process.
Whatever it reports, the handler then passes the signal to the handler that
was there before (Wine's), so Wine's own handling is unchanged.
`WINEMETALGL_CRASH_HANDLER=0` installs no handlers.

Symbols in a report are mangled C++ names; `c++filt` turns them back:

```sh
c++filt < ~/Library/Logs/WineMetalGL/Game.exe-20261017-115905-4242.log
```

## When the core cannot be loaded

The log lives in the core, so `winemetalgl.so` cannot use it when the core
fails to load (a missing or unsigned `libwinemetalgl.dylib`). Its errors then
go to stderr and to `~/Library/Logs/WineMetalGL/winemetalgl-module.log`.

## For bug reports

Attach the program's log file from `~/Library/Logs/WineMetalGL`. For a
rendering problem, run with `WINEMETALGL_LOG=2` first; for a GPU hang or
corruption, `WINEMETALGL_LOG=2,gpu:3`.
