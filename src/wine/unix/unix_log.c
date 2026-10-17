/* winemetalgl.so: logging. Messages of the unix side ("wine") and of the PE
 * side ("pe", wmgl_unix_log) go to the core's log (wmgl_log_write: its
 * levels, log file and crash history) once the core is loaded. Before then,
 * or with a core that has no log (before 3.3.0), errors go to stderr and to
 * ~/Library/Logs/WineMetalGL/winemetalgl-module.log, so that a core that
 * cannot be loaded is on record too. */

#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "wmgl_unix.h"

int wmgl_log_max;
static int pe_log_max;
static void (*core_write)( int level, const char *category, const char *text );

static int fallback_level(void)
{
    return getenv( "WINEMETALGL_WINE_DEBUG" ) ? 3 : 0;
}

/* Before the core: stderr, and the module's own log file for errors. */
static void write_fallback( int level, const char *category, const char *text )
{
    static const char *const names[] = {"err", "warn", "info", "trace"};
    const char *home = getenv( "HOME" );
    char path[1024];
    time_t now = time( NULL );
    struct tm local;
    FILE *file;

    fprintf( stderr, "winemetalgl:%s: %s\n", names[level], text );
    if (level > 0 || !home || !*home) return;
    snprintf( path, sizeof(path), "%s/Library/Logs", home );
    mkdir( path, 0755 );
    snprintf( path, sizeof(path), "%s/Library/Logs/WineMetalGL", home );
    mkdir( path, 0755 );
    snprintf( path, sizeof(path), "%s/Library/Logs/WineMetalGL/winemetalgl-module.log", home );
    if (!(file = fopen( path, "a" ))) return;
    localtime_r( &now, &local );
    fprintf( file, "%04d-%02d-%02d %02d:%02d:%02d %d %s %s: %s\n", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
             local.tm_hour, local.tm_min, local.tm_sec, (int)getpid(), names[level], category, text );
    fclose( file );
}

static void log_text( int level, const char *category, char *text )
{
    size_t len = strlen( text );
    if (level < 0) return;
    if (level > 3) level = 3;
    while (len && text[len - 1] == '\n') text[--len] = 0;
    if (core_write) core_write( level, category, text );
    else write_fallback( level, category, text );
}

void wmgl_log( int level, const char *format, ... )
{
    char stack[1024], *text = stack;
    va_list args;
    int len;

    if (level > wmgl_log_max) return;
    va_start( args, format );
    len = vsnprintf( stack, sizeof(stack), format, args );
    va_end( args );
    if (len < 0) return;
    if ((size_t)len >= sizeof(stack) && (text = malloc( (size_t)len + 1 )))
    {
        va_start( args, format );
        vsnprintf( text, (size_t)len + 1, format, args );
        va_end( args );
    }
    if (!text) text = stack;
    log_text( level, "wine", text );
    if (text != stack) free( text );
}

void wmgl_log_start( void *core, const char *program, const char *host_version )
{
    void (*set_host)( const char *, const char * );
    int (*level)( const char * );

    wmgl_log_max = pe_log_max = fallback_level();
    if (!core) return;
    *(void **)&set_host = dlsym( core, "wmgl_log_set_host" );
    *(void **)&level = dlsym( core, "wmgl_log_level" );
    *(void **)&core_write = dlsym( core, "wmgl_log_write" );
    if (!set_host || !level || !core_write)
    {
        core_write = NULL;
        return;
    }
    set_host( program, host_version );
    wmgl_log_max = level( "wine" );
    pe_log_max = level( "pe" );
}

int wmgl_pe_log_level(void)
{
    return pe_log_max;
}

/* A message of the PE side (winemetalgl.dll). */
NTSTATUS wmgl_unix_log( void *args )
{
    struct wmgl_log_params *params = args;
    const char *text = (const char *)(ULONG_PTR)params->text;
    char *copy;

    if (!text || (int)params->level > pe_log_max) return STATUS_SUCCESS;
    if (!(copy = strdup( text ))) return STATUS_SUCCESS;
    log_text( (int)params->level, "pe", copy );
    free( copy );
    return STATUS_SUCCESS;
}
