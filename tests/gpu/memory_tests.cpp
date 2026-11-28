// Memory that GL objects hold on to after they are deleted or replaced: a
// game that streams textures, vertex arrays and buffers every frame (issue
// #16) must not grow the Metal device's allocations, nor the host memory a
// 32-bit application's buffers live in.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "gpu.h"

namespace {

unsigned long long deviceBytes() {
    unsigned long long value = 0;
    CHECK(wmgl_stats_counter("deviceAllocatedBytes", &value));
    return value;
}

// The host allocator the Wine module installs for 32-bit applications
// (wmgl_set_buffer_memory): here, aligned heap memory counted while out.
std::atomic<long long> g_hostBytes{0};
void* hostAllocate(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 65536, bytes)) return nullptr;
    g_hostBytes += static_cast<long long>(bytes);
    return p;
}
void hostRelease(void* p, size_t bytes) {
    g_hostBytes -= static_cast<long long>(bytes);
    std::free(p);
}

// One frame of streamed draws, as SOMA's (issue #16): per draw, a uniform
// buffer the GPU is still reading is updated in part (glBufferSubData, and
// a range-invalidating map), so each update gives the buffer new storage,
// the old one to go once the GPU is done with it. Then a vertex buffer
// specified again, two new textures (DXT1 and RGBA8) sampled and deleted, a
// vertex array made and deleted, and an occlusion query.
struct Streamer {
    GLuint program = 0, vbo = 0, stream = 0, query = 0, scratch = 0, ubos[4] = {};
    std::vector<uint8_t> dxt1 = std::vector<uint8_t>(256 * 256 / 2, 0x55), rgba = std::vector<uint8_t>(256 * 256 * 4, 0x80),
                         data = std::vector<uint8_t>(64 * 1024, 0x22);

    Streamer() {
        program = gpu::program("#version 330 core\nlayout(location = 0) in vec2 pos;\nlayout(std140) uniform U { vec4 c; };\n"
                               "out vec2 uv;\nout vec4 color;\n"
                               "void main() { uv = pos * 0.5 + 0.5; color = c; gl_Position = vec4(pos, 0.0, 1.0); }\n",
                               "#version 330 core\nin vec2 uv;\nin vec4 color;\nuniform sampler2D a;\nuniform sampler2D b;\n"
                               "out vec4 o;\nvoid main() { o = color + texture(a, uv) + texture(b, uv); }\n");
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "a"), 0);
        glUniform1i(glGetUniformLocation(program, "b"), 1);
        glUniformBlockBinding(program, glGetUniformBlockIndex(program, "U"), 0);
        const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glGenBuffers(4, ubos);
        for (GLuint u : ubos) {
            glBindBuffer(GL_UNIFORM_BUFFER, u);
            glBufferData(GL_UNIFORM_BUFFER, 64 * 1024, nullptr, GL_STREAM_DRAW);
        }
        glGenBuffers(1, &stream);
        glGenBuffers(1, &scratch);
        glGenQueries(1, &query);
    }
    ~Streamer() {
        glDeleteQueries(1, &query);
        glDeleteBuffers(1, &stream);
        glDeleteBuffers(1, &scratch);
        glDeleteBuffers(4, ubos);
        glDeleteBuffers(1, &vbo);
        glDeleteProgram(program);
    }

    void frame(int f) {
        GLuint textures[2] = {}, vao = 0;
        glGenTextures(2, textures);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures[0]);
        glCompressedTexImage2D(GL_TEXTURE_2D, 0, 0x83F0 /* GL_COMPRESSED_RGB_S3TC_DXT1_EXT */, 256, 256, 0,
                               static_cast<GLsizei>(dxt1.size()), dxt1.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBeginQuery(GL_SAMPLES_PASSED, query);
        for (int d = 0; d < 32; ++d) {
            const GLuint u = ubos[d % 4];
            glBindBuffer(GL_UNIFORM_BUFFER, u);
            if (d % 8 < 4) {
                glBufferSubData(GL_UNIFORM_BUFFER, 256, 256, data.data());
            } else if (void* at = glMapBufferRange(GL_UNIFORM_BUFFER, 512, 256, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT)) {
                std::memset(at, d, 256);
                glUnmapBuffer(GL_UNIFORM_BUFFER);
            }
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, u, 0, 256);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        glEndQuery(GL_SAMPLES_PASSED);
        GLuint ready = 0;
        glGetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &ready);
        glBindBuffer(GL_ARRAY_BUFFER, stream);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size()), data.data(), GL_STREAM_DRAW);
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
        glDeleteTextures(2, textures);
        if (f % 2 == 1) glFlush();
    }
};

// `measure` once the GPU is done and Metal has let go of what it used. Metal
// drops storage on its own completion thread, possibly after glFinish
// returns, and host memory goes back to the host at the next allocation:
// allocate (a small buffer), wait a little, until the value stays put.
template <typename Measure>
long long settled(Streamer& s, Measure measure) {
    long long value = 0, previous = -1;
    for (int i = 0; i < 200 && value != previous; ++i) {
        previous = value;
        glFinish();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        glBindBuffer(GL_COPY_WRITE_BUFFER, s.scratch);
        glBufferData(GL_COPY_WRITE_BUFFER, 16, nullptr, GL_STREAM_DRAW);
        value = measure();
    }
    return value;
}

// Runs `frames` streamed frames after a warm-up and returns how much
// `measure` grew.
template <typename Measure>
long long growth(Streamer& s, int frames, Measure measure) {
    for (int f = 0; f < 100; ++f) s.frame(f);
    const long long before = settled(s, measure);
    for (int f = 0; f < frames; ++f) s.frame(f);
    return settled(s, measure) - before;
}

}  // namespace

GPU_TEST_VERSION(gpu_memory_streamed_objects_are_freed, 46) {
    // Before the fix, every replaced uniform storage (2 MiB a frame here)
    // stayed: Metal's autoreleased objects were never drained (ARC
    // autoreleases the receiver of `contents`, and render encoders, which
    // hold their command buffers and so all they used). The bound allows for
    // the device's own pools: on the GPU of macOS 15 virtual machines (CI),
    // its allocations step up 16 MiB at a time; the leak grows ~2 GiB over
    // these 1000 frames.
    Streamer s;
    const long long grew = growth(s, 1000, [] { return static_cast<long long>(deviceBytes()); });
    if (grew >= 256ll << 20) std::fprintf(stderr, "  device allocations grew %.1f MiB over 1000 frames\n", static_cast<double>(grew) / 1048576.0);
    CHECK(grew < 256ll << 20);
    CHECK_GL();
}

TEST(gpu_memory_info_fits_32_bit_programs) {
    // A 32-bit program (buffer storage in host memory, as the Wine module
    // places it for them) is told at most 2047 MiB of GPU memory, which it
    // can count in bytes in a signed 32-bit integer (City of Heroes, #94);
    // 64-bit programs get Metal's working set.
    constexpr GLint kCap = 2047 * 1024;
    GLint wide = 0;
    {
        gpu::Fixture fixture(64, 64, false, 46);
        CHECK(fixture.ok);
        if (fixture.ok) glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &wide);
    }
    wmgl_set_buffer_memory(hostAllocate, hostRelease);
    {
        gpu::Fixture fixture(64, 64, false, 46);
        CHECK(fixture.ok);
        if (fixture.ok) {
            GLint dedicated = 0, total = 0, available = 0, ati[4] = {}, vbo[4] = {};
            glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &dedicated);
            glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &total);
            glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &available);
            glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, ati);
            glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, vbo);
            CHECK_EQ(dedicated, std::min(wide, kCap));
            CHECK_EQ(total, dedicated);
            CHECK(available > 0 && available <= dedicated);
            CHECK(ati[0] > 0 && ati[0] <= kCap && ati[1] == ati[0]);
            CHECK(vbo[0] > 0 && vbo[0] <= kCap);
            CHECK_GL();
        }
    }
    wmgl_set_buffer_memory(nullptr, nullptr);
    if (wide <= kCap) std::fprintf(stderr, "  this GPU's working set (%d KiB) is under the cap; the cap is not exercised\n", wide);
}

TEST(gpu_memory_streamed_buffers_with_host_memory_are_freed) {
    // The same frames with buffer storage in host memory, as the Wine module
    // places it for 32-bit applications: the host memory out levels off.
    wmgl_set_buffer_memory(hostAllocate, hostRelease);
    {
        gpu::Fixture fixture(64, 64, true, 46);
        CHECK(fixture.ok);
        if (fixture.ok) {
            Streamer s;
            // Metal on the GPU of macOS 15 virtual machines (CI) holds on to
            // released buffers in host memory, up to a bound: measured, it
            // filled to about 125 MiB at some point of the run and stayed
            // there through 1000 frames. Over 1000 frames, that pool fits
            // under 256 MiB whenever it fills; the leak this guards against
            // grows ~2 GiB.
            const long long grew = growth(s, 1000, [] { return g_hostBytes.load(); });
            if (grew >= 256ll << 20) std::fprintf(stderr, "  host memory grew %.1f MiB over 1000 frames\n", static_cast<double>(grew) / 1048576.0);
            CHECK(grew < 256ll << 20);
            CHECK_GL();
        }
    }
    wmgl_set_buffer_memory(nullptr, nullptr);
}
