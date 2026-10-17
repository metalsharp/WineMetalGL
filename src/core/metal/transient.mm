#include "core/metal/transient.h"

#include <algorithm>

#include "core/metal/device.h"
#include "core/stats.h"
#include "core/util/log.h"

namespace wmgl {

namespace {
constexpr size_t kChunkSize = 4u << 20;
}

TransientAllocation TransientAllocator::allocate(size_t size, uint64_t serial, size_t alignment) {
    size = std::max<size_t>(size, 4);
    stats::count(&stats::Frame::transientBytes, size);
    // Each context owns its allocator and only appends, so earlier regions of
    // a chunk stay valid while their command buffers run. A chunk is rewound
    // only once the last command buffer that used it has completed.
    if (current_ != SIZE_MAX) {
        Chunk& chunk = chunks_[current_];
        const size_t start = (chunk.used + alignment - 1) / alignment * alignment;
        if (start + size <= chunk.buffer.length) {
            chunk.used = start + size;
            chunk.lastSerial = std::max(chunk.lastSerial, serial);
            return {chunk.buffer, start, static_cast<uint8_t*>(chunk.buffer.contents) + start};
        }
    }
    for (size_t i = 0; i < chunks_.size(); ++i) {
        Chunk& chunk = chunks_[i];
        if (i == current_ || chunk.buffer.length < size) continue;
        if (chunk.lastSerial == serial || !queue_.isComplete(chunk.lastSerial)) continue;
        chunk.used = size;
        chunk.lastSerial = serial;
        current_ = i;
        return {chunk.buffer, 0, chunk.buffer.contents};
    }
    if (size > kChunkSize) {
        // Chunks made for one large allocation go once their work is done,
        // so a few huge draws do not keep their memory.
        for (size_t i = chunks_.size(); i-- > 0;) {
            if (i == current_ || chunks_[i].buffer.length <= kChunkSize || !queue_.isComplete(chunks_[i].lastSerial)) continue;
            chunks_.erase(chunks_.begin() + static_cast<long>(i));
            if (current_ != SIZE_MAX && current_ > i) --current_;
        }
    }
    Chunk chunk;
    chunk.buffer = [Device::get().mtl() newBufferWithLength:std::max(kChunkSize, size)
                                                    options:MTLResourceStorageModeShared |
                                                            MTLResourceCPUCacheModeWriteCombined];
    if (!chunk.buffer) {
        logOnce(LogLevel::Warn, "transient-memory", "no GPU memory for %zu bytes of per-draw data", size);
        return {};
    }
    chunk.buffer.label = @"WineMetalGL transient";
    chunk.used = size;
    chunk.lastSerial = serial;
    chunks_.push_back(chunk);
    current_ = chunks_.size() - 1;
    return {chunk.buffer, 0, chunk.buffer.contents};
}

}  // namespace wmgl
