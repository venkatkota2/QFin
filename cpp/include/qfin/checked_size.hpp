#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qfin {

// Shared with qfin._memory: persistent native buffers per operation, not RSS.
inline constexpr std::size_t max_operation_bytes = 512ULL * 1024ULL * 1024ULL;

inline std::size_t checked_add(const std::size_t a, const std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::length_error("native dimension addition exceeds size_t capacity");
    }
    return a + b;
}

inline std::size_t checked_multiply(const std::size_t a, const std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("native dimension product exceeds size_t capacity");
    }
    return a * b;
}

inline std::size_t checked_allocation(
    const std::size_t count, const std::size_t arrays = 1,
    const std::size_t item_size = sizeof(double)
) {
    const auto bytes = checked_multiply(checked_multiply(count, arrays), item_size);
    if (bytes > max_operation_bytes ||
        count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("native buffers exceed the 512 MiB operation allocation limit");
    }
    return count;
}

}  // namespace qfin
