#pragma once

#include <cmath>
#include <span>
#include <stdexcept>

namespace qfin {
inline void require_finite_result(const double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("financial result exceeds the finite double range");
    }
}
inline void require_finite_result(const std::span<const double> values) {
    for (const auto value : values) {
        require_finite_result(value);
    }
}
}  // namespace qfin
