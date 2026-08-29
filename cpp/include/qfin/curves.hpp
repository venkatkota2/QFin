#pragma once

#include <cstddef>
#include <span>

namespace qfin {

void validate_curve(std::span<const double> times, std::span<const double> zero_rates);

[[nodiscard]] double interpolate_flat_linear(
    double time,
    std::span<const double> curve_times,
    std::span<const double> values
);

[[nodiscard]] double discount_factor(
    double time,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    double parallel_shift = 0.0
);

}  // namespace qfin
