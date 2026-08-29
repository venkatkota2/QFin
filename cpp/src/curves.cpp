#include "qfin/curves.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qfin {

void validate_curve(const std::span<const double> times, const std::span<const double> zero_rates) {
    if (times.empty() || times.size() != zero_rates.size()) {
        throw std::invalid_argument("curve times and zero rates must have equal non-zero length");
    }
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (!std::isfinite(times[index]) || !std::isfinite(zero_rates[index]) || times[index] < 0.0) {
            throw std::invalid_argument("curve values must be finite and times non-negative");
        }
        if (index > 0 && !(times[index] > times[index - 1])) {
            throw std::invalid_argument("curve times must be strictly increasing");
        }
    }
}

double interpolate_flat_linear(
    const double time,
    const std::span<const double> curve_times,
    const std::span<const double> values
) {
    if (time <= curve_times.front()) {
        return values.front();
    }
    if (time >= curve_times.back()) {
        return values.back();
    }
    const auto upper = std::upper_bound(curve_times.begin(), curve_times.end(), time);
    const auto upper_index = static_cast<std::size_t>(upper - curve_times.begin());
    const auto lower_index = upper_index - 1;
    const double fraction = (time - curve_times[lower_index]) /
                            (curve_times[upper_index] - curve_times[lower_index]);
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index]);
}

double discount_factor(
    const double time,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const double parallel_shift
) {
    if (!std::isfinite(time) || time < 0.0 || !std::isfinite(parallel_shift)) {
        throw std::invalid_argument("discount time and shift must be finite and time non-negative");
    }
    const double rate = interpolate_flat_linear(time, curve_times, zero_rates) + parallel_shift;
    return std::exp(-rate * time);
}

}  // namespace qfin
