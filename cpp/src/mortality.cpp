#include "qfin/mortality.hpp"

#include "qfin/curves.hpp"

#include <cmath>
#include <stdexcept>

namespace qfin {

void validate_mortality_table(const std::span<const double> ages, const std::span<const double> qx) {
    if (ages.empty() || ages.size() != qx.size()) {
        throw std::invalid_argument("mortality ages and qx must have equal non-zero length");
    }
    for (std::size_t index = 0; index < ages.size(); ++index) {
        if (!std::isfinite(ages[index]) || !std::isfinite(qx[index]) ||
            ages[index] < 0.0 || qx[index] < 0.0 || qx[index] > 1.0) {
            throw std::invalid_argument("mortality ages and probabilities are invalid");
        }
        if (index > 0 && !(ages[index] > ages[index - 1])) {
            throw std::invalid_argument("mortality ages must be strictly increasing");
        }
    }
}

double mortality_rate(
    const double age,
    const std::span<const double> table_ages,
    const std::span<const double> qx
) {
    if (!std::isfinite(age) || age < 0.0) {
        throw std::invalid_argument("age must be finite and non-negative");
    }
    return interpolate_flat_linear(age, table_ages, qx);
}

}  // namespace qfin
