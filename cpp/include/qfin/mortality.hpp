#pragma once

#include <span>

namespace qfin {

void validate_mortality_table(std::span<const double> ages, std::span<const double> qx);

[[nodiscard]] double mortality_rate(
    double age,
    std::span<const double> table_ages,
    std::span<const double> qx
);

}  // namespace qfin
