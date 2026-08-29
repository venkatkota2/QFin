#pragma once

#include <span>

namespace qfin {

struct RiskSummaryNative {
    double mean{};
    double standard_deviation{};
    double minimum{};
    double maximum{};
    double value_at_risk{};
    double expected_shortfall{};
};

[[nodiscard]] RiskSummaryNative aggregate_tail_risk(
    std::span<const double> losses,
    std::span<const double> probabilities,
    double confidence
);

}  // namespace qfin
