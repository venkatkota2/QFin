#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qfin {

struct BatchBondMetrics {
    std::vector<double> prices;
    std::vector<double> macaulay_durations;
    std::vector<double> convexities;
    std::vector<double> dv01;
};

struct YieldSolveResult {
    std::vector<double> yields;
    std::vector<std::uint8_t> converged;
    std::vector<std::int32_t> iterations;
};

[[nodiscard]] BatchBondMetrics price_cashflow_batches(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    double parallel_shift = 0.0
);

[[nodiscard]] BatchBondMetrics price_cashflow_batches_from_yield(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> yields,
    std::span<const std::int32_t> frequencies
);

[[nodiscard]] YieldSolveResult solve_yields_from_prices(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> target_prices,
    std::span<const std::int32_t> frequencies,
    double tolerance,
    std::int32_t max_iterations
);

}  // namespace qfin
