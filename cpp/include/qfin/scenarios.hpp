#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qfin {

void scenario_portfolio_present_values_into(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> position_weights,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_shocks,
    std::size_t scenario_count,
    std::span<double> output
);

[[nodiscard]] std::vector<double> scenario_portfolio_present_values(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> position_weights,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_shocks,
    std::size_t scenario_count
);

void scenario_instrument_present_values_into(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const std::int64_t> offsets,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_shocks,
    std::size_t scenario_count,
    std::span<double> output
);

void scenario_indexed_cashflow_present_values_into(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const double> inflation_linkage,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_rate_shocks,
    std::span<const double> scenario_inflation_rates,
    std::size_t scenario_count,
    std::span<double> output
);

[[nodiscard]] std::vector<double> scenario_indexed_cashflow_present_values(
    std::span<const double> cashflow_times,
    std::span<const double> cashflow_amounts,
    std::span<const double> inflation_linkage,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> scenario_rate_shocks,
    std::span<const double> scenario_inflation_rates,
    std::size_t scenario_count
);

}  // namespace qfin
