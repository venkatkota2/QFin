#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qfin {

struct ALMPathProjectionResult {
    std::vector<double> asset_values;
    std::vector<double> bond_values;
    std::vector<double> cash_values;
    std::vector<double> equity_values;
    std::vector<double> liability_values;
    std::vector<double> liability_payments;
    std::vector<double> surplus;
    std::vector<double> funding_ratio;
    std::vector<double> transaction_costs;
};

[[nodiscard]] ALMPathProjectionResult project_alm_paths(
    std::span<const double> asset_cashflow_times,
    std::span<const double> asset_cashflow_amounts,
    std::span<const double> liability_cashflow_times,
    std::span<const double> liability_cashflow_amounts,
    std::span<const double> liability_inflation_linkage,
    std::span<const double> curve_times,
    std::span<const double> zero_rates,
    std::span<const double> rate_shocks,
    std::span<const double> credit_spread_shocks,
    std::span<const double> equity_returns,
    std::span<const double> inflation_rates,
    std::size_t scenario_count,
    std::size_t period_count,
    double period_length,
    double initial_equity_value,
    double initial_cash_value,
    double target_equity_weight,
    std::int32_t rebalance_frequency,
    double transaction_cost_rate,
    bool pay_liabilities
);

}  // namespace qfin
