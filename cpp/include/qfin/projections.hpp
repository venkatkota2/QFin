#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace qfin {

struct PolicyProjectionResult {
    std::vector<double> expected_premiums;
    std::vector<double> expected_benefits;
    std::vector<double> expected_expenses;
    std::vector<double> net_liability_cashflows;
    std::vector<double> in_force;
    std::vector<double> policy_present_values;
    double present_value{};
    double duration{};
};

[[nodiscard]] PolicyProjectionResult project_term_life_policies(
    std::span<const double> attained_ages,
    std::span<const double> sums_assured,
    std::span<const double> annual_premiums,
    std::span<const std::int32_t> remaining_terms,
    std::span<const double> table_ages,
    std::span<const double> mortality_qx,
    std::span<const double> lapse_rates,
    double expense_per_policy,
    double mortality_multiplier,
    std::span<const double> curve_times,
    std::span<const double> zero_rates
);

}  // namespace qfin
