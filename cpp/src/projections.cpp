#include "qfin/projections.hpp"

#include "qfin/curves.hpp"
#include "qfin/mortality.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qfin {

PolicyProjectionResult project_term_life_policies(
    const std::span<const double> attained_ages,
    const std::span<const double> sums_assured,
    const std::span<const double> annual_premiums,
    const std::span<const std::int32_t> remaining_terms,
    const std::span<const double> table_ages,
    const std::span<const double> mortality_qx,
    const std::span<const double> lapse_rates,
    const double expense_per_policy,
    const double mortality_multiplier,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates
) {
    const std::size_t policy_count = attained_ages.size();
    if (sums_assured.size() != policy_count || annual_premiums.size() != policy_count ||
        remaining_terms.size() != policy_count || lapse_rates.empty() ||
        !std::isfinite(expense_per_policy) || expense_per_policy < 0.0 ||
        !std::isfinite(mortality_multiplier) || mortality_multiplier < 0.0) {
        throw std::invalid_argument("invalid policy projection inputs");
    }
    validate_mortality_table(table_ages, mortality_qx);
    validate_curve(curve_times, zero_rates);
    std::int32_t maximum_term = 0;
    for (std::size_t policy = 0; policy < policy_count; ++policy) {
        if (!std::isfinite(attained_ages[policy]) || attained_ages[policy] < 0.0 ||
            !std::isfinite(sums_assured[policy]) || sums_assured[policy] < 0.0 ||
            !std::isfinite(annual_premiums[policy]) || annual_premiums[policy] < 0.0 ||
            remaining_terms[policy] < 0) {
            throw std::invalid_argument("policy attributes must be finite and non-negative");
        }
        maximum_term = std::max(maximum_term, remaining_terms[policy]);
    }
    if (lapse_rates.size() != 1 && lapse_rates.size() < static_cast<std::size_t>(maximum_term)) {
        throw std::invalid_argument("lapse rates must be scalar or cover every projection year");
    }
    for (const double lapse : lapse_rates) {
        if (!std::isfinite(lapse) || lapse < 0.0 || lapse > 1.0) {
            throw std::invalid_argument("lapse rates must lie in [0, 1]");
        }
    }

    const auto output_size = static_cast<std::size_t>(maximum_term) + 1;
    PolicyProjectionResult result{
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(policy_count, 0.0),
        0.0,
        0.0,
    };

    for (std::size_t policy = 0; policy < policy_count; ++policy) {
        double in_force = 1.0;
        double policy_pv = 0.0;
        const auto term = remaining_terms[policy];
        for (std::int32_t year = 0; year < term; ++year) {
            const auto year_index = static_cast<std::size_t>(year);
            result.in_force[year_index] += in_force;
            const double premium = in_force * annual_premiums[policy];
            const double expense = in_force * expense_per_policy;
            const double base_qx = mortality_rate(
                attained_ages[policy] + static_cast<double>(year), table_ages, mortality_qx
            );
            const double qx = std::clamp(base_qx * mortality_multiplier, 0.0, 1.0);
            const double deaths = in_force * qx;
            const double benefit = deaths * sums_assured[policy];
            const double lapse = lapse_rates.size() == 1 ? lapse_rates.front() : lapse_rates[year_index];
            const double survivors = in_force - deaths;
            in_force = survivors * (1.0 - lapse);

            result.expected_premiums[year_index] += premium;
            result.expected_expenses[year_index] += expense;
            result.expected_benefits[year_index + 1] += benefit;
            const double start_outflow = expense - premium;
            result.net_liability_cashflows[year_index] += start_outflow;
            result.net_liability_cashflows[year_index + 1] += benefit;
            policy_pv += start_outflow * discount_factor(
                static_cast<double>(year), curve_times, zero_rates
            );
            policy_pv += benefit * discount_factor(
                static_cast<double>(year + 1), curve_times, zero_rates
            );
        }
        result.policy_present_values[policy] = policy_pv;
    }

    double duration_numerator = 0.0;
    for (std::size_t year = 0; year < output_size; ++year) {
        const double present_value = result.net_liability_cashflows[year] * discount_factor(
            static_cast<double>(year), curve_times, zero_rates
        );
        result.present_value += present_value;
        duration_numerator += static_cast<double>(year) * present_value;
    }
    if (std::abs(result.present_value) > 1.0e-15) {
        result.duration = duration_numerator / result.present_value;
    }
    return result;
}

}  // namespace qfin
