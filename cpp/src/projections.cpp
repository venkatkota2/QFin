#include "qfin/projections.hpp"

#include "qfin/curves.hpp"
#include "qfin/mortality.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace qfin {
namespace {

double path_value(const std::span<const double> values, const std::size_t year) {
    return values.size() == 1 ? values.front() : values[year];
}

void validate_projection_path(
    const std::span<const double> values,
    const std::size_t maximum_term,
    const bool probability,
    const bool greater_than_minus_one,
    const char* name
) {
    if (values.empty() || (values.size() != 1 && values.size() < maximum_term)) {
        throw std::invalid_argument(
            std::string(name) + " must be scalar or cover every projection year"
        );
    }
    for (const double value : values) {
        if (!std::isfinite(value) || (probability && (value < 0.0 || value > 1.0)) ||
            (greater_than_minus_one && value <= -1.0)) {
            throw std::invalid_argument(std::string("invalid ") + name);
        }
    }
}

}  // namespace

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

LifeModelPointProjectionResult project_life_model_points(
    const std::span<const double> attained_ages,
    const std::span<const double> sums_assured,
    const std::span<const double> annual_premiums,
    const std::span<const std::int32_t> remaining_terms,
    const std::span<const std::int32_t> policy_durations,
    const std::span<const std::int32_t> original_terms,
    const std::span<const std::int32_t> product_codes,
    const std::span<const double> model_point_counts,
    const std::span<const double> annual_benefits,
    const std::span<const double> account_values,
    const std::span<const double> annual_charges,
    const std::span<const double> crediting_spreads,
    const std::span<const double> bonus_rates,
    const std::span<const double> disability_benefits,
    const std::span<const double> benefit_inflation_linkage,
    const std::span<const double> table_ages,
    const std::span<const double> mortality_qx,
    const std::span<const double> lapse_rates,
    const std::span<const double> disability_rates,
    const std::span<const double> recovery_rates,
    const std::span<const double> crediting_rates,
    const std::span<const double> expense_inflation_rates,
    const double expense_per_policy,
    const double mortality_multiplier,
    const double disabled_mortality_multiplier,
    const double premium_multiplier,
    const double benefit_multiplier,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates
) {
    const std::size_t point_count = attained_ages.size();
    if (sums_assured.size() != point_count || annual_premiums.size() != point_count ||
        remaining_terms.size() != point_count || policy_durations.size() != point_count ||
        original_terms.size() != point_count || product_codes.size() != point_count ||
        model_point_counts.size() != point_count || annual_benefits.size() != point_count ||
        account_values.size() != point_count || annual_charges.size() != point_count ||
        crediting_spreads.size() != point_count || bonus_rates.size() != point_count ||
        disability_benefits.size() != point_count ||
        benefit_inflation_linkage.size() != point_count ||
        !std::isfinite(expense_per_policy) || expense_per_policy < 0.0 ||
        !std::isfinite(mortality_multiplier) || mortality_multiplier < 0.0 ||
        !std::isfinite(disabled_mortality_multiplier) ||
        disabled_mortality_multiplier < 0.0 || !std::isfinite(premium_multiplier) ||
        premium_multiplier < 0.0 || !std::isfinite(benefit_multiplier) ||
        benefit_multiplier < 0.0) {
        throw std::invalid_argument("invalid life model-point projection inputs");
    }
    validate_mortality_table(table_ages, mortality_qx);
    validate_curve(curve_times, zero_rates);
    std::int32_t maximum_term = 0;
    for (std::size_t point = 0; point < point_count; ++point) {
        const bool invalid = !std::isfinite(attained_ages[point]) ||
                             attained_ages[point] < 0.0 ||
                             !std::isfinite(sums_assured[point]) ||
                             sums_assured[point] < 0.0 ||
                             !std::isfinite(annual_premiums[point]) ||
                             annual_premiums[point] < 0.0 || remaining_terms[point] < 0 ||
                             policy_durations[point] < 0 || original_terms[point] <= 0 ||
                             product_codes[point] < 0 || product_codes[point] > 3 ||
                             !std::isfinite(model_point_counts[point]) ||
                             model_point_counts[point] <= 0.0 ||
                             !std::isfinite(annual_benefits[point]) ||
                             annual_benefits[point] < 0.0 ||
                             !std::isfinite(account_values[point]) ||
                             account_values[point] < 0.0 ||
                             !std::isfinite(annual_charges[point]) ||
                             annual_charges[point] < 0.0 ||
                             !std::isfinite(crediting_spreads[point]) ||
                             !std::isfinite(bonus_rates[point]) || bonus_rates[point] < 0.0 ||
                             !std::isfinite(disability_benefits[point]) ||
                             disability_benefits[point] < 0.0 ||
                             !std::isfinite(benefit_inflation_linkage[point]) ||
                             benefit_inflation_linkage[point] < 0.0;
        if (invalid) {
            throw std::invalid_argument("invalid life model-point attributes");
        }
        maximum_term = std::max(maximum_term, remaining_terms[point]);
    }
    const auto horizon = static_cast<std::size_t>(maximum_term);
    validate_projection_path(lapse_rates, horizon, true, false, "lapse rates");
    validate_projection_path(disability_rates, horizon, true, false, "disability rates");
    validate_projection_path(recovery_rates, horizon, true, false, "recovery rates");
    validate_projection_path(crediting_rates, horizon, false, true, "crediting rates");
    validate_projection_path(
        expense_inflation_rates, horizon, false, true, "expense inflation rates"
    );

    const std::size_t output_size = horizon + 1;
    LifeModelPointProjectionResult result{
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(point_count, 0.0),
        0.0,
        0.0,
    };

    for (std::size_t point = 0; point < point_count; ++point) {
        double active = 1.0;
        double disabled = 0.0;
        double account = account_values[point];
        double inflation_index = 1.0;
        double point_pv = 0.0;
        const double count = model_point_counts[point];
        for (std::int32_t year = 0; year < remaining_terms[point]; ++year) {
            const auto year_index = static_cast<std::size_t>(year);
            result.active[year_index] += count * active;
            result.disabled[year_index] += count * disabled;
            const double premium = active * annual_premiums[point] * premium_multiplier;
            const double expense = (active + disabled) * expense_per_policy *
                                   inflation_index;
            if (product_codes[point] == 2) {
                account = std::max(
                    0.0,
                    (account + annual_premiums[point] * premium_multiplier -
                     annual_charges[point]) *
                        (1.0 + path_value(crediting_rates, year_index) +
                         crediting_spreads[point])
                );
            }
            const double base_qx = mortality_rate(
                attained_ages[point] + static_cast<double>(year), table_ages, mortality_qx
            );
            const double active_qx = std::clamp(
                base_qx * mortality_multiplier, 0.0, 1.0
            );
            const double disabled_qx = std::clamp(
                active_qx * disabled_mortality_multiplier, 0.0, 1.0
            );
            const double active_deaths = active * active_qx;
            const double disabled_deaths = disabled * disabled_qx;
            const double active_survivors = active - active_deaths;
            const double disabled_survivors = disabled - disabled_deaths;
            const double new_disabled = active_survivors *
                                        path_value(disability_rates, year_index);
            const double recoveries = disabled_survivors *
                                      path_value(recovery_rates, year_index);
            const double active_before_lapse = active_survivors - new_disabled + recoveries;
            const double disabled_before_lapse = disabled_survivors - recoveries + new_disabled;
            const double lapse = path_value(lapse_rates, year_index);
            const double lapse_count = (active_before_lapse + disabled_before_lapse) * lapse;
            active = active_before_lapse * (1.0 - lapse);
            disabled = disabled_before_lapse * (1.0 - lapse);
            const double next_inflation = inflation_index *
                                          (1.0 + path_value(
                                                     expense_inflation_rates, year_index
                                                 ));
            const double benefit_scale = benefit_multiplier * std::pow(
                next_inflation, benefit_inflation_linkage[point]
            );
            const double deaths = active_deaths + disabled_deaths;
            double death_benefit = 0.0;
            double annuity_benefit = 0.0;
            if (product_codes[point] == 3) {
                annuity_benefit = (active_survivors + disabled_survivors) *
                                  annual_benefits[point] * benefit_scale;
            } else {
                double insured_amount = sums_assured[point];
                if (product_codes[point] == 1) {
                    insured_amount *= std::pow(
                        1.0 + bonus_rates[point],
                        static_cast<double>(policy_durations[point] + year + 1)
                    );
                } else if (product_codes[point] == 2) {
                    insured_amount = std::max(insured_amount, account);
                }
                death_benefit = deaths * insured_amount * benefit_scale;
            }
            const double disability_benefit = disabled_before_lapse *
                                              disability_benefits[point] *
                                              benefit_scale;
            const double surrender = product_codes[point] == 2
                                         ? lapse_count * account
                                         : 0.0;
            double maturity = 0.0;
            if (year + 1 == remaining_terms[point]) {
                const double ending_in_force = active + disabled;
                if (product_codes[point] == 1) {
                    maturity = ending_in_force * sums_assured[point] *
                               std::pow(
                                   1.0 + bonus_rates[point],
                                   static_cast<double>(original_terms[point])
                               ) *
                               benefit_scale;
                } else if (product_codes[point] == 2) {
                    maturity = ending_in_force * account * benefit_scale;
                }
            }
            const double end_benefit = death_benefit + annuity_benefit +
                                       disability_benefit + maturity;
            const double start_net = expense - premium;
            const double end_net = end_benefit + surrender;
            result.expected_premiums[year_index] += count * premium;
            result.expected_expenses[year_index] += count * expense;
            result.expected_benefits[year_index + 1] += count * end_benefit;
            result.expected_surrenders[year_index + 1] += count * surrender;
            result.deaths[year_index + 1] += count * deaths;
            result.net_liability_cashflows[year_index] += count * start_net;
            result.net_liability_cashflows[year_index + 1] += count * end_net;
            point_pv += count * start_net * discount_factor(
                static_cast<double>(year), curve_times, zero_rates
            );
            point_pv += count * end_net * discount_factor(
                static_cast<double>(year + 1), curve_times, zero_rates
            );
            inflation_index = next_inflation;
        }
        result.model_point_present_values[point] = point_pv;
    }

    double duration_numerator = 0.0;
    for (std::size_t year = 0; year < output_size; ++year) {
        const double value = result.net_liability_cashflows[year] * discount_factor(
            static_cast<double>(year), curve_times, zero_rates
        );
        result.present_value += value;
        duration_numerator += static_cast<double>(year) * value;
    }
    if (std::abs(result.present_value) > 1.0e-15) {
        result.duration = duration_numerator / result.present_value;
    }
    return result;
}

LifeScenarioProjectionResult project_life_scenarios(
    const std::span<const double> attained_ages,
    const std::span<const double> sums_assured,
    const std::span<const double> annual_premiums,
    const std::span<const std::int32_t> remaining_terms,
    const std::span<const std::int32_t> policy_durations,
    const std::span<const std::int32_t> original_terms,
    const std::span<const std::int32_t> product_codes,
    const std::span<const double> model_point_counts,
    const std::span<const double> annual_benefits,
    const std::span<const double> account_values,
    const std::span<const double> annual_charges,
    const std::span<const double> crediting_spreads,
    const std::span<const double> bonus_rates,
    const std::span<const double> disability_benefits,
    const std::span<const double> benefit_inflation_linkage,
    const std::span<const double> table_ages,
    const std::span<const double> mortality_qx,
    const std::span<const double> lapse_rates,
    const std::span<const double> disability_rates,
    const std::span<const double> recovery_rates,
    const std::span<const double> crediting_rates,
    const std::span<const double> base_expense_inflation_rates,
    const double expense_per_policy,
    const double mortality_multiplier,
    const double disabled_mortality_multiplier,
    const double premium_multiplier,
    const double benefit_multiplier,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_rate_shocks,
    const std::span<const double> scenario_mortality_multipliers,
    const std::span<const double> scenario_lapse_multipliers,
    const std::span<const double> scenario_inflation_rates,
    const std::size_t scenario_count,
    const std::size_t period_count
) {
    const std::size_t point_count = attained_ages.size();
    const bool misaligned = sums_assured.size() != point_count ||
                            annual_premiums.size() != point_count ||
                            remaining_terms.size() != point_count ||
                            policy_durations.size() != point_count ||
                            original_terms.size() != point_count ||
                            product_codes.size() != point_count ||
                            model_point_counts.size() != point_count ||
                            annual_benefits.size() != point_count ||
                            account_values.size() != point_count ||
                            annual_charges.size() != point_count ||
                            crediting_spreads.size() != point_count ||
                            bonus_rates.size() != point_count ||
                            disability_benefits.size() != point_count ||
                            benefit_inflation_linkage.size() != point_count;
    const std::size_t scenario_periods = scenario_count * period_count;
    if (misaligned || scenario_count == 0 || period_count == 0 ||
        scenario_rate_shocks.size() != scenario_periods * curve_times.size() ||
        scenario_mortality_multipliers.size() != scenario_periods ||
        scenario_lapse_multipliers.size() != scenario_periods ||
        scenario_inflation_rates.size() != scenario_periods ||
        !std::isfinite(expense_per_policy) || expense_per_policy < 0.0 ||
        !std::isfinite(mortality_multiplier) || mortality_multiplier < 0.0 ||
        !std::isfinite(disabled_mortality_multiplier) ||
        disabled_mortality_multiplier < 0.0 || !std::isfinite(premium_multiplier) ||
        premium_multiplier < 0.0 || !std::isfinite(benefit_multiplier) ||
        benefit_multiplier < 0.0) {
        throw std::invalid_argument("invalid life scenario projection inputs");
    }
    validate_mortality_table(table_ages, mortality_qx);
    validate_curve(curve_times, zero_rates);
    std::size_t maximum_term = 0;
    for (std::size_t point = 0; point < point_count; ++point) {
        if (!std::isfinite(attained_ages[point]) || attained_ages[point] < 0.0 ||
            !std::isfinite(sums_assured[point]) || sums_assured[point] < 0.0 ||
            !std::isfinite(annual_premiums[point]) || annual_premiums[point] < 0.0 ||
            remaining_terms[point] < 0 || policy_durations[point] < 0 ||
            original_terms[point] <= 0 || product_codes[point] < 0 ||
            product_codes[point] > 3 || !std::isfinite(model_point_counts[point]) ||
            model_point_counts[point] <= 0.0 || !std::isfinite(annual_benefits[point]) ||
            annual_benefits[point] < 0.0 || !std::isfinite(account_values[point]) ||
            account_values[point] < 0.0 || !std::isfinite(annual_charges[point]) ||
            annual_charges[point] < 0.0 || !std::isfinite(crediting_spreads[point]) ||
            !std::isfinite(bonus_rates[point]) || bonus_rates[point] < 0.0 ||
            !std::isfinite(disability_benefits[point]) ||
            disability_benefits[point] < 0.0 ||
            !std::isfinite(benefit_inflation_linkage[point]) ||
            benefit_inflation_linkage[point] < 0.0) {
            throw std::invalid_argument("invalid life scenario model point");
        }
        maximum_term = std::max(
            maximum_term, static_cast<std::size_t>(remaining_terms[point])
        );
    }
    if (maximum_term > period_count) {
        throw std::invalid_argument("life scenarios must cover every projection year");
    }
    validate_projection_path(lapse_rates, maximum_term, true, false, "lapse rates");
    validate_projection_path(
        disability_rates, maximum_term, true, false, "disability rates"
    );
    validate_projection_path(recovery_rates, maximum_term, true, false, "recovery rates");
    validate_projection_path(crediting_rates, maximum_term, false, true, "crediting rates");
    validate_projection_path(
        base_expense_inflation_rates,
        maximum_term,
        false,
        true,
        "expense inflation rates"
    );
    for (std::size_t index = 0; index < scenario_periods; ++index) {
        if (!std::isfinite(scenario_mortality_multipliers[index]) ||
            scenario_mortality_multipliers[index] < 0.0 ||
            !std::isfinite(scenario_lapse_multipliers[index]) ||
            scenario_lapse_multipliers[index] < 0.0 ||
            !std::isfinite(scenario_inflation_rates[index]) ||
            scenario_inflation_rates[index] <= -1.0) {
            throw std::invalid_argument("invalid life scenario factor");
        }
    }
    for (const double shock : scenario_rate_shocks) {
        if (!std::isfinite(shock)) {
            throw std::invalid_argument("life scenario rate shocks must be finite");
        }
    }

    LifeScenarioProjectionResult result{
        std::vector<double>(scenario_count, 0.0),
        std::vector<double>(scenario_count, 0.0),
        std::vector<double>(scenario_count, 0.0),
        std::vector<double>(scenario_count, 0.0),
        std::vector<double>(scenario_count, 0.0),
    };
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        double scenario_pv = 0.0;
        for (std::size_t point = 0; point < point_count; ++point) {
            double active = 1.0;
            double disabled = 0.0;
            double account = account_values[point];
            double inflation_index = 1.0;
            double discount = 1.0;
            const double count = model_point_counts[point];
            for (std::int32_t year = 0; year < remaining_terms[point]; ++year) {
                const auto year_index = static_cast<std::size_t>(year);
                const std::size_t path_index = scenario * period_count + year_index;
                const auto rate_path = scenario_rate_shocks.subspan(
                    path_index * curve_times.size(), curve_times.size()
                );
                const double premium = active * annual_premiums[point] *
                                       premium_multiplier;
                const double expense = (active + disabled) * expense_per_policy *
                                       inflation_index;
                if (product_codes[point] == 2) {
                    account = std::max(
                        0.0,
                        (account + annual_premiums[point] * premium_multiplier -
                         annual_charges[point]) *
                            (1.0 + path_value(crediting_rates, year_index) +
                             crediting_spreads[point])
                    );
                }
                const double base_qx = mortality_rate(
                    attained_ages[point] + static_cast<double>(year),
                    table_ages,
                    mortality_qx
                );
                const double active_qx = std::clamp(
                    base_qx * mortality_multiplier *
                        scenario_mortality_multipliers[path_index],
                    0.0,
                    1.0
                );
                const double disabled_qx = std::clamp(
                    active_qx * disabled_mortality_multiplier, 0.0, 1.0
                );
                const double active_deaths = active * active_qx;
                const double disabled_deaths = disabled * disabled_qx;
                const double active_survivors = active - active_deaths;
                const double disabled_survivors = disabled - disabled_deaths;
                const double new_disabled = active_survivors *
                                            path_value(disability_rates, year_index);
                const double recoveries = disabled_survivors *
                                          path_value(recovery_rates, year_index);
                const double active_before_lapse = active_survivors - new_disabled +
                                                   recoveries;
                const double disabled_before_lapse = disabled_survivors - recoveries +
                                                     new_disabled;
                const double lapse = std::clamp(
                    path_value(lapse_rates, year_index) *
                        scenario_lapse_multipliers[path_index],
                    0.0,
                    1.0
                );
                const double lapse_count = (active_before_lapse + disabled_before_lapse) *
                                           lapse;
                active = active_before_lapse * (1.0 - lapse);
                disabled = disabled_before_lapse * (1.0 - lapse);
                inflation_index *= (
                    1.0 + path_value(base_expense_inflation_rates, year_index)
                ) * (1.0 + scenario_inflation_rates[path_index]);
                const double benefit_scale = benefit_multiplier * std::pow(
                    inflation_index, benefit_inflation_linkage[point]
                );
                const double deaths = active_deaths + disabled_deaths;
                double death_benefit = 0.0;
                double annuity_benefit = 0.0;
                if (product_codes[point] == 3) {
                    annuity_benefit = (active_survivors + disabled_survivors) *
                                      annual_benefits[point] * benefit_scale;
                } else {
                    double insured_amount = sums_assured[point];
                    if (product_codes[point] == 1) {
                        insured_amount *= std::pow(
                            1.0 + bonus_rates[point],
                            static_cast<double>(policy_durations[point] + year + 1)
                        );
                    } else if (product_codes[point] == 2) {
                        insured_amount = std::max(insured_amount, account);
                    }
                    death_benefit = deaths * insured_amount * benefit_scale;
                }
                const double disability_benefit = disabled_before_lapse *
                                                  disability_benefits[point] *
                                                  benefit_scale;
                const double surrender = product_codes[point] == 2
                                             ? lapse_count * account
                                             : 0.0;
                double maturity = 0.0;
                if (year + 1 == remaining_terms[point]) {
                    const double ending_in_force = active + disabled;
                    if (product_codes[point] == 1) {
                        maturity = ending_in_force * sums_assured[point] *
                                   std::pow(
                                       1.0 + bonus_rates[point],
                                       static_cast<double>(original_terms[point])
                                   ) *
                                   benefit_scale;
                    } else if (product_codes[point] == 2) {
                        maturity = ending_in_force * account * benefit_scale;
                    }
                }
                const double end_benefit = death_benefit + annuity_benefit +
                                           disability_benefit + maturity;
                const double start_time = static_cast<double>(year);
                const double end_time = static_cast<double>(year + 1);
                const double base_forward =
                    interpolate_flat_linear(end_time, curve_times, zero_rates) * end_time -
                    interpolate_flat_linear(start_time, curve_times, zero_rates) *
                        start_time;
                const double short_shock = interpolate_flat_linear(
                    1.0, curve_times, rate_path
                );
                const double end_discount = discount * std::exp(
                    -(base_forward + short_shock)
                );
                const double start_net = expense - premium;
                const double end_net = end_benefit + surrender;
                scenario_pv += count * (
                    start_net * discount + end_net * end_discount
                );
                result.expected_premiums[scenario] += count * premium;
                result.expected_benefits[scenario] += count * end_benefit;
                result.expected_expenses[scenario] += count * expense;
                result.expected_surrenders[scenario] += count * surrender;
                discount = end_discount;
            }
        }
        result.present_values[scenario] = scenario_pv;
    }
    return result;
}

}  // namespace qfin
