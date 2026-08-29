#include "qfin/alm.hpp"

#include "qfin/curves.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfin {
namespace {

constexpr double time_tolerance = 1.0e-12;

double present_value(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const double valuation_time,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> shocks,
    const double spread,
    const std::span<const double> inflation_linkage,
    const double inflation_index
) {
    double value = 0.0;
    for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
        if (cashflow_times[index] <= valuation_time + time_tolerance) {
            continue;
        }
        const double tenor = cashflow_times[index] - valuation_time;
        const double rate = interpolate_flat_linear(tenor, curve_times, zero_rates) +
                            interpolate_flat_linear(tenor, curve_times, shocks) + spread;
        const double scale = inflation_linkage.empty()
                                 ? 1.0
                                 : std::pow(inflation_index, inflation_linkage[index]);
        value += cashflow_amounts[index] * scale * std::exp(-rate * tenor);
    }
    return value;
}

void validate_cashflows(
    const std::span<const double> times,
    const std::span<const double> amounts,
    const char* name
) {
    if (times.size() != amounts.size()) {
        throw std::invalid_argument(std::string(name) + " cash-flow buffers must align");
    }
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (!std::isfinite(times[index]) || times[index] < 0.0 ||
            !std::isfinite(amounts[index])) {
            throw std::invalid_argument(
                std::string(name) + " cash flows must be finite and non-negative in time"
            );
        }
    }
}

}  // namespace

ALMPathProjectionResult project_alm_paths(
    const std::span<const double> asset_cashflow_times,
    const std::span<const double> asset_cashflow_amounts,
    const std::span<const double> liability_cashflow_times,
    const std::span<const double> liability_cashflow_amounts,
    const std::span<const double> liability_inflation_linkage,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> rate_shocks,
    const std::span<const double> credit_spread_shocks,
    const std::span<const double> equity_returns,
    const std::span<const double> inflation_rates,
    const std::size_t scenario_count,
    const std::size_t period_count,
    const double period_length,
    const double initial_equity_value,
    const double initial_cash_value,
    const double target_equity_weight,
    const std::int32_t rebalance_frequency,
    const double transaction_cost_rate,
    const bool pay_liabilities
) {
    validate_curve(curve_times, zero_rates);
    validate_cashflows(asset_cashflow_times, asset_cashflow_amounts, "asset");
    validate_cashflows(liability_cashflow_times, liability_cashflow_amounts, "liability");
    const std::size_t path_size = scenario_count * period_count;
    if (liability_inflation_linkage.size() != liability_cashflow_times.size() ||
        rate_shocks.size() != path_size * curve_times.size() ||
        credit_spread_shocks.size() != path_size || equity_returns.size() != path_size ||
        inflation_rates.size() != path_size || scenario_count == 0 || period_count == 0 ||
        !std::isfinite(period_length) || period_length <= 0.0 ||
        !std::isfinite(initial_equity_value) || initial_equity_value < 0.0 ||
        !std::isfinite(initial_cash_value) || initial_cash_value < 0.0 ||
        !std::isfinite(target_equity_weight) ||
        (target_equity_weight != -1.0 &&
         (target_equity_weight < 0.0 || target_equity_weight > 1.0)) ||
        rebalance_frequency <= 0 || !std::isfinite(transaction_cost_rate) ||
        transaction_cost_rate < 0.0 || transaction_cost_rate >= 1.0) {
        throw std::invalid_argument("invalid ALM path inputs");
    }
    for (const double linkage : liability_inflation_linkage) {
        if (!std::isfinite(linkage) || linkage < 0.0) {
            throw std::invalid_argument("liability inflation linkage must be non-negative");
        }
    }
    for (std::size_t index = 0; index < path_size; ++index) {
        if (!std::isfinite(credit_spread_shocks[index]) ||
            !std::isfinite(equity_returns[index]) || equity_returns[index] <= -1.0 ||
            !std::isfinite(inflation_rates[index]) || inflation_rates[index] <= -1.0) {
            throw std::invalid_argument("invalid economic scenario factor path");
        }
    }
    for (const double shock : rate_shocks) {
        if (!std::isfinite(shock)) {
            throw std::invalid_argument("rate shocks must be finite");
        }
    }

    const std::size_t output_width = period_count + 1;
    const std::size_t output_size = scenario_count * output_width;
    ALMPathProjectionResult result{
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(path_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(output_size, 0.0),
        std::vector<double>(path_size, 0.0),
    };
    const std::vector<double> zero_shock_values(curve_times.size(), 0.0);
    const std::span<const double> zero_shocks(zero_shock_values);
    const std::span<const double> no_linkage;
    const double initial_bond = present_value(
        asset_cashflow_times,
        asset_cashflow_amounts,
        0.0,
        curve_times,
        zero_rates,
        zero_shocks,
        0.0,
        no_linkage,
        1.0
    );
    const double initial_liability = present_value(
        liability_cashflow_times,
        liability_cashflow_amounts,
        0.0,
        curve_times,
        zero_rates,
        zero_shocks,
        0.0,
        no_linkage,
        1.0
    );
    const double base_short_rate = interpolate_flat_linear(
        period_length, curve_times, zero_rates
    );

    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        double bond_balance = initial_bond;
        double cash_balance = initial_cash_value;
        double equity_balance = initial_equity_value;
        double inflation_index = 1.0;
        const std::size_t initial_output = scenario * output_width;
        result.bond_values[initial_output] = bond_balance;
        result.cash_values[initial_output] = cash_balance;
        result.equity_values[initial_output] = equity_balance;
        result.asset_values[initial_output] = bond_balance + cash_balance + equity_balance;
        result.liability_values[initial_output] = initial_liability;
        result.surplus[initial_output] = result.asset_values[initial_output] - initial_liability;
        result.funding_ratio[initial_output] = initial_liability == 0.0
                                                    ? std::numeric_limits<double>::infinity()
                                                    : result.asset_values[initial_output] /
                                                          initial_liability;

        for (std::size_t period = 0; period < period_count; ++period) {
            const std::size_t path_index = scenario * period_count + period;
            const std::size_t shock_offset = path_index * curve_times.size();
            const auto end_shocks = rate_shocks.subspan(shock_offset, curve_times.size());
            const auto start_shocks = period == 0
                                          ? zero_shocks
                                          : rate_shocks.subspan(
                                                (path_index - 1) * curve_times.size(),
                                                curve_times.size()
                                            );
            const double start_spread = period == 0
                                            ? 0.0
                                            : credit_spread_shocks[path_index - 1];
            const double end_spread = credit_spread_shocks[path_index];
            const double period_start = static_cast<double>(period) * period_length;
            const double period_end = static_cast<double>(period + 1) * period_length;
            const double start_index = present_value(
                asset_cashflow_times,
                asset_cashflow_amounts,
                period_start,
                curve_times,
                zero_rates,
                start_shocks,
                start_spread,
                no_linkage,
                1.0
            );
            const double end_index = present_value(
                asset_cashflow_times,
                asset_cashflow_amounts,
                period_end,
                curve_times,
                zero_rates,
                end_shocks,
                end_spread,
                no_linkage,
                1.0
            );
            const double average_short_shock = 0.5 * (
                interpolate_flat_linear(period_length, curve_times, start_shocks) +
                interpolate_flat_linear(period_length, curve_times, end_shocks)
            );
            const double cash_rate = base_short_rate + average_short_shock;
            const double cash_growth = std::exp(cash_rate * period_length);
            double received_wealth = 0.0;
            for (std::size_t index = 0; index < asset_cashflow_times.size(); ++index) {
                const double time = asset_cashflow_times[index];
                if (time > period_start + time_tolerance &&
                    time <= period_end + time_tolerance) {
                    received_wealth += asset_cashflow_amounts[index] *
                                       std::exp(cash_rate * (period_end - time));
                }
            }
            const double bond_return = std::abs(start_index) > 1.0e-15
                                           ? (end_index + received_wealth) / start_index
                                           : cash_growth;
            bond_balance *= bond_return;
            cash_balance *= cash_growth;
            equity_balance *= 1.0 + equity_returns[path_index];
            inflation_index *= std::pow(
                1.0 + inflation_rates[path_index], period_length
            );

            double liability_payment = 0.0;
            for (std::size_t index = 0; index < liability_cashflow_times.size(); ++index) {
                const double time = liability_cashflow_times[index];
                if (time > period_start + time_tolerance &&
                    time <= period_end + time_tolerance) {
                    liability_payment += liability_cashflow_amounts[index] * std::pow(
                        inflation_index, liability_inflation_linkage[index]
                    );
                }
            }
            result.liability_payments[path_index] = liability_payment;
            if (pay_liabilities) {
                const double before_payment = bond_balance + cash_balance + equity_balance;
                const double after_payment = before_payment - liability_payment;
                const double scale = std::abs(before_payment) > 1.0e-15
                                         ? after_payment / before_payment
                                         : 1.0;
                bond_balance *= scale;
                cash_balance *= scale;
                equity_balance *= scale;
            }

            if (target_equity_weight >= 0.0 &&
                (period + 1) % static_cast<std::size_t>(rebalance_frequency) == 0) {
                const double total = bond_balance + cash_balance + equity_balance;
                const double desired_equity = target_equity_weight * std::max(total, 0.0);
                const double cost = std::abs(desired_equity - equity_balance) *
                                    transaction_cost_rate;
                result.transaction_costs[path_index] = cost;
                const double after_cost = total - cost;
                if (after_cost > 0.0) {
                    const double non_equity_before = bond_balance + cash_balance;
                    const double cash_share = std::abs(non_equity_before) > 1.0e-15
                                                  ? std::clamp(
                                                        cash_balance / non_equity_before,
                                                        0.0,
                                                        1.0
                                                    )
                                                  : 0.0;
                    const double non_equity = (1.0 - target_equity_weight) *
                                              after_cost;
                    equity_balance = target_equity_weight * after_cost;
                    cash_balance = cash_share * non_equity;
                    bond_balance = non_equity - cash_balance;
                }
            }

            const double liability_value = present_value(
                liability_cashflow_times,
                liability_cashflow_amounts,
                period_end,
                curve_times,
                zero_rates,
                end_shocks,
                0.0,
                liability_inflation_linkage,
                inflation_index
            );
            const std::size_t output_index = scenario * output_width + period + 1;
            result.bond_values[output_index] = bond_balance;
            result.cash_values[output_index] = cash_balance;
            result.equity_values[output_index] = equity_balance;
            result.asset_values[output_index] = bond_balance + cash_balance + equity_balance;
            result.liability_values[output_index] = liability_value;
            result.surplus[output_index] = result.asset_values[output_index] - liability_value;
            result.funding_ratio[output_index] = liability_value == 0.0
                                                      ? std::numeric_limits<double>::infinity()
                                                      : result.asset_values[output_index] /
                                                            liability_value;
        }
    }
    return result;
}

}  // namespace qfin
