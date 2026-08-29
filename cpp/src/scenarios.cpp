#include "qfin/scenarios.hpp"

#include "qfin/curves.hpp"

#include <cmath>
#include <stdexcept>

namespace qfin {

std::vector<double> scenario_portfolio_present_values(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> position_weights,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_shocks,
    const std::size_t scenario_count
) {
    validate_curve(curve_times, zero_rates);
    if (cashflow_times.size() != cashflow_amounts.size() || offsets.empty() ||
        offsets.front() != 0 || offsets.back() != static_cast<std::int64_t>(cashflow_times.size()) ||
        position_weights.size() + 1 != offsets.size() ||
        scenario_shocks.size() != scenario_count * curve_times.size()) {
        throw std::invalid_argument("invalid scenario portfolio buffers");
    }
    for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
        if (!std::isfinite(cashflow_times[index]) || cashflow_times[index] < 0.0 ||
            !std::isfinite(cashflow_amounts[index])) {
            throw std::invalid_argument("scenario cash flows must be finite and non-negative in time");
        }
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument("scenario cash-flow offsets must be non-decreasing");
        }
    }
    for (const double weight : position_weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument("scenario position weights must be finite");
        }
    }
    for (const double shock : scenario_shocks) {
        if (!std::isfinite(shock)) {
            throw std::invalid_argument("scenario shocks must be finite");
        }
    }
    std::vector<double> result(scenario_count, 0.0);
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const auto shocks = scenario_shocks.subspan(
            scenario * curve_times.size(), curve_times.size()
        );
        double portfolio_value = 0.0;
        for (std::size_t instrument = 0; instrument < position_weights.size(); ++instrument) {
            double instrument_value = 0.0;
            const auto begin = static_cast<std::size_t>(offsets[instrument]);
            const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
            for (std::size_t index = begin; index < end; ++index) {
                const double time = cashflow_times[index];
                const double rate = interpolate_flat_linear(time, curve_times, zero_rates) +
                                    interpolate_flat_linear(time, curve_times, shocks);
                instrument_value += cashflow_amounts[index] * std::exp(-rate * time);
            }
            portfolio_value += position_weights[instrument] * instrument_value;
        }
        result[scenario] = portfolio_value;
    }
    return result;
}

std::vector<double> scenario_indexed_cashflow_present_values(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const double> inflation_linkage,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_rate_shocks,
    const std::span<const double> scenario_inflation_rates,
    const std::size_t scenario_count
) {
    validate_curve(curve_times, zero_rates);
    if (cashflow_times.size() != cashflow_amounts.size() ||
        cashflow_times.size() != inflation_linkage.size() ||
        scenario_rate_shocks.size() != scenario_count * curve_times.size() ||
        scenario_inflation_rates.size() != scenario_count) {
        throw std::invalid_argument("invalid indexed scenario cash-flow buffers");
    }
    for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
        if (!std::isfinite(cashflow_times[index]) || cashflow_times[index] < 0.0 ||
            !std::isfinite(cashflow_amounts[index]) ||
            !std::isfinite(inflation_linkage[index]) || inflation_linkage[index] < 0.0) {
            throw std::invalid_argument("invalid indexed scenario cash flow");
        }
    }
    std::vector<double> result(scenario_count, 0.0);
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const double inflation = scenario_inflation_rates[scenario];
        if (!std::isfinite(inflation) || inflation <= -1.0) {
            throw std::invalid_argument("scenario inflation rates must be greater than -1");
        }
        const auto shocks = scenario_rate_shocks.subspan(
            scenario * curve_times.size(), curve_times.size()
        );
        double value = 0.0;
        for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
            const double time = cashflow_times[index];
            const double rate = interpolate_flat_linear(time, curve_times, zero_rates) +
                                interpolate_flat_linear(time, curve_times, shocks);
            const double scale = std::pow(
                1.0 + inflation, time * inflation_linkage[index]
            );
            value += cashflow_amounts[index] * scale * std::exp(-rate * time);
        }
        result[scenario] = value;
    }
    return result;
}

}  // namespace qfin
