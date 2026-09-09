#include "qfin/scenarios.hpp"

#include "qfin/curves.hpp"
#include "qfin/numerics.hpp"

#include <cmath>
#include <stdexcept>

namespace qfin {

namespace {

void validate_scenario_cashflows(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_shocks,
    const std::size_t scenario_count
) {
    validate_curve(curve_times, zero_rates);
    if (cashflow_times.size() != cashflow_amounts.size() || offsets.empty() ||
        offsets.front() != 0 ||
        offsets.back() != static_cast<std::int64_t>(cashflow_times.size()) ||
        scenario_shocks.size() != scenario_count * curve_times.size()) {
        throw std::invalid_argument("invalid scenario cash-flow buffers");
    }
    for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
        if (!std::isfinite(cashflow_times[index]) || cashflow_times[index] < 0.0 ||
            !std::isfinite(cashflow_amounts[index])) {
            throw std::invalid_argument(
                "scenario cash flows must be finite and non-negative in time"
            );
        }
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument(
                "scenario cash-flow offsets must be non-decreasing"
            );
        }
    }
    for (const double shock : scenario_shocks) {
        if (!std::isfinite(shock)) {
            throw std::invalid_argument("scenario shocks must be finite");
        }
    }
}

}  // namespace

void scenario_portfolio_present_values_into(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> position_weights,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_shocks,
    const std::size_t scenario_count,
    const std::span<double> output
) {
    validate_scenario_cashflows(
        cashflow_times,
        cashflow_amounts,
        offsets,
        curve_times,
        zero_rates,
        scenario_shocks,
        scenario_count
    );
    if (position_weights.size() + 1 != offsets.size() || output.size() != scenario_count) {
        throw std::invalid_argument("invalid scenario portfolio buffers");
    }
    for (const double weight : position_weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument("scenario position weights must be finite");
        }
    }
    bool has_positive_position = false;
    bool has_negative_position = false;
    for (std::size_t instrument = 0; instrument < position_weights.size(); ++instrument) {
        const auto begin = static_cast<std::size_t>(offsets[instrument]);
        const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
        for (std::size_t index = begin; index < end; ++index) {
            const double weighted_amount = position_weights[instrument] * cashflow_amounts[index];
            has_positive_position = has_positive_position || weighted_amount > 0.0;
            has_negative_position = has_negative_position || weighted_amount < 0.0;
        }
    }
    const bool cancellation_possible = has_positive_position && has_negative_position;
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const auto shocks = scenario_shocks.subspan(
            scenario * curve_times.size(), curve_times.size()
        );
        double portfolio_value = 0.0;
        double portfolio_magnitude = 0.0;
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
            const double position_value = position_weights[instrument] * instrument_value;
            portfolio_value += position_value;
            if (cancellation_possible) {
                portfolio_magnitude += std::abs(position_value);
            }
        }
        if (cancellation_possible && portfolio_magnitude > 0.0 &&
            std::abs(portfolio_value) <= 1.0e-10 * portfolio_magnitude) {
            NeumaierSum accurate_value;
            for (std::size_t instrument = 0; instrument < position_weights.size(); ++instrument) {
                double instrument_value = 0.0;
                const auto begin = static_cast<std::size_t>(offsets[instrument]);
                const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
                for (std::size_t index = begin; index < end; ++index) {
                    const double time = cashflow_times[index];
                    const double rate =
                        interpolate_flat_linear(time, curve_times, zero_rates) +
                        interpolate_flat_linear(time, curve_times, shocks);
                    instrument_value += cashflow_amounts[index] * std::exp(-rate * time);
                }
                accurate_value.add(position_weights[instrument] * instrument_value);
            }
            portfolio_value = accurate_value.value();
        }
        output[scenario] = portfolio_value;
    }
}

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
    std::vector<double> result(scenario_count, 0.0);
    scenario_portfolio_present_values_into(
        cashflow_times,
        cashflow_amounts,
        offsets,
        position_weights,
        curve_times,
        zero_rates,
        scenario_shocks,
        scenario_count,
        result
    );
    return result;
}

void scenario_instrument_present_values_into(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_shocks,
    const std::size_t scenario_count,
    const std::span<double> output,
    const bool changes_from_base
) {
    validate_scenario_cashflows(
        cashflow_times,
        cashflow_amounts,
        offsets,
        curve_times,
        zero_rates,
        scenario_shocks,
        scenario_count
    );
    const std::size_t instrument_count = offsets.size() - 1;
    if (output.size() != scenario_count * instrument_count) {
        throw std::invalid_argument("scenario instrument output dimensions do not align");
    }
    std::vector<double> base_present_values;
    if (changes_from_base) {
        // Row zero reports base PV; later rows report changes. This private mode
        // lets key-rate central differences avoid subtracting two large PVs.
        if (scenario_count > 0) {
            for (const double shock : scenario_shocks.first(curve_times.size())) {
                if (shock != 0.0) {
                    throw std::invalid_argument("base scenario must contain zero shocks");
                }
            }
        }
        base_present_values.resize(cashflow_times.size());
        for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
            const double time = cashflow_times[index];
            const double rate = interpolate_flat_linear(time, curve_times, zero_rates);
            base_present_values[index] = cashflow_amounts[index] * std::exp(-rate * time);
        }
    }
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const auto shocks = scenario_shocks.subspan(
            scenario * curve_times.size(), curve_times.size()
        );
        for (std::size_t instrument = 0; instrument < instrument_count; ++instrument) {
            double instrument_value = 0.0;
            const auto begin = static_cast<std::size_t>(offsets[instrument]);
            const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
            for (std::size_t index = begin; index < end; ++index) {
                const double time = cashflow_times[index];
                if (changes_from_base) {
                    const double shock = interpolate_flat_linear(time, curve_times, shocks);
                    instrument_value += scenario == 0 ? base_present_values[index] :
                        base_present_values[index] * std::expm1(-shock * time);
                } else {
                    const double rate = interpolate_flat_linear(time, curve_times, zero_rates) +
                                        interpolate_flat_linear(time, curve_times, shocks);
                    instrument_value += cashflow_amounts[index] * std::exp(-rate * time);
                }
            }
            output[scenario * instrument_count + instrument] = instrument_value;
        }
    }
}

void scenario_indexed_cashflow_present_values_into(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const double> inflation_linkage,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const std::span<const double> scenario_rate_shocks,
    const std::span<const double> scenario_inflation_rates,
    const std::size_t scenario_count,
    const std::span<double> output
) {
    validate_curve(curve_times, zero_rates);
    if (cashflow_times.size() != cashflow_amounts.size() ||
        cashflow_times.size() != inflation_linkage.size() ||
        scenario_rate_shocks.size() != scenario_count * curve_times.size() ||
        scenario_inflation_rates.size() != scenario_count || output.size() != scenario_count) {
        throw std::invalid_argument("invalid indexed scenario cash-flow buffers");
    }
    for (std::size_t index = 0; index < cashflow_times.size(); ++index) {
        if (!std::isfinite(cashflow_times[index]) || cashflow_times[index] < 0.0 ||
            !std::isfinite(cashflow_amounts[index]) ||
            !std::isfinite(inflation_linkage[index]) || inflation_linkage[index] < 0.0) {
            throw std::invalid_argument("invalid indexed scenario cash flow");
        }
    }
    for (const double shock : scenario_rate_shocks) {
        if (!std::isfinite(shock)) {
            throw std::invalid_argument("scenario shocks must be finite");
        }
    }
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
        output[scenario] = value;
    }
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
    std::vector<double> result(scenario_count, 0.0);
    scenario_indexed_cashflow_present_values_into(
        cashflow_times,
        cashflow_amounts,
        inflation_linkage,
        curve_times,
        zero_rates,
        scenario_rate_shocks,
        scenario_inflation_rates,
        scenario_count,
        result
    );
    return result;
}

}  // namespace qfin
