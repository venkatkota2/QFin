#include "qfin/fixed_income.hpp"

#include "qfin/curves.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qfin {
namespace {

void validate_cashflows(
    const std::span<const double> times,
    const std::span<const double> amounts,
    const std::span<const std::int64_t> offsets
) {
    if (times.size() != amounts.size() || offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<std::int64_t>(times.size())) {
        throw std::invalid_argument("invalid flattened cash-flow buffers");
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            throw std::invalid_argument("cash-flow offsets must be non-decreasing");
        }
    }
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (!std::isfinite(times[index]) || !std::isfinite(amounts[index]) || times[index] < 0.0) {
            throw std::invalid_argument("cash-flow times and amounts must be finite");
        }
    }
}

double yield_discount(const double time, const double yield, const std::int32_t frequency) {
    const double base = 1.0 + yield / static_cast<double>(frequency);
    if (!(base > 0.0) || !std::isfinite(base)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::pow(base, -static_cast<double>(frequency) * time);
}

double price_one_from_yield(
    const std::span<const double> times,
    const std::span<const double> amounts,
    const double yield,
    const std::int32_t frequency
) {
    double price = 0.0;
    for (std::size_t index = 0; index < times.size(); ++index) {
        price += amounts[index] * yield_discount(times[index], yield, frequency);
    }
    return price;
}

}  // namespace

BatchBondMetrics price_cashflow_batches(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> curve_times,
    const std::span<const double> zero_rates,
    const double parallel_shift
) {
    validate_cashflows(cashflow_times, cashflow_amounts, offsets);
    validate_curve(curve_times, zero_rates);
    if (!std::isfinite(parallel_shift)) {
        throw std::invalid_argument("parallel shift must be finite");
    }
    const std::size_t count = offsets.size() - 1;
    BatchBondMetrics result{
        std::vector<double>(count),
        std::vector<double>(count),
        std::vector<double>(count),
        std::vector<double>(count),
    };
    constexpr double bump = 1.0e-4;
    for (std::size_t instrument = 0; instrument < count; ++instrument) {
        double price = 0.0;
        double first_moment = 0.0;
        double second_moment = 0.0;
        double price_down = 0.0;
        double price_up = 0.0;
        const auto begin = static_cast<std::size_t>(offsets[instrument]);
        const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
        for (std::size_t index = begin; index < end; ++index) {
            const double time = cashflow_times[index];
            const double amount = cashflow_amounts[index];
            const double present_value = amount * discount_factor(
                time, curve_times, zero_rates, parallel_shift
            );
            price += present_value;
            first_moment += time * present_value;
            second_moment += time * time * present_value;
            price_down += present_value * std::exp(bump * time);
            price_up += present_value * std::exp(-bump * time);
        }
        result.prices[instrument] = price;
        if (std::abs(price) > 1.0e-15) {
            result.macaulay_durations[instrument] = first_moment / price;
            result.convexities[instrument] = second_moment / price;
        }
        result.dv01[instrument] = 0.5 * (price_down - price_up);
    }
    return result;
}

BatchBondMetrics price_cashflow_batches_from_yield(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> yields,
    const std::span<const std::int32_t> frequencies
) {
    validate_cashflows(cashflow_times, cashflow_amounts, offsets);
    const std::size_t count = offsets.size() - 1;
    if (yields.size() != count || frequencies.size() != count) {
        throw std::invalid_argument("one yield and frequency are required per instrument");
    }
    BatchBondMetrics result{
        std::vector<double>(count),
        std::vector<double>(count),
        std::vector<double>(count),
        std::vector<double>(count),
    };
    constexpr double bump = 1.0e-4;
    for (std::size_t instrument = 0; instrument < count; ++instrument) {
        const auto frequency = frequencies[instrument];
        const double yield = yields[instrument];
        if (frequency <= 0 || !std::isfinite(yield) ||
            1.0 + yield / static_cast<double>(frequency) <= 0.0) {
            throw std::invalid_argument("invalid yield or coupon frequency");
        }
        double price = 0.0;
        double first_moment = 0.0;
        double convexity_numerator = 0.0;
        const auto begin = static_cast<std::size_t>(offsets[instrument]);
        const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
        for (std::size_t index = begin; index < end; ++index) {
            const double time = cashflow_times[index];
            const double present_value = cashflow_amounts[index] *
                                         yield_discount(time, yield, frequency);
            price += present_value;
            first_moment += time * present_value;
            const double periods = static_cast<double>(frequency) * time;
            convexity_numerator += periods * (periods + 1.0) * present_value;
        }
        result.prices[instrument] = price;
        if (std::abs(price) > 1.0e-15) {
            result.macaulay_durations[instrument] = first_moment / price;
            const double denominator = static_cast<double>(frequency * frequency) *
                                       std::pow(1.0 + yield / frequency, 2.0);
            result.convexities[instrument] = convexity_numerator / (price * denominator);
        }
        const auto times = cashflow_times.subspan(begin, end - begin);
        const auto amounts = cashflow_amounts.subspan(begin, end - begin);
        const double up = price_one_from_yield(times, amounts, yield + bump, frequency);
        if (1.0 + (yield - bump) / static_cast<double>(frequency) > 0.0) {
            const double down = price_one_from_yield(times, amounts, yield - bump, frequency);
            result.dv01[instrument] = 0.5 * (down - up);
        } else {
            result.dv01[instrument] = price - up;
        }
    }
    return result;
}

YieldSolveResult solve_yields_from_prices(
    const std::span<const double> cashflow_times,
    const std::span<const double> cashflow_amounts,
    const std::span<const std::int64_t> offsets,
    const std::span<const double> target_prices,
    const std::span<const std::int32_t> frequencies,
    const double tolerance,
    const std::int32_t max_iterations
) {
    validate_cashflows(cashflow_times, cashflow_amounts, offsets);
    const std::size_t count = offsets.size() - 1;
    if (target_prices.size() != count || frequencies.size() != count ||
        !(tolerance > 0.0) || !std::isfinite(tolerance) || max_iterations <= 0) {
        throw std::invalid_argument("invalid yield-solver inputs");
    }
    YieldSolveResult result{
        std::vector<double>(count),
        std::vector<std::uint8_t>(count),
        std::vector<std::int32_t>(count),
    };
    for (std::size_t instrument = 0; instrument < count; ++instrument) {
        const auto frequency = frequencies[instrument];
        const double target = target_prices[instrument];
        if (frequency <= 0 || !(target > 0.0) || !std::isfinite(target)) {
            throw std::invalid_argument("prices must be finite and positive");
        }
        const auto begin = static_cast<std::size_t>(offsets[instrument]);
        const auto end = static_cast<std::size_t>(offsets[instrument + 1]);
        if (begin == end) {
            throw std::invalid_argument("cannot solve yield for an empty cash-flow stream");
        }
        const auto times = cashflow_times.subspan(begin, end - begin);
        const auto amounts = cashflow_amounts.subspan(begin, end - begin);
        double lower = -0.999999 * static_cast<double>(frequency);
        double upper = 1.0;
        double upper_price = price_one_from_yield(times, amounts, upper, frequency);
        while (upper_price > target && upper < 1.0e6) {
            upper = 2.0 * upper + 1.0;
            upper_price = price_one_from_yield(times, amounts, upper, frequency);
        }
        if (upper_price > target) {
            result.yields[instrument] = upper;
            result.iterations[instrument] = 0;
            continue;
        }
        bool converged = false;
        double middle = 0.0;
        std::int32_t iteration = 0;
        for (; iteration < max_iterations; ++iteration) {
            middle = 0.5 * (lower + upper);
            const double value = price_one_from_yield(times, amounts, middle, frequency);
            const double error = value - target;
            if (std::abs(error) <= tolerance * std::max(1.0, target) ||
                std::abs(upper - lower) <= tolerance) {
                converged = true;
                ++iteration;
                break;
            }
            if (error > 0.0) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        result.yields[instrument] = middle;
        result.converged[instrument] = static_cast<std::uint8_t>(converged);
        result.iterations[instrument] = iteration;
    }
    return result;
}

}  // namespace qfin
