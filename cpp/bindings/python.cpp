#include "qfin/fixed_income.hpp"
#include "qfin/mortality.hpp"
#include "qfin/projections.hpp"
#include "qfin/risk.hpp"
#include "qfin/scenarios.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <typename T>
using InputArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
std::span<const T> as_span(const InputArray<T>& array) {
    if (array.ndim() != 1) {
        throw py::value_error("native numeric buffers must be one-dimensional");
    }
    const auto buffer = array.request();
    return {static_cast<const T*>(buffer.ptr), static_cast<std::size_t>(buffer.size)};
}

template <typename T>
std::span<const T> as_flat_span(const InputArray<T>& array) {
    const auto buffer = array.request();
    return {static_cast<const T*>(buffer.ptr), static_cast<std::size_t>(buffer.size)};
}

template <typename T>
py::array_t<T> move_array(std::vector<T>&& values) {
    py::array_t<T> output(values.size());
    const auto buffer = output.request();
    auto* destination = static_cast<T*>(buffer.ptr);
    std::move(values.begin(), values.end(), destination);
    return output;
}

py::dict metrics_to_dict(qfin::BatchBondMetrics&& metrics) {
    py::dict result;
    result["prices"] = move_array(std::move(metrics.prices));
    result["macaulay_durations"] = move_array(std::move(metrics.macaulay_durations));
    result["convexities"] = move_array(std::move(metrics.convexities));
    result["dv01"] = move_array(std::move(metrics.dv01));
    return result;
}

}  // namespace

PYBIND11_MODULE(_qfin_native, module) {
    module.doc() = "QFin-owned C++20 financial kernels (not a quantum simulator)";
    module.attr("cpp_standard") = "C++20";
    module.attr("compiler") = QFIN_CXX_COMPILER;
    module.attr("backend_name") = "qfin-native";

    module.def(
        "price_cashflow_batches",
        [](const InputArray<double>& cashflow_times,
           const InputArray<double>& cashflow_amounts,
           const InputArray<std::int64_t>& offsets,
           const InputArray<double>& curve_times,
           const InputArray<double>& zero_rates,
           const double parallel_shift) {
            const auto times_span = as_span(cashflow_times);
            const auto amounts_span = as_span(cashflow_amounts);
            const auto offsets_span = as_span(offsets);
            const auto curve_times_span = as_span(curve_times);
            const auto rates_span = as_span(zero_rates);
            qfin::BatchBondMetrics metrics;
            {
                py::gil_scoped_release release;
                metrics = qfin::price_cashflow_batches(
                    times_span,
                    amounts_span,
                    offsets_span,
                    curve_times_span,
                    rates_span,
                    parallel_shift
                );
            }
            return metrics_to_dict(std::move(metrics));
        },
        py::arg("cashflow_times"),
        py::arg("cashflow_amounts"),
        py::arg("offsets"),
        py::arg("curve_times"),
        py::arg("zero_rates"),
        py::arg("parallel_shift") = 0.0
    );

    module.def(
        "price_cashflow_batches_from_yield",
        [](const InputArray<double>& cashflow_times,
           const InputArray<double>& cashflow_amounts,
           const InputArray<std::int64_t>& offsets,
           const InputArray<double>& yields,
           const InputArray<std::int32_t>& frequencies) {
            const auto times_span = as_span(cashflow_times);
            const auto amounts_span = as_span(cashflow_amounts);
            const auto offsets_span = as_span(offsets);
            const auto yields_span = as_span(yields);
            const auto frequencies_span = as_span(frequencies);
            qfin::BatchBondMetrics metrics;
            {
                py::gil_scoped_release release;
                metrics = qfin::price_cashflow_batches_from_yield(
                    times_span,
                    amounts_span,
                    offsets_span,
                    yields_span,
                    frequencies_span
                );
            }
            return metrics_to_dict(std::move(metrics));
        }
    );

    module.def(
        "solve_yields_from_prices",
        [](const InputArray<double>& cashflow_times,
           const InputArray<double>& cashflow_amounts,
           const InputArray<std::int64_t>& offsets,
           const InputArray<double>& prices,
           const InputArray<std::int32_t>& frequencies,
           const double tolerance,
           const std::int32_t max_iterations) {
            const auto times_span = as_span(cashflow_times);
            const auto amounts_span = as_span(cashflow_amounts);
            const auto offsets_span = as_span(offsets);
            const auto prices_span = as_span(prices);
            const auto frequencies_span = as_span(frequencies);
            qfin::YieldSolveResult solved;
            {
                py::gil_scoped_release release;
                solved = qfin::solve_yields_from_prices(
                    times_span,
                    amounts_span,
                    offsets_span,
                    prices_span,
                    frequencies_span,
                    tolerance,
                    max_iterations
                );
            }
            py::dict result;
            result["yields"] = move_array(std::move(solved.yields));
            result["converged"] = move_array(std::move(solved.converged));
            result["iterations"] = move_array(std::move(solved.iterations));
            return result;
        },
        py::arg("cashflow_times"),
        py::arg("cashflow_amounts"),
        py::arg("offsets"),
        py::arg("prices"),
        py::arg("frequencies"),
        py::arg("tolerance") = 1.0e-12,
        py::arg("max_iterations") = 200
    );

    module.def(
        "scenario_portfolio_present_values",
        [](const InputArray<double>& cashflow_times,
           const InputArray<double>& cashflow_amounts,
           const InputArray<std::int64_t>& offsets,
           const InputArray<double>& position_weights,
           const InputArray<double>& curve_times,
           const InputArray<double>& zero_rates,
           const InputArray<double>& scenario_shocks) {
            if (scenario_shocks.ndim() != 2) {
                throw py::value_error("scenario_shocks must be two-dimensional");
            }
            const auto times_span = as_span(cashflow_times);
            const auto amounts_span = as_span(cashflow_amounts);
            const auto offsets_span = as_span(offsets);
            const auto weights_span = as_span(position_weights);
            const auto curve_times_span = as_span(curve_times);
            const auto rates_span = as_span(zero_rates);
            const auto shocks_span = as_flat_span(scenario_shocks);
            std::vector<double> values;
            {
                py::gil_scoped_release release;
                values = qfin::scenario_portfolio_present_values(
                    times_span,
                    amounts_span,
                    offsets_span,
                    weights_span,
                    curve_times_span,
                    rates_span,
                    shocks_span,
                    static_cast<std::size_t>(scenario_shocks.shape(0))
                );
            }
            return move_array(std::move(values));
        }
    );

    module.def(
        "mortality_rates",
        [](const InputArray<double>& query_ages,
           const InputArray<double>& table_ages,
           const InputArray<double>& qx) {
            const auto queries = as_span(query_ages);
            const auto ages_span = as_span(table_ages);
            const auto qx_span = as_span(qx);
            qfin::validate_mortality_table(ages_span, qx_span);
            std::vector<double> values(static_cast<std::size_t>(query_ages.size()));
            {
                py::gil_scoped_release release;
                for (std::size_t index = 0; index < queries.size(); ++index) {
                    values[index] = qfin::mortality_rate(queries[index], ages_span, qx_span);
                }
            }
            return move_array(std::move(values));
        }
    );

    module.def(
        "project_term_life_policies",
        [](const InputArray<double>& attained_ages,
           const InputArray<double>& sums_assured,
           const InputArray<double>& annual_premiums,
           const InputArray<std::int32_t>& remaining_terms,
           const InputArray<double>& table_ages,
           const InputArray<double>& mortality_qx,
           const InputArray<double>& lapse_rates,
           const double expense_per_policy,
           const double mortality_multiplier,
           const InputArray<double>& curve_times,
           const InputArray<double>& zero_rates) {
            const auto attained_span = as_span(attained_ages);
            const auto assured_span = as_span(sums_assured);
            const auto premiums_span = as_span(annual_premiums);
            const auto terms_span = as_span(remaining_terms);
            const auto table_ages_span = as_span(table_ages);
            const auto qx_span = as_span(mortality_qx);
            const auto lapse_span = as_span(lapse_rates);
            const auto curve_times_span = as_span(curve_times);
            const auto zero_rates_span = as_span(zero_rates);
            qfin::PolicyProjectionResult projection;
            {
                py::gil_scoped_release release;
                projection = qfin::project_term_life_policies(
                    attained_span,
                    assured_span,
                    premiums_span,
                    terms_span,
                    table_ages_span,
                    qx_span,
                    lapse_span,
                    expense_per_policy,
                    mortality_multiplier,
                    curve_times_span,
                    zero_rates_span
                );
            }
            py::dict result;
            result["expected_premiums"] = move_array(std::move(projection.expected_premiums));
            result["expected_benefits"] = move_array(std::move(projection.expected_benefits));
            result["expected_expenses"] = move_array(std::move(projection.expected_expenses));
            result["net_liability_cashflows"] = move_array(
                std::move(projection.net_liability_cashflows)
            );
            result["in_force"] = move_array(std::move(projection.in_force));
            result["policy_present_values"] = move_array(
                std::move(projection.policy_present_values)
            );
            result["present_value"] = projection.present_value;
            result["duration"] = projection.duration;
            return result;
        }
    );

    module.def(
        "aggregate_tail_risk",
        [](const InputArray<double>& losses,
           const InputArray<double>& probabilities,
           const double confidence) {
            const auto losses_span = as_span(losses);
            const auto probabilities_span = as_span(probabilities);
            qfin::RiskSummaryNative summary;
            {
                py::gil_scoped_release release;
                summary = qfin::aggregate_tail_risk(
                    losses_span, probabilities_span, confidence
                );
            }
            py::dict result;
            result["mean"] = summary.mean;
            result["standard_deviation"] = summary.standard_deviation;
            result["minimum"] = summary.minimum;
            result["maximum"] = summary.maximum;
            result["value_at_risk"] = summary.value_at_risk;
            result["expected_shortfall"] = summary.expected_shortfall;
            return result;
        }
    );
}
