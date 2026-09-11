#include "qfin/checked_size.hpp"
#include "qfin/risk.hpp"

#include "qfin/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qfin {

RiskSummaryNative aggregate_tail_risk(
    const std::span<const double> losses,
    const std::span<const double> probabilities,
    const double confidence
) {
    if (losses.empty() || losses.size() != probabilities.size() ||
        !std::isfinite(confidence) || confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("invalid loss distribution or confidence level");
    }
    double maximum_probability = 0.0;
    NeumaierSum probability_total_sum;
    NeumaierSum mean_sum;
    std::vector<std::pair<double, double>> ordered;
    ordered.reserve(checked_allocation(losses.size(), 2));
    for (std::size_t index = 0; index < losses.size(); ++index) {
        const double loss = losses[index];
        const double probability = probabilities[index];
        if (!std::isfinite(loss) || !std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument("losses must be finite and probabilities non-negative");
        }
        maximum_probability = std::max(maximum_probability, probability);
        probability_total_sum.add(probability);
        mean_sum.add(probability * loss);
        ordered.emplace_back(loss, probability);
    }
    if (!(maximum_probability > 0.0)) {
        throw std::invalid_argument("probabilities must have positive total mass");
    }
    double probability_total = probability_total_sum.value();
    double weighted_mean = mean_sum.value();
    if (!std::isfinite(probability_total) || !std::isfinite(weighted_mean)) {
        NeumaierSum scaled_probability_total;
        NeumaierSum scaled_mean;
        for (auto& item : ordered) {
            item.second /= maximum_probability;
            scaled_probability_total.add(item.second);
            scaled_mean.add(item.second * item.first);
        }
        probability_total = scaled_probability_total.value();
        weighted_mean = scaled_mean.value();
        if (!std::isfinite(probability_total) || !std::isfinite(weighted_mean)) {
            throw std::invalid_argument("weighted loss moments must remain finite");
        }
    }
    const double mean = weighted_mean / probability_total;
    for (auto& item : ordered) {
        item.second /= probability_total;
    }
    std::sort(ordered.begin(), ordered.end());
    NeumaierSum variance_sum;
    for (const auto& [loss, probability] : ordered) {
        variance_sum.add(probability * (loss - mean) * (loss - mean));
    }
    const double variance = variance_sum.value();
    if (!std::isfinite(variance)) {
        throw std::invalid_argument("weighted loss moments exceed the finite double range");
    }

    double cumulative = 0.0;
    double value_at_risk = ordered.back().first;
    for (const auto& [loss, probability] : ordered) {
        const double next = cumulative + probability;
        if (cumulative < confidence && next >= confidence) {
            value_at_risk = loss;
        }
        cumulative = next;
    }
    NeumaierSum excess_sum;
    for (const auto& [loss, probability] : ordered) {
        excess_sum.add(probability * std::max(0.0, loss - value_at_risk));
    }
    const double expected_shortfall = value_at_risk + excess_sum.value() / (1.0 - confidence);
    if (!std::isfinite(expected_shortfall)) {
        throw std::invalid_argument("expected shortfall exceeds the finite double range");
    }
    return RiskSummaryNative{
        mean,
        std::sqrt(std::max(0.0, variance)),
        ordered.front().first,
        ordered.back().first,
        value_at_risk,
        expected_shortfall,
    };
}

}  // namespace qfin
