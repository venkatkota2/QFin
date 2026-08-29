#include "qfin/risk.hpp"

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
    double probability_total = 0.0;
    double mean = 0.0;
    std::vector<std::pair<double, double>> ordered;
    ordered.reserve(losses.size());
    for (std::size_t index = 0; index < losses.size(); ++index) {
        const double loss = losses[index];
        const double probability = probabilities[index];
        if (!std::isfinite(loss) || !std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument("losses must be finite and probabilities non-negative");
        }
        maximum_probability = std::max(maximum_probability, probability);
        probability_total += probability;
        mean += probability * loss;
        ordered.emplace_back(loss, probability);
    }
    if (!(maximum_probability > 0.0)) {
        throw std::invalid_argument("probabilities must have positive total mass");
    }
    if (!std::isfinite(probability_total) || !std::isfinite(mean)) {
        probability_total = 0.0;
        mean = 0.0;
        for (auto& item : ordered) {
            item.second /= maximum_probability;
            probability_total += item.second;
            mean += item.second * item.first;
        }
        if (!std::isfinite(probability_total) || !std::isfinite(mean)) {
            throw std::invalid_argument("weighted loss moments must remain finite");
        }
    }
    mean /= probability_total;
    for (auto& item : ordered) {
        item.second /= probability_total;
    }
    std::sort(ordered.begin(), ordered.end());
    double variance = 0.0;
    for (const auto& [loss, probability] : ordered) {
        variance += probability * (loss - mean) * (loss - mean);
    }
    if (!std::isfinite(variance)) {
        throw std::invalid_argument("weighted loss moments exceed the finite double range");
    }

    double cumulative = 0.0;
    double value_at_risk = ordered.back().first;
    double tail_integral = 0.0;
    for (const auto& [loss, probability] : ordered) {
        const double next = cumulative + probability;
        if (cumulative < confidence && next >= confidence) {
            value_at_risk = loss;
        }
        const double overlap = std::max(0.0, next - std::max(cumulative, confidence));
        tail_integral += overlap * loss;
        cumulative = next;
    }
    const double expected_shortfall = tail_integral / (1.0 - confidence);
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
