#pragma once

#include <cmath>

namespace qfin {

class NeumaierSum {
  public:
    void add(const double value) noexcept {
        const double next = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - next) + value;
        } else {
            correction_ += (value - next) + sum_;
        }
        sum_ = next;
    }

    [[nodiscard]] double value() const noexcept {
        return sum_ + correction_;
    }

  private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

}  // namespace qfin
