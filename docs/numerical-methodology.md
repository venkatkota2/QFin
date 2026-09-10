# Numerical methodology

The strongest practical reference is used for each calculation: analytical
identities and Decimal arithmetic where possible, independent QuantLib for the
supported dated bond conventions, then deterministic NumPy/native comparison.
A shared bug in Python and C++ cannot be excluded by parity alone.

Scenario shocks are zero-node changes followed by the base interpolation; slow
per-scenario `curve.shifted` repricing is the regression oracle. Yield inversions
are checked in yield units as well as price units. Key-rate sensitivities use
stable cash-flow PV changes, and a 70-digit Decimal case tests a tiny nonzero node
exposure on a trillion-unit principal.

Portfolio cancellation is measured before selecting a more expensive reduction.
NumPy pairwise summation remains the normal path; severe mixed-sign cancellation
uses `math.fsum`, and corresponding native aggregates use Neumaier compensation.
The trigger compares residual with sum of magnitudes. This does not make every
sum arbitrarily precise: multiplication, discounting and input float rounding
remain finite precision. Positive-only loops retain ordinary arithmetic unless
stress evidence justifies extra cost.

Segmented reductions preserve segment order and support empty segments. Large
buffers use `np.add.reduceat` without a repeated instrument-index array; small
buffers retain the measured simpler path. Cumulative-sum differencing is not used
because a small later segment can lose significance against a large prefix.

`FinancialTolerance` requires both floating-point parity and an independent
financial-materiality bound when configured. Named quantity profiles express
price/PV/DV01/VaR in currency, duration in years, convexity in its defined units and
probability in probability units. Callers choose portfolio-appropriate notional
scales; a relative tolerance alone is not a universal materiality standard.

Negative rates/forwards are not globally forbidden. Nonfinite values, nonpositive
discounts, invalid compounding domains and unbounded covariance-null-space
optimization are mathematical failures. Diagnostics distinguish those failures
from economic unusualness. Unestimated errors and unsuccessful convergence are
never deliberately labelled zero or success.
