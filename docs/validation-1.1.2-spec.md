# 1.1.2 validation specification

The starting commit is `f9d4be69432ee813fccdde609420cdee9eb15580`. Its reproduced
local baseline is 654 passing tests, two optional Qiskit skips, 92.9599% Python
line coverage and 81.0391% branch coverage. The untouched baseline and hardened
installation are isolated. Final measured totals and CI links live in
[the hardening report](hardening-1.1.2.md); earlier milestone numbers remain in
[historical validation](history/validation-1.1.1.md).

Checked-in fixtures contain 1,073 independent configurations: 144 compounding,
60 day count, 52 schedule, 162 yield/bond, 432 curve, four mixed bootstrap, 30 risk,
nine ALM, 144 scalar life and 36 dated bond cases. Python/native parameterization
adds execution paths, not additional independent financial cases. The 26 inline
Decimal/calendar cases additionally cover nonlinear PCHIP, dated bootstrapping
and tiny/long zero-coupon sensitivities. Every corpus record names its origin.
QuantLib is required to regenerate its fixtures, not to run the stored corpus.

The baseline API manifest records 312 exported objects across four documented
namespaces. Compatibility checks preserve existing callable parameters/defaults,
methods, positional dataclass fields and enum members. Representative nested
serialization keys are captured separately from the untouched 1.1.1 installation.
Python's own changing Enum/exception constructor internals are not QFin APIs.

Generated properties check economic identities, yield/price round trips,
scaling, key-rate reconciliation, curve/scenario permutation and chunking,
extreme probability rescaling, life grouping, settlement and optimization budget
constraints. The stress profile has 500 examples per property; two direct native
buffer properties each use 100 deterministic examples. Shrunk invalid adjusted
stub schedules are rejected intentionally; valid schedule generation properties
use unadjusted conventions so they do not assume every business adjustment can
preserve strictly increasing dates.

CI retains the overall pytest-cov 90% combined floor and separately requires
93% line and 81% branch coverage in `examples/check_coverage.py`. Existing numerical
line floors remain finance 95%, compiler 92%, representation 92%, resources 90%,
backends 85% and native loader 95%. Branch floors are 85% for curves/fixed income/
scenarios, 80% for calibration/ALM paths/compiler/MLAE, 90% for life/risk, 75% for
ALM/life scenarios/optimization/representation, and 100% for the native loader.
These are Python coverage gates; C++ branch coverage is not claimed.

Native validation includes direct malformed offsets, zero-length streams,
strides/dtypes/read-only buffers, output lifetimes, non-finite inputs/outputs,
large logical dimensions without materialization and size arithmetic unit checks.
Both GCC and Clang run strict warnings plus address/undefined-behavior sanitizers.
Leak detection is disabled for the Python process; sanitizers do not certify every
possible input or absence of all leaks.

The ordinary test matrix includes deterministic real-circuit risk regressions.
Large empirical coverage, mutation and timing campaigns are scheduled/manual;
timing variation alone cannot fail a PR. Mutation testing is a nine-operator
regression-sensitivity sample, with an unmodified isolated control before mutated
runs. It is not an exhaustive codebase mutation score.

Build gates include editable source, clean sdist installation, five ordinary
OS/interpreter wheel smoke jobs and nine portable wheels. The full Linux 3.12 job
also installs Qiskit and QuantLib; minimum-dependency and isolated-wheel jobs prove
the core remains independent of these optional packages.
