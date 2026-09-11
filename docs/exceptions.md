# Exception compatibility

`QFinError` is the common library root. Existing documented catch behavior remains
valid. External library failures that are not explicitly translated can retain
their original exception type; this is not a promise to wrap every Python error.

| Exception | Purpose and compatibility |
| --- | --- |
| `QFinValidationError` | Financial values, shapes and mathematical domains; also `ValueError`. |
| `QFinTypeError` | Unsupported public object types; also `TypeError`. |
| `NumericalError` | Numerical execution failure; also `ArithmeticError`. |
| `PricingError` | Pricing failure; also `NumericalError` and `ValueError`. |
| `CalibrationError` | Calibration failure; also `ValueError`. Existing `CurveBootstrapError` inherits this and retains its canonical module. |
| `ScenarioError` | Scenario validation; inherits `QFinValidationError`. |
| `CompilationError` | Unsupported or infeasible compilation. |
| `BackendError` | Backend execution/configuration failure. |
| `BackendUnavailableError` | Missing optional backend; inherits `BackendError`. Existing `NativeBackendUnavailableError` inherits this. |
| `ResourceLimitError` | Explicit memory, wire, code or dimensional resource limit. |
| `OptimizationError` | Existing optimization failure. |
| `FinancialValidationError` | Validation report rejection; preserves `AssertionError` compatibility. |

The native binding translates invalid arguments and out-of-range values to
`QFinValidationError`, allocation/length failures to `ResourceLimitError`, and
overflow errors to `NumericalError`. Pybind's own argument-conversion errors remain
ordinary binding `TypeError`s. Native numeric result-range checks are validation
errors so existing `ValueError` handlers keep working.

Public validation does not rely on `assert`. Remaining production assertions
express internal normalized-state/type invariants; invalid public financial inputs
are still rejected under `python -O`, as exercised by the regression suite.
