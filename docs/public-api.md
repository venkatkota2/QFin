# Public API contract

QFin's stable top-level namespace is intended for financial users. It includes
the principal financial domain objects and results, `compile`, `system_info`,
and core convention, valuation, scenario, life, risk, and optimization
utilities.

Implementation-oriented APIs live in their canonical namespaces:

| Namespace | Scope |
| --- | --- |
| `qfin.finance` | Financial models, conventions, valuation, and results |
| `qfin.compiler` | Compilation models, budgets, and result metadata |
| `qfin.representation` | Encodings and reversible-arithmetic plans |
| `qfin.resources` | Logical and device resource reports |
| `qfin.backends` | Private/runtime backend adapters and interoperability |
| `qfin.circuits` | Circuit construction primitives |

The following top-level compatibility aliases are deprecated in QFin 1.1 and
emit `DeprecationWarning`; import them from the listed canonical namespace.

| Deprecated alias | Canonical path |
| --- | --- |
| `qfin.IntegerHingePlan` | `qfin.representation.IntegerHingePlan` |
| `qfin.IntegerPolynomialPlan` | `qfin.representation.IntegerPolynomialPlan` |
| `qfin.IntegerQuadraticTerm` | `qfin.representation.IntegerQuadraticTerm` |
| `qfin.ReversibleAffineTransformPlan` | `qfin.representation.ReversibleAffineTransformPlan` |
| `qfin.StatePreparationCost` | `qfin.representation.StatePreparationCost` |
| `qfin.StructuredLossOraclePlan` | `qfin.representation.StructuredLossOraclePlan` |
| `qfin.ProbabilityTreePreparation` | `qfin.circuits.ProbabilityTreePreparation` |
| `qfin.WalshTerm` | `qfin.circuits.WalshTerm` |

No canonical module path is removed by this deprecation. Other existing
low-level top-level names remain available for compatibility but should be
treated as provisional; new code should prefer their canonical namespaces.

## Command line

The command-line interface intentionally supports only the existing
European-option quantum demonstration. It is not intended to mirror the
financial Python API and does not establish new modelling scope. Run `qfin
--help` for its supported arguments and `qfin --version` for the version read
from installed package metadata.
