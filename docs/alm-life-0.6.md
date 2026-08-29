# QFin 0.6: multi-period ALM and life foundations

QFin 0.6 extends the finance preprocessing layer while preserving the original
responsibility boundary:

- Python owns financial objects, validation, scenario construction,
  orchestration, reporting, compiler policy, and PennyLane circuits;
- QFin-owned C++20 executes batched financial and actuarial loops; and
- PennyLane-Lightning C++ remains the quantum state-vector simulator.

No QFin-native kernel applies quantum gates or evolves a state vector.

## Economic scenario paths

`EconomicScenarioSet` stores:

| Factor | Shape | Semantics |
| --- | --- | --- |
| Zero-rate shocks | scenario × period × curve node | Additive continuously compounded rate shock |
| Credit spread | scenario × period | Additive bond discount-rate shock |
| Equity | scenario × period | One-period total return |
| Inflation | scenario × period | One-period inflation rate |
| Mortality | scenario × period | Non-negative multiplier on base `qx` |
| Lapse | scenario × period | Non-negative multiplier on base lapse |

Scenario probabilities are normalized once and reused by financial results and
loss distributions. Labels must be unique. The period length is explicit.
Every set records a non-empty dependence-assumption string.

`EconomicScenarioSet.correlated_gaussian(...)` validates a six-factor
correlation matrix and produces independent period innovations. Equity and
inflation use log-return mappings; mortality and lapse use lognormal
multipliers. This generator is useful for deterministic tests and research
examples, but it is not a calibrated economic-scenario generator and does not
claim realistic tails.

## ALM execution modes

`ALMModel.evaluate()` retains base PV, surplus, funding, duration-gap, and
convexity-gap reporting. `AssetPortfolio` additionally accepts aggregate equity
and cash values. `LiabilityPortfolio` accepts a non-negative inflation-linkage
exponent per cash flow.

`run_factor_scenarios()` performs one-period full revaluation. Its attribution
evaluates rates, credit spread, equity, and inflation in isolation and defines
interaction exactly as:

```text
interaction = full surplus change - sum(isolated factor changes)
```

Consequently, the reported factors plus interaction always reconcile to the
full scenario change. Mortality and lapse stay on the shared scenario object
but affect life projection, not deterministic liability cash flows.

`project_paths()` performs a multi-period roll-forward:

1. remaining bond cash flows are revalued under stochastic zero rates and
   credit spreads;
2. coupons and principal received during a period are reinvested to period end;
3. cash earns the average shocked short rate over the period;
4. aggregate equity exposure earns the supplied scenario return;
5. inflation-linked liability payments are deducted proportionally from the
   asset buckets when due;
6. remaining liability PV is revalued at the new horizon; and
7. an optional `RebalancingStrategy` restores a target equity weight, preserves
   the bond/cash mix inside the non-equity allocation, and
   deducts proportional transaction costs.

The result contains scenario × period asset, bond, cash, equity, liability,
surplus, funding, payment, and transaction-cost aggregates. It does not expose an
instrument cube. `loss_distribution(period=-1)` maps deterioration from the
initial surplus into the probability-aware tail-risk layer.

`sensitivities()` reports transparent forward bump-and-revalue impacts for
rates, spreads, equity, and inflation. It is a diagnostic, not a replacement
for full nonlinear scenarios.

## Life products and states

`LifePolicy.product_type` currently supports:

| Product | Initial foundation |
| --- | --- |
| `term_life` | Premiums, expenses, death and disability benefits |
| `participating_life` | Term protection plus compound bonus and maturity benefit |
| `universal_life` | Credited account, premium, charge, surrender, death, and maturity values |
| `annuity` | Survival-contingent annual benefit |

The annual state model tracks active, disabled, and dead lives. Surviving
active lives can become disabled; surviving disabled lives can recover;
mortality applies separately with a disabled-life multiplier. Lapse applies
after mortality and state transitions. Benefits can carry an inflation-linkage
exponent.

`PolicyModelPointSet.from_policies()` groups exactly equal immutable policies
and sums their exposure counts while preserving first-seen order. It reports
the model-point count, represented policy count, and compression ratio. This
is exact grouping, not approximate clustering.

`LifeAssumptionSet` aliases `ProjectionAssumptions` and contains reusable
mortality, lapse, disability, recovery, expense, crediting, inflation, premium,
and benefit assumptions. Scalar or annual paths are validated against the
projection horizon.

`project_liabilities()` reports aggregate cash flows and states, per-model-point
PVs, product-level PV attribution, portfolio PV, duration, and the selected
engine. Defaults preserve the original term-life semantics from QFin 0.4.

## Scenario life projection and memory

`project_liability_scenarios()` applies rate, inflation, mortality, and lapse
paths to model points. It has independent `scenario_chunk_size` and
`policy_chunk_size` controls. Each chunk crosses the Python/C++ boundary once;
C++ runs the scenario × model-point × year loop and returns five arrays over
the scenario chunk:

- present values;
- expected premiums;
- expected benefits;
- expected expenses; and
- expected surrenders.

The final output therefore scales with scenario count, not scenario × policy ×
time. `working_set_estimate_bytes` reports the deterministic numeric buffers
used by the configured peak chunk; allocator and interpreter overhead are not
included.

`LifeScenarioResult.loss_distribution()` defines loss as scenario liability PV
minus base liability PV. `life_sensitivities()` separately reports forward
mortality, lapse, rate, and expense impacts.

## Compiler connection

The financial engine does not silently select a quantum workflow:

```python
scenario_result = alm.project_paths(economic_scenarios)
losses = scenario_result.loss_distribution()
problem = qfin.CVaR(losses, confidence=0.995)
compiled = qfin.compile(problem, target_error=100_000)
```

The same boundary applies to `LifeScenarioResult`. Capability metadata
distinguishes the financial model, classical/native implementation, finite
distribution representation, and implemented quantum tail-risk algorithm.
QFin's first empirical state and objective loaders still require
`O(2**data_qubits)` rotations; the bridge does not imply quantum advantage.

## Numerical and implementation limits

- Life steps are annual and expected-value based.
- Product rules omit taxes, reserves, guarantees, dynamic policyholder
  behavior, management actions, and jurisdiction-specific contract terms.
- Bond assets are fixed-rate cash flows; equity and cash are aggregate
  allocations rather than security-level models.
- Spread shocks are aggregate additive rates and do not model migration or
  default.
- Rebalancing uses a target aggregate equity weight.
- The portable native kernels are deterministic and single-threaded.
- The supplied Gaussian generator is synthetic and does not model calibrated
  heavy tails or temporal dependence.

NumPy reference paths remain available with `engine="numpy"`; native dispatch
can be forced with `engine="native"`. Both are parity-tested, including chunk
invariance and analytical product cases. See
[alm-life-performance.md](alm-life-performance.md) for measured results.

The final measured 0.6 ALM path speedups range from 0.88× to 1.09× and do not
establish a native crossover, so `engine="auto"` deliberately selects NumPy
there. The native ALM path remains available for controlled
profiling and future kernel work. The life benchmark established a native
benefit even for one policy, one scenario, and one year; non-empty base and
scenario life workloads therefore dispatch to native execution when the
extension is available.
