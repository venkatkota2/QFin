# Fixed-income and life-insurance ALM

QFin 0.4 adds a classical cash-flow engine and a quantum scenario-risk
compiler. The division is intentional:

```text
Bond and policy inputs
        -> expected cash flows
        -> curve valuation and rate scenarios (NumPy)
        -> shortfall distribution
        -> amplitude-estimation circuit (PennyLane)
        -> lightning.qubit simulation
```

Cash-flow generation and curve valuation are ordinary deterministic numerical
work and are faster as vectorized classical operations. PennyLane is used at
the quantum-relevant boundary: encoding a discrete scenario distribution and
estimating its shortfall probability or expected shortfall.

## Model construction

```python
import numpy as np
import qfin

curve = qfin.DiscountCurve(
    times=np.array([0, 1, 5, 10, 20, 30]),
    zero_rates=np.array([0.035, 0.036, 0.038, 0.040, 0.042, 0.043]),
)

assets = qfin.FixedIncomePortfolio((
    qfin.BondPosition(qfin.FixedRateBond(1_000, 0.04, 10, 2), 600),
    qfin.BondPosition(qfin.FixedRateBond(1_000, 0.045, 20, 2), 450),
))

mortality = qfin.MortalityTable.illustrative_gompertz_makeham()
liabilities = qfin.LifePolicyPortfolio((
    qfin.PolicyPosition(
        qfin.TermLifePolicy(40, 25, 100_000, annual_expense=25),
        80,
    ),
    qfin.PolicyPosition(
        qfin.WholeLifePolicy(50, 50_000, annual_expense=20),
        40,
    ),
))

alm = qfin.AssetLiabilityModel(assets, liabilities, curve, mortality)
base = alm.evaluate()
```

`base` reports asset and liability present values, surplus, funding ratio,
parallel duration, dollar-duration gap, and parallel convexity. Rates are
continuously compounded. A shift of `0.01` means 100 basis points.

The illustrative Gompertz-Makeham table is only synthetic example data. A real
valuation should construct `MortalityTable` from an approved select/ultimate
or aggregate basis appropriate to the use case.

## Fast scenario valuation

```python
shocks = np.linspace(-0.05, 0.05, 100_000)
scenarios = alm.run_parallel_shocks(
    shocks,
    max_working_bytes=64 * 1024 * 1024,
)

print(scenarios.expected_surplus)
print(scenarios.shortfall_probability)
print(scenarios.expected_shortfall)
```

The model aggregates asset and liability cash flows once. For a parallel shift
`s`, it reuses the base discounted cash flows and multiplies by
`exp(-s * t)`. Scenario blocks are evaluated with matrix multiplication and a
bounded temporary array. `max_working_bytes` controls the block size, so large
scenario sets do not require one unbounded scenario-by-cash-flow matrix.

## Quantum ALM risk

```python
shocks = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])

compiled = qfin.compile_alm(
    alm,
    shocks,
    metric="expected_shortfall",
    target_error=25_000,
)

print(compiled.explain())
result = compiled.run(
    shots=2_000,
    schedule=(0, 1, 2, 4),
    seed=7,
)
```

Available objectives are:

- `shortfall_probability`: `E[1(surplus < 0)]`;
- `expected_shortfall`: `E[max(-surplus, 0)]` in the portfolio currency.

If the scenario count is a power of two and probabilities are uniform, QFin
prepares the scenario register with Hadamards and compiles the objective into a
sparse Walsh/Pauli circuit. Otherwise, it uses the exact probability-tree
loader. Both paths report the exact classical scenario value, compiled-circuit
value, approximation error, sampling error, logical qubits, shots, and payoff
term count.

`lightning.qubit` is the default device. It uses PennyLane Lightning's compiled
C++ state-vector engine. A complete MLAE schedule reuses one device instead of
creating a plugin instance for every Grover power. `device_name="auto"` prefers
Lightning and falls back to `default.qubit`; an explicit device never falls
back silently. `precision="complex64"` is an opt-in speed/memory tradeoff, while
the default `complex128` protects financial numerical accuracy.

## Current actuarial assumptions and limits

- Death benefits are paid at the end of the policy year of death.
- Premiums and per-policy expenses occur at the start of each policy year while
  the policy is in force.
- Policy counts can be fractional expected exposures.
- Mortality is deterministic; there are no lapse, morbidity, expense-inflation,
  reserve, tax, reinsurance, or policyholder-option models yet.
- Assets are bullet fixed-rate bonds; credit migration/default, callable bonds,
  and reinvestment strategies are not yet modelled.
- Rate scenarios are parallel zero-curve shifts. Key-rate, stochastic-rate,
  and joint economic/mortality scenarios remain future work.
- Results are research and educational outputs, not a production valuation,
  pricing, capital, or regulatory engine.

The quantum simulator path demonstrates a valid finance-to-circuit mapping. It
does not establish quantum advantage, because scenario loading and objective
synthesis can remain exponential in the number of scenario qubits.
