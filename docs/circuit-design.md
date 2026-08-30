# QFin circuit design

## 1. Inverse-CDF quantile representation

For `N = 2**n`, the compiler chooses midpoint probabilities within a truncated
central domain and maps them through the terminal distribution:

```text
u_i = q_low + (i + 1/2)(q_high - q_low)/N
x_i = F^-1(u_i)
```

Each point has probability `1/N`. The data register is therefore prepared as
`H^n |0>` using `n` gates and zero stored distribution angles. The nonlinear
lognormal shape lives in the classical mapping from basis labels to `x_i`.

## 2. Sparse Walsh payoff

For normalized payoff `f_i`, the exact objective rotation is

```text
alpha_i = 2 asin(sqrt(f_i)).
```

QFin computes its Walsh coefficients

```text
c_s = (1/N) sum_i alpha_i (-1)^popcount(s & i).
```

The constant term is an `RY(c_0)` rotation. Every other term is a PennyLane
`PauliRot(c_s, Z...ZY)` on the data bits selected by mask `s` and the objective
qubit. On basis state `|i>`, that term contributes the signed angle
`c_s (-1)^popcount(s & i)`. All terms commute.

Terms are added in descending `|c_s|`. After each term, QFin measures the exact
discrete financial price error and angle RMSE classically. Compilation stops
when both tolerances are met. If `payoff_max_terms` prevents convergence, the
model remains inspectable and executable but `met_tolerance` is false.

## 3. Amplitude-estimation iterate

The prepared state has an objective success probability equal to the compiled
payoff expectation. QFin implements

```text
Q = -A S_0 A† S_good.
```

`S_good` is a Z gate on the objective qubit. `S_0` flips the phase of the
all-zero state using X gates, Hadamards, a multi-controlled X, and one reusable
work qubit. MLAE executes selected powers of `Q` and fits their binomial
observations without phase-estimation qubits.

## 4. Resource consequences

| Quantity | Dense v0.1 | Structured v0.2 | Quantile/Walsh v0.3 |
| --- | ---: | ---: | ---: |
| Stored joint-unitary entries | `4**(n+1)` | `0` | `0` |
| Distribution parameters | In matrix | `2**n - 1` | `0` |
| Distribution gates | Backend decomposition | `2**n - 1` | `n` Hadamards |
| Payoff parameters | In matrix | `2**n` | `K <= 2**n` |
| Work qubits | `0` | `1` | `1` |
| Default representation | Joint state | Uniform-price grid | Quantile grid |

`K` depends on the payoff, qubit resolution, and requested tolerances. The
compiler reports `K/2**n`, approximation errors, and failure to converge under
a cap. This milestone removes the exponential distribution-angle table but
does not claim that every payoff has a compact Walsh expansion.

## 5. Empirical tail-risk objectives (v0.5)

ALM, life, or factor scenarios produce finite losses `L_i` with probabilities
`p_i`. Unlike the option quantile path, the weights need not be uniform, so the
v0.5 risk runtime uses the structured probability-tree loader.

The same normalized-objective interface constructs:

```text
tail probability:  f_i(K) = 1[L_i > K]
CDF search:         f_i(K) = 1[L_i <= K]
tail excess:        f_i(v) = max(L_i-v, 0) / max_j max(L_j-v, 0).
```

All three use the existing Grover iterate and MLAE implementation. VaR wraps
CDF amplitudes in a classical binary search over occupied loss points. CVaR
uses the selected threshold and the tail-excess identity

```text
CVaR_alpha = v + E[max(L-v, 0)] / (1-alpha).
```

There is no new state-vector simulator. `RiskPennyLaneBackend` orchestrates the
existing structured circuit, and PennyLane-Lightning performs simulation.

The first risk oracle is exact on the encoded grid but not asymptotically
compact: distribution and objective rotations are both `O(2**n)`. Resource
reports expose that cost, threshold evaluations, and classical preprocessing.

## 6. Device decomposition and routing (v0.7)

The compressed and structured runtimes now expose measurement-free circuit
tapes without changing their execution semantics. QFin recursively decomposes
those tapes into the portable basis

```text
{RX, RY, RZ, CNOT}
```

and then routes CNOT edges over an explicit coupling graph. Inserted SWAPs are
decomposed to three CNOTs. Reports retain pre-routing and post-routing depth,
gate counts, used edges, and the final logical-to-physical permutation.

The all-to-all and line targets are synthetic comparison surfaces. They are
not calibrated processors. OpenQASM export measures every physical wire and
reports the routed location of the objective qubit. Numerical tests load the
exported Qiskit circuit as a state vector and compare that physical objective
probability with the original PennyLane circuit.

Noise analysis is separate from target routing. It uses explicit local channel
assumptions on `default.mixed`, global unitary folding, and polynomial
zero-noise extrapolation. This separation prevents topology counts from being
misrepresented as calibrated hardware accuracy.
