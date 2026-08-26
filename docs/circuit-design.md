# QFin v0.3 circuit design

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
