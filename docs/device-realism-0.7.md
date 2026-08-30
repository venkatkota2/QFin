# QFin 0.7: device realism and interoperability

QFin 0.7 adds a tested boundary between logical finance circuits and realistic
device constraints. It does not add a QFin simulator, submit hardware jobs, or
claim that a routed circuit is currently useful on a quantum processor.

## Tested execution devices

The runtime registry contains only:

| Device | Role |
| --- | --- |
| `lightning.qubit` | Preferred ideal state-vector simulator when PennyLane-Lightning is installed |
| `default.qubit` | Tested ideal PennyLane fallback |
| `default.mixed` | Tested mixed-state simulator used by explicit noise experiments |

`device_name="auto"` prefers Lightning. An unregistered name such as
`lightning.gpu` is rejected instead of being presented as supported. OpenQASM
or Qiskit export remains available for interoperability without implying that
QFin can execute the resulting circuit on a provider.

## Portable target resources

`DeviceTarget` describes a basis gate set and an undirected coupling graph.
The built-in `all_to_all` and `linear` targets are synthetic research targets,
not copies of vendor hardware. Both currently use the portable basis
`RX`, `RY`, `RZ`, and `CNOT`.

For every requested MLAE Grover power, QFin:

1. records the original high-level PennyLane tape;
2. recursively decomposes it into the target basis;
3. routes two-qubit gates through the coupling graph with SWAP insertion;
4. decomposes inserted SWAPs into basis gates;
5. verifies every final two-qubit edge against the coupling graph; and
6. records the final logical-to-physical wire permutation.

The report includes high-level, decomposed, and routed gate/depth counts,
inserted SWAPs, one- and two-qubit gates, used coupling edges, objective-wire
placement, total shots, objective evaluations, and total executed gates.

```python
report = compiled.device_resources(
    schedule=(0, 1, 2, 4),
    shots=2_000,
    target="linear",
)
print(report.maximum_routed_depth)
print(report.circuits[-1].routing_swaps)
```

For VaR and CVaR, QFin profiles one representative structured objective and
multiplies circuit/shot/gate totals by the hybrid threshold and excess
evaluation count. The adaptive threshold sequence remains classical.

These are portable circuit-level estimates. They exclude pulse scheduling,
calibration, queue latency, control electronics, error correction, and
provider-specific transpiler optimizations.

## Synthetic noise and mitigation

`NoiseModel` makes two assumptions explicit:

- a local single-qubit depolarizing channel is inserted on every affected wire
  after each queued gate; and
- an optional bit-flip channel is inserted before measurement on each wire.

`compiled.noise_analysis(...)` executes the circuit with PennyLane
`default.mixed`. It applies global unitary folding at user-supplied scale
factors and polynomial zero-noise extrapolation (ZNE). The report retains the
ideal, unmitigated, raw extrapolated, and probability-clipped estimates plus
both absolute errors and whether mitigation improved this particular run.

```python
noise = qfin.NoiseModel(
    depolarizing_probability=0.001,
    readout_bit_flip_probability=0.002,
)
report = compiled.noise_analysis(
    noise,
    power=0,
    scale_factors=(1.0, 3.0, 5.0),
    extrapolation_order=1,
)
```

Analytic execution is deterministic. Supplying `shots` activates repeatable
finite-shot sampling through a seed. ZNE is an experiment, not a guarantee;
the report does not hide cases where extrapolation worsens the estimate or
leaves the probability interval before clipping.

## OpenQASM and Qiskit

`compiled.to_openqasm(...)` returns a `QasmExport` containing:

- an OpenQASM 2 program using the final routed basis circuit;
- the target and resource report;
- the final logical-to-physical wire map; and
- a SHA-256 digest for reproducibility.

Terminal measurements cover all wires. Consumers should use
`objective_physical_wire` to locate the routed objective result.

The optional Qiskit extra implements the former stub:

```bash
python -m pip install -e ".[quantum,qiskit]"
```

```python
quantum_circuit = compiled.to_qiskit(power=1, target="linear")
```

QFin exports through OpenQASM 2 and Qiskit's public parser. Tests compare the
routed Qiskit state-vector objective probability with the original
PennyLane-Lightning circuit. This is circuit export, not Qiskit runtime or IBM
hardware execution support.

`inspect_qiskit_backend(...)` reads BackendV2-style width, operation names,
coupling edges, measurement/reset/entangling primitives, and dynamic-circuit
signals. It does not authenticate, read calibration data, transpile for that
backend, or submit a job.

## Responsibility boundary

- QFin Python constructs finance objects, circuits, targets, noise assumptions,
  reports, and exports.
- QFin C++20 remains responsible only for financial and actuarial numerical
  kernels.
- PennyLane-Lightning remains responsible for compiled ideal quantum
  simulation.
- PennyLane `default.mixed` performs the explicit research noise simulation.
- Qiskit parses exported circuits when the optional extra is installed.

See [device-realism-performance.md](device-realism-performance.md) for the
measured development environment, timings, routing overhead, numerical parity,
and noise experiment.
