# Memory and allocation contract

Chunk sizes bound work processed together. Returned arrays and caller inputs still
exist in full. Neither a working-set estimate nor `tracemalloc` is a process RSS
limit. Public immutable models copy their input arrays and make their own copies
read-only; they do not change caller writeability. Native return arrays own their
storage and remain valid after inputs and intermediate C++ objects are destroyed.

| Path | Persistent storage | Working storage and control |
| --- | --- | --- |
| Bond batch | Flattened cashflows O(F), offsets and analytics O(B) | Several O(F) valuation buffers; no scenario cube. Schedule dimensions are checked before creation. |
| Rate/indexed scenarios | Input shocks O(SN), aggregate result O(S) | NumPy scenario blocks O(C max(F,N)); each main matrix is bounded toward 16 MiB. Multiple matrices coexist. `chunk_size` is an upper bound. A single row can exceed this soft matrix target. |
| Multi-period ALM | Nine aggregates O(S(T+1)); input paths O(STN) | Scenario chunks O(C(T+1)); no returned scenario-by-instrument cube. |
| Life projection | Policy buffers O(P), aggregate timeline O(T), point PV O(P) | Iteration over points/years; represented policy count does not expand grouped model points. |
| Life scenarios | Five aggregate vectors O(S), input paths O(STN) | `scenario_chunk_size` and `policy_chunk_size` control native working blocks. Reported working-set bytes are an estimate. |
| Structured validation | Marginal encodings plus occupied-code summaries | Joint points are streamed in `factor_validation_chunk_size` blocks; `max_factor_validation_points` guards exponential work. Code/arithmetic limits also apply. |
| MLAE likelihood | One-dimensional search/refinement arrays | O(G) in likelihood grid size, with the existing grid guard. It is not an amplitude-state simulation buffer. |
| PennyLane simulation | Device-managed quantum state | State memory can grow exponentially with wires. QFin's classical buffer checks do not cap a third-party simulator's RSS. |

Here B is bonds, F cashflows, S scenarios, N curve nodes, T periods, P model points,
C a chunk size and G the likelihood grid size.

Major output/schedule preflights and native allocations use a conservative
**512 MiB per guarded buffer group**. Dimension additions/products are checked
before `size_t`/NumPy-size conversion. Native groups also reject sizes beyond
`ptrdiff_t`. Requests exceeding these bounds raise `ResourceLimitError` before
allocation. Native `length_error` and `bad_alloc` map to the same exception.
Python-owned third-party allocations can still raise `MemoryError`; process OOM
termination cannot be converted into a Python exception. The 512 MiB guard is
not a total process budget or a promise that every smaller request fits RAM.

Finite input values can still produce an unrepresentable financial result. Bond,
life and ALM checks reject non-finite financial outputs explicitly. An infinite
funding ratio is intentional only for a zero liability denominator.

`python tools/memory_benchmark.py --output memory.json --repeats 3` starts a fresh
process for each public API workload/engine/chunk combination. It records absolute
process peak RSS, the pre-call high-water mark, repeated timings and dispersion.
Linux/BSD `ru_maxrss` is converted from KiB, macOS uses bytes, and Windows uses
`PeakWorkingSetSize`. These observations include imports, inputs and warm-up;
they must not be labelled pure kernel allocation. See the 1.1.2 hardening report
for the actual measurements and comparisons.
