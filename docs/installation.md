# Install and use QFin

QFin's distribution name is **qfin-quantum** and its Python import is **qfin**.
The tested interpreter range is CPython 3.11–3.13. Core calculations need only
NumPy and SciPy; pip installs both. Quantum simulation, Qiskit export and QuantLib
validation are optional. The intended quantum workflow runs locally with
PennyLane/Lightning simulation; no physical quantum computer or hardware account
is needed.

## Install directly from GitHub

Create an environment and activate it:

```bash
python -m venv .venv
```

| Shell | Activation |
| --- | --- |
| macOS/Linux bash or zsh | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Then install the current repository version (Git must be installed):

```bash
python -m pip install "qfin-quantum[quantum] @ git+https://github.com/venkatkota2/QFin.git"
python -m pip check
python -m qfin --version
```

This installs the simulator dependencies. Omit `[quantum]` if you need only the
classical utilities. For a repeatable environment, replace the URL's trailing `.git` with
`.git@<full-commit-sha>`, using an actual tested commit. Record `pip freeze` with
your analysis. This project has not been published to PyPI; do not assume that
`pip install qfin` or `pip install qfin-quantum` retrieves this repository.

Alternatively, download the repository's source ZIP, extract it, open a terminal
in the directory containing `pyproject.toml`, and run `python -m pip install .`.
This path does not require Git. An editable development install uses
`python -m pip install -e ".[dev]"`.

## Compiler-free installation

A source install first tries to compile the optional C++20 extension. If this
fails, the build backend retries as a pure-Python package using the existing
NumPy implementations. To skip the compilation attempt entirely:

```bash
python -m pip install -Cwheel.cmake=false "qfin-quantum[quantum] @ git+https://github.com/venkatkota2/QFin.git"
```

From a checkout, the equivalent command is:

```bash
python -m pip install . -Cwheel.cmake=false
```

The financial API and model conventions are the same. Automatic execution uses
the existing NumPy fallback; explicitly requesting `engine="native"` without the
extension raises `NativeBackendUnavailableError`. Some large workloads may be
slower without C++. Check the installed capability:

```python
import qfin
print(qfin.__version__)
print(qfin.system_info()["native_extension"])
```

This uses the build backend's documented
[pure-Python fallback](https://scikit-build-core.readthedocs.io/en/stable/configuration/overrides.html#failed-bool).

## Native acceleration

Source builds use a C++20 compiler: GCC/Clang on Linux, Apple Clang on macOS, or
Visual Studio C++ Build Tools on Windows. The build backend manages CMake and
pybind11. All native CI jobs require the extension so a failed native build cannot
silently become a successful pure-Python test.

Set `QFIN_REQUIRE_NATIVE=1` before installing to make a native build failure fatal:

```bash
# macOS/Linux
QFIN_REQUIRE_NATIVE=1 python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

```powershell
# Windows PowerShell, from a checkout
$env:QFIN_REQUIRE_NATIVE = "1"
python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

Use these rebuild commands after a core install has supplied NumPy/SciPy. Omit
`--no-deps` for a fresh environment. Do not combine the native requirement with
`-Cwheel.cmake=false`, which explicitly disables native compilation.

The portable binary matrix covers CPython 3.11–3.13 on Linux x86_64, macOS arm64
and Windows AMD64. A binary wheel contains QFin's C++ extension; a
`py3-none-any.whl` contains the compiler-free package. Install a downloaded wheel
with `python -m pip install /path/to/the.whl`. Keep the native wheel matching your
Python version and platform. The universal wheel is suitable for the supported
Python versions wherever compatible NumPy/SciPy distributions are available.

## Optional dependencies

From a checkout:

| Command | Purpose |
| --- | --- |
| `python -m pip install ".[quantum]"` | PennyLane and Lightning simulation |
| `python -m pip install ".[qiskit]"` | Existing Qiskit interoperability |
| `python -m pip install ".[validation]"` | QuantLib comparisons |
| `python -m pip install -e ".[dev]"` | Tests, type checks, lint and build tooling |

For direct GitHub installs, put extras after the distribution name, for example:

```bash
python -m pip install "qfin-quantum[quantum,qiskit] @ git+https://github.com/venkatkota2/QFin.git"
```

Append `-Cwheel.cmake=false` when you also want to skip QFin's native build.
Quantum simulation still uses the installed PennyLane/Lightning packages.

## First simulated quantum calculation

This uses the existing European-option workflow with the local Lightning
simulator. The compiled object carries the selected quantum representation,
algorithm and accuracy/resource metadata.

```python
import qfin

market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
option = qfin.EuropeanCall(strike=105, maturity=1.0)
program = qfin.compile(option, market, target_error=0.10, max_qubits=8)
print(program.explain())
result = program.run(
    shots=2_000, schedule=(0, 1, 2, 4), seed=7,
    device_name="lightning.qubit",
)
print(result.value, result.confidence_interval_95, result.backend)
```

`program.to_pennylane(device_name="lightning.qubit").draw(power=0)` displays the
logical circuit. Quantum simulation uses classical computing resources and is
subject to the existing qubit/memory limits and approximation errors.

## Classical finance utilities

Save this anywhere outside the repository and run it with your environment's
Python:

```python
import qfin

curve = qfin.YieldCurve([0, 1, 5], [0.03, 0.03, 0.03])
bond = qfin.FixedRateBond(maturity=5, coupon_rate=0.04)
prices = qfin.price_bonds([bond], curve)
print("Dirty price:", prices.dirty_prices[0])

losses = qfin.LossDistribution([0, 10, 30, 100])
risk = qfin.compile(qfin.CVaR(losses, confidence=0.75), backend="classical").run()
print("VaR:", risk.var, "CVaR:", risk.cvar)
```

The existing CLI can inspect an option without installing quantum extras:

```bash
python -m qfin price --spot 100 --strike 105 --maturity 1 --rate 0.04 --volatility 0.2 --compile-only
```

Remove `--compile-only` after installing `[quantum]` to execute the simulation.
The Python API exposes the rest of the library.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No module named qfin` | Activate the intended environment and use its `python -m pip`; check `python -m pip show qfin-quantum`. |
| Compiler not found | Use `-Cwheel.cmake=false`, or install a C++20 compiler and rebuild. Check that an inherited `CXX` points to a real compiler. |
| `native_extension` is `False` | A pure package is installed or the extension could not load. Use `engine="numpy"` or rebuild/install a compatible native wheel. |
| `BackendUnavailableError` in a quantum call | Install the `[quantum]` extra into the same environment. |
| Notebook cannot find the package | Select the environment's kernel or install with that notebook's `%pip`. Restart its kernel after installation. |
| `qfin` command not found | Use `python -m qfin` or activate the environment so its scripts directory is on PATH. |

See [release engineering](release-engineering.md) for the exact installed-package
checks and [limitations](limitations.md) for the existing model boundaries.
