"""Run the complete QFin European-call MVP."""

import qfin

market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
option = qfin.EuropeanCall(strike=105, maturity=1.0)
compiled = qfin.compile(option, market, target_error=0.10, max_qubits=8)

print(compiled.explain())
result = compiled.run(shots=2_000, schedule=(0, 1, 2, 4), seed=7)
print(result.to_dict())
