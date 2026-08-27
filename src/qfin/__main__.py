"""Command-line entry point for the European-option MVP."""

import argparse
import json
from collections.abc import Sequence

from qfin.compiler import compile
from qfin.finance import BlackScholes, EuropeanCall, EuropeanPut


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qfin",
        description="Compile and run a European-option quantum pricing model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    price = subparsers.add_parser("price", help="price a European call or put")
    price.add_argument("--kind", choices=("call", "put"), default="call")
    price.add_argument("--spot", type=float, required=True)
    price.add_argument("--strike", type=float, required=True)
    price.add_argument("--maturity", type=float, required=True)
    price.add_argument("--rate", type=float, required=True)
    price.add_argument("--volatility", type=float, required=True)
    price.add_argument("--dividend-yield", type=float, default=0.0)
    price.add_argument("--target-error", type=float, default=0.10)
    price.add_argument("--min-qubits", type=int, default=3)
    price.add_argument("--max-qubits", type=int, default=8)
    price.add_argument(
        "--representation",
        choices=("quantile", "probability"),
        default="quantile",
        help="quantile enables the v0.3 parameter-free distribution loader",
    )
    price.add_argument("--payoff-angle-tolerance", type=float, default=0.1)
    price.add_argument("--payoff-max-terms", type=int, default=None)
    price.add_argument("--shots", type=int, default=2_000)
    price.add_argument("--schedule", type=int, nargs="+", default=[0, 1, 2, 4])
    price.add_argument("--seed", type=int, default=None)
    price.add_argument(
        "--device",
        default="lightning.qubit",
        help="PennyLane device (default: lightning.qubit)",
    )
    price.add_argument(
        "--circuit-backend",
        choices=("compressed", "structured", "dense"),
        default=None,
        help="default: compressed for quantile encoding, otherwise structured",
    )
    price.add_argument(
        "--compile-only",
        action="store_true",
        help="show compiler decisions without executing PennyLane",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    market = BlackScholes(
        spot=args.spot,
        rate=args.rate,
        volatility=args.volatility,
        dividend_yield=args.dividend_yield,
    )
    option_type = EuropeanCall if args.kind == "call" else EuropeanPut
    option = option_type(strike=args.strike, maturity=args.maturity)
    model = compile(
        option,
        market,
        target_error=args.target_error,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        representation_method=args.representation,
        payoff_angle_tolerance=args.payoff_angle_tolerance,
        payoff_max_terms=args.payoff_max_terms,
    )
    if args.compile_only:
        print(model.explain())
        return 0
    result = model.run(
        shots=args.shots,
        schedule=tuple(args.schedule),
        seed=args.seed,
        backend_mode=args.circuit_backend,
        device_name=args.device,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
