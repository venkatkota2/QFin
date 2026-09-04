"""Classical benchmarks and numerical validation."""

from qfin.validation.classical import black_scholes_price, put_call_parity_residual
from qfin.validation.financial import (
    GOLDEN_BOND_CASES,
    FinancialTolerance,
    FinancialValidationCheck,
    FinancialValidationError,
    FinancialValidationReport,
    GoldenBondCase,
    ReferenceBondAnalytics,
    quantlib_available,
    quantlib_bond_from_yield,
    quantlib_bond_schedule,
    reference_bond_from_yield,
    validate_financial_values,
)

__all__ = [
    "GOLDEN_BOND_CASES",
    "FinancialTolerance",
    "FinancialValidationCheck",
    "FinancialValidationError",
    "FinancialValidationReport",
    "GoldenBondCase",
    "ReferenceBondAnalytics",
    "black_scholes_price",
    "put_call_parity_residual",
    "quantlib_available",
    "quantlib_bond_from_yield",
    "quantlib_bond_schedule",
    "reference_bond_from_yield",
    "validate_financial_values",
]
