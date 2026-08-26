"""QFin public API."""

from qfin.circuits import (
    PayoffRotation,
    ProbabilityTreePreparation,
    UniformQuantilePreparation,
    WalshPayoffApproximation,
    WalshTerm,
)
from qfin.compiler import CompiledPricingModel, ErrorBudget, PricingResult, compile
from qfin.exceptions import (
    BackendUnavailableError,
    CompilationError,
    QFinError,
    ResourceLimitError,
)
from qfin.finance import (
    BlackScholes,
    EmpiricalDistribution,
    EuropeanCall,
    EuropeanPut,
    GeometricBrownianMotion,
    LogNormal,
    Normal,
)
from qfin.representation import DistributionEncoding, encode, encode_quantiles
from qfin.validation import black_scholes_price

__version__ = "0.3.0"

__all__ = [
    "BackendUnavailableError",
    "BlackScholes",
    "CompilationError",
    "CompiledPricingModel",
    "DistributionEncoding",
    "EmpiricalDistribution",
    "ErrorBudget",
    "EuropeanCall",
    "EuropeanPut",
    "GeometricBrownianMotion",
    "LogNormal",
    "Normal",
    "PayoffRotation",
    "PricingResult",
    "ProbabilityTreePreparation",
    "QFinError",
    "ResourceLimitError",
    "UniformQuantilePreparation",
    "WalshPayoffApproximation",
    "WalshTerm",
    "__version__",
    "black_scholes_price",
    "compile",
    "encode",
    "encode_quantiles",
]
