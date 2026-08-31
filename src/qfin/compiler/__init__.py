"""Financial-to-quantum compilation API."""

from qfin.compiler.capabilities import ProblemCapabilities, problem_capabilities
from qfin.compiler.compile import compile
from qfin.compiler.factorized_models import (
    CompiledFactorTailModel,
    FactorQuantumTailResult,
    StructuredOracleErrorBudget,
)
from qfin.compiler.factorized_risk_models import (
    CompiledFactorRiskModel,
    FactorQuantumObjectiveEstimate,
    FactorQuantumRiskResult,
    FactorQuantumVaRSearch,
    StructuredRiskErrorBudget,
)
from qfin.compiler.models import (
    CompiledPricingModel,
    ErrorBudget,
    PricingResult,
)
from qfin.compiler.optimization_models import CompiledOptimizationModel
from qfin.compiler.risk_models import (
    CompiledRiskModel,
    QuantumRiskResult,
    QuantumThresholdEstimate,
    QuantumVaRSearch,
    RiskErrorBudget,
)

__all__ = [
    "CompiledFactorRiskModel",
    "CompiledFactorTailModel",
    "CompiledOptimizationModel",
    "CompiledPricingModel",
    "CompiledRiskModel",
    "ErrorBudget",
    "FactorQuantumObjectiveEstimate",
    "FactorQuantumRiskResult",
    "FactorQuantumTailResult",
    "FactorQuantumVaRSearch",
    "PricingResult",
    "ProblemCapabilities",
    "QuantumRiskResult",
    "QuantumThresholdEstimate",
    "QuantumVaRSearch",
    "RiskErrorBudget",
    "StructuredOracleErrorBudget",
    "StructuredRiskErrorBudget",
    "compile",
    "problem_capabilities",
]
