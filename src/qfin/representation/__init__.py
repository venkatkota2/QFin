"""Financial-distribution representation tools."""

from qfin.representation.arithmetic import (
    AffineOutputPlan,
    AffineTransformValidation,
    IntegerHingePlan,
    IntegerPolynomialPlan,
    IntegerQuadraticTerm,
    ReversibleAffineTransformPlan,
    StructuredLossOraclePlan,
    StructuredTailOracleValidation,
    compile_affine_transform,
    compile_structured_loss_oracle,
    validate_affine_transform,
    validate_structured_tail_oracle,
)
from qfin.representation.encoding import DistributionEncoding, encode, encode_quantiles
from qfin.representation.factorized import (
    FactorizedDistributionEncoding,
    LinearFactorTransform,
    MaterializedFactorGrid,
    encode_gaussian_factors,
    encode_independent_factors,
)
from qfin.representation.feasibility import (
    BlockEncodingFeasibility,
    analyze_block_encoding,
)
from qfin.representation.objectives import (
    QuantumObjectiveEncoding,
    cdf_objective,
    tail_excess_objective,
    tail_probability_objective,
)
from qfin.representation.strategies import (
    StatePreparationCost,
    StatePreparationStrategyReport,
    compare_state_preparation_strategies,
)

__all__ = [
    "AffineOutputPlan",
    "AffineTransformValidation",
    "BlockEncodingFeasibility",
    "DistributionEncoding",
    "FactorizedDistributionEncoding",
    "IntegerHingePlan",
    "IntegerPolynomialPlan",
    "IntegerQuadraticTerm",
    "LinearFactorTransform",
    "MaterializedFactorGrid",
    "QuantumObjectiveEncoding",
    "ReversibleAffineTransformPlan",
    "StatePreparationCost",
    "StatePreparationStrategyReport",
    "StructuredLossOraclePlan",
    "StructuredTailOracleValidation",
    "analyze_block_encoding",
    "cdf_objective",
    "compare_state_preparation_strategies",
    "compile_affine_transform",
    "compile_structured_loss_oracle",
    "encode",
    "encode_gaussian_factors",
    "encode_independent_factors",
    "encode_quantiles",
    "tail_excess_objective",
    "tail_probability_objective",
    "validate_affine_transform",
    "validate_structured_tail_oracle",
]
