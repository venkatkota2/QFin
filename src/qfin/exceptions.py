"""Public operational errors, preserving built-in catch compatibility in 1.x."""


class QFinError(Exception):
    """Base class for QFin operational errors."""


class QFinValidationError(QFinError, ValueError):
    """Invalid financial inputs; remains catchable as ValueError."""


class QFinTypeError(QFinError, TypeError):
    """Invalid input type; remains catchable as TypeError."""


class NumericalError(QFinError, ArithmeticError):
    """A valid calculation cannot be represented or numerically resolved."""


class PricingError(NumericalError, ValueError):
    """Pricing or yield inversion failed within its supported domain."""


class CalibrationError(QFinError, ValueError):
    """Calibration or independent repricing failed."""


class ScenarioError(QFinValidationError):
    """A scenario violates the supported financial domain."""


class CompilationError(QFinError):
    """Raised when a financial problem cannot be compiled."""


class BackendError(QFinError):
    """Backend execution or availability error."""


class BackendUnavailableError(BackendError):
    """Raised when an optional quantum backend is not installed."""


class ResourceLimitError(QFinError):
    """Rejected dimensions or allocation exceed an explicit safe limit."""


class NativeBackendUnavailableError(BackendUnavailableError):
    """Raised when native execution is explicitly requested but unavailable."""


class OptimizationError(QFinError):
    """Raised when a validated classical optimization problem cannot be solved."""
