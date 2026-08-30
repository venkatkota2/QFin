"""QFin-specific exception types."""


class QFinError(Exception):
    """Base class for QFin errors."""


class CompilationError(QFinError):
    """Raised when a financial problem cannot be compiled."""


class BackendUnavailableError(QFinError):
    """Raised when an optional quantum backend is not installed."""


class ResourceLimitError(QFinError):
    """Raised before a simulator allocation would exceed a safe MVP limit."""


class NativeBackendUnavailableError(QFinError):
    """Raised when native execution is explicitly requested but unavailable."""


class OptimizationError(QFinError):
    """Raised when a validated classical optimization problem cannot be solved."""
