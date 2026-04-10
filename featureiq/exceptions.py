"""Custom exceptions for the FeatureIQ framework."""


class FeatureIQError(Exception):
    """Base exception for all FeatureIQ errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ProfilingError(FeatureIQError):
    """Raised when data profiling fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class OntologyError(FeatureIQError):
    """Raised when ontology loading or rule application fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RuleValidationError(OntologyError):
    """Raised when a contributed rule fails validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RecommenderError(FeatureIQError):
    """Raised when the meta-learning recommender fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TransformerError(FeatureIQError):
    """Raised when pipeline construction or transformation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedAlgorithmError(FeatureIQError):
    """Raised when an unsupported algorithm is declared."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedProblemTypeError(FeatureIQError):
    """Raised when an unsupported problem type is declared."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InsufficientDataError(FeatureIQError):
    """Raised when dataset is too small for reliable profiling.

    Threshold: fewer than 50 rows raises this exception.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
