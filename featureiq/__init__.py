"""FeatureIQ: Context-aware automated feature engineering."""

from featureiq.core import FeatureIQ
from featureiq.exceptions import (
    FeatureIQError,
    InsufficientDataError,
    OntologyError,
    ProfilingError,
    RecommenderError,
    TransformerError,
    UnsupportedAlgorithmError,
    UnsupportedProblemTypeError,
)
from featureiq.utils.validation import AlgorithmFamily, ProblemType

__version__ = "0.1.0"
__all__ = [
    "FeatureIQ",
    "ProblemType",
    "AlgorithmFamily",
    "FeatureIQError",
    "ProfilingError",
    "OntologyError",
    "RecommenderError",
    "TransformerError",
    "UnsupportedAlgorithmError",
    "UnsupportedProblemTypeError",
    "InsufficientDataError",
]
