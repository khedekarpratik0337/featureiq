"""Constants, enums, and controlled vocabularies for FeatureIQ."""

from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    """Supported machine learning problem types."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"


class AlgorithmFamily(str, Enum):
    """Supported algorithm families."""

    TREE_BASED = "tree_based"
    LINEAR_MODEL = "linear_model"
    DISTANCE_BASED = "distance_based"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


ALGORITHM_FAMILY_MAP: dict[str, AlgorithmFamily] = {
    "xgboost": AlgorithmFamily.TREE_BASED,
    "lightgbm": AlgorithmFamily.TREE_BASED,
    "random_forest": AlgorithmFamily.TREE_BASED,
    "decision_tree": AlgorithmFamily.TREE_BASED,
    "gradient_boosting": AlgorithmFamily.TREE_BASED,
    "logistic_regression": AlgorithmFamily.LINEAR_MODEL,
    "linear_regression": AlgorithmFamily.LINEAR_MODEL,
    "ridge": AlgorithmFamily.LINEAR_MODEL,
    "lasso": AlgorithmFamily.LINEAR_MODEL,
    "svm": AlgorithmFamily.DISTANCE_BASED,
    "knn": AlgorithmFamily.DISTANCE_BASED,
    "mlp": AlgorithmFamily.NEURAL_NETWORK,
    "adaboost": AlgorithmFamily.ENSEMBLE,
    "bagging": AlgorithmFamily.ENSEMBLE,
    # Time-series forecasting algorithms
    "prophet": AlgorithmFamily.LINEAR_MODEL,
    "arima": AlgorithmFamily.LINEAR_MODEL,
    "ets": AlgorithmFamily.LINEAR_MODEL,
    "theta": AlgorithmFamily.LINEAR_MODEL,
    "tbats": AlgorithmFamily.LINEAR_MODEL,
    "temporal_fusion_transformer": AlgorithmFamily.NEURAL_NETWORK,
    "nbeats": AlgorithmFamily.NEURAL_NETWORK,
    "deepar": AlgorithmFamily.NEURAL_NETWORK,
    # Anomaly detection algorithms
    "isolation_forest": AlgorithmFamily.TREE_BASED,
    "one_class_svm": AlgorithmFamily.DISTANCE_BASED,
    "lof": AlgorithmFamily.DISTANCE_BASED,
    "dbscan": AlgorithmFamily.DISTANCE_BASED,
    "autoencoder": AlgorithmFamily.NEURAL_NETWORK,
    "elliptic_envelope": AlgorithmFamily.DISTANCE_BASED,
}


class ColumnType(str, Enum):
    """Detected column semantic types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


MIN_ROWS_FOR_PROFILING: int = 50
MIN_ROWS_FOR_META_LEARNING: int = 100
HIGH_CARDINALITY_THRESHOLD: float = 0.5
LOW_CARDINALITY_MAX: int = 10
MEDIUM_CARDINALITY_MAX: int = 50
SKEWNESS_THRESHOLD: float = 1.0
MISSING_THRESHOLD: float = 0.05
OUTLIER_IQR_FACTOR: float = 1.5
