"""Convert DatasetProfile into a flat numeric vector for the meta-learner.

The meta-feature vector has a fixed length of 24. If this composition changes,
all dependent modules (MetaLearner, openml_collector) must be updated.
"""

from __future__ import annotations

import numpy as np

from featureiq.exceptions import UnsupportedAlgorithmError
from featureiq.profiler.dataset_profiler import DatasetProfile
from featureiq.utils.validation import (
    ALGORITHM_FAMILY_MAP,
    AlgorithmFamily,
    ColumnType,
    ProblemType,
)

META_FEATURE_VECTOR_LENGTH = 26


def encode_problem_type(problem_type: ProblemType) -> list[int]:
    """One-hot encode problem type.

    Returns a list of 5 ints (one per ProblemType enum value),
    in alphabetical order of enum values.

    Args:
        problem_type: Declared problem type.

    Returns:
        One-hot encoded list of length 5.
    """
    sorted_members = sorted(ProblemType, key=lambda m: m.value)
    return [1 if m == problem_type else 0 for m in sorted_members]


def encode_algorithm_family(algorithm: str) -> list[int]:
    """One-hot encode algorithm family.

    Resolves algorithm name to AlgorithmFamily via ALGORITHM_FAMILY_MAP.
    Returns a list of 5 ints (one per AlgorithmFamily enum value),
    in alphabetical order of enum values.

    Args:
        algorithm: Algorithm name string (case-insensitive).

    Returns:
        One-hot encoded list of length 5.

    Raises:
        UnsupportedAlgorithmError: If algorithm not found in ALGORITHM_FAMILY_MAP.
    """
    key = algorithm.lower()
    if key not in ALGORITHM_FAMILY_MAP:
        raise UnsupportedAlgorithmError(
            f"Algorithm '{algorithm}' is not supported. "
            f"Supported algorithms: {list(ALGORITHM_FAMILY_MAP.keys())}"
        )

    family = ALGORITHM_FAMILY_MAP[key]
    sorted_members = sorted(AlgorithmFamily, key=lambda m: m.value)
    return [1 if m == family else 0 for m in sorted_members]


def build_meta_feature_vector(
    profile: DatasetProfile,
    problem_type: ProblemType,
    algorithm: str,
) -> np.ndarray:
    """Build a flat numeric meta-feature vector from a DatasetProfile.

    Vector composition (in this exact order):
    1. Dataset-level features: [n_rows, n_columns, n_numerical, n_categorical,
       n_datetime, overall_missing_rate, class_imbalance_ratio (0 if None),
       feature_to_row_ratio, has_temporal_structure (0 or 1)]
    2. Aggregate column stats: [mean_skewness, mean_missing_rate,
       fraction_high_cardinality, fraction_skewed, fraction_has_outliers,
       mean_correlation, max_vif]
    3. Problem type one-hot (5 values)
    4. Algorithm family one-hot (5 values)

    Total vector length: 9 + 7 + 5 + 5 = 26

    All None values must be replaced with 0.0 before returning.
    The vector must be a float64 numpy array.

    Depends on META_FEATURE_VECTOR_LENGTH constant (24). If the composition
    of this vector changes, MetaLearner and openml_collector must be updated.

    Args:
        profile: DatasetProfile from profile_dataset.
        problem_type: Declared problem type.
        algorithm: Algorithm name string.

    Returns:
        numpy array of shape (24,), dtype float64.
    """
    dataset_features = [
        float(profile.n_rows),
        float(profile.n_columns),
        float(profile.n_numerical),
        float(profile.n_categorical),
        float(profile.n_datetime),
        profile.overall_missing_rate,
        (
            profile.class_imbalance_ratio
            if profile.class_imbalance_ratio is not None
            else 0.0
        ),
        profile.feature_to_row_ratio,
        1.0 if profile.has_temporal_structure else 0.0,
    ]

    skewness_values: list[float] = []
    missing_rates: list[float] = []
    n_high_cardinality = 0
    n_skewed = 0
    n_has_outliers = 0
    n_cols = len(profile.column_profiles)

    for cp in profile.column_profiles.values():
        missing_rates.append(cp.missing_rate)
        if cp.is_high_cardinality:
            n_high_cardinality += 1
        if cp.column_type == ColumnType.NUMERICAL:
            if cp.skewness is not None:
                skewness_values.append(cp.skewness)
            if cp.is_skewed:
                n_skewed += 1
            if cp.outlier_fraction is not None and cp.outlier_fraction > 0:
                n_has_outliers += 1

    mean_skewness = float(np.mean(skewness_values)) if skewness_values else 0.0
    mean_missing_rate = float(np.mean(missing_rates)) if missing_rates else 0.0
    fraction_high_cardinality = n_high_cardinality / n_cols if n_cols > 0 else 0.0
    fraction_skewed = n_skewed / n_cols if n_cols > 0 else 0.0
    fraction_has_outliers = n_has_outliers / n_cols if n_cols > 0 else 0.0

    corr_values: list[float] = []
    if profile.correlation_matrix:
        cols = list(profile.correlation_matrix.keys())
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                corr_values.append(abs(profile.correlation_matrix[c1].get(c2, 0.0)))
    mean_correlation = float(np.mean(corr_values)) if corr_values else 0.0

    max_vif = 0.0
    if profile.vif_scores:
        finite_vifs = [v for v in profile.vif_scores.values() if np.isfinite(v)]
        max_vif = float(max(finite_vifs)) if finite_vifs else 0.0

    aggregate_stats = [
        mean_skewness,
        mean_missing_rate,
        fraction_high_cardinality,
        fraction_skewed,
        fraction_has_outliers,
        mean_correlation,
        max_vif,
    ]

    problem_type_enc = encode_problem_type(problem_type)
    algorithm_enc = encode_algorithm_family(algorithm)

    vector = dataset_features + aggregate_stats + problem_type_enc + algorithm_enc

    result = np.array(vector, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0)

    assert result.shape == (META_FEATURE_VECTOR_LENGTH,), (
        f"Meta-feature vector has wrong shape: {result.shape}, "
        f"expected ({META_FEATURE_VECTOR_LENGTH},)"
    )

    return result
