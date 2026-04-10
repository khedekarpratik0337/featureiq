"""Tests for validation constants/enums and dataset profiler."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from featureiq.exceptions import InsufficientDataError, ProfilingError
from featureiq.profiler.dataset_profiler import (
    DatasetProfile,
    compute_class_imbalance,
    profile_dataset,
)
from featureiq.utils.validation import (
    ALGORITHM_FAMILY_MAP,
    HIGH_CARDINALITY_THRESHOLD,
    MIN_ROWS_FOR_META_LEARNING,
    MIN_ROWS_FOR_PROFILING,
    MISSING_THRESHOLD,
    OUTLIER_IQR_FACTOR,
    SKEWNESS_THRESHOLD,
    AlgorithmFamily,
    ProblemType,
)


class TestValidationEnums:
    """Tests for ProblemType and AlgorithmFamily enums."""

    def test_problem_type_values_are_strings(self) -> None:
        for member in ProblemType:
            assert isinstance(member.value, str)

    def test_algorithm_family_map_covers_all_families(self) -> None:
        families_in_map = set(ALGORITHM_FAMILY_MAP.values())
        all_families = set(AlgorithmFamily)
        assert families_in_map == all_families

    def test_threshold_constants_are_positive(self) -> None:
        assert MIN_ROWS_FOR_PROFILING > 0
        assert MIN_ROWS_FOR_META_LEARNING > 0
        assert HIGH_CARDINALITY_THRESHOLD > 0
        assert SKEWNESS_THRESHOLD > 0
        assert MISSING_THRESHOLD > 0
        assert OUTLIER_IQR_FACTOR > 0


class TestComputeClassImbalance:
    """Tests for compute_class_imbalance."""

    def test_balanced_classes(self) -> None:
        y = pd.Series([0, 1] * 50)
        ratio = compute_class_imbalance(y, ProblemType.BINARY_CLASSIFICATION)
        assert ratio == pytest.approx(1.0)

    def test_imbalanced_90_10(self) -> None:
        y = pd.Series([0] * 90 + [1] * 10)
        ratio = compute_class_imbalance(y, ProblemType.BINARY_CLASSIFICATION)
        assert ratio == pytest.approx(10 / 90, abs=0.01)

    def test_regression_returns_none(self, target_regression: pd.Series) -> None:
        ratio = compute_class_imbalance(target_regression, ProblemType.REGRESSION)
        assert ratio is None

    def test_single_class_returns_one(self) -> None:
        y = pd.Series([1] * 100)
        ratio = compute_class_imbalance(y, ProblemType.BINARY_CLASSIFICATION)
        assert ratio == 1.0

    def test_all_zero_counts(self) -> None:
        y = pd.Series([], dtype=int)
        ratio = compute_class_imbalance(y, ProblemType.BINARY_CLASSIFICATION)
        assert ratio == 1.0

    def test_exact_imbalance_ratio(self) -> None:
        y = pd.Series([0] * 80 + [1] * 20)
        ratio = compute_class_imbalance(y, ProblemType.BINARY_CLASSIFICATION)
        assert ratio == pytest.approx(20 / 80)


class TestProfileDataset:
    """Tests for profile_dataset."""

    def test_mixed_type_dataframe(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        assert isinstance(profile, DatasetProfile)
        assert profile.n_rows == 100
        assert profile.n_columns == 6
        assert profile.n_numerical >= 2
        assert profile.n_categorical >= 1
        assert profile.n_datetime >= 1

    def test_type_counts_correct(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        total = (
            profile.n_numerical
            + profile.n_categorical
            + profile.n_datetime
            + profile.n_boolean
            + profile.n_text
            + profile.n_unknown
        )
        assert total == profile.n_columns

    def test_insufficient_data_raises(self) -> None:
        small_df = pd.DataFrame({"a": range(10)})
        small_y = pd.Series(range(10))
        with pytest.raises(InsufficientDataError):
            profile_dataset(small_df, small_y, ProblemType.REGRESSION)

    def test_generic_exception_in_column_profiling(self) -> None:
        df = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        with patch(
            "featureiq.profiler.dataset_profiler.profile_column",
            side_effect=RuntimeError("unexpected column error"),
        ):
            with pytest.raises(ProfilingError, match="Failed to profile"):
                profile_dataset(df, y, ProblemType.REGRESSION)

    def test_profiling_error_reraise_through_dataset(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0] + [np.nan] * 48})
        y = pd.Series(range(50))
        with pytest.raises(ProfilingError, match="fewer than 3"):
            profile_dataset(df, y, ProblemType.REGRESSION)
