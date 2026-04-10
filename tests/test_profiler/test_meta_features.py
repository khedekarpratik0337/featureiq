"""Tests for the meta-features vector builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featureiq.exceptions import UnsupportedAlgorithmError
from featureiq.profiler.dataset_profiler import profile_dataset
from featureiq.profiler.meta_features import (
    META_FEATURE_VECTOR_LENGTH,
    build_meta_feature_vector,
    encode_algorithm_family,
    encode_problem_type,
)
from featureiq.utils.validation import ProblemType


class TestBuildMetaFeatureVector:
    """Tests for build_meta_feature_vector."""

    def test_output_shape(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        vec = build_meta_feature_vector(
            profile, ProblemType.BINARY_CLASSIFICATION, "xgboost"
        )
        assert vec.shape == (META_FEATURE_VECTOR_LENGTH,)

    def test_output_dtype(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        vec = build_meta_feature_vector(
            profile, ProblemType.BINARY_CLASSIFICATION, "xgboost"
        )
        assert vec.dtype == np.float64

    def test_no_nan_values(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        vec = build_meta_feature_vector(
            profile, ProblemType.BINARY_CLASSIFICATION, "xgboost"
        )
        assert not np.any(np.isnan(vec))

    def test_problem_type_encoding_consistency(self) -> None:
        enc1 = encode_problem_type(ProblemType.REGRESSION)
        enc2 = encode_problem_type(ProblemType.REGRESSION)
        assert enc1 == enc2

    def test_unknown_algorithm_raises(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        with pytest.raises(UnsupportedAlgorithmError):
            build_meta_feature_vector(
                profile, ProblemType.BINARY_CLASSIFICATION, "not_a_real_algo"
            )

    def test_algorithm_family_case_insensitive(self) -> None:
        enc1 = encode_algorithm_family("XGBoost")
        enc2 = encode_algorithm_family("xgboost")
        assert enc1 == enc2

    def test_vector_composition_known_values(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        vec = build_meta_feature_vector(
            profile, ProblemType.BINARY_CLASSIFICATION, "xgboost"
        )
        assert vec[0] == float(profile.n_rows)
        assert vec[1] == float(profile.n_columns)
        assert vec[2] == float(profile.n_numerical)
        assert vec[3] == float(profile.n_categorical)
        assert vec[4] == float(profile.n_datetime)
        assert vec[5] == profile.overall_missing_rate
        cir = (
            profile.class_imbalance_ratio
            if profile.class_imbalance_ratio is not None
            else 0.0
        )
        assert vec[6] == pytest.approx(cir)
        assert vec[7] == pytest.approx(profile.feature_to_row_ratio)
        assert vec[8] == (1.0 if profile.has_temporal_structure else 0.0)

    def test_one_hot_problem_type_ordering(self) -> None:
        sorted_members = sorted(ProblemType, key=lambda m: m.value)
        enc = encode_problem_type(ProblemType.BINARY_CLASSIFICATION)
        idx = sorted_members.index(ProblemType.BINARY_CLASSIFICATION)
        assert enc[idx] == 1
        assert sum(enc) == 1

    def test_one_hot_algorithm_family_sums_to_one(self) -> None:
        enc = encode_algorithm_family("xgboost")
        assert sum(enc) == 1

    def test_assertion_shape_passes_for_valid_input(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        vec = build_meta_feature_vector(
            profile, ProblemType.BINARY_CLASSIFICATION, "xgboost"
        )
        assert vec.shape == (META_FEATURE_VECTOR_LENGTH,)
