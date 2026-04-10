"""Tests for the column profiler module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from featureiq.exceptions import ProfilingError
from featureiq.profiler.column_profiler import (
    ColumnProfile,
    detect_column_type,
    profile_categorical_column,
    profile_column,
    profile_datetime_column,
    profile_numerical_column,
)
from featureiq.utils.validation import ColumnType


class TestDetectColumnType:
    """Tests for detect_column_type."""

    def test_int_series(self, int_series: pd.Series) -> None:
        assert detect_column_type(int_series) == ColumnType.NUMERICAL

    def test_float_series(self, float_series: pd.Series) -> None:
        assert detect_column_type(float_series) == ColumnType.NUMERICAL

    def test_object_series(self, object_series: pd.Series) -> None:
        assert detect_column_type(object_series) == ColumnType.CATEGORICAL

    def test_bool_series(self, bool_series: pd.Series) -> None:
        assert detect_column_type(bool_series) == ColumnType.BOOLEAN

    def test_datetime_series(self, datetime_series: pd.Series) -> None:
        assert detect_column_type(datetime_series) == ColumnType.DATETIME

    def test_mixed_series(self) -> None:
        mixed = pd.Series([1, "a", 3.14, None], name="mixed")
        result = detect_column_type(mixed)
        assert result in (ColumnType.CATEGORICAL, ColumnType.UNKNOWN)

    def test_all_nan_returns_unknown(self) -> None:
        s = pd.Series([np.nan, np.nan, np.nan], name="allnan")
        assert detect_column_type(s) == ColumnType.UNKNOWN

    def test_object_series_parseable_as_datetime(self) -> None:
        s = pd.Series(["2020-01-01", "2020-02-01", "2020-03-01"], name="date_str")
        assert detect_column_type(s) == ColumnType.DATETIME

    def test_timedelta_dtype_returns_unknown(self) -> None:
        s = pd.Series(
            [pd.Timedelta("1 day"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            name="td_col",
        )
        assert detect_column_type(s) == ColumnType.UNKNOWN


class TestProfileNumericalColumn:
    """Tests for profile_numerical_column."""

    def test_normal_distribution(self, float_series: pd.Series) -> None:
        result = profile_numerical_column(float_series)
        assert "mean" in result
        assert "std" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert isinstance(result["is_skewed"], bool)
        assert isinstance(result["outlier_fraction"], float)

    def test_skewed_distribution(self, skewed_series: pd.Series) -> None:
        result = profile_numerical_column(skewed_series)
        assert result["is_skewed"] is True

    def test_series_with_outliers(self, series_with_outliers: pd.Series) -> None:
        result = profile_numerical_column(series_with_outliers)
        assert result["outlier_fraction"] > 0

    def test_series_with_nulls(self, series_with_nulls: pd.Series) -> None:
        result = profile_numerical_column(series_with_nulls)
        assert "mean" in result

    def test_fewer_than_3_non_null(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan], name="tiny")
        with pytest.raises(ProfilingError, match="fewer than 3"):
            profile_numerical_column(s)


class TestProfileCategoricalColumn:
    """Tests for profile_categorical_column."""

    def test_low_cardinality(self, low_cardinality_series: pd.Series) -> None:
        result = profile_categorical_column(low_cardinality_series)
        assert 0.0 < result["top_frequency"] <= 1.0

    def test_high_cardinality(self, high_cardinality_series: pd.Series) -> None:
        result = profile_categorical_column(high_cardinality_series)
        assert result["top_frequency"] == pytest.approx(0.01, abs=0.001)

    def test_single_value(self, single_value_series: pd.Series) -> None:
        result = profile_categorical_column(single_value_series)
        assert result["top_frequency"] == 1.0

    def test_all_null_categorical(self) -> None:
        s = pd.Series([np.nan, np.nan, np.nan], dtype=object, name="null_cat")
        result = profile_categorical_column(s)
        assert result["top_frequency"] == 0.0


class TestProfileColumn:
    """Integration tests for profile_column."""

    def test_numerical_column(self, float_series: pd.Series) -> None:
        profile = profile_column(float_series, "float_col")
        assert isinstance(profile, ColumnProfile)
        assert profile.column_type == ColumnType.NUMERICAL
        assert profile.mean is not None

    def test_categorical_column(self, object_series: pd.Series) -> None:
        profile = profile_column(object_series, "cat_col")
        assert profile.column_type == ColumnType.CATEGORICAL
        assert profile.top_frequency is not None

    def test_datetime_column(self, datetime_series: pd.Series) -> None:
        profile = profile_column(datetime_series, "date_col")
        assert profile.column_type == ColumnType.DATETIME
        assert profile.is_monotonic is not None

    def test_boolean_column(self, bool_series: pd.Series) -> None:
        profile = profile_column(bool_series, "bool_col")
        assert profile.column_type == ColumnType.BOOLEAN

    def test_empty_series_raises(self, empty_series: pd.Series) -> None:
        with pytest.raises(ProfilingError, match="empty"):
            profile_column(empty_series, "empty")

    def test_all_null_series_raises(self, all_null_series: pd.Series) -> None:
        with pytest.raises(ProfilingError, match="null"):
            profile_column(all_null_series, "all_null")

    def test_generic_exception_catch_all(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0] * 20, name="num_col")
        with patch(
            "featureiq.profiler.column_profiler.profile_numerical_column",
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(ProfilingError, match="Failed to profile"):
                profile_column(s, "num_col")

    def test_profiling_error_reraise_via_profile_column(self) -> None:
        s = pd.Series([1.0, 2.0] + [np.nan] * 48, name="sparse_num")
        with pytest.raises(ProfilingError, match="fewer than 3"):
            profile_column(s, "sparse_num")


class TestProfileDatetimeColumn:
    """Tests for profile_datetime_column edge cases."""

    def test_single_datetime_value(self) -> None:
        s = pd.Series(pd.to_datetime(["2020-01-01"]), name="single_dt")
        result = profile_datetime_column(s)
        assert result["is_monotonic"] is True
        assert result["has_regular_frequency"] is True

    def test_two_identical_datetime_values(self) -> None:
        s = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-01"]), name="dup_dt")
        result = profile_datetime_column(s)
        assert "is_monotonic" in result
        assert "has_regular_frequency" in result

    def test_normal_datetime_series(self) -> None:
        s = pd.Series(
            pd.date_range("2020-01-01", periods=10, freq="D"), name="normal_dt"
        )
        result = profile_datetime_column(s)
        assert result["is_monotonic"] is True
        assert result["has_regular_frequency"] is True


class TestProfileNumericalEdgeCases:
    """Additional edge-case tests for numerical profiling."""

    def test_is_normal_none_for_large_series(self) -> None:
        rng = np.random.RandomState(42)
        s = pd.Series(rng.randn(5001), name="large")
        result = profile_numerical_column(s)
        assert result["is_normal"] is None

    def test_skewness_boundary_not_skewed(self) -> None:
        rng = np.random.RandomState(42)
        s = pd.Series(rng.randn(1000), name="normal_ish")
        result = profile_numerical_column(s)
        assert result["is_skewed"] == (abs(result["skewness"]) > 1.0)

    def test_outlier_fraction_computed_correctly(self) -> None:
        data = list(range(100)) + [10000, -10000]
        s = pd.Series(data, name="outlier_test", dtype=float)
        result = profile_numerical_column(s)
        assert result["outlier_fraction"] > 0
