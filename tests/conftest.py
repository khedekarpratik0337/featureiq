"""Shared pytest fixtures for FeatureIQ tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def int_series() -> pd.Series:
    return pd.Series(range(100), name="int_col")


@pytest.fixture
def float_series() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.randn(200), name="float_col")


@pytest.fixture
def object_series() -> pd.Series:
    return pd.Series(["a", "b", "c", "a", "b"] * 20, name="cat_col")


@pytest.fixture
def bool_series() -> pd.Series:
    return pd.Series([True, False, True, True, False] * 20, name="bool_col")


@pytest.fixture
def datetime_series() -> pd.Series:
    return pd.Series(
        pd.date_range("2020-01-01", periods=100, freq="D"), name="date_col"
    )


@pytest.fixture
def skewed_series() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.exponential(scale=2.0, size=200), name="skewed_col")


@pytest.fixture
def series_with_outliers() -> pd.Series:
    rng = np.random.RandomState(42)
    data = rng.randn(200)
    data[0] = 100.0
    data[1] = -100.0
    return pd.Series(data, name="outlier_col")


@pytest.fixture
def series_with_nulls() -> pd.Series:
    rng = np.random.RandomState(42)
    data = rng.randn(200).tolist()
    for i in range(0, 200, 5):
        data[i] = np.nan
    return pd.Series(data, name="null_col")


@pytest.fixture
def low_cardinality_series() -> pd.Series:
    return pd.Series(["red", "green", "blue"] * 50, name="low_card")


@pytest.fixture
def high_cardinality_series() -> pd.Series:
    return pd.Series([f"val_{i}" for i in range(100)], name="high_card")


@pytest.fixture
def single_value_series() -> pd.Series:
    return pd.Series(["same"] * 100, name="single_val")


@pytest.fixture
def empty_series() -> pd.Series:
    return pd.Series([], dtype=float, name="empty")


@pytest.fixture
def all_null_series() -> pd.Series:
    return pd.Series([np.nan] * 50, name="all_null")


@pytest.fixture
def mixed_dataframe() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randint(0, 100, n).astype(float),
            "cat1": np.random.choice(["a", "b", "c"], n),
            "cat2": np.random.choice(["x", "y"], n),
            "date1": pd.date_range("2020-01-01", periods=n, freq="D"),
            "bool1": np.random.choice([True, False], n),
        }
    )


@pytest.fixture
def target_binary() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1], size=100), name="target")


@pytest.fixture
def target_multiclass() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1, 2], size=100), name="target")


@pytest.fixture
def target_regression() -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(rng.randn(100), name="target")
