"""Tests for Differencing, Fourier, TimeSinceReference transformers and registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featureiq.transformer.registry import (
    DifferencingTransformer,
    FourierFeaturesTransformer,
    TimeSinceReferenceTransformer,
    get_transformer,
)


class TestDifferencingTransformer:
    """Tests for DifferencingTransformer."""

    def test_first_order_differencing(self) -> None:
        df = pd.DataFrame({"val": list(range(1, 21))})
        t = DifferencingTransformer(variables=["val"])
        t.fit(df)
        result = t.transform(df)
        assert np.isnan(result["val"].iloc[0])
        assert (result["val"].iloc[1:] == 1.0).all()

    def test_preserves_non_target_columns(self) -> None:
        df = pd.DataFrame(
            {
                "val": list(range(1, 21)),
                "other": list(range(100, 120)),
            }
        )
        t = DifferencingTransformer(variables=["val"])
        t.fit(df)
        result = t.transform(df)
        pd.testing.assert_series_equal(result["other"], df["other"])


class TestFourierFeaturesTransformer:
    """Tests for FourierFeaturesTransformer."""

    def test_creates_sin_cos_columns(self) -> None:
        df = pd.DataFrame({"val": list(range(20))})
        t = FourierFeaturesTransformer(variables=["val"], period=7, n_harmonics=2)
        t.fit(df)
        result = t.transform(df)
        assert "val_sin_1" in result.columns
        assert "val_cos_1" in result.columns
        assert "val_sin_2" in result.columns
        assert "val_cos_2" in result.columns

    def test_output_values_bounded(self) -> None:
        df = pd.DataFrame({"val": list(range(20))})
        t = FourierFeaturesTransformer(variables=["val"], period=7, n_harmonics=2)
        t.fit(df)
        result = t.transform(df)
        for col in ["val_sin_1", "val_cos_1", "val_sin_2", "val_cos_2"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0


class TestTimeSinceReferenceTransformer:
    """Tests for TimeSinceReferenceTransformer."""

    def test_days_since_reference(self) -> None:
        df = pd.DataFrame(
            {
                "dt": pd.date_range("2020-01-01", periods=10, freq="D"),
            }
        )
        t = TimeSinceReferenceTransformer(variables=["dt"], unit="days")
        t.fit(df)
        result = t.transform(df)
        assert "dt_days_since_ref" in result.columns
        assert result["dt_days_since_ref"].iloc[0] == pytest.approx(0.0)
        assert result["dt_days_since_ref"].iloc[1] == pytest.approx(1.0)

    def test_drops_original_column(self) -> None:
        df = pd.DataFrame(
            {
                "dt": pd.date_range("2020-01-01", periods=10, freq="D"),
            }
        )
        t = TimeSinceReferenceTransformer(variables=["dt"], unit="days")
        t.fit(df)
        result = t.transform(df)
        assert "dt" not in result.columns


class TestNewRegistryEntries:
    """Tests that new transformers are registered in the registry."""

    def test_robust_scaler_in_registry(self) -> None:
        spec = get_transformer("robust_scaler")
        assert spec is not None

    def test_differencing_in_registry(self) -> None:
        spec = get_transformer("differencing")
        assert spec is not None

    def test_fourier_features_in_registry(self) -> None:
        spec = get_transformer("fourier_features")
        assert spec is not None

    def test_time_since_reference_in_registry(self) -> None:
        spec = get_transformer("time_since_reference")
        assert spec is not None
