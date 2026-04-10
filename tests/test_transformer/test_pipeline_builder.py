"""Tests for the pipeline builder and transformer registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from featureiq.exceptions import TransformerError
from featureiq.ontology.engine import OntologyEngine
from featureiq.profiler.dataset_profiler import profile_dataset
from featureiq.transformer.pipeline_builder import (
    build_column_transformer,
    build_pipeline,
)
from featureiq.transformer.registry import (
    BinaryEncoder,
    DateComponentExtractor,
    IQRClipper,
    LagFeatureGenerator,
    RareLabelGrouper,
    RollingStatsGenerator,
    get_transformer,
)
from featureiq.utils.validation import AlgorithmFamily, ProblemType


class TestTransformerRegistry:
    """Tests for the transformer registry."""

    def test_get_known_transformer(self) -> None:
        spec = get_transformer("standard_scaler")
        assert spec.description != ""

    def test_get_unknown_transformer_raises(self) -> None:
        with pytest.raises(TransformerError):
            get_transformer("nonexistent_transformer")


class TestCustomTransformers:
    """Tests for custom sklearn-compatible transformers."""

    def test_rare_label_grouper(self) -> None:
        df = pd.DataFrame({"col": ["a"] * 90 + ["b"] * 5 + ["c"] * 3 + ["d"] * 2})
        rlg = RareLabelGrouper(tol=0.05)
        result = rlg.fit_transform(df)
        assert "Rare" in result["col"].values
        assert "a" in result["col"].values

    def test_iqr_clipper(self) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"val": rng.randn(100)})
        df.loc[0, "val"] = 100.0
        clipper = IQRClipper(factor=1.5)
        result = clipper.fit_transform(df)
        assert result["val"].max() < 100.0

    def test_date_component_extractor(self) -> None:
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=10, freq="D")})
        ext = DateComponentExtractor(components=["year", "month", "day", "weekday"])
        result = ext.fit_transform(df)
        assert "dt_year" in result.columns
        assert "dt_month" in result.columns
        assert "dt" not in result.columns

    def test_lag_feature_generator(self) -> None:
        df = pd.DataFrame({"val": range(20)})
        lag = LagFeatureGenerator(variables=["val"], lags=[1, 2])
        result = lag.fit_transform(df)
        assert "val_lag_1" in result.columns
        assert "val_lag_2" in result.columns

    def test_rolling_stats_generator(self) -> None:
        df = pd.DataFrame({"val": range(20)})
        roll = RollingStatsGenerator(variables=["val"], windows=[3])
        result = roll.fit_transform(df)
        assert "val_rolling_mean_3" in result.columns
        assert "val_rolling_std_3" in result.columns


class TestBinaryEncoder:
    """Tests for BinaryEncoder transformer."""

    def test_fit_transform_categorical(self) -> None:
        df = pd.DataFrame({"color": ["red", "green", "blue", "red", "green"] * 20})
        enc = BinaryEncoder()
        result = enc.fit_transform(df)
        assert "color" not in result.columns
        bin_cols = [c for c in result.columns if c.startswith("color_bin_")]
        assert len(bin_cols) >= 1

    def test_no_categorical_columns(self) -> None:
        df = pd.DataFrame({"val": range(20)})
        enc = BinaryEncoder()
        result = enc.fit_transform(df)
        assert "val" in result.columns
        assert result.shape == (20, 1)

    def test_unknown_values_at_transform(self) -> None:
        train_df = pd.DataFrame({"cat": ["a", "b", "c"] * 20})
        test_df = pd.DataFrame({"cat": ["a", "d", "e"]})
        enc = BinaryEncoder()
        enc.fit(train_df)
        result = enc.transform(test_df)
        assert "cat" not in result.columns
        bin_cols = [c for c in result.columns if c.startswith("cat_bin_")]
        assert len(bin_cols) >= 1

    def test_multiple_categorical_columns(self) -> None:
        df = pd.DataFrame(
            {
                "a": ["x", "y", "z"] * 10,
                "b": ["p", "q"] * 15,
                "num": range(30),
            }
        )
        enc = BinaryEncoder(variables=["a", "b"])
        result = enc.fit_transform(df)
        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "num" in result.columns


class TestBuildColumnTransformer:
    """Tests for build_column_transformer."""

    def test_mixed_recommendations(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        engine = OntologyEngine()
        recs = engine.recommend_for_dataset(
            profile, AlgorithmFamily.LINEAR_MODEL, ProblemType.BINARY_CLASSIFICATION
        )
        ct = build_column_transformer(recs, profile)
        assert ct is not None

    def test_empty_recommendations(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        ct = build_column_transformer({}, profile)
        assert ct is not None


class TestBuildPipeline:
    """Tests for build_pipeline."""

    def test_pipeline_fittable(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        df = mixed_dataframe.drop(columns=["date1"])
        profile = profile_dataset(df, target_binary, ProblemType.BINARY_CLASSIFICATION)
        engine = OntologyEngine()
        recs = engine.recommend_for_dataset(
            profile, AlgorithmFamily.LINEAR_MODEL, ProblemType.BINARY_CLASSIFICATION
        )
        pipe = build_pipeline(recs, profile)
        result = pipe.fit_transform(df, target_binary)
        assert result is not None

    def test_empty_recommendations_passthrough(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        df = mixed_dataframe.drop(columns=["date1"])
        profile = profile_dataset(df, target_binary, ProblemType.BINARY_CLASSIFICATION)
        pipe = build_pipeline({}, profile)
        result = pipe.fit_transform(df)
        assert result is not None

    def test_pipeline_with_estimator(
        self, mixed_dataframe: pd.DataFrame, target_binary: pd.Series
    ) -> None:
        df = mixed_dataframe.drop(columns=["date1"])
        profile = profile_dataset(df, target_binary, ProblemType.BINARY_CLASSIFICATION)
        engine = OntologyEngine()
        recs = engine.recommend_for_dataset(
            profile, AlgorithmFamily.LINEAR_MODEL, ProblemType.BINARY_CLASSIFICATION
        )
        pipe = build_pipeline(
            recs, profile, estimator=LogisticRegression(max_iter=1000)
        )
        pipe.fit(df, target_binary)
        preds = pipe.predict(df)
        assert len(preds) == len(df)
