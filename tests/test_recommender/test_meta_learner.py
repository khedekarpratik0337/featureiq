"""Tests for the meta-learner module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from featureiq.exceptions import RecommenderError
from featureiq.recommender.meta_learner import MetaLearner
from featureiq.recommender.openml_collector import (
    load_training_data,
    save_training_data,
)


def _make_synthetic_training_data(n: int = 50) -> list[dict]:
    """Create synthetic training data for testing."""
    rng = np.random.RandomState(42)
    records = []
    transforms = ["standard_scaler", "log_transform", "one_hot_encoder"]
    for _ in range(n):
        meta = rng.randn(26).astype(np.float64)
        t = rng.choice(transforms)
        label = rng.choice([0, 1])
        records.append({
            "task_id": rng.randint(1, 10000),
            "meta_features": meta,
            "transformation": t,
            "performance_delta": rng.uniform(-0.1, 0.1),
            "label": int(label),
        })
    return records


class TestMetaLearner:
    """Tests for the MetaLearner class."""

    def test_train_returns_scores(self) -> None:
        data = _make_synthetic_training_data(100)
        ml = MetaLearner()
        scores = ml.train(data)
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for auc in scores.values():
            assert 0.0 <= auc <= 1.0

    def test_predict_after_train(self) -> None:
        data = _make_synthetic_training_data(100)
        ml = MetaLearner()
        ml.train(data)
        vec = np.random.randn(26).astype(np.float64)
        results = ml.predict(vec, ["standard_scaler", "log_transform"])
        assert len(results) == 2
        assert all(0.0 <= score <= 1.0 for _, score in results)

    def test_predict_before_train_raises(self) -> None:
        ml = MetaLearner()
        vec = np.random.randn(26).astype(np.float64)
        with pytest.raises(RecommenderError, match="not fitted"):
            ml.predict(vec, ["standard_scaler"])

    def test_save_and_load(self) -> None:
        data = _make_synthetic_training_data(100)
        ml = MetaLearner()
        ml.train(data)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        ml.save(path)

        ml2 = MetaLearner(model_path=path)
        assert ml2.is_fitted
        assert len(ml2.models) > 0

        Path(path).unlink(missing_ok=True)

    def test_train_skips_transform_with_few_records(self) -> None:
        rng = np.random.RandomState(42)
        records = []
        for _ in range(3):
            records.append({
                "task_id": rng.randint(1, 10000),
                "meta_features": rng.randn(26).astype(np.float64),
                "transformation": "rare_transform",
                "performance_delta": 0.01,
                "label": 1,
            })
        for _ in range(50):
            records.append({
                "task_id": rng.randint(1, 10000),
                "meta_features": rng.randn(26).astype(np.float64),
                "transformation": "standard_scaler",
                "performance_delta": 0.01,
                "label": rng.choice([0, 1]),
            })
        ml = MetaLearner()
        scores = ml.train(records)
        assert "rare_transform" not in ml.models
        assert "standard_scaler" in ml.models

    def test_train_skips_single_class_transform(self) -> None:
        rng = np.random.RandomState(42)
        records = []
        for _ in range(20):
            records.append({
                "task_id": rng.randint(1, 10000),
                "meta_features": rng.randn(26).astype(np.float64),
                "transformation": "mono_transform",
                "performance_delta": 0.01,
                "label": 1,
            })
        ml = MetaLearner()
        scores = ml.train(records)
        assert scores["mono_transform"] == 0.5
        assert "mono_transform" not in ml.models

    def test_predict_unknown_transformation_returns_half(self) -> None:
        data = _make_synthetic_training_data(100)
        ml = MetaLearner()
        ml.train(data)
        vec = np.random.randn(26).astype(np.float64)
        results = ml.predict(vec, ["unknown_transform_xyz"])
        assert results[0][0] == "unknown_transform_xyz"
        assert results[0][1] == 0.5

    def test_load_corrupted_file_raises(self, tmp_path) -> None:
        bad_file = tmp_path / "bad.joblib"
        bad_file.write_text("this is not joblib data")
        with pytest.raises(RecommenderError, match="Failed to load"):
            MetaLearner(model_path=str(bad_file))

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(RecommenderError, match="not found"):
            MetaLearner(model_path="/tmp/nonexistent_model_xyz.joblib")

    def test_train_roc_auc_value_error(self) -> None:
        data = _make_synthetic_training_data(100)
        ml = MetaLearner()
        with patch(
            "sklearn.metrics.roc_auc_score",
            side_effect=ValueError("Only one class"),
        ):
            scores = ml.train(data)
        for score in scores.values():
            assert score == 0.5


class TestTrainingDataIO:
    """Tests for save/load training data."""

    def test_save_and_load_roundtrip(self) -> None:
        data = _make_synthetic_training_data(10)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        save_training_data(data, path)
        loaded = load_training_data(path)
        assert len(loaded) == 10

        Path(path).unlink(missing_ok=True)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(RecommenderError, match="not found"):
            load_training_data("/tmp/nonexistent_file_12345.joblib")
