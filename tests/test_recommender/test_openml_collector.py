"""Tests for the OpenML data collector — mock-based, no network calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from featureiq.exceptions import RecommenderError
from featureiq.recommender.openml_collector import (
    fetch_openml_tasks,
    load_training_data,
)


class TestFetchOpenmlTasks:
    """Mock-based tests for fetch_openml_tasks."""

    def test_openml_import_error(self) -> None:
        with patch.dict("sys.modules", {"openml": None}):
            with pytest.raises(RecommenderError, match="openml library"):
                fetch_openml_tasks()

    def test_api_failure_raises(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.list_tasks.side_effect = RuntimeError("API down")
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1
        with patch.dict("sys.modules", {"openml": mock_openml}):
            with pytest.raises(RecommenderError, match="Failed to fetch"):
                fetch_openml_tasks()

    def test_empty_filtered_results_raises(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1
        empty_df = pd.DataFrame({"NumberOfInstances": [], "tid": []})
        mock_openml.tasks.list_tasks.return_value = empty_df
        with patch.dict("sys.modules", {"openml": mock_openml}):
            with pytest.raises(RecommenderError, match="No matching tasks"):
                fetch_openml_tasks()

    def test_successful_task_collection(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200, 300],
                "tid": [1, 2],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        rng = np.random.RandomState(42)
        X_data = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y_data = pd.Series(np.concatenate([np.zeros(100), np.ones(100)]), name="target")

        mock_dataset = MagicMock()
        mock_dataset.default_target_attribute = "target"
        mock_dataset.get_data.return_value = (X_data, y_data, None, None)

        mock_task = MagicMock()
        mock_task.get_dataset.return_value = mock_dataset

        mock_openml.tasks.get_task.return_value = mock_task

        with patch.dict("sys.modules", {"openml": mock_openml}):
            records = fetch_openml_tasks(
                n_tasks=1, min_instances=100, max_instances=1000
            )

        assert len(records) > 0
        for rec in records:
            assert "meta_features" in rec
            assert "transformation" in rec
            assert "label" in rec
            assert rec["meta_features"].shape == (26,)

    def test_skips_task_with_none_data(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200],
                "tid": [1],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        mock_dataset = MagicMock()
        mock_dataset.default_target_attribute = "target"
        mock_dataset.get_data.return_value = (None, None, None, None)

        mock_task = MagicMock()
        mock_task.get_dataset.return_value = mock_dataset
        mock_openml.tasks.get_task.return_value = mock_task

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with pytest.raises(RecommenderError, match="Failed to collect"):
                fetch_openml_tasks(n_tasks=1, min_instances=100)

    def test_skips_task_with_no_numeric_columns(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200],
                "tid": [1],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        X_data = pd.DataFrame({"cat1": ["a"] * 200, "cat2": ["b"] * 200})
        y_data = pd.Series(["pos"] * 100 + ["neg"] * 100, name="target")

        mock_dataset = MagicMock()
        mock_dataset.default_target_attribute = "target"
        mock_dataset.get_data.return_value = (X_data, y_data, None, None)

        mock_task = MagicMock()
        mock_task.get_dataset.return_value = mock_dataset
        mock_openml.tasks.get_task.return_value = mock_task

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with pytest.raises(RecommenderError, match="Failed to collect"):
                fetch_openml_tasks(n_tasks=1, min_instances=100)

    def test_skips_task_with_too_few_rows(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200],
                "tid": [1],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        rng = np.random.RandomState(42)
        X_data = pd.DataFrame(rng.randn(50, 3), columns=["f0", "f1", "f2"])
        y_data = pd.Series(["pos"] * 25 + ["neg"] * 25, name="target")

        mock_dataset = MagicMock()
        mock_dataset.default_target_attribute = "target"
        mock_dataset.get_data.return_value = (X_data, y_data, None, None)

        mock_task = MagicMock()
        mock_task.get_dataset.return_value = mock_dataset
        mock_openml.tasks.get_task.return_value = mock_task

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with pytest.raises(RecommenderError, match="Failed to collect"):
                fetch_openml_tasks(n_tasks=1, min_instances=100)

    def test_skips_task_on_general_exception(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200, 300],
                "tid": [1, 2],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        rng = np.random.RandomState(42)
        X_good = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y_good = pd.Series(np.concatenate([np.zeros(100), np.ones(100)]), name="target")

        mock_dataset_good = MagicMock()
        mock_dataset_good.default_target_attribute = "target"
        mock_dataset_good.get_data.return_value = (X_good, y_good, None, None)

        mock_task_good = MagicMock()
        mock_task_good.get_dataset.return_value = mock_dataset_good

        def get_task_side_effect(tid):
            if tid == 1:
                raise RuntimeError("task download failed")
            return mock_task_good

        mock_openml.tasks.get_task.side_effect = get_task_side_effect

        with patch.dict("sys.modules", {"openml": mock_openml}):
            records = fetch_openml_tasks(
                n_tasks=1, min_instances=100, max_instances=1000
            )
        assert len(records) > 0

    def test_skips_task_on_baseline_cv_failure(self) -> None:
        mock_openml = MagicMock()
        mock_openml.tasks.TaskType.SUPERVISED_CLASSIFICATION = 1

        tasks_df = pd.DataFrame(
            {
                "NumberOfInstances": [200],
                "tid": [1],
            }
        )
        mock_openml.tasks.list_tasks.return_value = tasks_df

        rng = np.random.RandomState(42)
        X_data = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y_data = pd.Series(np.concatenate([np.zeros(100), np.ones(100)]), name="target")

        mock_dataset = MagicMock()
        mock_dataset.default_target_attribute = "target"
        mock_dataset.get_data.return_value = (X_data, y_data, None, None)

        mock_task = MagicMock()
        mock_task.get_dataset.return_value = mock_dataset
        mock_openml.tasks.get_task.return_value = mock_task

        with patch.dict("sys.modules", {"openml": mock_openml}):
            with patch(
                "featureiq.recommender.openml_collector.cross_val_score",
                side_effect=ValueError("CV failed"),
            ):
                with pytest.raises(RecommenderError, match="Failed to collect"):
                    fetch_openml_tasks(n_tasks=1, min_instances=100)


class TestTrainingDataEdgeCases:
    """Tests for load/save edge cases."""

    def test_load_corrupted_file_raises(self, tmp_path) -> None:
        bad_file = tmp_path / "bad.joblib"
        bad_file.write_text("not joblib data")
        with pytest.raises(RecommenderError, match="Failed to load"):
            load_training_data(str(bad_file))

    def test_load_non_list_data_raises(self, tmp_path) -> None:
        import joblib

        dict_file = tmp_path / "dict.joblib"
        joblib.dump({"key": "value"}, str(dict_file))
        with pytest.raises(RecommenderError, match="Invalid training data format"):
            load_training_data(str(dict_file))
