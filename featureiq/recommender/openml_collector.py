"""Collect training data from OpenML for the meta-learner."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder

from featureiq.exceptions import RecommenderError
from featureiq.profiler.dataset_profiler import profile_dataset
from featureiq.profiler.meta_features import build_meta_feature_vector
from featureiq.transformer.registry import TRANSFORMER_REGISTRY
from featureiq.utils.validation import ProblemType

logger = logging.getLogger(__name__)

_CLASSIFICATION_TRANSFORMS = [
    "standard_scaler",
    "min_max_scaler",
    "one_hot_encoder",
    "ordinal_encoder",
    "log_transform",
]


def fetch_openml_tasks(
    n_tasks: int = 50,
    task_type: str = "Supervised Classification",
    min_instances: int = 100,
    max_instances: int = 100000,
) -> list[dict[str, Any]]:
    """Fetch task metadata from OpenML for meta-learning training.

    Uses the openml Python library (real library -- import openml).
    Fetches tasks, downloads datasets, and computes meta-feature vectors.

    For each task:
    1. Download dataset via openml.datasets.get_dataset()
    2. Run profile_dataset() to get DatasetProfile
    3. Run build_meta_feature_vector() to get meta-feature vector
    4. Record which transformations (from registry) improve accuracy
       vs. a no-transform baseline using 3-fold CV with LogisticRegression

    Args:
        n_tasks: Number of OpenML tasks to collect.
        task_type: OpenML task type string.
        min_instances: Minimum dataset size to include.
        max_instances: Maximum dataset size to include.

    Returns:
        List of dicts, each with keys:
        {
            "task_id": int,
            "meta_features": np.ndarray,  # shape (26,)
            "transformation": str,         # transformation name
            "performance_delta": float,    # accuracy improvement vs baseline
            "label": int                   # 1 if helpful, 0 if not
        }

    Raises:
        RecommenderError: If OpenML API is unreachable or returns no tasks.
    """
    try:
        import openml
    except ImportError as exc:
        raise RecommenderError(
            "openml library is required for data collection."
        ) from exc

    try:
        tasks = openml.tasks.list_tasks(
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            output_format="dataframe",
        )
    except Exception as exc:
        raise RecommenderError(
            f"Failed to fetch tasks from OpenML: {exc}"
        ) from exc

    filtered = tasks[
        (tasks["NumberOfInstances"] >= min_instances)
        & (tasks["NumberOfInstances"] <= max_instances)
    ].head(n_tasks * 3)

    if filtered.empty:
        raise RecommenderError("No matching tasks found on OpenML.")

    records: list[dict[str, Any]] = []
    collected = 0

    for task_id in filtered["tid"].values:
        if collected >= n_tasks:
            break

        try:
            task = openml.tasks.get_task(int(task_id))
            dataset = task.get_dataset()
            X_raw, y_raw, _, _ = dataset.get_data(
                target=dataset.default_target_attribute
            )

            if X_raw is None or y_raw is None:
                continue

            X = pd.DataFrame(X_raw) if not isinstance(X_raw, pd.DataFrame) else X_raw
            y = pd.Series(y_raw) if not isinstance(y_raw, pd.Series) else y_raw

            if len(X) < 100 or len(X.columns) == 0:
                continue

            le = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y.astype(str)), name="target")

            n_classes = y_enc.nunique()
            problem_type = (
                ProblemType.BINARY_CLASSIFICATION
                if n_classes == 2
                else ProblemType.MULTICLASS_CLASSIFICATION
            )

            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] == 0:
                continue
            X_numeric = X_numeric.fillna(X_numeric.median())

            profile = profile_dataset(X, y_enc, problem_type)
            meta_vec = build_meta_feature_vector(
                profile, problem_type, "logistic_regression"
            )

            baseline_clf = LogisticRegression(max_iter=500, solver="lbfgs")
            try:
                baseline_scores = cross_val_score(
                    baseline_clf, X_numeric, y_enc, cv=3, scoring="accuracy"
                )
                baseline_acc = float(baseline_scores.mean())
            except Exception:
                continue

            for transform_name in _CLASSIFICATION_TRANSFORMS:
                if transform_name not in TRANSFORMER_REGISTRY:  # pragma: no cover
                    continue

                spec = TRANSFORMER_REGISTRY[transform_name]
                transformer = spec.transformer_class(**spec.default_kwargs)

                try:
                    pipe = SkPipeline([
                        ("transform", transformer),
                        ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
                    ])
                    scores = cross_val_score(
                        pipe, X_numeric, y_enc, cv=3, scoring="accuracy"
                    )
                    transform_acc = float(scores.mean())
                except Exception:
                    continue

                delta = transform_acc - baseline_acc
                records.append({
                    "task_id": int(task_id),
                    "meta_features": meta_vec,
                    "transformation": transform_name,
                    "performance_delta": delta,
                    "label": 1 if delta > 0.001 else 0,
                })

            collected += 1
            logger.info(f"Collected task {task_id} ({collected}/{n_tasks})")

        except Exception as exc:
            logger.warning(f"Skipping task {task_id}: {exc}")
            continue

    if not records:
        raise RecommenderError(
            "Failed to collect any training data from OpenML."
        )

    return records


def save_training_data(records: list[dict[str, Any]], output_path: str) -> None:
    """Serialise collected training data to disk using joblib.

    Args:
        records: Output of fetch_openml_tasks.
        output_path: File path to save to (recommended: .joblib extension).
    """
    joblib.dump(records, output_path)


def load_training_data(input_path: str) -> list[dict[str, Any]]:
    """Load previously saved training data from disk.

    Args:
        input_path: File path to load from.

    Returns:
        List of record dicts.

    Raises:
        RecommenderError: If file not found or format is invalid.
    """
    try:
        data = joblib.load(input_path)
    except FileNotFoundError as exc:
        raise RecommenderError(
            f"Training data file not found: {input_path}"
        ) from exc
    except Exception as exc:
        raise RecommenderError(
            f"Failed to load training data from {input_path}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise RecommenderError(
            f"Invalid training data format in {input_path}: expected list."
        )

    return data
