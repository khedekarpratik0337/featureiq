"""LightGBM-based meta-learner for transformation recommendation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from featureiq.exceptions import RecommenderError

logger = logging.getLogger(__name__)

META_LEARNER_VERSION = "v1"


class MetaLearner:
    """LightGBM-based meta-learner for transformation recommendation.

    Trained on OpenML task meta-features. Predicts which transformations
    will improve model performance for a given dataset + algorithm combination.

    Args:
        model_path: Optional path to a pre-trained model file.
                    If None, auto-discovers the bundled pretrained model.

    Attributes:
        model: Dict mapping transformation name to trained LightGBM classifier.
        is_fitted: Boolean flag.
        supported_transformations: List of transformation names the model can predict for.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.models: dict[str, Any] = {}
        self.is_fitted: bool = False
        self.supported_transformations: list[str] = []

        if model_path is not None:
            self.load(model_path)
        else:
            default_path = Path(__file__).parent / "pretrained" / f"meta_learner_{META_LEARNER_VERSION}.joblib"
            if default_path.exists():
                self.load(str(default_path))
                logger.info("Loaded bundled meta-learner model (%s).", META_LEARNER_VERSION)

    def train(
        self,
        training_data: list[dict[str, Any]],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Train the meta-learner on collected OpenML data.

        One LightGBM binary classifier per transformation type.
        (i.e., one model that predicts whether transformation T is beneficial
        for a given meta-feature vector)

        Args:
            training_data: Output of fetch_openml_tasks or load_training_data.
            test_size: Fraction of data held out for evaluation.
            random_state: Random seed for reproducibility.

        Returns:
            Dict mapping transformation name to validation AUC score.
        """
        import lightgbm as lgb

        transform_records: dict[str, list[tuple[np.ndarray, int]]] = {}
        for record in training_data:
            t_name = record["transformation"]
            if t_name not in transform_records:
                transform_records[t_name] = []
            transform_records[t_name].append(
                (record["meta_features"], record["label"])
            )

        scores: dict[str, float] = {}
        self.models = {}

        for t_name, records in transform_records.items():
            if len(records) < 5:
                logger.warning(
                    f"Skipping '{t_name}': only {len(records)} records."
                )
                continue

            X_all = np.array([r[0] for r in records])
            y_all = np.array([r[1] for r in records])

            if len(np.unique(y_all)) < 2:
                logger.warning(
                    f"Skipping '{t_name}': only one class present."
                )
                scores[t_name] = 0.5
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=test_size, random_state=random_state,
                stratify=y_all,
            )

            clf = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
            )
            clf.fit(X_train, y_train)

            proba = clf.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score

            try:
                auc = float(roc_auc_score(y_val, proba))
            except ValueError:
                auc = 0.5

            self.models[t_name] = clf
            scores[t_name] = auc
            logger.info(f"Trained model for '{t_name}': AUC = {auc:.4f}")

        self.supported_transformations = list(self.models.keys())
        self.is_fitted = len(self.models) > 0
        return scores

    def predict(
        self,
        meta_feature_vector: np.ndarray,
        candidate_transformations: list[str],
    ) -> list[tuple[str, float]]:
        """Predict utility of each candidate transformation.

        Args:
            meta_feature_vector: Shape (24,) numpy array from build_meta_feature_vector.
            candidate_transformations: List of transformation names to score.

        Returns:
            List of (transformation_name, probability_score) sorted by score descending.

        Raises:
            RecommenderError: If model is not fitted.
        """
        if not self.is_fitted:
            raise RecommenderError(
                "MetaLearner is not fitted. Call train() first or load a model."
            )

        results: list[tuple[str, float]] = []
        X_input = meta_feature_vector.reshape(1, -1)

        for t_name in candidate_transformations:
            if t_name in self.models:
                proba = float(self.models[t_name].predict_proba(X_input)[0, 1])
                results.append((t_name, proba))
            else:
                results.append((t_name, 0.5))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def save(self, path: str) -> None:
        """Serialise the trained model to disk using joblib.

        Args:
            path: File path to save the model.
        """
        data = {
            "models": self.models,
            "supported_transformations": self.supported_transformations,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        """Load a pre-trained model from disk.

        Args:
            path: File path to load the model from.
        """
        try:
            data = joblib.load(path)
        except FileNotFoundError as exc:
            raise RecommenderError(
                f"Model file not found: {path}"
            ) from exc
        except Exception as exc:
            raise RecommenderError(
                f"Failed to load model from {path}: {exc}"
            ) from exc

        self.models = data["models"]
        self.supported_transformations = data["supported_transformations"]
        self.is_fitted = data["is_fitted"]
