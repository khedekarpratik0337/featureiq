"""End-to-end integration tests for FeatureIQ using real sklearn datasets."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from featureiq import (
    FeatureIQ,
    FeatureIQError,
    InsufficientDataError,
    UnsupportedAlgorithmError,
)
from featureiq.exceptions import UnsupportedProblemTypeError
from featureiq.recommender.meta_learner import MetaLearner
from featureiq.utils.validation import ProblemType


@pytest.fixture
def iris_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


@pytest.fixture
def breast_cancer_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


@pytest.fixture
def diabetes_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


class TestIntegration:
    """End-to-end integration tests."""

    def test_fit_transform_binary_classification(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        result = fiq.fit_transform(X, y)
        assert result is not None
        assert fiq.is_fitted_

    def test_fit_transform_regression(
        self, diabetes_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = diabetes_data
        fiq = FeatureIQ(
            problem_type="regression",
            algorithm="ridge",
            use_meta_learner=False,
            verbose=False,
        )
        result = fiq.fit_transform(X, y)
        assert result is not None

    def test_sklearn_pipeline_compatibility(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        y_train = y.iloc[:400]

        pipe = Pipeline(
            [
                (
                    "fiq",
                    FeatureIQ(
                        problem_type="binary_classification",
                        algorithm="logistic_regression",
                        use_meta_learner=False,
                        verbose=False,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_recommend_returns_non_empty(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="multiclass_classification",
            algorithm="random_forest",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        recs = fiq.recommend()
        assert isinstance(recs, dict)
        assert len(recs) > 0

    def test_score_report_is_string(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="multiclass_classification",
            algorithm="knn",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        report = fiq.score_report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert "FeatureIQ" in report

    def test_transform_before_fit_raises(self) -> None:
        fiq = FeatureIQ(verbose=False)
        with pytest.raises(FeatureIQError, match="not been fitted"):
            fiq.transform(pd.DataFrame({"a": [1, 2, 3]}))

    def test_unsupported_algorithm_raises(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="not_a_real_algo",
            use_meta_learner=False,
            verbose=False,
        )
        with pytest.raises(UnsupportedAlgorithmError):
            fiq.fit(X, y)

    def test_insufficient_data_raises(self) -> None:
        X = pd.DataFrame({"a": range(10), "b": range(10)})
        y = pd.Series(range(10))
        fiq = FeatureIQ(
            problem_type="regression",
            algorithm="ridge",
            use_meta_learner=False,
            verbose=False,
        )
        with pytest.raises(InsufficientDataError):
            fiq.fit(X, y)


class TestResolveProblemType:
    """Tests for _resolve_problem_type edge cases."""

    def test_problem_type_enum_directly(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        assert fiq.is_fitted_

    def test_invalid_problem_type_string_raises(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="invalid_type",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        with pytest.raises(UnsupportedProblemTypeError):
            fiq.fit(X, y)


class TestInputConversion:
    """Tests for automatic input type conversion."""

    def test_y_none_raises(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        with pytest.raises(FeatureIQError, match="Target series y must be provided"):
            fiq.fit(X, None)

    def test_non_series_y_input(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y.values)
        assert fiq.is_fitted_

    def test_non_dataframe_x_input(self) -> None:
        rng = np.random.RandomState(42)
        X_dict = {"a": rng.randn(100).tolist(), "b": rng.randn(100).tolist()}
        y = pd.Series(rng.choice([0, 1], 100), name="target")
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X_dict, y)
        assert fiq.is_fitted_

    def test_transform_with_non_dataframe(self) -> None:
        rng = np.random.RandomState(42)
        X_dict = {"a": rng.randn(100).tolist(), "b": rng.randn(100).tolist()}
        y = pd.Series(rng.choice([0, 1], 100), name="target")
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        X_df = pd.DataFrame(X_dict)
        fiq.fit(X_df, y)
        result = fiq.transform(X_dict)
        assert result is not None


class TestRecommendBeforeFit:
    """Tests for calling recommend() and score_report() before fit()."""

    def test_recommend_before_fit_raises(self) -> None:
        fiq = FeatureIQ(verbose=False)
        with pytest.raises(FeatureIQError, match="not been fitted"):
            fiq.recommend()

    def test_score_report_before_fit_raises(self) -> None:
        fiq = FeatureIQ(verbose=False)
        with pytest.raises(FeatureIQError, match="not been fitted"):
            fiq.score_report()


class TestVerboseOutput:
    """Tests for verbose/rich output."""

    def test_fit_with_verbose_true(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=True,
        )
        fiq.fit(X, y)
        assert fiq.is_fitted_

    def test_print_summary_import_error_fallback(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series], capsys
    ) -> None:
        from unittest.mock import patch

        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)

        import builtins

        real_import = builtins.__import__

        def fail_rich(name, *args, **kwargs):
            if "rich" in name:
                raise ImportError("no rich")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_rich):
            with patch("featureiq.core.logger") as mock_logger:
                fiq._print_summary()
                mock_logger.info.assert_called()
                log_output = " ".join(
                    str(call.args[0]) for call in mock_logger.info.call_args_list
                )
                assert "FeatureIQ" in log_output


class TestMetaLearnerIntegration:
    """Tests for FeatureIQ with use_meta_learner=True."""

    def test_meta_learner_with_trained_model(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        rng = np.random.RandomState(42)
        records = []
        transforms = ["standard_scaler", "log_transform", "one_hot_encoder"]
        for _ in range(100):
            records.append(
                {
                    "task_id": rng.randint(1, 10000),
                    "meta_features": rng.randn(26).astype(np.float64),
                    "transformation": rng.choice(transforms),
                    "performance_delta": rng.uniform(-0.1, 0.1),
                    "label": int(rng.choice([0, 1])),
                }
            )

        ml = MetaLearner()
        ml.train(records)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model_path = f.name
        ml.save(model_path)

        try:
            fiq = FeatureIQ(
                problem_type="binary_classification",
                algorithm="xgboost",
                use_meta_learner=True,
                meta_learner_path=model_path,
                verbose=False,
            )
            fiq.fit(X, y)
            assert fiq.is_fitted_
            assert fiq._used_meta_learner is True
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_meta_learner_boosting_path(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        from unittest.mock import MagicMock, patch

        X, y = breast_cancer_data
        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict.return_value = [
            ("standard_scaler", 0.95),
            ("log_transform", 0.90),
            ("iqr_clipper", 0.85),
        ]

        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="logistic_regression",
            use_meta_learner=True,
            verbose=False,
        )

        with patch("featureiq.core.MetaLearner", return_value=mock_ml):
            fiq.fit(X, y)
            assert fiq._used_meta_learner is True

    def test_meta_learner_fallback_on_bad_path(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="xgboost",
            use_meta_learner=True,
            meta_learner_path="/tmp/nonexistent_model.joblib",
            verbose=False,
        )
        fiq.fit(X, y)
        assert fiq.is_fitted_
        assert fiq._used_meta_learner is False


class TestFallbackRecommender:
    """Test for ontology_only_recommend direct invocation."""

    def test_ontology_only_recommend(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        from featureiq.ontology.engine import OntologyEngine
        from featureiq.profiler.dataset_profiler import profile_dataset
        from featureiq.recommender.fallback import ontology_only_recommend
        from featureiq.utils.validation import AlgorithmFamily

        X, y = iris_data
        profile = profile_dataset(X, y, ProblemType.MULTICLASS_CLASSIFICATION)
        engine = OntologyEngine()
        result = ontology_only_recommend(
            profile,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.MULTICLASS_CLASSIFICATION,
            engine,
        )
        assert isinstance(result, dict)


class TestAnomalyDetection:
    """Tests for anomaly detection problem type support."""

    def test_anomaly_detection_y_none_works(self) -> None:
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            {
                "a": rng.randn(100),
                "b": rng.randn(100),
                "c": rng.randn(100),
            }
        )
        fiq = FeatureIQ(
            problem_type="anomaly_detection",
            algorithm="isolation_forest",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X)
        assert fiq.is_fitted_ is True

    def test_anomaly_detection_y_none_regression_raises(self) -> None:
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            {
                "a": rng.randn(100),
                "b": rng.randn(100),
                "c": rng.randn(100),
            }
        )
        fiq = FeatureIQ(
            problem_type="regression",
            algorithm="isolation_forest",
            use_meta_learner=False,
            verbose=False,
        )
        with pytest.raises(FeatureIQError, match="Target series y must be provided"):
            fiq.fit(X, None)


class TestExplain:
    """Tests for the explain() and explain_report() methods."""

    def test_explain_before_fit_raises(self) -> None:
        fiq = FeatureIQ(verbose=False)
        with pytest.raises(FeatureIQError):
            fiq.explain()

    def test_explain_all_columns(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="logistic_regression",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        result = fiq.explain()
        assert isinstance(result, dict)
        assert len(result) > 0
        required_keys = {
            "column",
            "transformation",
            "confidence",
            "explanation",
            "evidence",
            "source",
            "rule_id",
        }
        for col_name, explanations in result.items():
            for entry in explanations:
                assert required_keys.issubset(entry.keys())

    def test_explain_single_column(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="logistic_regression",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        col = next(iter(fiq.recommendations_))
        result = fiq.explain(column=col)
        assert isinstance(result, list)
        required_keys = {
            "column",
            "transformation",
            "confidence",
            "explanation",
            "evidence",
            "source",
            "rule_id",
        }
        for entry in result:
            assert required_keys.issubset(entry.keys())

    def test_explain_report_runs_without_error(
        self, breast_cancer_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = breast_cancer_data
        fiq = FeatureIQ(
            problem_type="binary_classification",
            algorithm="logistic_regression",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        fiq.explain_report()


class TestRecommendAndScoreReport:
    """Tests for recommend() and score_report() output structure."""

    def test_recommend_structure(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="multiclass_classification",
            algorithm="random_forest",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        recs = fiq.recommend()
        for col_name, entries in recs.items():
            for entry in entries:
                assert "transformation" in entry
                assert "confidence" in entry
                assert "reason" in entry
                assert "source" in entry

    def test_score_report_content(
        self, iris_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = iris_data
        fiq = FeatureIQ(
            problem_type="multiclass_classification",
            algorithm="knn",
            use_meta_learner=False,
            verbose=False,
        )
        fiq.fit(X, y)
        report = fiq.score_report()
        assert "Columns profiled" in report
        assert "Transforms recommended" in report
        assert "Algorithm family" in report
        assert "Recommendation method" in report
        assert "ontology-only" in report
