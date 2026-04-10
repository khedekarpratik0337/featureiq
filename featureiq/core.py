"""Main FeatureIQ orchestrator class."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from featureiq.exceptions import (
    FeatureIQError,
    UnsupportedAlgorithmError,
    UnsupportedProblemTypeError,
)
from featureiq.ontology.engine import OntologyEngine
from featureiq.ontology.rule_loader import OntologyRule, RuleCondition
from featureiq.profiler.dataset_profiler import DatasetProfile, profile_dataset
from featureiq.profiler.meta_features import build_meta_feature_vector
from featureiq.recommender.fallback import ontology_only_recommend
from featureiq.recommender.meta_learner import MetaLearner
from featureiq.transformer.pipeline_builder import build_pipeline
from featureiq.transformer.registry import get_transformer
from featureiq.utils.validation import (
    ALGORITHM_FAMILY_MAP,
    MIN_ROWS_FOR_META_LEARNING,
    AlgorithmFamily,
    ProblemType,
)

logger = logging.getLogger(__name__)


class FeatureIQ(BaseEstimator, TransformerMixin):
    """Context-aware automated feature engineering orchestrator.

    Combines ontology-based reasoning and meta-learning to recommend
    and apply feature engineering transformations conditioned on:
    - Data characteristics (profiled automatically)
    - Problem type (declared by user)
    - Algorithm choice (declared by user)

    Scikit-learn compatible: implements fit() and transform(),
    works inside sklearn.pipeline.Pipeline.

    Args:
        problem_type: One of ProblemType enum values or equivalent string.
        algorithm: Algorithm name string (e.g., "xgboost", "logistic_regression").
        use_meta_learner: If True, use MetaLearner for recommendations.
                          Falls back to ontology-only if model not available.
        meta_learner_path: Path to pre-trained MetaLearner model file.
        custom_rules_dir: Path to directory containing custom YAML rule files.
        verbose: If True, print recommendation summary using rich.
        exclude_columns: Columns to pass through untouched.
        override_transforms: Dict mapping column name to list of transform names,
            replacing ontology recommendations for those columns.
        pre_transforms: Transforms injected before the ColumnTransformer.
        post_transforms: Transforms injected after the ColumnTransformer.

    Attributes:
        profile_: DatasetProfile computed during fit().
        recommendations_: Dict of column -> applicable rules, set during fit().
        pipeline_: sklearn Pipeline built during fit().
        is_fitted_: Boolean flag.

    Example:
        >>> import pandas as pd
        >>> from featureiq import FeatureIQ
        >>> fiq = FeatureIQ(problem_type="binary_classification", algorithm="xgboost")
        >>> X_transformed = fiq.fit_transform(X_train, y_train)
        >>> X_test_transformed = fiq.transform(X_test)
    """

    def __init__(
        self,
        problem_type: str | ProblemType = ProblemType.BINARY_CLASSIFICATION,
        algorithm: str = "xgboost",
        use_meta_learner: bool = True,
        meta_learner_path: str | None = None,
        custom_rules_dir: str | None = None,
        verbose: bool = True,
        exclude_columns: list[str] | None = None,
        override_transforms: dict[str, list[str]] | None = None,
        pre_transforms: list[tuple[str, BaseEstimator]] | None = None,
        post_transforms: list[tuple[str, BaseEstimator]] | None = None,
    ) -> None:
        self.problem_type = problem_type
        self.algorithm = algorithm
        self.use_meta_learner = use_meta_learner
        self.meta_learner_path = meta_learner_path
        self.custom_rules_dir = custom_rules_dir
        self.verbose = verbose
        self.exclude_columns = exclude_columns
        self.override_transforms = override_transforms
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        self.profile_: DatasetProfile | None = None
        self.recommendations_: dict[str, list[OntologyRule]] | None = None
        self.pipeline_: Any = None
        self.is_fitted_: bool = False
        self._used_meta_learner: bool = False
        self._validation_results: dict[str, Any] | None = None

    def _resolve_problem_type(self) -> ProblemType:
        """Convert string problem_type to ProblemType enum if needed."""
        if isinstance(self.problem_type, ProblemType):
            return self.problem_type

        try:
            return ProblemType(self.problem_type)
        except ValueError:
            raise UnsupportedProblemTypeError(
                f"Problem type '{self.problem_type}' is not supported. "
                f"Supported types: {[pt.value for pt in ProblemType]}"
            )

    def _resolve_algorithm_family(self) -> AlgorithmFamily:
        """Resolve algorithm name to AlgorithmFamily."""
        key = self.algorithm.lower()
        if key not in ALGORITHM_FAMILY_MAP:
            raise UnsupportedAlgorithmError(
                f"Algorithm '{self.algorithm}' is not supported. "
                f"Supported algorithms: {list(ALGORITHM_FAMILY_MAP.keys())}"
            )
        return ALGORITHM_FAMILY_MAP[key]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureIQ":
        """Profile the dataset, generate recommendations, and build the pipeline.

        Steps:
        1. Validate inputs (types, shapes, minimum size)
        2. Resolve problem_type and algorithm to enums
        3. Run profile_dataset(X, y, problem_type)
        4. Run OntologyEngine.recommend_for_dataset()
        5. If use_meta_learner and model available: run MetaLearner.predict()
           and merge/re-rank recommendations
        6. Build pipeline via build_pipeline()
        7. Fit the pipeline on X, y
        8. Set is_fitted_ = True

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            self (fitted FeatureIQ instance).

        Raises:
            InsufficientDataError: If X has fewer than MIN_ROWS_FOR_PROFILING rows.
            UnsupportedAlgorithmError: If algorithm is not in ALGORITHM_FAMILY_MAP.
            UnsupportedProblemTypeError: If problem_type is not a valid ProblemType.
        """
        pt = self._resolve_problem_type()

        if y is None and pt != ProblemType.ANOMALY_DETECTION:
            raise FeatureIQError("Target series y must be provided to fit().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")

        af = self._resolve_algorithm_family()

        self.profile_ = profile_dataset(X, y, pt)

        engine = OntologyEngine(self.custom_rules_dir)
        self.recommendations_ = engine.recommend_for_dataset(
            self.profile_, af, pt, algorithm=self.algorithm
        )

        self._used_meta_learner = False
        if (
            y is not None
            and self.use_meta_learner
            and len(X) >= MIN_ROWS_FOR_META_LEARNING
        ):
            try:
                ml = MetaLearner(model_path=self.meta_learner_path)
                if ml.is_fitted:
                    meta_vec = build_meta_feature_vector(
                        self.profile_, pt, self.algorithm
                    )
                    candidate_transforms = set()
                    for rules in self.recommendations_.values():
                        for rule in rules:
                            candidate_transforms.add(rule.transformation)

                    if candidate_transforms:
                        scored = ml.predict(meta_vec, list(candidate_transforms))
                        high_score = {t for t, s in scored if s > 0.6}

                        for col_name, rules in self.recommendations_.items():
                            boosted: list[OntologyRule] = []
                            for rule in rules:
                                if rule.transformation in high_score:
                                    boosted.insert(0, rule)
                                else:
                                    boosted.append(rule)
                            self.recommendations_[col_name] = boosted
                        self._used_meta_learner = True
                    logger.info("Using bundled meta-learner for recommendations.")
                else:
                    logger.debug("MetaLearner model not found, using ontology-only.")
            except Exception:
                logger.debug("MetaLearner not available, using ontology-only fallback.")

        if (
            not self._used_meta_learner and self.recommendations_ is None
        ):  # pragma: no cover
            self.recommendations_ = ontology_only_recommend(
                self.profile_, af, pt, engine
            )

        if self.exclude_columns:
            for col in self.exclude_columns:
                self.recommendations_.pop(col, None)

        if self.override_transforms:
            for col, transform_names in self.override_transforms.items():
                override_rules = []
                for t_name in transform_names:
                    get_transformer(t_name)
                    override_rules.append(
                        OntologyRule(
                            id=f"OVERRIDE_{col}_{t_name}",
                            description=f"User override: {t_name} for {col}",
                            version="1.0",
                            conditions=RuleCondition(),
                            algorithm_families=[af],
                            transformation=t_name,
                            contraindicated_for=[],
                            confidence=1.0,
                            source="user_override",
                            tags=["override"],
                        )
                    )
                self.recommendations_[col] = override_rules

        pre_steps = list(self.pre_transforms) if self.pre_transforms else None
        post_steps = list(self.post_transforms) if self.post_transforms else None

        self.pipeline_ = build_pipeline(
            self.recommendations_,
            self.profile_,
            pre_steps=pre_steps,
            post_steps=post_steps,
        )
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True

        if self.verbose:
            self._print_summary()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature engineering pipeline to new data.

        Args:
            X: Feature DataFrame with same columns as training data.

        Returns:
            Transformed DataFrame.

        Raises:
            FeatureIQError: If called before fit().
        """
        if not self.is_fitted_:
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return self.pipeline_.transform(X)

    def recommend(self) -> dict[str, list[dict[str, Any]]]:
        """Return human-readable recommendations generated during fit().

        Returns:
            Dict mapping column name -> list of recommendation dicts,
            each with keys: transformation, confidence, reason, source.

        Raises:
            FeatureIQError: If called before fit().
        """
        if not self.is_fitted_ or self.recommendations_ is None:
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        result: dict[str, list[dict[str, Any]]] = {}
        for col_name, rules in self.recommendations_.items():
            col_recs = []
            for rule in rules:
                rec: dict[str, Any] = {
                    "transformation": rule.transformation,
                    "confidence": rule.confidence,
                    "reason": rule.description,
                    "source": rule.source,
                }
                if self._validation_results is not None:
                    rec["validated_confidence"] = self._validation_results.get(
                        "featureiq_score"
                    )
                col_recs.append(rec)
            result[col_name] = col_recs
        return result

    def get_pipeline(self) -> Pipeline:
        """Return the fitted sklearn Pipeline for inspection or integration.

        Returns:
            The fitted sklearn Pipeline instance.

        Raises:
            FeatureIQError: If called before fit().
        """
        if not self.is_fitted_ or self.pipeline_ is None:
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")
        return self.pipeline_

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: BaseEstimator,
        cv: int = 5,
        scoring: str = "r2",
    ) -> dict[str, Any]:
        """Run cross-validation comparing FeatureIQ pipeline vs a baseline.

        The baseline uses only numeric columns with no transformations.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            estimator: sklearn-compatible estimator to evaluate.
            cv: Number of cross-validation folds.
            scoring: Scoring metric name.

        Returns:
            Dict with featureiq_score, baseline_score, improvement.
        """
        from sklearn.base import clone
        from sklearn.compose import ColumnTransformer as CT
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline as Pipe

        if (
            not self.is_fitted_
            or self.recommendations_ is None
            or self.profile_ is None
        ):
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        fiq_pipe = build_pipeline(
            self.recommendations_,
            self.profile_,
            estimator=clone(estimator),
            pre_steps=list(self.pre_transforms) if self.pre_transforms else None,
            post_steps=list(self.post_transforms) if self.post_transforms else None,
        )

        fiq_scores = cross_val_score(fiq_pipe, X, y, cv=cv, scoring=scoring)
        fiq_mean = float(np.mean(fiq_scores))

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            baseline_pipe = Pipe(
                [
                    ("passthrough", CT([], remainder="passthrough")),
                    ("estimator", clone(estimator)),
                ]
            )
            baseline_scores = cross_val_score(
                baseline_pipe, X[numeric_cols], y, cv=cv, scoring=scoring
            )
            baseline_mean = float(np.mean(baseline_scores))
        else:
            baseline_mean = 0.0

        self._validation_results = {
            "featureiq_score": fiq_mean,
            "baseline_score": baseline_mean,
            "improvement": fiq_mean - baseline_mean,
        }
        return self._validation_results

    def ablate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: BaseEstimator,
        cv: int = 3,
        scoring: str = "r2",
    ) -> list[dict[str, Any]]:
        """Ablation analysis: measure the impact of each individual transform.

        For each recommended transform, builds a pipeline without that
        one transform and measures the cross-validation impact.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            estimator: sklearn-compatible estimator.
            cv: Number of cross-validation folds.
            scoring: Scoring metric name.

        Returns:
            List of dicts with column, transformation, score_with,
            score_without, impact, sorted by impact descending.
        """
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        if (
            not self.is_fitted_
            or self.recommendations_ is None
            or self.profile_ is None
        ):
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        full_pipe = build_pipeline(
            self.recommendations_,
            self.profile_,
            estimator=clone(estimator),
            pre_steps=list(self.pre_transforms) if self.pre_transforms else None,
            post_steps=list(self.post_transforms) if self.post_transforms else None,
        )
        full_scores = cross_val_score(full_pipe, X, y, cv=cv, scoring=scoring)
        full_mean = float(np.mean(full_scores))

        results: list[dict[str, Any]] = []
        for col_name, rules in self.recommendations_.items():
            if col_name == "__dataset__":
                continue
            for rule in rules:
                ablated_recs = {}
                for k, v in self.recommendations_.items():
                    if k == col_name:
                        ablated_recs[k] = [r for r in v if r.id != rule.id]
                        if not ablated_recs[k]:
                            continue
                    else:
                        ablated_recs[k] = v

                try:
                    ablated_pipe = build_pipeline(
                        ablated_recs,
                        self.profile_,
                        estimator=clone(estimator),
                        pre_steps=(
                            list(self.pre_transforms) if self.pre_transforms else None
                        ),
                        post_steps=(
                            list(self.post_transforms) if self.post_transforms else None
                        ),
                    )
                    ablated_scores = cross_val_score(
                        ablated_pipe, X, y, cv=cv, scoring=scoring
                    )
                    ablated_mean = float(np.mean(ablated_scores))
                except Exception:
                    ablated_mean = full_mean

                results.append(
                    {
                        "column": col_name,
                        "transformation": rule.transformation,
                        "score_with": full_mean,
                        "score_without": ablated_mean,
                        "impact": full_mean - ablated_mean,
                    }
                )

        results.sort(key=lambda x: x["impact"], reverse=True)
        return results

    def score_report(self) -> str:
        """Return a formatted string summarising recommendations and coverage.

        Includes:
        - Number of columns profiled
        - Number of transforms recommended
        - Algorithm family detected
        - Whether meta-learner or fallback was used

        Returns:
            Formatted string (plain text, no rich markup).
        """
        if (
            not self.is_fitted_
            or self.profile_ is None
            or self.recommendations_ is None
        ):
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        n_profiled = len(self.profile_.column_profiles)
        n_transforms = sum(
            len(rules)
            for col, rules in self.recommendations_.items()
            if col != "__dataset__"
        )
        af = self._resolve_algorithm_family()
        method = "meta-learner" if self._used_meta_learner else "ontology-only"

        lines = [
            "FeatureIQ Score Report",
            "=" * 40,
            f"Columns profiled: {n_profiled}",
            f"Transforms recommended: {n_transforms}",
            f"Algorithm family: {af.value}",
            f"Recommendation method: {method}",
            f"Problem type: {self._resolve_problem_type().value}",
            f"Dataset rows: {self.profile_.n_rows}",
            f"Dataset columns: {self.profile_.n_columns}",
        ]
        return "\n".join(lines)

    def explain(
        self,
        column: str | None = None,
    ) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
        """Return detailed explanations linking data profile to recommendations.

        Each explanation includes the column characteristics that triggered
        the rule, why the transform is appropriate for the algorithm family,
        and what it is contraindicated for.

        Args:
            column: If provided, return explanations for that column only.
                    If None, return explanations for all columns.

        Returns:
            If column is given: list of explanation dicts for that column.
            If column is None: dict mapping column name -> list of explanations.

        Raises:
            FeatureIQError: If called before fit().
        """
        if (
            not self.is_fitted_
            or self.recommendations_ is None
            or self.profile_ is None
        ):
            raise FeatureIQError("FeatureIQ has not been fitted. Call fit() first.")

        af = self._resolve_algorithm_family()
        pt = self._resolve_problem_type()

        if column is not None:
            rules = self.recommendations_.get(column, [])
            col_profile = self.profile_.column_profiles.get(column)
            return [
                self._build_explanation(column, rule, col_profile, af, pt)
                for rule in rules
            ]

        result: dict[str, list[dict[str, Any]]] = {}
        for col_name, rules in self.recommendations_.items():
            if col_name == "__dataset__":
                explanations = [
                    self._build_explanation(col_name, rule, None, af, pt)
                    for rule in rules
                ]
            else:
                col_profile = self.profile_.column_profiles.get(col_name)
                explanations = [
                    self._build_explanation(col_name, rule, col_profile, af, pt)
                    for rule in rules
                ]
            if explanations:
                result[col_name] = explanations
        return result

    def _build_explanation(
        self,
        column: str,
        rule: OntologyRule,
        col_profile: Any | None,
        algorithm_family: "AlgorithmFamily",
        problem_type: ProblemType,
    ) -> dict[str, Any]:
        """Build a detailed explanation dict for a single rule application."""
        parts: list[str] = []
        evidence: dict[str, Any] = {
            "algorithm_family": algorithm_family.value,
            "problem_type": problem_type.value,
            "rule_id": rule.id,
        }

        if col_profile is not None:
            evidence["column_type"] = col_profile.column_type.value

            cond = rule.conditions
            if cond.is_skewed is not None and col_profile.skewness is not None:
                evidence["skewness"] = col_profile.skewness
                evidence["is_skewed"] = col_profile.is_skewed
                direction = "+" if col_profile.skewness > 0 else ""
                skew_prefix = "skewed " if col_profile.is_skewed else ""
                parts.append(
                    f"{column} is a {skew_prefix}"
                    f"numerical feature (skewness={direction}"
                    f"{col_profile.skewness:.2f}, threshold={1.0})"
                )

            if cond.has_missing is not None:
                evidence["missing_rate"] = col_profile.missing_rate
                evidence["has_missing"] = col_profile.has_missing
                if col_profile.has_missing:
                    parts.append(
                        f"{column} has {col_profile.missing_rate:.1%} missing values"
                    )

            if (
                cond.cardinality_level is not None
                or cond.is_high_cardinality is not None
            ):
                evidence["n_unique"] = col_profile.n_unique
                evidence["unique_ratio"] = col_profile.unique_ratio
                evidence["cardinality_level"] = col_profile.cardinality_level
                evidence["is_high_cardinality"] = col_profile.is_high_cardinality
                if col_profile.cardinality_level in ("medium", "high"):
                    parts.append(
                        f"{column} has {col_profile.cardinality_level} cardinality "
                        f"({col_profile.n_unique} unique values)"
                    )

            if cond.is_monotonic is not None and col_profile.is_monotonic is not None:
                evidence["is_monotonic"] = col_profile.is_monotonic

            if (
                cond.has_regular_frequency is not None
                and col_profile.has_regular_frequency is not None
            ):
                evidence["has_regular_frequency"] = col_profile.has_regular_frequency

            if col_profile.outlier_fraction is not None:
                evidence["outlier_fraction"] = col_profile.outlier_fraction
        else:
            parts.append(f"Dataset-level rule for {problem_type.value}")

        parts.append(
            f"For {algorithm_family.value} models ({self.algorithm}), "
            f"applying {rule.transformation} {rule.description.lower()}"
        )

        if rule.contraindicated_for:
            contra_names = [af.value for af in rule.contraindicated_for]
            evidence["contraindicated_for"] = contra_names
            parts.append(
                f"This is contraindicated for {', '.join(contra_names)} models "
                f"which handle this natively"
            )

        return {
            "column": column,
            "transformation": rule.transformation,
            "confidence": rule.confidence,
            "explanation": ". ".join(parts) + ".",
            "evidence": evidence,
            "source": rule.source,
            "rule_id": rule.id,
        }

    def explain_report(self, column: str | None = None) -> None:
        """Print a rich-formatted explanation report.

        Args:
            column: If provided, show explanations for that column only.
        """
        explanations = self.explain(column)

        if isinstance(explanations, list):
            items = {column or "column": explanations}
        else:
            items = explanations

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            console.print()
            console.print("[bold cyan]FeatureIQ Explanation Report[/bold cyan]")
            console.print()

            for col_name, exps in items.items():
                table = Table(
                    title=f"Column: {col_name}",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Transform", style="green", width=25)
                table.add_column("Confidence", style="yellow", width=12)
                table.add_column("Explanation", style="white", width=60)
                table.add_column("Source", style="dim", width=30)

                for exp in exps:
                    table.add_row(
                        exp["transformation"],
                        f"{exp['confidence']:.2f}",
                        exp["explanation"],
                        exp["source"],
                    )
                console.print(table)
                console.print()
        except ImportError:
            for col_name, exps in items.items():
                logger.info("=" * 60)
                logger.info("Column: %s", col_name)
                logger.info("=" * 60)
                for exp in exps:
                    logger.info("  Transform: %s", exp["transformation"])
                    logger.info("  Confidence: %.2f", exp["confidence"])
                    logger.info("  Explanation: %s", exp["explanation"])
                    logger.info("  Source: %s", exp["source"])
                    logger.info("  ---")

    def _print_summary(self) -> None:
        """Print a rich-formatted summary of recommendations."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="FeatureIQ Recommendations")
            table.add_column("Column", style="cyan")
            table.add_column("Transformation", style="green")
            table.add_column("Confidence", style="yellow")

            if self.recommendations_:
                for col_name, rules in self.recommendations_.items():
                    if col_name == "__dataset__":
                        continue
                    for rule in rules[:2]:
                        table.add_row(
                            col_name,
                            rule.transformation,
                            f"{rule.confidence:.2f}",
                        )

            console.print(table)
        except ImportError:
            logger.info(self.score_report())
