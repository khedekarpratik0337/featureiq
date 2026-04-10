"""Build sklearn Pipelines from ontology recommendations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from featureiq.ontology.rule_loader import OntologyRule
from featureiq.profiler.dataset_profiler import DatasetProfile
from featureiq.transformer.registry import get_transformer

_DATE_EXTRACTION_TRANSFORMS = frozenset(
    {
        "date_component_extractor",
        "time_since_reference",
    }
)

_IMPUTATION_TRANSFORMS = frozenset(
    {
        "mean_imputer",
        "median_imputer",
        "mode_imputer",
    }
)

_CLIPPING_TRANSFORMS = frozenset(
    {
        "iqr_clipper",
    }
)

_NONLINEAR_TRANSFORMS = frozenset(
    {
        "log_transform",
        "differencing",
    }
)

_SCALING_TRANSFORMS = frozenset(
    {
        "standard_scaler",
        "min_max_scaler",
        "robust_scaler",
    }
)

_ENCODING_TRANSFORMS = frozenset(
    {
        "one_hot_encoder",
        "ordinal_encoder",
        "target_encoder",
        "binary_encoder",
        "rare_label_grouper",
    }
)

_EXPANSION_TRANSFORMS = frozenset(
    {
        "polynomial_features",
        "fourier_features",
    }
)

_TEMPORAL_TRANSFORMS = frozenset(
    {
        "lag_feature_generator",
        "rolling_stats_generator",
    }
)

_SKIP_IN_COLUMN_TRANSFORMER = frozenset(
    {
        "class_weight_balancing",
        "stratified_sampling",
        "contamination_calibration",
        "lag_feature_generator",
        "rolling_stats_generator",
        "differencing",
        "fourier_features",
    }
)

_POST_CT_TRANSFORMS = frozenset(
    {
        "lag_feature_generator",
        "rolling_stats_generator",
        "differencing",
        "fourier_features",
    }
)

_STEP_ORDER = [
    _DATE_EXTRACTION_TRANSFORMS,
    _IMPUTATION_TRANSFORMS,
    _CLIPPING_TRANSFORMS,
    _NONLINEAR_TRANSFORMS,
    _SCALING_TRANSFORMS,
    _ENCODING_TRANSFORMS,
    _EXPANSION_TRANSFORMS,
    _TEMPORAL_TRANSFORMS,
]


def _step_priority(transformation: str) -> int:
    """Return ordering priority for a transformation (lower = earlier)."""
    for i, group in enumerate(_STEP_ORDER):
        if transformation in group:
            return i
    return len(_STEP_ORDER)


def _prioritise_rules(
    rules: list[OntologyRule],
) -> list[OntologyRule]:
    """Order rules: imputation -> clipping -> nonlinear -> scaling ->
    encoding -> expansion.

    Deduplicates by transformation name, keeping highest confidence.
    """
    seen: set[str] = set()
    deduped: list[OntologyRule] = []
    for rule in sorted(rules, key=lambda r: r.confidence, reverse=True):
        if rule.transformation not in seen:
            seen.add(rule.transformation)
            deduped.append(rule)

    return sorted(deduped, key=lambda r: _step_priority(r.transformation))


def _has_imputation_step(steps: list[tuple[str, Any]]) -> bool:
    """Check whether pipeline steps already contain an imputation transform."""
    for step_name, _ in steps:
        if step_name in _IMPUTATION_TRANSFORMS:
            return True
    return False


def build_column_transformer(
    recommendations: dict[str, list[OntologyRule]],
    dataset_profile: DatasetProfile,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer from ontology recommendations.

    For each column in recommendations (excluding "__dataset__" key):
        - Take the highest-confidence applicable rule
        - Look up the corresponding transformer in TRANSFORMER_REGISTRY
        - Group columns by transformer type
        - Build a ColumnTransformer

    NaN safety: any numerical column with raw_has_missing=True that has
    transforms but no imputation step gets a safety median imputer prepended.

    Deduplication logic:
        - If multiple rules recommend transforms for the same column,
          apply them in order: imputation first, then encoding/scaling
        - Use a Pipeline per column group where multiple steps are needed

    Args:
        recommendations: Output of OntologyEngine.recommend_for_dataset().
        dataset_profile: DatasetProfile for column type lookup.

    Returns:
        Configured ColumnTransformer (not yet fitted).

    Raises:
        TransformerError: If a recommended transformation is not in registry.
    """
    from featureiq.utils.validation import ColumnType as CT

    col_step_map: dict[str, list[tuple[str, Any]]] = defaultdict(list)

    for col_name, rules in recommendations.items():
        if col_name == "__dataset__":
            continue

        ordered_rules = _prioritise_rules(rules)
        for rule in ordered_rules:
            if rule.transformation in _SKIP_IN_COLUMN_TRANSFORMER:
                continue

            spec = get_transformer(rule.transformation)
            transformer = spec.transformer_class(**spec.default_kwargs)
            step_name = f"{rule.transformation}"
            col_step_map[col_name].append((step_name, transformer))

    for col_name, steps in col_step_map.items():
        cp = dataset_profile.column_profiles.get(col_name)
        if cp is None:
            continue
        if (
            cp.raw_has_missing
            and cp.column_type == CT.NUMERICAL
            and not _has_imputation_step(steps)
        ):
            from sklearn.impute import SimpleImputer

            safety_imputer = SimpleImputer(strategy="median")
            steps.insert(0, ("_safety_median_imputer", safety_imputer))
            col_step_map[col_name] = steps

    group_key_counter: dict[str, int] = defaultdict(int)
    transformers_list: list[tuple[str, Any, list[str]]] = []

    columns_by_steps: dict[str, list[str]] = defaultdict(list)
    steps_by_key: dict[str, Any] = {}

    for col_name, steps in col_step_map.items():
        if len(steps) == 1:
            step_name, transformer = steps[0]
            key = step_name
            columns_by_steps[key].append(col_name)
            steps_by_key[key] = transformer
        else:
            pipe = Pipeline(steps)
            unique_key = f"pipe_{'_'.join(s[0] for s in steps)}_{col_name}"
            transformers_list.append((unique_key, pipe, [col_name]))

    for key, cols in columns_by_steps.items():
        transformer = steps_by_key[key]
        group_key_counter[key] += 1
        name = key if group_key_counter[key] == 1 else f"{key}_{group_key_counter[key]}"
        transformers_list.append((name, transformer, cols))

    if not transformers_list:
        return ColumnTransformer(
            transformers=[],
            remainder="passthrough",
        )

    return ColumnTransformer(
        transformers=transformers_list,
        remainder="passthrough",
    )


def _collect_post_ct_transforms(
    recommendations: dict[str, list[OntologyRule]],
) -> list[tuple[str, Any]]:
    """Collect transforms after ColumnTransformer (row-wise temporal ops)."""
    seen: set[str] = set()
    post_steps: list[tuple[str, Any]] = []

    for col_name, rules in recommendations.items():
        if col_name == "__dataset__":
            continue
        for rule in rules:
            if (
                rule.transformation in _POST_CT_TRANSFORMS
                and rule.transformation not in seen
            ):
                seen.add(rule.transformation)
                spec = get_transformer(rule.transformation)
                transformer = spec.transformer_class(**spec.default_kwargs)
                post_steps.append((rule.transformation, transformer))

    return sorted(post_steps, key=lambda s: _step_priority(s[0]))


def build_pipeline(
    recommendations: dict[str, list[OntologyRule]],
    dataset_profile: DatasetProfile,
    estimator: Any | None = None,
    pre_steps: list[tuple[str, Any]] | None = None,
    post_steps: list[tuple[str, Any]] | None = None,
) -> Pipeline:
    """Build a full sklearn Pipeline including the optional final estimator.

    Steps:
    1. Optional pre-steps (user-injected transforms before ColumnTransformer)
    2. ColumnTransformer for feature engineering
    3. Post-CT temporal transforms (lag, rolling, differencing, fourier)
    4. Optional post-steps (user-injected transforms after ColumnTransformer)
    5. (Optional) Final estimator if provided

    Args:
        recommendations: Output of OntologyEngine.recommend_for_dataset().
        dataset_profile: DatasetProfile for column type lookup.
        estimator: Optional sklearn-compatible estimator to append.
        pre_steps: Optional transforms to insert before the ColumnTransformer.
        post_steps: Optional transforms to insert after the ColumnTransformer.

    Returns:
        sklearn Pipeline instance (not yet fitted).
    """
    steps: list[tuple[str, Any]] = []

    if pre_steps:
        steps.extend(pre_steps)

    ct = build_column_transformer(recommendations, dataset_profile)
    steps.append(("feature_engineering", ct))

    post_ct = _collect_post_ct_transforms(recommendations)
    for step_name, transformer in post_ct:
        steps.append((step_name, transformer))

    if post_steps:
        steps.extend(post_steps)

    if estimator is not None:
        steps.append(("estimator", estimator))

    return Pipeline(steps)
