"""Dataset-level profiling combining column profiles with dataset meta-features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel

from featureiq.exceptions import InsufficientDataError, ProfilingError
from featureiq.profiler.column_profiler import ColumnProfile, profile_column
from featureiq.utils.validation import (
    MIN_ROWS_FOR_PROFILING,
    ColumnType,
    ProblemType,
)

logger = logging.getLogger(__name__)

CORRELATION_THRESHOLD = 0.85


class DatasetProfile(BaseModel):
    """Structured profile of an entire dataset."""

    n_rows: int
    n_columns: int
    n_numerical: int
    n_categorical: int
    n_datetime: int
    n_boolean: int
    n_text: int
    n_unknown: int
    overall_missing_rate: float
    class_imbalance_ratio: float | None
    feature_to_row_ratio: float
    has_temporal_structure: bool
    column_profiles: dict[str, ColumnProfile]
    detected_frequency: str | None = None
    n_temporal_columns: int = 0
    correlation_matrix: dict[str, dict[str, float]] | None = None
    highly_correlated_pairs: list[tuple[str, str, float]] | None = None
    vif_scores: dict[str, float] | None = None


def compute_class_imbalance(
    y: pd.Series,
    problem_type: ProblemType,
) -> float | None:
    """Compute class imbalance ratio for classification problems.

    Ratio = n_minority_class / n_majority_class.
    Returns None for regression and forecasting.
    Returns 1.0 for balanced classes.

    Args:
        y: Target series.
        problem_type: Declared problem type.

    Returns:
        Imbalance ratio or None.
    """
    if problem_type in (
        ProblemType.REGRESSION,
        ProblemType.TIME_SERIES_FORECASTING,
        ProblemType.ANOMALY_DETECTION,
    ):
        return None

    value_counts = y.value_counts()
    if len(value_counts) < 2:
        return 1.0

    minority = int(value_counts.min())
    majority = int(value_counts.max())
    if majority == 0:  # pragma: no cover – unreachable when len(value_counts) >= 2
        return 1.0

    return minority / majority


def _infer_frequency(
    X: pd.DataFrame, column_profiles: dict[str, ColumnProfile]
) -> str | None:
    """Infer temporal frequency from the first datetime column found."""
    for col_name, cp in column_profiles.items():
        if cp.column_type != ColumnType.DATETIME:
            continue
        try:
            dt_col = pd.to_datetime(X[col_name], errors="coerce").dropna().sort_values()
            if len(dt_col) < 3:
                continue
            freq = pd.infer_freq(dt_col)
            if freq is not None:
                return freq
        except Exception:
            continue
    return None


def _compute_correlation_matrix(
    X: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
) -> tuple[
    dict[str, dict[str, float]],
    list[tuple[str, str, float]],
]:
    """Compute pairwise Pearson correlation for numerical columns."""
    num_cols = [
        name
        for name, cp in column_profiles.items()
        if cp.column_type == ColumnType.NUMERICAL and name in X.columns
    ]
    if len(num_cols) < 2:
        return {}, []

    corr_df = X[num_cols].corr(method="pearson")
    corr_dict: dict[str, dict[str, float]] = {}
    for col in num_cols:
        corr_dict[col] = {c: float(corr_df.loc[col, c]) for c in num_cols}

    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1 :]:
            r = abs(float(corr_df.loc[c1, c2]))
            if r > CORRELATION_THRESHOLD:
                pair = (c1, c2) if c1 < c2 else (c2, c1)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append((c1, c2, float(corr_df.loc[c1, c2])))

    return corr_dict, pairs


def _compute_vif_scores(
    X: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
) -> dict[str, float]:
    """Compute Variance Inflation Factor for numerical columns using OLS."""
    num_cols = [
        name
        for name, cp in column_profiles.items()
        if cp.column_type == ColumnType.NUMERICAL and name in X.columns
    ]
    if len(num_cols) < 2:
        return {}

    data = X[num_cols].dropna()
    if len(data) < len(num_cols) + 1:
        return {}

    vif_scores: dict[str, float] = {}
    X_mat = data.values.astype(np.float64)
    for i, col in enumerate(num_cols):
        try:
            y_col = X_mat[:, i]
            X_others = np.delete(X_mat, i, axis=1)
            X_others = np.column_stack([np.ones(len(X_others)), X_others])
            coeffs, residuals, _, _ = np.linalg.lstsq(X_others, y_col, rcond=None)
            y_pred = X_others @ coeffs
            ss_res = np.sum((y_col - y_pred) ** 2)
            ss_tot = np.sum((y_col - y_col.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
            vif_scores[col] = round(float(vif), 4)
        except Exception:
            vif_scores[col] = float("nan")

    return vif_scores


def _annotate_cross_column(
    X: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
) -> None:
    """Annotate column profiles with cross-column awareness fields."""
    corr_dict, pairs = _compute_correlation_matrix(X, column_profiles)
    vif_scores = _compute_vif_scores(X, column_profiles)

    correlated_map: dict[str, list[str]] = {}
    for c1, c2, _ in pairs:
        correlated_map.setdefault(c1, []).append(c2)
        correlated_map.setdefault(c2, []).append(c1)

    for name, cp in column_profiles.items():
        if name in correlated_map:
            cp.correlated_with = correlated_map[name]
            cp.is_highly_correlated = True
        if name in vif_scores and vif_scores[name] > 10.0:
            cp.has_high_vif = True

    return corr_dict, pairs, vif_scores  # type: ignore[return-value]


def profile_dataset(
    X: pd.DataFrame,
    y: pd.Series | None,
    problem_type: ProblemType,
) -> DatasetProfile:
    """Profile a full dataset and return a DatasetProfile.

    Calls profile_column for each column in X.
    Computes dataset-level meta-features.

    Args:
        X: Feature DataFrame.
        y: Target Series (None for unsupervised anomaly detection).
        problem_type: Declared problem type.

    Returns:
        DatasetProfile instance.

    Raises:
        InsufficientDataError: If X has fewer than MIN_ROWS_FOR_PROFILING rows.
        ProfilingError: If profiling fails for any column.
    """
    if len(X) < MIN_ROWS_FOR_PROFILING:
        raise InsufficientDataError(
            f"Dataset has {len(X)} rows, minimum required is "
            f"{MIN_ROWS_FOR_PROFILING}."
        )

    column_profiles: dict[str, ColumnProfile] = {}
    for col_name in X.columns:
        try:
            column_profiles[str(col_name)] = profile_column(X[col_name], str(col_name))
        except ProfilingError:
            raise
        except Exception as exc:
            raise ProfilingError(
                f"Failed to profile column '{col_name}': {exc}"
            ) from exc

    type_counts = {ct: 0 for ct in ColumnType}
    for cp in column_profiles.values():
        type_counts[cp.column_type] += 1

    n_rows = len(X)
    n_columns = len(X.columns)
    overall_missing_rate = float(X.isna().mean().mean())
    class_imbalance_ratio = (
        compute_class_imbalance(y, problem_type) if y is not None else None
    )
    feature_to_row_ratio = n_columns / n_rows if n_rows > 0 else 0.0
    n_temporal = type_counts[ColumnType.DATETIME]
    has_temporal_structure = n_temporal > 0
    detected_frequency = _infer_frequency(X, column_profiles)

    corr_dict, pairs, vif_scores = _annotate_cross_column(X, column_profiles)

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        n_numerical=type_counts[ColumnType.NUMERICAL],
        n_categorical=type_counts[ColumnType.CATEGORICAL],
        n_datetime=type_counts[ColumnType.DATETIME],
        n_boolean=type_counts[ColumnType.BOOLEAN],
        n_text=type_counts[ColumnType.TEXT],
        n_unknown=type_counts[ColumnType.UNKNOWN],
        overall_missing_rate=overall_missing_rate,
        class_imbalance_ratio=class_imbalance_ratio,
        feature_to_row_ratio=feature_to_row_ratio,
        has_temporal_structure=has_temporal_structure,
        column_profiles=column_profiles,
        detected_frequency=detected_frequency,
        n_temporal_columns=n_temporal,
        correlation_matrix=corr_dict if corr_dict else None,
        highly_correlated_pairs=pairs if pairs else None,
        vif_scores=vif_scores if vif_scores else None,
    )
