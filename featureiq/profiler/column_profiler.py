"""Column-level profiling for individual pandas Series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from featureiq.exceptions import ProfilingError
from featureiq.utils.validation import (
    HIGH_CARDINALITY_THRESHOLD,
    LOW_CARDINALITY_MAX,
    MEDIUM_CARDINALITY_MAX,
    MISSING_THRESHOLD,
    OUTLIER_IQR_FACTOR,
    SKEWNESS_THRESHOLD,
    ColumnType,
)


class ColumnProfile(BaseModel):
    """Structured profile of a single DataFrame column."""

    name: str
    column_type: ColumnType
    n_unique: int
    unique_ratio: float
    missing_rate: float
    raw_has_missing: bool = False
    cardinality_level: str = "low"
    has_missing: bool

    mean: float | None = None
    std: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    is_skewed: bool | None = None
    outlier_fraction: float | None = None
    is_normal: bool | None = None

    top_frequency: float | None = None

    is_monotonic: bool | None = None
    has_regular_frequency: bool | None = None

    correlated_with: list[str] | None = None
    is_highly_correlated: bool = False
    has_high_vif: bool = False

    @property
    def is_high_cardinality(self) -> bool:
        return self.cardinality_level == "high"


def detect_column_type(series: pd.Series) -> ColumnType:
    """Detect the semantic type of a pandas Series.

    Detection order:
    1. If dtype is bool -> BOOLEAN
    2. If dtype is datetime or parseable as datetime -> DATETIME
    3. If dtype is object or category -> CATEGORICAL
    4. If dtype is numeric -> NUMERICAL
    5. Otherwise -> UNKNOWN

    Args:
        series: Input pandas Series.

    Returns:
        Detected ColumnType enum value.
    """
    if series.dropna().empty:
        return ColumnType.UNKNOWN

    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    if pd.api.types.is_object_dtype(series) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        non_null = series.dropna()
        if len(non_null) > 0:
            try:
                pd.to_datetime(non_null, format="mixed")
                return ColumnType.DATETIME
            except (ValueError, TypeError, OverflowError):
                pass
        return ColumnType.CATEGORICAL

    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERICAL

    return ColumnType.UNKNOWN


def profile_numerical_column(series: pd.Series) -> dict:
    """Compute numerical meta-features for a Series.

    Must compute: mean, std, skewness, kurtosis, is_skewed,
    outlier_fraction, is_normal.

    Shapiro-Wilk is only run if series.dropna().shape[0] <= 5000.
    For larger series, is_normal = None.

    Args:
        series: Numeric pandas Series.

    Returns:
        Dict of numerical meta-features.

    Raises:
        ProfilingError: If series has fewer than 3 non-null values.
    """
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        raise ProfilingError(
            f"Series '{series.name}' has fewer than 3 non-null values "
            f"({len(clean)}), cannot compute numerical statistics."
        )

    mean_val = float(clean.mean())
    std_val = float(clean.std())
    skewness_val = float(stats.skew(clean, bias=False))
    kurtosis_val = float(stats.kurtosis(clean, bias=False))
    is_skewed = abs(skewness_val) > SKEWNESS_THRESHOLD

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - OUTLIER_IQR_FACTOR * iqr
    upper_bound = q3 + OUTLIER_IQR_FACTOR * iqr
    outlier_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
    outlier_fraction = outlier_count / len(clean)

    is_normal: bool | None = None
    if len(clean) <= 5000:
        _, p_value = stats.shapiro(clean)
        is_normal = p_value > 0.05

    return {
        "mean": mean_val,
        "std": std_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "is_skewed": is_skewed,
        "outlier_fraction": outlier_fraction,
        "is_normal": is_normal,
    }


def profile_categorical_column(series: pd.Series) -> dict:
    """Compute categorical meta-features for a Series.

    Must compute: top_frequency.

    Args:
        series: Categorical pandas Series.

    Returns:
        Dict of categorical meta-features.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"top_frequency": 0.0}
    value_counts = non_null.value_counts()
    top_frequency = float(value_counts.iloc[0] / len(non_null))
    return {"top_frequency": top_frequency}


def profile_datetime_column(series: pd.Series) -> dict:
    """Compute datetime meta-features for a Series.

    Must compute: is_monotonic, has_regular_frequency.
    has_regular_frequency: True if the median time delta
    equals the mode time delta (i.e., gaps are consistent).

    Args:
        series: Datetime pandas Series.

    Returns:
        Dict of datetime meta-features.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return {"is_monotonic": True, "has_regular_frequency": True}

    clean_sorted = clean.sort_values()
    is_monotonic = bool(clean.is_monotonic_increasing or clean.is_monotonic_decreasing)

    diffs = clean_sorted.diff().dropna()
    if len(diffs) == 0:  # pragma: no cover – defensive guard; unreachable when len(clean) >= 2
        return {"is_monotonic": is_monotonic, "has_regular_frequency": True}

    median_delta = diffs.median()
    mode_result = diffs.mode()
    if len(mode_result) == 0:  # pragma: no cover – defensive guard; mode() on non-empty always returns ≥1
        has_regular_frequency = True
    else:
        mode_delta = mode_result.iloc[0]
        has_regular_frequency = median_delta == mode_delta

    return {
        "is_monotonic": is_monotonic,
        "has_regular_frequency": bool(has_regular_frequency),
    }


def profile_column(series: pd.Series, name: str) -> ColumnProfile:
    """Profile a single column and return a ColumnProfile.

    This is the main entry point for column profiling.
    Calls detect_column_type, then the appropriate profiler.

    Args:
        series: Input pandas Series.
        name: Column name.

    Returns:
        ColumnProfile instance.

    Raises:
        ProfilingError: If profiling fails for any reason.
    """
    if series.empty:
        raise ProfilingError(f"Column '{name}' is empty, cannot profile.")

    if series.isna().all():
        raise ProfilingError(
            f"Column '{name}' contains only null values, cannot profile."
        )

    col_type = detect_column_type(series)
    n_total = len(series)
    n_unique = int(series.nunique(dropna=True))
    unique_ratio = n_unique / n_total if n_total > 0 else 0.0
    missing_rate = float(series.isna().mean())
    raw_has_missing = missing_rate > 0
    has_missing = missing_rate > MISSING_THRESHOLD

    if n_unique <= LOW_CARDINALITY_MAX:
        cardinality_level = "low"
    elif n_unique <= MEDIUM_CARDINALITY_MAX:
        cardinality_level = "medium"
    else:
        cardinality_level = "high"

    extra: dict = {}
    try:
        if col_type == ColumnType.NUMERICAL:
            extra = profile_numerical_column(series)
        elif col_type == ColumnType.CATEGORICAL:
            extra = profile_categorical_column(series)
        elif col_type == ColumnType.DATETIME:
            dt_series = pd.to_datetime(series, errors="coerce")
            extra = profile_datetime_column(dt_series)
    except ProfilingError:
        raise
    except Exception as exc:
        raise ProfilingError(
            f"Failed to profile column '{name}': {exc}"
        ) from exc

    return ColumnProfile(
        name=name,
        column_type=col_type,
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        missing_rate=missing_rate,
        raw_has_missing=raw_has_missing,
        cardinality_level=cardinality_level,
        has_missing=has_missing,
        **extra,
    )
