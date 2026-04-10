"""Transformer registry mapping rule transformation names to sklearn transformers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from featureiq.exceptions import TransformerError
from featureiq.utils.validation import ColumnType

# ---------------------------------------------------------------------------
# Custom sklearn-compatible transformers
# ---------------------------------------------------------------------------


class RareLabelGrouper(BaseEstimator, TransformerMixin):
    """Replace rare categorical labels with 'Rare'.

    Args:
        tol: Minimum frequency threshold; labels below this are grouped.
        variables: Columns to transform. None means all object/category columns.
    """

    def __init__(
        self,
        tol: float = 0.05,
        variables: list[str] | None = None,
    ) -> None:
        self.tol = tol
        self.variables = variables
        self.frequent_labels_: dict[str, list[Any]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RareLabelGrouper":
        """Identify frequent labels per column.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.variables or [
            c
            for c in df.columns
            if df[c].dtype == object or hasattr(df[c].dtype, "categories")
        ]
        for col in cols:
            freq = df[col].value_counts(normalize=True)
            self.frequent_labels_[col] = freq[freq >= self.tol].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare labels with 'Rare'.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col, frequent in self.frequent_labels_.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x, f=frequent: x if x in f else "Rare")
        return df


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip values to IQR-based bounds.

    Args:
        factor: IQR multiplier for bounds.
        variables: Columns to clip. None means all numeric columns.
    """

    def __init__(
        self,
        factor: float = 1.5,
        variables: list[str] | None = None,
    ) -> None:
        self.factor = factor
        self.variables = variables
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IQRClipper":
        """Compute Q1, Q3, and IQR-based bounds per column.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            self.bounds_[col] = (
                q1 - self.factor * iqr,
                q3 + self.factor * iqr,
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip values to precomputed bounds.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df


class DateComponentExtractor(BaseEstimator, TransformerMixin):
    """Extract date components from datetime columns.

    Args:
        variables: Datetime columns to process. None means all datetime columns.
        components: Which components to extract.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        components: list[str] | None = None,
    ) -> None:
        self.variables = variables
        self.components = components or ["year", "month", "day", "weekday"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DateComponentExtractor":
        """No-op fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract date components and drop original datetime columns.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame with new component columns.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
        ]

        component_map = {
            "year": lambda s: s.dt.year,
            "month": lambda s: s.dt.month,
            "day": lambda s: s.dt.day,
            "weekday": lambda s: s.dt.weekday,
            "hour": lambda s: s.dt.hour,
        }

        for col in cols:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            for comp in self.components:
                if comp in component_map:
                    df[f"{col}_{comp}"] = component_map[comp](dt_series)
            df = df.drop(columns=[col])
        return df


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """Create lag features for time-series data.

    Args:
        variables: Columns to create lags for.
        lags: List of lag periods.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        lags: list[int] | None = None,
    ) -> None:
        self.variables = variables
        self.lags = lags or [1, 2, 3]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LagFeatureGenerator":
        """Store lag periods (no computation needed at fit time).

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag columns for each variable and lag period.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with additional lag columns.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df


class RollingStatsGenerator(BaseEstimator, TransformerMixin):
    """Create rolling mean and rolling std features.

    Args:
        variables: Columns to compute rolling stats for.
        windows: List of window sizes.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        windows: list[int] | None = None,
    ) -> None:
        self.variables = variables
        self.windows = windows or [3, 7]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RollingStatsGenerator":
        """Store window sizes (no computation needed at fit time).

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and rolling std columns.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with additional rolling stat columns.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            for w in self.windows:
                df[f"{col}_rolling_mean_{w}"] = df[col].rolling(window=w).mean()
                df[f"{col}_rolling_std_{w}"] = df[col].rolling(window=w).std()
        return df


class DifferencingTransformer(BaseEstimator, TransformerMixin):
    """First-order differencing for stationarity in time-series data.

    Args:
        variables: Columns to difference. None means all numeric columns.
        order: Differencing order (1 = first-order).
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        order: int = 1,
    ) -> None:
        self.variables = variables
        self.order = order
        self.first_values_: dict[str, list[float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DifferencingTransformer":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            self.first_values_[col] = df[col].iloc[: self.order].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col in self.first_values_:
            if col in df.columns:
                for _ in range(self.order):
                    df[col] = df[col].diff()
        return df


class FourierFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Generate sin/cos Fourier features for capturing seasonality.

    Args:
        variables: Columns to create Fourier features for.
        period: Seasonal period (e.g., 7 for weekly, 12 for monthly).
        n_harmonics: Number of Fourier harmonics to generate.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        period: int = 7,
        n_harmonics: int = 2,
    ) -> None:
        self.variables = variables
        self.period = period
        self.n_harmonics = n_harmonics

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FourierFeaturesTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            for k in range(1, self.n_harmonics + 1):
                df[f"{col}_sin_{k}"] = np.sin(2 * np.pi * k * df[col] / self.period)
                df[f"{col}_cos_{k}"] = np.cos(2 * np.pi * k * df[col] / self.period)
        return df


class TimeSinceReferenceTransformer(BaseEstimator, TransformerMixin):
    """Compute elapsed time since the first date in a datetime column.

    Args:
        variables: Datetime columns to process. None means all datetime columns.
        unit: Time unit for the output ('days', 'hours', 'seconds').
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        unit: str = "days",
    ) -> None:
        self.variables = variables
        self.unit = unit
        self.reference_dates_: dict[str, pd.Timestamp] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TimeSinceReferenceTransformer":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.variables or [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        for col in cols:
            dt_series = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dt_series) > 0:
                self.reference_dates_[col] = dt_series.min()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col, ref_date in self.reference_dates_.items():
            if col in df.columns:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                delta = dt_series - ref_date
                if self.unit == "hours":
                    df[f"{col}_hours_since_ref"] = delta.dt.total_seconds() / 3600
                elif self.unit == "seconds":
                    df[f"{col}_seconds_since_ref"] = delta.dt.total_seconds()
                else:
                    df[f"{col}_days_since_ref"] = delta.dt.total_seconds() / 86400
                df = df.drop(columns=[col])
        return df


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Binary encoding via OrdinalEncoder + binary expansion.

    Args:
        variables: Columns to encode. None means all object/category columns.
    """

    def __init__(self, variables: list[str] | None = None) -> None:
        self.variables = variables
        self.ordinal_encoder_: OrdinalEncoder | None = None
        self.n_bits_: dict[str, int] = {}
        self.cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BinaryEncoder":
        """Fit ordinal encoder and compute bit widths.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.cols_ = self.variables or [
            c
            for c in df.columns
            if df[c].dtype == object or hasattr(df[c].dtype, "categories")
        ]
        if not self.cols_:
            return self

        self.ordinal_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.ordinal_encoder_.fit(df[self.cols_].astype(str))

        for i, col in enumerate(self.cols_):
            n_categories = len(self.ordinal_encoder_.categories_[i])
            self.n_bits_[col] = max(1, int(np.ceil(np.log2(n_categories + 1))))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binary encoding.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with binary-encoded columns replacing originals.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        if not self.cols_ or self.ordinal_encoder_ is None:
            return df

        encoded = self.ordinal_encoder_.transform(df[self.cols_].astype(str))
        for i, col in enumerate(self.cols_):
            ordinal_vals = encoded[:, i].astype(int)
            ordinal_vals = np.clip(ordinal_vals, 0, None)
            n_bits = self.n_bits_[col]
            for bit in range(n_bits):
                df[f"{col}_bin_{bit}"] = (ordinal_vals >> bit) & 1
            df = df.drop(columns=[col])
        return df


# ---------------------------------------------------------------------------
# TransformerSpec and Registry
# ---------------------------------------------------------------------------


@dataclass
class TransformerSpec:
    """Specification for a transformer in the registry.

    Args:
        transformer_class: The sklearn-compatible transformer class.
        default_kwargs: Default keyword arguments for instantiation.
        applies_to: Column types this transformer can be applied to.
        description: Human-readable description.
    """

    transformer_class: type[BaseEstimator]
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    applies_to: list[ColumnType] = field(default_factory=list)
    description: str = ""


def _make_target_encoder_spec() -> TransformerSpec:
    """Create the target encoder spec, using sklearn >= 1.3 TargetEncoder."""
    try:
        from sklearn.preprocessing import TargetEncoder
    except ImportError:
        return TransformerSpec(
            transformer_class=OrdinalEncoder,
            default_kwargs={"handle_unknown": "use_encoded_value", "unknown_value": -1},
            applies_to=[ColumnType.CATEGORICAL],
            description="Target encoding (fallback to ordinal, sklearn < 1.3)",
        )

    return TransformerSpec(
        transformer_class=TargetEncoder,
        default_kwargs={},
        applies_to=[ColumnType.CATEGORICAL],
        description="Target encoding for high-cardinality categorical features",
    )


TRANSFORMER_REGISTRY: dict[str, TransformerSpec] = {
    "log_transform": TransformerSpec(
        transformer_class=FunctionTransformer,
        default_kwargs={"func": np.log1p, "validate": True},
        applies_to=[ColumnType.NUMERICAL],
        description="Natural log transform (log1p) for skewed features",
    ),
    "standard_scaler": TransformerSpec(
        transformer_class=StandardScaler,
        default_kwargs={},
        applies_to=[ColumnType.NUMERICAL],
        description=(
            "Standardize features by removing the mean and " "scaling to unit variance"
        ),
    ),
    "min_max_scaler": TransformerSpec(
        transformer_class=MinMaxScaler,
        default_kwargs={},
        applies_to=[ColumnType.NUMERICAL],
        description="Scale features to a [0, 1] range",
    ),
    "one_hot_encoder": TransformerSpec(
        transformer_class=OneHotEncoder,
        default_kwargs={"sparse_output": False, "handle_unknown": "ignore"},
        applies_to=[ColumnType.CATEGORICAL],
        description="One-hot encode categorical features",
    ),
    "ordinal_encoder": TransformerSpec(
        transformer_class=OrdinalEncoder,
        default_kwargs={"handle_unknown": "use_encoded_value", "unknown_value": -1},
        applies_to=[ColumnType.CATEGORICAL],
        description="Ordinal encode categorical features",
    ),
    "target_encoder": _make_target_encoder_spec(),
    "binary_encoder": TransformerSpec(
        transformer_class=BinaryEncoder,
        default_kwargs={},
        applies_to=[ColumnType.CATEGORICAL],
        description="Binary encoding via ordinal + binary expansion",
    ),
    "mean_imputer": TransformerSpec(
        transformer_class=SimpleImputer,
        default_kwargs={"strategy": "mean"},
        applies_to=[ColumnType.NUMERICAL],
        description="Impute missing values with column mean",
    ),
    "median_imputer": TransformerSpec(
        transformer_class=SimpleImputer,
        default_kwargs={"strategy": "median"},
        applies_to=[ColumnType.NUMERICAL],
        description="Impute missing values with column median",
    ),
    "mode_imputer": TransformerSpec(
        transformer_class=SimpleImputer,
        default_kwargs={"strategy": "most_frequent"},
        applies_to=[ColumnType.CATEGORICAL],
        description="Impute missing values with mode",
    ),
    "rare_label_grouper": TransformerSpec(
        transformer_class=RareLabelGrouper,
        default_kwargs={"tol": 0.05},
        applies_to=[ColumnType.CATEGORICAL],
        description="Group rare categorical labels into 'Rare'",
    ),
    "iqr_clipper": TransformerSpec(
        transformer_class=IQRClipper,
        default_kwargs={"factor": 1.5},
        applies_to=[ColumnType.NUMERICAL],
        description="Clip outliers to IQR-based bounds",
    ),
    "polynomial_features": TransformerSpec(
        transformer_class=PolynomialFeatures,
        default_kwargs={"degree": 2, "include_bias": False, "interaction_only": False},
        applies_to=[ColumnType.NUMERICAL],
        description="Generate polynomial features (degree 2)",
    ),
    "date_component_extractor": TransformerSpec(
        transformer_class=DateComponentExtractor,
        default_kwargs={"components": ["year", "month", "day", "weekday"]},
        applies_to=[ColumnType.DATETIME],
        description="Extract date components from datetime columns",
    ),
    "lag_feature_generator": TransformerSpec(
        transformer_class=LagFeatureGenerator,
        default_kwargs={"lags": [1, 2, 3]},
        applies_to=[ColumnType.NUMERICAL],
        description="Generate lag features for time-series data",
    ),
    "rolling_stats_generator": TransformerSpec(
        transformer_class=RollingStatsGenerator,
        default_kwargs={"windows": [3, 7]},
        applies_to=[ColumnType.NUMERICAL],
        description="Generate rolling mean and std features",
    ),
    "robust_scaler": TransformerSpec(
        transformer_class=RobustScaler,
        default_kwargs={},
        applies_to=[ColumnType.NUMERICAL],
        description="Scale using median and IQR, robust to outliers",
    ),
    "differencing": TransformerSpec(
        transformer_class=DifferencingTransformer,
        default_kwargs={"order": 1},
        applies_to=[ColumnType.NUMERICAL],
        description="First-order differencing for stationarity",
    ),
    "fourier_features": TransformerSpec(
        transformer_class=FourierFeaturesTransformer,
        default_kwargs={"period": 7, "n_harmonics": 2},
        applies_to=[ColumnType.NUMERICAL],
        description="Sin/cos Fourier features for seasonality",
    ),
    "time_since_reference": TransformerSpec(
        transformer_class=TimeSinceReferenceTransformer,
        default_kwargs={"unit": "days"},
        applies_to=[ColumnType.DATETIME],
        description="Days/hours since the first date in the column",
    ),
}


def get_transformer(name: str) -> TransformerSpec:
    """Retrieve a TransformerSpec by name.

    Args:
        name: Transformation name matching an ontology rule's transformation field.

    Returns:
        TransformerSpec instance.

    Raises:
        TransformerError: If name not found in registry.
    """
    if name not in TRANSFORMER_REGISTRY:
        raise TransformerError(
            f"Transformation '{name}' not found in registry. "
            f"Available: {list(TRANSFORMER_REGISTRY.keys())}"
        )
    return TRANSFORMER_REGISTRY[name]
