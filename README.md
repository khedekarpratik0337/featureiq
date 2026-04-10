# FeatureIQ

[![CI](https://github.com/khedekarpratik0337/featureiq/actions/workflows/ci.yml/badge.svg)](https://github.com/khedekarpratik0337/featureiq/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/featureiq.svg)](https://pypi.org/project/featureiq/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/pypi/pyversions/featureiq.svg)](https://pypi.org/project/featureiq/)

**A knowledge graph and meta-learning framework for context-aware automated feature engineering.**

## The Problem

Feature engineering is the most impactful yet most time-consuming step in building machine learning models. Practitioners repeatedly make the same decisions -- log-transforming skewed features, one-hot encoding categoricals, scaling for distance-based models -- but these decisions are rarely captured in a reusable, systematic way. The right transformations depend on three interacting factors: the data characteristics, the problem type, and the downstream algorithm. Getting this wrong wastes time and hurts model performance.

FeatureIQ solves this by encoding feature engineering expertise into a structured ontology of rules, each backed by real citations from ML literature. It profiles your dataset, matches applicable rules based on column statistics and your declared algorithm, and builds an sklearn-compatible transformation pipeline automatically. For advanced users, a meta-learning layer trained on OpenML benchmarks can re-rank recommendations based on what historically worked for similar datasets.

## Installation

```bash
pip install featureiq
```

With meta-learning support (LightGBM + OpenML):

```bash
pip install featureiq[meta]
```

Or with Poetry:

```bash
poetry add featureiq
poetry add featureiq -E meta  # with meta-learning
```

## Quick Start

```python
import pandas as pd
from sklearn.datasets import load_iris
from featureiq import FeatureIQ

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

fiq = FeatureIQ(
    problem_type="multiclass_classification",
    algorithm="random_forest",
)
X_transformed = fiq.fit_transform(X, y)

# Inspect recommendations
print(fiq.score_report())
for col, recs in fiq.recommend().items():
    for rec in recs:
        print(f"  {col}: {rec['transformation']} (confidence: {rec['confidence']})")
```

FeatureIQ works inside sklearn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("features", FeatureIQ(problem_type="binary_classification", algorithm="logistic_regression")),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

## Key Features

### Composability

Control exactly what FeatureIQ does:

```python
fiq = FeatureIQ(
    problem_type="regression",
    algorithm="ridge",
    exclude_columns=["id", "timestamp"],           # pass through untouched
    override_transforms={"price": ["log_transform", "standard_scaler"]},  # your rules
    pre_transforms=[("custom_step", MyTransformer())],   # before FeatureIQ
    post_transforms=[("pca", PCA(n_components=10))],     # after FeatureIQ
)
```

### Validation and Ablation

Verify that transforms actually help:

```python
from sklearn.linear_model import Ridge

fiq.fit(X_train, y_train)
results = fiq.validate(X_train, y_train, estimator=Ridge(), cv=5, scoring="r2")
print(f"FeatureIQ: {results['featureiq_score']:.3f}  Baseline: {results['baseline_score']:.3f}")

# Per-transform impact analysis
ablation = fiq.ablate(X_train, y_train, estimator=Ridge(), cv=3)
for item in ablation[:5]:
    print(f"  {item['column']}/{item['transformation']}: impact={item['impact']:.4f}")
```

### Cross-Column Awareness

FeatureIQ detects multicollinearity, computes VIF scores, and recommends accordingly for linear models -- while correctly skipping these checks for tree-based models that handle correlation natively.

### Explainability

Every recommendation cites a real source:

```python
fiq.explain_report()  # rich-formatted table with evidence and citations
```

## How It Works

FeatureIQ is built on three pillars:

1. **Data Profiling** -- Automatically analyzes every column: distribution shape, missing rates, cardinality, outliers, temporal structure, and cross-column correlations.

2. **Ontology Rules** -- A curated knowledge graph of feature engineering rules, each specifying when a transformation should (or should not) be applied. Rules are written in YAML, cite real ML literature, and can be extended by anyone.

3. **Meta-Learning** (optional) -- A LightGBM model trained on OpenML benchmarks that predicts which transformations will improve performance for datasets similar to yours. Falls back gracefully to ontology-only mode when no trained model is available.

## Supported Algorithms

| Algorithm | Family |
|-----------|--------|
| XGBoost | Tree-based |
| LightGBM | Tree-based |
| Random Forest | Tree-based |
| Decision Tree | Tree-based |
| Gradient Boosting | Tree-based |
| Logistic Regression | Linear Model |
| Linear Regression | Linear Model |
| Ridge | Linear Model |
| Lasso | Linear Model |
| SVM | Distance-based |
| KNN | Distance-based |
| MLP | Neural Network |
| AdaBoost | Ensemble |
| Bagging | Ensemble |

## Supported Problem Types

| Problem Type | Value |
|-------------|-------|
| Binary Classification | `binary_classification` |
| Multiclass Classification | `multiclass_classification` |
| Regression | `regression` |
| Time Series Forecasting | `time_series_forecasting` |
| Anomaly Detection | `anomaly_detection` |

## API Overview

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Profile data, generate recommendations, build pipeline |
| `transform(X)` | Apply fitted pipeline to new data |
| `recommend()` | Return human-readable recommendations dict |
| `validate(X, y, estimator)` | Cross-validate FeatureIQ pipeline vs baseline |
| `ablate(X, y, estimator)` | Per-transform ablation analysis |
| `explain(column=None)` | Detailed explanations with evidence |
| `explain_report()` | Rich-formatted explanation table |
| `score_report()` | Plain-text summary of recommendations |
| `get_pipeline()` | Return the fitted sklearn Pipeline |

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

To contribute ontology rules, see [docs/contributing_rules.md](docs/contributing_rules.md).

## Citation

```bibtex
@software{featureiq2024,
  title={FeatureIQ: Context-Aware Automated Feature Engineering},
  author={Pratik Khedekar},
  year={2026},
  url={https://github.com/khedekarpratik0337/featureiq},
}
```

## License

FeatureIQ is licensed under the [Apache License 2.0](LICENSE).
