# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-10

### Changed

- Minimum Python version raised from 3.9 to 3.10 (Python 3.9 reached EOL October 2025)

### Fixed

- Added `.flake8` config to align line-length (88) with black formatter
- Fixed mypy strict mode compatibility with untyped third-party libraries (sklearn, scipy)
- Fixed incorrect return type on `_annotate_cross_column` in dataset profiler
- Fixed type errors in `FeatureIQ.ablate()` and override transform construction

## [0.1.0] - 2024-12-01

### Added

- Initial release of FeatureIQ
- Ontology-based rule engine with YAML rules for numerical, categorical, temporal, anomaly detection, and target transforms
- Automatic data profiling (column-level and dataset-level)
- Cross-column awareness: correlation matrix, VIF scores, multicollinearity detection
- Tiered cardinality thresholds (low/medium/high) replacing naive unique-ratio threshold
- NaN safety: automatic median imputer injection for columns with missing values but no explicit imputation step
- Algorithm-specific confidence adjustments via `preferred_for` / `not_recommended_for` hints
- LightGBM-based meta-learner with auto-discovery of bundled pretrained model
- Composability: `exclude_columns`, `override_transforms`, `pre_transforms`, `post_transforms`
- Transform validation via `validate()` and ablation analysis via `ablate()`
- `get_pipeline()` accessor for sklearn Pipeline inspection and integration
- sklearn-compatible `fit()` / `transform()` / `fit_transform()` API
- Rich-formatted recommendation and explanation reports
- 20+ built-in transformers including custom implementations for binary encoding, IQR clipping, date component extraction, lag features, rolling stats, differencing, and Fourier features
- Full test suite with 160+ tests
- CI/CD with GitHub Actions
- Contributing guide for ontology rules
