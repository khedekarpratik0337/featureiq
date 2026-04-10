# Contributing Ontology Rules to FeatureIQ

FeatureIQ's knowledge graph is built from **ontology rules** -- YAML-based declarations that encode expert feature engineering heuristics. This guide explains how to contribute new rules.

## Rule YAML Schema

Every rule must follow this exact schema:

```yaml
rules:
  - id: "TYPE_NNN"
    description: "Human-readable description of what this rule does"
    version: "1.0"
    conditions:
      column_type: "numerical"      # or "categorical", "datetime", null
      is_skewed: true               # or false, null
      has_missing: false            # or true, null
      cardinality_level: "low"     # "low", "medium", "high", or null
      is_highly_correlated: true   # or false, null
      has_high_vif: true           # or false, null
      is_monotonic: true           # or false, null (datetime only)
    algorithm_families:
      - "tree_based"
      - "linear_model"
    transformation: "standard_scaler"
    contraindicated_for:
      - "tree_based"
    confidence: 0.85
    source: "Author, Title, Year"
    tags: ["numerical", "scaling"]
    preferred_for: ["linear_regression"]         # optional
    not_recommended_for: ["lasso", "ridge"]      # optional
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier in format `{TYPE}_{NNN}` |
| `description` | string | Clear explanation of the rule's purpose |
| `version` | string | Semantic version of the rule |
| `conditions` | object | Conditions that must be met for the rule to fire |
| `algorithm_families` | list | Algorithm families this rule applies to |
| `transformation` | string | Must match a key in the transformer registry |
| `contraindicated_for` | list | Algorithm families where this rule should NOT fire |
| `confidence` | float | Value between 0.0 and 1.0 indicating rule reliability |
| `source` | string | Citation for the heuristic (must be a real reference) |
| `tags` | list | Searchable tags for categorisation |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `preferred_for` | list | Specific algorithms where confidence is boosted (+0.05) |
| `not_recommended_for` | list | Specific algorithms where confidence is reduced (-0.10) |
| `problem_types` | list | Restrict rule to specific problem types (null = all) |

## Rule ID Format

Rule IDs follow the pattern `{TYPE}_{NNN}` where:

- `TYPE` is one of:
  - `NUM` -- Numerical feature rules
  - `CAT` -- Categorical feature rules
  - `TMP` -- Temporal/datetime feature rules
  - `TGT` -- Target variable rules
  - `AD` -- Anomaly detection rules
  - `MCL` -- Multicollinearity rules
- `NNN` is a zero-padded three-digit number (e.g., `001`, `002`)

Check existing rule files to find the next available number for your type.

## Writing the `conditions` Block

Each field in `conditions` acts as a filter. A rule fires only when **all non-null conditions match** the column profile:

| Condition | Meaning | Relevant For |
|-----------|---------|-------------|
| `column_type` | The detected type of the column | All rules |
| `is_skewed` | Whether abs(skewness) > 1.0 | Numerical columns |
| `has_missing` | Whether missing rate > 5% | All columns |
| `cardinality_level` | `"low"` (<=10 unique), `"medium"` (11-50), `"high"` (>50) | Categorical columns |
| `is_highly_correlated` | Whether the column has Pearson |r| > 0.85 with another column | Numerical columns |
| `has_high_vif` | Whether VIF > 10 (multicollinearity) | Numerical columns |
| `is_monotonic` | Whether the datetime column is monotonically increasing/decreasing | Datetime columns |

Set a field to `null` (or omit it) to make the rule apply regardless of that condition.

**Note:** `is_high_cardinality` is still accepted for backward compatibility but `cardinality_level` is preferred.

## Valid `transformation` Names

The transformation field must exactly match a key in the transformer registry:

`log_transform`, `standard_scaler`, `min_max_scaler`, `robust_scaler`, `one_hot_encoder`, `ordinal_encoder`, `target_encoder`, `binary_encoder`, `mean_imputer`, `median_imputer`, `mode_imputer`, `rare_label_grouper`, `iqr_clipper`, `polynomial_features`, `date_component_extractor`, `time_since_reference`, `lag_feature_generator`, `rolling_stats_generator`, `differencing`, `fourier_features`

## Writing a Valid `source` Citation

Every rule must cite a real, verifiable source. Acceptable formats:

- **Book:** `Author(s), Title, Publisher, Year`
- **Paper:** `Author(s), Title, Journal/Conference, Year`
- **Documentation:** `Library Name documentation, Component Name`

Examples:
- `Kuhn & Johnson, Applied Predictive Modeling, Springer, 2013`
- `Micci-Barreca, A Preprocessing Scheme for High-Cardinality Categorical Attributes, SIGKDD Explorations, 2001`
- `scikit-learn documentation, StandardScaler`

Do NOT use blog posts, Stack Overflow answers, or undocumented heuristics.

## How to Submit a Rule

1. **Fork** the FeatureIQ repository
2. **Add your rule** to the appropriate YAML file in `featureiq/ontology/rules/`
3. **Run validation tests** to ensure your rule is valid:
   ```bash
   poetry run pytest tests/test_ontology/ -v
   ```
4. **Open a Pull Request** with:
   - A clear description of the rule
   - The source citation
   - Any benchmark evidence supporting the rule's confidence value

## Automatic Validation on PRs

When you open a PR, the CI pipeline automatically validates:

1. **Rule ID uniqueness** -- No duplicate IDs across all YAML files
2. **Schema validation** -- All required fields present and correctly typed
3. **Transformation name validation** -- The `transformation` field matches a registered transformer
4. **Confidence range** -- Value is between 0.0 and 1.0
5. **Algorithm family validation** -- All entries are valid `AlgorithmFamily` enum values
