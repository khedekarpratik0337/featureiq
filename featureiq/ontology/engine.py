"""Ontology reasoning engine for matching rules to columns and datasets."""

from __future__ import annotations

from featureiq.ontology.rule_loader import OntologyRule, load_all_rules
from featureiq.profiler.column_profiler import ColumnProfile
from featureiq.profiler.dataset_profiler import DatasetProfile
from featureiq.utils.validation import AlgorithmFamily, ColumnType, ProblemType


def match_rule_to_column(
    rule: OntologyRule,
    column_profile: ColumnProfile,
    algorithm_family: AlgorithmFamily,
    problem_type: ProblemType,
) -> bool:
    """Evaluate whether a rule applies to a specific column.

    A rule matches if ALL of the following are true:
    1. rule.conditions.column_type matches column_profile.column_type (or is None)
    2. Every non-None field in rule.conditions matches the corresponding
       field in column_profile
    3. algorithm_family is in rule.algorithm_families
    4. algorithm_family is NOT in rule.contraindicated_for

    Args:
        rule: An OntologyRule instance.
        column_profile: Profile of the column to check.
        algorithm_family: Declared algorithm family.
        problem_type: Declared problem type.

    Returns:
        True if rule applies, False otherwise.
    """
    if rule.problem_types is not None and problem_type not in rule.problem_types:
        return False

    cond = rule.conditions

    if cond.column_type is not None and cond.column_type != column_profile.column_type:
        return False

    if cond.is_skewed is not None:
        if column_profile.is_skewed is None or cond.is_skewed != column_profile.is_skewed:
            return False

    if cond.has_missing is not None:
        if cond.has_missing != column_profile.has_missing:
            return False

    if cond.cardinality_level is not None:
        if cond.cardinality_level != column_profile.cardinality_level:
            return False

    if cond.is_high_cardinality is not None:
        if cond.is_high_cardinality != column_profile.is_high_cardinality:
            return False

    if cond.is_highly_correlated is not None:
        if cond.is_highly_correlated != column_profile.is_highly_correlated:
            return False

    if cond.has_high_vif is not None:
        if cond.has_high_vif != column_profile.has_high_vif:
            return False

    if cond.is_monotonic is not None:
        if column_profile.is_monotonic is None or cond.is_monotonic != column_profile.is_monotonic:
            return False

    if cond.has_regular_frequency is not None:
        if (
            column_profile.has_regular_frequency is None
            or cond.has_regular_frequency != column_profile.has_regular_frequency
        ):
            return False

    if algorithm_family not in rule.algorithm_families:
        return False

    if algorithm_family in rule.contraindicated_for:
        return False

    return True


def _adjust_confidence_for_algorithm(
    rule: OntologyRule,
    algorithm: str | None,
) -> float:
    """Adjust rule confidence based on algorithm-specific hints."""
    confidence = rule.confidence
    if algorithm is None:
        return confidence
    algo_lower = algorithm.lower()
    if rule.preferred_for and algo_lower in rule.preferred_for:
        confidence = min(1.0, confidence + 0.05)
    if rule.not_recommended_for and algo_lower in rule.not_recommended_for:
        confidence = max(0.0, confidence - 0.10)
    return confidence


def get_applicable_rules(
    column_profile: ColumnProfile,
    all_rules: list[OntologyRule],
    algorithm_family: AlgorithmFamily,
    problem_type: ProblemType,
    algorithm: str | None = None,
) -> list[OntologyRule]:
    """Return all rules applicable to a column, sorted by confidence descending.

    Args:
        column_profile: Profile of a single column.
        all_rules: Full list of loaded ontology rules.
        algorithm_family: Declared algorithm family.
        problem_type: Declared problem type.
        algorithm: Optional specific algorithm string for confidence adjustment.

    Returns:
        Filtered and sorted list of applicable OntologyRule instances.
    """
    applicable: list[tuple[float, OntologyRule]] = []
    for rule in all_rules:
        if rule.id.startswith("TGT_"):
            continue
        if match_rule_to_column(rule, column_profile, algorithm_family, problem_type):
            adjusted = _adjust_confidence_for_algorithm(rule, algorithm)
            applicable.append((adjusted, rule))

    applicable.sort(key=lambda x: x[0], reverse=True)
    results = []
    for adj_conf, rule in applicable:
        if adj_conf != rule.confidence:
            rule = rule.model_copy(update={"confidence": adj_conf})
        results.append(rule)
    return results


def get_dataset_level_rules(
    dataset_profile: DatasetProfile,
    all_rules: list[OntologyRule],
    algorithm_family: AlgorithmFamily,
    problem_type: ProblemType,
) -> list[OntologyRule]:
    """Return dataset-level rules (e.g., class balancing, target transform).

    These are rules from target.yaml that apply to the dataset as a whole,
    not to individual columns.

    Args:
        dataset_profile: Full DatasetProfile.
        all_rules: Full list of loaded ontology rules.
        algorithm_family: Declared algorithm family.
        problem_type: Declared problem type.

    Returns:
        List of applicable dataset-level OntologyRule instances.
    """
    dataset_rules: list[OntologyRule] = []
    for rule in all_rules:
        if not rule.id.startswith("TGT_"):
            continue

        if algorithm_family not in rule.algorithm_families:
            continue
        if algorithm_family in rule.contraindicated_for:
            continue

        if rule.problem_types is not None and problem_type not in rule.problem_types:
            continue

        if rule.id == "TGT_002" and problem_type != ProblemType.BINARY_CLASSIFICATION:
            continue
        if rule.id == "TGT_003" and problem_type != ProblemType.MULTICLASS_CLASSIFICATION:
            continue
        if rule.id == "TGT_001" and problem_type not in (
            ProblemType.REGRESSION,
            ProblemType.TIME_SERIES_FORECASTING,
        ):
            continue

        dataset_rules.append(rule)

    return sorted(dataset_rules, key=lambda r: r.confidence, reverse=True)


class OntologyEngine:
    """Manages rule loading and reasoning for feature engineering recommendations.

    Args:
        rules_dir: Optional path to custom rules directory.

    Attributes:
        rules: List of all loaded OntologyRule instances.
    """

    def __init__(self, rules_dir: str | None = None) -> None:
        self.rules = load_all_rules(rules_dir)

    def recommend_for_column(
        self,
        column_profile: ColumnProfile,
        algorithm_family: AlgorithmFamily,
        problem_type: ProblemType,
        algorithm: str | None = None,
    ) -> list[OntologyRule]:
        """Return applicable rules for a single column.

        Args:
            column_profile: Profile of the column.
            algorithm_family: Declared algorithm family.
            problem_type: Declared problem type.
            algorithm: Optional specific algorithm name for confidence adjustment.

        Returns:
            List of applicable rules sorted by confidence descending.
        """
        return get_applicable_rules(
            column_profile, self.rules, algorithm_family, problem_type, algorithm
        )

    def recommend_for_dataset(
        self,
        dataset_profile: DatasetProfile,
        algorithm_family: AlgorithmFamily,
        problem_type: ProblemType,
        algorithm: str | None = None,
    ) -> dict[str, list[OntologyRule]]:
        """Return recommendations for all columns in the dataset.

        Args:
            dataset_profile: Full DatasetProfile.
            algorithm_family: Declared algorithm family.
            problem_type: Declared problem type.
            algorithm: Optional specific algorithm name for confidence adjustment.

        Returns:
            Dict mapping column name to list of applicable rules.
            Includes a special key "__dataset__" for dataset-level rules.
        """
        recommendations: dict[str, list[OntologyRule]] = {}

        for col_name, col_profile in dataset_profile.column_profiles.items():
            applicable = self.recommend_for_column(
                col_profile, algorithm_family, problem_type, algorithm
            )
            if applicable:
                recommendations[col_name] = applicable

        dataset_rules = get_dataset_level_rules(
            dataset_profile, self.rules, algorithm_family, problem_type
        )
        if dataset_rules:
            recommendations["__dataset__"] = dataset_rules

        return recommendations
