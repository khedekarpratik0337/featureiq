"""Load and validate ontology rules from YAML files."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from featureiq.exceptions import OntologyError, RuleValidationError
from featureiq.utils.validation import AlgorithmFamily, ColumnType, ProblemType


class RuleCondition(BaseModel):
    """Conditions that must be met for a rule to apply."""

    column_type: ColumnType | None = None
    is_skewed: bool | None = None
    has_missing: bool | None = None
    is_high_cardinality: bool | None = None
    cardinality_level: str | None = None
    is_monotonic: bool | None = None
    has_regular_frequency: bool | None = None
    is_highly_correlated: bool | None = None
    has_high_vif: bool | None = None


class OntologyRule(BaseModel):
    """A single ontology rule for feature engineering."""

    id: str
    description: str
    version: str
    conditions: RuleCondition
    algorithm_families: list[AlgorithmFamily]
    problem_types: list[ProblemType] | None = None
    transformation: str
    contraindicated_for: list[AlgorithmFamily]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    tags: list[str]
    preferred_for: list[str] | None = None
    not_recommended_for: list[str] | None = None


class RuleSet(BaseModel):
    """A collection of ontology rules loaded from a YAML file."""

    rules: list[OntologyRule]


def _get_default_rules_dir() -> str:
    """Return the path to the built-in rules directory."""
    return str(Path(__file__).parent / "rules")


def load_rules_from_yaml(filepath: str) -> RuleSet:
    """Load and validate an ontology rule file.

    Args:
        filepath: Absolute path to a YAML rule file.

    Returns:
        Validated RuleSet instance.

    Raises:
        OntologyError: If file not found or YAML is malformed.
        RuleValidationError: If Pydantic validation fails on any rule.
    """
    if not os.path.exists(filepath):
        raise OntologyError(f"Rule file not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise OntologyError(f"Malformed YAML in {filepath}: {exc}") from exc

    if raw is None or "rules" not in raw:
        raise OntologyError(f"Rule file {filepath} has no 'rules' key.")

    try:
        validate = getattr(RuleSet, "model_validate", None) or RuleSet.parse_obj
        return validate(raw)
    except Exception as exc:
        raise RuleValidationError(
            f"Validation failed for rules in {filepath}: {exc}"
        ) from exc


def load_all_rules(rules_dir: str | None = None) -> list[OntologyRule]:
    """Load all YAML rule files from the rules directory.

    Deduplicates rules by ID. If two rules share an ID, raises RuleValidationError.
    Default rules_dir is the package's built-in ontology/rules/ directory.

    Args:
        rules_dir: Optional path to a custom rules directory.

    Returns:
        List of all validated OntologyRule instances.

    Raises:
        OntologyError: If rules directory does not exist.
        RuleValidationError: If duplicate rule IDs are found.
    """
    if rules_dir is None:
        rules_dir = _get_default_rules_dir()

    if not os.path.isdir(rules_dir):
        raise OntologyError(f"Rules directory does not exist: {rules_dir}")

    all_rules: list[OntologyRule] = []
    seen_ids: dict[str, str] = {}

    for filename in sorted(os.listdir(rules_dir)):
        if not filename.endswith((".yaml", ".yml")):
            continue

        filepath = os.path.join(rules_dir, filename)
        rule_set = load_rules_from_yaml(filepath)

        for rule in rule_set.rules:
            if rule.id in seen_ids:
                raise RuleValidationError(
                    f"Duplicate rule ID '{rule.id}' found in "
                    f"'{filename}' and '{seen_ids[rule.id]}'."
                )
            seen_ids[rule.id] = filename
            all_rules.append(rule)

    return all_rules
