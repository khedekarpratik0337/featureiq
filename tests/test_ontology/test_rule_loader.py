"""Tests for the ontology rule loader – all error/edge-case paths."""

from __future__ import annotations

import os

import pytest

from featureiq.exceptions import OntologyError, RuleValidationError
from featureiq.ontology.rule_loader import (
    RuleSet,
    load_all_rules,
    load_rules_from_yaml,
)


class TestLoadRulesFromYaml:
    """Tests for load_rules_from_yaml error paths."""

    def test_nonexistent_file_raises_ontology_error(self) -> None:
        with pytest.raises(OntologyError, match="not found"):
            load_rules_from_yaml("/tmp/nonexistent_rule_file_xyz.yaml")

    def test_malformed_yaml_raises_ontology_error(self, tmp_path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{{not valid yaml: [")
        with pytest.raises(OntologyError, match="Malformed YAML"):
            load_rules_from_yaml(str(bad_yaml))

    def test_missing_rules_key_raises_ontology_error(self, tmp_path) -> None:
        no_rules = tmp_path / "no_rules.yaml"
        no_rules.write_text("data:\n  - item: 1\n")
        with pytest.raises(OntologyError, match="no 'rules' key"):
            load_rules_from_yaml(str(no_rules))

    def test_empty_yaml_raises_ontology_error(self, tmp_path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(OntologyError, match="no 'rules' key"):
            load_rules_from_yaml(str(empty))

    def test_invalid_rule_schema_raises_rule_validation_error(self, tmp_path) -> None:
        invalid_schema = tmp_path / "invalid.yaml"
        invalid_schema.write_text("rules:\n" "  - id: 123\n" "    description: null\n")
        with pytest.raises(RuleValidationError, match="Validation failed"):
            load_rules_from_yaml(str(invalid_schema))

    def test_valid_yaml_loads_successfully(self) -> None:
        from featureiq.ontology.rule_loader import _get_default_rules_dir

        rules_dir = _get_default_rules_dir()
        yaml_files = [f for f in os.listdir(rules_dir) if f.endswith(".yaml")]
        assert len(yaml_files) > 0
        first = os.path.join(rules_dir, yaml_files[0])
        rs = load_rules_from_yaml(first)
        assert isinstance(rs, RuleSet)
        assert len(rs.rules) > 0


class TestLoadAllRules:
    """Tests for load_all_rules error paths."""

    def test_nonexistent_directory_raises(self) -> None:
        with pytest.raises(OntologyError, match="does not exist"):
            load_all_rules("/tmp/nonexistent_rules_dir_xyz")

    def test_non_yaml_files_are_skipped(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("not a rule file")
        (tmp_path / "data.json").write_text("{}")
        rules = load_all_rules(str(tmp_path))
        assert rules == []

    def test_duplicate_rule_ids_raises(self, tmp_path) -> None:
        rule_content = (
            "rules:\n"
            "  - id: DUP_001\n"
            "    description: Test rule\n"
            "    version: '1.0'\n"
            "    conditions: {}\n"
            "    algorithm_families: [tree_based]\n"
            "    transformation: log_transform\n"
            "    contraindicated_for: []\n"
            "    confidence: 0.9\n"
            "    source: test\n"
            "    tags: [test]\n"
        )
        (tmp_path / "a.yaml").write_text(rule_content)
        (tmp_path / "b.yaml").write_text(rule_content)
        with pytest.raises(RuleValidationError, match="Duplicate rule ID"):
            load_all_rules(str(tmp_path))

    def test_default_rules_dir_loads(self) -> None:
        rules = load_all_rules()
        assert len(rules) >= 21
