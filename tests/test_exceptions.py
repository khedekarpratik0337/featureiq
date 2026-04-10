"""Tests for direct instantiation of all FeatureIQ exception classes."""

from __future__ import annotations

from featureiq.exceptions import (
    FeatureIQError,
    InsufficientDataError,
    OntologyError,
    ProfilingError,
    RecommenderError,
    RuleValidationError,
    TransformerError,
    UnsupportedAlgorithmError,
    UnsupportedProblemTypeError,
)


class TestExceptions:
    """Verify every exception stores .message and inherits correctly."""

    def test_featureiq_error(self) -> None:
        exc = FeatureIQError("base error")
        assert exc.message == "base error"
        assert str(exc) == "base error"

    def test_profiling_error(self) -> None:
        exc = ProfilingError("profiling failed")
        assert exc.message == "profiling failed"
        assert isinstance(exc, FeatureIQError)

    def test_ontology_error(self) -> None:
        exc = OntologyError("ontology issue")
        assert exc.message == "ontology issue"
        assert isinstance(exc, FeatureIQError)

    def test_rule_validation_error(self) -> None:
        exc = RuleValidationError("bad rule")
        assert exc.message == "bad rule"
        assert isinstance(exc, OntologyError)

    def test_recommender_error(self) -> None:
        exc = RecommenderError("recommender failed")
        assert exc.message == "recommender failed"
        assert isinstance(exc, FeatureIQError)

    def test_transformer_error(self) -> None:
        exc = TransformerError("transform failed")
        assert exc.message == "transform failed"
        assert isinstance(exc, FeatureIQError)

    def test_unsupported_algorithm_error(self) -> None:
        exc = UnsupportedAlgorithmError("unknown algo")
        assert exc.message == "unknown algo"
        assert isinstance(exc, FeatureIQError)

    def test_unsupported_problem_type_error(self) -> None:
        exc = UnsupportedProblemTypeError("unknown problem")
        assert exc.message == "unknown problem"
        assert isinstance(exc, FeatureIQError)

    def test_insufficient_data_error(self) -> None:
        exc = InsufficientDataError("too few rows")
        assert exc.message == "too few rows"
        assert isinstance(exc, FeatureIQError)
