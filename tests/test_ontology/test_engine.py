"""Tests for the ontology engine."""

from __future__ import annotations

import pytest

from featureiq.ontology.engine import (
    OntologyEngine,
    get_applicable_rules,
    get_dataset_level_rules,
    match_rule_to_column,
)
from featureiq.ontology.rule_loader import OntologyRule, RuleCondition, load_all_rules
from featureiq.profiler.column_profiler import ColumnProfile
from featureiq.utils.validation import AlgorithmFamily, ColumnType, ProblemType


@pytest.fixture
def all_rules() -> list[OntologyRule]:
    return load_all_rules()


@pytest.fixture
def skewed_numerical_profile() -> ColumnProfile:
    return ColumnProfile(
        name="skewed_num",
        column_type=ColumnType.NUMERICAL,
        n_unique=50,
        unique_ratio=0.5,
        missing_rate=0.0,
        is_high_cardinality=False,
        has_missing=False,
        mean=5.0,
        std=2.0,
        skewness=2.5,
        kurtosis=3.0,
        is_skewed=True,
        outlier_fraction=0.05,
        is_normal=False,
    )


@pytest.fixture
def normal_numerical_profile() -> ColumnProfile:
    return ColumnProfile(
        name="normal_num",
        column_type=ColumnType.NUMERICAL,
        n_unique=50,
        unique_ratio=0.5,
        missing_rate=0.0,
        is_high_cardinality=False,
        has_missing=False,
        mean=0.0,
        std=1.0,
        skewness=0.1,
        kurtosis=0.0,
        is_skewed=False,
        outlier_fraction=0.0,
        is_normal=True,
    )


@pytest.fixture
def high_cardinality_categorical_profile() -> ColumnProfile:
    return ColumnProfile(
        name="high_card_cat",
        column_type=ColumnType.CATEGORICAL,
        n_unique=100,
        unique_ratio=0.8,
        missing_rate=0.0,
        is_high_cardinality=True,
        has_missing=False,
        top_frequency=0.02,
    )


class TestRuleMatching:
    """Tests for rule matching logic."""

    def test_skewed_numerical_linear_gets_log_transform(
        self,
        all_rules: list[OntologyRule],
        skewed_numerical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            skewed_numerical_profile,
            all_rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.BINARY_CLASSIFICATION,
        )
        transformations = [r.transformation for r in applicable]
        assert "log_transform" in transformations

    def test_tree_based_no_standard_scaler(
        self,
        all_rules: list[OntologyRule],
        normal_numerical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            normal_numerical_profile,
            all_rules,
            AlgorithmFamily.TREE_BASED,
            ProblemType.BINARY_CLASSIFICATION,
        )
        transformations = [r.transformation for r in applicable]
        assert "standard_scaler" not in transformations

    def test_contraindication_excludes_rule(
        self,
        all_rules: list[OntologyRule],
        normal_numerical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            normal_numerical_profile,
            all_rules,
            AlgorithmFamily.ENSEMBLE,
            ProblemType.BINARY_CLASSIFICATION,
        )
        transformations = [r.transformation for r in applicable]
        assert "standard_scaler" not in transformations

    def test_high_cardinality_tree_gets_target_encoding(
        self,
        all_rules: list[OntologyRule],
        high_cardinality_categorical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            high_cardinality_categorical_profile,
            all_rules,
            AlgorithmFamily.TREE_BASED,
            ProblemType.BINARY_CLASSIFICATION,
        )
        transformations = [r.transformation for r in applicable]
        assert "target_encoder" in transformations

    def test_high_cardinality_linear_gets_binary_encoding(
        self,
        all_rules: list[OntologyRule],
        high_cardinality_categorical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            high_cardinality_categorical_profile,
            all_rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.BINARY_CLASSIFICATION,
        )
        transformations = [r.transformation for r in applicable]
        assert "binary_encoder" in transformations

    def test_empty_rule_list_returns_empty(
        self,
        normal_numerical_profile: ColumnProfile,
    ) -> None:
        applicable = get_applicable_rules(
            normal_numerical_profile,
            [],
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.REGRESSION,
        )
        assert applicable == []

    def test_is_monotonic_mismatch_rejects_rule(self) -> None:
        rule = OntologyRule(
            id="TEST_MONO",
            description="needs monotonic",
            version="1.0",
            conditions=RuleCondition(
                column_type=ColumnType.DATETIME, is_monotonic=True
            ),
            algorithm_families=[AlgorithmFamily.TREE_BASED],
            transformation="lag_feature_generator",
            contraindicated_for=[],
            confidence=0.85,
            source="test",
            tags=["test"],
        )
        profile = ColumnProfile(
            name="dt_col",
            column_type=ColumnType.DATETIME,
            n_unique=10,
            unique_ratio=0.1,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            is_monotonic=False,
            has_regular_frequency=True,
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.TREE_BASED,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is False
        )

    def test_is_monotonic_none_rejects_rule(self) -> None:
        rule = OntologyRule(
            id="TEST_MONO2",
            description="needs monotonic",
            version="1.0",
            conditions=RuleCondition(
                column_type=ColumnType.DATETIME, is_monotonic=True
            ),
            algorithm_families=[AlgorithmFamily.TREE_BASED],
            transformation="lag_feature_generator",
            contraindicated_for=[],
            confidence=0.85,
            source="test",
            tags=["test"],
        )
        profile = ColumnProfile(
            name="dt_col",
            column_type=ColumnType.DATETIME,
            n_unique=10,
            unique_ratio=0.1,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            is_monotonic=None,
            has_regular_frequency=True,
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.TREE_BASED,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is False
        )

    def test_algorithm_in_both_families_and_contraindicated(self) -> None:
        rule = OntologyRule(
            id="TEST_CONTRA",
            description="contra rule",
            version="1.0",
            conditions=RuleCondition(column_type=ColumnType.NUMERICAL),
            algorithm_families=[
                AlgorithmFamily.LINEAR_MODEL,
                AlgorithmFamily.ENSEMBLE,
            ],
            transformation="standard_scaler",
            contraindicated_for=[AlgorithmFamily.ENSEMBLE],
            confidence=0.9,
            source="test",
            tags=["test"],
        )
        profile = ColumnProfile(
            name="num",
            column_type=ColumnType.NUMERICAL,
            n_unique=50,
            unique_ratio=0.5,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=0.0,
            is_skewed=False,
            outlier_fraction=0.0,
            is_normal=True,
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.ENSEMBLE,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is False
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is True
        )


class TestDatasetLevelRules:
    """Tests for get_dataset_level_rules."""

    def test_dataset_level_rules_for_binary(self) -> None:
        rules = load_all_rules()
        from featureiq.profiler.dataset_profiler import DatasetProfile

        profile = DatasetProfile(
            n_rows=100,
            n_columns=5,
            n_numerical=3,
            n_categorical=1,
            n_datetime=0,
            n_boolean=1,
            n_text=0,
            n_unknown=0,
            overall_missing_rate=0.0,
            class_imbalance_ratio=0.5,
            feature_to_row_ratio=0.05,
            has_temporal_structure=False,
            column_profiles={},
        )
        ds_rules = get_dataset_level_rules(
            profile,
            rules,
            AlgorithmFamily.TREE_BASED,
            ProblemType.BINARY_CLASSIFICATION,
        )
        rule_ids = [r.id for r in ds_rules]
        assert "TGT_002" in rule_ids
        assert "TGT_001" not in rule_ids
        assert "TGT_003" not in rule_ids

    def test_dataset_level_rules_for_regression(self) -> None:
        rules = load_all_rules()
        from featureiq.profiler.dataset_profiler import DatasetProfile

        profile = DatasetProfile(
            n_rows=100,
            n_columns=5,
            n_numerical=3,
            n_categorical=1,
            n_datetime=0,
            n_boolean=1,
            n_text=0,
            n_unknown=0,
            overall_missing_rate=0.0,
            class_imbalance_ratio=None,
            feature_to_row_ratio=0.05,
            has_temporal_structure=False,
            column_profiles={},
        )
        ds_rules = get_dataset_level_rules(
            profile,
            rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.REGRESSION,
        )
        rule_ids = [r.id for r in ds_rules]
        assert "TGT_001" in rule_ids
        assert "TGT_002" not in rule_ids
        assert "TGT_003" not in rule_ids

    def test_dataset_level_contraindication_skips_rule(self) -> None:
        rule = OntologyRule(
            id="TGT_099",
            description="contra dataset rule",
            version="1.0",
            conditions=RuleCondition(),
            algorithm_families=[AlgorithmFamily.LINEAR_MODEL],
            transformation="class_weight_balancing",
            contraindicated_for=[AlgorithmFamily.LINEAR_MODEL],
            confidence=0.85,
            source="test",
            tags=["test"],
        )
        from featureiq.profiler.dataset_profiler import DatasetProfile

        profile = DatasetProfile(
            n_rows=100,
            n_columns=5,
            n_numerical=3,
            n_categorical=1,
            n_datetime=0,
            n_boolean=1,
            n_text=0,
            n_unknown=0,
            overall_missing_rate=0.0,
            class_imbalance_ratio=0.5,
            feature_to_row_ratio=0.05,
            has_temporal_structure=False,
            column_profiles={},
        )
        ds_rules = get_dataset_level_rules(
            profile,
            [rule],
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.BINARY_CLASSIFICATION,
        )
        assert len(ds_rules) == 0


class TestOntologyEngine:
    """Tests for the OntologyEngine class."""

    def test_engine_loads_rules(self) -> None:
        engine = OntologyEngine()
        assert len(engine.rules) >= 21

    def test_recommend_for_dataset(
        self,
        mixed_dataframe,
        target_binary,
    ) -> None:
        from featureiq.profiler.dataset_profiler import profile_dataset

        profile = profile_dataset(
            mixed_dataframe, target_binary, ProblemType.BINARY_CLASSIFICATION
        )
        engine = OntologyEngine()
        recs = engine.recommend_for_dataset(
            profile, AlgorithmFamily.LINEAR_MODEL, ProblemType.BINARY_CLASSIFICATION
        )
        assert isinstance(recs, dict)
        assert len(recs) > 0


class TestTimeSeriesRules:
    """Tests for time-series-specific ontology rules."""

    @pytest.fixture
    def numerical_profile(self) -> ColumnProfile:
        return ColumnProfile(
            name="ts_value",
            column_type=ColumnType.NUMERICAL,
            n_unique=50,
            unique_ratio=0.5,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            mean=10.0,
            std=3.0,
            skewness=0.2,
            kurtosis=0.1,
            is_skewed=False,
            outlier_fraction=0.01,
            is_normal=True,
        )

    def test_temporal_rules_fire_for_time_series_forecasting(
        self, numerical_profile: ColumnProfile
    ) -> None:
        all_rules = load_all_rules()
        applicable = get_applicable_rules(
            numerical_profile,
            all_rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.TIME_SERIES_FORECASTING,
        )
        transformations = [r.transformation for r in applicable]
        assert "lag_feature_generator" in transformations
        assert "rolling_stats_generator" in transformations
        assert "differencing" in transformations
        assert "fourier_features" in transformations

    def test_temporal_rules_dont_fire_for_regression(
        self, numerical_profile: ColumnProfile
    ) -> None:
        all_rules = load_all_rules()
        applicable = get_applicable_rules(
            numerical_profile,
            all_rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.REGRESSION,
        )
        transformations = [r.transformation for r in applicable]
        assert "lag_feature_generator" not in transformations
        assert "rolling_stats_generator" not in transformations
        assert "differencing" not in transformations
        assert "fourier_features" not in transformations

    def test_problem_types_filter_works(self) -> None:
        rule = OntologyRule(
            id="TEST_PT_FILTER",
            description="only for anomaly detection",
            version="1.0",
            conditions=RuleCondition(column_type=ColumnType.NUMERICAL),
            algorithm_families=[AlgorithmFamily.LINEAR_MODEL],
            problem_types=[ProblemType.ANOMALY_DETECTION],
            transformation="robust_scaler",
            contraindicated_for=[],
            confidence=0.85,
            source="test",
            tags=["test"],
        )
        profile = ColumnProfile(
            name="num",
            column_type=ColumnType.NUMERICAL,
            n_unique=50,
            unique_ratio=0.5,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=0.0,
            is_skewed=False,
            outlier_fraction=0.0,
            is_normal=True,
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.REGRESSION,
            )
            is False
        )
        assert (
            match_rule_to_column(
                rule,
                profile,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.ANOMALY_DETECTION,
            )
            is True
        )

    def test_has_regular_frequency_condition(self) -> None:
        rule = OntologyRule(
            id="TEST_FREQ",
            description="needs regular frequency",
            version="1.0",
            conditions=RuleCondition(
                column_type=ColumnType.DATETIME,
                has_regular_frequency=True,
            ),
            algorithm_families=[AlgorithmFamily.LINEAR_MODEL],
            transformation="fourier_features",
            contraindicated_for=[],
            confidence=0.8,
            source="test",
            tags=["test"],
        )
        profile_match = ColumnProfile(
            name="dt",
            column_type=ColumnType.DATETIME,
            n_unique=10,
            unique_ratio=0.1,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            has_regular_frequency=True,
        )
        profile_no_match = ColumnProfile(
            name="dt",
            column_type=ColumnType.DATETIME,
            n_unique=10,
            unique_ratio=0.1,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            has_regular_frequency=False,
        )
        profile_none = ColumnProfile(
            name="dt",
            column_type=ColumnType.DATETIME,
            n_unique=10,
            unique_ratio=0.1,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            has_regular_frequency=None,
        )
        assert (
            match_rule_to_column(
                rule,
                profile_match,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is True
        )
        assert (
            match_rule_to_column(
                rule,
                profile_no_match,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is False
        )
        assert (
            match_rule_to_column(
                rule,
                profile_none,
                AlgorithmFamily.LINEAR_MODEL,
                ProblemType.BINARY_CLASSIFICATION,
            )
            is False
        )


class TestAnomalyDetectionRules:
    """Tests for anomaly-detection-specific ontology rules."""

    def test_ad_rules_fire_for_anomaly_detection(self) -> None:
        all_rules = load_all_rules()
        profile = ColumnProfile(
            name="num_feat",
            column_type=ColumnType.NUMERICAL,
            n_unique=50,
            unique_ratio=0.5,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            mean=0.0,
            std=1.0,
            skewness=0.1,
            kurtosis=0.0,
            is_skewed=False,
            outlier_fraction=0.02,
            is_normal=True,
        )
        applicable = get_applicable_rules(
            profile,
            all_rules,
            AlgorithmFamily.DISTANCE_BASED,
            ProblemType.ANOMALY_DETECTION,
        )
        transformations = [r.transformation for r in applicable]
        assert "robust_scaler" in transformations

    def test_ad_rules_dont_fire_for_regression(self) -> None:
        all_rules = load_all_rules()
        profile = ColumnProfile(
            name="num_feat",
            column_type=ColumnType.NUMERICAL,
            n_unique=50,
            unique_ratio=0.5,
            missing_rate=0.0,
            is_high_cardinality=False,
            has_missing=False,
            mean=0.0,
            std=1.0,
            skewness=0.1,
            kurtosis=0.0,
            is_skewed=False,
            outlier_fraction=0.02,
            is_normal=True,
        )
        applicable = get_applicable_rules(
            profile,
            all_rules,
            AlgorithmFamily.DISTANCE_BASED,
            ProblemType.REGRESSION,
        )
        transformations = [r.transformation for r in applicable]
        assert "robust_scaler" not in transformations

    def test_dataset_level_tgt_006_fires_for_ad(self) -> None:
        from featureiq.profiler.dataset_profiler import DatasetProfile

        all_rules = load_all_rules()
        profile = DatasetProfile(
            n_rows=100,
            n_columns=5,
            n_numerical=4,
            n_categorical=1,
            n_datetime=0,
            n_boolean=0,
            n_text=0,
            n_unknown=0,
            overall_missing_rate=0.0,
            class_imbalance_ratio=None,
            feature_to_row_ratio=0.05,
            has_temporal_structure=False,
            column_profiles={},
        )
        ds_rules = get_dataset_level_rules(
            profile,
            all_rules,
            AlgorithmFamily.DISTANCE_BASED,
            ProblemType.ANOMALY_DETECTION,
        )
        rule_ids = [r.id for r in ds_rules]
        assert "TGT_006" in rule_ids

    def test_dataset_level_tgt_006_skipped_for_regression(self) -> None:
        from featureiq.profiler.dataset_profiler import DatasetProfile

        all_rules = load_all_rules()
        profile = DatasetProfile(
            n_rows=100,
            n_columns=5,
            n_numerical=4,
            n_categorical=1,
            n_datetime=0,
            n_boolean=0,
            n_text=0,
            n_unknown=0,
            overall_missing_rate=0.0,
            class_imbalance_ratio=None,
            feature_to_row_ratio=0.05,
            has_temporal_structure=False,
            column_profiles={},
        )
        ds_rules = get_dataset_level_rules(
            profile,
            all_rules,
            AlgorithmFamily.LINEAR_MODEL,
            ProblemType.REGRESSION,
        )
        rule_ids = [r.id for r in ds_rules]
        assert "TGT_006" not in rule_ids
