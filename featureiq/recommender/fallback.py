"""Fallback recommender using ontology rules only (no meta-learning)."""

from __future__ import annotations

from featureiq.ontology.engine import OntologyEngine
from featureiq.ontology.rule_loader import OntologyRule
from featureiq.profiler.dataset_profiler import DatasetProfile
from featureiq.utils.validation import AlgorithmFamily, ProblemType


def ontology_only_recommend(
    dataset_profile: DatasetProfile,
    algorithm_family: AlgorithmFamily,
    problem_type: ProblemType,
    ontology_engine: OntologyEngine,
) -> dict[str, list[OntologyRule]]:
    """Return ontology-only recommendations without meta-learning.

    This is the fallback used when:
    - MetaLearner is not fitted
    - Dataset is too small for reliable meta-learning prediction

    Args:
        dataset_profile: DatasetProfile from profile_dataset.
        algorithm_family: Declared algorithm family.
        problem_type: Declared problem type.
        ontology_engine: Loaded OntologyEngine instance.

    Returns:
        Same structure as OntologyEngine.recommend_for_dataset().
    """
    return ontology_engine.recommend_for_dataset(
        dataset_profile, algorithm_family, problem_type
    )
