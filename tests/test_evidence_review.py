"""Core consistency tests for the mixed climate-fisheries evidence workflow."""

import pytest
from pydantic import ValidationError

from evidence_review.quality import appraisal_criteria, appraisal_domain
from evidence_review.schema import (
    EvidenceFinding,
    QualityAppraisal,
    ScreeningDecision,
    SourceType,
)
from evidence_review.scoring import score_transferability


def test_screening_decision_requires_all_three_inclusion_criteria() -> None:
    decision = ScreeningDecision(
        source_id="paper_001",
        include=True,
        source_type=SourceType.SCIENTIFIC_ARTICLE,
        fisheries_or_marine_relevant=True,
        climate_environment_or_adaptation_element=True,
        contributes_codable_evidence=True,
        screening_stage="title_abstract",
    )
    assert decision.include is True

    with pytest.raises(ValidationError):
        ScreeningDecision(
            source_id="paper_002",
            include=True,
            source_type=SourceType.SCIENTIFIC_ARTICLE,
            fisheries_or_marine_relevant=True,
            climate_environment_or_adaptation_element=False,
            contributes_codable_evidence=True,
            screening_stage="title_abstract",
        )


def test_excluded_source_requires_reason() -> None:
    with pytest.raises(ValidationError):
        ScreeningDecision(
            source_id="report_001",
            include=False,
            source_type=SourceType.TECHNICAL_REPORT,
            fisheries_or_marine_relevant=False,
            climate_environment_or_adaptation_element=True,
            contributes_codable_evidence=True,
            screening_stage="full_text",
        )


def test_finding_preserves_locator_and_evidence() -> None:
    finding = EvidenceFinding(
        finding_id="finding_001",
        source_id="paper_001",
        source_type=SourceType.SCIENTIFIC_ARTICLE,
        finding_type="empirical_result",
        evidence_streams=["ecological_response"],
        unit_locator="Results, p. 12, Table 3",
        supporting_excerpt="Recruitment declined during the warm period.",
        evidence_summary="The study reports lower recruitment during the analysed warm period.",
        climate_drivers=["ocean_warming"],
        biological_responses=["recruitment"],
        relevance_to_small_pelagics="direct",
        relevance_to_anchoveta="indirect_high",
    )
    assert finding.unit_locator.startswith("Results")


def test_quality_scores_are_calculated_and_bounded() -> None:
    appraisal = QualityAppraisal(
        appraisal_id="quality_001",
        source_id="project_001",
        appraisal_domain="project",
        criteria_scores={"baseline_quality": 1, "outcome_evidence": 2},
        overall_rating="moderate",
    )
    assert appraisal.total_score == 3
    assert appraisal.maximum_score == 4

    with pytest.raises(ValidationError):
        QualityAppraisal(
            appraisal_id="quality_002",
            source_id="project_002",
            appraisal_domain="project",
            criteria_scores={"baseline_quality": 3},
            overall_rating="high",
        )


def test_quality_domain_depends_on_source_type() -> None:
    assert appraisal_domain(SourceType.SCIENTIFIC_ARTICLE) == "scientific"
    assert appraisal_domain(SourceType.PROJECT_EVALUATION) == "project"
    assert appraisal_domain(SourceType.NATIONAL_LAW) == "legal"
    assert "design_appropriateness" in appraisal_criteria(
        SourceType.SCIENTIFIC_ARTICLE
    )


def test_transferability_is_component_based() -> None:
    assessment = score_transferability(
        ecological_similarity=3,
        climate_relevance=3,
        management_relevance=2,
        evidence_strength=2,
        operational_specificity=2,
        data_feasibility_peru=2,
        rationale="Direct small-pelagic evidence with an implemented management response.",
    )
    assert assessment.total_score == 14
    assert assessment.transferability_class == "very_high"
