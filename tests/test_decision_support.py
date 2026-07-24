from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_review.decision_support import (
    build_anchoveta_recommendation_framework,
    build_climate_response_management_chains,
    build_decision_support_summary,
    build_management_option_portfolio,
    build_operational_readiness_matrix,
    build_research_priority_matrix,
    initialise_recommendation_review,
    load_decision_support_config,
    split_recommendation_review,
    validate_decision_support_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "decision_support.yml"


@pytest.fixture
def config():
    return load_decision_support_config(CONFIG_PATH, project_root=ROOT)


@pytest.fixture
def matrix():
    return pd.DataFrame(
        [
            {
                "finding_id": "finding_001",
                "source_id": "source_001",
                "title": "Evaluated adaptive monitoring",
                "evidence_summary": (
                    "Environmental information was used to adapt monitoring and "
                    "the implementation outcome was evaluated."
                ),
                "supporting_excerpt": "Monitoring was adapted and evaluated.",
                "unit_locator": "p. 4",
                "climate_drivers": "enso",
                "biological_responses": "spatial_distribution",
                "fishery_responses": "availability_to_fleet",
                "management_measures": "adaptive_survey_design",
                "implementation_stage": "evaluated_outcome",
                "implementation_evidence_level": "evaluated",
                "proposed_use": "direct_operational_input",
                "synthesis_role": "core",
                "quality_overall_rating": "high",
                "transferability_class": "high",
                "normalized_score": "0.86",
                "score__operational_specificity": "2",
                "score__data_feasibility_peru": "2",
                "data_requirements_peru": "acoustic distribution | SST",
                "institutional_requirements_peru": "IMARPE survey protocol",
                "adaptation_required": "validate local thresholds",
                "critical_caveats": "Local effectiveness remains to be tested.",
            },
            {
                "finding_id": "finding_002",
                "source_id": "source_002",
                "title": "Proposed adaptive monitoring",
                "evidence_summary": (
                    "The study recommends adaptive survey design but does not "
                    "evaluate implementation."
                ),
                "supporting_excerpt": "Adaptive surveys are recommended.",
                "unit_locator": "p. 8",
                "climate_drivers": "ocean_warming",
                "biological_responses": "spatial_distribution",
                "fishery_responses": "",
                "management_measures": "adaptive_survey_design",
                "implementation_stage": "proposed",
                "implementation_evidence_level": "proposed_or_planned",
                "proposed_use": "management_option",
                "synthesis_role": "supporting",
                "quality_overall_rating": "high",
                "transferability_class": "moderate",
                "normalized_score": "0.64",
                "score__operational_specificity": "1",
                "score__data_feasibility_peru": "2",
                "data_requirements_peru": "acoustic distribution",
                "institutional_requirements_peru": "survey planning review",
                "adaptation_required": "define trigger rules",
                "critical_caveats": "Recommendation is not implementation evidence.",
            },
            {
                "finding_id": "finding_003",
                "source_id": "source_003",
                "title": "Scenario evidence",
                "evidence_summary": (
                    "A scenario analysis explores dynamic closure responses under "
                    "warming without observed implementation."
                ),
                "supporting_excerpt": "Dynamic closures were tested in scenarios.",
                "unit_locator": "p. 11",
                "climate_drivers": "ocean_warming",
                "biological_responses": "spatial_distribution",
                "fishery_responses": "fishing_ground_shift",
                "management_measures": "dynamic_closure",
                "implementation_stage": "proposed",
                "implementation_evidence_level": "proposed_or_planned",
                "proposed_use": "scenario_or_operating_model",
                "synthesis_role": "supporting",
                "quality_overall_rating": "moderate",
                "transferability_class": "moderate",
                "normalized_score": "0.57",
                "score__operational_specificity": "1",
                "score__data_feasibility_peru": "1",
                "data_requirements_peru": "vessel tracking | juvenile incidence",
                "institutional_requirements_peru": "closure simulation capacity",
                "adaptation_required": "parameterize Peruvian fleet response",
                "critical_caveats": "Scenario result is not observed effectiveness.",
            },
        ]
    )


@pytest.fixture
def gaps():
    return pd.DataFrame(
        [
            {
                "finding_id": "finding_002",
                "source_id": "source_002",
                "synthesis_theme": "management_and_governance",
                "gap_type": "implementation_evidence_missing",
                "gap_description": (
                    "A management option is present without evaluated implementation evidence."
                ),
            },
            {
                "finding_id": "finding_003",
                "source_id": "source_003",
                "synthesis_theme": "management_and_governance",
                "gap_type": "implementation_evidence_missing",
                "gap_description": (
                    "A management option is present without evaluated implementation evidence."
                ),
            },
        ]
    )


def test_load_config(config):
    assert config["project"]["target_fishery"].startswith("North-central")
    assert config["readiness_scoring"]["maximum_score"] == 13


def test_build_chains_preserves_distinctions(matrix, config):
    chains = build_climate_response_management_chains(matrix, config)
    assert len(chains) == 3
    assert chains.loc[0, "chain_class"] == "complete_management_chain"
    assert chains.loc[1, "chain_class"] == "management_link_without_full_pressure_response"
    assert chains["chain_id"].is_unique


def test_portfolio_aggregates_explicit_measures(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    assert set(portfolio["management_measure"]) == {
        "adaptive_survey_design",
        "dynamic_closure",
    }
    adaptive = portfolio.loc[
        portfolio["management_measure"].eq("adaptive_survey_design")
    ].iloc[0]
    assert adaptive["findings"] == 2
    assert adaptive["sources"] == 2
    assert adaptive["evaluated_implementation"] == 1
    assert adaptive["readiness_total_score"] <= 13


def test_decision_support_ready_requires_evaluated_implementation(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    adaptive = portfolio.loc[
        portfolio["management_measure"].eq("adaptive_survey_design")
    ].iloc[0]
    dynamic = portfolio.loc[
        portfolio["management_measure"].eq("dynamic_closure")
    ].iloc[0]
    assert adaptive["readiness_class"] in {
        "decision_support_ready",
        "pilot_ready",
        "evidence_development",
    }
    assert dynamic["readiness_class"] != "decision_support_ready"


def test_operational_readiness_has_guardrail_notes(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    readiness = build_operational_readiness_matrix(portfolio)
    assert len(readiness) == len(portfolio)
    assert readiness["readiness_gate_note"].str.len().min() > 20


def test_research_priorities_do_not_treat_absence_as_no_effect(gaps, matrix, config):
    priorities = build_research_priority_matrix(gaps, matrix, config)
    assert len(priorities) == 1
    assert priorities.loc[0, "gap_type"] == "implementation_evidence_missing"
    assert "missing evidence" in priorities.loc[0, "priority_statement"]
    assert "evidence of no effect" in priorities.loc[0, "priority_statement"]


def test_recommendations_are_conditional(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    assert len(framework) == len(portfolio)
    assert framework["expert_review_status"].eq("not_reviewed").all()
    text = " ".join(framework["candidate_statement"]).lower()
    assert any(term in text for term in ["assess", "test", "pilot", "subject to"])
    assert "should be implemented immediately" not in text


def test_review_initialisation_preserves_human_work(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    existing = framework.copy()
    existing.loc[0, "expert_review_status"] = "corrected"
    existing.loc[0, "expert_review_notes"] = "Reviewed by fisheries experts."
    refreshed = initialise_recommendation_review(framework, existing)
    assert refreshed.loc[0, "expert_review_status"] == "corrected"
    assert refreshed.loc[0, "expert_review_notes"] == "Reviewed by fisheries experts."


def test_split_review(matrix, config):
    portfolio = build_management_option_portfolio(matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    framework.loc[0, "expert_review_status"] = "accepted"
    groups = split_recommendation_review(framework)
    assert len(groups["validated"]) == 1
    assert len(groups["pending"]) == len(framework) - 1


def test_validation_passes_for_generated_outputs(gaps, matrix, config):
    chains = build_climate_response_management_chains(matrix, config)
    portfolio = build_management_option_portfolio(matrix, config)
    readiness = build_operational_readiness_matrix(portfolio)
    priorities = build_research_priority_matrix(gaps, matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    issues = validate_decision_support_outputs(
        chains,
        portfolio,
        readiness,
        priorities,
        framework,
        config,
    )
    assert issues.empty


def test_validation_detects_unconditional_overclaim(gaps, matrix, config):
    chains = build_climate_response_management_chains(matrix, config)
    portfolio = build_management_option_portfolio(matrix, config)
    readiness = build_operational_readiness_matrix(portfolio)
    priorities = build_research_priority_matrix(gaps, matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    framework.loc[0, "candidate_statement"] = (
        "This measure is proven effective and should be implemented immediately."
    )
    issues = validate_decision_support_outputs(
        chains,
        portfolio,
        readiness,
        priorities,
        framework,
        config,
    )
    assert "prohibited_effectiveness_or_immediacy_claim" in set(issues["issue"])


def test_decision_support_summary(gaps, matrix, config):
    chains = build_climate_response_management_chains(matrix, config)
    portfolio = build_management_option_portfolio(matrix, config)
    readiness = build_operational_readiness_matrix(portfolio)
    priorities = build_research_priority_matrix(gaps, matrix, config)
    framework = build_anchoveta_recommendation_framework(portfolio, config)
    summary = build_decision_support_summary(
        chains,
        portfolio,
        readiness,
        priorities,
        framework,
    )
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["chains_total"] == 3
    assert metrics["management_options"] == 2
    assert metrics["recommendations_not_reviewed"] == 2
