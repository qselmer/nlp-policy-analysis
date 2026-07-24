from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_review.publication_products import (
    build_evidence_summary,
    build_final_recommendations,
    build_management_portfolio_table,
    build_product_manifest,
    build_publication_metrics,
    build_research_gap_table,
    build_source_influence_table,
    generate_publication_figures,
    load_publication_config,
    validate_publication_tables,
    write_publication_reports,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "publication_products.yml"


@pytest.fixture
def config():
    return load_publication_config(CONFIG_PATH, project_root=ROOT)


@pytest.fixture
def matrix():
    return pd.DataFrame(
        [
            {
                "finding_id": "finding_001",
                "source_id": "source_001",
                "synthesis_theme": "management_and_governance",
                "quality_overall_rating": "high",
                "transferability_class": "high",
                "implementation_evidence_level": "evaluated",
                "mechanism_chain_stage_count": "4",
                "climate_drivers": "marine_heatwave",
                "biological_responses": "spatial_distribution",
                "fishery_responses": "fleet_availability",
                "management_measures": "climate_risk_assessment",
            },
            {
                "finding_id": "finding_002",
                "source_id": "source_002",
                "synthesis_theme": "management_and_governance",
                "quality_overall_rating": "moderate",
                "transferability_class": "moderate",
                "implementation_evidence_level": "implemented_or_piloted",
                "mechanism_chain_stage_count": "3",
                "climate_drivers": "enso",
                "biological_responses": "recruitment",
                "fishery_responses": "catch_variability",
                "management_measures": "climate_risk_assessment | catch_limit",
            },
            {
                "finding_id": "finding_003",
                "source_id": "source_003",
                "synthesis_theme": "ecological_response",
                "quality_overall_rating": "high",
                "transferability_class": "moderate",
                "implementation_evidence_level": "proposed_or_planned",
                "mechanism_chain_stage_count": "2",
                "climate_drivers": "ocean_warming",
                "biological_responses": "growth",
                "fishery_responses": "",
                "management_measures": "catch_limit",
            },
            {
                "finding_id": "finding_004",
                "source_id": "source_003",
                "synthesis_theme": "implementation",
                "quality_overall_rating": "high",
                "transferability_class": "low",
                "implementation_evidence_level": "not_applicable_or_unclear",
                "mechanism_chain_stage_count": "1",
                "climate_drivers": "",
                "biological_responses": "",
                "fishery_responses": "",
                "management_measures": "seasonal_closure",
            },
        ]
    )


@pytest.fixture
def portfolio():
    return pd.DataFrame(
        [
            {
                "option_id": "option_risk",
                "management_measure": "climate_risk_assessment",
                "findings": 2,
                "sources": 2,
                "high_quality": 1,
                "high_transferability": 1,
                "evaluated_implementation": 1,
                "implemented_or_piloted": 1,
                "proposed_or_planned": 0,
                "mean_transferability_score": "0.78",
                "mean_operational_specificity": "1.50",
                "mean_data_feasibility_peru": "1.50",
                "readiness_normalized_score": "0.77",
                "readiness_class": "decision_support_ready",
                "decision_pathway": "operational_integration_candidate",
            },
            {
                "option_id": "option_catch",
                "management_measure": "catch_limit",
                "findings": 2,
                "sources": 2,
                "high_quality": 1,
                "high_transferability": 0,
                "evaluated_implementation": 0,
                "implemented_or_piloted": 1,
                "proposed_or_planned": 1,
                "mean_transferability_score": "0.61",
                "mean_operational_specificity": "1.00",
                "mean_data_feasibility_peru": "1.50",
                "readiness_normalized_score": "0.62",
                "readiness_class": "pilot_ready",
                "decision_pathway": "mse_or_scenario_candidate",
            },
            {
                "option_id": "option_closure",
                "management_measure": "seasonal_closure",
                "findings": 1,
                "sources": 1,
                "high_quality": 1,
                "high_transferability": 0,
                "evaluated_implementation": 0,
                "implemented_or_piloted": 0,
                "proposed_or_planned": 0,
                "mean_transferability_score": "0.42",
                "mean_operational_specificity": "0.50",
                "mean_data_feasibility_peru": "1.00",
                "readiness_normalized_score": "0.31",
                "readiness_class": "evidence_development",
                "decision_pathway": "research_priority",
            },
        ]
    )


@pytest.fixture
def recommendations():
    return pd.DataFrame(
        [
            {
                "recommendation_id": "rec_risk",
                "option_id": "option_risk",
                "management_measure": "climate_risk_assessment",
                "candidate_statement": (
                    "Subject to expert review, assess climate risk assessment in existing advice procedures."
                ),
                "expert_review_status": "accepted",
            },
            {
                "recommendation_id": "rec_catch",
                "option_id": "option_catch",
                "management_measure": "catch_limit",
                "candidate_statement": (
                    "Test catch limit alternatives in management strategy evaluation before adoption."
                ),
                "expert_review_status": "corrected",
            },
            {
                "recommendation_id": "rec_closure",
                "option_id": "option_closure",
                "management_measure": "seasonal_closure",
                "candidate_statement": (
                    "Prioritize research on seasonal closure before management application."
                ),
                "expert_review_status": "accepted",
            },
        ]
    )


@pytest.fixture
def stability():
    return pd.DataFrame(
        [
            {
                "recommendation_id": "rec_risk",
                "option_id": "option_risk",
                "management_measure": "climate_risk_assessment",
                "base_readiness_class": "decision_support_ready",
                "base_decision_pathway": "operational_integration_candidate",
                "findings": 2,
                "sources": 2,
                "scenario_retention_rate": 0.875,
                "leave_one_source_out_retention": 1.0,
                "top_source_id": "source_001",
                "top_source_share": 0.50,
                "source_hhi": 0.50,
                "effective_source_number": 2.0,
                "evaluated_implementation_findings": 1,
                "implementation_dependency": "single_source",
                "readiness_class_changes": 1,
                "decision_pathway_changes": 1,
                "stability_class": "generally_stable",
            },
            {
                "recommendation_id": "rec_catch",
                "option_id": "option_catch",
                "management_measure": "catch_limit",
                "base_readiness_class": "pilot_ready",
                "base_decision_pathway": "mse_or_scenario_candidate",
                "findings": 2,
                "sources": 2,
                "scenario_retention_rate": 0.50,
                "leave_one_source_out_retention": 1.0,
                "top_source_id": "source_002",
                "top_source_share": 0.50,
                "source_hhi": 0.50,
                "effective_source_number": 2.0,
                "evaluated_implementation_findings": 0,
                "implementation_dependency": "no_evaluated_implementation",
                "readiness_class_changes": 4,
                "decision_pathway_changes": 2,
                "stability_class": "assumption_sensitive",
            },
            {
                "recommendation_id": "rec_closure",
                "option_id": "option_closure",
                "management_measure": "seasonal_closure",
                "base_readiness_class": "evidence_development",
                "base_decision_pathway": "research_priority",
                "findings": 1,
                "sources": 1,
                "scenario_retention_rate": 0.875,
                "leave_one_source_out_retention": 0.9375,
                "top_source_id": "source_003",
                "top_source_share": 1.0,
                "source_hhi": 1.0,
                "effective_source_number": 1.0,
                "evaluated_implementation_findings": 0,
                "implementation_dependency": "no_evaluated_implementation",
                "readiness_class_changes": 1,
                "decision_pathway_changes": 0,
                "stability_class": "insufficiently_supported",
            },
        ]
    )


@pytest.fixture
def independence():
    return pd.DataFrame(
        [
            {
                "option_id": "option_risk",
                "findings_per_source": 1.0,
                "independence_flag": "multi_source",
            },
            {
                "option_id": "option_catch",
                "findings_per_source": 1.0,
                "independence_flag": "multi_source",
            },
            {
                "option_id": "option_closure",
                "findings_per_source": 1.0,
                "independence_flag": "single_source",
            },
        ]
    )


@pytest.fixture
def priorities():
    return pd.DataFrame(
        [
            {
                "priority_id": "priority_001",
                "gap_type": "implementation_evidence_missing",
                "synthesis_theme": "management_and_governance",
                "findings": 2,
                "sources": 2,
                "priority_score": 4,
                "priority_class": "moderate",
                "priority_statement": "Evaluate implementation outcomes locally.",
                "interpretation_guardrail": "Missing evidence is not evidence of no effect.",
            }
        ]
    )


def test_load_config(config):
    assert config["project"]["target_fishery"].startswith("North-central")
    assert config["stability_to_publication_tier"]["robust"] == "conditional_priority"


def test_build_evidence_summary(matrix):
    summary = build_evidence_summary(matrix)
    management = summary.loc[
        summary["synthesis_theme"].eq("management_and_governance")
    ].iloc[0]
    assert management["findings"] == 2
    assert management["independent_sources"] == 2
    assert management["evaluated_implementation"] == 1
    assert management["complete_mechanism_chains"] == 1


def test_build_management_portfolio_table(portfolio, stability):
    table = build_management_portfolio_table(portfolio, stability)
    assert len(table) == 3
    assert table.iloc[0]["management_measure"] == "climate_risk_assessment"
    assert table.loc[
        table["option_id"].eq("option_catch"), "stability_class"
    ].iloc[0] == "assumption_sensitive"


def test_final_recommendation_hierarchy(recommendations, stability, config):
    final = build_final_recommendations(recommendations, stability, config)
    tiers = dict(zip(final["management_measure"], final["publication_tier"]))
    assert tiers["climate_risk_assessment"] == "local_validation_priority"
    assert tiers["catch_limit"] == "scenario_evaluation_required"
    assert tiers["seasonal_closure"] == "research_or_bounded_pilot"
    assert "does not demonstrate" in final.loc[
        final["management_measure"].eq("climate_risk_assessment"),
        "publication_interpretation",
    ].iloc[0]


def test_research_and_source_tables(
    recommendations,
    stability,
    priorities,
    independence,
    config,
):
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    assert "robustness_audit" in set(gaps["gap_origin"])
    source = build_source_influence_table(stability, independence)
    assert len(source) == 3
    assert source.iloc[0]["top_source_share"] == 1.0


def test_validation_passes_and_detects_prohibited_claim(
    portfolio,
    recommendations,
    stability,
    priorities,
    config,
):
    portfolio_table = build_management_portfolio_table(portfolio, stability)
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    issues = validate_publication_tables(
        portfolio_table,
        final,
        gaps,
        config,
        expected_recommendation_ids=list(recommendations["recommendation_id"]),
        robustness_issues=pd.DataFrame(),
    )
    assert issues.empty
    bad = final.copy()
    bad.loc[0, "final_statement"] = "This measure is proven effective."
    bad_issues = validate_publication_tables(
        portfolio_table,
        bad,
        gaps,
        config,
        expected_recommendation_ids=list(recommendations["recommendation_id"]),
    )
    assert "prohibited_effectiveness_or_immediacy_claim" in set(bad_issues["issue"])


def test_generate_figures_and_reports(
    tmp_path,
    matrix,
    portfolio,
    recommendations,
    stability,
    priorities,
    independence,
    config,
):
    portfolio_table = build_management_portfolio_table(portfolio, stability)
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    source = build_source_influence_table(stability, independence)
    summary = build_evidence_summary(matrix)
    metrics = build_publication_metrics(matrix, portfolio_table, final, gaps)
    figures = generate_publication_figures(
        matrix=matrix,
        portfolio=portfolio_table,
        final_recommendations=final,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        settings=config["figure_settings"],
    )
    assert len(figures) == 6
    assert all((tmp_path / path).exists() for path in figures["relative_path"])
    reports = write_publication_reports(
        evidence_summary=summary,
        portfolio=portfolio_table,
        final_recommendations=final,
        research_gaps=gaps,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        config=config,
    )
    assert len(reports) == 3
    assert reports["characters"].min() >= 500
    assert all((tmp_path / path).exists() for path in reports["relative_path"])


def test_product_manifest(
    tmp_path,
    matrix,
    portfolio,
    recommendations,
    stability,
    priorities,
    independence,
    config,
):
    portfolio_table = build_management_portfolio_table(portfolio, stability)
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    source = build_source_influence_table(stability, independence)
    summary = build_evidence_summary(matrix)
    metrics = build_publication_metrics(matrix, portfolio_table, final, gaps)
    generate_publication_figures(
        matrix=matrix,
        portfolio=portfolio_table,
        final_recommendations=final,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        settings=config["figure_settings"],
    )
    write_publication_reports(
        evidence_summary=summary,
        portfolio=portfolio_table,
        final_recommendations=final,
        research_gaps=gaps,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        config=config,
    )
    keys = [
        "evidence_flow_png",
        "quality_transferability_png",
        "management_readiness_png",
        "recommendation_stability_png",
        "source_concentration_png",
        "climate_response_management_network_png",
        "evidence_synthesis_report_md",
        "methods_and_results_md",
        "executive_decision_brief_md",
    ]
    manifest = build_product_manifest(tmp_path, config["paths"], keys)
    assert manifest["exists"].all()
    assert (manifest["size_bytes"] > 0).all()
