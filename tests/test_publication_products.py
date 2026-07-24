from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def sample_data():
    matrix = pd.DataFrame(
        [
            {
                "finding_id": "f1",
                "source_id": "s1",
                "synthesis_theme": "management_and_governance",
                "quality_overall_rating": "high",
                "transferability_class": "high",
                "implementation_evidence_level": "evaluated",
                "mechanism_chain_stage_count": "4",
                "climate_drivers": "marine_heatwave",
                "biological_responses": "distribution",
                "fishery_responses": "availability",
                "management_measures": "climate_risk_assessment",
            },
            {
                "finding_id": "f2",
                "source_id": "s2",
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
                "finding_id": "f3",
                "source_id": "s3",
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
        ]
    )
    portfolio = pd.DataFrame(
        [
            {
                "option_id": "o1",
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
                "option_id": "o2",
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
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": "r1",
                "option_id": "o1",
                "management_measure": "climate_risk_assessment",
                "candidate_statement": "Subject to expert review, assess integration in existing advice procedures.",
                "expert_review_status": "accepted",
            },
            {
                "recommendation_id": "r2",
                "option_id": "o2",
                "management_measure": "catch_limit",
                "candidate_statement": "Test catch-limit alternatives in management strategy evaluation before adoption.",
                "expert_review_status": "corrected",
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "recommendation_id": "r1",
                "option_id": "o1",
                "management_measure": "climate_risk_assessment",
                "base_readiness_class": "decision_support_ready",
                "base_decision_pathway": "operational_integration_candidate",
                "findings": 2,
                "sources": 2,
                "scenario_retention_rate": 0.875,
                "leave_one_source_out_retention": 1.0,
                "top_source_id": "s1",
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
                "recommendation_id": "r2",
                "option_id": "o2",
                "management_measure": "catch_limit",
                "base_readiness_class": "pilot_ready",
                "base_decision_pathway": "mse_or_scenario_candidate",
                "findings": 2,
                "sources": 2,
                "scenario_retention_rate": 0.50,
                "leave_one_source_out_retention": 1.0,
                "top_source_id": "s2",
                "top_source_share": 0.50,
                "source_hhi": 0.50,
                "effective_source_number": 2.0,
                "evaluated_implementation_findings": 0,
                "implementation_dependency": "no_evaluated_implementation",
                "readiness_class_changes": 4,
                "decision_pathway_changes": 2,
                "stability_class": "assumption_sensitive",
            },
        ]
    )
    priorities = pd.DataFrame(
        [
            {
                "priority_id": "p1",
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
    independence = pd.DataFrame(
        [
            {"option_id": "o1", "findings_per_source": 1.0, "independence_flag": "multi_source"},
            {"option_id": "o2", "findings_per_source": 1.0, "independence_flag": "multi_source"},
        ]
    )
    return matrix, portfolio, recommendations, stability, priorities, independence


def test_phase13_tables_and_validation():
    config = load_publication_config(
        ROOT / "config" / "publication_products.yml", project_root=ROOT
    )
    matrix, portfolio, recommendations, stability, priorities, independence = sample_data()
    evidence = build_evidence_summary(matrix)
    management = build_management_portfolio_table(portfolio, stability)
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    source = build_source_influence_table(stability, independence)
    metrics = build_publication_metrics(matrix, management, final, gaps)
    assert evidence["findings"].sum() == 3
    assert len(management) == len(final) == len(source) == 2
    assert set(final["publication_tier"]) == {
        "local_validation_priority",
        "scenario_evaluation_required",
    }
    assert "robustness_audit" in set(gaps["gap_origin"])
    assert int(metrics.loc[metrics["metric"].eq("independent_sources"), "value"].iloc[0]) == 3
    issues = validate_publication_tables(
        management,
        final,
        gaps,
        config,
        expected_recommendation_ids=["r1", "r2"],
        robustness_issues=pd.DataFrame(),
    )
    assert issues.empty
    bad = final.copy()
    bad.loc[0, "final_statement"] = "This option is proven effective."
    bad_issues = validate_publication_tables(management, bad, gaps, config)
    assert "prohibited_effectiveness_or_immediacy_claim" in set(bad_issues["issue"])


def test_phase13_figures_reports_and_manifest(tmp_path):
    config = load_publication_config(
        ROOT / "config" / "publication_products.yml", project_root=ROOT
    )
    matrix, portfolio, recommendations, stability, priorities, independence = sample_data()
    evidence = build_evidence_summary(matrix)
    management = build_management_portfolio_table(portfolio, stability)
    final = build_final_recommendations(recommendations, stability, config)
    gaps = build_research_gap_table(priorities, final)
    source = build_source_influence_table(stability, independence)
    metrics = build_publication_metrics(matrix, management, final, gaps)
    figures = generate_publication_figures(
        matrix=matrix,
        portfolio=management,
        final_recommendations=final,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        settings=config["figure_settings"],
    )
    reports = write_publication_reports(
        evidence_summary=evidence,
        portfolio=management,
        final_recommendations=final,
        research_gaps=gaps,
        source_table=source,
        metrics=metrics,
        root=tmp_path,
        paths=config["paths"],
        config=config,
    )
    assert len(figures) == 6
    assert len(reports) == 3
    keys = list(figures["product_key"]) + list(reports["product_key"])
    manifest = build_product_manifest(tmp_path, config["paths"], keys)
    assert manifest["exists"].all()
    assert (manifest["size_bytes"] > 0).all()
    assert reports["characters"].min() >= 500
