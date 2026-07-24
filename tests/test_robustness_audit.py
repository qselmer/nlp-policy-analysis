from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_review.decision_support_core import stable_id
from evidence_review.decision_support_portfolio import build_management_option_portfolio
from evidence_review.robustness_audit import (
    build_evidence_independence_summary,
    build_recommendation_stability,
    build_robustness_summary,
    build_sensitivity_scenario_results,
    build_source_concentration_summary,
    build_source_influence_matrix,
    leave_one_source_out_summary,
    load_robustness_config,
    scenario_retention_summary,
    split_stability_outputs,
    validate_robustness_outputs,
)
from evidence_review.sensitivity_scenarios import (
    adjusted_decision_support_config,
    filter_evidence_for_scenario,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robustness_audit.yml"


@pytest.fixture
def config():
    return load_robustness_config(CONFIG_PATH, project_root=ROOT)


@pytest.fixture
def matrix():
    records = []
    robust_sources = ["source_a", "source_a", "source_b", "source_c", "source_d", "source_d"]
    for index, source_id in enumerate(robust_sources, start=1):
        records.append(
            {
                "finding_id": f"robust_{index}",
                "source_id": source_id,
                "title": "Comparable implemented evidence",
                "finding_type": "empirical_result",
                "method_or_design": "observational_time_series",
                "management_measures": "dynamic_closure",
                "quality_overall_rating": "high",
                "transferability_class": "high",
                "normalized_score": "0.90",
                "score__operational_specificity": "2",
                "score__data_feasibility_peru": "2",
                "implementation_evidence_level": "evaluated",
                "proposed_use": "direct_operational_input",
                "synthesis_role": "core",
            }
        )
    records.append(
        {
            "finding_id": "weak_1",
            "source_id": "source_e",
            "title": "Single-source proposal",
            "finding_type": "recommendation",
            "method_or_design": "conceptual",
            "management_measures": "early_warning_system",
            "quality_overall_rating": "moderate",
            "transferability_class": "moderate",
            "normalized_score": "0.55",
            "score__operational_specificity": "1",
            "score__data_feasibility_peru": "1",
            "implementation_evidence_level": "proposed_or_planned",
            "proposed_use": "management_option",
            "synthesis_role": "supporting",
        }
    )
    return pd.DataFrame(records)


def _products(matrix, config):
    decision = config["decision_support_config"]
    portfolio = build_management_option_portfolio(matrix, decision)
    scenarios = build_sensitivity_scenario_results(matrix, decision, config)
    scenario_summary = scenario_retention_summary(scenarios, config)
    influence = build_source_influence_matrix(matrix, decision, config)
    loso = leave_one_source_out_summary(influence)
    concentration = build_source_concentration_summary(matrix, config)
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": stable_id("recommendation", row.option_id),
                "option_id": row.option_id,
                "management_measure": row.management_measure,
                "candidate_statement": "Assess this candidate under local testing.",
                "expert_review_status": "accepted",
            }
            for row in portfolio.itertuples()
        ]
    )
    stability = build_recommendation_stability(
        recommendations,
        portfolio,
        scenario_summary,
        loso,
        concentration,
        config,
    )
    return portfolio, scenarios, influence, concentration, stability


def test_load_config(config):
    assert "base" in config["scenarios"]
    assert len(config["scenarios"]) == 8
    assert config["decision_support_config"]["readiness_scoring"]["maximum_score"] == 13


def test_filter_high_quality_and_observational(matrix, config):
    high = filter_evidence_for_scenario(
        matrix,
        config["scenarios"]["high_quality_only"],
    )
    assert len(high) == 6
    assert set(high["quality_overall_rating"]) == {"high"}

    observational = filter_evidence_for_scenario(
        matrix,
        config["scenarios"]["observational_only"],
    )
    assert len(observational) == 6
    assert set(observational["finding_type"]) == {"empirical_result"}


def test_threshold_shift_preserves_order(config):
    conservative = adjusted_decision_support_config(
        config["decision_support_config"],
        0.10,
    )
    thresholds = conservative["readiness_thresholds"]
    assert thresholds["evidence_development_minimum"] == pytest.approx(0.35)
    assert thresholds["pilot_ready_minimum"] == pytest.approx(0.60)
    assert thresholds["decision_support_ready_minimum"] == pytest.approx(0.85)


def test_scenarios_cover_every_base_option(matrix, config):
    decision = config["decision_support_config"]
    portfolio = build_management_option_portfolio(matrix, decision)
    results = build_sensitivity_scenario_results(matrix, decision, config)
    assert len(portfolio) == 2
    assert len(results) == 2 * len(config["scenarios"])
    assert not results.duplicated(["scenario_id", "option_id"]).any()
    assert set(results["scenario_id"]) == set(config["scenarios"])


def test_leave_one_source_out_and_concentration(matrix, config):
    decision = config["decision_support_config"]
    portfolio = build_management_option_portfolio(matrix, decision)
    influence = build_source_influence_matrix(matrix, decision, config)
    assert len(influence) == len(portfolio) * matrix["source_id"].nunique()

    concentration = build_source_concentration_summary(matrix, config)
    robust = concentration.loc[
        concentration["management_measure"].eq("dynamic_closure")
    ].iloc[0]
    weak = concentration.loc[
        concentration["management_measure"].eq("early_warning_system")
    ].iloc[0]
    assert robust["sources"] == 4
    assert robust["top_source_share"] == pytest.approx(2 / 6)
    assert weak["top_source_share"] == pytest.approx(1.0)
    assert weak["source_concentration_class"] == "high"


def test_evidence_independence_distinguishes_findings_and_sources(matrix, config):
    portfolio = build_management_option_portfolio(
        matrix,
        config["decision_support_config"],
    )
    concentration = build_source_concentration_summary(matrix, config)
    independence = build_evidence_independence_summary(portfolio, concentration)
    robust = independence.loc[
        independence["management_measure"].eq("dynamic_closure")
    ].iloc[0]
    assert robust["findings"] == 6
    assert robust["independent_sources"] == 4
    assert robust["effective_source_number"] > 3


def test_recommendation_stability_classification(matrix, config):
    portfolio, scenarios, influence, concentration, stability = _products(matrix, config)
    robust = stability.loc[
        stability["management_measure"].eq("dynamic_closure")
    ].iloc[0]
    weak = stability.loc[
        stability["management_measure"].eq("early_warning_system")
    ].iloc[0]
    assert robust["stability_class"] == "robust"
    assert robust["scenario_retention_rate"] == pytest.approx(1.0)
    assert robust["leave_one_source_out_retention"] == pytest.approx(1.0)
    assert weak["stability_class"] == "insufficiently_supported"

    split = split_stability_outputs(stability)
    assert len(split["robust"]) == 1
    assert len(split["sensitive"]) == 1


def test_validation_and_summary(matrix, config):
    portfolio, scenarios, influence, concentration, stability = _products(matrix, config)
    issues = validate_robustness_outputs(
        scenarios,
        influence,
        concentration,
        stability,
        config,
    )
    assert issues.empty
    summary = build_robustness_summary(
        matrix,
        scenarios,
        influence,
        concentration,
        stability,
        issues,
    )
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["evidence_findings"] == 7
    assert metrics["independent_sources"] == 5
    assert metrics["recommendations_audited"] == 2
    assert metrics["validation_issues"] == 0


def test_validation_detects_duplicate_and_bad_rate(matrix, config):
    _, scenarios, influence, concentration, stability = _products(matrix, config)
    bad = pd.concat([scenarios, scenarios.iloc[[0]]], ignore_index=True)
    stability = stability.copy()
    stability.loc[0, "scenario_retention_rate"] = 1.2
    issues = validate_robustness_outputs(
        bad,
        influence,
        concentration,
        stability,
        config,
    )
    assert "duplicate_option_scenario" in set(issues["issue"])
    assert "rate_out_of_range" in set(issues["issue"])
