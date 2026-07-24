from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evidence_review.transferability_synthesis import (
    build_evidence_gap_matrix,
    build_evidence_synthesis_matrix,
    build_management_option_summary,
    build_recommendation_candidates,
    build_synthesis_theme_summary,
    build_transferability_prompt,
    expected_evidence_strength,
    expected_transferability_class,
    export_transferability_prompts_jsonl,
    initialise_transferability_sheet,
    load_transferability_config,
    merge_transferability_assessments,
    read_transferability_responses_jsonl,
    score_column,
    split_transferability_outputs,
    transferability_summary,
    validate_transferability_sheet,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "transferability_synthesis.yml"


@pytest.fixture
def config():
    return load_transferability_config(CONFIG_PATH, project_root=ROOT)


@pytest.fixture
def finding():
    return pd.DataFrame(
        [
            {
                "finding_id": "finding_001",
                "source_id": "source_001",
                "title": "Comparable anchovy evidence",
                "source_type": "scientific_article",
                "finding_type": "empirical_result",
                "evidence_streams": "climate_hazard | ecological_response",
                "unit_locator": "p. 4",
                "supporting_excerpt": (
                    "Ocean warming shifted the spatial distribution of anchovy."
                ),
                "evidence_summary": (
                    "Ocean warming shifted anchovy distribution and altered "
                    "availability to the fleet."
                ),
                "fishery_scope": "small_pelagics",
                "species": "anchovy",
                "climate_drivers": "ocean_warming",
                "biological_responses": "spatial_distribution",
                "fishery_responses": "availability_to_fleet",
                "management_measures": "adaptive_survey_design",
                "implementation_stage": "implemented",
                "quality_overall_rating": "high",
                "quality_normalized_score": "0.86",
                "quality_human_validation_status": "accepted",
            }
        ]
    )


def completed_response():
    return {
        "finding_id": "finding_001",
        "source_id": "source_001",
        "transferability_status": "completed",
        "proposed_use": "monitoring_indicator",
        "synthesis_role": "core",
        "scores": {
            "ecological_similarity": {
                "score": 3,
                "note": (
                    "The source concerns anchovy and a directly comparable "
                    "small-pelagic context."
                ),
            },
            "climate_relevance": {
                "score": 2,
                "note": (
                    "The finding documents an explicit warming-to-distribution "
                    "mechanism."
                ),
            },
            "management_relevance": {
                "score": 2,
                "note": (
                    "The result directly informs acoustic monitoring and "
                    "survey interpretation."
                ),
            },
            "evidence_strength": {
                "score": 0,
                "note": "This model-provided value must be ignored.",
            },
            "operational_specificity": {
                "score": 1,
                "note": (
                    "A monitoring application is specified, but no automatic "
                    "decision rule is evaluated."
                ),
            },
            "data_feasibility_peru": {
                "score": 2,
                "note": (
                    "Acoustic distribution and environmental observations are "
                    "routinely available."
                ),
            },
        },
        "data_requirements_peru": [
            "acoustic distribution",
            "sea-surface temperature",
        ],
        "institutional_requirements_peru": [
            "integration in IMARPE survey interpretation"
        ],
        "adaptation_required": ["define local thresholds"],
        "transferability_summary": (
            "The finding can support environmentally informed interpretation "
            "of acoustic distribution and adaptive survey design for the "
            "north-central anchoveta stock."
        ),
        "critical_caveats": [
            "The management effectiveness of the proposed application was not evaluated."
        ],
        "reviewer_notes": "",
    }


def test_load_config_and_expected_scores(config):
    assert expected_evidence_strength("high", config) == 2
    assert expected_evidence_strength("moderate", config) == 1
    assert expected_evidence_strength("low", config) == 0
    assert expected_transferability_class(0.75, config) == "high"
    assert expected_transferability_class(0.50, config) == "moderate"
    assert expected_transferability_class(0.49, config) == "low"


def test_initialise_inherits_quality_strength(finding, config):
    sheet = initialise_transferability_sheet(finding, config)
    assert len(sheet) == 1
    assert sheet.loc[0, "transferability_status"] == "pending"
    assert sheet.loc[0, score_column("evidence_strength")] == "2"
    assert "validated source quality" in sheet.loc[
        0, "note__evidence_strength"
    ]


def test_initialise_preserves_human_validated_row(finding, config):
    existing = initialise_transferability_sheet(finding, config)
    existing.loc[0, "human_validation_status"] = "corrected"
    existing.loc[0, "transferability_status"] = "completed"
    existing.loc[0, "score__ecological_similarity"] = "1"
    refreshed = initialise_transferability_sheet(finding, config, existing)
    assert refreshed.loc[0, "human_validation_status"] == "corrected"
    assert refreshed.loc[0, "score__ecological_similarity"] == "1"


def test_prompt_preserves_methodological_distinctions(finding, config):
    sheet = initialise_transferability_sheet(finding, config)
    prompt = json.loads(build_transferability_prompt(sheet.iloc[0], config))
    rules = " ".join(prompt["rules"])
    assert "recommendation or proposal is not evidence of implementation" in rules
    assert "projection or scenario is not an observed trend" in rules
    assert prompt["inherited_evidence_strength"] == 2
    assert prompt["finding"]["finding_id"] == "finding_001"


def test_export_prompts_one_per_finding(tmp_path, finding, config):
    sheet = initialise_transferability_sheet(finding, config)
    output = tmp_path / "prompts.jsonl"
    export_transferability_prompts_jsonl(sheet, config, output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["finding_id"] == "finding_001"


def test_read_response_overrides_evidence_strength(
    tmp_path,
    finding,
    config,
):
    sheet = initialise_transferability_sheet(finding, config)
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        json.dumps(completed_response()) + "\n",
        encoding="utf-8",
    )
    incoming = read_transferability_responses_jsonl(
        response_path,
        sheet,
        config,
    )
    assert incoming.loc[0, "score__evidence_strength"] == "2"
    assert incoming.loc[0, "total_score"] == "12"
    assert incoming.loc[0, "maximum_score"] == "14"
    assert incoming.loc[0, "transferability_class"] == "high"


def test_merge_protects_human_validated_rows(
    tmp_path,
    finding,
    config,
):
    sheet = initialise_transferability_sheet(finding, config)
    sheet.loc[0, "human_validation_status"] = "accepted"
    sheet.loc[0, "transferability_summary"] = "Human-reviewed summary."
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        json.dumps(completed_response()) + "\n",
        encoding="utf-8",
    )
    incoming = read_transferability_responses_jsonl(
        response_path,
        sheet,
        config,
    )
    merged = merge_transferability_assessments(sheet, incoming, config)
    assert merged.loc[0, "transferability_summary"] == "Human-reviewed summary."


def test_completed_response_validates_without_issues(
    tmp_path,
    finding,
    config,
):
    sheet = initialise_transferability_sheet(finding, config)
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        json.dumps(completed_response()) + "\n",
        encoding="utf-8",
    )
    incoming = read_transferability_responses_jsonl(
        response_path,
        sheet,
        config,
    )
    issues = validate_transferability_sheet(incoming, config)
    assert issues.empty


def test_validation_detects_quality_mismatch_and_bad_total(
    tmp_path,
    finding,
    config,
):
    sheet = initialise_transferability_sheet(finding, config)
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        json.dumps(completed_response()) + "\n",
        encoding="utf-8",
    )
    incoming = read_transferability_responses_jsonl(
        response_path,
        sheet,
        config,
    )
    incoming.loc[0, "score__evidence_strength"] = "0"
    incoming.loc[0, "total_score"] = "1"
    issues = validate_transferability_sheet(incoming, config)
    assert "evidence_strength" in " ".join(issues["issue"])
    assert "recorded_1.0_calculated_10" in set(issues["issue"])


def test_split_and_flow(finding, config):
    sheet = initialise_transferability_sheet(finding, config)
    sheet.loc[0, "transferability_status"] = "completed"
    sheet.loc[0, "human_validation_status"] = "accepted"
    groups = split_transferability_outputs(sheet)
    assert len(groups["validated"]) == 1
    flow = transferability_summary(sheet)
    metrics = dict(zip(flow["metric"], flow["value"]))
    assert metrics["findings_total"] == 1
    assert metrics["assessments_accepted"] == 1


def validated_matrix_input(tmp_path, finding, config):
    sheet = initialise_transferability_sheet(finding, config)
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        json.dumps(completed_response()) + "\n",
        encoding="utf-8",
    )
    incoming = read_transferability_responses_jsonl(
        response_path,
        sheet,
        config,
    )
    incoming.loc[0, "human_validation_status"] = "accepted"
    return build_evidence_synthesis_matrix(incoming)


def test_synthesis_matrix_and_theme_summary(
    tmp_path,
    finding,
    config,
):
    matrix = validated_matrix_input(tmp_path, finding, config)
    assert matrix.loc[0, "synthesis_theme"] == "management_and_governance"
    assert matrix.loc[0, "mechanism_chain_stage_count"] == 4
    themes = build_synthesis_theme_summary(matrix)
    assert themes.loc[0, "findings"] == 1
    assert themes.loc[0, "high_transferability"] == 1


def test_management_gap_and_recommendation_outputs(
    tmp_path,
    finding,
    config,
):
    matrix = validated_matrix_input(tmp_path, finding, config)
    management = build_management_option_summary(matrix)
    assert management.loc[0, "management_measure"] == "adaptive_survey_design"
    assert management.loc[0, "implemented_or_piloted"] == 1

    gaps = build_evidence_gap_matrix(matrix)
    assert "implementation_evidence_missing" not in set(
        gaps.get("gap_type", pd.Series(dtype=str))
    )

    recommendations = build_recommendation_candidates(matrix)
    assert len(recommendations) == 1
    assert recommendations.loc[0, "recommendation_readiness"] in {
        "candidate_for_expert_review",
        "supporting_candidate",
    }
