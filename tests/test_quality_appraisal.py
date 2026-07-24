from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_review.quality_appraisal import (
    appraisal_domain_for_source_type,
    appraisal_summary,
    build_finding_quality_link,
    build_quality_appraisal_prompt,
    criteria_for_domain,
    expected_rating,
    initialise_quality_appraisal_sheet,
    load_quality_appraisal_config,
    merge_quality_appraisals,
    note_column,
    locator_column,
    read_quality_appraisal_responses_jsonl,
    score_column,
    select_quality_appraisal_packets,
    split_quality_appraisal_outputs,
    validate_quality_appraisal_sheet,
)


@pytest.fixture()
def config(tmp_path: Path) -> dict:
    taxonomy = tmp_path / "taxonomy.yml"
    taxonomy.write_text(
        "source_type:\n  - scientific_article\n  - technical_report\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "quality.yml"
    config_path.write_text(
        f"""
taxonomy_path: {taxonomy.as_posix()}
paths: {{}}
appraisal_statuses: [pending, completed, insufficient_text, appraisal_error]
human_validation_status: [not_reviewed, accepted, corrected, rejected]
overall_ratings: [low, moderate, high, not_applicable]
score_values:
  0: absent
  1: partial
  2: adequate
rating_thresholds:
  moderate_minimum: 0.5
  high_minimum: 0.8
source_type_to_domain:
  scientific_article: scientific
  technical_report: technical_report
  other: scientific
criteria:
  common:
    source_authenticity: Authenticity
    evidence_traceability: Traceability
  scientific:
    design_appropriateness: Design
    conclusion_support: Conclusions
  technical_report:
    institutional_authority: Authority
    conclusion_support: Conclusions
packet_selection:
  max_chunks_per_source: 4
  max_characters_per_source: 5000
  edge_chunks: 1
  spread_chunks: 1
  keyword_terms: [methods, limitation, results]
validation:
  require_all_domain_criteria_for_completed: true
  require_note_for_scored_criterion: true
  require_locator_for_scored_criterion: true
  require_explicit_page_locator: true
  minimum_note_characters: 12
  minimum_summary_characters: 20
  enforce_calculated_scores: true
  require_human_review_for_final_use: true
principles: []
""",
        encoding="utf-8",
    )
    return load_quality_appraisal_config(
        config_path,
        project_root=tmp_path,
    )


@pytest.fixture()
def sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "Scientific source",
                "source_type": "scientific_article",
                "page_count": "20",
            },
            {
                "source_id": "src_002",
                "title": "Technical source",
                "source_type": "technical_report",
                "page_count": "10",
            },
        ]
    )


@pytest.fixture()
def findings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"finding_id": "f1", "source_id": "src_001"},
            {"finding_id": "f2", "source_id": "src_001"},
            {"finding_id": "f3", "source_id": "src_002"},
        ]
    )


def completed_row(
    sheet: pd.DataFrame,
    config: dict,
    source_id: str = "src_001",
) -> pd.DataFrame:
    output = sheet.copy()
    index = output.index[output["source_id"].eq(source_id)][0]
    domain = output.at[index, "appraisal_domain"]
    criteria = criteria_for_domain(config, domain)
    for criterion in criteria:
        output.at[index, score_column(criterion)] = "2"
        output.at[index, note_column(criterion)] = (
            f"Adequate evidence for {criterion}."
        )
        output.at[index, locator_column(criterion)] = "p. 2"
    output.at[index, "appraisal_status"] = "completed"
    output.at[index, "total_score"] = str(2 * len(criteria))
    output.at[index, "maximum_score"] = str(2 * len(criteria))
    output.at[index, "normalized_score"] = "1.000000"
    output.at[index, "overall_rating"] = "high"
    output.at[index, "appraisal_summary"] = (
        "The source is methodologically strong and well documented."
    )
    output.at[index, "human_validation_status"] = "not_reviewed"
    return output


def test_config_and_domain_mapping(config: dict) -> None:
    assert appraisal_domain_for_source_type(
        "scientific_article",
        config,
    ) == "scientific"
    assert appraisal_domain_for_source_type(
        "technical_report",
        config,
    ) == "technical_report"
    assert criteria_for_domain(config, "scientific") == [
        "source_authenticity",
        "evidence_traceability",
        "design_appropriateness",
        "conclusion_support",
    ]


def test_initialise_sheet_preserves_existing(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    first = initialise_quality_appraisal_sheet(
        sources,
        findings,
        config,
    )
    assert len(first) == 2
    assert first.loc[
        first["source_id"].eq("src_001"),
        "validated_findings_count",
    ].iloc[0] == "2"

    existing = completed_row(first, config)
    existing.loc[
        existing["source_id"].eq("src_001"),
        "human_validation_status",
    ] = "accepted"
    second = initialise_quality_appraisal_sheet(
        sources,
        findings,
        config,
        existing,
    )
    row = second.loc[second["source_id"].eq("src_001")].iloc[0]
    assert row["appraisal_status"] == "completed"
    assert row["human_validation_status"] == "accepted"


def test_packet_selection_is_deterministic(
    config: dict,
    sources: pd.DataFrame,
) -> None:
    corpus = pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "chunk_id": f"c{i}",
                "start_page": i,
                "end_page": i,
                "text": (
                    "methods and results with limitations"
                    if i == 3
                    else f"ordinary text {i}"
                ),
            }
            for i in range(1, 7)
        ]
    )
    first = select_quality_appraisal_packets(
        corpus,
        sources,
        config,
    )
    second = select_quality_appraisal_packets(
        corpus,
        sources,
        config,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) <= 4
    assert "c3" in first["chunk_id"].tolist()


def test_prompt_contains_domain_criteria(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    packet = pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "Scientific source",
                "chunk_id": "c1",
                "section_heading": "Methods",
                "start_page": 2,
                "end_page": 2,
                "character_count": 50,
                "keyword_score": 2,
                "selection_reason": "quality_keyword_priority",
                "packet_order": 1,
                "text": "Methods are described.",
            }
        ]
    )
    prompt = json.loads(
        build_quality_appraisal_prompt(
            sheet.iloc[0].to_dict(),
            packet,
            config,
        )
    )
    assert prompt["appraisal_domain"] == "scientific"
    assert "design_appropriateness" in prompt["criteria"]
    assert prompt["quality_packet"][0]["pages"] == "2"


def test_read_response_and_calculate_scores(
    tmp_path: Path,
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    criteria = criteria_for_domain(config, "scientific")
    payload = {
        "source_id": "src_001",
        "appraisal_status": "completed",
        "appraisal_domain": "scientific",
        "criteria": {
            criterion: {
                "score": 2,
                "note": f"Adequate evidence for {criterion}.",
                "locator": "p. 2",
            }
            for criterion in criteria
        },
        "appraisal_summary": (
            "The source is transparent and methodologically appropriate."
        ),
        "critical_limitations": ["Limited external validation"],
    }
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    frame = read_quality_appraisal_responses_jsonl(
        path,
        sheet,
        config,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["total_score"] == "8"
    assert frame.iloc[0]["normalized_score"] == "1.000000"
    assert frame.iloc[0]["overall_rating"] == "high"


def test_merge_protects_human_validated(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    sheet = completed_row(sheet, config)
    sheet.loc[0, "human_validation_status"] = "accepted"
    incoming = sheet.copy()
    incoming.loc[0, "overall_rating"] = "low"
    merged = merge_quality_appraisals(
        sheet,
        incoming,
        config,
        overwrite_human_validated=False,
    )
    assert merged.loc[0, "overall_rating"] == "high"


def test_valid_completed_sheet_has_no_issues(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    sheet = completed_row(sheet, config)
    issues = validate_quality_appraisal_sheet(sheet, config)
    assert issues.empty


def test_validation_detects_locator_and_total_errors(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    sheet = completed_row(sheet, config)
    criterion = criteria_for_domain(config, "scientific")[0]
    sheet.loc[0, locator_column(criterion)] = "section methods"
    sheet.loc[0, "total_score"] = "1"
    issues = validate_quality_appraisal_sheet(sheet, config)
    assert "explicit_page_locator_required" in issues["issue"].tolist()
    assert any(
        issue.startswith("recorded_1.0_differs_from_calculated_")
        for issue in issues["issue"]
    )


def test_rating_thresholds(config: dict) -> None:
    assert expected_rating(0.49, config) == "low"
    assert expected_rating(0.50, config) == "moderate"
    assert expected_rating(0.79, config) == "moderate"
    assert expected_rating(0.80, config) == "high"
    assert expected_rating(None, config) == "not_applicable"


def test_split_and_finding_link(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    sheet = completed_row(sheet, config)
    sheet.loc[0, "human_validation_status"] = "accepted"
    groups = split_quality_appraisal_outputs(sheet)
    assert len(groups["validated"]) == 1
    assert len(groups["pending_validation"]) == 0

    linked = build_finding_quality_link(
        findings.loc[findings["source_id"].eq("src_001")],
        groups["validated"],
    )
    assert len(linked) == 2
    assert linked["quality_overall_rating"].eq("high").all()


def test_summary_counts(
    config: dict,
    sources: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    sheet = initialise_quality_appraisal_sheet(
        sources.iloc[[0]],
        findings,
        config,
    )
    sheet = completed_row(sheet, config)
    summary = appraisal_summary(sheet).set_index("metric")["value"]
    assert summary["sources_total"] == 1
    assert summary["appraisals_completed"] == 1
    assert summary["quality_high"] == 1
