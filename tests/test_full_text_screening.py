from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evidence_review.full_text_screening import (
    FullTextScreeningRecord,
    build_evidence_extraction_corpus,
    build_full_text_screening_prompt,
    completed_full_text_decisions,
    initialise_full_text_screening_sheet,
    load_full_text_screening_config,
    merge_full_text_screening_decisions,
    parse_page_locators,
    read_full_text_screening_decisions_jsonl,
    screening_summary,
    select_full_text_screening_packets,
    split_full_text_screening_outputs,
    validate_full_text_screening_sheet,
)


@pytest.fixture()
def config(tmp_path: Path) -> dict:
    path = tmp_path / "config.yml"
    path.write_text(
        """
paths: {}
operational_rule: {}
eligibility_criteria:
  fisheries_or_marine_relevant: {question: "fishery?"}
  climate_environment_or_adaptation_element: {question: "climate?"}
  contributes_codable_evidence: {question: "evidence?"}
  sufficient_full_text_for_verification: {question: "enough text?"}
criterion_values: ["yes", "no", "unclear"]
decisions: [include, exclude, uncertain]
eligibility_exclusion_reasons:
  - not_marine_or_fisheries
  - no_climate_environment_adaptation_element
  - no_codable_evidence
administrative_exclusion_reasons:
  - duplicate_record
  - superseded_version
  - wrong_document
  - publication_type_not_eligible
  - outside_date_range
priority_groups:
  - marine_fisheries
  - climate_hazard
human_validation_status:
  - not_reviewed
  - accepted
  - corrected
  - rejected
packet_selection:
  max_chunks_per_source: 4
  max_characters_per_source: 500
  edge_chunks: 1
  spread_chunks: 2
  keyword_terms: [climate, fishery, management]
""".strip(),
        encoding="utf-8",
    )
    return load_full_text_screening_config(path)


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "Climate and fisheries",
                "screening_decision": "include",
                "full_text_screening_required": "false",
                "local_path": "data/raw/full_texts/src_001.pdf",
                "source_checksum_sha256": "abc",
                "page_count": "10",
                "total_characters": "10000",
                "ocr_recommended": "false",
                "parsing_status": "parsed",
                "full_text": "full text",
            },
            {
                "source_id": "src_002",
                "title": "Uncertain source",
                "screening_decision": "uncertain",
                "full_text_screening_required": "true",
                "local_path": "data/raw/full_texts/src_002.pdf",
                "source_checksum_sha256": "def",
                "page_count": "8",
                "total_characters": "8000",
                "ocr_recommended": "false",
                "parsing_status": "parsed",
                "full_text": "full text",
            },
        ]
    )


def _valid_include_row() -> dict[str, str]:
    return {
        "source_id": "src_001",
        "title": "Climate and fisheries",
        "source_type": "article",
        "doi": "",
        "primary_url": "",
        "title_abstract_decision": "include",
        "full_text_screening_required": "false",
        "local_path": "x.pdf",
        "source_checksum_sha256": "abc",
        "page_count": "10",
        "total_characters": "1000",
        "ocr_recommended": "false",
        "parsing_status": "parsed",
        "screening_stage": "full_text",
        "decision": "include",
        "fisheries_or_marine_relevant": "yes",
        "climate_environment_or_adaptation_element": "yes",
        "contributes_codable_evidence": "yes",
        "sufficient_full_text_for_verification": "yes",
        "exclusion_reason": "",
        "priority_groups": "marine_fisheries | climate_hazard",
        "decision_basis": "The document reports a locatable climate-fishery result.",
        "supporting_page_locators": "pp. 3-4",
        "reviewer": "Reviewer",
        "review_date": "2026-07-22",
        "reviewer_notes": "",
        "human_validation_status": "accepted",
    }


def test_initialise_sheet_preserves_existing_decisions() -> None:
    existing = pd.DataFrame([_valid_include_row()])
    sheet = initialise_full_text_screening_sheet(_corpus(), existing=existing)

    first = sheet.loc[sheet["source_id"].eq("src_001")].iloc[0]
    second = sheet.loc[sheet["source_id"].eq("src_002")].iloc[0]

    assert first["decision"] == "include"
    assert first["human_validation_status"] == "accepted"
    assert first["title_abstract_decision"] == "include"
    assert second["decision"] == ""
    assert "full_text" not in sheet.columns


def test_full_text_record_logic() -> None:
    FullTextScreeningRecord(
        source_id="src_001",
        decision="include",
        fisheries_or_marine_relevant="yes",
        climate_environment_or_adaptation_element="yes",
        contributes_codable_evidence="yes",
        sufficient_full_text_for_verification="yes",
        decision_basis="This is a sufficiently detailed eligibility basis.",
        supporting_page_locators="p. 3",
    )

    with pytest.raises(ValueError, match="include requires yes"):
        FullTextScreeningRecord(
            source_id="src_001",
            decision="include",
            fisheries_or_marine_relevant="yes",
            climate_environment_or_adaptation_element="unclear",
            contributes_codable_evidence="yes",
            sufficient_full_text_for_verification="yes",
            decision_basis="This is a sufficiently detailed eligibility basis.",
            supporting_page_locators="p. 3",
        )

    FullTextScreeningRecord(
        source_id="src_001",
        decision="exclude",
        fisheries_or_marine_relevant="yes",
        climate_environment_or_adaptation_element="yes",
        contributes_codable_evidence="yes",
        sufficient_full_text_for_verification="yes",
        exclusion_reason="duplicate_record",
        decision_basis="This is a duplicate of the retained source version.",
        supporting_page_locators="p. 1",
    )


def test_parse_page_locators() -> None:
    assert parse_page_locators("p. 3 | pp. 5-7; page 10") == [3, 5, 6, 7, 10]
    assert parse_page_locators("") == []


def test_validation_accepts_valid_row_and_detects_out_of_range(config: dict) -> None:
    row = _valid_include_row()
    assert validate_full_text_screening_sheet(pd.DataFrame([row]), config).empty

    row["supporting_page_locators"] = "p. 11"
    issues = validate_full_text_screening_sheet(pd.DataFrame([row]), config)
    assert "page_locator_out_of_range:11" in set(issues["issue"])


def test_validation_does_not_treat_blank_decision_as_error(config: dict) -> None:
    sheet = initialise_full_text_screening_sheet(_corpus())
    issues = validate_full_text_screening_sheet(sheet, config)
    assert issues.empty


def test_packet_selection_is_deterministic_and_traceable(config: dict) -> None:
    chunks = pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "A",
                "chunk_id": f"chk_{index}",
                "section_heading": "Results",
                "section_index": 1,
                "chunk_index": index,
                "start_page": index,
                "end_page": index,
                "character_count": 120,
                "text": (
                    "climate fishery management result " + ("x" * 80)
                    if index == 4
                    else f"ordinary content {index} " + ("x" * 80)
                ),
            }
            for index in range(1, 7)
        ]
    )

    first = select_full_text_screening_packets(chunks, config)
    second = select_full_text_screening_packets(chunks, config)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) <= 4
    assert first["start_page"].min() == 1
    assert 4 in set(first["start_page"])
    assert first["packet_order"].tolist() == list(range(1, len(first) + 1))


def test_prompt_is_valid_json(config: dict) -> None:
    sheet = initialise_full_text_screening_sheet(_corpus())
    packet = pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "A",
                "chunk_id": "chk_1",
                "section_heading": "Results",
                "start_page": 3,
                "end_page": 4,
                "character_count": 100,
                "keyword_score": 2,
                "selection_reason": "keyword_priority",
                "packet_order": 1,
                "text": "Climate-related fisheries result.",
            }
        ]
    )
    prompt = build_full_text_screening_prompt(
        sheet.iloc[0].to_dict(),
        packet,
        config,
    )
    payload = json.loads(prompt)
    assert payload["source"]["source_id"] == "src_001"
    assert payload["screening_packet"][0]["pages"] == "3-4"
    assert payload["allowed_decisions"] == ["include", "exclude", "uncertain"]


def test_read_and_merge_jsonl_protects_human_validated(
    config: dict,
    tmp_path: Path,
) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_id": "src_001",
                "decision": "exclude",
                "fisheries_or_marine_relevant": "no",
                "climate_environment_or_adaptation_element": "yes",
                "contributes_codable_evidence": "yes",
                "sufficient_full_text_for_verification": "yes",
                "exclusion_reason": "not_marine_or_fisheries",
                "priority_groups": ["climate_hazard"],
                "decision_basis": "The document does not concern marine fisheries.",
                "supporting_page_locators": "p. 2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    incoming = read_full_text_screening_decisions_jsonl(path)
    current = pd.DataFrame([_valid_include_row()])
    merged = merge_full_text_screening_decisions(current, incoming)
    assert merged.iloc[0]["decision"] == "include"

    overwritten = merge_full_text_screening_decisions(
        current,
        incoming,
        overwrite_human_validated=True,
    )
    assert overwritten.iloc[0]["decision"] == "exclude"
    assert overwritten.iloc[0]["priority_groups"] == "climate_hazard"


def test_completed_outputs_and_extraction_corpus() -> None:
    include = _valid_include_row()
    exclude = {**_valid_include_row()}
    exclude.update(
        {
            "source_id": "src_002",
            "title": "Excluded",
            "decision": "exclude",
            "fisheries_or_marine_relevant": "no",
            "exclusion_reason": "not_marine_or_fisheries",
            "priority_groups": "",
            "decision_basis": "The document is outside the marine fisheries scope.",
            "supporting_page_locators": "p. 1",
        }
    )
    sheet = pd.DataFrame([include, exclude])

    completed = completed_full_text_decisions(sheet)
    groups = split_full_text_screening_outputs(sheet)

    assert len(completed) == 2
    assert completed.set_index("source_id").loc["src_001", "include"]
    assert len(groups["included"]) == 1
    assert len(groups["excluded"]) == 1

    chunks = pd.DataFrame(
        [
            {"source_id": "src_001", "chunk_id": "a", "text": "included"},
            {"source_id": "src_002", "chunk_id": "b", "text": "excluded"},
        ]
    )
    corpus = build_evidence_extraction_corpus(chunks, sheet)
    assert corpus["source_id"].tolist() == ["src_001"]
    assert "full_text_decision_basis" in corpus.columns


def test_screening_summary_tracks_resolution() -> None:
    include = _valid_include_row()
    uncertain = {**_valid_include_row()}
    uncertain.update(
        {
            "source_id": "src_002",
            "title_abstract_decision": "uncertain",
            "decision": "exclude",
            "fisheries_or_marine_relevant": "no",
            "exclusion_reason": "not_marine_or_fisheries",
            "decision_basis": "The complete document is outside marine fisheries.",
            "supporting_page_locators": "p. 1",
        }
    )
    summary = screening_summary(pd.DataFrame([include, uncertain]))
    values = dict(zip(summary["metric"], summary["value"], strict=True))
    assert values["total_sources"] == 2
    assert values["title_uncertain_resolved"] == 1
    assert values["completed_binary_decisions"] == 2
