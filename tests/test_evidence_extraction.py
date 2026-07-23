import json

import pandas as pd
import pytest

from evidence_review.evidence_extraction import (
    FINDING_COLUMNS,
    annotate_near_duplicate_findings,
    build_source_finding_summary,
    build_structured_finding_prompt,
    enrich_extraction_corpus,
    extraction_summary,
    initialise_extraction_queue,
    merge_extraction_queue,
    merge_findings_working,
    parse_page_locators,
    read_structured_finding_responses_jsonl,
    split_structured_finding_outputs,
    synchronise_queue_with_findings,
    validate_structured_finding_outputs,
)


@pytest.fixture
def config():
    return {
        "extraction_statuses": [
            "pending",
            "findings_extracted",
            "no_codable_finding",
            "needs_more_context",
            "extraction_error",
        ],
        "no_finding_reasons": [
            "bibliographic_or_front_matter",
            "methods_without_codable_contribution",
            "repeated_context_only",
            "references_only",
            "insufficient_context",
            "no_climate_fisheries_finding",
            "other",
        ],
        "human_validation_status": [
            "not_reviewed",
            "accepted",
            "corrected",
            "rejected",
        ],
        "prompting": {
            "max_findings_per_chunk": 6,
            "max_excerpt_characters": 1200,
        },
        "validation": {
            "require_supporting_excerpt": True,
            "excerpt_must_match_chunk": True,
            "require_page_locator": True,
            "page_locator_within_chunk": True,
            "minimum_summary_characters": 20,
            "maximum_excerpt_characters": 1200,
            "coder_confidence_minimum": 0.0,
            "coder_confidence_maximum": 1.0,
        },
        "near_duplicate_detection": {
            "enabled": True,
            "token_jaccard_threshold": 0.82,
            "minimum_tokens": 8,
            "require_page_overlap_or_adjacency": True,
        },
        "new_controlled_vocabularies": {
            "evidence_basis": [
                "observed",
                "experimental",
                "modelled_hindcast",
                "modelled_projection",
                "scenario",
                "legal_text",
                "policy_statement",
                "project_monitoring",
                "recommendation",
                "conceptual",
                "mixed",
                "unclear",
            ],
            "implementation_stage": [
                "not_applicable",
                "proposed",
                "planned",
                "authorized",
                "piloted",
                "implemented",
                "evaluated_activity",
                "evaluated_output",
                "evaluated_outcome",
                "evaluated_impact",
                "unclear",
            ],
            "causal_interpretation": [
                "descriptive",
                "associational",
                "causal",
                "mechanistic_model",
                "normative",
                "unclear",
            ],
            "result_direction": [
                "positive",
                "negative",
                "mixed",
                "null",
                "non_directional",
                "unclear",
            ],
            "deduplication_status": [
                "unique",
                "primary",
                "possible_duplicate",
                "confirmed_duplicate",
            ],
        },
        "taxonomy": {
            "source_type": [
                "scientific_article",
                "systematic_review",
                "other",
            ],
            "finding_type": [
                "empirical_result",
                "model_projection",
                "scenario_result",
                "project_activity",
                "project_output",
                "project_outcome",
                "policy_commitment",
                "recommendation",
                "method_or_indicator",
            ],
            "evidence_stream": [
                "climate_hazard",
                "ecological_response",
                "fishery_response",
                "management_option",
                "methods_and_data",
            ],
            "fishery_scope": [
                "marine_ecosystem",
                "small_pelagics",
                "anchovy",
            ],
            "climate_driver": [
                "ocean_warming",
                "marine_heatwave",
                "enso",
            ],
            "biological_response": [
                "biomass",
                "spatial_distribution",
                "recruitment",
            ],
            "fishery_response": [
                "catch",
                "fishing_ground_shift",
            ],
            "management_measure": [
                "catch_limit",
                "early_warning_system",
            ],
            "method_or_design": [
                "observational_time_series",
                "ecosystem_model",
            ],
            "relevance_level": [
                "direct",
                "indirect_high",
                "indirect_moderate",
                "indirect_low",
                "none",
                "unclear",
            ],
        },
    }


@pytest.fixture
def corpus():
    return pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "title": "Climate effects on anchovy",
                "source_type": "scientific_article",
                "doi": "10.1/test",
                "primary_url": "https://example.org",
                "chunk_id": "chk_001",
                "section_id": "sec_001",
                "section_heading": "Results",
                "start_page": 3,
                "end_page": 4,
                "character_count": 150,
                "source_checksum_sha256": "abc",
                "full_text_priority_groups": "anchoveta_direct",
                "full_text_human_validation_status": "accepted",
                "text": (
                    "Ocean warming was associated with a southward shift in "
                    "anchovy distribution. The estimated shift was 120 km "
                    "during the study period."
                ),
            },
            {
                "source_id": "src_001",
                "title": "Climate effects on anchovy",
                "source_type": "scientific_article",
                "doi": "10.1/test",
                "primary_url": "https://example.org",
                "chunk_id": "chk_002",
                "section_id": "sec_001",
                "section_heading": "Discussion",
                "start_page": 4,
                "end_page": 5,
                "character_count": 140,
                "source_checksum_sha256": "abc",
                "full_text_priority_groups": "anchoveta_direct",
                "full_text_human_validation_status": "accepted",
                "text": (
                    "Ocean warming was associated with a southward shift in "
                    "anchovy distribution. This pattern may alter fishing "
                    "grounds."
                ),
            },
        ]
    )


def finding_row(**updates):
    row = {column: "" for column in FINDING_COLUMNS}
    row.update(
        {
            "finding_id": "fnd_001",
            "source_id": "src_001",
            "title": "Climate effects on anchovy",
            "source_type": "scientific_article",
            "doi": "10.1/test",
            "primary_url": "https://example.org",
            "chunk_id": "chk_001",
            "section_id": "sec_001",
            "section_heading": "Results",
            "chunk_start_page": "3",
            "chunk_end_page": "4",
            "source_checksum_sha256": "abc",
            "finding_type": "empirical_result",
            "evidence_streams": (
                "climate_hazard | ecological_response"
            ),
            "unit_locator": "p. 3",
            "supporting_excerpt": (
                "Ocean warming was associated with a southward shift "
                "in anchovy distribution."
            ),
            "evidence_summary": (
                "Ocean warming was associated with a southward shift "
                "in anchovy distribution."
            ),
            "fishery_scope": "anchovy",
            "species": "Engraulis ringens",
            "climate_drivers": "ocean_warming",
            "biological_responses": "spatial_distribution",
            "method_or_design": "observational_time_series",
            "result_direction": "negative",
            "evidence_basis": "observed",
            "implementation_stage": "not_applicable",
            "causal_interpretation": "associational",
            "uncertainty_reported": "true",
            "relevance_to_small_pelagics": "direct",
            "relevance_to_anchoveta": "direct",
            "transferability_rationale": "Direct anchovy evidence.",
            "coder_confidence": "0.9",
            "deduplication_status": "unique",
            "reviewer": (
                "AI-assisted preliminary finding extraction"
            ),
            "human_validation_status": "not_reviewed",
        }
    )
    row.update(updates)
    return row


def test_enrich_corpus_adds_metadata_and_counts_characters(corpus):
    raw = corpus.drop(
        columns=["source_type", "doi", "primary_url"]
    ).copy()
    metadata = pd.DataFrame(
        [
            {
                "source_id": "src_001",
                "source_type": "scientific_article",
                "doi": "10.1/test",
                "primary_url": "https://example.org",
            }
        ]
    )
    enriched = enrich_extraction_corpus(raw, metadata)
    assert enriched.loc[0, "source_type"] == "scientific_article"
    assert enriched.loc[0, "character_count"] == len(
        corpus.loc[0, "text"]
    )


def test_queue_preserves_existing_human_review(corpus):
    existing = pd.DataFrame(
        [
            {
                "chunk_id": "chk_001",
                "extraction_status": "no_codable_finding",
                "no_finding_reason": "other",
                "human_validation_status": "accepted",
            }
        ]
    )
    queue = initialise_extraction_queue(corpus, existing)
    row = queue.loc[queue["chunk_id"].eq("chk_001")].iloc[0]
    assert row["extraction_status"] == "no_codable_finding"
    assert row["human_validation_status"] == "accepted"


def test_prompt_contains_critical_distinctions(corpus, config):
    prompt = json.loads(
        build_structured_finding_prompt(
            corpus.iloc[0].to_dict(),
            config,
        )
    )
    rules = " ".join(prompt["rules"])
    assert "recommendation is not an observed result" in rules
    assert "not evidence of implementation" in rules
    assert prompt["chunk"]["chunk_id"] == "chk_001"


def test_response_parser_flattens_findings(
    tmp_path,
    corpus,
    config,
):
    path = tmp_path / "responses.jsonl"
    response = {
        "source_id": "src_001",
        "chunk_id": "chk_001",
        "extraction_status": "findings_extracted",
        "findings": [
            {
                "finding_type": "empirical_result",
                "evidence_streams": [
                    "climate_hazard",
                    "ecological_response",
                ],
                "unit_locator": "p. 3",
                "supporting_excerpt": (
                    "Ocean warming was associated with a southward "
                    "shift in anchovy distribution."
                ),
                "evidence_summary": (
                    "Ocean warming was associated with a southward "
                    "distribution shift."
                ),
                "fishery_scope": ["anchovy"],
                "climate_drivers": ["ocean_warming"],
                "biological_responses": [
                    "spatial_distribution"
                ],
                "method_or_design": [
                    "observational_time_series"
                ],
                "result_direction": "negative",
                "evidence_basis": "observed",
                "implementation_stage": "not_applicable",
                "causal_interpretation": "associational",
                "relevance_to_small_pelagics": "direct",
                "relevance_to_anchoveta": "direct",
                "coder_confidence": 0.8,
            }
        ],
    }
    path.write_text(
        json.dumps(response) + "\n",
        encoding="utf-8",
    )
    updates, findings = read_structured_finding_responses_jsonl(
        path,
        corpus,
        config,
    )
    assert updates.loc[0, "candidate_findings_count"] == "1"
    assert findings.loc[0, "finding_id"].startswith("fnd_")
    assert findings.loc[0, "evidence_streams"] == (
        "climate_hazard | ecological_response"
    )


def test_response_parser_rejects_unknown_chunk(
    tmp_path,
    corpus,
    config,
):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_id": "src_001",
                "chunk_id": "missing",
                "extraction_status": "no_codable_finding",
                "findings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown chunk_id"):
        read_structured_finding_responses_jsonl(
            path,
            corpus,
            config,
        )


def test_queue_merge_protects_validated_rows(corpus):
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "human_validation_status"] = "accepted"
    queue.loc[0, "extraction_status"] = "no_codable_finding"
    updates = pd.DataFrame(
        [
            {
                "chunk_id": "chk_001",
                "extraction_status": "findings_extracted",
                "human_validation_status": "not_reviewed",
            }
        ]
    )
    merged = merge_extraction_queue(queue, updates)
    assert merged.loc[0, "extraction_status"] == (
        "no_codable_finding"
    )


def test_findings_merge_protects_corrected_row():
    existing = pd.DataFrame(
        [
            finding_row(
                evidence_summary=(
                    "Human corrected summary with sufficient detail."
                ),
                human_validation_status="corrected",
            )
        ]
    )
    incoming = pd.DataFrame(
        [
            finding_row(
                evidence_summary=(
                    "Model replacement summary with sufficient detail."
                ),
                human_validation_status="not_reviewed",
            )
        ]
    )
    merged = merge_findings_working(existing, incoming)
    assert merged.loc[0, "evidence_summary"].startswith(
        "Human corrected"
    )


def test_valid_outputs_have_no_issues(corpus, config):
    findings = pd.DataFrame([finding_row()])
    queue = initialise_extraction_queue(corpus)
    queue.loc[
        queue["chunk_id"].eq("chk_001"),
        "extraction_status",
    ] = "findings_extracted"
    queue = synchronise_queue_with_findings(queue, findings)
    issues = validate_structured_finding_outputs(
        queue,
        findings,
        corpus,
        config,
    )
    assert issues.empty


def test_validator_flags_excerpt_not_in_chunk(corpus, config):
    findings = pd.DataFrame(
        [
            finding_row(
                supporting_excerpt=(
                    "This sentence is absent from the source chunk."
                )
            )
        ]
    )
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "extraction_status"] = "findings_extracted"
    queue = synchronise_queue_with_findings(queue, findings)
    issues = validate_structured_finding_outputs(
        queue,
        findings,
        corpus,
        config,
    )
    assert "excerpt_not_found_in_chunk" in set(issues["issue"])


def test_validator_flags_page_outside_chunk(corpus, config):
    findings = pd.DataFrame([finding_row(unit_locator="p. 8")])
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "extraction_status"] = "findings_extracted"
    queue = synchronise_queue_with_findings(queue, findings)
    issues = validate_structured_finding_outputs(
        queue,
        findings,
        corpus,
        config,
    )
    assert any(
        value.startswith("page_locator_outside_chunk")
        for value in issues["issue"]
    )


def test_validator_flags_unknown_vocabulary(corpus, config):
    findings = pd.DataFrame(
        [finding_row(climate_drivers="invented_driver")]
    )
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "extraction_status"] = "findings_extracted"
    queue = synchronise_queue_with_findings(queue, findings)
    issues = validate_structured_finding_outputs(
        queue,
        findings,
        corpus,
        config,
    )
    assert "unknown_values:invented_driver" in set(
        issues["issue"]
    )


def test_parse_page_locators_requires_explicit_page_prefix():
    assert parse_page_locators("pp. 3-5 | p. 8") == [
        3,
        4,
        5,
        8,
    ]
    assert parse_page_locators("section 3-5") == []


def test_near_duplicate_annotation(config):
    first = finding_row()
    second = finding_row(
        finding_id="fnd_002",
        chunk_id="chk_002",
        chunk_start_page="4",
        chunk_end_page="5",
        unit_locator="p. 4",
        evidence_summary=(
            "Ocean warming was associated with a southward shift "
            "in anchovy distribution."
        ),
        supporting_excerpt=(
            "Ocean warming was associated with a southward shift "
            "in anchovy distribution."
        ),
    )
    annotated = annotate_near_duplicate_findings(
        pd.DataFrame([first, second]),
        config,
    )
    assert set(annotated["deduplication_status"]) == {
        "primary",
        "possible_duplicate",
    }
    assert annotated["duplicate_group_id"].nunique() == 1


def test_split_outputs_separates_validation_and_queue_states(
    corpus,
):
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "extraction_status"] = "no_codable_finding"
    queue.loc[0, "no_finding_reason"] = "other"
    findings = pd.DataFrame(
        [
            finding_row(human_validation_status="accepted"),
            finding_row(
                finding_id="fnd_002",
                human_validation_status="not_reviewed",
            ),
            finding_row(
                finding_id="fnd_003",
                human_validation_status="rejected",
            ),
        ]
    )
    groups = split_structured_finding_outputs(queue, findings)
    assert len(groups["validated"]) == 1
    assert len(groups["pending_validation"]) == 1
    assert len(groups["rejected"]) == 1
    assert len(groups["no_finding_chunks"]) == 1


def test_summary_and_source_summary(corpus):
    queue = initialise_extraction_queue(corpus)
    queue.loc[0, "extraction_status"] = "findings_extracted"
    findings = pd.DataFrame(
        [finding_row(human_validation_status="accepted")]
    )
    summary = extraction_summary(queue, findings).set_index(
        "metric"
    )["value"]
    assert summary["findings_total"] == 1
    assert summary["findings_accepted"] == 1
    source = build_source_finding_summary(findings)
    assert source.loc[0, "maximum_anchoveta_relevance"] == (
        "direct"
    )
