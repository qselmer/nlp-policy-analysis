from pathlib import Path

import pandas as pd
import pytest

from evidence_review.screening import (
    TitleAbstractScreening,
    apply_keyword_triage,
    build_screening_corpus,
    completed_binary_decisions,
    initialise_screening_sheet,
    load_screening_config,
    select_pilot_sample,
    validate_screening_sheet,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "screening.yml"


def _retained_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": "Climate variability and Peruvian anchoveta fisheries",
                "abstract": (
                    "We estimate how ENSO changes anchoveta distribution and discuss "
                    "adaptive fishery management responses."
                ),
                "authors_or_organisation": "Example Author",
                "publisher": "Example Journal",
                "year": 2024,
                "source_type": "scientific_article",
                "source_status": "peer_reviewed",
                "source_language": "en",
                "doi": "https://doi.org/10.1000/example",
                "primary_url": "https://example.org/1",
                "import_file": "pilot.csv",
                "import_platform": "openalex",
            },
            {
                "title": "Freshwater aquaculture production",
                "abstract": (
                    "This study evaluates feed conversion in farmed tilapia in lakes "
                    "without a marine fisheries or climate component."
                ),
                "year": 2022,
                "source_type": "scientific_article",
                "doi": "",
            },
        ]
    )


def test_screening_config_criterion_values_are_strings():
    config = load_screening_config(CONFIG_PATH)

    assert config["criterion_values"] == ["yes", "no", "unclear"]
    assert all(isinstance(value, str) for value in config["criterion_values"])


def test_include_requires_three_yes():
    record = TitleAbstractScreening(
        source_id="src_001",
        decision="include",
        source_type="scientific_article",
        fisheries_or_marine_relevant="yes",
        climate_environment_or_adaptation_element="yes",
        contributes_codable_evidence="yes",
    )
    assert record.decision.value == "include"

    with pytest.raises(ValueError):
        TitleAbstractScreening(
            source_id="src_002",
            decision="include",
            source_type="scientific_article",
            fisheries_or_marine_relevant="yes",
            climate_environment_or_adaptation_element="unclear",
            contributes_codable_evidence="yes",
        )


def test_exclude_requires_failed_criterion_and_reason():
    with pytest.raises(ValueError):
        TitleAbstractScreening(
            source_id="src_003",
            decision="exclude",
            source_type="scientific_article",
            fisheries_or_marine_relevant="no",
            climate_environment_or_adaptation_element="yes",
            contributes_codable_evidence="yes",
        )


def test_uncertain_requires_unclear_criterion():
    record = TitleAbstractScreening(
        source_id="src_004",
        decision="uncertain",
        source_type="scientific_article",
        fisheries_or_marine_relevant="yes",
        climate_environment_or_adaptation_element="unclear",
        contributes_codable_evidence="yes",
    )
    assert record.decision.value == "uncertain"


def test_corpus_and_triage_are_traceable():
    config = load_screening_config(CONFIG_PATH)
    corpus = build_screening_corpus(_retained_frame())
    triaged = apply_keyword_triage(corpus, config)

    assert len(triaged) == 2
    assert triaged["source_id"].is_unique
    assert triaged.loc[0, "triage_priority"] == "high"
    assert "anchoveta_direct" in triaged.loc[0, "suggested_priority_groups"]
    assert "aquaculture" in triaged.loc[1, "exclusion_keyword_hits"]


def test_initialise_preserves_existing_review_fields():
    corpus = build_screening_corpus(_retained_frame())
    first = initialise_screening_sheet(corpus)
    first.loc[0, "decision"] = "include"
    first.loc[0, "fisheries_or_marine_relevant"] = "yes"
    first.loc[0, "climate_environment_or_adaptation_element"] = "yes"
    first.loc[0, "contributes_codable_evidence"] = "yes"

    refreshed = initialise_screening_sheet(corpus, first)
    assert refreshed.loc[0, "decision"] == "include"


def test_validation_flags_inconsistent_decisions():
    config = load_screening_config(CONFIG_PATH)
    corpus = apply_keyword_triage(build_screening_corpus(_retained_frame()), config)
    sheet = initialise_screening_sheet(corpus)
    sheet.loc[0, "decision"] = "include"
    sheet.loc[0, "fisheries_or_marine_relevant"] = "yes"
    sheet.loc[0, "climate_environment_or_adaptation_element"] = "unclear"
    sheet.loc[0, "contributes_codable_evidence"] = "yes"

    issues = validate_screening_sheet(sheet, config)
    assert not issues.empty
    assert "screening_logic" in set(issues["field"])


def test_pilot_sample_and_binary_export():
    config = load_screening_config(CONFIG_PATH)
    retained = pd.concat([_retained_frame()] * 20, ignore_index=True)
    retained["title"] = retained["title"] + retained.index.astype(str)
    retained["doi"] = ""
    corpus = apply_keyword_triage(build_screening_corpus(retained), config)
    sheet = initialise_screening_sheet(corpus)

    sample_a = select_pilot_sample(sheet, n=10, random_state=7)
    sample_b = select_pilot_sample(sheet, n=10, random_state=7)
    assert list(sample_a["source_id"]) == list(sample_b["source_id"])

    sheet.loc[0, "decision"] = "include"
    sheet.loc[0, "fisheries_or_marine_relevant"] = "yes"
    sheet.loc[0, "climate_environment_or_adaptation_element"] = "yes"
    sheet.loc[0, "contributes_codable_evidence"] = "yes"
    binary = completed_binary_decisions(sheet)
    assert len(binary) == 1
    assert bool(binary.loc[0, "include"]) is True
