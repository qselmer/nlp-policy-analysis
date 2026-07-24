import pandas as pd

from evidence_review.importing import (
    deduplicate_sources,
    normalise_doi,
    normalise_title,
    stable_source_id,
    standardise_export,
    to_source_registry,
)


def test_normalise_doi_removes_resolver_prefix():
    assert normalise_doi("https://doi.org/10.1000/ABC.123") == "10.1000/abc.123"
    assert normalise_doi("doi: 10.1000/ABC.123") == "10.1000/abc.123"


def test_title_normalisation_handles_accents_and_punctuation():
    assert normalise_title("Cambio climático: pesquerías") == "cambio climatico pesquerias"


def test_stable_source_id_is_deterministic():
    first = stable_source_id(title="A title", year=2020, doi="10.1000/example")
    second = stable_source_id(title="Different title", year=2024, doi="https://doi.org/10.1000/EXAMPLE")
    assert first == second


def test_standardise_export_matches_case_insensitive_aliases():
    raw = pd.DataFrame(
        {
            "Title": ["Climate and fisheries"],
            "DOI": ["10.1234/example"],
            "Year": [2022],
        }
    )
    mapping = {
        "title": ["title"],
        "doi": ["doi"],
        "year": ["year"],
    }
    standard = standardise_export(raw, mapping, import_platform="generic")
    assert standard.loc[0, "title"] == "Climate and fisheries"
    assert standard.loc[0, "doi_normalised"] == "10.1234/example"
    assert standard.loc[0, "year_normalised"] == "2022"


def test_deduplication_prefers_more_complete_doi_record():
    frame = pd.DataFrame(
        [
            {
                "title": "Climate effects on anchovy",
                "year": 2020,
                "doi": "10.1000/anchovy",
                "abstract": "",
            },
            {
                "title": "Climate effects on anchovy",
                "year": 2020,
                "doi": "https://doi.org/10.1000/ANCHOVY",
                "abstract": "Detailed abstract",
                "publisher": "Journal",
            },
        ]
    )
    retained, audit = deduplicate_sources(frame)
    assert len(retained) == 1
    assert retained.loc[0, "abstract"] == "Detailed abstract"
    assert len(audit) == 2
    assert set(audit["record_status"]) == {"retained", "removed_duplicate"}


def test_official_versions_with_different_dates_are_preserved():
    frame = pd.DataFrame(
        [
            {
                "title": "National fisheries adaptation plan",
                "year": 2021,
                "version_date": "2021-01-01",
                "source_type": "adaptation_plan",
            },
            {
                "title": "National fisheries adaptation plan",
                "year": 2021,
                "version_date": "2021-09-01",
                "source_type": "adaptation_plan",
            },
        ]
    )
    retained, audit = deduplicate_sources(frame)
    assert len(retained) == 2
    assert audit.empty


def test_registry_generation_creates_unique_stable_ids():
    frame = pd.DataFrame(
        [
            {"title": "Source A", "year": 2020, "doi": "10.1/a"},
            {"title": "Source B", "year": 2021, "doi": "10.1/b"},
        ]
    )
    registry = to_source_registry(frame)
    assert registry["source_id"].nunique() == 2
    assert registry.columns[0] == "source_id"
