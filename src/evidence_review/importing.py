"""Utilities for importing, normalising, and deduplicating evidence sources."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping

import pandas as pd
import yaml


OFFICIAL_VERSIONED_TYPES = {
    "international_agreement",
    "regional_measure",
    "national_law",
    "regulation",
    "policy",
    "strategy",
    "adaptation_plan",
    "fishery_management_plan",
    "harvest_strategy",
    "technical_protocol",
    "guideline_or_manual",
}

REGISTRY_COLUMNS = [
    "source_id",
    "title",
    "source_type",
    "source_subtype",
    "authors_or_organisation",
    "publisher",
    "year",
    "version_date",
    "jurisdiction",
    "geographic_scope",
    "fishery_scope",
    "species",
    "source_status",
    "legal_status",
    "doi",
    "primary_url",
    "landing_page_url",
    "access_date",
    "source_language",
    "local_path",
    "checksum_sha256",
    "supersedes_source_id",
    "notes",
]


def load_import_mappings(path: str | Path = "config/import_mappings.yml") -> dict:
    """Load platform-specific column mappings."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _clean_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalise_doi(value: object) -> str:
    """Return a DOI without URL or resolver prefixes."""
    doi = _clean_scalar(value).lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().rstrip(".,;)")


def normalise_title(value: object) -> str:
    """Create a conservative title key for duplicate detection."""
    title = _clean_scalar(value).lower()
    decomposed = unicodedata.normalize("NFKD", title)
    title = "".join(char for char in decomposed if not unicodedata.combining(char))
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def normalise_year(value: object) -> str:
    """Extract a four-digit publication year where possible."""
    text = _clean_scalar(value)
    match = re.search(r"(?:18|19|20|21)\d{2}", text)
    return match.group(0) if match else ""


def stable_source_id(*, title: object, year: object = "", doi: object = "") -> str:
    """Build a deterministic identifier from DOI or title-year metadata."""
    doi_key = normalise_doi(doi)
    if doi_key:
        payload = f"doi:{doi_key}"
    else:
        title_key = normalise_title(title)
        if not title_key:
            raise ValueError("A source_id requires a DOI or non-empty title")
        payload = f"title:{title_key}|year:{normalise_year(year)}"
    digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


def read_tabular_export(path: str | Path) -> pd.DataFrame:
    """Read a CSV or TSV bibliographic export."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported tabular export: {path.name}")


def _first_matching_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {str(column).strip().casefold(): column for column in columns}
    for alias in aliases:
        match = lookup.get(str(alias).strip().casefold())
        if match is not None:
            return match
    return None


def standardise_export(
    frame: pd.DataFrame,
    mapping: Mapping[str, Iterable[str]],
    *,
    defaults: Mapping[str, object] | None = None,
    import_file: str = "",
    import_platform: str = "",
) -> pd.DataFrame:
    """Map a platform export to canonical evidence-source fields."""
    defaults = defaults or {}
    output = pd.DataFrame(index=frame.index)

    for canonical, aliases in mapping.items():
        source_column = _first_matching_column(frame.columns, aliases)
        if source_column is None:
            output[canonical] = defaults.get(canonical, "")
        else:
            output[canonical] = frame[source_column]

    for canonical, value in defaults.items():
        if canonical not in output:
            output[canonical] = value
        else:
            empty = output[canonical].isna() | output[canonical].astype(str).str.strip().eq("")
            output.loc[empty, canonical] = value

    output["import_file"] = import_file
    output["import_platform"] = import_platform
    output["original_row"] = range(1, len(output) + 1)
    output["doi_normalised"] = output.get("doi", "").map(normalise_doi)
    output["title_normalised"] = output.get("title", "").map(normalise_title)
    output["year_normalised"] = output.get("year", "").map(normalise_year)
    return output


def _deduplication_key(row: pd.Series) -> tuple[str, str]:
    doi = row.get("doi_normalised", "")
    if doi:
        return f"doi:{doi}", "exact_doi"

    title = row.get("title_normalised", "")
    year = row.get("year_normalised", "")
    source_type = _clean_scalar(row.get("source_type", ""))
    version_date = _clean_scalar(row.get("version_date", ""))

    if title:
        key = f"title_year:{title}|{year}"
        if source_type in OFFICIAL_VERSIONED_TYPES and version_date:
            key += f"|version:{version_date}"
            return key, "official_title_year_version"
        return key, "exact_title_year"

    fallback = f"unresolved:{row.name}"
    return fallback, "insufficient_metadata"


def _metadata_completeness(row: pd.Series) -> int:
    fields = [
        "title",
        "authors_or_organisation",
        "publisher",
        "year",
        "doi",
        "primary_url",
        "abstract",
        "source_type",
        "source_status",
    ]
    return sum(bool(_clean_scalar(row.get(field, ""))) for field in fields)


def deduplicate_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate sources by DOI, then normalised title and year.

    Returns a retained-source table and an audit table containing every member of
    duplicate groups. Distinct dated versions of official instruments are preserved.
    """
    data = frame.copy().reset_index(drop=True)
    for field in ("doi", "title", "year"):
        if field not in data:
            data[field] = ""

    if "doi_normalised" not in data:
        data["doi_normalised"] = data["doi"].map(normalise_doi)
    if "title_normalised" not in data:
        data["title_normalised"] = data["title"].map(normalise_title)
    if "year_normalised" not in data:
        data["year_normalised"] = data["year"].map(normalise_year)

    keys = data.apply(_deduplication_key, axis=1, result_type="expand")
    data["duplicate_key"] = keys[0]
    data["duplicate_rule"] = keys[1]
    data["metadata_completeness"] = data.apply(_metadata_completeness, axis=1)
    data["input_order"] = range(len(data))

    data = data.sort_values(
        ["duplicate_key", "metadata_completeness", "input_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    data["duplicate_rank"] = data.groupby("duplicate_key").cumcount() + 1
    data["duplicate_group_size"] = data.groupby("duplicate_key")["duplicate_key"].transform("size")
    data["record_status"] = data["duplicate_rank"].map(lambda rank: "retained" if rank == 1 else "removed_duplicate")

    duplicate_audit = data.loc[data["duplicate_group_size"] > 1].copy()
    retained = data.loc[data["duplicate_rank"] == 1].copy()
    retained = retained.sort_values("input_order", kind="stable").reset_index(drop=True)
    duplicate_audit = duplicate_audit.sort_values(
        ["duplicate_key", "duplicate_rank"], kind="stable"
    ).reset_index(drop=True)

    return retained, duplicate_audit


def to_source_registry(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert standardised retained records to the master registry schema."""
    registry = pd.DataFrame(index=frame.index)
    for column in REGISTRY_COLUMNS:
        registry[column] = frame[column] if column in frame else ""

    registry["source_id"] = [
        stable_source_id(title=title, year=year, doi=doi)
        for title, year, doi in zip(
            registry["title"], registry["year"], registry["doi"], strict=True
        )
    ]
    registry["doi"] = registry["doi"].map(normalise_doi)
    return registry[REGISTRY_COLUMNS].reset_index(drop=True)
