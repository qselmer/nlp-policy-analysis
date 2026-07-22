"""Title-and-abstract screening for the climate-fisheries evidence map."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from evidence_review.importing import stable_source_id


class CriterionStatus(str, Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class ScreeningOutcome(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


class TitleAbstractScreening(BaseModel):
    """Validated title-and-abstract screening record."""

    source_id: str = Field(min_length=3)
    decision: ScreeningOutcome
    source_type: str
    fisheries_or_marine_relevant: CriterionStatus
    climate_environment_or_adaptation_element: CriterionStatus
    contributes_codable_evidence: CriterionStatus
    exclusion_reason: str | None = None
    priority_groups: list[str] = Field(default_factory=list)
    screening_stage: str = "title_abstract"
    reviewer: str | None = None
    review_date: str | None = None
    reviewer_notes: str | None = None
    human_validation_status: str = "not_reviewed"

    @model_validator(mode="after")
    def validate_logic(self) -> "TitleAbstractScreening":
        criteria = (
            self.fisheries_or_marine_relevant,
            self.climate_environment_or_adaptation_element,
            self.contributes_codable_evidence,
        )
        if self.decision == ScreeningOutcome.INCLUDE:
            if any(value != CriterionStatus.YES for value in criteria):
                raise ValueError("include requires yes for all three criteria")
            if self.exclusion_reason:
                raise ValueError("included sources cannot have an exclusion_reason")
        elif self.decision == ScreeningOutcome.EXCLUDE:
            if CriterionStatus.NO not in criteria:
                raise ValueError("exclude requires at least one no criterion")
            if not self.exclusion_reason:
                raise ValueError("excluded sources require an exclusion_reason")
        else:
            if CriterionStatus.UNCLEAR not in criteria:
                raise ValueError("uncertain requires at least one unclear criterion")
            if self.exclusion_reason:
                raise ValueError("uncertain sources cannot have an exclusion_reason")
        return self


CONTEXT_COLUMNS = [
    "source_id", "title", "abstract", "authors_or_organisation", "publisher",
    "year", "source_type", "source_status", "source_language", "doi",
    "primary_url", "import_file", "import_platform", "abstract_available",
    "title_abstract_characters", "triage_priority", "climate_keyword_hits",
    "fisheries_keyword_hits", "priority_taxa_keyword_hits",
    "exclusion_keyword_hits", "suggested_priority_groups",
]

REVIEW_COLUMNS = [
    "decision", "fisheries_or_marine_relevant",
    "climate_environment_or_adaptation_element", "contributes_codable_evidence",
    "exclusion_reason", "priority_groups", "reviewer", "review_date",
    "reviewer_notes", "human_validation_status",
]

BINARY_COLUMNS = [
    "source_id", "include", "source_type", "fisheries_or_marine_relevant",
    "climate_environment_or_adaptation_element", "contributes_codable_evidence",
    "exclusion_reason", "priority_groups", "screening_stage", "reviewer_notes",
    "human_validation_status",
]


def load_screening_config(path: str | Path = "config/screening.yml") -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("screening configuration must be a YAML mapping")
    return config


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", _clean(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _hits(text: str, terms: Iterable[str]) -> list[str]:
    normalised = _normalise(text)
    found: list[str] = []
    for term in terms:
        candidate = _normalise(term).strip('"')
        if not candidate:
            continue
        matched = (
            bool(re.search(rf"\b{re.escape(candidate[:-1])}\w*", normalised))
            if candidate.endswith("*")
            else candidate in normalised
        )
        if matched:
            found.append(str(term))
    return sorted(set(found))


def build_screening_corpus(retained: pd.DataFrame) -> pd.DataFrame:
    """Preserve title/abstract and create stable IDs for screening."""
    if retained.empty:
        return pd.DataFrame(columns=CONTEXT_COLUMNS)
    data = retained.copy().reset_index(drop=True)
    base_fields = (
        "title", "abstract", "authors_or_organisation", "publisher", "year",
        "source_type", "source_status", "source_language", "doi", "primary_url",
        "import_file", "import_platform",
    )
    for field in base_fields:
        if field not in data:
            data[field] = ""
    data["title"] = data["title"].map(_clean)
    data["abstract"] = data["abstract"].map(_clean)
    if "source_id" not in data:
        data["source_id"] = ""
    missing = data["source_id"].map(_clean).eq("")
    data.loc[missing, "source_id"] = [
        stable_source_id(title=title, year=year, doi=doi)
        for title, year, doi in zip(
            data.loc[missing, "title"], data.loc[missing, "year"],
            data.loc[missing, "doi"], strict=True,
        )
    ]
    data["abstract_available"] = data["abstract"].str.len().ge(40)
    data["title_abstract_characters"] = data["title"].str.len() + data["abstract"].str.len()
    for field in CONTEXT_COLUMNS[15:]:
        data[field] = ""
    return data[CONTEXT_COLUMNS].copy()


def apply_keyword_triage(corpus: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add diagnostic hits; never use them as automatic eligibility decisions."""
    if corpus.empty:
        return corpus.copy()
    output = corpus.copy()
    terms = config["diagnostic_terms"]
    rows: list[dict[str, str]] = []
    for row in output.to_dict(orient="records"):
        text = f"{row.get('title', '')}\n{row.get('abstract', '')}"
        climate = _hits(text, terms.get("climate_environment", []))
        fisheries = _hits(text, terms.get("fisheries_marine", []))
        taxa = _hits(text, terms.get("priority_taxa", []))
        exclusions = _hits(text, terms.get("possible_exclusions", []))
        normalised = _normalise(text)
        groups: list[str] = []
        if "engraulis ringens" in normalised or "peruvian anchoveta" in normalised:
            groups.append("anchoveta_direct")
        if taxa:
            groups.append("small_pelagics_or_forage_fish")
        if fisheries:
            groups.append("marine_fisheries")
        if any(term in normalised for term in ("management", "governance", "policy", "legislation", "adaptation")):
            groups.append("management_governance")
        priority = (
            "high" if climate and taxa else
            "medium" if climate and fisheries else
            "review" if climate or fisheries or taxa else
            "low_information"
        )
        rows.append({
            "triage_priority": priority,
            "climate_keyword_hits": " | ".join(climate),
            "fisheries_keyword_hits": " | ".join(fisheries),
            "priority_taxa_keyword_hits": " | ".join(taxa),
            "exclusion_keyword_hits": " | ".join(exclusions),
            "suggested_priority_groups": " | ".join(sorted(set(groups))),
        })
    diagnostics = pd.DataFrame(rows, index=output.index)
    for field in diagnostics:
        output[field] = diagnostics[field]
    return output


def initialise_screening_sheet(
    corpus: pd.DataFrame,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refresh metadata while preserving existing reviewer fields by source_id."""
    sheet = corpus.copy()
    for field in REVIEW_COLUMNS:
        sheet[field] = ""
    sheet["screening_stage"] = "title_abstract"
    if existing is not None and not existing.empty and "source_id" in existing:
        previous = existing.drop_duplicates("source_id", keep="last").set_index("source_id")
        for field in REVIEW_COLUMNS:
            if field in previous:
                values = sheet["source_id"].map(previous[field])
                mask = values.notna()
                sheet.loc[mask, field] = values.loc[mask].astype(str)
    return sheet[CONTEXT_COLUMNS + ["screening_stage"] + REVIEW_COLUMNS]


def select_pilot_sample(
    sheet: pd.DataFrame,
    n: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select a reproducible sample across diagnostic-priority strata."""
    if sheet.empty or len(sheet) <= n:
        return sheet.copy().reset_index(drop=True)
    groups = [
        group for name in ("high", "medium", "review", "low_information")
        if not (group := sheet.loc[sheet["triage_priority"] == name]).empty
    ]
    pieces: list[pd.DataFrame] = []
    remaining = n
    for group in groups:
        take = min(max(1, n // len(groups)), len(group), remaining)
        pieces.append(group.sample(take, random_state=random_state))
        remaining -= take
    selected = pd.concat(pieces, ignore_index=True)
    if remaining:
        pool = sheet.loc[~sheet["source_id"].isin(selected["source_id"])]
        selected = pd.concat([
            selected,
            pool.sample(min(remaining, len(pool)), random_state=random_state + 1),
        ], ignore_index=True)
    return selected.drop_duplicates("source_id").head(n).reset_index(drop=True)


def _split(value: object) -> list[str]:
    return [item.strip() for item in re.split(r"[|;,]", _clean(value)) if item.strip()]


def validate_screening_sheet(sheet: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return row-level issues; blank decisions remain valid work in progress."""
    issues: list[dict[str, object]] = []
    decisions = {item.value for item in ScreeningOutcome}
    criteria = {item.value for item in CriterionStatus}
    reasons = set(config["exclusion_reasons"])
    groups = set(config["priority_groups"])
    if "source_id" in sheet:
        duplicated = sheet["source_id"].duplicated(keep=False)
        for index, source_id in sheet.loc[duplicated, "source_id"].items():
            issues.append({"row_number": index + 2, "source_id": source_id, "field": "source_id", "issue": "duplicate_source_id"})
    criterion_fields = (
        "fisheries_or_marine_relevant",
        "climate_environment_or_adaptation_element",
        "contributes_codable_evidence",
    )
    for index, row in sheet.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        decision = _clean(row.get("decision"))
        if not decision:
            continue
        if decision not in decisions:
            issues.append({"row_number": index + 2, "source_id": source_id, "field": "decision", "issue": f"invalid_decision:{decision}"})
            continue
        statuses = {field: _clean(row.get(field)) for field in criterion_fields}
        for field, status in statuses.items():
            if status not in criteria:
                issues.append({"row_number": index + 2, "source_id": source_id, "field": field, "issue": f"invalid_or_blank_criterion:{status}"})
        if any(status not in criteria for status in statuses.values()):
            continue
        reason = _clean(row.get("exclusion_reason"))
        priority_groups = _split(row.get("priority_groups"))
        try:
            TitleAbstractScreening(
                source_id=source_id,
                decision=decision,
                source_type=_clean(row.get("source_type")) or "other",
                fisheries_or_marine_relevant=statuses[criterion_fields[0]],
                climate_environment_or_adaptation_element=statuses[criterion_fields[1]],
                contributes_codable_evidence=statuses[criterion_fields[2]],
                exclusion_reason=reason or None,
                priority_groups=priority_groups,
                reviewer=_clean(row.get("reviewer")) or None,
                review_date=_clean(row.get("review_date")) or None,
                reviewer_notes=_clean(row.get("reviewer_notes")) or None,
                human_validation_status=_clean(row.get("human_validation_status")) or "not_reviewed",
            )
        except ValueError as exc:
            issues.append({"row_number": index + 2, "source_id": source_id, "field": "screening_logic", "issue": str(exc).replace("\n", " ")})
        if reason and reason not in reasons:
            issues.append({"row_number": index + 2, "source_id": source_id, "field": "exclusion_reason", "issue": f"unknown_exclusion_reason:{reason}"})
        unknown = sorted(set(priority_groups) - groups)
        if unknown:
            issues.append({"row_number": index + 2, "source_id": source_id, "field": "priority_groups", "issue": f"unknown_priority_groups:{'|'.join(unknown)}"})
    return pd.DataFrame(issues, columns=["row_number", "source_id", "field", "issue"])


def build_screening_prompt(row: dict, config: dict) -> str:
    """Create a JSON-only prompt without calling an external model."""
    payload = {
        "task": "Screen one source for a climate-fisheries systematic evidence map.",
        "rules": [
            "Use only supplied title, abstract, and metadata.",
            "Include only when all three criteria are yes.",
            "Exclude only when at least one criterion is clearly no.",
            "Use uncertain for absent, truncated, or ambiguous information.",
            "Small pelagics and anchoveta are priorities, not mandatory conditions.",
            "Return valid JSON only.",
        ],
        "allowed_decisions": config["decisions"],
        "allowed_criterion_values": config["criterion_values"],
        "allowed_exclusion_reasons": config["exclusion_reasons"],
        "allowed_priority_groups": config["priority_groups"],
        "source": {key: row.get(key, "") for key in (
            "source_id", "title", "abstract", "authors_or_organisation", "publisher",
            "year", "source_type", "source_status", "source_language", "doi", "primary_url",
        )},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_prompt_jsonl(sample: pd.DataFrame, config: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in sample.to_dict(orient="records"):
            stream.write(json.dumps({"source_id": row["source_id"], "prompt": build_screening_prompt(row, config)}, ensure_ascii=False) + "\n")
    return output


def screening_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    decisions = sheet.get("decision", pd.Series("", index=sheet.index)).fillna("").astype(str).str.strip()
    return pd.DataFrame({
        "metric": ["total_sources", "not_screened", "include", "exclude", "uncertain", "completed_decisions"],
        "value": [
            len(sheet), int(decisions.eq("").sum()), int(decisions.eq("include").sum()),
            int(decisions.eq("exclude").sum()), int(decisions.eq("uncertain").sum()),
            int(decisions.ne("").sum()),
        ],
    })


def completed_binary_decisions(sheet: pd.DataFrame) -> pd.DataFrame:
    """Convert completed include/exclude rows to the project's binary schema."""
    completed = sheet.loc[sheet["decision"].isin(["include", "exclude"])].copy()
    if completed.empty:
        return pd.DataFrame(columns=BINARY_COLUMNS)
    mapping = {"yes": True, "no": False}
    output = pd.DataFrame({
        "source_id": completed["source_id"],
        "include": completed["decision"].eq("include"),
        "source_type": completed["source_type"],
        "fisheries_or_marine_relevant": completed["fisheries_or_marine_relevant"].map(mapping),
        "climate_environment_or_adaptation_element": completed["climate_environment_or_adaptation_element"].map(mapping),
        "contributes_codable_evidence": completed["contributes_codable_evidence"].map(mapping),
        "exclusion_reason": completed["exclusion_reason"],
        "priority_groups": completed["priority_groups"],
        "screening_stage": "title_abstract",
        "reviewer_notes": completed["reviewer_notes"],
        "human_validation_status": completed["human_validation_status"].replace("", "not_reviewed"),
    })
    return output[BINARY_COLUMNS].reset_index(drop=True)
