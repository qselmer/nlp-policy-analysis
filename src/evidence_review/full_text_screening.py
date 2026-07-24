"""Full-text eligibility screening for the climate-fisheries evidence map."""

from __future__ import annotations

from enum import Enum
from hashlib import sha1
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping

import pandas as pd
from pydantic import BaseModel, Field, model_validator
import yaml


class CriterionStatus(str, Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class ScreeningOutcome(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


ADMINISTRATIVE_EXCLUSION_REASONS = {
    "duplicate_record",
    "superseded_version",
    "wrong_document",
    "publication_type_not_eligible",
    "outside_date_range",
}

CONTEXT_COLUMNS = [
    "source_id",
    "title",
    "source_type",
    "doi",
    "primary_url",
    "title_abstract_decision",
    "full_text_screening_required",
    "local_path",
    "source_checksum_sha256",
    "page_count",
    "total_characters",
    "ocr_recommended",
    "parsing_status",
]

REVIEW_COLUMNS = [
    "decision",
    "fisheries_or_marine_relevant",
    "climate_environment_or_adaptation_element",
    "contributes_codable_evidence",
    "sufficient_full_text_for_verification",
    "exclusion_reason",
    "priority_groups",
    "decision_basis",
    "supporting_page_locators",
    "reviewer",
    "review_date",
    "reviewer_notes",
    "human_validation_status",
]

PACKET_COLUMNS = [
    "source_id",
    "title",
    "chunk_id",
    "section_heading",
    "start_page",
    "end_page",
    "character_count",
    "keyword_score",
    "selection_reason",
    "packet_order",
    "text",
]

DECISION_COLUMNS = [
    "source_id",
    "include",
    "title_abstract_decision",
    "decision",
    "fisheries_or_marine_relevant",
    "climate_environment_or_adaptation_element",
    "contributes_codable_evidence",
    "sufficient_full_text_for_verification",
    "exclusion_reason",
    "priority_groups",
    "decision_basis",
    "supporting_page_locators",
    "screening_stage",
    "reviewer",
    "review_date",
    "reviewer_notes",
    "human_validation_status",
]

EXCLUSION_LOG_COLUMNS = [
    "source_id",
    "title",
    "source_type",
    "title_abstract_decision",
    "exclusion_reason",
    "decision_basis",
    "supporting_page_locators",
    "reviewer",
    "review_date",
    "human_validation_status",
]


class FullTextScreeningRecord(BaseModel):
    """Validated full-text eligibility decision."""

    source_id: str = Field(min_length=3)
    decision: ScreeningOutcome
    fisheries_or_marine_relevant: CriterionStatus
    climate_environment_or_adaptation_element: CriterionStatus
    contributes_codable_evidence: CriterionStatus
    sufficient_full_text_for_verification: CriterionStatus
    exclusion_reason: str | None = None
    priority_groups: list[str] = Field(default_factory=list)
    decision_basis: str = Field(min_length=10)
    supporting_page_locators: str = Field(min_length=1)
    reviewer: str | None = None
    review_date: str | None = None
    reviewer_notes: str | None = None
    human_validation_status: str = "not_reviewed"

    @model_validator(mode="after")
    def validate_logic(self) -> "FullTextScreeningRecord":
        scope = (
            self.fisheries_or_marine_relevant,
            self.climate_environment_or_adaptation_element,
            self.contributes_codable_evidence,
        )
        all_criteria = scope + (self.sufficient_full_text_for_verification,)

        if self.decision == ScreeningOutcome.INCLUDE:
            if any(value != CriterionStatus.YES for value in all_criteria):
                raise ValueError("include requires yes for all four criteria")
            if self.exclusion_reason:
                raise ValueError("included sources cannot have an exclusion_reason")

        elif self.decision == ScreeningOutcome.EXCLUDE:
            if not self.exclusion_reason:
                raise ValueError("excluded sources require an exclusion_reason")
            administrative = self.exclusion_reason in ADMINISTRATIVE_EXCLUSION_REASONS
            if CriterionStatus.NO not in scope and not administrative:
                raise ValueError(
                    "exclude requires a no scope criterion or an administrative exclusion"
                )

        else:
            if CriterionStatus.NO in scope:
                raise ValueError("uncertain cannot contain a no scope criterion")
            if CriterionStatus.UNCLEAR not in all_criteria:
                raise ValueError("uncertain requires at least one unclear criterion")
            if self.exclusion_reason:
                raise ValueError("uncertain sources cannot have an exclusion_reason")

        return self


def load_full_text_screening_config(
    path: str | Path = "config/full_text_screening.yml",
) -> dict:
    """Load and validate full-text screening configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("full-text screening configuration must be a YAML mapping")
    required = {
        "paths",
        "operational_rule",
        "eligibility_criteria",
        "criterion_values",
        "decisions",
        "eligibility_exclusion_reasons",
        "administrative_exclusion_reasons",
        "priority_groups",
        "human_validation_status",
        "packet_selection",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing full-text screening sections: {missing}")
    packet = config["packet_selection"]
    if int(packet["max_chunks_per_source"]) <= 0:
        raise ValueError("max_chunks_per_source must be positive")
    if int(packet["max_characters_per_source"]) <= 0:
        raise ValueError("max_characters_per_source must be positive")
    if int(packet["edge_chunks"]) < 0 or int(packet["spread_chunks"]) < 0:
        raise ValueError("packet chunk counts cannot be negative")
    return config


def read_csv_robust(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV using common UTF-8 and Windows encodings."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Unable to decode CSV: " + " | ".join(errors))


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", _clean(value).casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", text).strip()


def _split(value: object) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[|;,]", _clean(value))
        if item.strip()
    ]


def _stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(_clean(value) for value in values)
    return f"{prefix}_{sha1(payload.encode('utf-8')).hexdigest()[:14]}"


def _metadata_map(
    metadata: pd.DataFrame | None,
    field: str,
) -> pd.Series | None:
    if metadata is None or metadata.empty or "source_id" not in metadata or field not in metadata:
        return None
    unique = metadata.drop_duplicates("source_id", keep="last").set_index("source_id")
    return unique[field]


def initialise_full_text_screening_sheet(
    corpus: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a compact review sheet and preserve prior reviewer fields."""
    if corpus is None or corpus.empty:
        return pd.DataFrame(columns=CONTEXT_COLUMNS + ["screening_stage"] + REVIEW_COLUMNS)
    if "source_id" not in corpus:
        raise ValueError("full-text screening corpus must include source_id")

    sheet = corpus.copy()
    sheet["source_id"] = sheet["source_id"].map(_clean)
    if sheet["source_id"].eq("").any():
        raise ValueError("full-text screening corpus contains blank source_id values")
    sheet = sheet.drop_duplicates("source_id", keep="last").reset_index(drop=True)

    rename = {
        "screening_decision": "title_abstract_decision",
        "source_checksum_sha256": "source_checksum_sha256",
    }
    sheet = sheet.rename(columns=rename)

    for field in CONTEXT_COLUMNS:
        if field not in sheet:
            sheet[field] = ""

    for field in ("source_type", "doi", "primary_url"):
        mapping = _metadata_map(metadata, field)
        if mapping is not None:
            values = sheet["source_id"].map(mapping)
            current = sheet[field].map(_clean)
            mask = current.eq("") & values.notna()
            sheet.loc[mask, field] = values.loc[mask].astype(str)

    sheet["screening_stage"] = "full_text"
    for field in REVIEW_COLUMNS:
        sheet[field] = ""

    if existing is not None and not existing.empty and "source_id" in existing:
        previous = (
            existing.drop_duplicates("source_id", keep="last")
            .set_index("source_id")
        )
        for field in REVIEW_COLUMNS:
            if field not in previous:
                continue
            values = sheet["source_id"].map(previous[field])
            mask = values.notna()
            sheet.loc[mask, field] = values.loc[mask].astype(str)

    return sheet[CONTEXT_COLUMNS + ["screening_stage"] + REVIEW_COLUMNS].copy()


def parse_page_locators(value: object) -> list[int]:
    """Parse page locators such as 'p. 3 | pp. 5-7' into page numbers."""
    text = _clean(value)
    if not text:
        return []
    pages: set[int] = set()
    pieces = re.split(r"[|;,]", text)
    for piece in pieces:
        cleaned = re.sub(
            r"\b(?:pages|page|pp|p|pags|pag|paginas|pagina|páginas|página)\.?\s*",
            "",
            piece,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            continue
        match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", cleaned)
        if match:
            start, end = map(int, match.groups())
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
            continue
        for number in re.findall(r"\d+", cleaned):
            pages.add(int(number))
    return sorted(pages)


def validate_full_text_screening_sheet(
    sheet: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Return row-level validation issues; blank decisions remain work in progress."""
    issues: list[dict[str, object]] = []
    decisions = set(config["decisions"])
    criteria = set(config["criterion_values"])
    reasons = set(config["eligibility_exclusion_reasons"]) | set(
        config["administrative_exclusion_reasons"]
    )
    groups = set(config["priority_groups"])
    human_statuses = set(config["human_validation_status"])
    criterion_fields = (
        "fisheries_or_marine_relevant",
        "climate_environment_or_adaptation_element",
        "contributes_codable_evidence",
        "sufficient_full_text_for_verification",
    )

    if "source_id" not in sheet:
        return pd.DataFrame(
            [{
                "row_number": 1,
                "source_id": "",
                "field": "source_id",
                "issue": "missing_column",
            }]
        )

    duplicated = sheet["source_id"].astype(str).duplicated(keep=False)
    for index, source_id in sheet.loc[duplicated, "source_id"].items():
        issues.append({
            "row_number": index + 2,
            "source_id": source_id,
            "field": "source_id",
            "issue": "duplicate_source_id",
        })

    for index, row in sheet.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        decision = _clean(row.get("decision"))
        if not decision:
            continue
        if decision not in decisions:
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": "decision",
                "issue": f"invalid_decision:{decision}",
            })
            continue

        statuses = {field: _clean(row.get(field)) for field in criterion_fields}
        invalid = {
            field: status for field, status in statuses.items()
            if status not in criteria
        }
        for field, status in invalid.items():
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": field,
                "issue": f"invalid_or_blank_criterion:{status}",
            })
        if invalid:
            continue

        reason = _clean(row.get("exclusion_reason"))
        priority_groups = _split(row.get("priority_groups"))
        basis = _clean(row.get("decision_basis"))
        locators = _clean(row.get("supporting_page_locators"))
        human_status = _clean(row.get("human_validation_status")) or "not_reviewed"

        try:
            FullTextScreeningRecord(
                source_id=source_id,
                decision=decision,
                fisheries_or_marine_relevant=statuses[
                    "fisheries_or_marine_relevant"
                ],
                climate_environment_or_adaptation_element=statuses[
                    "climate_environment_or_adaptation_element"
                ],
                contributes_codable_evidence=statuses[
                    "contributes_codable_evidence"
                ],
                sufficient_full_text_for_verification=statuses[
                    "sufficient_full_text_for_verification"
                ],
                exclusion_reason=reason or None,
                priority_groups=priority_groups,
                decision_basis=basis,
                supporting_page_locators=locators,
                reviewer=_clean(row.get("reviewer")) or None,
                review_date=_clean(row.get("review_date")) or None,
                reviewer_notes=_clean(row.get("reviewer_notes")) or None,
                human_validation_status=human_status,
            )
        except ValueError as exc:
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": "screening_logic",
                "issue": str(exc).replace("\n", " "),
            })

        if reason and reason not in reasons:
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": "exclusion_reason",
                "issue": f"unknown_exclusion_reason:{reason}",
            })

        unknown_groups = sorted(set(priority_groups) - groups)
        if unknown_groups:
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": "priority_groups",
                "issue": f"unknown_priority_groups:{'|'.join(unknown_groups)}",
            })

        if human_status not in human_statuses:
            issues.append({
                "row_number": index + 2,
                "source_id": source_id,
                "field": "human_validation_status",
                "issue": f"unknown_human_validation_status:{human_status}",
            })

        page_count_text = _clean(row.get("page_count"))
        pages = parse_page_locators(locators)
        if page_count_text.isdigit() and pages:
            page_count = int(page_count_text)
            outside = [page for page in pages if page < 1 or page > page_count]
            if outside:
                issues.append({
                    "row_number": index + 2,
                    "source_id": source_id,
                    "field": "supporting_page_locators",
                    "issue": "page_locator_out_of_range:" + "|".join(map(str, outside)),
                })

    return pd.DataFrame(
        issues,
        columns=["row_number", "source_id", "field", "issue"],
    )


def _keyword_score(text: object, terms: Iterable[str]) -> int:
    normalised = _normalise(text)
    score = 0
    for term in terms:
        candidate = _normalise(term)
        if candidate and candidate in normalised:
            score += normalised.count(candidate)
    return score


def _spread_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    })


def select_full_text_screening_packets(
    chunks: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select deterministic, page-traceable chunks for screening prompts."""
    if chunks is None or chunks.empty:
        return pd.DataFrame(columns=PACKET_COLUMNS)
    required = {"source_id", "chunk_id", "start_page", "end_page", "text"}
    missing = sorted(required - set(chunks.columns))
    if missing:
        raise ValueError(f"document chunks missing required columns: {missing}")

    settings = config["packet_selection"]
    max_chunks = int(settings["max_chunks_per_source"])
    max_characters = int(settings["max_characters_per_source"])
    edge = int(settings["edge_chunks"])
    spread = int(settings["spread_chunks"])
    terms = list(settings.get("keyword_terms", []))
    output: list[dict[str, object]] = []

    for source_id, source_chunks in chunks.fillna("").groupby("source_id", sort=False):
        ordered = source_chunks.copy()
        for field in ("section_index", "start_page", "end_page", "chunk_index"):
            if field not in ordered:
                ordered[field] = 0
            ordered[field] = pd.to_numeric(ordered[field], errors="coerce").fillna(0)
        ordered = ordered.sort_values(
            ["start_page", "section_index", "chunk_index", "chunk_id"]
        ).reset_index(drop=True)
        ordered["_keyword_score"] = ordered["text"].map(
            lambda value: _keyword_score(value, terms)
        )

        reasons: dict[int, set[str]] = {}

        def mark(indices: Iterable[int], reason: str) -> None:
            for index in indices:
                if 0 <= index < len(ordered):
                    reasons.setdefault(index, set()).add(reason)

        mark(range(min(edge, len(ordered))), "document_start")
        mark(
            range(max(0, len(ordered) - edge), len(ordered)),
            "document_end",
        )
        mark(_spread_indices(len(ordered), spread), "document_spread")

        ranked = ordered.sort_values(
            ["_keyword_score", "character_count", "start_page"],
            ascending=[False, False, True],
        ).index.tolist()
        mark(ranked[:max_chunks], "keyword_priority")

        candidate_indices = sorted(
            reasons,
            key=lambda index: (
                0 if "document_start" in reasons[index] else 1,
                0 if "document_end" in reasons[index] else 1,
                -int(ordered.loc[index, "_keyword_score"]),
                int(ordered.loc[index, "start_page"]),
                index,
            ),
        )

        chosen: list[int] = []
        characters = 0
        for index in candidate_indices:
            value = _clean(ordered.loc[index, "text"])
            if not value:
                continue
            if chosen and (
                len(chosen) >= max_chunks
                or characters + len(value) > max_characters
            ):
                continue
            chosen.append(index)
            characters += len(value)
            if len(chosen) >= max_chunks:
                break

        if not chosen and len(ordered):
            chosen = [0]

        chosen = sorted(
            chosen,
            key=lambda index: (
                int(ordered.loc[index, "start_page"]),
                int(ordered.loc[index, "end_page"]),
                index,
            ),
        )

        for packet_order, index in enumerate(chosen, start=1):
            row = ordered.loc[index]
            output.append({
                "source_id": _clean(source_id),
                "title": _clean(row.get("title")),
                "chunk_id": _clean(row.get("chunk_id")),
                "section_heading": _clean(row.get("section_heading")),
                "start_page": int(row.get("start_page") or 0),
                "end_page": int(row.get("end_page") or 0),
                "character_count": len(_clean(row.get("text"))),
                "keyword_score": int(row.get("_keyword_score") or 0),
                "selection_reason": " | ".join(sorted(reasons.get(index, {"fallback"}))),
                "packet_order": packet_order,
                "text": _clean(row.get("text")),
            })

    return pd.DataFrame(output, columns=PACKET_COLUMNS)


def build_full_text_screening_prompt(
    row: Mapping[str, object],
    packet: pd.DataFrame,
    config: Mapping[str, object],
) -> str:
    """Build a JSON-only prompt without calling an external model."""
    packet_records = []
    for item in packet.sort_values("packet_order").to_dict(orient="records"):
        packet_records.append({
            "chunk_id": item.get("chunk_id", ""),
            "section_heading": item.get("section_heading", ""),
            "pages": (
                str(item.get("start_page", ""))
                if item.get("start_page") == item.get("end_page")
                else f"{item.get('start_page', '')}-{item.get('end_page', '')}"
            ),
            "selection_reason": item.get("selection_reason", ""),
            "text": item.get("text", ""),
        })

    payload = {
        "task": (
            "Screen one source at full text for a systematic evidence map on "
            "climate change, environmental variability, adaptation, and marine fisheries."
        ),
        "rules": [
            "Use only the supplied source metadata and page-traceable text packet.",
            "The packet is representative; do not claim that unshown content was reviewed.",
            "Include only when all four criteria are yes.",
            "Exclude only when at least one scope criterion is no or a valid administrative reason applies.",
            "Use uncertain when scope is not clearly excluded but evidence or readable text remains insufficient.",
            "Lack of access or unreadable text is not an eligibility exclusion.",
            "Provide a concise decision_basis and exact supporting_page_locators.",
            "Small pelagics and anchoveta are priorities, not mandatory eligibility conditions.",
            "Return one valid JSON object only.",
        ],
        "criteria": config["eligibility_criteria"],
        "allowed_decisions": config["decisions"],
        "allowed_criterion_values": config["criterion_values"],
        "allowed_exclusion_reasons": (
            list(config["eligibility_exclusion_reasons"])
            + list(config["administrative_exclusion_reasons"])
        ),
        "allowed_priority_groups": config["priority_groups"],
        "required_output_fields": [
            "source_id",
            "decision",
            "fisheries_or_marine_relevant",
            "climate_environment_or_adaptation_element",
            "contributes_codable_evidence",
            "sufficient_full_text_for_verification",
            "exclusion_reason",
            "priority_groups",
            "decision_basis",
            "supporting_page_locators",
            "reviewer_notes",
        ],
        "source": {
            key: row.get(key, "")
            for key in (
                "source_id",
                "title",
                "source_type",
                "doi",
                "primary_url",
                "title_abstract_decision",
                "page_count",
                "total_characters",
                "ocr_recommended",
                "parsing_status",
            )
        },
        "screening_packet": packet_records,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_full_text_screening_prompts_jsonl(
    sheet: pd.DataFrame,
    packets: pd.DataFrame,
    config: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write one prompt per source to JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    packet_groups = {
        source_id: group.copy()
        for source_id, group in packets.groupby("source_id", sort=False)
    }
    with output.open("w", encoding="utf-8") as stream:
        for row in sheet.to_dict(orient="records"):
            source_id = _clean(row.get("source_id"))
            packet = packet_groups.get(
                source_id,
                pd.DataFrame(columns=PACKET_COLUMNS),
            )
            stream.write(json.dumps({
                "source_id": source_id,
                "prompt": build_full_text_screening_prompt(row, packet, config),
            }, ensure_ascii=False) + "\n")
    return output


def read_full_text_screening_decisions_jsonl(
    path: str | Path,
) -> pd.DataFrame:
    """Read model- or reviewer-produced JSONL decisions."""
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if isinstance(payload.get("result"), dict):
                payload = payload["result"]
            elif isinstance(payload.get("response"), str):
                payload = json.loads(payload["response"])
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(payload)

    columns = ["source_id"] + REVIEW_COLUMNS
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    if "priority_groups" in frame:
        frame["priority_groups"] = frame["priority_groups"].map(
            lambda value: " | ".join(value)
            if isinstance(value, list)
            else _clean(value)
        )
    frame["reviewer"] = frame["reviewer"].replace(
        "",
        "AI-assisted preliminary full-text screening",
    )
    frame["human_validation_status"] = frame["human_validation_status"].replace(
        "",
        "not_reviewed",
    )
    return frame[columns].copy()


def merge_full_text_screening_decisions(
    sheet: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    overwrite_human_validated: bool = False,
) -> pd.DataFrame:
    """Merge decisions by source_id while protecting validated human records."""
    if decisions is None or decisions.empty:
        return sheet.copy()
    if "source_id" not in decisions:
        raise ValueError("decision file must include source_id")

    output = sheet.copy()
    incoming = (
        decisions.drop_duplicates("source_id", keep="last")
        .set_index("source_id")
    )
    protected = {"accepted", "corrected", "rejected"}

    for index, row in output.iterrows():
        source_id = _clean(row.get("source_id"))
        if source_id not in incoming.index:
            continue
        status = _clean(row.get("human_validation_status"))
        if not overwrite_human_validated and status in protected:
            continue
        for field in REVIEW_COLUMNS:
            if field in incoming:
                value = incoming.at[source_id, field]
                if isinstance(value, pd.Series):
                    value = value.iloc[-1]
                output.at[index, field] = _clean(value)
    return output


def screening_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    """Summarise full-text screening progress and decisions."""
    decisions = sheet.get(
        "decision",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    previous = sheet.get(
        "title_abstract_decision",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    rows = [
        ("total_sources", len(sheet)),
        ("title_abstract_include", int(previous.eq("include").sum())),
        ("title_abstract_uncertain", int(previous.eq("uncertain").sum())),
        ("not_screened", int(decisions.eq("").sum())),
        ("include", int(decisions.eq("include").sum())),
        ("exclude", int(decisions.eq("exclude").sum())),
        ("uncertain", int(decisions.eq("uncertain").sum())),
        ("completed_binary_decisions", int(decisions.isin(["include", "exclude"]).sum())),
        (
            "title_include_changed_to_exclude",
            int(previous.eq("include").mul(decisions.eq("exclude")).sum()),
        ),
        (
            "title_uncertain_resolved",
            int(previous.eq("uncertain").mul(decisions.isin(["include", "exclude"])).sum()),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def completed_full_text_decisions(sheet: pd.DataFrame) -> pd.DataFrame:
    """Return final include/exclude rows in a stable schema."""
    completed = sheet.loc[sheet["decision"].isin(["include", "exclude"])].copy()
    if completed.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    output = pd.DataFrame({
        "source_id": completed["source_id"],
        "include": completed["decision"].eq("include"),
        "title_abstract_decision": completed["title_abstract_decision"],
        "decision": completed["decision"],
        "fisheries_or_marine_relevant": completed["fisheries_or_marine_relevant"],
        "climate_environment_or_adaptation_element": completed[
            "climate_environment_or_adaptation_element"
        ],
        "contributes_codable_evidence": completed["contributes_codable_evidence"],
        "sufficient_full_text_for_verification": completed[
            "sufficient_full_text_for_verification"
        ],
        "exclusion_reason": completed["exclusion_reason"],
        "priority_groups": completed["priority_groups"],
        "decision_basis": completed["decision_basis"],
        "supporting_page_locators": completed["supporting_page_locators"],
        "screening_stage": "full_text",
        "reviewer": completed["reviewer"],
        "review_date": completed["review_date"],
        "reviewer_notes": completed["reviewer_notes"],
        "human_validation_status": completed["human_validation_status"].replace(
            "",
            "not_reviewed",
        ),
    })
    return output[DECISION_COLUMNS].reset_index(drop=True)


def split_full_text_screening_outputs(
    sheet: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split included, excluded, uncertain, and pending records."""
    return {
        "included": sheet.loc[sheet["decision"].eq("include")].reset_index(drop=True),
        "excluded": sheet.loc[sheet["decision"].eq("exclude")].reset_index(drop=True),
        "uncertain": sheet.loc[sheet["decision"].eq("uncertain")].reset_index(drop=True),
        "pending": sheet.loc[
            sheet["decision"].astype(str).str.strip().eq("")
        ].reset_index(drop=True),
    }


def build_full_text_exclusion_log(sheet: pd.DataFrame) -> pd.DataFrame:
    """Return an auditable full-text exclusion log."""
    excluded = sheet.loc[sheet["decision"].eq("exclude")].copy()
    for column in EXCLUSION_LOG_COLUMNS:
        if column not in excluded:
            excluded[column] = ""
    return excluded[EXCLUSION_LOG_COLUMNS].reset_index(drop=True)


def build_evidence_extraction_corpus(
    chunks: pd.DataFrame,
    sheet: pd.DataFrame,
) -> pd.DataFrame:
    """Retain chunks only from sources included after full-text screening."""
    included = sheet.loc[
        sheet["decision"].eq("include"),
        [
            "source_id",
            "decision_basis",
            "supporting_page_locators",
            "priority_groups",
            "human_validation_status",
        ],
    ].drop_duplicates("source_id", keep="last")
    if chunks is None or chunks.empty or included.empty:
        columns = list(chunks.columns) if chunks is not None else []
        return pd.DataFrame(columns=columns + [
            "full_text_decision_basis",
            "full_text_supporting_page_locators",
            "full_text_priority_groups",
            "full_text_human_validation_status",
        ])
    output = chunks.merge(included, on="source_id", how="inner")
    return output.rename(columns={
        "decision_basis": "full_text_decision_basis",
        "supporting_page_locators": "full_text_supporting_page_locators",
        "priority_groups": "full_text_priority_groups",
        "human_validation_status": "full_text_human_validation_status",
    }).reset_index(drop=True)
