"""Source-type-specific critical appraisal for climate-fisheries evidence."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping

import pandas as pd
from pydantic import BaseModel, Field, model_validator
import yaml


SOURCE_CONTEXT_COLUMNS = [
    "source_id",
    "title",
    "source_type",
    "doi",
    "primary_url",
    "page_count",
    "total_characters",
    "parsing_status",
    "full_text_priority_groups",
    "full_text_human_validation_status",
    "validated_findings_count",
]

BASE_REVIEW_COLUMNS = [
    "appraisal_status",
    "appraisal_domain",
    "total_score",
    "maximum_score",
    "normalized_score",
    "overall_rating",
    "appraisal_summary",
    "critical_limitations",
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

ISSUE_COLUMNS = [
    "row_number",
    "source_id",
    "field",
    "issue",
]


class AppraisalStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    INSUFFICIENT_TEXT = "insufficient_text"
    APPRAISAL_ERROR = "appraisal_error"


class HumanValidationStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class SourceQualityRecord(BaseModel):
    """Validated source-level quality appraisal."""

    source_id: str = Field(min_length=3)
    appraisal_status: AppraisalStatus
    appraisal_domain: str = Field(min_length=1)
    criteria_scores: dict[str, int] = Field(default_factory=dict)
    criteria_notes: dict[str, str] = Field(default_factory=dict)
    criteria_locators: dict[str, str] = Field(default_factory=dict)
    total_score: int | None = Field(default=None, ge=0)
    maximum_score: int | None = Field(default=None, ge=0)
    normalized_score: float | None = Field(default=None, ge=0, le=1)
    overall_rating: str = "not_applicable"
    appraisal_summary: str | None = None
    critical_limitations: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    review_date: str | None = None
    reviewer_notes: str | None = None
    human_validation_status: HumanValidationStatus = (
        HumanValidationStatus.NOT_REVIEWED
    )

    @model_validator(mode="after")
    def validate_scores(self) -> "SourceQualityRecord":
        invalid = {
            criterion: value
            for criterion, value in self.criteria_scores.items()
            if value not in {0, 1, 2}
        }
        if invalid:
            raise ValueError(f"quality scores must be 0, 1, or 2: {invalid}")

        if self.appraisal_status == AppraisalStatus.COMPLETED:
            if not self.criteria_scores:
                raise ValueError("completed appraisal requires criterion scores")
            calculated_total = sum(self.criteria_scores.values())
            calculated_maximum = 2 * len(self.criteria_scores)
            calculated_normalized = (
                calculated_total / calculated_maximum
                if calculated_maximum
                else 0.0
            )
            if (
                self.total_score is not None
                and self.total_score != calculated_total
            ):
                raise ValueError("total_score does not match criterion scores")
            if (
                self.maximum_score is not None
                and self.maximum_score != calculated_maximum
            ):
                raise ValueError(
                    "maximum_score does not match criterion scores"
                )
            if (
                self.normalized_score is not None
                and abs(self.normalized_score - calculated_normalized) > 1e-6
            ):
                raise ValueError(
                    "normalized_score does not match criterion scores"
                )
            self.total_score = calculated_total
            self.maximum_score = calculated_maximum
            self.normalized_score = calculated_normalized
        return self


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
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", text).strip()


def _split(value: object) -> list[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    return [
        item.strip()
        for item in re.split(r"[|;]", _clean(value))
        if item.strip()
    ]


def _serialise_list(value: object) -> str:
    return " | ".join(dict.fromkeys(_split(value)))


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


def load_quality_appraisal_config(
    path: str | Path = "config/quality_appraisal.yml",
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Load and validate quality-appraisal configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("quality appraisal configuration must be a mapping")

    required = {
        "paths",
        "taxonomy_path",
        "appraisal_statuses",
        "human_validation_status",
        "overall_ratings",
        "score_values",
        "rating_thresholds",
        "source_type_to_domain",
        "criteria",
        "packet_selection",
        "validation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f"missing quality appraisal configuration sections: {missing}"
        )

    root = Path(project_root) if project_root else config_path.parent.parent
    taxonomy_path = Path(config["taxonomy_path"])
    if not taxonomy_path.is_absolute():
        taxonomy_path = root / taxonomy_path
    with taxonomy_path.open("r", encoding="utf-8") as stream:
        taxonomy = yaml.safe_load(stream)
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be a mapping")
    config["taxonomy"] = taxonomy

    domains = set(config["criteria"]) - {"common"}
    mapped_domains = set(config["source_type_to_domain"].values())
    unknown_domains = sorted(mapped_domains - domains)
    if unknown_domains:
        raise ValueError(
            f"source type mapping contains unknown domains: {unknown_domains}"
        )

    moderate = float(config["rating_thresholds"]["moderate_minimum"])
    high = float(config["rating_thresholds"]["high_minimum"])
    if not 0 <= moderate < high <= 1:
        raise ValueError("rating thresholds must satisfy 0 <= moderate < high <= 1")

    packet = config["packet_selection"]
    if int(packet["max_chunks_per_source"]) <= 0:
        raise ValueError("max_chunks_per_source must be positive")
    if int(packet["max_characters_per_source"]) <= 0:
        raise ValueError("max_characters_per_source must be positive")

    return config


def all_criteria(config: Mapping[str, object]) -> list[str]:
    """Return a stable union of all common and domain-specific criteria."""
    output: list[str] = []
    for _, criteria in config["criteria"].items():
        for criterion in criteria:
            if criterion not in output:
                output.append(criterion)
    return output


def criteria_for_domain(
    config: Mapping[str, object],
    domain: str,
) -> list[str]:
    """Return common plus domain-specific criteria."""
    common = list(config["criteria"]["common"])
    specific = list(config["criteria"].get(domain, {}))
    return common + specific


def score_column(criterion: str) -> str:
    return f"score__{criterion}"


def note_column(criterion: str) -> str:
    return f"note__{criterion}"


def locator_column(criterion: str) -> str:
    return f"locator__{criterion}"


def criterion_columns(config: Mapping[str, object]) -> list[str]:
    columns: list[str] = []
    for criterion in all_criteria(config):
        columns.extend(
            [
                score_column(criterion),
                note_column(criterion),
                locator_column(criterion),
            ]
        )
    return columns


def appraisal_columns(config: Mapping[str, object]) -> list[str]:
    return SOURCE_CONTEXT_COLUMNS + BASE_REVIEW_COLUMNS + criterion_columns(
        config
    )


def appraisal_domain_for_source_type(
    source_type: object,
    config: Mapping[str, object],
) -> str:
    mapping = config["source_type_to_domain"]
    return mapping.get(_clean(source_type), mapping.get("other", "scientific"))


def _source_counts(findings: pd.DataFrame | None) -> pd.Series:
    if (
        findings is None
        or findings.empty
        or "source_id" not in findings
    ):
        return pd.Series(dtype=int)
    return findings.groupby("source_id").size()


def initialise_quality_appraisal_sheet(
    sources: pd.DataFrame,
    findings: pd.DataFrame | None,
    config: Mapping[str, object],
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one quality-appraisal row per included source."""
    if sources is None or sources.empty:
        return pd.DataFrame(columns=appraisal_columns(config))
    if "source_id" not in sources:
        raise ValueError("sources must contain source_id")

    source_rows = sources.drop_duplicates("source_id", keep="last").copy()
    for field in SOURCE_CONTEXT_COLUMNS:
        if field not in source_rows:
            source_rows[field] = ""
    counts = _source_counts(findings)
    source_rows["validated_findings_count"] = (
        source_rows["source_id"].map(counts).fillna(0).astype(int).astype(str)
    )

    sheet = source_rows[SOURCE_CONTEXT_COLUMNS].copy()
    for field in BASE_REVIEW_COLUMNS + criterion_columns(config):
        sheet[field] = ""
    sheet["appraisal_status"] = "pending"
    sheet["appraisal_domain"] = sheet["source_type"].map(
        lambda value: appraisal_domain_for_source_type(value, config)
    )
    sheet["overall_rating"] = "not_applicable"
    sheet["human_validation_status"] = "not_reviewed"

    if (
        existing is not None
        and not existing.empty
        and "source_id" in existing
    ):
        previous = (
            existing.drop_duplicates("source_id", keep="last")
            .set_index("source_id")
        )
        preservable = BASE_REVIEW_COLUMNS + criterion_columns(config)
        for field in preservable:
            if field in previous:
                mapped = sheet["source_id"].map(previous[field])
                mask = mapped.notna()
                sheet.loc[mask, field] = mapped.loc[mask].astype(str)

    protected = sheet["human_validation_status"].isin(
        ["accepted", "corrected", "rejected"]
    )
    expected_domains = sheet["source_type"].map(
        lambda value: appraisal_domain_for_source_type(value, config)
    )
    sheet.loc[~protected, "appraisal_domain"] = expected_domains.loc[
        ~protected
    ]

    return sheet[appraisal_columns(config)].reset_index(drop=True)


def parse_page_locators(value: object) -> list[int]:
    """Extract explicit p./pp./page/pages references."""
    pages: set[int] = set()
    pattern = re.compile(
        r"(?i)\b(?:pp?\.?|pages?)\s*(\d+)"
        r"(?:\s*[-–]\s*(\d+))?"
    )
    for match in pattern.finditer(_clean(value)):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        pages.update(range(start, end + 1))
    return sorted(pages)


def _keyword_score(text: object, terms: Iterable[str]) -> int:
    normalised = _normalise(text)
    score = 0
    for term in terms:
        candidate = _normalise(term)
        if candidate:
            score += normalised.count(candidate)
    return score


def _spread_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted(
        {
            round(index * (length - 1) / (count - 1))
            for index in range(count)
        }
    )


def select_quality_appraisal_packets(
    corpus: pd.DataFrame,
    sources: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Select deterministic source-level packets emphasizing quality evidence."""
    if corpus is None or corpus.empty:
        return pd.DataFrame(columns=PACKET_COLUMNS)
    required = {"source_id", "chunk_id", "start_page", "end_page", "text"}
    missing = sorted(required - set(corpus.columns))
    if missing:
        raise ValueError(f"appraisal corpus missing columns: {missing}")

    source_titles = (
        sources.drop_duplicates("source_id", keep="last")
        .set_index("source_id")["title"]
        if sources is not None
        and not sources.empty
        and {"source_id", "title"} <= set(sources.columns)
        else pd.Series(dtype=str)
    )
    settings = config["packet_selection"]
    maximum = int(settings["max_chunks_per_source"])
    max_characters = int(settings["max_characters_per_source"])
    edge = int(settings["edge_chunks"])
    spread = int(settings["spread_chunks"])
    terms = list(settings.get("keyword_terms", []))

    records: list[dict[str, object]] = []
    for source_id, group in corpus.fillna("").groupby(
        "source_id",
        sort=False,
    ):
        ordered = group.copy()
        for field in ("start_page", "end_page", "section_index", "chunk_index"):
            if field not in ordered:
                ordered[field] = 0
            ordered[field] = pd.to_numeric(
                ordered[field],
                errors="coerce",
            ).fillna(0)
        ordered["character_count"] = ordered["text"].map(
            lambda value: len(_clean(value))
        )
        ordered["_keyword_score"] = ordered["text"].map(
            lambda value: _keyword_score(value, terms)
        )
        ordered = ordered.sort_values(
            ["start_page", "section_index", "chunk_index", "chunk_id"]
        ).reset_index(drop=True)

        reasons: dict[int, set[str]] = defaultdict(set)

        def mark(indices: Iterable[int], reason: str) -> None:
            for index in indices:
                if 0 <= index < len(ordered):
                    reasons[index].add(reason)

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
        mark(ranked[:maximum], "quality_keyword_priority")

        candidates = sorted(
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
        for index in candidates:
            text = _clean(ordered.loc[index, "text"])
            if not text:
                continue
            if chosen and (
                len(chosen) >= maximum
                or characters + len(text) > max_characters
            ):
                continue
            chosen.append(index)
            characters += len(text)
            if len(chosen) >= maximum:
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
        title = _clean(source_titles.get(source_id, ""))
        for packet_order, index in enumerate(chosen, start=1):
            row = ordered.loc[index]
            records.append(
                {
                    "source_id": _clean(source_id),
                    "title": title or _clean(row.get("title")),
                    "chunk_id": _clean(row.get("chunk_id")),
                    "section_heading": _clean(row.get("section_heading")),
                    "start_page": int(row.get("start_page") or 0),
                    "end_page": int(row.get("end_page") or 0),
                    "character_count": int(
                        row.get("character_count") or 0
                    ),
                    "keyword_score": int(
                        row.get("_keyword_score") or 0
                    ),
                    "selection_reason": " | ".join(
                        sorted(reasons.get(index, {"fallback"}))
                    ),
                    "packet_order": packet_order,
                    "text": _clean(row.get("text")),
                }
            )
    return pd.DataFrame(records, columns=PACKET_COLUMNS)


def _criterion_descriptions(
    config: Mapping[str, object],
    domain: str,
) -> dict[str, str]:
    output = dict(config["criteria"]["common"])
    output.update(config["criteria"].get(domain, {}))
    return output


def build_quality_appraisal_prompt(
    row: Mapping[str, object],
    packet: pd.DataFrame,
    config: Mapping[str, object],
) -> str:
    """Build one JSON-only quality-appraisal prompt."""
    domain = _clean(row.get("appraisal_domain")) or (
        appraisal_domain_for_source_type(row.get("source_type"), config)
    )
    packet_records = []
    for item in packet.sort_values("packet_order").to_dict(
        orient="records"
    ):
        pages = (
            str(item.get("start_page", ""))
            if item.get("start_page") == item.get("end_page")
            else f"{item.get('start_page', '')}-{item.get('end_page', '')}"
        )
        packet_records.append(
            {
                "chunk_id": item.get("chunk_id", ""),
                "section_heading": item.get("section_heading", ""),
                "pages": pages,
                "selection_reason": item.get("selection_reason", ""),
                "text": item.get("text", ""),
            }
        )

    payload = {
        "task": (
            "Appraise the methodological and documentary quality of one "
            "climate-fisheries evidence source."
        ),
        "rules": [
            "Use only the supplied source metadata and page-traceable packet.",
            "Score every common and domain-specific criterion as 0, 1, or 2.",
            "Every score requires a concise note and an exact page locator.",
            "Do not use journal prestige, citation count, or institutional reputation as quality proxies.",
            "Do not treat legal authority as methodological quality or methodological quality as legal authority.",
            "Do not treat project activities or outputs as evaluated outcomes or impacts.",
            "Do not treat recommendations, projections, or scenarios as observed effectiveness evidence.",
            "Absence of implementation evidence is not evidence of implementation failure.",
            "Use insufficient_text when the packet cannot support a defensible appraisal.",
            "Return one valid JSON object only.",
        ],
        "score_scale": config["score_values"],
        "appraisal_domain": domain,
        "criteria": _criterion_descriptions(config, domain),
        "required_response_shape": {
            "source_id": "string",
            "appraisal_status": (
                "completed | insufficient_text | appraisal_error"
            ),
            "appraisal_domain": domain,
            "criteria": {
                "criterion_name": {
                    "score": "0 | 1 | 2",
                    "note": "concise justification",
                    "locator": "p. x or pp. x-y",
                }
            },
            "appraisal_summary": "source-level synthesis",
            "critical_limitations": [],
            "reviewer_notes": "",
        },
        "source": {
            key: row.get(key, "")
            for key in SOURCE_CONTEXT_COLUMNS
        },
        "quality_packet": packet_records,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_quality_appraisal_prompts_jsonl(
    sheet: pd.DataFrame,
    packets: pd.DataFrame,
    config: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write one prompt per source without calling an external model."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        source_id: frame.copy()
        for source_id, frame in packets.groupby("source_id", sort=False)
    }
    with output.open("w", encoding="utf-8") as stream:
        for row in sheet.to_dict(orient="records"):
            source_id = _clean(row.get("source_id"))
            packet = groups.get(
                source_id,
                pd.DataFrame(columns=PACKET_COLUMNS),
            )
            stream.write(
                json.dumps(
                    {
                        "source_id": source_id,
                        "prompt": build_quality_appraisal_prompt(
                            row,
                            packet,
                            config,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def expected_rating(
    normalized_score: float | None,
    config: Mapping[str, object],
) -> str:
    if normalized_score is None:
        return "not_applicable"
    thresholds = config["rating_thresholds"]
    if normalized_score >= float(thresholds["high_minimum"]):
        return "high"
    if normalized_score >= float(thresholds["moderate_minimum"]):
        return "moderate"
    return "low"


def _response_payload(payload: object, line_number: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"JSONL line {line_number} is not an object")
    if isinstance(payload.get("result"), dict):
        return payload["result"]
    if isinstance(payload.get("response"), str):
        nested = json.loads(payload["response"])
        if not isinstance(nested, dict):
            raise ValueError(
                f"response on line {line_number} is not an object"
            )
        return nested
    return payload


def _flatten_response(
    payload: Mapping[str, object],
    source_row: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    row = {
        field: _clean(source_row.get(field))
        for field in SOURCE_CONTEXT_COLUMNS
    }
    for field in BASE_REVIEW_COLUMNS + criterion_columns(config):
        row[field] = ""

    status = _clean(payload.get("appraisal_status")) or "appraisal_error"
    domain = _clean(payload.get("appraisal_domain")) or (
        appraisal_domain_for_source_type(source_row.get("source_type"), config)
    )
    row["appraisal_status"] = status
    row["appraisal_domain"] = domain
    row["appraisal_summary"] = _clean(
        payload.get("appraisal_summary")
    )
    row["critical_limitations"] = _serialise_list(
        payload.get("critical_limitations", [])
    )
    row["reviewer"] = (
        _clean(payload.get("reviewer"))
        or "AI-assisted preliminary quality appraisal"
    )
    row["review_date"] = _clean(payload.get("review_date"))
    row["reviewer_notes"] = _clean(payload.get("reviewer_notes"))
    row["human_validation_status"] = (
        _clean(payload.get("human_validation_status"))
        or "not_reviewed"
    )

    criteria = payload.get("criteria", {})
    if not isinstance(criteria, dict):
        raise ValueError("criteria response must be an object")

    scores: dict[str, int] = {}
    for criterion in criteria_for_domain(config, domain):
        item = criteria.get(criterion, {})
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if score not in ("", None):
            try:
                score_value = int(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"non-integer score for {criterion}: {score}"
                ) from exc
            row[score_column(criterion)] = str(score_value)
            scores[criterion] = score_value
        row[note_column(criterion)] = _clean(item.get("note"))
        row[locator_column(criterion)] = _clean(item.get("locator"))

    if status == "completed" and scores:
        total = sum(scores.values())
        maximum = 2 * len(scores)
        normalized = total / maximum if maximum else 0.0
        row["total_score"] = str(total)
        row["maximum_score"] = str(maximum)
        row["normalized_score"] = f"{normalized:.6f}"
        row["overall_rating"] = expected_rating(normalized, config)
    else:
        row["overall_rating"] = "not_applicable"

    return {
        field: row.get(field, "")
        for field in appraisal_columns(config)
    }


def read_quality_appraisal_responses_jsonl(
    path: str | Path,
    sheet: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Read one model- or reviewer-produced appraisal per source."""
    source_lookup = {
        _clean(row["source_id"]): row
        for row in sheet.to_dict(orient="records")
    }
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = _response_payload(
                    json.loads(line),
                    line_number,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc}"
                ) from exc
            source_id = _clean(payload.get("source_id"))
            if source_id not in source_lookup:
                raise ValueError(
                    f"unknown source_id on line {line_number}: {source_id}"
                )
            records.append(
                _flatten_response(
                    payload,
                    source_lookup[source_id],
                    config,
                )
            )
    return pd.DataFrame(records, columns=appraisal_columns(config))


def merge_quality_appraisals(
    sheet: pd.DataFrame,
    incoming: pd.DataFrame,
    config: Mapping[str, object],
    *,
    overwrite_human_validated: bool = False,
) -> pd.DataFrame:
    """Merge source appraisals while protecting human-validated rows."""
    if incoming is None or incoming.empty:
        return sheet.copy()
    if "source_id" not in incoming:
        raise ValueError("incoming appraisals require source_id")

    output = sheet.copy()
    indexed = (
        incoming.drop_duplicates("source_id", keep="last")
        .set_index("source_id")
    )
    protected = {"accepted", "corrected", "rejected"}
    fields = BASE_REVIEW_COLUMNS + criterion_columns(config)

    for index, row in output.iterrows():
        source_id = _clean(row.get("source_id"))
        if source_id not in indexed.index:
            continue
        if (
            not overwrite_human_validated
            and _clean(row.get("human_validation_status")) in protected
        ):
            continue
        for field in fields:
            if field in indexed:
                value = indexed.at[source_id, field]
                if isinstance(value, pd.Series):
                    value = value.iloc[-1]
                output.at[index, field] = _clean(value)
    return output[appraisal_columns(config)].reset_index(drop=True)


def _numeric(value: object) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate_quality_appraisal_sheet(
    sheet: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate source-level quality scores, evidence, and calculations."""
    issues: list[dict[str, object]] = []
    if "source_id" not in sheet:
        return pd.DataFrame(
            [
                {
                    "row_number": 1,
                    "source_id": "",
                    "field": "source_id",
                    "issue": "missing_column",
                }
            ],
            columns=ISSUE_COLUMNS,
        )

    duplicated = sheet["source_id"].astype(str).duplicated(keep=False)
    for index, row in sheet.loc[duplicated].iterrows():
        issues.append(
            {
                "row_number": index + 2,
                "source_id": _clean(row.get("source_id")),
                "field": "source_id",
                "issue": "duplicate_source_id",
            }
        )

    valid_statuses = set(config["appraisal_statuses"])
    valid_human = set(config["human_validation_status"])
    valid_ratings = set(config["overall_ratings"])
    validation = config["validation"]

    for index, row in sheet.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        status = _clean(row.get("appraisal_status"))
        domain = _clean(row.get("appraisal_domain"))
        human = (
            _clean(row.get("human_validation_status"))
            or "not_reviewed"
        )
        base = {
            "row_number": index + 2,
            "source_id": source_id,
        }

        if status not in valid_statuses:
            issues.append(
                {
                    **base,
                    "field": "appraisal_status",
                    "issue": f"unknown_appraisal_status:{status}",
                }
            )
            continue
        if human not in valid_human:
            issues.append(
                {
                    **base,
                    "field": "human_validation_status",
                    "issue": f"unknown_human_validation_status:{human}",
                }
            )

        expected_domain = appraisal_domain_for_source_type(
            row.get("source_type"),
            config,
        )
        if domain != expected_domain:
            issues.append(
                {
                    **base,
                    "field": "appraisal_domain",
                    "issue": (
                        f"domain_{domain}_differs_from_expected_"
                        f"{expected_domain}"
                    ),
                }
            )

        if status == "pending":
            continue

        required = criteria_for_domain(config, domain)
        scored: dict[str, int] = {}
        for criterion in required:
            score_text = _clean(row.get(score_column(criterion)))
            note = _clean(row.get(note_column(criterion)))
            locator = _clean(row.get(locator_column(criterion)))
            if status == "completed" and not score_text:
                issues.append(
                    {
                        **base,
                        "field": score_column(criterion),
                        "issue": "completed_appraisal_requires_score",
                    }
                )
                continue
            if not score_text:
                continue
            try:
                score = int(score_text)
            except ValueError:
                issues.append(
                    {
                        **base,
                        "field": score_column(criterion),
                        "issue": f"score_not_integer:{score_text}",
                    }
                )
                continue
            if score not in {0, 1, 2}:
                issues.append(
                    {
                        **base,
                        "field": score_column(criterion),
                        "issue": f"score_out_of_range:{score}",
                    }
                )
                continue
            scored[criterion] = score

            if (
                validation.get("require_note_for_scored_criterion", True)
                and len(note)
                < int(validation["minimum_note_characters"])
            ):
                issues.append(
                    {
                        **base,
                        "field": note_column(criterion),
                        "issue": "criterion_note_missing_or_too_short",
                    }
                )
            pages = parse_page_locators(locator)
            if (
                validation.get(
                    "require_locator_for_scored_criterion",
                    True,
                )
                and not locator
            ):
                issues.append(
                    {
                        **base,
                        "field": locator_column(criterion),
                        "issue": "criterion_locator_required",
                    }
                )
            elif (
                validation.get("require_explicit_page_locator", True)
                and locator
                and not pages
            ):
                issues.append(
                    {
                        **base,
                        "field": locator_column(criterion),
                        "issue": "explicit_page_locator_required",
                    }
                )

            page_count = _clean(row.get("page_count"))
            if page_count.isdigit() and pages:
                outside = [
                    page
                    for page in pages
                    if page < 1 or page > int(page_count)
                ]
                if outside:
                    issues.append(
                        {
                            **base,
                            "field": locator_column(criterion),
                            "issue": (
                                "page_locator_out_of_range:"
                                + "|".join(map(str, outside))
                            ),
                        }
                    )

        irrelevant = set(all_criteria(config)) - set(required)
        for criterion in irrelevant:
            if _clean(row.get(score_column(criterion))):
                issues.append(
                    {
                        **base,
                        "field": score_column(criterion),
                        "issue": "criterion_not_applicable_to_domain",
                    }
                )

        if status == "completed":
            summary = _clean(row.get("appraisal_summary"))
            if len(summary) < int(
                validation["minimum_summary_characters"]
            ):
                issues.append(
                    {
                        **base,
                        "field": "appraisal_summary",
                        "issue": "appraisal_summary_missing_or_too_short",
                    }
                )

            total = sum(scored.values())
            maximum = 2 * len(scored)
            normalized = total / maximum if maximum else 0.0
            recorded_total = _numeric(row.get("total_score"))
            recorded_maximum = _numeric(row.get("maximum_score"))
            recorded_normalized = _numeric(row.get("normalized_score"))
            rating = _clean(row.get("overall_rating"))
            expected = expected_rating(normalized, config)

            if recorded_total != total:
                issues.append(
                    {
                        **base,
                        "field": "total_score",
                        "issue": (
                            f"recorded_{recorded_total}_differs_from_"
                            f"calculated_{total}"
                        ),
                    }
                )
            if recorded_maximum != maximum:
                issues.append(
                    {
                        **base,
                        "field": "maximum_score",
                        "issue": (
                            f"recorded_{recorded_maximum}_differs_from_"
                            f"calculated_{maximum}"
                        ),
                    }
                )
            if (
                recorded_normalized is None
                or abs(recorded_normalized - normalized) > 1e-5
            ):
                issues.append(
                    {
                        **base,
                        "field": "normalized_score",
                        "issue": (
                            f"recorded_{recorded_normalized}_differs_from_"
                            f"calculated_{normalized:.6f}"
                        ),
                    }
                )
            if rating not in valid_ratings:
                issues.append(
                    {
                        **base,
                        "field": "overall_rating",
                        "issue": f"unknown_overall_rating:{rating}",
                    }
                )
            elif rating != expected:
                issues.append(
                    {
                        **base,
                        "field": "overall_rating",
                        "issue": (
                            f"rating_{rating}_differs_from_expected_"
                            f"{expected}"
                        ),
                    }
                )

            try:
                SourceQualityRecord(
                    source_id=source_id,
                    appraisal_status=status,
                    appraisal_domain=domain,
                    criteria_scores=scored,
                    criteria_notes={
                        criterion: _clean(
                            row.get(note_column(criterion))
                        )
                        for criterion in scored
                    },
                    criteria_locators={
                        criterion: _clean(
                            row.get(locator_column(criterion))
                        )
                        for criterion in scored
                    },
                    total_score=int(recorded_total)
                    if recorded_total is not None
                    else None,
                    maximum_score=int(recorded_maximum)
                    if recorded_maximum is not None
                    else None,
                    normalized_score=recorded_normalized,
                    overall_rating=rating,
                    appraisal_summary=summary,
                    critical_limitations=_split(
                        row.get("critical_limitations")
                    ),
                    reviewer=_clean(row.get("reviewer")) or None,
                    review_date=_clean(row.get("review_date")) or None,
                    reviewer_notes=_clean(
                        row.get("reviewer_notes")
                    )
                    or None,
                    human_validation_status=human,
                )
            except (ValueError, TypeError) as exc:
                issues.append(
                    {
                        **base,
                        "field": "appraisal_logic",
                        "issue": str(exc).replace("\n", " "),
                    }
                )
        else:
            rating = _clean(row.get("overall_rating"))
            if rating not in {"", "not_applicable"}:
                issues.append(
                    {
                        **base,
                        "field": "overall_rating",
                        "issue": (
                            "non_completed_appraisal_must_be_"
                            "not_applicable"
                        ),
                    }
                )

    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def appraisal_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    """Summarise appraisal completion, validation, domain, and rating."""
    status = sheet.get(
        "appraisal_status",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    human = sheet.get(
        "human_validation_status",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    rating = sheet.get(
        "overall_rating",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    rows = [
        ("sources_total", len(sheet)),
        ("appraisals_pending", int(status.eq("pending").sum())),
        ("appraisals_completed", int(status.eq("completed").sum())),
        (
            "appraisals_insufficient_text",
            int(status.eq("insufficient_text").sum()),
        ),
        (
            "appraisal_errors",
            int(status.eq("appraisal_error").sum()),
        ),
        (
            "appraisals_not_reviewed",
            int(human.eq("not_reviewed").sum()),
        ),
        ("appraisals_accepted", int(human.eq("accepted").sum())),
        ("appraisals_corrected", int(human.eq("corrected").sum())),
        ("appraisals_rejected", int(human.eq("rejected").sum())),
        ("quality_low", int(rating.eq("low").sum())),
        ("quality_moderate", int(rating.eq("moderate").sum())),
        ("quality_high", int(rating.eq("high").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def split_quality_appraisal_outputs(
    sheet: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split active, validated, pending-validation, and rejected appraisals."""
    rejected = sheet.loc[
        sheet["human_validation_status"].eq("rejected")
    ].copy()
    active = sheet.loc[
        ~sheet["human_validation_status"].eq("rejected")
    ].copy()
    validated = active.loc[
        active["appraisal_status"].eq("completed")
        & active["human_validation_status"].isin(
            ["accepted", "corrected"]
        )
    ].copy()
    pending = active.loc[
        active["human_validation_status"].eq("not_reviewed")
        | ~active["appraisal_status"].eq("completed")
    ].copy()
    return {
        "appraisal": active.reset_index(drop=True),
        "validated": validated.reset_index(drop=True),
        "pending_validation": pending.reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
    }


def build_finding_quality_link(
    findings: pd.DataFrame,
    validated_appraisals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach validated source-quality results to validated findings."""
    if findings is None or findings.empty:
        return pd.DataFrame()
    fields = [
        "source_id",
        "appraisal_domain",
        "total_score",
        "maximum_score",
        "normalized_score",
        "overall_rating",
        "appraisal_summary",
        "critical_limitations",
        "human_validation_status",
    ]
    if (
        validated_appraisals is None
        or validated_appraisals.empty
    ):
        output = findings.copy()
        for field in fields[1:]:
            output[f"quality_{field}"] = ""
        return output
    quality = validated_appraisals.copy()
    for field in fields:
        if field not in quality:
            quality[field] = ""
    quality = quality[fields].drop_duplicates(
        "source_id",
        keep="last",
    )
    quality = quality.rename(
        columns={
            field: f"quality_{field}"
            for field in fields
            if field != "source_id"
        }
    )
    return findings.merge(quality, on="source_id", how="left")


def build_source_quality_summary(
    sheet: pd.DataFrame,
) -> pd.DataFrame:
    """Return one compact source-level quality summary."""
    columns = [
        "source_id",
        "title",
        "source_type",
        "appraisal_domain",
        "validated_findings_count",
        "appraisal_status",
        "total_score",
        "maximum_score",
        "normalized_score",
        "overall_rating",
        "critical_limitations",
        "human_validation_status",
    ]
    output = sheet.copy()
    for column in columns:
        if column not in output:
            output[column] = ""
    return output[columns].sort_values(
        ["overall_rating", "normalized_score", "title"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
