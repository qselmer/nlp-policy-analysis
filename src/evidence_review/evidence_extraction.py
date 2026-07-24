"""Structured, traceable finding extraction from screened document chunks."""

from __future__ import annotations

from collections import defaultdict
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


QUEUE_CONTEXT_COLUMNS = [
    "source_id",
    "title",
    "source_type",
    "doi",
    "primary_url",
    "chunk_id",
    "section_id",
    "section_heading",
    "start_page",
    "end_page",
    "character_count",
    "source_checksum_sha256",
    "full_text_priority_groups",
    "full_text_human_validation_status",
]

QUEUE_REVIEW_COLUMNS = [
    "extraction_status",
    "no_finding_reason",
    "candidate_findings_count",
    "reviewer",
    "review_date",
    "reviewer_notes",
    "human_validation_status",
]

FINDING_COLUMNS = [
    "finding_id",
    "source_id",
    "title",
    "source_type",
    "doi",
    "primary_url",
    "chunk_id",
    "section_id",
    "section_heading",
    "chunk_start_page",
    "chunk_end_page",
    "source_checksum_sha256",
    "finding_type",
    "evidence_streams",
    "unit_locator",
    "supporting_excerpt",
    "evidence_summary",
    "geographic_scope",
    "jurisdiction",
    "ecosystem_or_region",
    "fishery_scope",
    "species",
    "stock_or_management_unit",
    "climate_drivers",
    "environmental_variables",
    "biological_responses",
    "fishery_responses",
    "governance_mechanisms",
    "management_measures",
    "method_or_design",
    "study_period",
    "temporal_scale",
    "scenarios",
    "sample_or_data_description",
    "quantitative_result",
    "result_direction",
    "evidence_basis",
    "implementation_stage",
    "causal_interpretation",
    "uncertainty_reported",
    "relevance_to_small_pelagics",
    "relevance_to_anchoveta",
    "transferability_rationale",
    "limitations",
    "coder_confidence",
    "finding_fingerprint",
    "duplicate_group_id",
    "deduplication_status",
    "reviewer",
    "review_date",
    "validation_notes",
    "human_validation_status",
]

ISSUE_COLUMNS = [
    "output",
    "row_number",
    "source_id",
    "chunk_id",
    "finding_id",
    "field",
    "issue",
]

LIST_FIELDS = {
    "evidence_streams",
    "geographic_scope",
    "ecosystem_or_region",
    "fishery_scope",
    "species",
    "climate_drivers",
    "environmental_variables",
    "biological_responses",
    "fishery_responses",
    "governance_mechanisms",
    "management_measures",
    "method_or_design",
    "temporal_scale",
    "scenarios",
    "limitations",
}

TAXONOMY_LIST_FIELDS = {
    "source_type": "source_type",
    "finding_type": "finding_type",
    "evidence_streams": "evidence_stream",
    "fishery_scope": "fishery_scope",
    "climate_drivers": "climate_driver",
    "biological_responses": "biological_response",
    "fishery_responses": "fishery_response",
    "management_measures": "management_measure",
    "method_or_design": "method_or_design",
    "relevance_to_small_pelagics": "relevance_level",
    "relevance_to_anchoveta": "relevance_level",
}


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    FINDINGS_EXTRACTED = "findings_extracted"
    NO_CODABLE_FINDING = "no_codable_finding"
    NEEDS_MORE_CONTEXT = "needs_more_context"
    EXTRACTION_ERROR = "extraction_error"


class StructuredFindingRecord(BaseModel):
    """One independently verifiable finding extracted from one document chunk."""

    finding_id: str = Field(min_length=3)
    source_id: str = Field(min_length=3)
    chunk_id: str = Field(min_length=3)
    source_type: str = Field(min_length=1)
    finding_type: str = Field(min_length=1)
    evidence_streams: list[str] = Field(min_length=1)
    unit_locator: str = Field(min_length=1)
    supporting_excerpt: str = Field(min_length=1, max_length=1200)
    evidence_summary: str = Field(min_length=20)

    geographic_scope: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    ecosystem_or_region: list[str] = Field(default_factory=list)
    fishery_scope: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    stock_or_management_unit: str | None = None
    climate_drivers: list[str] = Field(default_factory=list)
    environmental_variables: list[str] = Field(default_factory=list)
    biological_responses: list[str] = Field(default_factory=list)
    fishery_responses: list[str] = Field(default_factory=list)
    governance_mechanisms: list[str] = Field(default_factory=list)
    management_measures: list[str] = Field(default_factory=list)
    method_or_design: list[str] = Field(default_factory=list)
    study_period: str | None = None
    temporal_scale: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    sample_or_data_description: str | None = None
    quantitative_result: str | None = None
    result_direction: str = "non_directional"
    evidence_basis: str = "unclear"
    implementation_stage: str = "not_applicable"
    causal_interpretation: str = "unclear"
    uncertainty_reported: bool | None = None
    relevance_to_small_pelagics: str = "unclear"
    relevance_to_anchoveta: str = "unclear"
    transferability_rationale: str | None = None
    limitations: list[str] = Field(default_factory=list)
    coder_confidence: float = Field(default=0.5, ge=0, le=1)
    reviewer: str | None = None
    review_date: str | None = None
    validation_notes: str | None = None
    human_validation_status: str = "not_reviewed"

    @model_validator(mode="after")
    def validate_distinctions(self) -> "StructuredFindingRecord":
        if self.finding_type == "model_projection" and self.evidence_basis not in {
            "modelled_projection",
            "scenario",
            "mixed",
            "unclear",
        }:
            raise ValueError(
                "model_projection requires a projection or scenario evidence_basis"
            )
        if self.finding_type == "scenario_result" and self.evidence_basis not in {
            "scenario",
            "modelled_projection",
            "mixed",
            "unclear",
        }:
            raise ValueError(
                "scenario_result requires a scenario-compatible evidence_basis"
            )
        if self.evidence_basis == "observed" and self.finding_type in {
            "model_projection",
            "scenario_result",
        }:
            raise ValueError(
                "projected or scenario findings cannot be coded as observed"
            )
        if self.implementation_stage in {
            "evaluated_outcome",
            "evaluated_impact",
        } and self.finding_type in {
            "project_activity",
            "project_output",
            "policy_commitment",
        }:
            raise ValueError(
                "activities, outputs, and commitments cannot be coded as "
                "evaluated outcomes or impacts"
            )
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
        character for character in text if not unicodedata.combining(character)
    )
    text = text.replace("…", "...")
    return re.sub(r"\s+", " ", text).strip()


def _split(value: object) -> list[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    return [
        item.strip()
        for item in re.split(r"[|;,]", _clean(value))
        if item.strip()
    ]


def _serialise_list(value: object) -> str:
    return " | ".join(dict.fromkeys(_split(value)))


def _stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(_clean(value) for value in values)
    return f"{prefix}_{sha1(payload.encode('utf-8')).hexdigest()[:14]}"


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


def load_finding_extraction_config(
    path: str | Path = "config/finding_extraction.yml",
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Load phase configuration and the shared controlled taxonomy."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(
            "finding extraction configuration must be a YAML mapping"
        )
    required = {
        "paths",
        "taxonomy_path",
        "extraction_statuses",
        "no_finding_reasons",
        "human_validation_status",
        "new_controlled_vocabularies",
        "prompting",
        "validation",
        "near_duplicate_detection",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f"missing finding extraction configuration sections: {missing}"
        )

    root = Path(project_root) if project_root else config_path.parent.parent
    taxonomy_path = Path(config["taxonomy_path"])
    if not taxonomy_path.is_absolute():
        taxonomy_path = root / taxonomy_path
    with taxonomy_path.open("r", encoding="utf-8") as stream:
        taxonomy = yaml.safe_load(stream)
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be a YAML mapping")
    config["taxonomy"] = taxonomy

    maximum = int(config["prompting"]["max_findings_per_chunk"])
    if maximum <= 0:
        raise ValueError("max_findings_per_chunk must be positive")
    threshold = float(
        config["near_duplicate_detection"]["token_jaccard_threshold"]
    )
    if not 0 <= threshold <= 1:
        raise ValueError(
            "token_jaccard_threshold must be between 0 and 1"
        )
    return config


def enrich_extraction_corpus(
    corpus: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add source metadata while preserving one row per chunk."""
    if corpus is None or corpus.empty:
        return pd.DataFrame(columns=QUEUE_CONTEXT_COLUMNS + ["text"])
    required = {
        "source_id",
        "chunk_id",
        "start_page",
        "end_page",
        "text",
    }
    missing = sorted(required - set(corpus.columns))
    if missing:
        raise ValueError(
            f"evidence extraction corpus missing required columns: {missing}"
        )

    output = corpus.copy().reset_index(drop=True)
    for field in QUEUE_CONTEXT_COLUMNS + ["text"]:
        if field not in output:
            output[field] = ""

    if (
        metadata is not None
        and not metadata.empty
        and "source_id" in metadata
    ):
        source_meta = (
            metadata.drop_duplicates("source_id", keep="last")
            .set_index("source_id")
        )
        for field in ("title", "source_type", "doi", "primary_url"):
            if field in source_meta:
                mapped = output["source_id"].map(source_meta[field])
                current_blank = output[field].map(_clean).eq("")
                mask = current_blank & mapped.notna()
                output.loc[mask, field] = mapped.loc[mask].astype(str)

    output["start_page"] = (
        pd.to_numeric(output["start_page"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    output["end_page"] = (
        pd.to_numeric(output["end_page"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    output["character_count"] = output["text"].map(
        lambda value: len(_clean(value))
    )
    return output[QUEUE_CONTEXT_COLUMNS + ["text"]].copy()


def initialise_extraction_queue(
    corpus: pd.DataFrame,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one auditable extraction decision row per chunk."""
    queue = corpus[QUEUE_CONTEXT_COLUMNS].copy()
    for field in QUEUE_REVIEW_COLUMNS:
        queue[field] = ""
    queue["extraction_status"] = "pending"
    queue["candidate_findings_count"] = "0"
    queue["human_validation_status"] = "not_reviewed"

    if (
        existing is not None
        and not existing.empty
        and "chunk_id" in existing
    ):
        previous = (
            existing.drop_duplicates("chunk_id", keep="last")
            .set_index("chunk_id")
        )
        for field in QUEUE_REVIEW_COLUMNS:
            if field in previous:
                mapped = queue["chunk_id"].map(previous[field])
                mask = mapped.notna()
                queue.loc[mask, field] = mapped.loc[mask].astype(str)
    return queue[
        QUEUE_CONTEXT_COLUMNS + QUEUE_REVIEW_COLUMNS
    ].reset_index(drop=True)


def initialise_findings_working(
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a stable findings sheet, preserving existing rows when supplied."""
    if existing is None or existing.empty:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    output = existing.copy()
    for field in FINDING_COLUMNS:
        if field not in output:
            output[field] = ""
    return output[FINDING_COLUMNS].copy().reset_index(drop=True)


def _taxonomy_payload(
    config: Mapping[str, object],
) -> dict[str, object]:
    taxonomy = config["taxonomy"]
    payload = {
        key: taxonomy.get(key, [])
        for key in (
            "source_type",
            "evidence_stream",
            "finding_type",
            "fishery_scope",
            "climate_driver",
            "biological_response",
            "fishery_response",
            "management_measure",
            "method_or_design",
            "relevance_level",
        )
    }
    payload.update(config["new_controlled_vocabularies"])
    return payload


def build_structured_finding_prompt(
    row: Mapping[str, object],
    config: Mapping[str, object],
) -> str:
    """Build one JSON-only extraction prompt for one page-traceable chunk."""
    payload = {
        "task": (
            "Extract zero or more independently verifiable climate-fisheries "
            "findings from one document chunk."
        ),
        "rules": [
            "Use only the supplied metadata and chunk text; do not add external facts.",
            "Extract the smallest independently verifiable finding rather than a whole-document summary.",
            "Return no_codable_finding when the chunk has no independently codable contribution.",
            "Use needs_more_context when the chunk hints at a finding but cannot support it on its own.",
            "Every finding must preserve the chunk_id, exact page locator, and a short verbatim supporting excerpt.",
            "A recommendation is not an observed result.",
            "A projection or scenario is not an observed trend.",
            "A proposal, policy commitment, or legal authorization is not evidence of implementation or effectiveness.",
            "A project activity or output is not an outcome or impact without evaluated change.",
            "Correlation is not causation unless the design supports causal interpretation.",
            "Use unclear or empty values rather than guessing.",
            "Return one valid JSON object only.",
        ],
        "allowed_extraction_statuses": config["extraction_statuses"],
        "allowed_no_finding_reasons": config["no_finding_reasons"],
        "maximum_findings": int(
            config["prompting"]["max_findings_per_chunk"]
        ),
        "controlled_vocabularies": _taxonomy_payload(config),
        "required_response_shape": {
            "source_id": "string",
            "chunk_id": "string",
            "extraction_status": (
                "findings_extracted | no_codable_finding | "
                "needs_more_context | extraction_error"
            ),
            "no_finding_reason": "allowed value or empty",
            "reviewer_notes": "string or empty",
            "findings": [
                {
                    "finding_type": "controlled value",
                    "evidence_streams": [
                        "one or more controlled values"
                    ],
                    "unit_locator": (
                        "page or page range within this chunk"
                    ),
                    "supporting_excerpt": (
                        "short verbatim excerpt from chunk"
                    ),
                    "evidence_summary": "faithful synthesis",
                    "geographic_scope": [],
                    "jurisdiction": "",
                    "ecosystem_or_region": [],
                    "fishery_scope": [],
                    "species": [],
                    "stock_or_management_unit": "",
                    "climate_drivers": [],
                    "environmental_variables": [],
                    "biological_responses": [],
                    "fishery_responses": [],
                    "governance_mechanisms": [],
                    "management_measures": [],
                    "method_or_design": [],
                    "study_period": "",
                    "temporal_scale": [],
                    "scenarios": [],
                    "sample_or_data_description": "",
                    "quantitative_result": "",
                    "result_direction": "controlled value",
                    "evidence_basis": "controlled value",
                    "implementation_stage": "controlled value",
                    "causal_interpretation": "controlled value",
                    "uncertainty_reported": "true | false | null",
                    "relevance_to_small_pelagics": (
                        "controlled value"
                    ),
                    "relevance_to_anchoveta": "controlled value",
                    "transferability_rationale": "",
                    "limitations": [],
                    "coder_confidence": "number from 0 to 1",
                }
            ],
        },
        "source": {
            key: row.get(key, "")
            for key in (
                "source_id",
                "title",
                "source_type",
                "doi",
                "primary_url",
                "full_text_priority_groups",
                "full_text_human_validation_status",
            )
        },
        "chunk": {
            key: row.get(key, "")
            for key in (
                "chunk_id",
                "section_id",
                "section_heading",
                "start_page",
                "end_page",
                "source_checksum_sha256",
                "text",
            )
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_structured_finding_prompts_jsonl(
    corpus: pd.DataFrame,
    config: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Write one prompt per chunk without calling an external model."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in corpus.to_dict(orient="records"):
            stream.write(
                json.dumps(
                    {
                        "source_id": _clean(row.get("source_id")),
                        "chunk_id": _clean(row.get("chunk_id")),
                        "prompt": build_structured_finding_prompt(
                            row,
                            config,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def _response_payload(
    payload: object,
    line_number: int,
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            f"JSONL line {line_number} is not an object"
        )
    if isinstance(payload.get("result"), dict):
        payload = payload["result"]
    elif isinstance(payload.get("response"), str):
        payload = json.loads(payload["response"])
    if not isinstance(payload, dict):
        raise ValueError(
            f"JSONL line {line_number} does not contain an object response"
        )
    return payload


def _bool_or_blank(value: object) -> bool | None:
    if value is None or _clean(value) == "":
        return None
    if isinstance(value, bool):
        return value
    normalised = _normalise(value)
    if normalised in {"true", "yes", "1"}:
        return True
    if normalised in {"false", "no", "0"}:
        return False
    return None


def _finding_row(
    item: Mapping[str, object],
    chunk: Mapping[str, object],
    finding_index: int,
) -> dict[str, object]:
    summary = _clean(item.get("evidence_summary"))
    excerpt = _clean(item.get("supporting_excerpt"))
    finding_id = _clean(item.get("finding_id")) or _stable_id(
        "fnd",
        chunk.get("source_id"),
        chunk.get("chunk_id"),
        finding_index,
        summary,
        excerpt,
    )
    row: dict[str, object] = {
        "finding_id": finding_id,
        "source_id": _clean(chunk.get("source_id")),
        "title": _clean(chunk.get("title")),
        "source_type": _clean(chunk.get("source_type")) or "other",
        "doi": _clean(chunk.get("doi")),
        "primary_url": _clean(chunk.get("primary_url")),
        "chunk_id": _clean(chunk.get("chunk_id")),
        "section_id": _clean(chunk.get("section_id")),
        "section_heading": _clean(chunk.get("section_heading")),
        "chunk_start_page": _clean(chunk.get("start_page")),
        "chunk_end_page": _clean(chunk.get("end_page")),
        "source_checksum_sha256": _clean(
            chunk.get("source_checksum_sha256")
        ),
        "finding_type": _clean(item.get("finding_type")),
        "unit_locator": _clean(item.get("unit_locator")),
        "supporting_excerpt": excerpt,
        "evidence_summary": summary,
        "jurisdiction": _clean(item.get("jurisdiction")),
        "stock_or_management_unit": _clean(
            item.get("stock_or_management_unit")
        ),
        "study_period": _clean(item.get("study_period")),
        "sample_or_data_description": _clean(
            item.get("sample_or_data_description")
        ),
        "quantitative_result": _clean(
            item.get("quantitative_result")
        ),
        "result_direction": (
            _clean(item.get("result_direction"))
            or "non_directional"
        ),
        "evidence_basis": (
            _clean(item.get("evidence_basis")) or "unclear"
        ),
        "implementation_stage": (
            _clean(item.get("implementation_stage"))
            or "not_applicable"
        ),
        "causal_interpretation": (
            _clean(item.get("causal_interpretation")) or "unclear"
        ),
        "uncertainty_reported": (
            ""
            if _bool_or_blank(item.get("uncertainty_reported")) is None
            else str(
                _bool_or_blank(item.get("uncertainty_reported"))
            ).lower()
        ),
        "relevance_to_small_pelagics": (
            _clean(item.get("relevance_to_small_pelagics"))
            or "unclear"
        ),
        "relevance_to_anchoveta": (
            _clean(item.get("relevance_to_anchoveta"))
            or "unclear"
        ),
        "transferability_rationale": _clean(
            item.get("transferability_rationale")
        ),
        "coder_confidence": (
            _clean(item.get("coder_confidence")) or "0.5"
        ),
        "finding_fingerprint": "",
        "duplicate_group_id": "",
        "deduplication_status": "unique",
        "reviewer": (
            _clean(item.get("reviewer"))
            or "AI-assisted preliminary finding extraction"
        ),
        "review_date": _clean(item.get("review_date")),
        "validation_notes": _clean(item.get("validation_notes")),
        "human_validation_status": (
            _clean(item.get("human_validation_status"))
            or "not_reviewed"
        ),
    }
    for field in LIST_FIELDS:
        row[field] = _serialise_list(item.get(field, []))
    return {
        field: row.get(field, "")
        for field in FINDING_COLUMNS
    }


def read_structured_finding_responses_jsonl(
    path: str | Path,
    corpus: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read chunk-level responses and flatten their findings."""
    chunk_lookup = {
        _clean(row["chunk_id"]): row
        for row in corpus.to_dict(orient="records")
    }
    queue_updates: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    maximum = int(
        config["prompting"]["max_findings_per_chunk"]
    )

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

            chunk_id = _clean(payload.get("chunk_id"))
            if chunk_id not in chunk_lookup:
                raise ValueError(
                    f"unknown chunk_id on line {line_number}: {chunk_id}"
                )
            chunk = chunk_lookup[chunk_id]
            supplied_source = _clean(payload.get("source_id"))
            if (
                supplied_source
                and supplied_source != _clean(chunk.get("source_id"))
            ):
                raise ValueError(
                    "source_id does not match chunk_id on line "
                    f"{line_number}"
                )

            status = (
                _clean(payload.get("extraction_status"))
                or "extraction_error"
            )
            items = payload.get("findings", [])
            if not isinstance(items, list):
                raise ValueError(
                    f"findings must be a list on line {line_number}"
                )
            if len(items) > maximum:
                raise ValueError(
                    f"too many findings on line {line_number}: "
                    f"{len(items)} > {maximum}"
                )

            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"finding {index} on line {line_number} "
                        "is not an object"
                    )
                findings.append(
                    _finding_row(item, chunk, index)
                )

            queue_updates.append(
                {
                    "chunk_id": chunk_id,
                    "extraction_status": status,
                    "no_finding_reason": _clean(
                        payload.get("no_finding_reason")
                    ),
                    "candidate_findings_count": str(len(items)),
                    "reviewer": (
                        _clean(payload.get("reviewer"))
                        or "AI-assisted preliminary finding extraction"
                    ),
                    "review_date": _clean(payload.get("review_date")),
                    "reviewer_notes": _clean(
                        payload.get("reviewer_notes")
                    ),
                    "human_validation_status": (
                        _clean(payload.get("human_validation_status"))
                        or "not_reviewed"
                    ),
                }
            )

    return (
        pd.DataFrame(
            queue_updates,
            columns=["chunk_id"] + QUEUE_REVIEW_COLUMNS,
        ),
        pd.DataFrame(findings, columns=FINDING_COLUMNS),
    )


def merge_extraction_queue(
    queue: pd.DataFrame,
    updates: pd.DataFrame,
    *,
    overwrite_human_validated: bool = False,
) -> pd.DataFrame:
    """Merge chunk decisions while protecting accepted or corrected human rows."""
    if updates is None or updates.empty:
        return queue.copy()
    incoming = (
        updates.drop_duplicates("chunk_id", keep="last")
        .set_index("chunk_id")
    )
    output = queue.copy()
    protected = {"accepted", "corrected", "rejected"}
    for index, row in output.iterrows():
        chunk_id = _clean(row.get("chunk_id"))
        if chunk_id not in incoming.index:
            continue
        if (
            not overwrite_human_validated
            and _clean(row.get("human_validation_status"))
            in protected
        ):
            continue
        for field in QUEUE_REVIEW_COLUMNS:
            if field in incoming:
                output.at[index, field] = _clean(
                    incoming.at[chunk_id, field]
                )
    return output


def merge_findings_working(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    overwrite_human_validated: bool = False,
) -> pd.DataFrame:
    """Upsert finding rows by finding_id without overwriting validated human edits."""
    current = initialise_findings_working(existing)
    if incoming is None or incoming.empty:
        return current
    new = initialise_findings_working(incoming)
    protected = {"accepted", "corrected", "rejected"}
    current_by_id = {
        value: index
        for index, value in current["finding_id"].items()
    }

    for _, row in new.iterrows():
        finding_id = _clean(row.get("finding_id"))
        if finding_id in current_by_id:
            index = current_by_id[finding_id]
            status = _clean(
                current.at[index, "human_validation_status"]
            )
            if (
                not overwrite_human_validated
                and status in protected
            ):
                continue
            for field in FINDING_COLUMNS:
                current.at[index, field] = _clean(row.get(field))
        else:
            current = pd.concat(
                [
                    current,
                    pd.DataFrame(
                        [row],
                        columns=FINDING_COLUMNS,
                    ),
                ],
                ignore_index=True,
            )
            current_by_id[finding_id] = len(current) - 1
    return current[FINDING_COLUMNS].reset_index(drop=True)


def parse_page_locators(value: object) -> list[int]:
    """Extract explicit p./pp./page/pages references from a locator string."""
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


def _excerpt_matches(
    excerpt: object,
    chunk_text: object,
) -> bool:
    quote = _normalise(excerpt)
    text = _normalise(chunk_text)
    if not quote or not text:
        return False
    if quote in text:
        return True
    parts = [
        part.strip()
        for part in re.split(
            r"\.{3}|\[\s*\.\.\.\s*\]",
            quote,
        )
        if len(part.strip()) >= 12
    ]
    if not parts:
        return False
    cursor = 0
    for part in parts:
        position = text.find(part, cursor)
        if position < 0:
            return False
        cursor = position + len(part)
    return True


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalise(value))
        if len(token) > 2
    }


def _finding_fingerprint(
    row: Mapping[str, object],
) -> str:
    payload = "|".join(
        [
            _clean(row.get("source_id")),
            _normalise(row.get("finding_type")),
            _normalise(row.get("supporting_excerpt"))
            or _normalise(row.get("evidence_summary")),
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def _page_span(
    row: Mapping[str, object],
) -> tuple[int, int]:
    pages = parse_page_locators(row.get("unit_locator"))
    if pages:
        return min(pages), max(pages)
    try:
        return (
            int(float(_clean(row.get("chunk_start_page")))),
            int(float(_clean(row.get("chunk_end_page")))),
        )
    except ValueError:
        return 0, 0


def annotate_near_duplicate_findings(
    findings: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Flag likely duplicates created by overlapping chunks within the same source."""
    output = initialise_findings_working(findings)
    if output.empty:
        return output
    for index, row in output.iterrows():
        output.at[index, "finding_fingerprint"] = (
            _finding_fingerprint(row)
        )
        if _clean(row.get("deduplication_status")) not in {
            "confirmed_duplicate",
            "primary",
            "possible_duplicate",
        }:
            output.at[index, "deduplication_status"] = "unique"
            output.at[index, "duplicate_group_id"] = ""

    settings = config["near_duplicate_detection"]
    if not settings.get("enabled", True):
        return output
    threshold = float(settings["token_jaccard_threshold"])
    minimum = int(settings["minimum_tokens"])
    require_near_pages = bool(
        settings.get(
            "require_page_overlap_or_adjacency",
            True,
        )
    )

    parents = list(range(len(output)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        first = find(left)
        second = find(right)
        if first != second:
            parents[second] = first

    for _, indices in output.groupby(
        "source_id",
        sort=False,
    ).groups.items():
        indices = list(indices)
        for offset, left in enumerate(indices):
            left_row = output.loc[left]
            left_tokens = _tokens(
                left_row.get("supporting_excerpt")
            ) or _tokens(left_row.get("evidence_summary"))
            if len(left_tokens) < minimum:
                continue
            left_start, left_end = _page_span(left_row)
            for right in indices[offset + 1 :]:
                right_row = output.loc[right]
                right_tokens = _tokens(
                    right_row.get("supporting_excerpt")
                ) or _tokens(right_row.get("evidence_summary"))
                if len(right_tokens) < minimum:
                    continue
                right_start, right_end = _page_span(right_row)
                near_pages = not (
                    left_end + 1 < right_start
                    or right_end + 1 < left_start
                )
                if require_near_pages and not near_pages:
                    continue
                union_size = len(left_tokens | right_tokens)
                score = (
                    len(left_tokens & right_tokens) / union_size
                    if union_size
                    else 0
                )
                same_type = _clean(
                    left_row.get("finding_type")
                ) == _clean(right_row.get("finding_type"))
                if score >= threshold and same_type:
                    union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(output)):
        groups[find(index)].append(index)
    for members in groups.values():
        if len(members) < 2:
            continue
        identifiers = sorted(
            _clean(output.at[index, "finding_id"])
            for index in members
        )
        group_id = _stable_id(
            "dup",
            _clean(output.at[members[0], "source_id"]),
            *identifiers,
        )
        ordered = sorted(
            members,
            key=lambda index: (
                int(
                    float(
                        _clean(
                            output.at[
                                index,
                                "chunk_start_page",
                            ]
                        )
                        or 0
                    )
                ),
                _clean(output.at[index, "finding_id"]),
            ),
        )
        for position, index in enumerate(ordered):
            if (
                _clean(
                    output.at[index, "deduplication_status"]
                )
                == "confirmed_duplicate"
            ):
                continue
            output.at[index, "duplicate_group_id"] = group_id
            output.at[index, "deduplication_status"] = (
                "primary" if position == 0 else "possible_duplicate"
            )
    return output


def synchronise_queue_with_findings(
    queue: pd.DataFrame,
    findings: pd.DataFrame,
) -> pd.DataFrame:
    """Refresh candidate counts without changing explicit extraction decisions."""
    output = queue.copy()
    if not findings.empty:
        active = findings.loc[
            ~findings["human_validation_status"].eq("rejected")
            & ~findings["deduplication_status"].eq(
                "confirmed_duplicate"
            )
        ]
    else:
        active = findings
    counts = (
        active.groupby("chunk_id").size()
        if active is not None and not active.empty
        else pd.Series(dtype=int)
    )
    for index, row in output.iterrows():
        count = int(
            counts.get(_clean(row.get("chunk_id")), 0)
        )
        output.at[index, "candidate_findings_count"] = str(count)
    return output


def _allowed(
    config: Mapping[str, object],
    field: str,
) -> set[str]:
    if field in TAXONOMY_LIST_FIELDS:
        return set(
            config["taxonomy"].get(
                TAXONOMY_LIST_FIELDS[field],
                [],
            )
        )
    return set(
        config["new_controlled_vocabularies"].get(field, [])
    )


def validate_structured_finding_outputs(
    queue: pd.DataFrame,
    findings: pd.DataFrame,
    corpus: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate queue completeness, controlled vocabularies, and source traceability."""
    issues: list[dict[str, object]] = []
    chunk_lookup = {
        _clean(row["chunk_id"]): row
        for row in corpus.to_dict(orient="records")
    }
    valid_statuses = set(config["extraction_statuses"])
    valid_reasons = set(config["no_finding_reasons"])
    valid_human = set(config["human_validation_status"])

    if "chunk_id" not in queue:
        return pd.DataFrame(
            [
                {
                    "output": "queue",
                    "row_number": 1,
                    "source_id": "",
                    "chunk_id": "",
                    "finding_id": "",
                    "field": "chunk_id",
                    "issue": "missing_column",
                }
            ],
            columns=ISSUE_COLUMNS,
        )

    duplicated_chunks = queue["chunk_id"].astype(str).duplicated(
        keep=False
    )
    for index, row in queue.loc[duplicated_chunks].iterrows():
        issues.append(
            {
                "output": "queue",
                "row_number": index + 2,
                "source_id": _clean(row.get("source_id")),
                "chunk_id": _clean(row.get("chunk_id")),
                "finding_id": "",
                "field": "chunk_id",
                "issue": "duplicate_chunk_id",
            }
        )

    if not findings.empty:
        active_findings = findings.loc[
            ~findings["human_validation_status"].eq("rejected")
            & ~findings["deduplication_status"].eq(
                "confirmed_duplicate"
            )
        ]
    else:
        active_findings = findings
    finding_counts = (
        active_findings.groupby("chunk_id").size()
        if not active_findings.empty
        else pd.Series(dtype=int)
    )

    for index, row in queue.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        chunk_id = _clean(row.get("chunk_id"))
        status = _clean(row.get("extraction_status"))
        reason = _clean(row.get("no_finding_reason"))
        human = (
            _clean(row.get("human_validation_status"))
            or "not_reviewed"
        )
        count = int(finding_counts.get(chunk_id, 0))
        recorded = _clean(row.get("candidate_findings_count"))
        base = {
            "output": "queue",
            "row_number": index + 2,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "finding_id": "",
        }
        if chunk_id not in chunk_lookup:
            issues.append(
                {
                    **base,
                    "field": "chunk_id",
                    "issue": "unknown_chunk_id",
                }
            )
        if status not in valid_statuses:
            issues.append(
                {
                    **base,
                    "field": "extraction_status",
                    "issue": f"unknown_extraction_status:{status}",
                }
            )
        if human not in valid_human:
            issues.append(
                {
                    **base,
                    "field": "human_validation_status",
                    "issue": (
                        "unknown_human_validation_status:"
                        f"{human}"
                    ),
                }
            )
        if status == "findings_extracted" and count < 1:
            issues.append(
                {
                    **base,
                    "field": "extraction_status",
                    "issue": "findings_extracted_requires_finding",
                }
            )
        if status == "no_codable_finding" and not reason:
            issues.append(
                {
                    **base,
                    "field": "no_finding_reason",
                    "issue": "no_codable_finding_requires_reason",
                }
            )
        if reason and reason not in valid_reasons:
            issues.append(
                {
                    **base,
                    "field": "no_finding_reason",
                    "issue": f"unknown_no_finding_reason:{reason}",
                }
            )
        if (
            status
            in {
                "no_codable_finding",
                "needs_more_context",
                "extraction_error",
            }
            and count
        ):
            issues.append(
                {
                    **base,
                    "field": "candidate_findings_count",
                    "issue": (
                        f"status_{status}_cannot_have_active_findings"
                    ),
                }
            )
        if recorded.isdigit() and int(recorded) != count:
            issues.append(
                {
                    **base,
                    "field": "candidate_findings_count",
                    "issue": (
                        f"recorded_count_{recorded}_differs_from_"
                        f"actual_{count}"
                    ),
                }
            )

    if findings.empty:
        return pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    duplicated_findings = findings["finding_id"].astype(
        str
    ).duplicated(keep=False)
    for index, row in findings.loc[duplicated_findings].iterrows():
        issues.append(
            {
                "output": "findings",
                "row_number": index + 2,
                "source_id": _clean(row.get("source_id")),
                "chunk_id": _clean(row.get("chunk_id")),
                "finding_id": _clean(row.get("finding_id")),
                "field": "finding_id",
                "issue": "duplicate_finding_id",
            }
        )

    validation = config["validation"]
    list_vocab_fields = [
        "evidence_streams",
        "fishery_scope",
        "climate_drivers",
        "biological_responses",
        "fishery_responses",
        "management_measures",
        "method_or_design",
    ]
    scalar_vocab_fields = [
        "source_type",
        "finding_type",
        "result_direction",
        "evidence_basis",
        "implementation_stage",
        "causal_interpretation",
        "relevance_to_small_pelagics",
        "relevance_to_anchoveta",
        "deduplication_status",
    ]

    for index, row in findings.fillna("").iterrows():
        source_id = _clean(row.get("source_id"))
        chunk_id = _clean(row.get("chunk_id"))
        finding_id = _clean(row.get("finding_id"))
        chunk = chunk_lookup.get(chunk_id)
        base = {
            "output": "findings",
            "row_number": index + 2,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "finding_id": finding_id,
        }
        if chunk is None:
            issues.append(
                {
                    **base,
                    "field": "chunk_id",
                    "issue": "unknown_chunk_id",
                }
            )
            continue
        if source_id != _clean(chunk.get("source_id")):
            issues.append(
                {
                    **base,
                    "field": "source_id",
                    "issue": "source_id_does_not_match_chunk",
                }
            )

        for field in list_vocab_fields:
            unknown = sorted(
                set(_split(row.get(field)))
                - _allowed(config, field)
            )
            if unknown:
                issues.append(
                    {
                        **base,
                        "field": field,
                        "issue": (
                            "unknown_values:" + "|".join(unknown)
                        ),
                    }
                )
        for field in scalar_vocab_fields:
            value = _clean(row.get(field))
            if value and value not in _allowed(config, field):
                issues.append(
                    {
                        **base,
                        "field": field,
                        "issue": f"unknown_value:{value}",
                    }
                )

        try:
            StructuredFindingRecord(
                finding_id=finding_id,
                source_id=source_id,
                chunk_id=chunk_id,
                source_type=(
                    _clean(row.get("source_type")) or "other"
                ),
                finding_type=_clean(row.get("finding_type")),
                evidence_streams=_split(
                    row.get("evidence_streams")
                ),
                unit_locator=_clean(row.get("unit_locator")),
                supporting_excerpt=_clean(
                    row.get("supporting_excerpt")
                ),
                evidence_summary=_clean(
                    row.get("evidence_summary")
                ),
                geographic_scope=_split(
                    row.get("geographic_scope")
                ),
                jurisdiction=(
                    _clean(row.get("jurisdiction")) or None
                ),
                ecosystem_or_region=_split(
                    row.get("ecosystem_or_region")
                ),
                fishery_scope=_split(row.get("fishery_scope")),
                species=_split(row.get("species")),
                stock_or_management_unit=(
                    _clean(row.get("stock_or_management_unit"))
                    or None
                ),
                climate_drivers=_split(
                    row.get("climate_drivers")
                ),
                environmental_variables=_split(
                    row.get("environmental_variables")
                ),
                biological_responses=_split(
                    row.get("biological_responses")
                ),
                fishery_responses=_split(
                    row.get("fishery_responses")
                ),
                governance_mechanisms=_split(
                    row.get("governance_mechanisms")
                ),
                management_measures=_split(
                    row.get("management_measures")
                ),
                method_or_design=_split(
                    row.get("method_or_design")
                ),
                study_period=(
                    _clean(row.get("study_period")) or None
                ),
                temporal_scale=_split(
                    row.get("temporal_scale")
                ),
                scenarios=_split(row.get("scenarios")),
                sample_or_data_description=(
                    _clean(
                        row.get("sample_or_data_description")
                    )
                    or None
                ),
                quantitative_result=(
                    _clean(row.get("quantitative_result"))
                    or None
                ),
                result_direction=(
                    _clean(row.get("result_direction"))
                    or "non_directional"
                ),
                evidence_basis=(
                    _clean(row.get("evidence_basis"))
                    or "unclear"
                ),
                implementation_stage=(
                    _clean(row.get("implementation_stage"))
                    or "not_applicable"
                ),
                causal_interpretation=(
                    _clean(row.get("causal_interpretation"))
                    or "unclear"
                ),
                uncertainty_reported=_bool_or_blank(
                    row.get("uncertainty_reported")
                ),
                relevance_to_small_pelagics=(
                    _clean(
                        row.get("relevance_to_small_pelagics")
                    )
                    or "unclear"
                ),
                relevance_to_anchoveta=(
                    _clean(row.get("relevance_to_anchoveta"))
                    or "unclear"
                ),
                transferability_rationale=(
                    _clean(
                        row.get("transferability_rationale")
                    )
                    or None
                ),
                limitations=_split(row.get("limitations")),
                coder_confidence=float(
                    _clean(row.get("coder_confidence")) or 0.5
                ),
                reviewer=_clean(row.get("reviewer")) or None,
                review_date=(
                    _clean(row.get("review_date")) or None
                ),
                validation_notes=(
                    _clean(row.get("validation_notes")) or None
                ),
                human_validation_status=(
                    _clean(row.get("human_validation_status"))
                    or "not_reviewed"
                ),
            )
        except (ValueError, TypeError) as exc:
            issues.append(
                {
                    **base,
                    "field": "finding_logic",
                    "issue": str(exc).replace("\n", " "),
                }
            )

        human = (
            _clean(row.get("human_validation_status"))
            or "not_reviewed"
        )
        if human not in valid_human:
            issues.append(
                {
                    **base,
                    "field": "human_validation_status",
                    "issue": (
                        "unknown_human_validation_status:"
                        f"{human}"
                    ),
                }
            )

        excerpt = _clean(row.get("supporting_excerpt"))
        if (
            validation.get("require_supporting_excerpt", True)
            and not excerpt
        ):
            issues.append(
                {
                    **base,
                    "field": "supporting_excerpt",
                    "issue": "supporting_excerpt_required",
                }
            )
        if excerpt and len(excerpt) > int(
            validation["maximum_excerpt_characters"]
        ):
            issues.append(
                {
                    **base,
                    "field": "supporting_excerpt",
                    "issue": "supporting_excerpt_too_long",
                }
            )
        if (
            validation.get("excerpt_must_match_chunk", True)
            and excerpt
            and not _excerpt_matches(excerpt, chunk.get("text"))
        ):
            issues.append(
                {
                    **base,
                    "field": "supporting_excerpt",
                    "issue": "excerpt_not_found_in_chunk",
                }
            )

        locator = _clean(row.get("unit_locator"))
        pages = parse_page_locators(locator)
        if (
            validation.get("require_page_locator", True)
            and not pages
        ):
            issues.append(
                {
                    **base,
                    "field": "unit_locator",
                    "issue": "explicit_page_locator_required",
                }
            )
        if (
            validation.get("page_locator_within_chunk", True)
            and pages
        ):
            start = int(chunk.get("start_page") or 0)
            end = int(chunk.get("end_page") or 0)
            outside = [
                page
                for page in pages
                if page < start or page > end
            ]
            if outside:
                issues.append(
                    {
                        **base,
                        "field": "unit_locator",
                        "issue": (
                            "page_locator_outside_chunk:"
                            + "|".join(map(str, outside))
                        ),
                    }
                )

        confidence = _clean(row.get("coder_confidence"))
        try:
            value = float(confidence)
            if not (
                float(validation["coder_confidence_minimum"])
                <= value
                <= float(
                    validation["coder_confidence_maximum"]
                )
            ):
                issues.append(
                    {
                        **base,
                        "field": "coder_confidence",
                        "issue": "coder_confidence_out_of_range",
                    }
                )
        except ValueError:
            issues.append(
                {
                    **base,
                    "field": "coder_confidence",
                    "issue": "coder_confidence_not_numeric",
                }
            )

    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def extraction_summary(
    queue: pd.DataFrame,
    findings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise chunk processing, findings, validation, and duplicate flags."""
    status = queue.get(
        "extraction_status",
        pd.Series("", index=queue.index),
    ).fillna("").astype(str).str.strip()
    human = findings.get(
        "human_validation_status",
        pd.Series("", index=findings.index),
    ).fillna("").astype(str).str.strip()
    dedup = findings.get(
        "deduplication_status",
        pd.Series("", index=findings.index),
    ).fillna("").astype(str).str.strip()
    rows = [
        ("chunks_total", len(queue)),
        ("chunks_pending", int(status.eq("pending").sum())),
        (
            "chunks_with_findings",
            int(status.eq("findings_extracted").sum()),
        ),
        (
            "chunks_without_codable_findings",
            int(status.eq("no_codable_finding").sum()),
        ),
        (
            "chunks_needing_context",
            int(status.eq("needs_more_context").sum()),
        ),
        (
            "chunk_extraction_errors",
            int(status.eq("extraction_error").sum()),
        ),
        ("findings_total", len(findings)),
        (
            "sources_with_findings",
            int(findings["source_id"].nunique())
            if not findings.empty
            else 0,
        ),
        (
            "findings_not_reviewed",
            int(human.eq("not_reviewed").sum()),
        ),
        (
            "findings_accepted",
            int(human.eq("accepted").sum()),
        ),
        (
            "findings_corrected",
            int(human.eq("corrected").sum()),
        ),
        (
            "findings_rejected",
            int(human.eq("rejected").sum()),
        ),
        (
            "possible_duplicate_findings",
            int(dedup.eq("possible_duplicate").sum()),
        ),
        (
            "confirmed_duplicate_findings",
            int(dedup.eq("confirmed_duplicate").sum()),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def split_structured_finding_outputs(
    queue: pd.DataFrame,
    findings: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create stable outputs for appraisal and human validation."""
    if not findings.empty:
        active = findings.loc[
            ~findings["human_validation_status"].eq("rejected")
            & ~findings["deduplication_status"].eq(
                "confirmed_duplicate"
            )
        ].copy()
    else:
        active = findings.copy()
    validated = active.loc[
        active["human_validation_status"].isin(
            ["accepted", "corrected"]
        )
    ].copy()
    pending = active.loc[
        active["human_validation_status"].eq("not_reviewed")
    ].copy()
    if not findings.empty:
        rejected = findings.loc[
            findings["human_validation_status"].eq("rejected")
            | findings["deduplication_status"].eq(
                "confirmed_duplicate"
            )
        ].copy()
    else:
        rejected = findings.copy()
    return {
        "findings": active.reset_index(drop=True),
        "validated": validated.reset_index(drop=True),
        "pending_validation": pending.reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
        "no_finding_chunks": queue.loc[
            queue["extraction_status"].eq(
                "no_codable_finding"
            )
        ].reset_index(drop=True),
        "needs_context_chunks": queue.loc[
            queue["extraction_status"].eq(
                "needs_more_context"
            )
        ].reset_index(drop=True),
        "extraction_errors": queue.loc[
            queue["extraction_status"].eq(
                "extraction_error"
            )
        ].reset_index(drop=True),
    }


def build_source_finding_summary(
    findings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise finding counts and coded dimensions by source."""
    columns = [
        "source_id",
        "title",
        "source_type",
        "findings",
        "accepted_or_corrected",
        "evidence_streams",
        "finding_types",
        "climate_drivers",
        "biological_responses",
        "fishery_responses",
        "management_measures",
        "maximum_anchoveta_relevance",
    ]
    if findings is None or findings.empty:
        return pd.DataFrame(columns=columns)

    relevance_order = {
        "none": 0,
        "indirect_low": 1,
        "indirect_moderate": 2,
        "indirect_high": 3,
        "direct": 4,
        "unclear": -1,
    }

    rows = []
    for source_id, group in findings.groupby(
        "source_id",
        sort=False,
    ):
        first = group.iloc[0]

        def combined(field: str) -> str:
            values: list[str] = []
            if field in group:
                for value in group[field]:
                    values.extend(_split(value))
            return " | ".join(sorted(set(values)))

        relevance_values = [
            _clean(value)
            for value in group["relevance_to_anchoveta"]
        ]
        maximum = (
            max(
                relevance_values,
                key=lambda value: relevance_order.get(
                    value,
                    -2,
                ),
            )
            if relevance_values
            else "unclear"
        )
        rows.append(
            {
                "source_id": source_id,
                "title": _clean(first.get("title")),
                "source_type": _clean(
                    first.get("source_type")
                ),
                "findings": len(group),
                "accepted_or_corrected": int(
                    group["human_validation_status"]
                    .isin(["accepted", "corrected"])
                    .sum()
                ),
                "evidence_streams": combined(
                    "evidence_streams"
                ),
                "finding_types": combined("finding_type"),
                "climate_drivers": combined(
                    "climate_drivers"
                ),
                "biological_responses": combined(
                    "biological_responses"
                ),
                "fishery_responses": combined(
                    "fishery_responses"
                ),
                "management_measures": combined(
                    "management_measures"
                ),
                "maximum_anchoveta_relevance": maximum,
            }
        )
    return pd.DataFrame(rows, columns=columns)
