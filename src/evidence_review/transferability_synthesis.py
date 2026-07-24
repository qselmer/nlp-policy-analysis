"""Finding-level transferability appraisal and evidence synthesis for anchoveta."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import re
import unicodedata
from typing import Mapping

import pandas as pd
from pydantic import BaseModel, Field, model_validator
import yaml


BASE_REVIEW_COLUMNS = [
    "transferability_status",
    "target_fishery",
    "proposed_use",
    "synthesis_role",
    "total_score",
    "maximum_score",
    "normalized_score",
    "transferability_class",
    "data_requirements_peru",
    "institutional_requirements_peru",
    "adaptation_required",
    "transferability_summary",
    "critical_caveats",
    "reviewer",
    "review_date",
    "reviewer_notes",
    "human_validation_status",
]

ISSUE_COLUMNS = [
    "row_number",
    "finding_id",
    "source_id",
    "field",
    "issue",
]


class TransferabilityStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ASSESSMENT_ERROR = "assessment_error"


class HumanValidationStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class TransferabilityRecord(BaseModel):
    """Validated transferability assessment for one finding."""

    finding_id: str = Field(min_length=3)
    source_id: str = Field(min_length=3)
    transferability_status: TransferabilityStatus
    proposed_use: str
    synthesis_role: str
    component_scores: dict[str, int] = Field(default_factory=dict)
    total_score: int | None = Field(default=None, ge=0)
    maximum_score: int | None = Field(default=None, ge=0)
    normalized_score: float | None = Field(default=None, ge=0, le=1)
    transferability_class: str = "not_applicable"
    transferability_summary: str | None = None
    critical_caveats: list[str] = Field(default_factory=list)
    human_validation_status: HumanValidationStatus = (
        HumanValidationStatus.NOT_REVIEWED
    )

    @model_validator(mode="after")
    def validate_totals(self) -> "TransferabilityRecord":
        if self.transferability_status == TransferabilityStatus.COMPLETED:
            if not self.component_scores:
                raise ValueError("completed assessment requires component scores")
            calculated_total = sum(self.component_scores.values())
            if self.total_score is not None and self.total_score != calculated_total:
                raise ValueError("total_score does not match component scores")
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


def _numeric(value: object) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_robust(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read CSV files generated on UTF-8 or common Windows encodings."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return (
                pd.read_csv(
                    path,
                    encoding=encoding,
                    dtype=str,
                    keep_default_na=False,
                ),
                encoding,
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Unable to decode CSV: " + " | ".join(errors))


def load_transferability_config(
    path: str | Path = "config/transferability_synthesis.yml",
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Load and validate transferability configuration and shared taxonomy."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("transferability configuration must be a mapping")

    required = {
        "project",
        "taxonomy_path",
        "paths",
        "transferability_statuses",
        "human_validation_status",
        "transferability_classes",
        "proposed_uses",
        "synthesis_roles",
        "rating_thresholds",
        "quality_to_evidence_strength",
        "components",
        "validation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f"missing transferability configuration sections: {missing}"
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

    moderate = float(config["rating_thresholds"]["moderate_minimum"])
    high = float(config["rating_thresholds"]["high_minimum"])
    if not 0 <= moderate < high <= 1:
        raise ValueError("rating thresholds must satisfy 0 <= moderate < high <= 1")

    for component, specification in config["components"].items():
        maximum = int(specification["maximum"])
        if maximum <= 0:
            raise ValueError(f"component maximum must be positive: {component}")

    return config


def component_names(config: Mapping[str, object]) -> list[str]:
    return list(config["components"])


def score_column(component: str) -> str:
    return f"score__{component}"


def note_column(component: str) -> str:
    return f"note__{component}"


def component_columns(config: Mapping[str, object]) -> list[str]:
    columns: list[str] = []
    for component in component_names(config):
        columns.extend([score_column(component), note_column(component)])
    return columns


def review_columns(config: Mapping[str, object]) -> list[str]:
    return BASE_REVIEW_COLUMNS + component_columns(config)


def expected_evidence_strength(
    quality_rating: object,
    config: Mapping[str, object],
) -> int:
    mapping = config["quality_to_evidence_strength"]
    rating = _clean(quality_rating).casefold()
    return int(mapping.get(rating, mapping.get("unclear", 0)))


def expected_transferability_class(
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


def _maximum_score(config: Mapping[str, object]) -> int:
    return sum(
        int(specification["maximum"])
        for specification in config["components"].values()
    )


def initialise_transferability_sheet(
    findings: pd.DataFrame,
    config: Mapping[str, object],
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one assessment row per validated finding and preserve human work."""
    if findings is None or findings.empty:
        return pd.DataFrame(columns=review_columns(config))
    if "finding_id" not in findings or "source_id" not in findings:
        raise ValueError("findings require finding_id and source_id")

    sheet = findings.drop_duplicates("finding_id", keep="last").copy()
    for field in review_columns(config):
        if field not in sheet:
            sheet[field] = ""

    sheet["transferability_status"] = "pending"
    sheet["target_fishery"] = config["project"]["target_fishery"]
    sheet["proposed_use"] = "unclear"
    sheet["synthesis_role"] = "unclear"
    sheet["transferability_class"] = "not_applicable"
    sheet["human_validation_status"] = "not_reviewed"

    if existing is not None and not existing.empty and "finding_id" in existing:
        previous = (
            existing.drop_duplicates("finding_id", keep="last")
            .set_index("finding_id")
        )
        for field in review_columns(config):
            if field in previous:
                mapped = sheet["finding_id"].map(previous[field])
                mask = mapped.notna()
                sheet.loc[mask, field] = mapped.loc[mask].astype(str)

    protected = sheet["human_validation_status"].isin(
        ["accepted", "corrected", "rejected"]
    )
    quality_field = (
        "quality_overall_rating"
        if "quality_overall_rating" in sheet
        else "overall_rating"
    )
    expected = sheet.get(
        quality_field,
        pd.Series("", index=sheet.index),
    ).map(lambda value: expected_evidence_strength(value, config))
    component = "evidence_strength"
    sheet.loc[~protected, score_column(component)] = (
        expected.loc[~protected].astype(int).astype(str)
    )
    sheet.loc[~protected, note_column(component)] = (
        "Deterministically inherited from validated source quality: "
        + sheet.get(
            quality_field,
            pd.Series("", index=sheet.index),
        ).loc[~protected].astype(str)
    )
    return sheet.reset_index(drop=True)


def _finding_payload(row: Mapping[str, object]) -> dict[str, object]:
    fields = [
        "finding_id",
        "source_id",
        "title",
        "source_type",
        "finding_type",
        "evidence_streams",
        "unit_locator",
        "supporting_excerpt",
        "evidence_summary",
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
        "quality_appraisal_domain",
        "quality_normalized_score",
        "quality_overall_rating",
        "quality_appraisal_summary",
        "quality_critical_limitations",
        "quality_human_validation_status",
    ]
    return {field: row.get(field, "") for field in fields}


def build_transferability_prompt(
    row: Mapping[str, object],
    config: Mapping[str, object],
) -> str:
    """Build one auditable JSON-only transferability prompt."""
    quality_rating = row.get("quality_overall_rating", "")
    inherited_strength = expected_evidence_strength(quality_rating, config)
    components = {
        component: {
            "maximum": int(specification["maximum"]),
            "description": specification["description"],
        }
        for component, specification in config["components"].items()
    }
    payload = {
        "task": (
            "Assess how this validated climate-fisheries finding can be used for "
            "north-central Peruvian anchoveta."
        ),
        "rules": [
            "Use only the supplied finding, validated source quality, and target profile.",
            "Transferability is not the same as source quality, relevance, implementation, or effectiveness.",
            "A recommendation or proposal is not evidence of implementation.",
            "A project activity or output is not an evaluated outcome or impact.",
            "A projection or scenario is not an observed trend.",
            "Do not infer effectiveness when implementation_stage lacks evaluated outcome or impact.",
            "Score each component independently within its stated maximum.",
            (
                "evidence_strength is locked to validated source quality and must "
                f"equal {inherited_strength}."
            ),
            "Use unclear or insufficient_context instead of inventing missing information.",
            "Return one valid JSON object only.",
        ],
        "target_profile": config["project"]["target_profile"],
        "target_fishery": config["project"]["target_fishery"],
        "allowed_statuses": config["transferability_statuses"],
        "allowed_proposed_uses": config["proposed_uses"],
        "allowed_synthesis_roles": config["synthesis_roles"],
        "components": components,
        "inherited_evidence_strength": inherited_strength,
        "required_response_shape": {
            "finding_id": "string",
            "source_id": "string",
            "transferability_status": (
                "completed | insufficient_context | assessment_error"
            ),
            "proposed_use": "controlled value",
            "synthesis_role": "controlled value",
            "scores": {
                "component_name": {
                    "score": "integer within component maximum",
                    "note": "concise finding-specific justification",
                }
            },
            "data_requirements_peru": [],
            "institutional_requirements_peru": [],
            "adaptation_required": [],
            "transferability_summary": "finding-level synthesis",
            "critical_caveats": [],
            "reviewer_notes": "",
        },
        "finding": _finding_payload(row),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_transferability_prompts_jsonl(
    sheet: pd.DataFrame,
    config: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Export one prompt per finding without calling an external model."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in sheet.to_dict(orient="records"):
            stream.write(
                json.dumps(
                    {
                        "finding_id": _clean(row.get("finding_id")),
                        "source_id": _clean(row.get("source_id")),
                        "prompt": build_transferability_prompt(row, config),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def _response_payload(payload: object, line_number: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"JSONL line {line_number} is not an object")
    if isinstance(payload.get("result"), dict):
        return payload["result"]
    if isinstance(payload.get("response"), str):
        nested = json.loads(payload["response"])
        if not isinstance(nested, dict):
            raise ValueError(f"response on line {line_number} is not an object")
        return nested
    return payload


def _flatten_response(
    payload: Mapping[str, object],
    source_row: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    row = dict(source_row)
    for field in review_columns(config):
        row[field] = ""

    status = _clean(payload.get("transferability_status")) or "assessment_error"
    row["transferability_status"] = status
    row["target_fishery"] = config["project"]["target_fishery"]
    row["proposed_use"] = _clean(payload.get("proposed_use")) or "unclear"
    row["synthesis_role"] = _clean(payload.get("synthesis_role")) or "unclear"
    row["data_requirements_peru"] = _serialise_list(
        payload.get("data_requirements_peru", [])
    )
    row["institutional_requirements_peru"] = _serialise_list(
        payload.get("institutional_requirements_peru", [])
    )
    row["adaptation_required"] = _serialise_list(
        payload.get("adaptation_required", [])
    )
    row["transferability_summary"] = _clean(
        payload.get("transferability_summary")
    )
    row["critical_caveats"] = _serialise_list(
        payload.get("critical_caveats", [])
    )
    row["reviewer"] = (
        _clean(payload.get("reviewer"))
        or "AI-assisted preliminary transferability assessment"
    )
    row["review_date"] = _clean(payload.get("review_date"))
    row["reviewer_notes"] = _clean(payload.get("reviewer_notes"))
    row["human_validation_status"] = (
        _clean(payload.get("human_validation_status")) or "not_reviewed"
    )

    scores_payload = payload.get("scores", {})
    if not isinstance(scores_payload, dict):
        raise ValueError("scores response must be an object")

    scores: dict[str, int] = {}
    for component in component_names(config):
        item = scores_payload.get(component, {})
        if not isinstance(item, dict):
            item = {}
        if component == "evidence_strength":
            score = expected_evidence_strength(
                source_row.get("quality_overall_rating"),
                config,
            )
            note = (
                "Deterministically inherited from validated source quality: "
                + _clean(source_row.get("quality_overall_rating"))
            )
        else:
            score = item.get("score")
            note = _clean(item.get("note"))
        if score not in ("", None):
            score_value = int(score)
            row[score_column(component)] = str(score_value)
            scores[component] = score_value
        row[note_column(component)] = note

    if status == "completed" and scores:
        total = sum(scores.values())
        maximum = _maximum_score(config)
        normalized = total / maximum if maximum else 0.0
        row["total_score"] = str(total)
        row["maximum_score"] = str(maximum)
        row["normalized_score"] = f"{normalized:.6f}"
        row["transferability_class"] = expected_transferability_class(
            normalized,
            config,
        )
    else:
        row["transferability_class"] = "not_applicable"

    return row


def read_transferability_responses_jsonl(
    path: str | Path,
    sheet: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Read one model- or reviewer-produced assessment per finding."""
    lookup = {
        _clean(row["finding_id"]): row
        for row in sheet.to_dict(orient="records")
    }
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = _response_payload(json.loads(line), line_number)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc}"
                ) from exc
            finding_id = _clean(payload.get("finding_id"))
            if finding_id not in lookup:
                raise ValueError(
                    f"unknown finding_id on line {line_number}: {finding_id}"
                )
            source_id = _clean(payload.get("source_id"))
            expected_source = _clean(lookup[finding_id].get("source_id"))
            if source_id and source_id != expected_source:
                raise ValueError(
                    f"source_id mismatch on line {line_number}: {source_id}"
                )
            records.append(
                _flatten_response(payload, lookup[finding_id], config)
            )
    return pd.DataFrame(records)


def merge_transferability_assessments(
    sheet: pd.DataFrame,
    incoming: pd.DataFrame,
    config: Mapping[str, object],
    *,
    overwrite_human_validated: bool = False,
) -> pd.DataFrame:
    """Merge assessments while preserving accepted, corrected, and rejected rows."""
    if incoming is None or incoming.empty:
        return sheet.copy()
    if "finding_id" not in incoming:
        raise ValueError("incoming assessments require finding_id")

    output = sheet.copy()
    indexed = incoming.drop_duplicates("finding_id", keep="last").set_index(
        "finding_id"
    )
    protected = {"accepted", "corrected", "rejected"}
    fields = review_columns(config)

    for index, row in output.iterrows():
        finding_id = _clean(row.get("finding_id"))
        if finding_id not in indexed.index:
            continue
        if (
            not overwrite_human_validated
            and _clean(row.get("human_validation_status")) in protected
        ):
            continue
        for field in fields:
            if field in indexed:
                value = indexed.at[finding_id, field]
                if isinstance(value, pd.Series):
                    value = value.iloc[-1]
                output.at[index, field] = _clean(value)
    return output.reset_index(drop=True)


def validate_transferability_sheet(
    sheet: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate controlled values, scoring, distinctions, and calculations."""
    issues: list[dict[str, object]] = []
    required_ids = {"finding_id", "source_id"}
    if not required_ids <= set(sheet.columns):
        for field in sorted(required_ids - set(sheet.columns)):
            issues.append(
                {
                    "row_number": 1,
                    "finding_id": "",
                    "source_id": "",
                    "field": field,
                    "issue": "missing_column",
                }
            )
        return pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    duplicated = sheet["finding_id"].astype(str).duplicated(keep=False)
    for index, row in sheet.loc[duplicated].iterrows():
        issues.append(
            {
                "row_number": index + 2,
                "finding_id": _clean(row.get("finding_id")),
                "source_id": _clean(row.get("source_id")),
                "field": "finding_id",
                "issue": "duplicate_finding_id",
            }
        )

    valid_statuses = set(config["transferability_statuses"])
    valid_human = set(config["human_validation_status"])
    valid_classes = set(config["transferability_classes"])
    valid_uses = set(config["proposed_uses"])
    valid_roles = set(config["synthesis_roles"])
    validation = config["validation"]

    for index, row in sheet.fillna("").iterrows():
        finding_id = _clean(row.get("finding_id"))
        source_id = _clean(row.get("source_id"))
        status = _clean(row.get("transferability_status"))
        human = _clean(row.get("human_validation_status")) or "not_reviewed"
        proposed_use = _clean(row.get("proposed_use"))
        role = _clean(row.get("synthesis_role"))
        base = {
            "row_number": index + 2,
            "finding_id": finding_id,
            "source_id": source_id,
        }

        controlled = [
            ("transferability_status", status, valid_statuses),
            ("human_validation_status", human, valid_human),
            ("proposed_use", proposed_use, valid_uses),
            ("synthesis_role", role, valid_roles),
        ]
        for field, value, allowed in controlled:
            if value not in allowed:
                issues.append(
                    {
                        **base,
                        "field": field,
                        "issue": f"unknown_value:{value}",
                    }
                )

        if status == "pending":
            continue

        scores: dict[str, int] = {}
        for component, specification in config["components"].items():
            score_text = _clean(row.get(score_column(component)))
            note = _clean(row.get(note_column(component)))
            if status == "completed" and not score_text:
                issues.append(
                    {
                        **base,
                        "field": score_column(component),
                        "issue": "completed_assessment_requires_score",
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
                        "field": score_column(component),
                        "issue": f"score_not_integer:{score_text}",
                    }
                )
                continue
            maximum = int(specification["maximum"])
            if score < 0 or score > maximum:
                issues.append(
                    {
                        **base,
                        "field": score_column(component),
                        "issue": f"score_out_of_range:{score}",
                    }
                )
                continue
            scores[component] = score
            if (
                validation.get("require_note_for_every_component", True)
                and len(note) < int(validation["minimum_note_characters"])
            ):
                issues.append(
                    {
                        **base,
                        "field": note_column(component),
                        "issue": "component_note_missing_or_too_short",
                    }
                )

        expected_strength = expected_evidence_strength(
            row.get("quality_overall_rating"),
            config,
        )
        if scores.get("evidence_strength") != expected_strength:
            issues.append(
                {
                    **base,
                    "field": score_column("evidence_strength"),
                    "issue": (
                        f"evidence_strength_{scores.get('evidence_strength')}_"
                        f"differs_from_quality_{expected_strength}"
                    ),
                }
            )

        if status != "completed":
            classification = _clean(row.get("transferability_class"))
            if classification not in {"", "not_applicable"}:
                issues.append(
                    {
                        **base,
                        "field": "transferability_class",
                        "issue": "non_completed_must_be_not_applicable",
                    }
                )
            continue

        summary = _clean(row.get("transferability_summary"))
        if len(summary) < int(validation["minimum_summary_characters"]):
            issues.append(
                {
                    **base,
                    "field": "transferability_summary",
                    "issue": "summary_missing_or_too_short",
                }
            )

        total = sum(scores.values())
        maximum = _maximum_score(config)
        normalized = total / maximum if maximum else 0.0
        recorded_total = _numeric(row.get("total_score"))
        recorded_maximum = _numeric(row.get("maximum_score"))
        recorded_normalized = _numeric(row.get("normalized_score"))
        classification = _clean(row.get("transferability_class"))
        expected_class = expected_transferability_class(normalized, config)

        if recorded_total != total:
            issues.append(
                {
                    **base,
                    "field": "total_score",
                    "issue": f"recorded_{recorded_total}_calculated_{total}",
                }
            )
        if recorded_maximum != maximum:
            issues.append(
                {
                    **base,
                    "field": "maximum_score",
                    "issue": f"recorded_{recorded_maximum}_calculated_{maximum}",
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
                        f"recorded_{recorded_normalized}_"
                        f"calculated_{normalized:.6f}"
                    ),
                }
            )
        if classification not in valid_classes:
            issues.append(
                {
                    **base,
                    "field": "transferability_class",
                    "issue": f"unknown_class:{classification}",
                }
            )
        elif classification != expected_class:
            issues.append(
                {
                    **base,
                    "field": "transferability_class",
                    "issue": (
                        f"class_{classification}_differs_from_{expected_class}"
                    ),
                }
            )

        if classification == "high":
            for component, minimum in validation[
                "high_transferability_minimums"
            ].items():
                if scores.get(component, -1) < int(minimum):
                    issues.append(
                        {
                            **base,
                            "field": score_column(component),
                            "issue": (
                                "high_transferability_below_minimum:"
                                f"{minimum}"
                            ),
                        }
                    )

        if proposed_use == "direct_operational_input":
            for component, minimum in validation[
                "direct_operational_input_minimums"
            ].items():
                if scores.get(component, -1) < int(minimum):
                    issues.append(
                        {
                            **base,
                            "field": score_column(component),
                            "issue": (
                                "direct_operational_input_below_minimum:"
                                f"{minimum}"
                            ),
                        }
                    )

        if proposed_use == "not_transferable" and classification != "low":
            issues.append(
                {
                    **base,
                    "field": "proposed_use",
                    "issue": "not_transferable_requires_low_class",
                }
            )

        try:
            TransferabilityRecord(
                finding_id=finding_id,
                source_id=source_id,
                transferability_status=status,
                proposed_use=proposed_use,
                synthesis_role=role,
                component_scores=scores,
                total_score=int(recorded_total)
                if recorded_total is not None
                else None,
                maximum_score=int(recorded_maximum)
                if recorded_maximum is not None
                else None,
                normalized_score=recorded_normalized,
                transferability_class=classification,
                transferability_summary=summary,
                critical_caveats=_split(row.get("critical_caveats")),
                human_validation_status=human,
            )
        except (ValueError, TypeError) as exc:
            issues.append(
                {
                    **base,
                    "field": "transferability_logic",
                    "issue": str(exc).replace("\n", " "),
                }
            )

    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def split_transferability_outputs(
    sheet: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split active, validated, pending, and rejected assessments."""
    rejected = sheet.loc[
        sheet["human_validation_status"].eq("rejected")
    ].copy()
    active = sheet.loc[
        ~sheet["human_validation_status"].eq("rejected")
    ].copy()
    validated = active.loc[
        active["transferability_status"].eq("completed")
        & active["human_validation_status"].isin(["accepted", "corrected"])
    ].copy()
    pending = active.loc[
        active["human_validation_status"].eq("not_reviewed")
        | ~active["transferability_status"].eq("completed")
    ].copy()
    return {
        "transferability": active.reset_index(drop=True),
        "validated": validated.reset_index(drop=True),
        "pending_validation": pending.reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
    }


def _theme(row: Mapping[str, object]) -> str:
    finding_type = _clean(row.get("finding_type"))
    if finding_type == "evidence_gap":
        return "evidence_gap"
    if finding_type in {
        "project_activity",
        "project_output",
        "project_outcome",
        "implementation_barrier",
        "implementation_enabler",
    }:
        return "implementation"
    if _clean(row.get("management_measures")) or finding_type in {
        "management_measure",
        "recommendation",
        "legal_mechanism",
        "policy_commitment",
    }:
        return "management_and_governance"
    if _clean(row.get("fishery_responses")):
        return "fishery_consequence"
    if _clean(row.get("biological_responses")):
        return "ecological_response"
    if _clean(row.get("climate_drivers")):
        return "climate_hazard"
    if finding_type in {
        "method_or_indicator",
        "data_resource",
        "conceptual_framework",
    }:
        return "methods_and_data"
    return "other"


def build_evidence_synthesis_matrix(
    validated: pd.DataFrame,
) -> pd.DataFrame:
    """Build a finding-level pressure-response-management synthesis matrix."""
    if validated is None or validated.empty:
        return pd.DataFrame()
    output = validated.copy()
    output["synthesis_theme"] = output.apply(
        lambda row: _theme(row),
        axis=1,
    )
    chain_fields = [
        "climate_drivers",
        "biological_responses",
        "fishery_responses",
        "management_measures",
    ]
    for field in chain_fields:
        if field not in output:
            output[field] = ""
        output[f"has_{field}"] = output[field].map(
            lambda value: bool(_clean(value))
        )
    output["mechanism_chain_stage_count"] = output[
        [f"has_{field}" for field in chain_fields]
    ].sum(axis=1)
    output["implementation_evidence_level"] = output.get(
        "implementation_stage",
        pd.Series("", index=output.index),
    ).map(
        lambda value: (
            "evaluated"
            if _clean(value).startswith("evaluated_")
            else (
                "implemented_or_piloted"
                if _clean(value) in {"implemented", "piloted"}
                else (
                    "proposed_or_planned"
                    if _clean(value)
                    in {"proposed", "planned", "authorized"}
                    else "not_applicable_or_unclear"
                )
            )
        )
    )
    return output.reset_index(drop=True)


def build_synthesis_theme_summary(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise themes without treating counts as effect sizes."""
    columns = [
        "synthesis_theme",
        "findings",
        "sources",
        "high_transferability",
        "moderate_transferability",
        "low_transferability",
        "high_quality",
        "evaluated_implementation",
    ]
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for theme, group in matrix.groupby("synthesis_theme", sort=True):
        records.append(
            {
                "synthesis_theme": theme,
                "findings": len(group),
                "sources": group["source_id"].nunique(),
                "high_transferability": int(
                    group["transferability_class"].eq("high").sum()
                ),
                "moderate_transferability": int(
                    group["transferability_class"].eq("moderate").sum()
                ),
                "low_transferability": int(
                    group["transferability_class"].eq("low").sum()
                ),
                "high_quality": int(
                    group.get(
                        "quality_overall_rating",
                        pd.Series("", index=group.index),
                    ).eq("high").sum()
                ),
                "evaluated_implementation": int(
                    group["implementation_evidence_level"].eq("evaluated").sum()
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_management_option_summary(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise management options while keeping implementation distinct."""
    columns = [
        "management_measure",
        "findings",
        "sources",
        "high_transferability",
        "moderate_transferability",
        "evaluated_implementation",
        "implemented_or_piloted",
        "recommendation_or_proposal_only",
    ]
    if matrix is None or matrix.empty or "management_measures" not in matrix:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    expanded = matrix.copy()
    expanded["_management_measure"] = expanded["management_measures"].map(_split)
    expanded = expanded.explode("_management_measure")
    expanded["_management_measure"] = expanded["_management_measure"].fillna("")
    expanded = expanded.loc[expanded["_management_measure"].ne("")]
    for measure, group in expanded.groupby("_management_measure", sort=True):
        stage = group.get(
            "implementation_evidence_level",
            pd.Series("", index=group.index),
        )
        records.append(
            {
                "management_measure": measure,
                "findings": len(group),
                "sources": group["source_id"].nunique(),
                "high_transferability": int(
                    group["transferability_class"].eq("high").sum()
                ),
                "moderate_transferability": int(
                    group["transferability_class"].eq("moderate").sum()
                ),
                "evaluated_implementation": int(stage.eq("evaluated").sum()),
                "implemented_or_piloted": int(
                    stage.eq("implemented_or_piloted").sum()
                ),
                "recommendation_or_proposal_only": int(
                    stage.eq("proposed_or_planned").sum()
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_evidence_gap_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Create auditable gap flags; absence is not interpreted as negative effect."""
    columns = [
        "finding_id",
        "source_id",
        "title",
        "synthesis_theme",
        "gap_type",
        "gap_description",
        "transferability_class",
        "quality_overall_rating",
        "proposed_use",
    ]
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    for row in matrix.to_dict(orient="records"):
        gaps: list[tuple[str, str]] = []
        if _clean(row.get("finding_type")) == "evidence_gap":
            gaps.append(
                (
                    "explicit_source_gap",
                    "The source explicitly identifies an evidence gap.",
                )
            )
        if not _clean(row.get("climate_drivers")):
            gaps.append(
                (
                    "climate_link_missing",
                    "No explicit climate or environmental driver is encoded.",
                )
            )
        if (
            _clean(row.get("management_measures"))
            and _clean(row.get("implementation_evidence_level"))
            in {"proposed_or_planned", "not_applicable_or_unclear"}
        ):
            gaps.append(
                (
                    "implementation_evidence_missing",
                    (
                        "A management option is present without evaluated "
                        "implementation evidence."
                    ),
                )
            )
        if _clean(row.get("score__data_feasibility_peru")) in {"", "0"}:
            gaps.append(
                (
                    "peru_data_feasibility_low",
                    "Required data are absent or not demonstrated for Peru.",
                )
            )
        if _clean(row.get("transferability_class")) == "low":
            gaps.append(
                (
                    "low_transferability",
                    "The finding has low overall transferability.",
                )
            )
        for gap_type, description in gaps:
            records.append(
                {
                    "finding_id": _clean(row.get("finding_id")),
                    "source_id": _clean(row.get("source_id")),
                    "title": _clean(row.get("title")),
                    "synthesis_theme": _clean(row.get("synthesis_theme")),
                    "gap_type": gap_type,
                    "gap_description": description,
                    "transferability_class": _clean(
                        row.get("transferability_class")
                    ),
                    "quality_overall_rating": _clean(
                        row.get("quality_overall_rating")
                    ),
                    "proposed_use": _clean(row.get("proposed_use")),
                }
            )
    return pd.DataFrame(records, columns=columns)


def build_recommendation_candidates(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Flag evidence for expert review without auto-generating recommendations."""
    columns = [
        "finding_id",
        "source_id",
        "title",
        "evidence_summary",
        "supporting_excerpt",
        "management_measures",
        "implementation_stage",
        "implementation_evidence_level",
        "proposed_use",
        "transferability_class",
        "quality_overall_rating",
        "recommendation_readiness",
        "transferability_summary",
        "critical_caveats",
    ]
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=columns)
    eligible = matrix.loc[
        matrix["transferability_class"].isin(["high", "moderate"])
        & matrix["proposed_use"].isin(
            [
                "direct_operational_input",
                "management_option",
                "monitoring_indicator",
                "scenario_or_operating_model",
                "research_design",
            ]
        )
    ].copy()
    eligible["recommendation_readiness"] = eligible.apply(
        lambda row: (
            "candidate_with_evaluated_implementation"
            if _clean(row.get("implementation_evidence_level")) == "evaluated"
            and _clean(row.get("quality_overall_rating")) == "high"
            else (
                "candidate_for_expert_review"
                if _clean(row.get("transferability_class")) == "high"
                else "supporting_candidate"
            )
        ),
        axis=1,
    )
    for column in columns:
        if column not in eligible:
            eligible[column] = ""
    return eligible[columns].reset_index(drop=True)


def transferability_summary(sheet: pd.DataFrame) -> pd.DataFrame:
    """Summarise completion, validation, classes, uses, and roles."""
    status = sheet.get(
        "transferability_status",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    human = sheet.get(
        "human_validation_status",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    classification = sheet.get(
        "transferability_class",
        pd.Series("", index=sheet.index),
    ).fillna("").astype(str).str.strip()
    rows = [
        ("findings_total", len(sheet)),
        ("assessments_pending", int(status.eq("pending").sum())),
        ("assessments_completed", int(status.eq("completed").sum())),
        (
            "assessments_insufficient_context",
            int(status.eq("insufficient_context").sum()),
        ),
        ("assessment_errors", int(status.eq("assessment_error").sum())),
        ("assessments_not_reviewed", int(human.eq("not_reviewed").sum())),
        ("assessments_accepted", int(human.eq("accepted").sum())),
        ("assessments_corrected", int(human.eq("corrected").sum())),
        ("assessments_rejected", int(human.eq("rejected").sum())),
        ("transferability_low", int(classification.eq("low").sum())),
        (
            "transferability_moderate",
            int(classification.eq("moderate").sum()),
        ),
        ("transferability_high", int(classification.eq("high").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
