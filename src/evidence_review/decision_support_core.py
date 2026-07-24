"""Core helpers and evidence-chain construction for phase 11."""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd
import yaml

ISSUE_COLUMNS = ["output", "row_number", "record_id", "field", "issue"]
REVIEW_COLUMNS = [
    "expert_review_status", "expert_reviewer", "expert_review_date",
    "expert_review_notes",
]


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def split_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [clean(x) for x in value if clean(x)]
    return [x.strip() for x in re.split(r"[|;]", clean(value)) if x.strip()]


def serialise(values: Iterable[object]) -> str:
    return " | ".join(dict.fromkeys(clean(x) for x in values if clean(x)))


def numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(clean(value)) if clean(value) else default
    except ValueError:
        return default


def integer(value: object, default: int = 0) -> int:
    return int(round(numeric(value, float(default))))


def stable_id(prefix: str, *values: object) -> str:
    text = "|".join(clean(v) for v in values)
    return f"{prefix}_{sha1(text.encode('utf-8')).hexdigest()[:14]}"


def mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0


def count_equals(frame: pd.DataFrame, column: str, value: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].fillna("").astype(str).str.strip().eq(value).sum())


def read_csv_robust(path: str | Path) -> tuple[pd.DataFrame, str]:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path, encoding=encoding, dtype=str, keep_default_na=False
            ), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Unable to decode CSV: " + " | ".join(errors))


def load_decision_support_config(
    path: str | Path = "config/decision_support.yml",
    *, project_root: str | Path | None = None,
) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "project", "taxonomy_path", "paths", "chain_classes",
        "decision_pathways", "readiness_classes", "research_priority_classes",
        "expert_review_statuses", "readiness_scoring", "readiness_thresholds",
        "readiness_gates", "research_priority_scoring", "recommendation_rules",
        "validation",
    }
    if not isinstance(config, dict):
        raise ValueError("decision-support configuration must be a mapping")
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing decision-support configuration sections: {missing}")
    root = Path(project_root) if project_root else config_path.parent.parent
    taxonomy = Path(config["taxonomy_path"])
    taxonomy = taxonomy if taxonomy.is_absolute() else root / taxonomy
    with taxonomy.open("r", encoding="utf-8") as stream:
        config["taxonomy"] = yaml.safe_load(stream)
    thresholds = config["readiness_thresholds"]
    ordered = [
        float(thresholds["evidence_development_minimum"]),
        float(thresholds["pilot_ready_minimum"]),
        float(thresholds["decision_support_ready_minimum"]),
    ]
    if not 0 <= ordered[0] < ordered[1] < ordered[2] <= 1:
        raise ValueError("readiness thresholds must be ordered between 0 and 1")
    scoring = config["readiness_scoring"]
    names = [
        "evidence_coverage", "source_quality", "transferability",
        "implementation_maturity", "operational_specificity", "data_feasibility",
    ]
    if sum(int(scoring[n]["maximum"]) for n in names) != int(scoring["maximum_score"]):
        raise ValueError("readiness maximum_score does not match component maxima")
    return config


def _chain_class(row: Mapping[str, object]) -> str:
    c = bool(clean(row.get("climate_drivers")))
    b = bool(clean(row.get("biological_responses")))
    f = bool(clean(row.get("fishery_responses")))
    m = bool(clean(row.get("management_measures")))
    n = sum((c, b, f, m))
    if n == 4:
        return "complete_management_chain"
    if c and b and f:
        return "climate_ecological_fishery_chain"
    if m and n >= 2:
        return "management_link_without_full_pressure_response"
    if c and b:
        return "climate_ecological_chain"
    return "partial_chain" if n >= 2 else "uncoupled_finding"


def _decision_relevance(row: Mapping[str, object]) -> str:
    use = clean(row.get("proposed_use"))
    transfer = clean(row.get("transferability_class"))
    if use == "direct_operational_input" and transfer == "high":
        return "direct_candidate_for_expert_review"
    if use in {
        "management_option", "monitoring_indicator", "scenario_or_operating_model"
    } and transfer in {"high", "moderate"}:
        return "decision_support_candidate"
    if use == "research_design" or clean(row.get("synthesis_role")) == "gap":
        return "research_candidate"
    return "contextual"


def build_climate_response_management_chains(
    matrix: pd.DataFrame, config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    columns = [
        "chain_id", "finding_id", "source_id", "title", "evidence_summary",
        "climate_drivers", "biological_responses", "fishery_responses",
        "management_measures", "implementation_stage",
        "implementation_evidence_level", "proposed_use", "synthesis_role",
        "quality_overall_rating", "transferability_class", "chain_stage_count",
        "chain_class", "decision_relevance", "supporting_excerpt", "unit_locator",
        "critical_caveats",
    ]
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for row in matrix.fillna("").to_dict("records"):
        stages = [
            bool(clean(row.get("climate_drivers"))),
            bool(clean(row.get("biological_responses"))),
            bool(clean(row.get("fishery_responses"))),
            bool(clean(row.get("management_measures"))),
        ]
        excluded = {"chain_id", "chain_stage_count", "chain_class", "decision_relevance"}
        record = {c: clean(row.get(c)) for c in columns if c not in excluded}
        record.update({
            "chain_id": stable_id("chain", row.get("finding_id"), row.get("source_id")),
            "chain_stage_count": sum(stages),
            "chain_class": _chain_class(row),
            "decision_relevance": _decision_relevance(row),
        })
        records.append(record)
    return pd.DataFrame(records, columns=columns)
