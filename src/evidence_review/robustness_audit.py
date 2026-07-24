"""Phase 12 robustness audit for anchoveta decision-support candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from .decision_support_core import load_decision_support_config, read_csv_robust
from .sensitivity_scenarios import (
    build_sensitivity_scenario_results,
    readiness_rank,
    scenario_retention_summary,
)
from .source_influence import (
    build_evidence_independence_summary,
    build_source_concentration_summary,
    build_source_influence_matrix,
    leave_one_source_out_summary,
)

ISSUE_COLUMNS = ["output", "row_number", "record_id", "field", "issue"]


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(_clean(value)) if _clean(value) else default
    except ValueError:
        return default


def _integer(value: object, default: int = 0) -> int:
    return int(round(_numeric(value, float(default))))


def load_robustness_config(
    path: str | Path = "config/robustness_audit.yml",
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Load robustness and nested decision-support configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("robustness configuration must be a mapping")
    required = {
        "project",
        "decision_support_config_path",
        "paths",
        "readiness_order",
        "stability_classes",
        "scenarios",
        "retention",
        "stability_thresholds",
        "source_concentration",
        "validation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing robustness configuration sections: {missing}")
    if "base" not in config["scenarios"]:
        raise ValueError("robustness configuration requires a base scenario")
    root = Path(project_root) if project_root else config_path.parent.parent
    decision_path = Path(config["decision_support_config_path"])
    if not decision_path.is_absolute():
        decision_path = root / decision_path
    config["decision_support_config"] = load_decision_support_config(
        decision_path,
        project_root=root,
    )
    return config


def _stability_class(
    row: Mapping[str, object],
    config: Mapping[str, object],
) -> str:
    thresholds = config["stability_thresholds"]
    scenario = _numeric(row.get("scenario_retention_rate"))
    loso = _numeric(row.get("leave_one_source_out_retention"))
    top_share = _numeric(row.get("top_source_share"))
    sources = _integer(row.get("sources"))
    base_rank = _integer(row.get("base_readiness_rank"), -1)

    insufficient = thresholds["insufficiently_supported"]
    if (
        sources <= int(insufficient["maximum_sources"])
        or base_rank <= int(insufficient["maximum_base_readiness_rank"])
    ):
        return "insufficiently_supported"

    dependent = thresholds["source_dependent"]
    if (
        loso <= float(dependent["maximum_leave_one_source_out_retention"])
        or top_share >= float(dependent["minimum_top_source_share"])
    ):
        return "source_dependent"

    robust = thresholds["robust"]
    if (
        scenario >= float(robust["minimum_scenario_retention"])
        and loso >= float(robust["minimum_leave_one_source_out_retention"])
        and top_share <= float(robust["maximum_top_source_share"])
        and sources >= int(robust["minimum_sources"])
    ):
        return "robust"

    stable = thresholds["generally_stable"]
    if (
        scenario >= float(stable["minimum_scenario_retention"])
        and loso >= float(stable["minimum_leave_one_source_out_retention"])
        and top_share <= float(stable["maximum_top_source_share"])
        and sources >= int(stable["minimum_sources"])
    ):
        return "generally_stable"
    return "assumption_sensitive"


def build_recommendation_stability(
    validated_recommendations: pd.DataFrame,
    portfolio: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    loso_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Combine scenario, source-removal, and concentration evidence per recommendation."""
    columns = [
        "recommendation_id",
        "option_id",
        "management_measure",
        "candidate_statement",
        "expert_review_status",
        "base_readiness_class",
        "base_readiness_rank",
        "base_decision_pathway",
        "findings",
        "sources",
        "scenario_retention_rate",
        "leave_one_source_out_retention",
        "minimum_readiness_class",
        "maximum_readiness_class",
        "readiness_class_changes",
        "decision_pathway_changes",
        "top_source_id",
        "top_source_share",
        "source_hhi",
        "effective_source_number",
        "source_concentration_class",
        "evaluated_implementation_findings",
        "implementation_source_count",
        "implementation_dependency",
        "stability_class",
        "robustness_interpretation",
    ]
    if validated_recommendations is None or validated_recommendations.empty:
        return pd.DataFrame(columns=columns)

    base = portfolio.drop_duplicates("option_id").set_index("option_id")
    scenario = scenario_summary.drop_duplicates("option_id").set_index("option_id")
    loso = loso_summary.drop_duplicates("option_id").set_index("option_id")
    concentration_lookup = concentration.drop_duplicates("option_id").set_index("option_id")
    records: list[dict[str, object]] = []

    for recommendation in validated_recommendations.fillna("").to_dict("records"):
        option_id = _clean(recommendation.get("option_id"))
        base_row = base.loc[option_id] if option_id in base.index else {}
        scenario_row = scenario.loc[option_id] if option_id in scenario.index else {}
        loso_row = loso.loc[option_id] if option_id in loso.index else {}
        concentration_row = (
            concentration_lookup.loc[option_id]
            if option_id in concentration_lookup.index
            else {}
        )
        for name, value in (
            ("base_row", base_row),
            ("scenario_row", scenario_row),
            ("loso_row", loso_row),
            ("concentration_row", concentration_row),
        ):
            if isinstance(value, pd.DataFrame):
                locals()[name] = value.iloc[-1]
        base_readiness = _clean(base_row.get("readiness_class"))
        record = {
            "recommendation_id": _clean(recommendation.get("recommendation_id")),
            "option_id": option_id,
            "management_measure": _clean(recommendation.get("management_measure")),
            "candidate_statement": _clean(recommendation.get("candidate_statement")),
            "expert_review_status": _clean(recommendation.get("expert_review_status")),
            "base_readiness_class": base_readiness,
            "base_readiness_rank": readiness_rank(base_readiness, config),
            "base_decision_pathway": _clean(base_row.get("decision_pathway")),
            "findings": _integer(base_row.get("findings")),
            "sources": _integer(base_row.get("sources")),
            "scenario_retention_rate": _numeric(
                scenario_row.get("scenario_retention_rate")
            ),
            "leave_one_source_out_retention": _numeric(
                loso_row.get("leave_one_source_out_retention")
            ),
            "minimum_readiness_class": _clean(
                scenario_row.get("minimum_readiness_class")
            ),
            "maximum_readiness_class": _clean(
                scenario_row.get("maximum_readiness_class")
            ),
            "readiness_class_changes": _integer(
                scenario_row.get("readiness_class_changes")
            ) + _integer(loso_row.get("readiness_class_changes")),
            "decision_pathway_changes": _integer(
                scenario_row.get("decision_pathway_changes")
            ) + _integer(loso_row.get("decision_pathway_changes")),
            "top_source_id": _clean(concentration_row.get("top_source_id")),
            "top_source_share": _numeric(concentration_row.get("top_source_share")),
            "source_hhi": _numeric(concentration_row.get("source_hhi")),
            "effective_source_number": _numeric(
                concentration_row.get("effective_source_number")
            ),
            "source_concentration_class": _clean(
                concentration_row.get("source_concentration_class")
            ),
            "evaluated_implementation_findings": _integer(
                concentration_row.get("evaluated_implementation_findings")
            ),
            "implementation_source_count": _integer(
                concentration_row.get("implementation_source_count")
            ),
            "implementation_dependency": _clean(
                concentration_row.get("implementation_dependency")
            ),
        }
        stability = _stability_class(record, config)
        record["stability_class"] = stability
        record["robustness_interpretation"] = {
            "robust": "Stable across scenario and source-removal analyses; effectiveness and local performance still require evaluation.",
            "generally_stable": "Generally stable, with limited sensitivity to assumptions or source removal.",
            "assumption_sensitive": "Classification or retention changes under plausible analytical assumptions.",
            "source_dependent": "Support is materially dependent on one source or a concentrated evidence base.",
            "insufficiently_supported": "Independent-source support or base readiness is insufficient for a stable recommendation.",
        }[stability]
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def split_stability_outputs(stability: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate robust recommendations from all sensitivity flags."""
    if stability is None or stability.empty:
        empty = pd.DataFrame(columns=getattr(stability, "columns", []))
        return {"robust": empty.copy(), "sensitive": empty.copy()}
    robust = stability.loc[stability["stability_class"].eq("robust")].copy()
    sensitive = stability.loc[~stability["stability_class"].eq("robust")].copy()
    return {
        "robust": robust.reset_index(drop=True),
        "sensitive": sensitive.reset_index(drop=True),
    }


def _issue(
    output: str,
    row: int,
    record: object,
    field: str,
    issue: str,
) -> dict[str, object]:
    return {
        "output": output,
        "row_number": row,
        "record_id": _clean(record),
        "field": field,
        "issue": issue,
    }


def validate_robustness_outputs(
    scenario_results: pd.DataFrame,
    influence: pd.DataFrame,
    concentration: pd.DataFrame,
    stability: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Validate identities, rates, controlled values, and complete scenario coverage."""
    issues: list[dict[str, object]] = []
    validation = config["validation"]
    readiness_allowed = set(config["readiness_order"]) | {"absent"}
    stability_allowed = set(config["stability_classes"])

    if not scenario_results.empty:
        duplicate = scenario_results.duplicated(["scenario_id", "option_id"], False)
        for index, row in scenario_results.loc[duplicate].iterrows():
            issues.append(_issue("scenarios", index + 2, row.get("option_id"), "option_id", "duplicate_option_scenario"))
        configured = set(map(str, config["scenarios"]))
        observed = set(scenario_results["scenario_id"].astype(str))
        if bool(validation.get("require_all_configured_scenarios", True)) and observed != configured:
            issues.append(_issue("scenarios", 1, "", "scenario_id", "configured_scenarios_not_fully_represented"))
        for index, row in scenario_results.fillna("").iterrows():
            if _clean(row.get("readiness_class")) not in readiness_allowed:
                issues.append(_issue("scenarios", index + 2, row.get("option_id"), "readiness_class", "unknown_readiness_class"))

    if not influence.empty:
        duplicate = influence.duplicated(["removed_source_id", "option_id"], False)
        for index, row in influence.loc[duplicate].iterrows():
            issues.append(_issue("source_influence", index + 2, row.get("option_id"), "option_id", "duplicate_option_source"))

    for output_name, frame, fields in (
        ("concentration", concentration, ["top_source_share", "source_hhi"]),
        ("stability", stability, ["scenario_retention_rate", "leave_one_source_out_retention", "top_source_share", "source_hhi"]),
    ):
        if frame is None or frame.empty:
            continue
        for index, row in frame.fillna("").iterrows():
            for field in fields:
                value = _numeric(row.get(field), -1.0)
                if not 0.0 <= value <= 1.0:
                    issues.append(_issue(output_name, index + 2, row.get("option_id"), field, "rate_out_of_range"))

    if not stability.empty:
        duplicate = stability["recommendation_id"].astype(str).duplicated(False)
        for index, row in stability.loc[duplicate].iterrows():
            issues.append(_issue("stability", index + 2, row.get("recommendation_id"), "recommendation_id", "duplicate_recommendation_id"))
        for index, row in stability.fillna("").iterrows():
            if _clean(row.get("stability_class")) not in stability_allowed:
                issues.append(_issue("stability", index + 2, row.get("recommendation_id"), "stability_class", "unknown_stability_class"))
            if _clean(row.get("base_readiness_class")) not in set(config["readiness_order"]):
                issues.append(_issue("stability", index + 2, row.get("recommendation_id"), "base_readiness_class", "unknown_base_readiness_class"))
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def build_robustness_summary(
    matrix: pd.DataFrame,
    scenario_results: pd.DataFrame,
    influence: pd.DataFrame,
    concentration: pd.DataFrame,
    stability: pd.DataFrame,
    issues: pd.DataFrame,
) -> pd.DataFrame:
    """Return compact phase-12 flow and stability metrics."""
    rows = [
        ("evidence_findings", len(matrix)),
        ("independent_sources", matrix.get("source_id", pd.Series(dtype=str)).nunique()),
        ("configured_scenarios", scenario_results.get("scenario_id", pd.Series(dtype=str)).nunique()),
        ("scenario_option_rows", len(scenario_results)),
        ("leave_one_source_out_rows", len(influence)),
        ("source_concentration_rows", len(concentration)),
        ("recommendations_audited", len(stability)),
        ("stability_robust", int(stability.get("stability_class", pd.Series(dtype=str)).eq("robust").sum())),
        ("stability_generally_stable", int(stability.get("stability_class", pd.Series(dtype=str)).eq("generally_stable").sum())),
        ("stability_assumption_sensitive", int(stability.get("stability_class", pd.Series(dtype=str)).eq("assumption_sensitive").sum())),
        ("stability_source_dependent", int(stability.get("stability_class", pd.Series(dtype=str)).eq("source_dependent").sum())),
        ("stability_insufficiently_supported", int(stability.get("stability_class", pd.Series(dtype=str)).eq("insufficiently_supported").sum())),
        ("validation_issues", len(issues)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


__all__ = [
    "build_evidence_independence_summary",
    "build_recommendation_stability",
    "build_robustness_summary",
    "build_sensitivity_scenario_results",
    "build_source_concentration_summary",
    "build_source_influence_matrix",
    "leave_one_source_out_summary",
    "load_robustness_config",
    "read_csv_robust",
    "scenario_retention_summary",
    "split_stability_outputs",
    "validate_robustness_outputs",
]
