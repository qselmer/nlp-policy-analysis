"""Sensitivity scenarios for phase 12 robustness auditing."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Mapping

import pandas as pd

from .decision_support_portfolio import build_management_option_portfolio


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _tokens(value: object) -> set[str]:
    return {
        item.strip()
        for item in re.split(r"[|;,]", _clean(value))
        if item.strip()
    }


def readiness_rank(value: object, config: Mapping[str, object]) -> int:
    order = list(config["readiness_order"])
    text = _clean(value)
    return order.index(text) if text in order else -1


def _allowed_mask(
    frame: pd.DataFrame,
    column: str,
    allowed: list[str] | tuple[str, ...],
) -> pd.Series:
    if not allowed:
        return pd.Series(True, index=frame.index)
    allowed_set = set(map(str, allowed))
    values = frame.get(column, pd.Series("", index=frame.index))
    return values.map(lambda value: bool(_tokens(value) & allowed_set))


def filter_evidence_for_scenario(
    matrix: pd.DataFrame,
    specification: Mapping[str, object],
) -> pd.DataFrame:
    """Apply only explicit scenario filters; missing fields never imply inclusion."""
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=getattr(matrix, "columns", []))
    mask = pd.Series(True, index=matrix.index)
    mask &= _allowed_mask(
        matrix,
        "quality_overall_rating",
        list(specification.get("quality_allowed", [])),
    )
    mask &= _allowed_mask(
        matrix,
        "transferability_class",
        list(specification.get("transferability_allowed", [])),
    )
    mask &= _allowed_mask(
        matrix,
        "implementation_evidence_level",
        list(specification.get("implementation_allowed", [])),
    )
    mask &= _allowed_mask(
        matrix,
        "finding_type",
        list(specification.get("finding_types_allowed", [])),
    )
    mask &= _allowed_mask(
        matrix,
        "method_or_design",
        list(specification.get("methods_allowed", [])),
    )
    return matrix.loc[mask].copy().reset_index(drop=True)


def adjusted_decision_support_config(
    decision_config: Mapping[str, object],
    threshold_shift: float,
) -> dict:
    """Shift readiness thresholds without altering component scores or gates."""
    output = deepcopy(dict(decision_config))
    thresholds = output["readiness_thresholds"]
    for key, value in list(thresholds.items()):
        thresholds[key] = min(1.0, max(0.0, float(value) + threshold_shift))
    ordered = [
        "evidence_development_minimum",
        "pilot_ready_minimum",
        "decision_support_ready_minimum",
    ]
    values = [float(thresholds[key]) for key in ordered]
    if values != sorted(values):
        raise ValueError("shifted readiness thresholds are not monotonic")
    return output


def build_sensitivity_scenario_results(
    matrix: pd.DataFrame,
    decision_config: Mapping[str, object],
    robustness_config: Mapping[str, object],
) -> pd.DataFrame:
    """Rebuild the option portfolio under every configured sensitivity scenario."""
    scenarios = robustness_config["scenarios"]
    if "base" not in scenarios:
        raise ValueError("robustness scenarios require a base scenario")

    scenario_portfolios: dict[str, pd.DataFrame] = {}
    scenario_evidence_rows: dict[str, int] = {}
    for scenario_id, specification in scenarios.items():
        filtered = filter_evidence_for_scenario(matrix, specification)
        shifted = adjusted_decision_support_config(
            decision_config,
            float(specification.get("threshold_shift", 0.0)),
        )
        portfolio = build_management_option_portfolio(filtered, shifted)
        scenario_portfolios[str(scenario_id)] = portfolio
        scenario_evidence_rows[str(scenario_id)] = len(filtered)

    base = scenario_portfolios["base"]
    if base.empty:
        return pd.DataFrame()
    base_options = base[["option_id", "management_measure"]].drop_duplicates()
    minimum = _clean(
        robustness_config["retention"].get(
            "minimum_readiness_class",
            "evidence_development",
        )
    )
    minimum_rank = readiness_rank(minimum, robustness_config)
    presence_is_retained = bool(
        robustness_config["retention"].get(
            "scenario_presence_counts_as_retained",
            True,
        )
    )

    records: list[dict[str, object]] = []
    for scenario_id, specification in scenarios.items():
        portfolio = scenario_portfolios[str(scenario_id)]
        lookup = (
            portfolio.drop_duplicates("option_id", keep="last").set_index("option_id")
            if not portfolio.empty
            else pd.DataFrame()
        )
        for option in base_options.to_dict("records"):
            option_id = _clean(option.get("option_id"))
            present = not lookup.empty and option_id in lookup.index
            row = lookup.loc[option_id] if present else {}
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            readiness = _clean(row.get("readiness_class")) if present else "absent"
            rank = readiness_rank(readiness, robustness_config) if present else -1
            meets_minimum = bool(present and rank >= minimum_rank)
            retained = bool(present if presence_is_retained else meets_minimum)
            records.append(
                {
                    "scenario_id": str(scenario_id),
                    "scenario_label": _clean(specification.get("label")),
                    "option_id": option_id,
                    "management_measure": _clean(option.get("management_measure")),
                    "evidence_rows_available": scenario_evidence_rows[str(scenario_id)],
                    "option_present": present,
                    "findings": int(float(row.get("findings", 0))) if present else 0,
                    "sources": int(float(row.get("sources", 0))) if present else 0,
                    "readiness_class": readiness,
                    "readiness_rank": rank,
                    "readiness_normalized_score": _clean(
                        row.get("readiness_normalized_score")
                    ) if present else "",
                    "decision_pathway": _clean(row.get("decision_pathway")) if present else "absent",
                    "meets_minimum_readiness": meets_minimum,
                    "retained": retained,
                }
            )
    return pd.DataFrame(records)


def scenario_retention_summary(
    scenario_results: pd.DataFrame,
    robustness_config: Mapping[str, object],
) -> pd.DataFrame:
    """Summarise scenario retention and readiness range for every base option."""
    columns = [
        "option_id",
        "management_measure",
        "configured_scenarios",
        "scenarios_present",
        "scenarios_retained",
        "scenario_retention_rate",
        "minimum_readiness_class",
        "maximum_readiness_class",
        "minimum_readiness_rank",
        "maximum_readiness_rank",
        "readiness_class_changes",
        "decision_pathway_changes",
    ]
    if scenario_results is None or scenario_results.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    count = len(robustness_config["scenarios"])
    order = list(robustness_config["readiness_order"])
    for option_id, group in scenario_results.groupby("option_id", sort=True):
        present = group.loc[group["option_present"].astype(bool)].copy()
        ranks = [int(value) for value in present.get("readiness_rank", []) if int(value) >= 0]
        min_rank = min(ranks) if ranks else -1
        max_rank = max(ranks) if ranks else -1
        records.append(
            {
                "option_id": option_id,
                "management_measure": _clean(group.iloc[0].get("management_measure")),
                "configured_scenarios": count,
                "scenarios_present": int(group["option_present"].astype(bool).sum()),
                "scenarios_retained": int(group["retained"].astype(bool).sum()),
                "scenario_retention_rate": (
                    float(group["retained"].astype(bool).sum()) / count if count else 0.0
                ),
                "minimum_readiness_class": order[min_rank] if min_rank >= 0 else "absent",
                "maximum_readiness_class": order[max_rank] if max_rank >= 0 else "absent",
                "minimum_readiness_rank": min_rank,
                "maximum_readiness_rank": max_rank,
                "readiness_class_changes": max(0, present["readiness_class"].nunique() - 1),
                "decision_pathway_changes": max(0, present["decision_pathway"].nunique() - 1),
            }
        )
    return pd.DataFrame(records, columns=columns)
