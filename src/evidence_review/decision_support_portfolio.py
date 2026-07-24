"""Management portfolio, readiness, and research-priority construction."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from .decision_support_core import (
    clean, count_equals, mean_numeric, serialise, split_values, stable_id,
)


def _readiness_scores(group: pd.DataFrame, config: Mapping[str, object]) -> dict[str, int]:
    s = config["readiness_scoring"]
    findings, sources = len(group), group["source_id"].nunique()
    coverage = s["evidence_coverage"]
    evidence = 2 if findings >= int(coverage["two_point_minimum_findings"]) and sources >= int(coverage["two_point_minimum_sources"]) else (1 if findings >= int(coverage["one_point_minimum_findings"]) else 0)
    high_quality = count_equals(group, "quality_overall_rating", "high")
    quality_share = high_quality / findings if findings else 0
    quality = 2 if quality_share >= float(s["source_quality"]["two_point_minimum_high_quality_share"]) else (1 if high_quality >= int(s["source_quality"]["one_point_minimum_high_quality"]) else 0)
    mean_transfer = mean_numeric(group, "normalized_score")
    transfer = 2 if mean_transfer >= float(s["transferability"]["two_point_minimum_mean"]) else (1 if mean_transfer >= float(s["transferability"]["one_point_minimum_mean"]) else 0)
    stage = group.get("implementation_evidence_level", pd.Series("", index=group.index)).fillna("").astype(str).str.strip()
    maturity = s["implementation_maturity"]
    implementation = int(maturity["evaluated"]) if stage.eq("evaluated").any() else (int(maturity["implemented_or_piloted"]) if stage.eq("implemented_or_piloted").any() else (int(maturity["proposed_or_planned"]) if stage.eq("proposed_or_planned").any() else 0))
    mean_specificity = mean_numeric(group, "score__operational_specificity")
    specificity = 2 if mean_specificity >= float(s["operational_specificity"]["two_point_minimum_mean"]) else (1 if mean_specificity >= float(s["operational_specificity"]["one_point_minimum_mean"]) else 0)
    mean_data = mean_numeric(group, "score__data_feasibility_peru")
    data = 2 if mean_data >= float(s["data_feasibility"]["two_point_minimum_mean"]) else (1 if mean_data >= float(s["data_feasibility"]["one_point_minimum_mean"]) else 0)
    return {
        "readiness_evidence_coverage": evidence,
        "readiness_source_quality": quality,
        "readiness_transferability": transfer,
        "readiness_implementation_maturity": implementation,
        "readiness_operational_specificity": specificity,
        "readiness_data_feasibility": data,
    }


def _readiness_class(score: float, evaluated: int, specificity: float, data: float, config: Mapping[str, object]) -> str:
    t, gates = config["readiness_thresholds"], config["readiness_gates"]
    dg = gates["decision_support_ready"]
    if score >= float(t["decision_support_ready_minimum"]) and evaluated >= int(dg["minimum_evaluated_implementation"]) and specificity >= float(dg["minimum_mean_operational_specificity"]) and data >= float(dg["minimum_mean_data_feasibility"]):
        return "decision_support_ready"
    pg = gates["pilot_ready"]
    if score >= float(t["pilot_ready_minimum"]) and specificity >= float(pg["minimum_mean_operational_specificity"]) and data >= float(pg["minimum_mean_data_feasibility"]):
        return "pilot_ready"
    return "evidence_development" if score >= float(t["evidence_development_minimum"]) else "not_ready"


def _pathway(group: pd.DataFrame, readiness: str) -> str:
    uses = set(group.get("proposed_use", pd.Series(dtype=str)).fillna("").astype(str).str.strip())
    if "direct_operational_input" in uses and readiness == "decision_support_ready":
        return "operational_integration_candidate"
    if "monitoring_indicator" in uses:
        return "monitoring_candidate"
    if "scenario_or_operating_model" in uses:
        return "mse_or_scenario_candidate"
    if readiness in {"decision_support_ready", "pilot_ready"}:
        return "bounded_pilot_candidate"
    if "research_design" in uses or readiness == "not_ready":
        return "research_priority"
    return "mse_or_scenario_candidate" if "management_option" in uses else "contextual_only"


def build_management_option_portfolio(matrix: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    if matrix is None or matrix.empty or "management_measures" not in matrix:
        return pd.DataFrame()
    expanded = matrix.copy()
    expanded["_measure"] = expanded["management_measures"].map(split_values)
    expanded = expanded.explode("_measure")
    expanded["_measure"] = expanded["_measure"].fillna("").astype(str).str.strip()
    expanded = expanded.loc[expanded["_measure"].ne("")]
    maximum = int(config["readiness_scoring"]["maximum_score"])
    records = []
    for measure, group in expanded.groupby("_measure", sort=True):
        components = _readiness_scores(group, config)
        total = sum(components.values())
        evaluated = count_equals(group, "implementation_evidence_level", "evaluated")
        specificity = mean_numeric(group, "score__operational_specificity")
        data = mean_numeric(group, "score__data_feasibility_peru")
        normalized = total / maximum if maximum else 0.0
        readiness = _readiness_class(normalized, evaluated, specificity, data, config)
        flatten = lambda column: serialise(item for value in group.get(column, []) for item in split_values(value))
        records.append({
            "option_id": stable_id("option", measure), "management_measure": measure,
            "findings": len(group), "sources": group["source_id"].nunique(),
            "finding_ids": serialise(group.get("finding_id", [])),
            "source_ids": serialise(group.get("source_id", [])),
            "proposed_uses": serialise(group.get("proposed_use", [])),
            "synthesis_roles": serialise(group.get("synthesis_role", [])),
            "high_transferability": count_equals(group, "transferability_class", "high"),
            "moderate_transferability": count_equals(group, "transferability_class", "moderate"),
            "low_transferability": count_equals(group, "transferability_class", "low"),
            "high_quality": count_equals(group, "quality_overall_rating", "high"),
            "moderate_quality": count_equals(group, "quality_overall_rating", "moderate"),
            "evaluated_implementation": evaluated,
            "implemented_or_piloted": count_equals(group, "implementation_evidence_level", "implemented_or_piloted"),
            "proposed_or_planned": count_equals(group, "implementation_evidence_level", "proposed_or_planned"),
            "mean_transferability_score": f"{mean_numeric(group, 'normalized_score'):.6f}",
            "mean_operational_specificity": f"{specificity:.6f}",
            "mean_data_feasibility_peru": f"{data:.6f}",
            "data_requirements_peru": flatten("data_requirements_peru"),
            "institutional_requirements_peru": flatten("institutional_requirements_peru"),
            "adaptation_required": flatten("adaptation_required"),
            "critical_caveats": flatten("critical_caveats"),
            **components,
            "readiness_total_score": total, "readiness_maximum_score": maximum,
            "readiness_normalized_score": f"{normalized:.6f}",
            "readiness_class": readiness, "decision_pathway": _pathway(group, readiness),
            "candidate_status": "candidate_for_expert_review",
        })
    return pd.DataFrame(records)


def build_operational_readiness_matrix(portfolio: pd.DataFrame) -> pd.DataFrame:
    if portfolio is None or portfolio.empty:
        return pd.DataFrame()
    output = portfolio.copy()
    output["readiness_gate_note"] = output["readiness_class"].map({
        "decision_support_ready": "Evaluated implementation and minimum operational/data gates are present.",
        "pilot_ready": "Evidence supports bounded testing; effectiveness is not assumed.",
        "evidence_development": "Additional evidence development is required before operational use.",
        "not_ready": "Current evidence is insufficient for operational progression.",
    })
    keep = [
        "option_id", "management_measure", "decision_pathway", "readiness_class",
        "readiness_total_score", "readiness_maximum_score", "readiness_normalized_score",
        "readiness_evidence_coverage", "readiness_source_quality",
        "readiness_transferability", "readiness_implementation_maturity",
        "readiness_operational_specificity", "readiness_data_feasibility",
        "evaluated_implementation", "implemented_or_piloted", "proposed_or_planned",
        "mean_operational_specificity", "mean_data_feasibility_peru",
        "readiness_gate_note",
    ]
    return output[keep].reset_index(drop=True)


def build_research_priority_matrix(gaps: pd.DataFrame, matrix: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    if gaps is None or gaps.empty:
        return pd.DataFrame()
    lookup = matrix.drop_duplicates("finding_id").set_index("finding_id") if matrix is not None and not matrix.empty else pd.DataFrame()
    scoring, records = config["research_priority_scoring"], []
    for (gap, theme), group in gaps.groupby(["gap_type", "synthesis_theme"], dropna=False, sort=True):
        ids = [clean(x) for x in group["finding_id"] if clean(x)]
        linked = lookup.loc[lookup.index.intersection(ids)] if not lookup.empty else pd.DataFrame()
        high_t = count_equals(linked, "transferability_class", "high")
        high_q = count_equals(linked, "quality_overall_rating", "high")
        score = int(scoring.get(clean(gap), 0))
        score += int(scoring["high_transferability_bonus"]) if high_t else 0
        score += int(scoring["high_quality_bonus"]) if high_q else 0
        priority = "high" if score >= int(scoring["high_minimum"]) else ("moderate" if score >= int(scoring["moderate_minimum"]) else "low")
        records.append({
            "priority_id": stable_id("priority", gap, theme), "gap_type": clean(gap),
            "synthesis_theme": clean(theme), "findings": len(group),
            "sources": group["source_id"].nunique(),
            "finding_ids": serialise(group.get("finding_id", [])),
            "source_ids": serialise(group.get("source_id", [])),
            "high_transferability": high_t, "high_quality": high_q,
            "priority_score": score, "priority_class": priority,
            "research_use": "targeted_validation_or_pilot" if priority == "high" else ("focused_research_design" if priority == "moderate" else "monitor_as_context"),
            "priority_statement": f"Assess the {clean(gap).replace('_', ' ')} gap for {clean(theme) or 'the evidence base'} using a design that distinguishes missing evidence from evidence of no effect.",
            "interpretation_guardrail": clean(group.iloc[0].get("gap_description")) or "Absence of evidence must not be coded as evidence of absence.",
        })
    return pd.DataFrame(records)
