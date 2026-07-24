"""Conditional recommendation framework, review persistence, and validation."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from .decision_support_core import (
    ISSUE_COLUMNS, REVIEW_COLUMNS, clean, count_equals, integer, numeric, stable_id,
)


def _statement(row: Mapping[str, object]) -> str:
    measure = clean(row.get("management_measure")).replace("_", " ")
    pathway = clean(row.get("decision_pathway"))
    if pathway == "operational_integration_candidate":
        return f"Subject to IMARPE–PRODUCE expert review, assess integration of {measure} into existing advice and decision procedures, retaining explicit safeguards and performance monitoring."
    if pathway == "bounded_pilot_candidate":
        return f"Design and evaluate a bounded pilot of {measure} before considering broader operational use."
    if pathway == "mse_or_scenario_candidate":
        return f"Test {measure} in a management strategy evaluation or operating-model scenario before policy adoption."
    if pathway == "monitoring_candidate":
        return f"Assess {measure} as a monitoring or early-warning candidate and define locally validated indicators, thresholds, and review rules."
    if pathway == "research_priority":
        return f"Prioritize research on {measure} to resolve evidence, data, or implementation gaps before management application."
    return f"Retain evidence on {measure} as contextual support; it is not currently an operational recommendation."


def build_anchoveta_recommendation_framework(portfolio: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    if portfolio is None or portfolio.empty:
        return pd.DataFrame()
    records = []
    for row in portfolio.fillna("").to_dict("records"):
        records.append({
            "recommendation_id": stable_id("recommendation", row.get("option_id"), row.get("decision_pathway")),
            "option_id": clean(row.get("option_id")),
            "management_measure": clean(row.get("management_measure")),
            "decision_pathway": clean(row.get("decision_pathway")),
            "candidate_statement": _statement(row),
            "readiness_class": clean(row.get("readiness_class")),
            "readiness_normalized_score": clean(row.get("readiness_normalized_score")),
            "supporting_findings": integer(row.get("findings")),
            "supporting_sources": integer(row.get("sources")),
            "high_transferability": integer(row.get("high_transferability")),
            "high_quality": integer(row.get("high_quality")),
            "evaluated_implementation": integer(row.get("evaluated_implementation")),
            "implemented_or_piloted": integer(row.get("implemented_or_piloted")),
            "proposed_or_planned": integer(row.get("proposed_or_planned")),
            "data_requirements_peru": clean(row.get("data_requirements_peru")),
            "institutional_requirements_peru": clean(row.get("institutional_requirements_peru")),
            "adaptation_required": clean(row.get("adaptation_required")),
            "critical_caveats": clean(row.get("critical_caveats")),
            "evidence_interpretation": "Evaluated implementation exists, but local effectiveness remains to be assessed." if integer(row.get("evaluated_implementation")) else ("Implementation or piloting exists without sufficient evaluated effectiveness evidence." if integer(row.get("implemented_or_piloted")) else "Evidence supports consideration or testing, not demonstrated effectiveness."),
            "expert_review_status": "not_reviewed", "expert_reviewer": "",
            "expert_review_date": "", "expert_review_notes": "",
        })
    return pd.DataFrame(records)


def initialise_recommendation_review(candidates: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    output = candidates.copy()
    for field in REVIEW_COLUMNS:
        if field not in output:
            output[field] = ""
    output["expert_review_status"] = output["expert_review_status"].replace("", "not_reviewed")
    if existing is not None and not existing.empty and "recommendation_id" in existing:
        previous = existing.drop_duplicates("recommendation_id", keep="last").set_index("recommendation_id")
        for field in REVIEW_COLUMNS:
            if field in previous:
                mapped = output["recommendation_id"].map(previous[field])
                output.loc[mapped.notna(), field] = mapped.loc[mapped.notna()].astype(str)
    return output.reset_index(drop=True)


def split_recommendation_review(framework: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if framework is None or framework.empty:
        empty = pd.DataFrame(columns=getattr(framework, "columns", []))
        return {k: empty.copy() for k in ("all", "validated", "pending", "rejected")}
    status = framework["expert_review_status"].fillna("").astype(str).str.strip()
    return {
        "all": framework.reset_index(drop=True),
        "validated": framework.loc[status.isin(["accepted", "corrected"])].reset_index(drop=True),
        "pending": framework.loc[status.eq("not_reviewed")].reset_index(drop=True),
        "rejected": framework.loc[status.eq("rejected")].reset_index(drop=True),
    }


def build_decision_support_summary(chains: pd.DataFrame, portfolio: pd.DataFrame, readiness: pd.DataFrame, priorities: pd.DataFrame, framework: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("chains_total", len(chains)),
        ("complete_management_chains", count_equals(chains, "chain_class", "complete_management_chain")),
        ("management_options", len(portfolio)),
        ("decision_support_ready_options", count_equals(readiness, "readiness_class", "decision_support_ready")),
        ("pilot_ready_options", count_equals(readiness, "readiness_class", "pilot_ready")),
        ("research_priorities", len(priorities)),
        ("high_research_priorities", count_equals(priorities, "priority_class", "high")),
        ("recommendation_candidates", len(framework)),
        ("recommendations_not_reviewed", count_equals(framework, "expert_review_status", "not_reviewed")),
        ("recommendations_accepted", count_equals(framework, "expert_review_status", "accepted")),
        ("recommendations_corrected", count_equals(framework, "expert_review_status", "corrected")),
        ("recommendations_rejected", count_equals(framework, "expert_review_status", "rejected")),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _issue(output: str, row: int, record: object, field: str, issue: str) -> dict[str, object]:
    return {"output": output, "row_number": row, "record_id": clean(record), "field": field, "issue": issue}


def validate_decision_support_outputs(chains: pd.DataFrame, portfolio: pd.DataFrame, readiness: pd.DataFrame, priorities: pd.DataFrame, framework: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    issues: list[dict[str, object]] = []
    if not chains.empty:
        for index, row in chains.loc[chains["chain_id"].astype(str).duplicated(False)].iterrows():
            issues.append(_issue("chains", index + 2, row.get("chain_id"), "chain_id", "duplicate_chain_id"))
        for index, row in chains.fillna("").iterrows():
            if clean(row.get("chain_class")) not in set(config["chain_classes"]):
                issues.append(_issue("chains", index + 2, row.get("chain_id"), "chain_class", "unknown_chain_class"))
            if not 0 <= integer(row.get("chain_stage_count"), -1) <= 4:
                issues.append(_issue("chains", index + 2, row.get("chain_id"), "chain_stage_count", "chain_stage_count_out_of_range"))
    if not portfolio.empty:
        maximum = int(config["readiness_scoring"]["maximum_score"])
        components = ["readiness_evidence_coverage", "readiness_source_quality", "readiness_transferability", "readiness_implementation_maturity", "readiness_operational_specificity", "readiness_data_feasibility"]
        for index, row in portfolio.fillna("").iterrows():
            record = row.get("option_id")
            if clean(row.get("decision_pathway")) not in set(config["decision_pathways"]):
                issues.append(_issue("portfolio", index + 2, record, "decision_pathway", "unknown_decision_pathway"))
            if clean(row.get("readiness_class")) not in set(config["readiness_classes"]):
                issues.append(_issue("portfolio", index + 2, record, "readiness_class", "unknown_readiness_class"))
            calculated = sum(integer(row.get(field)) for field in components)
            if integer(row.get("readiness_total_score"), -1) != calculated:
                issues.append(_issue("portfolio", index + 2, record, "readiness_total_score", "readiness_total_mismatch"))
            if integer(row.get("readiness_maximum_score"), -1) != maximum:
                issues.append(_issue("portfolio", index + 2, record, "readiness_maximum_score", "maximum_score_mismatch"))
            if abs(numeric(row.get("readiness_normalized_score"), -1) - calculated / maximum) > 1e-5:
                issues.append(_issue("portfolio", index + 2, record, "readiness_normalized_score", "normalized_score_mismatch"))
            if clean(row.get("readiness_class")) == "decision_support_ready" and integer(row.get("evaluated_implementation")) < 1:
                issues.append(_issue("portfolio", index + 2, record, "readiness_class", "decision_support_ready_requires_evaluated_implementation"))
    if not priorities.empty:
        for index, row in priorities.fillna("").iterrows():
            if clean(row.get("priority_class")) not in set(config["research_priority_classes"]):
                issues.append(_issue("research_priorities", index + 2, row.get("priority_id"), "priority_class", "unknown_priority_class"))
    if not framework.empty:
        rules = config["recommendation_rules"]
        prohibited = [x.casefold() for x in rules["prohibited_claim_phrases"]]
        conditional = [x.casefold() for x in rules["required_conditional_terms"]]
        for index, row in framework.fillna("").iterrows():
            record = row.get("recommendation_id")
            text = clean(row.get("candidate_statement")).casefold()
            if clean(row.get("expert_review_status")) not in set(config["expert_review_statuses"]):
                issues.append(_issue("recommendations", index + 2, record, "expert_review_status", "unknown_expert_review_status"))
            if any(x in text for x in prohibited):
                issues.append(_issue("recommendations", index + 2, record, "candidate_statement", "prohibited_effectiveness_or_immediacy_claim"))
            if not any(x in text for x in conditional):
                issues.append(_issue("recommendations", index + 2, record, "candidate_statement", "conditional_language_required"))
            if integer(row.get("evaluated_implementation")) == 0 and "effective" in text:
                issues.append(_issue("recommendations", index + 2, record, "candidate_statement", "effectiveness_claim_without_evaluated_implementation"))
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)
