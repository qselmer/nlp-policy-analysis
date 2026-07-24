"""Leave-one-source-out influence and evidence concentration diagnostics."""
from __future__ import annotations

import re
from typing import Mapping

import pandas as pd

from .decision_support_portfolio import build_management_option_portfolio
from .sensitivity_scenarios import readiness_rank


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
    return [
        item.strip()
        for item in re.split(r"[|;,]", _clean(value))
        if item.strip()
    ]


def _expanded_measure_evidence(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix is None or matrix.empty or "management_measures" not in matrix:
        return pd.DataFrame()
    expanded = matrix.copy()
    expanded["management_measure"] = expanded["management_measures"].map(_split)
    expanded = expanded.explode("management_measure")
    expanded["management_measure"] = (
        expanded["management_measure"].fillna("").astype(str).str.strip()
    )
    return expanded.loc[expanded["management_measure"].ne("")].reset_index(drop=True)


def build_source_influence_matrix(
    matrix: pd.DataFrame,
    decision_config: Mapping[str, object],
    robustness_config: Mapping[str, object],
) -> pd.DataFrame:
    """Rebuild all base options after removing each independent source."""
    baseline = build_management_option_portfolio(matrix, decision_config)
    if baseline.empty:
        return pd.DataFrame()
    sources = sorted(
        {
            _clean(value)
            for value in matrix.get("source_id", pd.Series(dtype=str))
            if _clean(value)
        }
    )
    expanded = _expanded_measure_evidence(matrix)
    support_lookup = {
        (measure, source)
        for measure, source in expanded[["management_measure", "source_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    } if not expanded.empty else set()
    presence_is_retained = bool(
        robustness_config["retention"].get(
            "leave_one_source_out_presence_counts_as_retained",
            True,
        )
    )
    minimum = _clean(
        robustness_config["retention"].get(
            "minimum_readiness_class",
            "evidence_development",
        )
    )
    minimum_rank = readiness_rank(minimum, robustness_config)
    base_lookup = baseline.drop_duplicates("option_id").set_index("option_id")
    records: list[dict[str, object]] = []

    for removed_source in sources:
        reduced = matrix.loc[
            matrix.get("source_id", pd.Series("", index=matrix.index))
            .astype(str)
            .str.strip()
            .ne(removed_source)
        ].copy()
        rebuilt = build_management_option_portfolio(reduced, decision_config)
        lookup = (
            rebuilt.drop_duplicates("option_id").set_index("option_id")
            if not rebuilt.empty
            else pd.DataFrame()
        )
        for option_id, base_row in base_lookup.iterrows():
            present = not lookup.empty and option_id in lookup.index
            row = lookup.loc[option_id] if present else {}
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            readiness = _clean(row.get("readiness_class")) if present else "absent"
            rank = readiness_rank(readiness, robustness_config) if present else -1
            meets_minimum = bool(present and rank >= minimum_rank)
            retained = bool(present if presence_is_retained else meets_minimum)
            base_readiness = _clean(base_row.get("readiness_class"))
            base_pathway = _clean(base_row.get("decision_pathway"))
            measure = _clean(base_row.get("management_measure"))
            records.append(
                {
                    "option_id": option_id,
                    "management_measure": measure,
                    "removed_source_id": removed_source,
                    "removed_source_supported_option": (measure, removed_source) in support_lookup,
                    "option_present": present,
                    "retained": retained,
                    "base_readiness_class": base_readiness,
                    "readiness_class_without_source": readiness,
                    "readiness_rank_without_source": rank,
                    "readiness_class_changed": bool(present and readiness != base_readiness),
                    "base_decision_pathway": base_pathway,
                    "decision_pathway_without_source": _clean(row.get("decision_pathway")) if present else "absent",
                    "decision_pathway_changed": bool(
                        present and _clean(row.get("decision_pathway")) != base_pathway
                    ),
                    "option_lost": not present,
                    "findings_without_source": int(float(row.get("findings", 0))) if present else 0,
                    "sources_without_source": int(float(row.get("sources", 0))) if present else 0,
                }
            )
    return pd.DataFrame(records)


def leave_one_source_out_summary(
    influence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise retention and classification changes across source removals."""
    columns = [
        "option_id",
        "management_measure",
        "sources_removed_total",
        "source_removals_retained",
        "leave_one_source_out_retention",
        "option_losses",
        "readiness_class_changes",
        "decision_pathway_changes",
        "supporting_source_removals_tested",
        "supporting_source_removal_losses",
    ]
    if influence is None or influence.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for option_id, group in influence.groupby("option_id", sort=True):
        total = len(group)
        supported = group["removed_source_supported_option"].astype(bool)
        lost = group["option_lost"].astype(bool)
        records.append(
            {
                "option_id": option_id,
                "management_measure": _clean(group.iloc[0].get("management_measure")),
                "sources_removed_total": total,
                "source_removals_retained": int(group["retained"].astype(bool).sum()),
                "leave_one_source_out_retention": (
                    float(group["retained"].astype(bool).sum()) / total if total else 0.0
                ),
                "option_losses": int(lost.sum()),
                "readiness_class_changes": int(
                    group["readiness_class_changed"].astype(bool).sum()
                ),
                "decision_pathway_changes": int(
                    group["decision_pathway_changed"].astype(bool).sum()
                ),
                "supporting_source_removals_tested": int(supported.sum()),
                "supporting_source_removal_losses": int((supported & lost).sum()),
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_source_concentration_summary(
    matrix: pd.DataFrame,
    robustness_config: Mapping[str, object],
) -> pd.DataFrame:
    """Quantify source concentration using finding shares and effective source number."""
    expanded = _expanded_measure_evidence(matrix)
    columns = [
        "option_id",
        "management_measure",
        "findings",
        "sources",
        "top_source_id",
        "top_source_findings",
        "top_source_share",
        "source_hhi",
        "effective_source_number",
        "source_concentration_class",
        "evaluated_implementation_findings",
        "implementation_source_count",
        "implementation_dependency",
    ]
    if expanded.empty:
        return pd.DataFrame(columns=columns)
    thresholds = robustness_config["source_concentration"]
    records: list[dict[str, object]] = []
    for measure, group in expanded.groupby("management_measure", sort=True):
        counts = group.groupby("source_id").size().sort_values(ascending=False)
        total = int(counts.sum())
        shares = counts / total if total else counts.astype(float)
        hhi = float((shares**2).sum()) if total else 0.0
        effective = 1.0 / hhi if hhi > 0 else 0.0
        top_id = _clean(counts.index[0]) if len(counts) else ""
        top_count = int(counts.iloc[0]) if len(counts) else 0
        top_share = float(shares.iloc[0]) if len(shares) else 0.0
        if top_share >= float(thresholds["high_concentration_threshold"]):
            concentration = "high"
        elif top_share >= float(thresholds["moderate_concentration_threshold"]):
            concentration = "moderate"
        else:
            concentration = "low"
        evaluated = group.loc[
            group.get(
                "implementation_evidence_level",
                pd.Series("", index=group.index),
            )
            .astype(str)
            .str.strip()
            .eq("evaluated")
        ]
        implementation_sources = evaluated.get(
            "source_id",
            pd.Series(dtype=str),
        ).nunique()
        dependency = (
            "no_evaluated_implementation"
            if evaluated.empty
            else (
                "single_source"
                if implementation_sources == 1
                else "multiple_sources"
            )
        )
        records.append(
            {
                "option_id": "option_" + __import__("hashlib").sha1(measure.encode("utf-8")).hexdigest()[:12],
                "management_measure": measure,
                "findings": len(group),
                "sources": int(counts.size),
                "top_source_id": top_id,
                "top_source_findings": top_count,
                "top_source_share": top_share,
                "source_hhi": hhi,
                "effective_source_number": effective,
                "source_concentration_class": concentration,
                "evaluated_implementation_findings": len(evaluated),
                "implementation_source_count": int(implementation_sources),
                "implementation_dependency": dependency,
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_evidence_independence_summary(
    portfolio: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    """Keep finding counts distinct from independent-source support."""
    columns = [
        "option_id",
        "management_measure",
        "findings",
        "independent_sources",
        "findings_per_source",
        "effective_source_number",
        "top_source_share",
        "evaluated_implementation_findings",
        "implementation_source_count",
        "implementation_dependency",
        "independence_flag",
    ]
    if portfolio is None or portfolio.empty:
        return pd.DataFrame(columns=columns)
    merged = portfolio.merge(
        concentration,
        on=["option_id", "management_measure"],
        how="left",
        suffixes=("", "_concentration"),
    )
    records: list[dict[str, object]] = []
    for row in merged.fillna("").to_dict("records"):
        findings = int(float(row.get("findings", 0) or 0))
        sources = int(float(row.get("sources", 0) or 0))
        top_share = float(row.get("top_source_share", 0) or 0)
        flag = (
            "single_source"
            if sources <= 1
            else (
                "concentrated"
                if top_share >= 0.75
                else "multi_source"
            )
        )
        records.append(
            {
                "option_id": _clean(row.get("option_id")),
                "management_measure": _clean(row.get("management_measure")),
                "findings": findings,
                "independent_sources": sources,
                "findings_per_source": findings / sources if sources else 0.0,
                "effective_source_number": float(row.get("effective_source_number", 0) or 0),
                "top_source_share": top_share,
                "evaluated_implementation_findings": int(float(row.get("evaluated_implementation_findings", 0) or 0)),
                "implementation_source_count": int(float(row.get("implementation_source_count", 0) or 0)),
                "implementation_dependency": _clean(row.get("implementation_dependency")),
                "independence_flag": flag,
            }
        )
    return pd.DataFrame(records, columns=columns)
