"""Phase 13 scientific tables, final recommendation framing, and validation."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd
import yaml

from .decision_support_core import read_csv_robust

ISSUE_COLUMNS = ["output", "row_number", "record_id", "field", "issue"]


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def numeric(value: object, default: float = 0.0) -> float:
    text = clean(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def integer(value: object, default: int = 0) -> int:
    return int(round(numeric(value, float(default))))


def split_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [clean(item) for item in value if clean(item)]
    return [
        item.strip()
        for item in re.split(r"[|;]", clean(value))
        if item.strip()
    ]


def load_publication_config(
    path: str | Path = "config/publication_products.yml",
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Load phase 13 configuration and its phase 11/12 dependencies."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("publication configuration must be a mapping")
    required = {
        "project",
        "decision_support_config_path",
        "robustness_config_path",
        "paths",
        "publication_tiers",
        "stability_to_publication_tier",
        "publication_tier_order",
        "publication_tier_labels",
        "recommendation_next_steps",
        "figure_settings",
        "report_settings",
        "controlled_values",
        "language_guardrails",
        "validation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing publication configuration sections: {missing}")
    mapping_values = set(config["stability_to_publication_tier"].values())
    tiers = set(config["publication_tiers"])
    if mapping_values - tiers:
        raise ValueError("stability mapping contains unknown publication tiers")
    if set(config["publication_tier_order"]) != tiers:
        raise ValueError("publication_tier_order must contain every publication tier once")
    root = Path(project_root) if project_root else config_path.parent.parent
    config["project_root"] = root
    return config


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()


def _count(frame: pd.DataFrame, column: str, value: str) -> int:
    return int(_series(frame, column).eq(value).sum())


def build_evidence_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarise evidence by synthesis theme without treating findings as studies."""
    columns = [
        "synthesis_theme",
        "findings",
        "independent_sources",
        "findings_per_source",
        "high_quality",
        "moderate_quality",
        "high_transferability",
        "moderate_transferability",
        "low_transferability",
        "evaluated_implementation",
        "implemented_or_piloted",
        "proposed_or_planned",
        "complete_mechanism_chains",
    ]
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=columns)
    data = matrix.copy()
    if "synthesis_theme" not in data:
        data["synthesis_theme"] = "other"
    records: list[dict[str, object]] = []
    for theme, group in data.groupby("synthesis_theme", dropna=False, sort=True):
        sources = group.get("source_id", pd.Series(dtype=str)).nunique()
        findings = len(group)
        implementation = _series(group, "implementation_evidence_level")
        stage_count = pd.to_numeric(
            group.get("mechanism_chain_stage_count", pd.Series(0, index=group.index)),
            errors="coerce",
        ).fillna(0)
        records.append(
            {
                "synthesis_theme": clean(theme) or "other",
                "findings": findings,
                "independent_sources": int(sources),
                "findings_per_source": findings / sources if sources else 0.0,
                "high_quality": _count(group, "quality_overall_rating", "high"),
                "moderate_quality": _count(group, "quality_overall_rating", "moderate"),
                "high_transferability": _count(group, "transferability_class", "high"),
                "moderate_transferability": _count(group, "transferability_class", "moderate"),
                "low_transferability": _count(group, "transferability_class", "low"),
                "evaluated_implementation": int(implementation.eq("evaluated").sum()),
                "implemented_or_piloted": int(
                    implementation.eq("implemented_or_piloted").sum()
                ),
                "proposed_or_planned": int(
                    implementation.eq("proposed_or_planned").sum()
                ),
                "complete_mechanism_chains": int(stage_count.eq(4).sum()),
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_management_portfolio_table(
    portfolio: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Join operational readiness and robustness while preserving distinct meanings."""
    columns = [
        "option_id",
        "management_measure",
        "findings",
        "independent_sources",
        "high_quality",
        "high_transferability",
        "evaluated_implementation",
        "implemented_or_piloted",
        "proposed_or_planned",
        "mean_transferability_score",
        "mean_operational_specificity",
        "mean_data_feasibility_peru",
        "readiness_normalized_score",
        "readiness_class",
        "decision_pathway",
        "scenario_retention_rate",
        "leave_one_source_out_retention",
        "top_source_share",
        "effective_source_number",
        "implementation_dependency",
        "stability_class",
    ]
    if portfolio is None or portfolio.empty:
        return pd.DataFrame(columns=columns)
    output = portfolio.copy()
    if stability is not None and not stability.empty:
        stability_fields = [
            "option_id",
            "scenario_retention_rate",
            "leave_one_source_out_retention",
            "top_source_share",
            "effective_source_number",
            "implementation_dependency",
            "stability_class",
        ]
        available = [field for field in stability_fields if field in stability]
        stability_one = stability[available].drop_duplicates("option_id", keep="last")
        output = output.merge(stability_one, on="option_id", how="left")
    output = output.rename(columns={"sources": "independent_sources"})
    for column in columns:
        if column not in output:
            output[column] = ""
    order = {
        "decision_support_ready": 3,
        "pilot_ready": 2,
        "evidence_development": 1,
        "not_ready": 0,
    }
    output["_readiness_order"] = output["readiness_class"].map(order).fillna(-1)
    output = output.sort_values(
        ["_readiness_order", "readiness_normalized_score", "management_measure"],
        ascending=[False, False, True],
    )
    return output[columns].reset_index(drop=True)


def _publication_interpretation(stability_class: str) -> str:
    return {
        "robust": (
            "Stable under configured scenario and source-removal analyses. This does "
            "not demonstrate local management effectiveness."
        ),
        "generally_stable": (
            "Generally stable, but local validation and performance evaluation remain required."
        ),
        "assumption_sensitive": (
            "The conclusion changes under plausible filters or readiness assumptions."
        ),
        "source_dependent": (
            "The evidence base is materially concentrated in one source."
        ),
        "insufficiently_supported": (
            "Independent-source support or operational readiness is currently insufficient."
        ),
    }.get(stability_class, "Stability was not classified.")


def build_final_recommendations(
    validated_recommendations: pd.DataFrame,
    stability: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build publication-ready conditional recommendations from reviewed candidates."""
    columns = [
        "recommendation_id",
        "option_id",
        "management_measure",
        "expert_review_status",
        "candidate_statement",
        "base_readiness_class",
        "base_decision_pathway",
        "findings",
        "independent_sources",
        "scenario_retention_rate",
        "leave_one_source_out_retention",
        "top_source_share",
        "effective_source_number",
        "evaluated_implementation_findings",
        "implementation_dependency",
        "stability_class",
        "publication_tier",
        "publication_tier_label",
        "final_statement",
        "recommended_next_step",
        "publication_interpretation",
    ]
    if validated_recommendations is None or validated_recommendations.empty:
        return pd.DataFrame(columns=columns)
    if stability is None or stability.empty:
        merged = validated_recommendations.copy()
    else:
        stability_one = stability.drop_duplicates("recommendation_id", keep="last")
        merged = validated_recommendations.merge(
            stability_one,
            on=["recommendation_id", "option_id", "management_measure"],
            how="left",
            suffixes=("", "_stability"),
        )
    records: list[dict[str, object]] = []
    mapping = config["stability_to_publication_tier"]
    labels = config["publication_tier_labels"]
    next_steps = config["recommendation_next_steps"]
    for row in merged.fillna("").to_dict("records"):
        stability_class = clean(row.get("stability_class"))
        tier = clean(mapping.get(stability_class, "research_or_bounded_pilot"))
        candidate = clean(row.get("candidate_statement"))
        interpretation = _publication_interpretation(stability_class)
        final_statement = (
            f"{candidate} Evidence classification: {interpretation}"
            if candidate
            else interpretation
        )
        records.append(
            {
                "recommendation_id": clean(row.get("recommendation_id")),
                "option_id": clean(row.get("option_id")),
                "management_measure": clean(row.get("management_measure")),
                "expert_review_status": clean(row.get("expert_review_status")),
                "candidate_statement": candidate,
                "base_readiness_class": clean(row.get("base_readiness_class")),
                "base_decision_pathway": clean(row.get("base_decision_pathway")),
                "findings": integer(row.get("findings")),
                "independent_sources": integer(row.get("sources")),
                "scenario_retention_rate": numeric(row.get("scenario_retention_rate")),
                "leave_one_source_out_retention": numeric(
                    row.get("leave_one_source_out_retention")
                ),
                "top_source_share": numeric(row.get("top_source_share")),
                "effective_source_number": numeric(row.get("effective_source_number")),
                "evaluated_implementation_findings": integer(
                    row.get("evaluated_implementation_findings")
                ),
                "implementation_dependency": clean(
                    row.get("implementation_dependency")
                ),
                "stability_class": stability_class,
                "publication_tier": tier,
                "publication_tier_label": clean(labels.get(tier, tier)),
                "final_statement": final_statement,
                "recommended_next_step": clean(next_steps.get(stability_class, "")),
                "publication_interpretation": interpretation,
            }
        )
    output = pd.DataFrame(records, columns=columns)
    tier_order = {
        tier: index for index, tier in enumerate(config["publication_tier_order"])
    }
    output["_tier_order"] = output["publication_tier"].map(tier_order).fillna(999)
    output = output.sort_values(
        ["_tier_order", "management_measure"], ascending=[True, True]
    )
    return output[columns].reset_index(drop=True)


def build_research_gap_table(
    research_priorities: pd.DataFrame,
    final_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Create a publication table for explicit and robustness-derived research gaps."""
    columns = [
        "gap_id",
        "gap_origin",
        "gap_type",
        "synthesis_theme",
        "management_measure",
        "findings",
        "independent_sources",
        "priority_score",
        "priority_class",
        "recommended_action",
        "interpretation_guardrail",
    ]
    records: list[dict[str, object]] = []
    if research_priorities is not None and not research_priorities.empty:
        for row in research_priorities.fillna("").to_dict("records"):
            records.append(
                {
                    "gap_id": clean(row.get("priority_id")),
                    "gap_origin": "evidence_gap_matrix",
                    "gap_type": clean(row.get("gap_type")),
                    "synthesis_theme": clean(row.get("synthesis_theme")),
                    "management_measure": "",
                    "findings": integer(row.get("findings")),
                    "independent_sources": integer(row.get("sources")),
                    "priority_score": integer(row.get("priority_score")),
                    "priority_class": clean(row.get("priority_class")),
                    "recommended_action": clean(row.get("priority_statement")),
                    "interpretation_guardrail": clean(
                        row.get("interpretation_guardrail")
                    ),
                }
            )
    if final_recommendations is not None and not final_recommendations.empty:
        sensitive = final_recommendations.loc[
            final_recommendations["stability_class"].isin(
                [
                    "assumption_sensitive",
                    "source_dependent",
                    "insufficiently_supported",
                ]
            )
        ]
        for row in sensitive.fillna("").to_dict("records"):
            stability = clean(row.get("stability_class"))
            gap_type = {
                "assumption_sensitive": "analytical_assumption_sensitivity",
                "source_dependent": "independent_replication_gap",
                "insufficiently_supported": "insufficient_independent_support",
            }[stability]
            records.append(
                {
                    "gap_id": f"robustness::{clean(row.get('recommendation_id'))}",
                    "gap_origin": "robustness_audit",
                    "gap_type": gap_type,
                    "synthesis_theme": "management_and_governance",
                    "management_measure": clean(row.get("management_measure")),
                    "findings": integer(row.get("findings")),
                    "independent_sources": integer(row.get("independent_sources")),
                    "priority_score": "",
                    "priority_class": "high" if stability != "assumption_sensitive" else "moderate",
                    "recommended_action": clean(row.get("recommended_next_step")),
                    "interpretation_guardrail": clean(
                        row.get("publication_interpretation")
                    ),
                }
            )
    return pd.DataFrame(records, columns=columns).drop_duplicates(
        ["gap_id"], keep="last"
    ).reset_index(drop=True)


def build_source_influence_table(
    stability: pd.DataFrame,
    evidence_independence: pd.DataFrame,
) -> pd.DataFrame:
    """Provide source-independence diagnostics for publication and supplements."""
    columns = [
        "option_id",
        "management_measure",
        "findings",
        "independent_sources",
        "findings_per_source",
        "effective_source_number",
        "top_source_id",
        "top_source_share",
        "source_hhi",
        "leave_one_source_out_retention",
        "readiness_class_changes",
        "decision_pathway_changes",
        "implementation_dependency",
        "stability_class",
        "independence_flag",
    ]
    if stability is None or stability.empty:
        return pd.DataFrame(columns=columns)
    data = stability.drop_duplicates("option_id", keep="last").copy()
    if evidence_independence is not None and not evidence_independence.empty:
        independence_fields = [
            "option_id",
            "findings_per_source",
            "independence_flag",
        ]
        available = [field for field in independence_fields if field in evidence_independence]
        data = data.merge(
            evidence_independence[available].drop_duplicates("option_id", keep="last"),
            on="option_id",
            how="left",
        )
    data = data.rename(columns={"sources": "independent_sources"})
    for column in columns:
        if column not in data:
            data[column] = ""
    return data[columns].sort_values(
        ["top_source_share", "management_measure"], ascending=[False, True]
    ).reset_index(drop=True)


def build_publication_metrics(
    matrix: pd.DataFrame,
    portfolio: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    research_gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Build load-bearing counts used by figures and deterministic reports."""
    implementation = _series(matrix, "implementation_evidence_level")
    rows = [
        ("validated_findings", len(matrix)),
        (
            "independent_sources",
            matrix.get("source_id", pd.Series(dtype=str)).nunique(),
        ),
        ("management_options", len(portfolio)),
        ("validated_recommendations", len(final_recommendations)),
        ("research_gap_rows", len(research_gaps)),
        ("evaluated_implementation_findings", int(implementation.eq("evaluated").sum())),
        (
            "implemented_or_piloted_findings",
            int(implementation.eq("implemented_or_piloted").sum()),
        ),
    ]
    for stability_class in [
        "robust",
        "generally_stable",
        "assumption_sensitive",
        "source_dependent",
        "insufficiently_supported",
    ]:
        rows.append(
            (
                f"recommendations_{stability_class}",
                _count(final_recommendations, "stability_class", stability_class),
            )
        )
    for readiness_class in [
        "not_ready",
        "evidence_development",
        "pilot_ready",
        "decision_support_ready",
    ]:
        rows.append(
            (
                f"options_{readiness_class}",
                _count(portfolio, "readiness_class", readiness_class),
            )
        )
    return pd.DataFrame(rows, columns=["metric", "value"])


def _issue(
    output: str,
    row_number: int,
    record_id: object,
    field: str,
    issue: str,
) -> dict[str, object]:
    return {
        "output": output,
        "row_number": row_number,
        "record_id": clean(record_id),
        "field": field,
        "issue": issue,
    }


def validate_publication_tables(
    portfolio_table: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    research_gaps: pd.DataFrame,
    config: Mapping[str, object],
    *,
    expected_recommendation_ids: Sequence[str] | None = None,
    robustness_issues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate traceability, controlled values, and publication language."""
    issues: list[dict[str, object]] = []
    validation = config["validation"]
    controlled = config["controlled_values"]
    prohibited = [
        phrase.casefold()
        for phrase in config["language_guardrails"]["prohibited_claim_phrases"]
    ]
    if (
        bool(validation.get("require_zero_robustness_issues", True))
        and robustness_issues is not None
        and not robustness_issues.empty
    ):
        issues.append(
            _issue(
                "publication_inputs",
                1,
                "",
                "robustness_issues",
                "phase_12_robustness_issues_must_be_zero",
            )
        )
    if portfolio_table is not None and not portfolio_table.empty:
        duplicate = portfolio_table["option_id"].astype(str).duplicated(False)
        for index, row in portfolio_table.loc[duplicate].iterrows():
            issues.append(
                _issue(
                    "management_portfolio",
                    index + 2,
                    row.get("option_id"),
                    "option_id",
                    "duplicate_option_id",
                )
            )
        allowed_readiness = set(controlled["readiness_classes"])
        for index, row in portfolio_table.fillna("").iterrows():
            if clean(row.get("readiness_class")) not in allowed_readiness:
                issues.append(
                    _issue(
                        "management_portfolio",
                        index + 2,
                        row.get("option_id"),
                        "readiness_class",
                        "unknown_readiness_class",
                    )
                )
    if final_recommendations is not None and not final_recommendations.empty:
        duplicate = final_recommendations["recommendation_id"].astype(str).duplicated(False)
        for index, row in final_recommendations.loc[duplicate].iterrows():
            issues.append(
                _issue(
                    "final_recommendations",
                    index + 2,
                    row.get("recommendation_id"),
                    "recommendation_id",
                    "duplicate_recommendation_id",
                )
            )
        allowed_statuses = set(controlled["expert_review_statuses"])
        allowed_stability = set(controlled["stability_classes"])
        allowed_tiers = set(config["publication_tiers"])
        for index, row in final_recommendations.fillna("").iterrows():
            record_id = row.get("recommendation_id")
            if clean(row.get("expert_review_status")) not in allowed_statuses:
                issues.append(
                    _issue(
                        "final_recommendations",
                        index + 2,
                        record_id,
                        "expert_review_status",
                        "recommendation_not_accepted_or_corrected",
                    )
                )
            if clean(row.get("stability_class")) not in allowed_stability:
                issues.append(
                    _issue(
                        "final_recommendations",
                        index + 2,
                        record_id,
                        "stability_class",
                        "unknown_or_missing_stability_class",
                    )
                )
            if clean(row.get("publication_tier")) not in allowed_tiers:
                issues.append(
                    _issue(
                        "final_recommendations",
                        index + 2,
                        record_id,
                        "publication_tier",
                        "unknown_publication_tier",
                    )
                )
            statement = clean(row.get("final_statement")).casefold()
            if any(phrase in statement for phrase in prohibited):
                issues.append(
                    _issue(
                        "final_recommendations",
                        index + 2,
                        record_id,
                        "final_statement",
                        "prohibited_effectiveness_or_immediacy_claim",
                    )
                )
            if integer(row.get("evaluated_implementation_findings")) == 0 and re.search(
                r"\beffective(?:ness)?\s+(?:was|is|has been)\s+(?:shown|demonstrated|proven)",
                statement,
            ):
                issues.append(
                    _issue(
                        "final_recommendations",
                        index + 2,
                        record_id,
                        "final_statement",
                        "effectiveness_claim_without_evaluated_implementation",
                    )
                )
        if expected_recommendation_ids is not None:
            expected = {clean(value) for value in expected_recommendation_ids if clean(value)}
            observed = set(final_recommendations["recommendation_id"].astype(str))
            if expected != observed:
                issues.append(
                    _issue(
                        "final_recommendations",
                        1,
                        "",
                        "recommendation_id",
                        "validated_recommendations_not_fully_represented",
                    )
                )
    elif expected_recommendation_ids:
        issues.append(
            _issue(
                "final_recommendations",
                1,
                "",
                "recommendation_id",
                "final_recommendations_empty",
            )
        )
    if research_gaps is not None and not research_gaps.empty:
        duplicate = research_gaps["gap_id"].astype(str).duplicated(False)
        for index, row in research_gaps.loc[duplicate].iterrows():
            issues.append(
                _issue(
                    "research_gaps",
                    index + 2,
                    row.get("gap_id"),
                    "gap_id",
                    "duplicate_gap_id",
                )
            )
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def build_product_manifest(
    root: str | Path,
    configured_paths: Mapping[str, object],
    product_keys: Sequence[str],
) -> pd.DataFrame:
    """Register generated products with size and deterministic type metadata."""
    base = Path(root)
    records: list[dict[str, object]] = []
    for key in product_keys:
        relative = clean(configured_paths.get(key))
        path = base / relative if relative else Path("")
        exists = bool(relative and path.exists())
        kind = path.suffix.lower().lstrip(".") if relative else ""
        records.append(
            {
                "product_key": key,
                "relative_path": relative,
                "product_type": kind,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
            }
        )
    return pd.DataFrame(records)


from .publication_figures import generate_publication_figures  # noqa: E402
from .publication_reports import write_publication_reports  # noqa: E402

__all__ = [
    "build_evidence_summary",
    "build_final_recommendations",
    "build_management_portfolio_table",
    "build_product_manifest",
    "build_publication_metrics",
    "build_research_gap_table",
    "build_source_influence_table",
    "generate_publication_figures",
    "load_publication_config",
    "read_csv_robust",
    "validate_publication_tables",
    "write_publication_reports",
]
