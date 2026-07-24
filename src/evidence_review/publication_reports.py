"""Deterministic Markdown reports for phase 13 scientific products."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


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


def _metric(metrics: pd.DataFrame, name: str) -> int:
    if metrics is None or metrics.empty:
        return 0
    selected = metrics.loc[metrics["metric"].astype(str).eq(name), "value"]
    return int(round(_numeric(selected.iloc[0]))) if len(selected) else 0


def _escape(value: object) -> str:
    return _clean(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: Mapping[str, str] | None = None,
    *,
    maximum_rows: int = 30,
) -> str:
    """Render a compact Markdown table without a tabulate dependency."""
    labels = labels or {}
    available = [column for column in columns if column in frame]
    if frame is None or frame.empty or not available:
        return "_No rows available._"
    data = frame[available].head(maximum_rows)
    header = "| " + " | ".join(labels.get(column, column) for column in available) + " |"
    separator = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(_escape(row.get(column)) for column in available) + " |"
        for row in data.to_dict("records")
    ]
    suffix = ""
    if len(frame) > maximum_rows:
        suffix = f"\n\n_Table truncated to {maximum_rows} of {len(frame)} rows._"
    return "\n".join([header, separator, *rows]) + suffix


def _recommendation_bullets(final_recommendations: pd.DataFrame, maximum: int) -> str:
    if final_recommendations is None or final_recommendations.empty:
        return "- No expert-validated recommendations were available."
    lines: list[str] = []
    for row in final_recommendations.head(maximum).fillna("").to_dict("records"):
        measure = _clean(row.get("management_measure")).replace("_", " ")
        tier = _clean(row.get("publication_tier_label"))
        next_step = _clean(row.get("recommended_next_step"))
        stability = _clean(row.get("stability_class")).replace("_", " ")
        lines.append(
            f"- **{measure}** — {tier}; stability: `{stability}`. {next_step}"
        )
    return "\n".join(lines)


def build_evidence_synthesis_report(
    *,
    evidence_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    research_gaps: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> str:
    """Build the complete scientific synthesis report from validated outputs."""
    maximum_rows = int(config["report_settings"].get("maximum_rows_per_markdown_table", 30))
    target = _clean(config["project"].get("target_fishery"))
    findings = _metric(metrics, "validated_findings")
    sources = _metric(metrics, "independent_sources")
    options = _metric(metrics, "management_options")
    recommendations = _metric(metrics, "validated_recommendations")
    evaluated = _metric(metrics, "evaluated_implementation_findings")
    stable = _metric(metrics, "recommendations_robust") + _metric(
        metrics, "recommendations_generally_stable"
    )
    theme_table = markdown_table(
        evidence_summary,
        [
            "synthesis_theme",
            "findings",
            "independent_sources",
            "high_quality",
            "high_transferability",
            "evaluated_implementation",
            "complete_mechanism_chains",
        ],
        {
            "synthesis_theme": "Theme",
            "independent_sources": "Sources",
            "high_quality": "High quality",
            "high_transferability": "High transferability",
            "evaluated_implementation": "Evaluated implementation",
            "complete_mechanism_chains": "Complete chains",
        },
        maximum_rows=maximum_rows,
    )
    portfolio_table = markdown_table(
        portfolio,
        [
            "management_measure",
            "independent_sources",
            "readiness_class",
            "decision_pathway",
            "scenario_retention_rate",
            "leave_one_source_out_retention",
            "top_source_share",
            "stability_class",
        ],
        {
            "management_measure": "Management option",
            "independent_sources": "Sources",
            "readiness_class": "Readiness",
            "decision_pathway": "Decision pathway",
            "scenario_retention_rate": "Scenario retention",
            "leave_one_source_out_retention": "LOSO retention",
            "top_source_share": "Top-source share",
            "stability_class": "Stability",
        },
        maximum_rows=maximum_rows,
    )
    recommendation_table = markdown_table(
        final_recommendations,
        [
            "management_measure",
            "publication_tier_label",
            "stability_class",
            "independent_sources",
            "evaluated_implementation_findings",
            "recommended_next_step",
        ],
        {
            "management_measure": "Management option",
            "publication_tier_label": "Publication category",
            "stability_class": "Stability",
            "independent_sources": "Sources",
            "evaluated_implementation_findings": "Evaluated implementation findings",
            "recommended_next_step": "Next step",
        },
        maximum_rows=maximum_rows,
    )
    gap_table = markdown_table(
        research_gaps,
        [
            "gap_origin",
            "gap_type",
            "management_measure",
            "independent_sources",
            "priority_class",
            "recommended_action",
        ],
        {
            "gap_origin": "Origin",
            "gap_type": "Gap",
            "management_measure": "Management option",
            "independent_sources": "Sources",
            "priority_class": "Priority",
            "recommended_action": "Required action",
        },
        maximum_rows=maximum_rows,
    )
    return f"""# Evidence synthesis and decision-support portfolio

## Scope

This report synthesizes validated climate–fisheries evidence for **{target}**. The analytical chain distinguishes source quality, finding-level transferability, operational readiness, expert acceptance, and robustness. None of these dimensions alone demonstrates management effectiveness.

## Evidence base

The synthesis contains **{findings} validated findings** from **{sources} independent sources**. It generated **{options} management-option candidates** and **{recommendations} expert-validated conditional recommendations**. Findings are not treated as independent studies; the source is the principal unit of independence.

{theme_table}

## Decision-support portfolio

Operational readiness describes whether evidence, data, specificity, and implementation maturity support progression to further testing or decision procedures. It does not establish biological, fishery, or governance effectiveness.

{portfolio_table}

## Stability and recommendation framing

Across the configured scenario and leave-one-source-out analyses, **{stable} recommendations** were robust or generally stable. The remaining recommendations were sensitive to assumptions, dependent on concentrated source support, or insufficiently supported. There were **{evaluated} finding-level records of evaluated implementation**; this count must not be interpreted as proof of effectiveness in the target fishery.

{recommendation_table}

## Research and validation gaps

The table combines explicit evidence gaps with gaps detected by the robustness audit. Absence of evidence is not coded as evidence of no effect.

{gap_table}

## Interpretation guardrails

1. A proposal or recommendation is not evidence of implementation.
2. Implementation or piloting is not evidence of effectiveness.
3. A project activity or output is not an evaluated outcome or impact.
4. A projection or scenario is not an observed trend.
5. Finding counts do not replace independent-source counts.
6. Analytical stability does not guarantee local management performance.
7. Every operational progression requires IMARPE–PRODUCE expert review, local validation, safeguards, and performance monitoring.

## Reproducibility

All tables, figures, and reports are generated deterministically from the phase 10–12 CSV products through `notebooks/13_build_scientific_products.ipynb`. Generated files are written under `data/processed/phase13/` and remain separate from the version-controlled code and templates.
"""


def build_methods_and_results_report(
    *,
    evidence_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    source_table: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> str:
    """Build a methods-and-results section suitable for manuscript development."""
    findings = _metric(metrics, "validated_findings")
    sources = _metric(metrics, "independent_sources")
    recommendations = _metric(metrics, "validated_recommendations")
    robust = _metric(metrics, "recommendations_robust")
    generally = _metric(metrics, "recommendations_generally_stable")
    sensitive = _metric(metrics, "recommendations_assumption_sensitive")
    dependent = _metric(metrics, "recommendations_source_dependent")
    insufficient = _metric(metrics, "recommendations_insufficiently_supported")
    readiness_table = markdown_table(
        portfolio,
        [
            "management_measure",
            "findings",
            "independent_sources",
            "readiness_normalized_score",
            "readiness_class",
            "stability_class",
        ],
        maximum_rows=int(config["report_settings"].get("maximum_rows_per_markdown_table", 30)),
    )
    source_table_md = markdown_table(
        source_table,
        [
            "management_measure",
            "findings",
            "independent_sources",
            "findings_per_source",
            "effective_source_number",
            "top_source_share",
            "leave_one_source_out_retention",
            "independence_flag",
        ],
        maximum_rows=int(config["report_settings"].get("maximum_rows_per_markdown_table", 30)),
    )
    return f"""# Methods and results

## Methods

### Evidence unit and independence

The analysis used validated findings as the unit of extracted evidence and source records as the principal unit of independence. A source could contribute multiple findings, so finding counts were never interpreted as equivalent to independent studies.

### Quality and transferability

Source quality was inherited from the validated phase 09 appraisal. Transferability to north-central Peruvian anchoveta was assessed separately at finding level using ecological similarity, climate relevance, management relevance, inherited evidence strength, operational specificity, and data feasibility in Peru.

### Decision-support readiness

Management measures were aggregated across validated findings. Readiness combined evidence coverage, source quality, transferability, implementation maturity, operational specificity, and Peruvian data feasibility. Gates prevented a measure from being classified as decision-support ready without minimum implementation, specificity, and data criteria.

### Robustness audit

Eight sensitivity scenarios were evaluated: base, conservative, permissive, high-quality only, high-transferability only, high quality plus high transferability, observational only, and implemented-or-evaluated only. A leave-one-source-out analysis removed each independent source and rebuilt the option portfolio. Source concentration was quantified with dominant-source share, HHI, and effective source number.

### Publication classification

Expert-validated recommendations were assigned one of five publication categories based on robustness: conditional priority, local-validation priority, scenario-evaluation required, independent-replication required, or research/bounded-pilot candidate. The classification governs communication and next steps; it does not estimate effect size or effectiveness.

## Results

The final synthesis included **{findings} findings from {sources} independent sources** and **{recommendations} expert-validated recommendations**. Stability classes were distributed as follows: **{robust} robust**, **{generally} generally stable**, **{sensitive} assumption sensitive**, **{dependent} source dependent**, and **{insufficient} insufficiently supported**.

### Management-option portfolio

{readiness_table}

### Source independence and concentration

{source_table_md}

## Reporting constraints

Results describe evidence coverage, transferability, readiness, and stability. They do not demonstrate that any option will improve biomass, recruitment, catch stability, compliance, or resilience in the target fishery. Such claims require explicit local implementation and outcome evaluation.
"""


def build_executive_decision_brief(
    *,
    final_recommendations: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> str:
    """Build a concise decision brief with conditional, stability-aware language."""
    maximum = int(
        config["report_settings"].get("maximum_recommendations_in_executive_brief", 10)
    )
    findings = _metric(metrics, "validated_findings")
    sources = _metric(metrics, "independent_sources")
    recommendations = _metric(metrics, "validated_recommendations")
    robust = _metric(metrics, "recommendations_robust")
    generally = _metric(metrics, "recommendations_generally_stable")
    evaluated = _metric(metrics, "evaluated_implementation_findings")
    bullets = _recommendation_bullets(final_recommendations, maximum)
    return f"""# Executive decision brief

## Decision context

The evidence review retained **{findings} validated findings from {sources} independent sources** and produced **{recommendations} expert-validated conditional recommendations** for north-central Peruvian anchoveta.

## Central result

Only **{robust} recommendations were classified as robust** and **{generally} as generally stable** under the configured sensitivity and source-removal analyses. Stability supports prioritization for local evaluation; it does not establish effectiveness. The evidence matrix contained **{evaluated} evaluated-implementation findings**, which remain context-specific.

## Recommended portfolio treatment

{bullets}

## Required governance safeguards

- Maintain IMARPE scientific review and PRODUCE decision authority.
- Define local indicators, thresholds, responsibilities, timing, and reversal rules before piloting.
- Use bounded pilots or management strategy evaluation when evidence is sensitive or incomplete.
- Monitor biological, fishery, compliance, distributional, and implementation outcomes.
- Predefine stopping or adaptation criteria.
- Do not infer effectiveness from implementation status, expert acceptance, or analytical stability alone.

## Use of this brief

This brief ranks evidence-informed candidates for further review, testing, or research. It is not an automatic management prescription and does not replace stock assessment, legal review, stakeholder processes, or formal advice procedures.
"""


def write_publication_reports(
    *,
    evidence_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    research_gaps: pd.DataFrame,
    source_table: pd.DataFrame,
    metrics: pd.DataFrame,
    root: str | Path,
    paths: Mapping[str, object],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Write the three configured phase 13 Markdown reports."""
    base = Path(root)
    reports = {
        "evidence_synthesis_report_md": build_evidence_synthesis_report(
            evidence_summary=evidence_summary,
            portfolio=portfolio,
            final_recommendations=final_recommendations,
            research_gaps=research_gaps,
            metrics=metrics,
            config=config,
        ),
        "methods_and_results_md": build_methods_and_results_report(
            evidence_summary=evidence_summary,
            portfolio=portfolio,
            final_recommendations=final_recommendations,
            source_table=source_table,
            metrics=metrics,
            config=config,
        ),
        "executive_decision_brief_md": build_executive_decision_brief(
            final_recommendations=final_recommendations,
            metrics=metrics,
            config=config,
        ),
    }
    records: list[dict[str, object]] = []
    for key, text in reports.items():
        path = base / str(paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        records.append(
            {
                "product_key": key,
                "relative_path": str(path.relative_to(base)),
                "characters": len(text),
            }
        )
    return pd.DataFrame(records)


__all__ = [
    "build_evidence_synthesis_report",
    "build_executive_decision_brief",
    "build_methods_and_results_report",
    "markdown_table",
    "write_publication_reports",
]
