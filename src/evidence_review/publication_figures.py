"""Deterministic matplotlib figures for phase 13 publication products."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import re
import textwrap
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyBboxPatch
import numpy as np
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


def _split(value: object) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[|;]", _clean(value))
        if item.strip()
    ]


def _label(value: object, width: int = 28) -> str:
    text = _clean(value).replace("_", " ")
    return textwrap.fill(text, width=width)


def _prepare(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _finish(fig: plt.Figure, path: str | Path, dpi: int) -> Path:
    output = _prepare(path)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _metric_lookup(metrics: pd.DataFrame) -> dict[str, int]:
    if metrics is None or metrics.empty:
        return {}
    return {
        _clean(row.get("metric")): int(round(_numeric(row.get("value"))))
        for row in metrics.fillna("").to_dict("records")
    }


def plot_evidence_flow(
    metrics: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Plot the validated evidence-to-recommendation workflow, not a PRISMA count."""
    values = _metric_lookup(metrics)
    stages = [
        ("Independent sources", values.get("independent_sources", 0)),
        ("Validated findings", values.get("validated_findings", 0)),
        ("Management options", values.get("management_options", 0)),
        ("Expert-validated recommendations", values.get("validated_recommendations", 0)),
        (
            "Robust or generally stable",
            values.get("recommendations_robust", 0)
            + values.get("recommendations_generally_stable", 0),
        ),
    ]
    width = float(settings.get("width_inches", 10))
    height = float(settings.get("height_inches", 6))
    dpi = int(settings.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_ylim(0, 1)
    for index, (name, value) in enumerate(stages):
        box = FancyBboxPatch(
            (index - 0.38, 0.36),
            0.76,
            0.28,
            boxstyle="round,pad=0.03",
            linewidth=1.4,
            facecolor=f"C{index % 10}",
            alpha=0.22,
        )
        ax.add_patch(box)
        ax.text(index, 0.52, f"{value}", ha="center", va="center", fontsize=16)
        ax.text(index, 0.29, _label(name, 20), ha="center", va="top", fontsize=10)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(index + 0.58, 0.50),
                xytext=(index + 0.42, 0.50),
                arrowprops={"arrowstyle": "->", "linewidth": 1.4},
            )
    ax.set_title("Validated evidence and decision-support workflow")
    ax.text(
        0.5,
        0.06,
        "Counts represent different analytical units; findings are not independent studies.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.axis("off")
    return _finish(fig, path, dpi)


def plot_quality_transferability(
    matrix: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Plot finding counts across source quality and transferability classes."""
    quality_order = ["high", "moderate", "low", "not_applicable", "unclear"]
    transfer_order = ["high", "moderate", "low", "not_applicable"]
    width = float(settings.get("width_inches", 10))
    height = float(settings.get("height_inches", 6))
    dpi = int(settings.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(width, height))
    if matrix is None or matrix.empty:
        ax.text(0.5, 0.5, "No evidence matrix available", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path, dpi)
    cross = pd.crosstab(
        matrix.get("quality_overall_rating", pd.Series("unclear", index=matrix.index)),
        matrix.get("transferability_class", pd.Series("not_applicable", index=matrix.index)),
    ).reindex(index=quality_order, columns=transfer_order, fill_value=0)
    bottom = np.zeros(len(cross), dtype=float)
    for index, transfer in enumerate(transfer_order):
        values = cross[transfer].to_numpy(dtype=float)
        ax.bar(
            np.arange(len(cross)),
            values,
            bottom=bottom,
            label=_label(transfer),
            color=f"C{index}",
        )
        bottom += values
    ax.set_xticks(np.arange(len(cross)), [_label(item) for item in cross.index])
    ax.set_ylabel("Validated findings")
    ax.set_xlabel("Validated source-quality class")
    ax.set_title("Source quality and finding-level transferability")
    ax.legend(title="Transferability", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, path, dpi)


def plot_management_readiness(
    portfolio: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Plot operational-readiness scores with class annotations."""
    width = float(settings.get("width_inches", 10))
    height = max(float(settings.get("height_inches", 6)), 0.52 * max(len(portfolio), 1))
    dpi = int(settings.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(width, height))
    if portfolio is None or portfolio.empty:
        ax.text(0.5, 0.5, "No management portfolio available", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path, dpi)
    data = portfolio.copy()
    data["_score"] = pd.to_numeric(
        data.get("readiness_normalized_score"), errors="coerce"
    ).fillna(0)
    data = data.sort_values(["_score", "management_measure"], ascending=[True, True])
    y = np.arange(len(data))
    bars = ax.barh(y, data["_score"].to_numpy(), color="C0", alpha=0.75)
    ax.set_yticks(y, [_label(value, 34) for value in data["management_measure"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized operational-readiness score")
    ax.set_title("Management-option readiness does not imply effectiveness")
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            min(bar.get_width() + 0.02, 1.01),
            bar.get_y() + bar.get_height() / 2,
            _label(row.get("readiness_class"), 20),
            va="center",
            fontsize=8,
        )
    ax.axvline(0.50, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(0.75, linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(axis="x", alpha=0.25)
    return _finish(fig, path, dpi)


def plot_recommendation_stability(
    final_recommendations: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Compare scenario retention, source-removal retention, and concentration."""
    width = float(settings.get("width_inches", 10))
    height = max(
        float(settings.get("height_inches", 6)),
        0.58 * max(len(final_recommendations), 1),
    )
    dpi = int(settings.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(width, height))
    if final_recommendations is None or final_recommendations.empty:
        ax.text(0.5, 0.5, "No validated recommendations available", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path, dpi)
    data = final_recommendations.copy()
    for column in [
        "scenario_retention_rate",
        "leave_one_source_out_retention",
        "top_source_share",
    ]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce").fillna(0)
    data = data.sort_values(
        ["scenario_retention_rate", "leave_one_source_out_retention", "management_measure"]
    )
    y = np.arange(len(data))
    ax.scatter(
        data["scenario_retention_rate"],
        y - 0.16,
        marker="o",
        label="Scenario retention",
        color="C0",
    )
    ax.scatter(
        data["leave_one_source_out_retention"],
        y,
        marker="s",
        label="Leave-one-source-out retention",
        color="C1",
    )
    ax.scatter(
        data["top_source_share"],
        y + 0.16,
        marker="^",
        label="Top-source share",
        color="C2",
    )
    ax.set_yticks(y, [_label(value, 34) for value in data["management_measure"]])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Rate or share")
    ax.set_title("Recommendation stability and evidence concentration")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return _finish(fig, path, dpi)


def plot_source_concentration(
    source_table: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Plot dominant-source shares and annotate independent-source support."""
    width = float(settings.get("width_inches", 10))
    height = max(float(settings.get("height_inches", 6)), 0.55 * max(len(source_table), 1))
    dpi = int(settings.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(width, height))
    if source_table is None or source_table.empty:
        ax.text(0.5, 0.5, "No source-concentration results available", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path, dpi)
    data = source_table.copy()
    data["_share"] = pd.to_numeric(data.get("top_source_share"), errors="coerce").fillna(0)
    data = data.sort_values(["_share", "management_measure"], ascending=[True, True])
    y = np.arange(len(data))
    bars = ax.barh(y, data["_share"], color="C3", alpha=0.72)
    ax.set_yticks(y, [_label(value, 34) for value in data["management_measure"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of findings from the dominant source")
    ax.set_title("Source concentration by management option")
    ax.axvline(0.50, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(0.75, linestyle="--", linewidth=1, alpha=0.6)
    for bar, (_, row) in zip(bars, data.iterrows()):
        sources = int(round(_numeric(row.get("independent_sources"))))
        effective = _numeric(row.get("effective_source_number"))
        ax.text(
            min(bar.get_width() + 0.02, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"n={sources}; effective={effective:.1f}",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", alpha=0.25)
    return _finish(fig, path, dpi)


def _top_mapping(values: list[list[str]], maximum: int) -> dict[str, str]:
    counts = Counter(item for items in values for item in items)
    top = {item for item, _ in counts.most_common(maximum)}
    return {item: item if item in top else "other" for item in counts}


def plot_chain_network(
    matrix: pd.DataFrame,
    path: str | Path,
    settings: Mapping[str, object],
) -> Path:
    """Plot an aggregated four-stage evidence network without causal overstatement."""
    width = float(settings.get("width_inches", 10)) + 2
    height = float(settings.get("height_inches", 6)) + 2
    dpi = int(settings.get("dpi", 220))
    maximum = int(settings.get("network_top_nodes_per_stage", 6))
    fig, ax = plt.subplots(figsize=(width, height))
    stages = [
        "climate_drivers",
        "biological_responses",
        "fishery_responses",
        "management_measures",
    ]
    stage_labels = ["Climate/environment", "Biological response", "Fishery response", "Management"]
    if matrix is None or matrix.empty:
        ax.text(0.5, 0.5, "No evidence chains available", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path, dpi)
    stage_values: dict[str, list[list[str]]] = {
        stage: [
            _split(value) or ["not reported"]
            for value in matrix.get(stage, pd.Series("", index=matrix.index))
        ]
        for stage in stages
    }
    maps = {
        stage: _top_mapping(stage_values[stage], maximum)
        for stage in stages
    }
    node_counts: dict[tuple[int, str], int] = Counter()
    edge_counts: dict[tuple[int, str, str], int] = Counter()
    for row_index in range(len(matrix)):
        mapped_stages: list[list[str]] = []
        for stage in stages:
            mapped = [maps[stage].get(item, item) for item in stage_values[stage][row_index]]
            mapped_stages.append(list(dict.fromkeys(mapped)))
        for stage_index, nodes in enumerate(mapped_stages):
            for node in nodes:
                node_counts[(stage_index, node)] += 1
        for stage_index in range(len(stages) - 1):
            for left, right in product(mapped_stages[stage_index], mapped_stages[stage_index + 1]):
                edge_counts[(stage_index, left, right)] += 1
    positions: dict[tuple[int, str], tuple[float, float]] = {}
    for stage_index in range(len(stages)):
        nodes = [
            node for (index, node), _ in node_counts.items() if index == stage_index
        ]
        nodes = sorted(nodes, key=lambda item: (-node_counts[(stage_index, item)], item))
        ys = np.linspace(0.88, 0.12, max(len(nodes), 1))
        for node, y in zip(nodes, ys):
            positions[(stage_index, node)] = (float(stage_index), float(y))
    maximum_edge = max(edge_counts.values()) if edge_counts else 1
    for (stage_index, left, right), count in edge_counts.items():
        x0, y0 = positions[(stage_index, left)]
        x1, y1 = positions[(stage_index + 1, right)]
        path_data = MplPath(
            [(x0 + 0.10, y0), (x0 + 0.45, y0), (x1 - 0.45, y1), (x1 - 0.10, y1)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        patch = PathPatch(
            path_data,
            facecolor="none",
            edgecolor=f"C{stage_index}",
            linewidth=0.35 + 4.0 * count / maximum_edge,
            alpha=0.20,
        )
        ax.add_patch(patch)
    maximum_node = max(node_counts.values()) if node_counts else 1
    for (stage_index, node), count in node_counts.items():
        x, y = positions[(stage_index, node)]
        ax.scatter(
            [x],
            [y],
            s=70 + 520 * count / maximum_node,
            color=f"C{stage_index}",
            alpha=0.80,
            zorder=3,
        )
        horizontal = "right" if stage_index == len(stages) - 1 else "left"
        offset = -0.12 if horizontal == "right" else 0.12
        ax.text(
            x + offset,
            y,
            _label(node, 22),
            ha=horizontal,
            va="center",
            fontsize=8,
        )
    for index, label in enumerate(stage_labels):
        ax.text(index, 0.98, label, ha="center", va="bottom", fontweight="bold")
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.02, 1.05)
    ax.set_title("Aggregated evidence links across climate, biological, fishery, and management stages")
    ax.text(
        0.5,
        -0.02,
        "Links indicate co-occurrence in validated findings; they are not automatically causal effects.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.axis("off")
    return _finish(fig, path, dpi)


def generate_publication_figures(
    *,
    matrix: pd.DataFrame,
    portfolio: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    source_table: pd.DataFrame,
    metrics: pd.DataFrame,
    root: str | Path,
    paths: Mapping[str, object],
    settings: Mapping[str, object],
) -> pd.DataFrame:
    """Generate all configured phase 13 figures and return their paths."""
    base = Path(root)
    products = [
        (
            "evidence_flow_png",
            plot_evidence_flow(metrics, base / str(paths["evidence_flow_png"]), settings),
        ),
        (
            "quality_transferability_png",
            plot_quality_transferability(
                matrix, base / str(paths["quality_transferability_png"]), settings
            ),
        ),
        (
            "management_readiness_png",
            plot_management_readiness(
                portfolio, base / str(paths["management_readiness_png"]), settings
            ),
        ),
        (
            "recommendation_stability_png",
            plot_recommendation_stability(
                final_recommendations,
                base / str(paths["recommendation_stability_png"]),
                settings,
            ),
        ),
        (
            "source_concentration_png",
            plot_source_concentration(
                source_table, base / str(paths["source_concentration_png"]), settings
            ),
        ),
        (
            "climate_response_management_network_png",
            plot_chain_network(
                matrix,
                base / str(paths["climate_response_management_network_png"]),
                settings,
            ),
        ),
    ]
    return pd.DataFrame(
        [
            {"product_key": key, "relative_path": str(path.relative_to(base))}
            for key, path in products
        ]
    )


__all__ = [
    "generate_publication_figures",
    "plot_chain_network",
    "plot_evidence_flow",
    "plot_management_readiness",
    "plot_quality_transferability",
    "plot_recommendation_stability",
    "plot_source_concentration",
]
