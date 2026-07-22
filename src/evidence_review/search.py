"""Utilities for building reproducible multilingual search queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class SearchQuery:
    """One reproducible query assembled from controlled concept blocks."""

    language: str
    query_type: str
    blocks: tuple[str, ...]
    query: str


def load_search_strategy(path: str | Path = "config/search_strategy.yml") -> dict:
    """Load the search strategy configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _join_terms(terms: Iterable[str], operator: str = "OR") -> str:
    clean = [term.strip() for term in terms if str(term).strip()]
    if not clean:
        raise ValueError("A concept block cannot be empty")
    return "(" + f" {operator} ".join(clean) + ")"


def build_query(
    strategy: dict,
    *,
    language: str,
    blocks: Iterable[str],
    query_type: str,
) -> SearchQuery:
    """Build a Boolean query from named concept blocks.

    The function preserves the exact controlled terms from the YAML file. Platform-
    specific syntax should be adapted only at export time and must be recorded in the
    search log.
    """
    block_names = tuple(blocks)
    if not block_names:
        raise ValueError("At least one concept block is required")

    concepts = strategy["concept_blocks"]
    within = strategy["query_design"].get("operator_within_block", "OR")
    between = strategy["query_design"].get("operator_between_blocks", "AND")

    rendered: list[str] = []
    for name in block_names:
        if name not in concepts:
            raise KeyError(f"Unknown concept block: {name}")
        language_terms = concepts[name].get(language)
        if not language_terms:
            raise KeyError(f"No terms defined for block={name!r}, language={language!r}")
        rendered.append(_join_terms(language_terms, within))

    return SearchQuery(
        language=language,
        query_type=query_type,
        blocks=block_names,
        query=f" {between} ".join(rendered),
    )


def build_standard_queries(strategy: dict) -> list[SearchQuery]:
    """Build broad and priority searches for every configured language."""
    languages = strategy["concept_blocks"]["climate"].keys()
    broad_blocks = strategy["query_design"]["broad_query_blocks"]
    priority_blocks = strategy["query_design"]["priority_query_blocks"]

    queries: list[SearchQuery] = []
    for language in languages:
        queries.append(
            build_query(
                strategy,
                language=language,
                blocks=broad_blocks,
                query_type="broad",
            )
        )
        queries.append(
            build_query(
                strategy,
                language=language,
                blocks=priority_blocks,
                query_type="priority_taxa",
            )
        )
    return queries
