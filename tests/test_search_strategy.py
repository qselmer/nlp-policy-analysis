from pathlib import Path

import pytest

from evidence_review.search import build_query, build_standard_queries, load_search_strategy


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "config" / "search_strategy.yml"


def test_standard_queries_cover_all_languages_and_levels():
    strategy = load_search_strategy(STRATEGY_PATH)
    queries = build_standard_queries(strategy)

    languages = set(strategy["concept_blocks"]["climate"].keys())
    observed = {(query.language, query.query_type) for query in queries}

    assert len(queries) == len(languages) * 2
    for language in languages:
        assert (language, "broad") in observed
        assert (language, "priority_taxa") in observed


def test_priority_query_contains_three_blocks():
    strategy = load_search_strategy(STRATEGY_PATH)
    query = build_query(
        strategy,
        language="en",
        blocks=("climate", "fisheries", "taxa"),
        query_type="priority_taxa",
    )

    assert query.blocks == ("climate", "fisheries", "taxa")
    assert " AND " in query.query
    assert '"climate change"' in query.query
    assert '"small pelagic*"' in query.query


def test_unknown_language_is_rejected():
    strategy = load_search_strategy(STRATEGY_PATH)

    with pytest.raises(KeyError):
        build_query(
            strategy,
            language="de",
            blocks=("climate", "fisheries"),
            query_type="broad",
        )
