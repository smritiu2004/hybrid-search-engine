"""
test_hybrid.py
Tests for the HybridSearch class.
Covers: RRF fusion, edge cases, explain, graceful degradation.
"""

import pytest
from src.hybrid import HybridSearch
from src.indexer import Indexer


@pytest.fixture
def hybrid():
    """Returns a HybridSearch instance with default k=60."""
    return HybridSearch()


@pytest.fixture
def bm25_results():
    """Sample BM25+ results as (url_id, score) tuples."""
    return [
        (0, 5.42),
        (1, 4.31),
        (2, 3.20),
        (3, 2.10),
        (4, 1.05)
    ]


@pytest.fixture
def semantic_results():
    """Sample semantic results as (url_id, score) tuples."""
    return [
        (2, 0.91),
        (0, 0.87),
        (4, 0.75),
        (1, 0.65),
        (5, 0.55)
    ]


@pytest.fixture
def built_indexer():
    """Returns a built indexer with sample pages."""
    indexer = Indexer()
    indexer.build({
        "https://quotes.toscrape.com/page/1/": (
            "good friends are important in life"
        ),
        "https://quotes.toscrape.com/page/2/": (
            "choices define who we truly are"
        ),
        "https://quotes.toscrape.com/page/3/": (
            "world thinking process creativity"
        ),
        "https://quotes.toscrape.com/page/4/": (
            "books reading knowledge wisdom"
        ),
        "https://quotes.toscrape.com/page/5/": (
            "love friendship trust loyalty"
        ),
        "https://quotes.toscrape.com/page/6/": (
            "truth lies honesty integrity"
        ),
    })
    return indexer


class TestHybridInit:
    def test_default_k(self, hybrid):
        """HybridSearch initialises with k=60."""
        assert hybrid.k == 60

    def test_custom_k(self):
        """HybridSearch accepts custom k value."""
        hybrid = HybridSearch(k=30)
        assert hybrid.k == 30


class TestRRF:
    def test_rrf_returns_dict(self, hybrid, bm25_results, semantic_results):
        """RRF returns a dictionary of scores."""
        result = hybrid._reciprocal_rank_fusion(
            [bm25_results, semantic_results]
        )
        assert isinstance(result, dict)

    def test_rrf_scores_all_documents(
        self, hybrid, bm25_results, semantic_results
    ):
        """RRF scores all unique documents across both lists."""
        result = hybrid._reciprocal_rank_fusion(
            [bm25_results, semantic_results]
        )
        # url_ids 0-5 should all be present
        all_ids = {uid for uid, _ in bm25_results + semantic_results}
        assert all(uid in result for uid in all_ids)

    def test_rrf_document_in_both_lists_scores_higher(
        self, hybrid, bm25_results, semantic_results
    ):
        """Document appearing in both lists scores higher than one list only."""
        result = hybrid._reciprocal_rank_fusion(
            [bm25_results, semantic_results]
        )
        # url_id 0 appears in both lists
        # url_id 5 appears only in semantic
        assert result[0] > result[5]

    def test_rrf_formula_correct(self, hybrid):
        """RRF score matches 1/(k+rank) formula."""
        ranked_list = [(0, 5.0), (1, 3.0)]
        result = hybrid._reciprocal_rank_fusion([ranked_list])
        expected_0 = 1.0 / (60 + 1)
        expected_1 = 1.0 / (60 + 2)
        assert abs(result[0] - expected_0) < 1e-6
        assert abs(result[1] - expected_1) < 1e-6

    def test_rrf_empty_lists(self, hybrid):
        """RRF handles empty ranked lists."""
        result = hybrid._reciprocal_rank_fusion([[], []])
        assert result == {}

    def test_rrf_single_list(self, hybrid, bm25_results):
        """RRF works with a single ranked list."""
        result = hybrid._reciprocal_rank_fusion([bm25_results])
        assert isinstance(result, dict)
        assert len(result) == len(bm25_results)

class TestHybridSearch:
    def test_search_returns_list(
        self, hybrid, bm25_results, semantic_results
    ):
        """search returns a list."""
        results = hybrid.search(bm25_results, semantic_results)
        assert isinstance(results, list)

    def test_search_returns_tuples(
        self, hybrid, bm25_results, semantic_results
    ):
        """search returns list of (url_id, score) tuples."""
        results = hybrid.search(bm25_results, semantic_results)
        for item in results:
            assert len(item) == 2

    def test_search_sorted_descending(
        self, hybrid, bm25_results, semantic_results
    ):
        """search results are sorted by score descending."""
        results = hybrid.search(bm25_results, semantic_results)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_n(
        self, hybrid, bm25_results, semantic_results
    ):
        """search returns at most top_n results."""
        results = hybrid.search(bm25_results, semantic_results, top_n=3)
        assert len(results) <= 3

    def test_search_empty_bm25_falls_back_to_semantic(
        self, hybrid, semantic_results
    ):
        """search falls back to semantic when BM25+ empty."""
        results = hybrid.search([], semantic_results)
        assert results == semantic_results[:10]

    def test_search_empty_semantic_falls_back_to_bm25(
        self, hybrid, bm25_results
    ):
        """search falls back to BM25+ when semantic empty."""
        results = hybrid.search(bm25_results, [])
        assert results == bm25_results[:10]

    def test_search_both_empty_returns_empty(self, hybrid):
        """search returns empty when both lists empty."""
        results = hybrid.search([], [])
        assert results == []

    def test_search_combines_both_lists(
        self, hybrid, bm25_results, semantic_results
    ):
        """search includes documents from both lists."""
        results = hybrid.search(bm25_results, semantic_results)
        result_ids = {uid for uid, _ in results}
        bm25_ids = {uid for uid, _ in bm25_results}
        semantic_ids = {uid for uid, _ in semantic_results}
        all_ids = bm25_ids | semantic_ids
        assert result_ids.issubset(all_ids)



class TestHybridExplain:
    def test_explain_returns_dict(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain returns a dictionary."""
        result = hybrid.explain(
            url_id=0,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        assert isinstance(result, dict)

    def test_explain_contains_required_keys(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain dict contains all required keys."""
        result = hybrid.explain(
            url_id=0,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        required_keys = [
            "url",
            "bm25_rank",
            "bm25_score",
            "bm25_rrf_contribution",
            "semantic_rank",
            "semantic_score",
            "semantic_rrf_contribution",
            "final_rrf_score"
        ]
        for key in required_keys:
            assert key in result

    def test_explain_correct_bm25_rank(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain correctly identifies BM25+ rank."""
        result = hybrid.explain(
            url_id=0,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        assert result["bm25_rank"] == 1

    def test_explain_correct_semantic_rank(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain correctly identifies semantic rank."""
        result = hybrid.explain(
            url_id=0,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        assert result["semantic_rank"] == 2

    def test_explain_rrf_score_is_sum_of_contributions(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain final RRF score equals sum of contributions."""
        result = hybrid.explain(
            url_id=0,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        expected = round(
            result["bm25_rrf_contribution"] +
            result["semantic_rrf_contribution"],
            6
        )
        assert result["final_rrf_score"] == expected

    def test_explain_url_id_not_in_results(
        self, hybrid, bm25_results, semantic_results, built_indexer
    ):
        """explain handles url_id not present in either list."""
        result = hybrid.explain(
            url_id=999,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=built_indexer
        )
        assert result["bm25_rank"] is None
        assert result["semantic_rank"] is None
        assert result["final_rrf_score"] == 0.0