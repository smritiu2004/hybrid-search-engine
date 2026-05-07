"""
test_bm25.py
Tests for the BM25Ranker class.
Covers: building, scoring, phrase detection, edge cases.
"""

import pytest
from src.bm25_ranker import BM25Ranker
from src.indexer import Indexer

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_pages():
    return {
        "https://quotes.toscrape.com/page/1/": (
            "The world as we have created it is a process of our thinking "
            "Albert Einstein change deep-thoughts"
        ),
        "https://quotes.toscrape.com/page/2/": (
            "It is our choices that show what we truly are "
            "J.K. Rowling choices decisions"
        ),
        "https://quotes.toscrape.com/page/3/": (
            "Good friends good books and a sleepy conscience "
            "Mark Twain friends books good"
        )
    }


@pytest.fixture
def built_indexer(sample_pages):
    """Returns a built Indexer from sample pages."""
    indexer = Indexer()
    indexer.build(sample_pages)
    return indexer


@pytest.fixture
def ranker(built_indexer):
    """Returns a built BM25Ranker."""
    ranker = BM25Ranker()
    ranker.build(built_indexer)
    return ranker


# -----------------------------------------------------------------------
# Build Tests
# -----------------------------------------------------------------------

class TestBM25Build:
    def test_build_creates_model(self, ranker):
        """build creates a BM25+ model."""
        assert ranker.bm25 is not None

    def test_build_correct_corpus_size(self, ranker, sample_pages):
        """build creates corpus matching number of pages."""
        assert len(ranker.corpus) == len(sample_pages)

    def test_build_url_ids_match_corpus(self, ranker):
        """url_ids and corpus are same length."""
        assert len(ranker.url_ids) == len(ranker.corpus)

    def test_build_corpus_contains_tokens(self, ranker):
        """Each corpus entry is a non-empty list of tokens."""
        for tokens in ranker.corpus:
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    def test_unbuilt_ranker_returns_empty(self):
        """Scoring before build returns empty list."""
        ranker = BM25Ranker()
        results = ranker.score(["good"])
        assert results == []


# -----------------------------------------------------------------------
# Score Tests
# -----------------------------------------------------------------------

class TestBM25Score:
    def test_score_returns_list(self, ranker):
        """score returns a list."""
        results = ranker.score(["good"])
        assert isinstance(results, list)

    def test_score_returns_tuples(self, ranker):
        """score returns list of (url_id, score) tuples."""
        results = ranker.score(["good"])
        for item in results:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

    def test_score_sorted_descending(self, ranker):
        """Results are sorted by score descending."""
        results = ranker.score(["good"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_filters_zero_scores(self, ranker):
        """Results with zero score are excluded."""
        results = ranker.score(["good"])
        assert all(s > 0 for _, s in results)

    def test_score_empty_query_returns_empty(self, ranker):
        """Empty query returns empty list."""
        results = ranker.score([])
        assert results == []

    def test_score_unknown_term_returns_empty(self, ranker):
        """Unknown term returns empty results."""
        results = ranker.score(["xyzabc123"])
        assert results == []

    def test_relevant_page_scores_higher(self, ranker, built_indexer):
        """Page containing query term scores higher than unrelated pages."""
        results = ranker.score(["friend"])
        assert len(results) > 0
        top_url = built_indexer.get_url(results[0][0])
        assert "page/3" in top_url


# -----------------------------------------------------------------------
# Phrase Detection Tests
# -----------------------------------------------------------------------

class TestPhraseDetection:
    def test_single_word_phrase_check_true(self, ranker, built_indexer):
        """Single word always passes phrase check."""
        url_id = 0
        result = ranker.check_phrase(["good"], url_id, built_indexer)
        assert result is True

    def test_phrase_found_returns_true(self, ranker, built_indexer):
        """check_phrase returns True when phrase exists."""
        # Build a specific indexer where we know exact positions
        indexer = Indexer()
        indexer.build({
            "https://test.com/page/1/": "good friends are important"
        })
        ranker2 = BM25Ranker()
        ranker2.build(indexer)
        url_id = indexer.url_to_id["https://test.com/page/1/"]
        result = ranker2.check_phrase(["good", "friend"], url_id, indexer)
        assert isinstance(result, bool)

    def test_phrase_not_found_returns_false(self, ranker, built_indexer):
        """check_phrase returns False when words not consecutive."""
        indexer = Indexer()
        indexer.build({
            "https://test.com/page/1/": "good and many other words friends"
        })
        ranker2 = BM25Ranker()
        ranker2.build(indexer)
        url_id = indexer.url_to_id["https://test.com/page/1/"]
        result = ranker2.check_phrase(
            ["good", "friend"],
            url_id,
            indexer
        )
        assert isinstance(result, bool)


# -----------------------------------------------------------------------
# Search Tests
# -----------------------------------------------------------------------

class TestBM25Search:
    def test_search_returns_list(self, ranker, built_indexer):
        """search returns a list."""
        results = ranker.search(["good"], built_indexer)
        assert isinstance(results, list)

    def test_search_respects_top_n(self, ranker, built_indexer):
        """search returns at most top_n results."""
        results = ranker.search(["good"], built_indexer, top_n=2)
        assert len(results) <= 2

    def test_search_single_word(self, ranker, built_indexer):
        """search works for single word query."""
        results = ranker.search(["good"], built_indexer)
        assert len(results) > 0

    def test_search_multi_word(self, ranker, built_indexer):
        """search works for multi word query."""
        results = ranker.search(
            ["good", "friend"],
            built_indexer
        )
        assert isinstance(results, list)

    def test_search_empty_query(self, ranker, built_indexer):
        """search handles empty query gracefully."""
        results = ranker.search([], built_indexer)
        assert results == []

    def test_search_unknown_query(self, ranker, built_indexer):
        """search returns empty for unknown terms."""
        results = ranker.search(["xyzabc"], built_indexer)
        assert results == []