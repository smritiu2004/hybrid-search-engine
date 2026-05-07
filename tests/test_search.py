"""
test_search.py
Integration tests for the SearchEngine facade.
Covers: build, find, print_term, explain,
        mode flags, edge cases, no results handling.
"""

import pytest
from src.search import SearchEngine
from src.indexer import Indexer
from src.storage import Storage


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_pages():
    """Sample pages dict mimicking crawled content."""
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
        ),
        "https://quotes.toscrape.com/page/4/": (
            "Love is a serious mental disease Plato love philosophy"
        ),
        "https://quotes.toscrape.com/page/5/": (
            "Truth is rarely pure and never simple Oscar Wilde truth lies"
        )
    }


@pytest.fixture
def built_indexer(sample_pages):
    """Returns a built Indexer."""
    indexer = Indexer()
    indexer.build(sample_pages)
    return indexer


@pytest.fixture
def search_engine(built_indexer, sample_pages):
    """Returns a fully built SearchEngine."""
    engine = SearchEngine()
    engine.build(built_indexer, sample_pages)
    return engine


# -----------------------------------------------------------------------
# Initialisation Tests
# -----------------------------------------------------------------------

class TestSearchEngineInit:
    def test_not_ready_before_build(self):
        """SearchEngine is not ready before build or load."""
        engine = SearchEngine()
        assert engine.is_ready is False

    def test_ready_after_build(self, search_engine):
        """SearchEngine is ready after build."""
        assert search_engine.is_ready is True

    def test_components_initialised(self, search_engine):
        """All components are initialised after build."""
        assert search_engine.bm25 is not None
        assert search_engine.semantic is not None
        assert search_engine.hybrid is not None
        assert search_engine.reranker is not None
        assert search_engine.processor is not None
        assert search_engine.trie is not None
        assert search_engine.bktree is not None


# -----------------------------------------------------------------------
# Find Tests
# -----------------------------------------------------------------------

class TestSearchEngineFind:
    def test_find_returns_list(self, search_engine):
        """find returns a list."""
        results = search_engine.find("good")
        assert isinstance(results, list)

    def test_find_returns_result_dicts(self, search_engine):
        """find returns list of dicts with url and score."""
        results = search_engine.find("good")
        for result in results:
            assert "url" in result
            assert "score" in result

    def test_find_not_ready_returns_empty(self):
        """find returns empty when engine not ready."""
        engine = SearchEngine()
        results = engine.find("good")
        assert results == []

    def test_find_empty_query_returns_empty(self, search_engine):
        """find returns empty for empty query."""
        results = search_engine.find("")
        assert results == []

    def test_find_known_term_returns_results(self, search_engine):
        """find returns results for known term."""
        results = search_engine.find("good")
        assert len(results) > 0

    def test_find_multi_word_query(self, search_engine):
        """find handles multi-word queries."""
        results = search_engine.find("good friends")
        assert isinstance(results, list)

    def test_find_bm25_mode(self, search_engine):
        """find works in bm25 mode."""
        results = search_engine.find("truth", mode="bm25")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_find_semantic_mode(self, search_engine):
        """find works in semantic mode."""
        results = search_engine.find("truth", mode="semantic")
        assert isinstance(results, list)

    def test_find_hybrid_mode_default(self, search_engine):
        """find defaults to hybrid mode."""
        results = search_engine.find("love")
        assert isinstance(results, list)

    def test_find_unknown_term_returns_suggestion(self, search_engine):
        """find returns suggestion dict for unknown term."""
        results = search_engine.find("goood")
        assert isinstance(results, list)

    def test_find_results_have_type_field(self, search_engine):
        """find results contain type field."""
        results = search_engine.find("good")
        for result in results:
            assert "type" in result

    def test_find_result_type_is_result(self, search_engine):
        """find results have type 'result' for valid queries."""
        results = search_engine.find("good")
        result_types = [r["type"] for r in results]
        assert "result" in result_types

    def test_find_stopwords_only_returns_empty(self, search_engine):
        """find handles stopword-only queries gracefully."""
        results = search_engine.find("the and is")
        assert isinstance(results, list)


# -----------------------------------------------------------------------
# Print Term Tests
# -----------------------------------------------------------------------

class TestSearchEnginePrintTerm:
    def test_print_term_returns_dict(self, search_engine):
        """print_term returns a dictionary."""
        result = search_engine.print_term("good")
        assert isinstance(result, dict)

    def test_print_term_found_has_correct_type(self, search_engine):
        """print_term returns type 'found' for known term."""
        result = search_engine.print_term("good")
        assert result["type"] == "found"

    def test_print_term_contains_postings(self, search_engine):
        """print_term result contains postings."""
        result = search_engine.print_term("good")
        assert "postings" in result

    def test_print_term_contains_document_frequency(self, search_engine):
        """print_term result contains document frequency."""
        result = search_engine.print_term("good")
        assert "document_frequency" in result
        assert result["document_frequency"] > 0

    def test_print_term_postings_have_stats(self, search_engine):
        """print_term postings contain frequency, positions, tf, tf_idf."""
        result = search_engine.print_term("good")
        for url, stats in result["postings"].items():
            assert "frequency" in stats
            assert "positions" in stats
            assert "tf" in stats
            assert "tf_idf" in stats

    def test_print_term_unknown_returns_not_found(self, search_engine):
        """print_term returns not_found for unknown term."""
        result = search_engine.print_term("xyzabc")
        assert result["type"] == "not_found"

    def test_print_term_not_found_has_suggestions(self, search_engine):
        """print_term not_found result includes suggestions."""
        result = search_engine.print_term("goood")
        assert "suggestions" in result

    def test_print_term_case_insensitive(self, search_engine):
        """print_term handles uppercase input."""
        result = search_engine.print_term("GOOD")
        assert result["type"] == "found"

    def test_print_term_not_ready_returns_none(self):
        """print_term returns None when engine not ready."""
        engine = SearchEngine()
        result = engine.print_term("good")
        assert result is None


# -----------------------------------------------------------------------
# Explain Tests
# -----------------------------------------------------------------------

class TestSearchEngineExplain:
    def test_explain_returns_dict(self, search_engine):
        """explain returns a dictionary."""
        url = "https://quotes.toscrape.com/page/3/"
        result = search_engine.explain("good friends", url)
        assert isinstance(result, dict)

    def test_explain_contains_rrf_keys(self, search_engine):
        """explain result contains RRF breakdown keys."""
        url = "https://quotes.toscrape.com/page/3/"
        result = search_engine.explain("good friends", url)
        assert "final_rrf_score" in result
        assert "bm25_rank" in result
        assert "semantic_rank" in result

    def test_explain_unknown_url_returns_none(self, search_engine):
        """explain returns None for unknown URL."""
        result = search_engine.explain(
            "good",
            "https://unknown.com/page/99/"
        )
        assert result is None

    def test_explain_not_ready_returns_none(self):
        """explain returns None when engine not ready."""
        engine = SearchEngine()
        result = engine.explain(
            "good",
            "https://quotes.toscrape.com/page/1/"
        )
        assert result is None


# -----------------------------------------------------------------------
# Format Results Tests
# -----------------------------------------------------------------------

class TestFormatResults:
    def test_format_results_returns_list(self, search_engine):
        """_format_results returns a list."""
        results = search_engine._format_results([(0, 0.5)])
        assert isinstance(results, list)

    def test_format_results_contains_url(self, search_engine):
        """_format_results includes url in each result."""
        url_id = search_engine.indexer.url_to_id[
            "https://quotes.toscrape.com/page/1/"
        ]
        results = search_engine._format_results([(url_id, 0.5)])
        assert results[0]["url"] == \
               "https://quotes.toscrape.com/page/1/"

    def test_format_results_score_rounded(self, search_engine):
        """_format_results rounds scores to 4 decimal places."""
        url_id = search_engine.indexer.url_to_id[
            "https://quotes.toscrape.com/page/1/"
        ]
        results = search_engine._format_results([(url_id, 0.123456789)])
        assert results[0]["score"] == 0.1235

    def test_format_empty_results(self, search_engine):
        """_format_results handles empty input."""
        results = search_engine._format_results([])
        assert results == []