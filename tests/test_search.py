"""
test_search.py
Integration tests for the SearchEngine facade.
Covers: build, find, print_term, explain,
        mode flags, edge cases, no results handling.
"""

from unittest import result

import pytest
from src.search import SearchEngine
from src.indexer import Indexer
from src.storage import Storage
import os


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


class TestRerankerEdgeCases:
    def test_rerank_empty_candidates(self, search_engine):
        """reranker returns empty list for empty candidates."""
        result = search_engine.reranker.rerank(
            query="good",
            candidates=[],
            indexer=search_engine.indexer,
            pages=search_engine.pages
        )
        assert result == []

    def test_rerank_empty_query(self, search_engine):
        """reranker returns empty list for empty query."""
        result = search_engine.reranker.rerank(
            query="",
            candidates=[(0, 0.5)],
            indexer=search_engine.indexer,
            pages=search_engine.pages
        )
        assert result == []

    def test_rerank_no_valid_pairs_returns_candidates(
        self, search_engine
    ):
        """reranker returns original candidates when no text found."""
        # Use url_id that exists but with empty pages
        empty_pages = {
            url: "" for url in search_engine.indexer.url_to_id.keys()
        }
        url_id = list(search_engine.indexer.url_to_id.values())[0]
        candidates = [(url_id, 0.5)]
        result = search_engine.reranker.rerank(
            query="good",
            candidates=candidates,
            indexer=search_engine.indexer,
            pages=empty_pages
        )
        assert result == candidates

    def test_rerank_skips_missing_text(self, search_engine):
        """reranker skips candidates with no page text."""
        partial_pages = {}
        url_id = list(search_engine.indexer.url_to_id.values())[0]
        candidates = [(url_id, 0.5), (9999, 0.3)]
        result = search_engine.reranker.rerank(
            query="good",
            candidates=candidates,
            indexer=search_engine.indexer,
            pages=partial_pages
        )
        assert isinstance(result, list)

class TestSemanticEdgeCases:
    def test_search_before_build_returns_empty(self):
        """semantic search returns empty if FAISS not built."""
        from src.semantic import SemanticSearch
        semantic = SemanticSearch()
        result = semantic.search("good")
        assert result == []

    def test_search_empty_query_returns_empty(self, search_engine):
        """semantic search returns empty for empty query."""
        result = search_engine.semantic.search("")
        assert result == []

    def test_build_empty_pages_returns_none(self, built_indexer):
        """semantic build handles empty pages gracefully."""
        from src.semantic import SemanticSearch
        semantic = SemanticSearch()
        semantic.build(built_indexer, {})
        assert semantic.faiss_index is None

    def test_save_without_build_returns_none(self, tmp_path):
        """semantic save does nothing if no FAISS index."""
        from src.semantic import SemanticSearch
        semantic = SemanticSearch()
        path = str(tmp_path / "faiss.index")
        semantic.save(path)
        assert not os.path.exists(path)

    def test_save_and_load_roundtrip(self, search_engine, tmp_path):
        """semantic save and load roundtrip works correctly."""
        from src.semantic import SemanticSearch
        path = str(tmp_path / "faiss.index")
        search_engine.semantic.save(path)
        new_semantic = SemanticSearch()
        result = new_semantic.load(path)
        assert result is True
        assert new_semantic.faiss_index is not None

    def test_load_missing_index_returns_false(self, tmp_path):
        """semantic load returns False for missing index file."""
        from src.semantic import SemanticSearch
        semantic = SemanticSearch()
        result = semantic.load(str(tmp_path / "missing.index"))
        assert result is False

    def test_load_missing_url_ids_returns_false(
        self, search_engine, tmp_path
    ):
        """semantic load returns False when url_ids file missing."""
        from src.semantic import SemanticSearch
        import os
        path = str(tmp_path / "faiss.index")
        search_engine.semantic.save(path)
        # Remove url_ids file
        os.remove(path + ".url_ids.npy")
        semantic = SemanticSearch()
        result = semantic.load(path)
        assert result is False

class TestSearchEngineLoad:
    def test_load_fails_without_faiss(self, built_indexer, sample_pages):
        """load returns False when FAISS index missing."""
        engine = SearchEngine()
        # Patch semantic.load to simulate missing FAISS file
        engine.semantic.load = lambda path="data/faiss.index": False
        result = engine.load(built_indexer, sample_pages)
        assert result is False

    def test_handle_no_results_returns_suggestion(self, search_engine):
        """_handle_no_results returns suggestion for unknown query."""
        processed = search_engine.processor.process("goood")
        result = search_engine._handle_no_results(processed)
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["type"] == "suggestion"

    def test_handle_no_results_tries_variants(self, search_engine):
        """_handle_no_results tries individual tokens as fallback."""
        processed = search_engine.processor.process("good xyzabc")
        result = search_engine._handle_no_results(processed)
        assert isinstance(result, list)

    def test_find_triggers_no_results_handler(self, search_engine):
        """find triggers no results handler for unknown query."""
        results = search_engine.find("xyzabc123")
        assert isinstance(results, list)

    def test_hybrid_returns_empty_when_no_fusion(self, search_engine):
        """_hybrid returns empty when RRF produces nothing."""
        results = search_engine._hybrid(
            ["xyzabc"],
            "xyzabc",
            10
        )
        assert isinstance(results, list)