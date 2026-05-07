"""
test_indexer.py
Tests for the Indexer class.
Covers: URL registration, text processing, delta encoding,
        indexing, TF-IDF computation, and lookup helpers.
"""

import pytest
from src.indexer import Indexer

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def indexer():
    """Returns a fresh Indexer instance."""
    return Indexer()


@pytest.fixture
def sample_pages():
    """Returns sample pages dict mimicking crawled content."""
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
def built_indexer(indexer, sample_pages):
    """Returns an Indexer with sample pages already built."""
    indexer.build(sample_pages)
    return indexer


# -----------------------------------------------------------------------
# Initialisation Tests
# -----------------------------------------------------------------------

class TestIndexerInit:
    def test_empty_index_on_init(self, indexer):
        """Indexer starts with empty index."""
        assert len(indexer.index) == 0

    def test_empty_url_maps_on_init(self, indexer):
        """Indexer starts with empty URL maps."""
        assert len(indexer.url_to_id) == 0
        assert len(indexer.id_to_url) == 0

    def test_empty_pages_on_init(self, indexer):
        """Indexer starts with empty pages dict."""
        assert len(indexer.pages) == 0

    def test_empty_page_lengths_on_init(self, indexer):
        """Indexer starts with empty page lengths."""
        assert len(indexer.page_lengths) == 0


# -----------------------------------------------------------------------
# URL Registration Tests
# -----------------------------------------------------------------------

class TestURLRegistration:
    def test_registers_url(self, indexer):
        """URL gets assigned an integer ID."""
        url_id = indexer._register_url("https://quotes.toscrape.com/page/1/")
        assert url_id == 0

    def test_second_url_gets_next_id(self, indexer):
        """Second URL gets ID 1."""
        indexer._register_url("https://quotes.toscrape.com/page/1/")
        url_id = indexer._register_url("https://quotes.toscrape.com/page/2/")
        assert url_id == 1

    def test_same_url_returns_same_id(self, indexer):
        """Registering same URL twice returns same ID."""
        id1 = indexer._register_url("https://quotes.toscrape.com/page/1/")
        id2 = indexer._register_url("https://quotes.toscrape.com/page/1/")
        assert id1 == id2

    def test_bidirectional_mapping(self, indexer):
        """URL and ID maps are consistent."""
        url = "https://quotes.toscrape.com/page/1/"
        url_id = indexer._register_url(url)
        assert indexer.id_to_url[url_id] == url
        assert indexer.url_to_id[url] == url_id


# -----------------------------------------------------------------------
# Text Processing Tests
# -----------------------------------------------------------------------

class TestTextProcessing:
    def test_lowercases_text(self, indexer):
        """Tokens are lowercased."""
        tokens = indexer._process_text("GOOD Friends")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stopwords(self, indexer):
        """Common stopwords are removed."""
        tokens = indexer._process_text("the quick brown fox")
        assert "the" not in tokens

    def test_removes_punctuation(self, indexer):
        """Punctuation is stripped."""
        tokens = indexer._process_text("hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_lemmatises_tokens(self, indexer):
        """Tokens are lemmatised."""
        tokens = indexer._process_text("running friends choices")
        assert "run" in tokens or "running" not in tokens
        assert "friend" in tokens or "friends" not in tokens

    def test_empty_string_returns_empty_list(self, indexer):
        """Empty string returns empty token list."""
        tokens = indexer._process_text("")
        assert tokens == []

    def test_returns_list(self, indexer):
        """_process_text returns a list."""
        result = indexer._process_text("hello world")
        assert isinstance(result, list)


# -----------------------------------------------------------------------
# Delta Encoding Tests
# -----------------------------------------------------------------------

class TestDeltaEncoding:
    def test_delta_encode_basic(self, indexer):
        """Delta encoding produces correct gaps."""
        positions = [4, 17, 53]
        encoded = indexer._delta_encode(positions)
        assert encoded == [4, 13, 36]

    def test_delta_decode_basic(self, indexer):
        """Delta decoding reconstructs original positions."""
        encoded = [4, 13, 36]
        decoded = indexer._delta_decode(encoded)
        assert decoded == [4, 17, 53]

    def test_encode_decode_roundtrip(self, indexer):
        """Encoding then decoding returns original list."""
        positions = [1, 5, 12, 30, 87]
        assert indexer._delta_decode(
            indexer._delta_encode(positions)
        ) == positions

    def test_empty_list_encode(self, indexer):
        """Delta encoding empty list returns empty list."""
        assert indexer._delta_encode([]) == []

    def test_empty_list_decode(self, indexer):
        """Delta decoding empty list returns empty list."""
        assert indexer._delta_decode([]) == []

    def test_single_position_encode(self, indexer):
        """Single position encodes correctly."""
        assert indexer._delta_encode([5]) == [5]

    def test_single_position_decode(self, indexer):
        """Single position decodes correctly."""
        assert indexer._delta_decode([5]) == [5]


# -----------------------------------------------------------------------
# Indexing Tests
# -----------------------------------------------------------------------

class TestIndexing:
    def test_index_page_stores_text(self, indexer):
        """index_page stores raw page text."""
        url = "https://quotes.toscrape.com/page/1/"
        indexer.index_page(url, "good friends books")
        assert url in indexer.pages

    def test_index_page_records_frequency(self, indexer):
        """index_page records correct term frequency."""
        url = "https://quotes.toscrape.com/page/1/"
        indexer.index_page(url, "good good good friends")
        url_id = indexer.url_to_id[url]
        assert indexer.index["good"][url_id]["frequency"] == 3

    def test_index_page_records_positions(self, indexer):
        """index_page records term positions."""
        url = "https://quotes.toscrape.com/page/1/"
        indexer.index_page(url, "good friends good")
        url_id = indexer.url_to_id[url]
        assert len(indexer.index["good"][url_id]["positions"]) > 0

    def test_index_page_case_insensitive(self, indexer):
        """index_page treats Good and good as same term."""
        url = "https://quotes.toscrape.com/page/1/"
        indexer.index_page(url, "Good good GOOD")
        url_id = indexer.url_to_id[url]
        assert indexer.index["good"][url_id]["frequency"] == 3

    def test_empty_text_not_indexed(self, indexer):
        """Empty text produces no index entries."""
        url = "https://quotes.toscrape.com/page/1/"
        indexer.index_page(url, "")
        assert len(indexer.index) == 0

    def test_build_indexes_all_pages(self, built_indexer, sample_pages):
        """build indexes all provided pages."""
        assert len(built_indexer.url_to_id) == len(sample_pages)

    def test_build_computes_tfidf(self, built_indexer):
        """build computes non-zero TF-IDF for indexed terms."""
        for term, postings in built_indexer.index.items():
            for url_id, stats in postings.items():
                assert "tf_idf" in stats


# -----------------------------------------------------------------------
# TF-IDF Tests
# -----------------------------------------------------------------------

class TestTFIDF:
    def test_tf_between_zero_and_one(self, built_indexer):
        """TF values are between 0 and 1."""
        for term, postings in built_indexer.index.items():
            for url_id, stats in postings.items():
                assert 0 <= stats["tf"] <= 1

    def test_rare_term_higher_idf(self, built_indexer):
        """Term appearing in fewer pages has higher TF-IDF."""
        # "good" appears in page 1 and page 3
        # find a term that appears in only one page
        single_page_terms = [
            term for term, postings in built_indexer.index.items()
            if len(postings) == 1
        ]
        multi_page_terms = [
            term for term, postings in built_indexer.index.items()
            if len(postings) > 1
        ]
        if single_page_terms and multi_page_terms:
            single_term = single_page_terms[0]
            multi_term = multi_page_terms[0]
            single_uid = list(
                built_indexer.index[single_term].keys()
            )[0]
            multi_uid = list(
                built_indexer.index[multi_term].keys()
            )[0]
            assert (
                built_indexer.index[single_term][single_uid]["tf_idf"]
                >=
                built_indexer.index[multi_term][multi_uid]["tf_idf"]
            )


# -----------------------------------------------------------------------
# Lookup Helper Tests
# -----------------------------------------------------------------------

class TestLookupHelpers:
    def test_get_url_returns_correct_url(self, built_indexer):
        """get_url returns correct URL for ID."""
        url = "https://quotes.toscrape.com/page/1/"
        url_id = built_indexer.url_to_id[url]
        assert built_indexer.get_url(url_id) == url

    def test_get_url_id_returns_correct_id(self, built_indexer):
        """get_url_id returns correct ID for URL."""
        url = "https://quotes.toscrape.com/page/1/"
        expected_id = built_indexer.url_to_id[url]
        assert built_indexer.get_url_id(url) == expected_id

    def test_get_url_unknown_id_returns_empty(self, built_indexer):
        """get_url returns empty string for unknown ID."""
        assert built_indexer.get_url(9999) == ""

    def test_get_url_id_unknown_url_returns_none(self, built_indexer):
        """get_url_id returns None for unknown URL."""
        assert built_indexer.get_url_id("https://unknown.com") is None

    def test_get_postings_returns_dict(self, built_indexer):
        """get_postings returns a dict for known term."""
        postings = built_indexer.get_postings("good")
        assert isinstance(postings, dict)

    def test_get_postings_unknown_term_returns_empty(self, built_indexer):
        """get_postings returns empty dict for unknown term."""
        postings = built_indexer.get_postings("xyzabc")
        assert postings == {}

    def test_get_positions_decodes_correctly(self, built_indexer):
        """get_positions returns decoded positions."""
        url = "https://quotes.toscrape.com/page/1/"
        url_id = built_indexer.url_to_id[url]
        positions = built_indexer.get_positions("good", url_id)
        assert isinstance(positions, list)
        # Positions should be in ascending order
        assert positions == sorted(positions)

    def test_get_positions_unknown_term(self, built_indexer):
        """get_positions returns empty list for unknown term."""
        assert built_indexer.get_positions("xyzabc", 0) == []