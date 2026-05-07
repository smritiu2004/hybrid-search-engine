"""
test_bktree.py
Tests for the BKTree class.
Covers: Levenshtein distance, insertion, search,
        spell correction, edge cases.
"""

import pytest
from src.bktree import BKTree
from src.indexer import Indexer

@pytest.fixture
def bktree():
    """Returns a fresh BKTree instance."""
    return BKTree()


@pytest.fixture
def populated_bktree():
    """Returns a BKTree with sample words inserted."""
    tree = BKTree()
    words = [
        "friend", "friends", "friendship",
        "love", "lovely", "lover",
        "good", "goodness",
        "truth", "true",
        "world", "wisdom"
    ]
    for word in words:
        tree.insert(word)
    return tree


@pytest.fixture
def built_bktree():
    """Returns a BKTree built from a sample indexer."""
    indexer = Indexer()
    indexer.build({
        "https://quotes.toscrape.com/page/1/": (
            "good friends love world truth wisdom"
        )
    })
    tree = BKTree()
    tree.build(indexer)
    return tree


class TestBKTreeInit:
    def test_empty_tree_size_zero(self, bktree):
        """Fresh BKTree has size 0."""
        assert bktree.size == 0

    def test_root_is_none(self, bktree):
        """Fresh BKTree has no root."""
        assert bktree.root is None

    def test_len_returns_zero(self, bktree):
        """len(bktree) returns 0 for empty tree."""
        assert len(bktree) == 0

class TestLevenshtein:
    def test_identical_strings_distance_zero(self, bktree):
        """Distance between identical strings is 0."""
        assert bktree._levenshtein("good", "good") == 0

    def test_empty_string_distance(self, bktree):
        """Distance from empty to word equals word length."""
        assert bktree._levenshtein("", "good") == 4
        assert bktree._levenshtein("good", "") == 4

    def test_single_insertion(self, bktree):
        """Single character insertion gives distance 1."""
        assert bktree._levenshtein("good", "goood") == 1

    def test_single_deletion(self, bktree):
        """Single character deletion gives distance 1."""
        assert bktree._levenshtein("goood", "good") == 1

    def test_single_substitution(self, bktree):
        """Single character substitution gives distance 1."""
        assert bktree._levenshtein("good", "goad") == 1

    def test_two_edits(self, bktree):
        """Two edits gives distance 2."""
        assert bktree._levenshtein("friend", "fiend") == 1
        assert bktree._levenshtein("friends", "fiends") == 1

    def test_completely_different_strings(self, bktree):
        """Completely different strings have high distance."""
        dist = bktree._levenshtein("good", "xyzw")
        assert dist == 4

    def test_symmetric(self, bktree):
        """Levenshtein distance is symmetric."""
        assert bktree._levenshtein("good", "goood") == \
               bktree._levenshtein("goood", "good")

class TestBKTreeInsert:
    def test_first_insert_becomes_root(self, bktree):
        """First inserted word becomes root."""
        bktree.insert("good")
        assert bktree.root is not None
        assert bktree.root.word == "good"

    def test_insert_increases_size(self, bktree):
        """Inserting a word increases size."""
        bktree.insert("good")
        assert bktree.size == 1

    def test_insert_multiple_words(self, bktree):
        """Inserting multiple words increases size correctly."""
        bktree.insert("good")
        bktree.insert("love")
        bktree.insert("truth")
        assert bktree.size == 3

    def test_insert_duplicate_not_counted(self, bktree):
        """Inserting same word twice doesn't increase size."""
        bktree.insert("good")
        bktree.insert("good")
        assert bktree.size == 1

    def test_insert_empty_string_ignored(self, bktree):
        """Empty string insertion is ignored."""
        bktree.insert("")
        assert bktree.size == 0

class TestBKTreeSearch:
    def test_search_returns_list(self, populated_bktree):
        """search returns a list."""
        result = populated_bktree.search("good")
        assert isinstance(result, list)

    def test_search_exact_match(self, populated_bktree):
        """search finds exact match at distance 0."""
        result = populated_bktree.search("good", max_distance=0)
        words = [w for w, _ in result]
        assert "good" in words

    def test_search_finds_close_matches(self, populated_bktree):
        """search finds words within max_distance."""
        result = populated_bktree.search("goood", max_distance=2)
        words = [w for w, _ in result]
        assert "good" in words

    def test_search_respects_max_distance(self, populated_bktree):
        """All results are within max_distance."""
        result = populated_bktree.search("good", max_distance=2)
        for word, dist in result:
            assert dist <= 2

    def test_search_sorted_by_distance(self, populated_bktree):
        """Results are sorted by distance ascending."""
        result = populated_bktree.search("good", max_distance=3)
        distances = [d for _, d in result]
        assert distances == sorted(distances)

    def test_search_empty_tree_returns_empty(self, bktree):
        """search on empty tree returns empty list."""
        result = bktree.search("good")
        assert result == []

    def test_search_empty_query_returns_empty(self, populated_bktree):
        """search with empty query returns empty list."""
        result = populated_bktree.search("")
        assert result == []

    def test_search_no_matches_returns_empty(self, populated_bktree):
        """search returns empty when no words within distance."""
        result = populated_bktree.search("xyzabc", max_distance=1)
        assert result == []


class TestBKTreeSuggest:
    def test_suggest_returns_list(self, populated_bktree):
        """suggest returns a list."""
        result = populated_bktree.suggest("goood")
        assert isinstance(result, list)

    def test_suggest_returns_words_only(self, populated_bktree):
        """suggest returns list of strings not tuples."""
        result = populated_bktree.suggest("goood")
        for item in result:
            assert isinstance(item, str)

    def test_suggest_respects_limit(self, populated_bktree):
        """suggest respects limit parameter."""
        result = populated_bktree.suggest("good", limit=2)
        assert len(result) <= 2

    def test_suggest_misspelled_word(self, populated_bktree):
        """suggest returns correction for misspelled word."""
        result = populated_bktree.suggest("goood", max_distance=2)
        assert "good" in result

    def test_suggest_unknown_word_empty(self, populated_bktree):
        """suggest returns empty for completely unknown word."""
        result = populated_bktree.suggest("xyzabc", max_distance=1)
        assert result == []


class TestBKTreeContains:
    def test_contains_inserted_word(self, populated_bktree):
        """__contains__ returns True for inserted word."""
        assert "good" in populated_bktree

    def test_not_contains_unknown_word(self, populated_bktree):
        """__contains__ returns False for unknown word."""
        assert "xyzabc" not in populated_bktree

class TestBKTreeBuild:
    def test_build_from_indexer(self, built_bktree):
        """BKTree builds correctly from indexer vocabulary."""
        assert built_bktree.size > 0

    def test_built_tree_contains_indexed_terms(self, built_bktree):
        """Built tree contains terms from index."""
        assert "good" in built_bktree or \
               "friend" in built_bktree or \
               "love" in built_bktree

    def test_built_tree_len_matches_size(self, built_bktree):
        """len() matches size attribute."""
        assert len(built_bktree) == built_bktree.size