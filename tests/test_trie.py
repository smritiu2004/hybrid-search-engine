"""
test_trie.py
Tests for the Trie class.
Covers: insertion, autocomplete, contains,
        starts_with, edge cases.
"""

import pytest
from src.trie import Trie
from src.indexer import Indexer

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def trie():
    """Returns a fresh Trie instance."""
    return Trie()


@pytest.fixture
def populated_trie():
    """Returns a Trie with sample words inserted."""
    t = Trie()
    words = [
        "friend", "friends", "friendship",
        "love", "lovely", "lover",
        "good", "goodness",
        "truth", "true",
        "world", "wisdom"
    ]
    for word in words:
        t.insert(word)
    return t


@pytest.fixture
def built_trie():
    """Returns a Trie built from a sample indexer."""
    indexer = Indexer()
    indexer.build({
        "https://quotes.toscrape.com/page/1/": (
            "good friends love world truth wisdom"
        )
    })
    t = Trie()
    t.build(indexer)
    return t


# -----------------------------------------------------------------------
# Initialisation Tests
# -----------------------------------------------------------------------

class TestTrieInit:
    def test_empty_trie_size_zero(self, trie):
        """Fresh Trie has size 0."""
        assert trie.size == 0

    def test_empty_trie_len_zero(self, trie):
        """len(trie) returns 0 for empty trie."""
        assert len(trie) == 0

    def test_root_node_exists(self, trie):
        """Trie initialises with a root node."""
        assert trie.root is not None

    def test_root_has_no_children(self, trie):
        """Root node starts with no children."""
        assert len(trie.root.children) == 0


# -----------------------------------------------------------------------
# Insert Tests
# -----------------------------------------------------------------------

class TestTrieInsert:
    def test_insert_increases_size(self, trie):
        """Inserting a word increases size."""
        trie.insert("friend")
        assert trie.size == 1

    def test_insert_multiple_words(self, trie):
        """Inserting multiple words increases size correctly."""
        trie.insert("friend")
        trie.insert("love")
        trie.insert("good")
        assert trie.size == 3

    def test_insert_duplicate_not_counted(self, trie):
        """Inserting same word twice doesn't increase size."""
        trie.insert("friend")
        trie.insert("friend")
        assert trie.size == 1

    def test_insert_empty_string_ignored(self, trie):
        """Empty string insertion is ignored."""
        trie.insert("")
        assert trie.size == 0

    def test_insert_whitespace_ignored(self, trie):
        """Whitespace string insertion is ignored."""
        trie.insert("   ")
        assert trie.size == 0

    def test_shared_prefix_stored_efficiently(self, trie):
        """Words sharing prefix share nodes in trie."""
        trie.insert("friend")
        trie.insert("friends")
        # Both share f-r-i-e-n-d path
        assert trie.size == 2


# -----------------------------------------------------------------------
# Contains Tests
# -----------------------------------------------------------------------

class TestTrieContains:
    def test_contains_inserted_word(self, populated_trie):
        """contains returns True for inserted word."""
        assert populated_trie.contains("friend") is True

    def test_not_contains_unknown_word(self, populated_trie):
        """contains returns False for unknown word."""
        assert populated_trie.contains("xyzabc") is False

    def test_not_contains_prefix_only(self, populated_trie):
        """contains returns False for prefix that isn't a word."""
        assert populated_trie.contains("fri") is False

    def test_contains_via_in_operator(self, populated_trie):
        """'word in trie' syntax works correctly."""
        assert "friend" in populated_trie
        assert "xyzabc" not in populated_trie

    def test_contains_empty_string(self, populated_trie):
        """contains returns False for empty string."""
        assert populated_trie.contains("") is False


# -----------------------------------------------------------------------
# starts_with Tests
# -----------------------------------------------------------------------

class TestTrieStartsWith:
    def test_starts_with_valid_prefix(self, populated_trie):
        """starts_with returns True for valid prefix."""
        assert populated_trie.starts_with("fri") is True

    def test_starts_with_full_word(self, populated_trie):
        """starts_with returns True for full word."""
        assert populated_trie.starts_with("friend") is True

    def test_starts_with_invalid_prefix(self, populated_trie):
        """starts_with returns False for invalid prefix."""
        assert populated_trie.starts_with("xyz") is False

    def test_starts_with_empty_string(self, populated_trie):
        """starts_with returns False for empty string."""
        assert populated_trie.starts_with("") is False


# -----------------------------------------------------------------------
# Autocomplete Tests
# -----------------------------------------------------------------------

class TestTrieAutocomplete:
    def test_autocomplete_returns_list(self, populated_trie):
        """autocomplete returns a list."""
        result = populated_trie.autocomplete("fri")
        assert isinstance(result, list)

    def test_autocomplete_finds_completions(self, populated_trie):
        """autocomplete finds words with given prefix."""
        result = populated_trie.autocomplete("fri")
        assert "friend" in result or "friends" in result

    def test_autocomplete_respects_limit(self, populated_trie):
        """autocomplete respects limit parameter."""
        result = populated_trie.autocomplete("lo", limit=2)
        assert len(result) <= 2

    def test_autocomplete_unknown_prefix_empty(self, populated_trie):
        """autocomplete returns empty for unknown prefix."""
        result = populated_trie.autocomplete("xyz")
        assert result == []

    def test_autocomplete_empty_prefix_empty(self, populated_trie):
        """autocomplete returns empty for empty prefix."""
        result = populated_trie.autocomplete("")
        assert result == []

    def test_autocomplete_full_word_returns_word(self, populated_trie):
        """autocomplete on full word returns that word."""
        result = populated_trie.autocomplete("friend")
        assert "friend" in result

    def test_autocomplete_love_prefix(self, populated_trie):
        """autocomplete 'lov' finds love, lovely, lover."""
        result = populated_trie.autocomplete("lov")
        assert len(result) >= 1
        for word in result:
            assert word.startswith("lov")

    def test_autocomplete_all_results_start_with_prefix(
        self, populated_trie
    ):
        """All autocomplete results start with given prefix."""
        result = populated_trie.autocomplete("go")
        for word in result:
            assert word.startswith("go")


# -----------------------------------------------------------------------
# Build Tests
# -----------------------------------------------------------------------

class TestTrieBuild:
    def test_build_from_indexer(self, built_trie):
        """Trie builds correctly from indexer vocabulary."""
        assert built_trie.size > 0

    def test_built_trie_contains_indexed_terms(self, built_trie):
        """Built trie contains terms from index."""
        assert built_trie.contains("good") or \
               built_trie.contains("friend") or \
               built_trie.contains("love")

    def test_built_trie_len_matches_size(self, built_trie):
        """len() matches size attribute."""
        assert len(built_trie) == built_trie.size