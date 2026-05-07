"""
test_query_processor.py
Tests for the QueryProcessor class.
Covers: processing, tokenisation, lemmatisation,
        stopword removal, phrase detection, edge cases.
"""

import pytest
from src.query_processor import QueryProcessor

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def processor():
    """Returns a fresh QueryProcessor instance."""
    return QueryProcessor()


# -----------------------------------------------------------------------
# Initialisation Tests
# -----------------------------------------------------------------------

class TestQueryProcessorInit:
    def test_processor_has_nlp(self, processor):
        """QueryProcessor initialises with spaCy nlp pipeline."""
        assert processor.nlp is not None


# -----------------------------------------------------------------------
# Process Tests
# -----------------------------------------------------------------------

class TestProcess:
    def test_returns_dict(self, processor):
        """process returns a dictionary."""
        result = processor.process("good friends")
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, processor):
        """process returns dict with all required keys."""
        result = processor.process("good friends")
        assert "raw" in result
        assert "tokens" in result
        assert "is_phrase" in result
        assert "normalised" in result

    def test_raw_preserved(self, processor):
        """process preserves original raw query."""
        result = processor.process("Good Friends!!")
        assert result["raw"] == "Good Friends!!"

    def test_normalised_lowercased(self, processor):
        """process lowercases the normalised query."""
        result = processor.process("GOOD FRIENDS")
        assert result["normalised"] == "good friends"

    def test_normalised_stripped(self, processor):
        """process strips whitespace from normalised query."""
        result = processor.process("  good friends  ")
        assert result["normalised"] == "good friends"

    def test_tokens_are_lowercase(self, processor):
        """All tokens are lowercase."""
        result = processor.process("GOOD FRIENDS")
        for token in result["tokens"]:
            assert token == token.lower()

    def test_stopwords_removed(self, processor):
        """Common stopwords are removed from tokens."""
        result = processor.process("the good and the beautiful")
        assert "the" not in result["tokens"]
        assert "and" not in result["tokens"]

    def test_punctuation_removed(self, processor):
        """Punctuation is removed from tokens."""
        result = processor.process("good friends!!")
        for token in result["tokens"]:
            assert "!" not in token

    def test_lemmatisation_applied(self, processor):
        """Tokens are lemmatised."""
        result = processor.process("friends running choices")
        assert "friend" in result["tokens"] or \
               "friends" not in result["tokens"]

    def test_empty_query_returns_empty_tokens(self, processor):
        """Empty query returns empty token list."""
        result = processor.process("")
        assert result["tokens"] == []

    def test_whitespace_only_returns_empty_tokens(self, processor):
        """Whitespace only query returns empty token list."""
        result = processor.process("   ")
        assert result["tokens"] == []

    def test_none_like_empty_returns_empty(self, processor):
        """None-like empty string handled gracefully."""
        result = processor.process("")
        assert result["is_phrase"] is False


# -----------------------------------------------------------------------
# Phrase Detection Tests
# -----------------------------------------------------------------------

class TestPhraseDetection:
    def test_single_word_not_phrase(self, processor):
        """Single word query is not a phrase."""
        result = processor.process("love")
        assert result["is_phrase"] is False

    def test_multi_word_is_phrase(self, processor):
        """Multi word query is a phrase."""
        result = processor.process("good friends")
        assert result["is_phrase"] is True

    def test_stopwords_only_not_phrase(self, processor):
        """Query with only stopwords produces no phrase."""
        result = processor.process("the and is")
        assert result["is_phrase"] is False

    def test_two_meaningful_words_is_phrase(self, processor):
        """Two meaningful words detected as phrase."""
        result = processor.process("love life")
        assert result["is_phrase"] is True


# -----------------------------------------------------------------------
# Helper Tests
# -----------------------------------------------------------------------

class TestHelpers:
    def test_is_empty_true_for_empty_tokens(self, processor):
        """is_empty returns True for empty token list."""
        processed = processor.process("")
        assert processor.is_empty(processed) is True

    def test_is_empty_false_for_valid_tokens(self, processor):
        """is_empty returns False for valid tokens."""
        processed = processor.process("good friends")
        assert processor.is_empty(processed) is False

    def test_get_tokens_returns_list(self, processor):
        """get_tokens returns a list."""
        processed = processor.process("good friends")
        tokens = processor.get_tokens(processed)
        assert isinstance(tokens, list)

    def test_get_raw_returns_original(self, processor):
        """get_raw returns original query string."""
        processed = processor.process("Good Friends!!")
        assert processor.get_raw(processed) == "Good Friends!!"

    def test_get_normalised_returns_lowercase(self, processor):
        """get_normalised returns lowercased query."""
        processed = processor.process("GOOD FRIENDS")
        assert processor.get_normalised(processed) == "good friends"


# -----------------------------------------------------------------------
# Phrase Variants Tests
# -----------------------------------------------------------------------

class TestPhraseVariants:
    def test_variants_returns_list(self, processor):
        """extract_phrase_variants returns a list."""
        processed = processor.process("good friends")
        variants = processor.extract_phrase_variants(processed)
        assert isinstance(variants, list)

    def test_full_phrase_is_first_variant(self, processor):
        """Full token list is first variant."""
        processed = processor.process("good friends")
        variants = processor.extract_phrase_variants(processed)
        assert variants[0] == processed["tokens"]

    def test_individual_tokens_are_variants(self, processor):
        """Individual tokens appear as fallback variants."""
        processed = processor.process("good friends")
        variants = processor.extract_phrase_variants(processed)
        tokens = processed["tokens"]
        for token in tokens:
            assert [token] in variants

    def test_single_word_has_one_variant(self, processor):
        """Single word query has only one variant."""
        processed = processor.process("love")
        variants = processor.extract_phrase_variants(processed)
        assert len(variants) == 1

    def test_empty_query_returns_empty_variants(self, processor):
        """Empty query returns empty variants list."""
        processed = processor.process("")
        variants = processor.extract_phrase_variants(processed)
        assert variants == []

    def test_three_word_query_has_four_variants(self, processor):
        """Three meaningful word query has full phrase plus 3 singles."""
        processed = processor.process("good friends life")
        variants = processor.extract_phrase_variants(processed)
        # First is full phrase, then one per token
        assert len(variants) == len(processed["tokens"]) + 1