"""
query_processor.py
Processes raw user queries before passing to retrieval pipeline.
Steps:
  1. Lowercase
  2. Tokenise
  3. Strip punctuation
  4. Remove stopwords
  5. Lemmatise (spaCy)
  6. Detect phrase queries
"""

import logging
import spacy

logger = logging.getLogger(__name__)

# Loading spaCy English model
nlp = spacy.load("en_core_web_sm")


class QueryProcessor:
    def __init__(self):
        self.nlp = nlp


    def process(self, raw_query: str) -> dict:
        """
        Processes a raw query string into clean tokens.
        Returns a dict with:
          - raw        : original query string
          - tokens     : processed token list
          - is_phrase  : whether query is multi-word
          - normalised : lowercased stripped query string
        """
        
        # Handling empty or whitespace-only queries
        if not raw_query or not raw_query.strip():
            logger.warning("Empty query received.")
            return {
                "raw": raw_query,
                "tokens": [],
                "is_phrase": False,
                "normalised": ""
            }

        # Step 1: lowercase and strip whitespace
        normalised = raw_query.lower().strip()

        # Step 2: run through spaCy pipeline
        doc = self.nlp(normalised)

        # Step 3: filter and lemmatise
        tokens = []
        for token in doc:
    
            # Skipping punctuation, spaces, stopwords
            if token.is_punct or token.is_space or token.is_stop:
                continue
            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)

        # Step 4: detecting phrase queries
        is_phrase = len(tokens) > 1

        logger.info(
            f"Query processed: '{raw_query}' → "
            f"tokens={tokens}, is_phrase={is_phrase}"
        )

        return {
            "raw": raw_query,
            "tokens": tokens,
            "is_phrase": is_phrase,
            "normalised": normalised
        }


    # --- Helpers ---
    def is_empty(self, processed: dict) -> bool:
        """
        Returns True if processed query has no tokens.
        """
        return len(processed["tokens"]) == 0

    def get_tokens(self, processed: dict) -> list[str]:
        """
        Returns token list from processed query dict.
        """
        return processed["tokens"]

    def get_raw(self, processed: dict) -> str:
        """
        Returns original raw query string.
        Used for Cross-Encoder which prefers natural language.
        """
        return processed["raw"]

    def get_normalised(self, processed: dict) -> str:
        """
        Returns normalised query string.
        Used for Trie and BK-Tree lookups.
        """
        return processed["normalised"]


    def extract_phrase_variants(self, processed: dict) -> list[list[str]]:
        """
        For a multi-word query, generates search variants.
        Example: "good friends" →
          [["good", "friend"],   ← both terms (phrase)
           ["good"],             ← first term only
           ["friend"]]           ← second term only
        Allows fallback if exact phrase not found.
        """
        tokens = processed["tokens"]

        if not tokens:
            return []

        variants = [tokens]  # full phrase first

        # Adds individual token variants as fallback
        if len(tokens) > 1:
            for token in tokens:
                variants.append([token])

        return variants