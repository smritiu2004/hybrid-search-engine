"""
indexer.py
Builds an inverted index from crawled page content.
Uses delta-encoded compressed posting lists for efficient storage.
Stores word frequency, positions, TF, and TF-IDF per page.
"""

import math
import logging
import spacy
from collections import defaultdict

logger = logging.getLogger(__name__)

# Loading the small English model for tokenisation, lemmatisation, and stopword removal.
nlp = spacy.load("en_core_web_sm")


class Indexer:
    def __init__(self):
        # URL-ID mapping:
        self.url_to_id: dict[str, int] = {}
        self.id_to_url: dict[int, str] = {}

        # Inverted index:
        # { term → { url_id → { "frequency": int, "positions": list, "tf": float } } }
        # Innermost dict stores frequency, positions, TF, and later TF-IDF for each term/page.
        # Middle and Outer maps url_id to stats dict and postings respectively.
        self.index: dict = defaultdict(lambda: defaultdict(lambda: {
            "frequency": 0,
            "positions": [],
            "tf": 0.0,
            "tf_idf": 0.0
        }))

        # Stores total number of words per page for later TF calculation
        self.page_lengths: dict[int, int] = {}
        self.pages: dict[str, str] = {}  

    

    def _register_url(self, url: str) -> int:
        """
        URL Mapping: Assigns a unique integer ID to each URL.
        
        Registers a URL and returns its integer ID.
        If already registered, function returns existing ID.
        """
        if url not in self.url_to_id:
            url_id = len(self.url_to_id)
            self.url_to_id[url] = url_id
            self.id_to_url[url_id] = url
        return self.url_to_id[url]

 

    def _process_text(self, text: str) -> list[str]:
        """
        Text Processing:
        Processes raw text into clean tokens using spaCy.
        Steps:
          1. Lowercase
          2. Tokenise
          3. Remove punctuation and stopwords
          4. Lemmatise
        Returns a list of clean tokens.
        """
        doc = nlp(text.lower())
        tokens = []

        for token in doc:
            # Skips punctuation, spaces, and stopwords
            if token.is_punct or token.is_space or token.is_stop:
                continue
            # Lemmatises and strips whitespaces, stopwords and punctuation.
            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)

        return tokens


    def _delta_encode(self, positions: list[int]) -> list[int]:
        """
        Delta-encoding: used as a method for efficient storage of position lists in the index.
        
        Function:
        Delta encodes a sorted list of positions.
        Stores gaps between positions instead of raw values.
        Example: [4, 17, 53] → [4, 13, 36]
        """
        if not positions:
            return []

        encoded = [positions[0]]
        for i in range(1, len(positions)):
            encoded.append(positions[i] - positions[i - 1])
        return encoded

    def _delta_decode(self, encoded: list[int]) -> list[int]:
        """
        Delta-decoding:reverses the delta encoding process to retrieve original positions.
        
        Function:
        Decodes a delta encoded list back to raw positions.
        Example: [4, 13, 36] → [4, 17, 53]
        """
        if not encoded:
            return []

        decoded = [encoded[0]]
        for i in range(1, len(encoded)):
            decoded.append(decoded[i - 1] + encoded[i])
        return decoded



    def index_page(self, url: str, text: str) -> None:
        """
        Indexes a single page.
        Tokenises text, records positions, frequency, and TF per term.
        """
        url_id = self._register_url(url)
        tokens = self._process_text(text)

        if not tokens:
            logger.warning(f"No tokens extracted from {url}")
            return
        
        self.pages[url] = text
        self.page_lengths[url_id] = len(tokens)

        # Records position of each token
        # enumerate gives us both position and token. 
        # The count and location of the word in the document is recorded here
        for position, token in enumerate(tokens):
            entry = self.index[token][url_id]
            entry["frequency"] += 1
            entry["positions"].append(position)

        # Computes TF for each token in this page
        total_tokens = len(tokens)
        seen_terms = set(tokens)
        for term in seen_terms:
            freq = self.index[term][url_id]["frequency"]
            # TF is term frequency divided by total tokens in the page.
            self.index[term][url_id]["tf"] = freq / total_tokens

        logger.info(f"Indexed {url} — {total_tokens} tokens, {len(seen_terms)} unique terms")


    def build(self, pages: dict[str, str]) -> None:
        """
        Builds the full inverted index from a dict of {url: text}.
        Then computes TF-IDF and delta encodes all position lists.
        """
        logger.info(f"Building index for {len(pages)} pages...")

        # Step 1 — indexing each page
        for url, text in pages.items():
            self.index_page(url, text)

        # Step 2 — computing TF-IDF
        self._compute_tfidf()

        # Step 3 — delta encoding all the position lists
        self._encode_all_positions()

        logger.info("Index build complete.")


    def _compute_tfidf(self) -> None:
        """
        TF-IDF 
        
        Computes TF-IDF for every term/page combination.
        TF-IDF = TF * log(N / df)
        Where:
          N  = total number of pages
          df = number of pages containing the term
        """
        N = len(self.url_to_id)

        for term, postings in self.index.items():
            # number of pages containing term
            df = len(postings)  
            idf = math.log(N / df) if df > 0 else 0.0

            for url_id in postings:
                tf = postings[url_id]["tf"]
                postings[url_id]["tf_idf"] = tf * idf



    def _encode_all_positions(self) -> None:
        """
        Delta encodes all position lists in the index.
        Runs after indexing is complete.
        """
        for term, postings in self.index.items():
            for url_id in postings:
                raw = postings[url_id]["positions"]
                postings[url_id]["positions"] = self._delta_encode(raw)


# helpers for retrieval and debugging

    def get_positions(self, term: str, url_id: int) -> list[int]:
        """
        Returns decoded (raw) positions for a term in a page.
        Decodes delta encoding on the fly.
        """
        if term not in self.index:
            return []
        if url_id not in self.index[term]:
            return []
        encoded = self.index[term][url_id]["positions"]
        return self._delta_decode(encoded)

    def get_postings(self, term: str) -> dict:
        """
        Returns the full postings dict for a term.
        { url_id: { frequency, positions, tf, tf_idf } }
        """
        return dict(self.index.get(term, {}))

    def get_url(self, url_id: int) -> str:
        """Returns the URL for a given integer ID."""
        return self.id_to_url.get(url_id, "")

    def get_url_id(self, url: str) -> int | None:
        """Returns the integer ID for a given URL."""
        return self.url_to_id.get(url, None)