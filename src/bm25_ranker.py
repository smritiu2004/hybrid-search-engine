"""
bm25_ranker.py
Implements BM25+ keyword ranking over the inverted index.
Initially considered BM25, but BM25+ adds a delta floor to term frequency, 
preventing over-penalisation of terms in longer documents.
"""

import logging
from rank_bm25 import BM25Plus

logger = logging.getLogger(__name__)


class BM25Ranker:
    def __init__(self):
        self.bm25 = None
        # ordered list of url_ids matching bm25 corpus
        self.url_ids = []  
        
        # tokenised pages in same order as url_ids
        self.corpus = []   

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, indexer) -> None:
        """
        This builds the BM25+ model from the indexer.
        Reconstructs tokenised corpus from the inverted index.
        """
        logger.info("Building BM25+ model...")

        # Reconstructs corpus from index
        # Each page becomes a list of tokens (ensuring repetition for frequency)
        page_tokens: dict[int, list[str]] = {}

        for term, postings in indexer.index.items():
            for url_id, stats in postings.items():
                if url_id not in page_tokens:
                    page_tokens[url_id] = []
                # Repeating term by its frequency to reconstruct the token list
                page_tokens[url_id].extend([term] * stats["frequency"])

        # Sorted by url_id for consistent ordering
        sorted_ids = sorted(page_tokens.keys())
        self.url_ids = sorted_ids
        self.corpus = [page_tokens[uid] for uid in sorted_ids]

        # Builds BM25+ model
        self.bm25 = BM25Plus(self.corpus)

        logger.info(f"BM25+ model built over {len(self.corpus)} pages")


    def score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """
        Scores all pages against a query.
        Returns a sorted list of (url_id, score) tuples.
        Highest scores first.
        """
        if self.bm25 is None:
            logger.error("BM25+ model not built. Call build() first.")
            return []

        if not query_tokens:
            logger.warning("Empty query tokens passed to BM25+ scorer.")
            return []

        # Get scores for all pages
        scores = self.bm25.get_scores(query_tokens)

        # Pair each score with its url_id
        scored = [
            (self.url_ids[i], float(scores[i]))
            for i in range(len(scores))
        ]

        # Sorted by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter out zero scores
        scored = [(uid, s) for uid, s in scored if s > 0]

        logger.info(f"BM25+ scored {len(scored)} pages for query: {query_tokens}")
        return scored


    def check_phrase(
        self,
        query_tokens: list[str],
        url_id: int,
        indexer
    ) -> bool:
        """
        Checks if query tokens appear consecutively in a page.
        Uses delta-decoded position lists from the indexer.
        Only relevant for multi-word queries.
        Returns True if phrase found, False otherwise.
        """
        if len(query_tokens) < 2:
            # if its a single word, no phrase check is needed
            return True  

        # Get decoded positions for first token
        first_positions = set(indexer.get_positions(query_tokens[0], url_id))

        if not first_positions:
            return False

        # For each subsequent token, checks if it appears
        # exactly 1 position after the previous token
        for offset, token in enumerate(query_tokens[1:], start=1):
            next_positions = set(indexer.get_positions(token, url_id))

            # Shifts first positions by offset and check overlap
            first_positions = {
                p for p in first_positions
                if p + offset in next_positions
            }
            if not first_positions:
                return False

        return True
    

    def search(
        self,
        query_tokens: list[str],
        indexer,
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Full BM25+ search with optional phrase filtering.
        Returns top_n results as (url_id, score) tuples.
        """
        # Step 1 — gets BM25+ scores
        scored = self.score(query_tokens)

        # Step 2 — phrases filter for multi-word queries
        if len(query_tokens) > 1:
            scored = [
                (uid, s) for uid, s in scored
                if self.check_phrase(query_tokens, uid, indexer)
            ]

        # Step 3 — returns top N
        return scored[:top_n]