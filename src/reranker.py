"""
reranker.py
Implements Cross-Encoder re-ranking on top of hybrid search results.
Cross-Encoder takes query and document together as input,
producing a more accurate relevance score than Bi-Encoder alone.
Used as a final precision layer on top N results.
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# The Cross-Encoder model
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        logger.info(f"Loading Cross-Encoder model: {MODEL_NAME}")
        self.model = CrossEncoder(MODEL_NAME)


    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        indexer,
        pages: dict[str, str],
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Re-ranks candidate results using Cross-Encoder.
        Accepts:
          query      — raw query string
          candidates — list of (url_id, score) from hybrid search
          indexer    — for url_id to URL mapping
          pages      — dict of {url: text} for document content
          top_n      — number of results to return
        Returns re-ranked list of (url_id, score) tuples.
        """
        if not candidates:
            logger.warning("No candidates to rerank.")
            return []

        if not query.strip():
            logger.warning("Empty query passed to reranker.")
            return []

        logger.info(f"Re-ranking {len(candidates)} candidates...")

        # Step 1 — building (query, document) pairs for the Cross-Encoder
        pairs = []
        valid_candidates = []

        # Tracking valid candidates that have corresponding text for scoring
        for url_id, _ in candidates:
            url = indexer.get_url(url_id)
            text = pages.get(url, "")

            if not text:
                logger.warning(f"No text found for url_id {url_id}, skipping.")
                continue

            pairs.append([query, text])
            valid_candidates.append(url_id)

        if not pairs:
            logger.error("No valid query-document pairs to rerank.")
            return candidates

        # Scoring all pairs with Cross-Encoder
        scores = self.model.predict(pairs)

        # Pairing url_ids with new scores
        reranked = [
            (valid_candidates[i], float(scores[i]))
            for i in range(len(valid_candidates))
        ]

        # Sorting by new score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Re-ranking complete. Top result url_id: {reranked[0][0]}")
        return reranked[:top_n]