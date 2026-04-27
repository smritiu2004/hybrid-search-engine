"""
hybrid.py
Implements Reciprocal Rank Fusion (RRF) to combine
BM25+ and semantic search ranked lists.
"""

import logging

logger = logging.getLogger(__name__)

# RRF constant
# Using value 60 from Cormack et al. 2009 paper.
RRF_K = 60


class HybridSearch:
    def __init__(self, k: int = RRF_K):
        """
        k — RRF constant. Higher k reduces impact of top rankings.
        """
        self.k = k

    
    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[tuple[int, float]]]
    ) -> dict[int, float]:
        """
        Fuses multiple ranked lists using RRF.
        For each document in each list:
            RRF score += 1 / (k + rank)
        Where rank starts at 1.
        Returns dict of {url_id: rrf_score}.
        """
        rrf_scores: dict[int, float] = {}

        for ranked_list in ranked_lists:
            # For each document in each ranked list we add 1/(k + rank) to its running total.
            for rank, (url_id, _) in enumerate(ranked_list, start=1):
                if url_id not in rrf_scores:
                    rrf_scores[url_id] = 0.0
                rrf_scores[url_id] += 1.0 / (self.k + rank)

        return rrf_scores


    def search(
        self,
        bm25_results: list[tuple[int, float]],
        semantic_results: list[tuple[int, float]],
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Combines BM25+ and semantic results using RRF.
        Returns top_n fused results as (url_id, rrf_score) tuples.
        Sorted by RRF score descending.
        """
        if not bm25_results and not semantic_results:
            logger.warning("Both BM25+ and semantic results are empty.")
            return []

        # Handles cases where one list is empty
        if not bm25_results:
            logger.warning("BM25+ results empty — using semantic only.")
            return semantic_results[:top_n]

        # Handles cases where one list is empty
        if not semantic_results:
            logger.warning("Semantic results empty — using BM25+ only.")
            return bm25_results[:top_n]

        logger.info(
            f"Fusing {len(bm25_results)} BM25+ results "
            f"and {len(semantic_results)} semantic results with RRF..."
        )

        # Step 1 — compute RRF scores
        rrf_scores = self._reciprocal_rank_fusion([
            bm25_results,
            semantic_results
        ])

        # Step 2 — sort by RRF score descending in order to get final ranking
        fused = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Step 3 — return top N
        results = fused[:top_n]

        logger.info(f"RRF fusion complete — {len(results)} results returned")
        return results


    def explain(
        self,
        url_id: int,
        bm25_results: list[tuple[int, float]],
        semantic_results: list[tuple[int, float]],
        indexer
    ) -> dict:
        """
        Explains the RRF score breakdown for a specific page.
        Returns dict with BM25+ rank/score, semantic rank/score,
        and final RRF score.
        """
        # Finding BM25+ rank and score
        bm25_rank = None
        bm25_score = None
        for rank, (uid, score) in enumerate(bm25_results, start=1):
            if uid == url_id:
                bm25_rank = rank
                bm25_score = score
                break

        # Finding semantic rank and score
        semantic_rank = None
        semantic_score = None
        for rank, (uid, score) in enumerate(semantic_results, start=1):
            if uid == url_id:
                semantic_rank = rank
                semantic_score = score
                break

        # Computing individual RRF contributions
        bm25_contribution = (
            1.0 / (self.k + bm25_rank) if bm25_rank else 0.0
        )
        semantic_contribution = (
            1.0 / (self.k + semantic_rank) if semantic_rank else 0.0
        )
        rrf_score = bm25_contribution + semantic_contribution

        return {
            "url": indexer.get_url(url_id),
            "bm25_rank": bm25_rank,
            "bm25_score": bm25_score,
            "bm25_rrf_contribution": round(bm25_contribution, 6),
            "semantic_rank": semantic_rank,
            "semantic_score": semantic_score,
            "semantic_rrf_contribution": round(semantic_contribution, 6),
            "final_rrf_score": round(rrf_score, 6)
        }