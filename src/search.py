"""
search.py
Facade that coordinates the full retrieval pipeline.
Ties together BM25+, semantic search, RRF fusion,
Cross-Encoder reranking, and query processing.
"""

import logging

from src.bm25_ranker import BM25Ranker
from src.semantic import SemanticSearch
from src.hybrid import HybridSearch
from src.reranker import Reranker
from src.query_processor import QueryProcessor
from src.trie import Trie
from src.bktree import BKTree

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self):
        self.bm25 = BM25Ranker()
        self.semantic = SemanticSearch()
        self.hybrid = HybridSearch()
        self.reranker = Reranker()
        self.processor = QueryProcessor()
        self.trie = Trie()
        self.bktree = BKTree()

        # Set after build/load
        self.indexer = None
        self.pages = None
        self.is_ready = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, indexer, pages: dict[str, str]) -> None:
        """
        Builds all search components from crawled pages and index.
        Called after crawler and indexer have run.
        """
        logger.info("Building search engine components...")

        self.indexer = indexer
        self.pages = pages

        # Build BM25+ model
        logger.info("Building BM25+ model...")
        self.bm25.build(indexer)

        # Build FAISS semantic index
        logger.info("Building semantic index...")
        self.semantic.build(indexer, pages)

        # Build Trie for autocomplete
        logger.info("Building Trie...")
        self.trie.build(indexer)

        # Build BK-Tree for spell correction
        logger.info("Building BK-Tree...")
        self.bktree.build(indexer)

        self.is_ready = True
        logger.info("Search engine ready.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, indexer, pages: dict[str, str]) -> bool:
        """
        Loads search components after index has been loaded from disk.
        Rebuilds BM25+, Trie, BK-Tree from loaded index.
        Loads FAISS index from disk.
        Returns True if successful.
        """
        logger.info("Loading search engine components...")

        self.indexer = indexer
        self.pages = pages

        # Rebuild BM25+ from loaded index
        logger.info("Rebuilding BM25+ model...")
        self.bm25.build(indexer)

        # Load FAISS index from disk
        logger.info("Loading FAISS index...")
        if not self.semantic.load():
            logger.error("Failed to load FAISS index.")
            return False

        # Rebuild Trie from loaded index
        logger.info("Rebuilding Trie...")
        self.trie.build(indexer)

        # Rebuild BK-Tree from loaded index
        logger.info("Rebuilding BK-Tree...")
        self.bktree.build(indexer)

        self.is_ready = True
        logger.info("Search engine loaded and ready.")
        return True

    # ------------------------------------------------------------------
    # Find
    # ------------------------------------------------------------------

    def find(
        self,
        raw_query: str,
        top_n: int = 10,
        mode: str = "hybrid"
    ) -> list[dict]:
        """
        Full retrieval pipeline for a user query.
        Modes:
          hybrid   — BM25+ + semantic + RRF + reranker (default)
          bm25     — BM25+ only
          semantic — semantic only
        Returns list of result dicts with url and score.
        """
        if not self.is_ready:
            logger.error("Search engine not ready. Run build or load first.")
            return []

        # Step 1 — process query
        processed = self.processor.process(raw_query)

        if self.processor.is_empty(processed):
            logger.warning("Query is empty after processing.")
            return []

        tokens = self.processor.get_tokens(processed)
        raw = self.processor.get_raw(processed)

        logger.info(f"Searching for: {tokens} (mode={mode})")

        # Step 2 — retrieval based on mode
        if mode == "bm25":
            results = self._bm25_only(tokens)
        elif mode == "semantic":
            results = self._semantic_only(raw)
        else:
            results = self._hybrid(tokens, raw, top_n)

        # Step 3 — handle no results
        if not results:
            return self._handle_no_results(processed)

        # Step 4 — format results
        return self._format_results(results)

    # ------------------------------------------------------------------
    # Retrieval Modes
    # ------------------------------------------------------------------

    def _bm25_only(
        self,
        tokens: list[str],
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """BM25+ only retrieval."""
        return self.bm25.search(tokens, self.indexer, top_n)

    def _semantic_only(
        self,
        raw: str,
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """Semantic only retrieval."""
        return self.semantic.search(raw, top_n)

    def _hybrid(
        self,
        tokens: list[str],
        raw: str,
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Full hybrid pipeline:
          BM25+ → semantic → RRF → Cross-Encoder reranker
        """
        # BM25+ results
        bm25_results = self.bm25.search(tokens, self.indexer, top_n)

        # Semantic results
        semantic_results = self.semantic.search(raw, top_n)

        # RRF fusion
        fused = self.hybrid.search(bm25_results, semantic_results, top_n)

        if not fused:
            return []

        # Cross-Encoder reranking
        reranked = self.reranker.rerank(
            query=raw,
            candidates=fused,
            indexer=self.indexer,
            pages=self.pages,
            top_n=top_n
        )

        return reranked

    # ------------------------------------------------------------------
    # No Results Handler
    # ------------------------------------------------------------------

    def _handle_no_results(self, processed: dict) -> list[dict]:
        """
        Called when retrieval returns no results.
        Tries phrase variants first, then suggests corrections.
        Returns list of suggestion dicts or empty list.
        """
        tokens = self.processor.get_tokens(processed)
        normalised = self.processor.get_normalised(processed)

        # Try phrase variants (fallback to individual terms)
        variants = self.processor.extract_phrase_variants(processed)

        # Skip first variant (full phrase) — already tried
        for variant in variants[1:]:
            results = self.bm25.search(variant, self.indexer)
            if results:
                logger.info(f"Found results using variant: {variant}")
                return self._format_results(results)

        # No results from any variant — suggest corrections
        suggestions = []

        for token in tokens:
            # BK-Tree spell correction
            corrections = self.bktree.suggest(token, max_distance=2)
            if corrections:
                suggestions.extend(corrections)

            # Trie autocomplete
            completions = self.trie.autocomplete(token, limit=3)
            if completions:
                suggestions.extend(completions)

        # Deduplicate suggestions
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return [{
            "type": "suggestion",
            "message": f"No results found for '{normalised}'.",
            "suggestions": unique_suggestions[:5]
        }]

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    def print_term(self, term: str) -> dict | None:
        """
        Returns inverted index entry for a term.
        Used by the print command in CLI.
        Decodes delta-encoded positions for display.
        """
        if not self.is_ready:
            logger.error("Search engine not ready.")
            return None

        term = term.lower().strip()
        postings = self.indexer.get_postings(term)

        if not postings:
            # Try BK-Tree suggestion
            suggestions = self.bktree.suggest(term, max_distance=2)
            return {
                "type": "not_found",
                "term": term,
                "suggestions": suggestions
            }

        result = {
            "type": "found",
            "term": term,
            "document_frequency": len(postings),
            "postings": {}
        }

        for url_id, stats in postings.items():
            url = self.indexer.get_url(url_id)
            decoded_positions = self.indexer.get_positions(term, url_id)
            result["postings"][url] = {
                "frequency": stats["frequency"],
                "positions": decoded_positions,
                "tf": round(stats["tf"], 4),
                "tf_idf": round(stats["tf_idf"], 4)
            }

        return result

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, raw_query: str, url: str) -> dict | None:
        """
        Explains RRF score breakdown for a specific result.
        Great for video demonstration.
        """
        if not self.is_ready:
            return None

        processed = self.processor.process(raw_query)
        tokens = self.processor.get_tokens(processed)
        raw = self.processor.get_raw(processed)

        url_id = self.indexer.get_url_id(url)
        if url_id is None:
            logger.error(f"URL not found in index: {url}")
            return None

        bm25_results = self.bm25.search(tokens, self.indexer)
        semantic_results = self.semantic.search(raw)

        return self.hybrid.explain(
            url_id=url_id,
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            indexer=self.indexer
        )

    # ------------------------------------------------------------------
    # Format Results
    # ------------------------------------------------------------------

    def _format_results(
        self,
        results: list[tuple[int, float]]
    ) -> list[dict]:
        """
        Converts (url_id, score) tuples to readable result dicts.
        """
        formatted = []
        for url_id, score in results:
            url = self.indexer.get_url(url_id)
            if url:
                formatted.append({
                    "type": "result",
                    "url": url,
                    "score": round(score, 4)
                })
        return formatted