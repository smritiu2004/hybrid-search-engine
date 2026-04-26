"""
semantic.py
Handles semantic search using a Bi-Encoder model.
Encodes pages and queries into dense vectors.
Uses FAISS IndexFlatIP for cosine similarity search.
"""

import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# The Bi-Encoder model
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self):
        logger.info(f"Loading Bi-Encoder model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        # Encoding a dummy sentence to get embedding dimension
        self.dimension = self.model.encode(["test"], convert_to_numpy=True).shape[1]

        # FAISS index — IndexFlatIP = inner product (cosine if normalised)
        self.faiss_index = None

        # Ordered list of url_ids matching FAISS index order
        self.url_ids = []


    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of texts into dense vectors.
        Normalises vectors for cosine similarity via inner product.
        Returns numpy array of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            # normalises for cosine similarity
            normalize_embeddings=True 
        )
        return embeddings.astype(np.float32)


    def build(self, indexer, pages: dict[str, str]) -> None:
        """
        Encodes all crawled pages and stores in FAISS index.
        Accepts:
          indexer — for URL to ID mapping
          pages   — dict of {url: raw text}
        """
        logger.info("Building FAISS index with page embeddings...")

        # Building a  ordered list of (url_id, text) pairs
        url_id_text_pairs = []
        for url, text in pages.items():
            url_id = indexer.url_to_id.get(url)
            if url_id is not None:
                url_id_text_pairs.append((url_id, text))

        if not url_id_text_pairs:
            logger.error("No pages to encode.")
            return

        # Separating into ordered lists
        self.url_ids = [pair[0] for pair in url_id_text_pairs]
        texts = [pair[1] for pair in url_id_text_pairs]

        # Encodes all pages
        logger.info(f"Encoding {len(texts)} pages...")
        embeddings = self._encode(texts)

        # Builds FAISS IndexFlatIP
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)

        logger.info(f"FAISS index built — {self.faiss_index.ntotal} vectors stored")


    def search(
        self,
        query: str,
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Encodes query and searches FAISS index.
        Returns list of (url_id, score) tuples sorted by score descending.
        """
        if self.faiss_index is None:
            logger.error("FAISS index not built. Call build() first.")
            return []

        if not query.strip():
            logger.warning("Empty query passed to semantic search.")
            return []

        # Encodes query — shape (1, dimension)
        query_embedding = self._encode([query])

        # Searches FAISS — returns distances and indices
        scores, indices = self.faiss_index.search(query_embedding, top_n)

        # scores shape: (1, top_n)
        # indices shape: (1, top_n)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for empty slots
                continue
            url_id = self.url_ids[idx]
            results.append((url_id, float(score)))

        logger.info(f"Semantic search returned {len(results)} results")
        return results


    def save(self, path: str = "data/faiss.index") -> None:
        """
        Saves FAISS index and url_ids to disk.
        """
        if self.faiss_index is None:
            logger.error("No FAISS index to save.")
            return

        faiss.write_index(self.faiss_index, path)

        # Saves url_ids as numpy array alongside
        np.save(path + ".url_ids.npy", np.array(self.url_ids))
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str = "data/faiss.index") -> bool:
        """
        Loads FAISS index and url_ids from disk.
        Returns True if successful, False otherwise.
        """
        import os
        if not os.path.exists(path):
            logger.error(f"FAISS index file not found: {path}")
            return False

        self.faiss_index = faiss.read_index(path)

        url_ids_path = path + ".url_ids.npy"
        if not os.path.exists(url_ids_path):
            logger.error(f"url_ids file not found: {url_ids_path}")
            return False

        self.url_ids = np.load(url_ids_path).tolist()
        logger.info(f"FAISS index loaded — {self.faiss_index.ntotal} vectors")
        return True