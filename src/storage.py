"""
storage.py
Handles serialisation and deserialisation of the inverted index.
Uses MessagePack for binary serialisation and SHA256 for integrity checking.
"""

import hashlib
import logging
import os
import msgpack
from collections import defaultdict

logger = logging.getLogger(__name__)

INDEX_PATH = "data/index.msgpack"
CHECKSUM_PATH = "data/index.sha256"


class Storage:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        checksum_path: str = CHECKSUM_PATH
    ):
        self.index_path = index_path
        self.checksum_path = checksum_path

   # Helpers for serialisation and integrity checking

    def _compute_checksum(self, data: bytes) -> str:
        """
        Computes SHA256 checksum of binary data.
        Returns hex string.
        """
        return hashlib.sha256(data).hexdigest()

    def _default_index(self) -> defaultdict:
        """
        Returns a fresh nested defaultdict matching the indexer structure.
        Used when rebuilding the index from loaded data.
        """
        return defaultdict(lambda: defaultdict(lambda: {
            "frequency": 0,
            "positions": [],
            "tf": 0.0,
            "tf_idf": 0.0
        }))

    def _index_to_dict(self, index: defaultdict) -> dict:
        """
        Converts nested defaultdict to a regular nested dict.
        MessagePack cannot serialise defaultdict directly.
        """
        return {
            term: {
                url_id: stats
                for url_id, stats in postings.items()
            }
            for term, postings in index.items()
        }

    def _dict_to_index(self, data: dict) -> defaultdict:
        """
        Rebuilds nested defaultdict from a regular dict.
        Called after deserialisation.
        """
        index = self._default_index()
        for term, postings in data.items():
            for url_id, stats in postings.items():
                # msgpack deserialises keys as bytes decode to int
                # as  MessagePack deserialises dictionary keys differently depending on their type. 
                int_id = int(url_id)
                index[term][int_id] = {
                    "frequency": stats["frequency"],
                    "positions": stats["positions"],
                    "tf": stats["tf"],
                    "tf_idf": stats["tf_idf"]
                }
        return index


    def save(self, indexer) -> None:
        """
        Serialises and saves the index to disk.
        Accepts an Indexer instance.
        Steps:
          1. Convert index to regular dict
          2. Bundle with URL maps and page lengths
          3. Serialise with MessagePack
          4. Compute and save SHA256 checksum
          5. Save binary to disk
        """
        logger.info("Saving index to disk...")

        # Ensures data directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Step 1 — converting defaultdict to a regular dict
        index_dict = self._index_to_dict(indexer.index)

        # Step 2 — bundling everything together
        payload = {
            "index": index_dict,
            "url_to_id": indexer.url_to_id,
            "id_to_url": {
                str(k): v
                for k, v in indexer.id_to_url.items()
            },
            "page_lengths": {
                str(k): v
                for k, v in indexer.page_lengths.items()
            }
        }

        # Step 3 — serialising with MessagePack
        serialised = msgpack.packb(payload, use_bin_type=True)

        # Step 4 — computing checksum
        checksum = self._compute_checksum(serialised)

        # Step 5 — saving binary index
        with open(self.index_path, "wb") as f:
            f.write(serialised)

        # Step 6 — saving checksum
        with open(self.checksum_path, "w") as f:
            f.write(checksum)

        logger.info(f"Index saved to {self.index_path}")
        logger.info(f"Checksum saved to {self.checksum_path}")
        logger.info(f"Index size: {len(serialised)} bytes")



    def load(self, indexer) -> bool:
        """
        Loads and deserialises the index from disk into an Indexer instance.
        Steps:
          1. Check files exist
          2. Read binary data
          3. Verify SHA256 checksum
          4. Deserialise with MessagePack
          5. Rebuild indexer state
        Returns True if successful, False otherwise.
        """
        logger.info("Loading index from disk...")

        # Step 1 — checking if files exist
        if not os.path.exists(self.index_path):
            logger.error(f"Index file not found: {self.index_path}")
            return False

        if not os.path.exists(self.checksum_path):
            logger.error(f"Checksum file not found: {self.checksum_path}")
            return False

        # Step 2 — reading binary data
        with open(self.index_path, "rb") as f:
            serialised = f.read()

        # Step 3 — verifying checksum
        computed = self._compute_checksum(serialised)
        with open(self.checksum_path, "r") as f:
            stored = f.read().strip()

        if computed != stored:
            logger.error("Checksum mismatch — index may be corrupted!")
            return False

        logger.info("Checksum verified")

        # Step 4 — deserialising
        payload = msgpack.unpackb(serialised, raw=False)

        # Step 5 — rebuild indexer state
        indexer.index = self._dict_to_index(payload["index"])

        indexer.url_to_id = payload["url_to_id"]

        indexer.id_to_url = {
            int(k): v
            for k, v in payload["id_to_url"].items()
        }

        indexer.page_lengths = {
            int(k): v
            for k, v in payload["page_lengths"].items()
        }

        logger.info(f"Index loaded — {len(indexer.url_to_id)} pages, {len(indexer.index)} terms")
        return True