"""
trie.py
Implements a Trie (prefix tree) for autocomplete query suggestions.
Built from the crawled vocabulary in the inverted index.
Provides O(m) prefix lookup where m = length of query prefix.
"""

import logging

logger = logging.getLogger(__name__)


# Each node represents one character: 
# 1. Children a dictionary mapping character to next TrieNode. O(1) lookup.
# 2. is_end boolean flag to mark end of a valid word.
# 3. word string to store the full word at end nodes for easy retrieval during autocomplete
class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.word: str = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.size = 0


    def insert(self, word: str) -> None:
        """
        Inserts a word into the Trie.
        O(m) where m = length of word.
        """
        if not word or not word.strip():
            return

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        # Marking the end of a word
        if not node.is_end:
            node.is_end = True
            node.word = word
            self.size += 1

    def build(self, indexer) -> None:
        """
        Builds Trie from all terms in the inverted index vocabulary.
        """
        logger.info("Building Trie from index vocabulary...")

        for term in indexer.index.keys():
            self.insert(term)

        logger.info(f"Trie built — {self.size} words inserted")


    def _collect_words(
        self,
        node: TrieNode,
        results: list[str],
        limit: int
    ) -> None:
        
        """
        Recursively collects all words from a given TrieNode.
        Stops once limit is reached.
        """
        if len(results) >= limit:
            return

        if node.is_end:
            results.append(node.word)

        for child in node.children.values():
            if len(results) >= limit:
                return
            self._collect_words(child, results, limit)

    def autocomplete(
        self,
        prefix: str,
        limit: int = 5
    ) -> list[str]:
        
        """
        Returns up to `limit` words that start with prefix.
        O(m + k) where m = prefix length, k = number of results.
        Returns empty list if no matches found.
        """
        if not prefix or not prefix.strip():
            return []

        prefix = prefix.lower().strip()
        node = self.root

        # Traversing to the end of a prefix
        for char in prefix:
            if char not in node.children:
                # Prefix not in Trie
                logger.info(f"No autocomplete results for prefix: '{prefix}'")
                return []
            node = node.children[char]

        # Collecting all the words from this node
        results = []
        self._collect_words(node, results, limit)

        logger.info(f"Autocomplete '{prefix}' → {results}")
        return results


    def contains(self, word: str) -> bool:
        """
        Returns True if exact word exists in Trie.
        O(m) where m = length of word.
        """
        
        if not word:
            return False

        node = self.root
        for char in word.lower().strip():
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        """
        Returns True if any word in Trie starts with prefix.
        O(m) where m = length of prefix.
        """
        if not prefix:
            return False

        node = self.root
        for char in prefix.lower().strip():
            if char not in node.children:
                return False
            node = node.children[char]

        return True


    def __len__(self) -> int:
        """Returns number of words in Trie."""
        return self.size

    def __contains__(self, word: str) -> bool:
        """Allows 'word in trie' syntax."""
        return self.contains(word)