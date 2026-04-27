"""
bktree.py
Implements a BK-Tree for spell correction using Levenshtein distance.
BK-Tree enables O(log n) approximate string matching, making it efficient for large vocabularies.
Built from the crawled vocabulary in the inverted index.
"""

import logging

logger = logging.getLogger(__name__)

# Each node represents one word and has children keyed by edit distance from that word.
class BKNode:
    def __init__(self, word: str):
        self.word = word
        self.children: dict[int, "BKNode"] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None
        self.size: int = 0


    def _levenshtein(self, s1: str, s2: str) -> int:
        """
        Levenshtein distance between two strings.
        Computes Levenshtein edit distance between two strings.
        Uses dynamic programming — O(mn) where m, n = string lengths.
        Counts minimum insertions, deletions, substitutions
        needed to transform s1 into s2.
        """
        m, n = len(s1), len(s2)

        # Creating a DP (Dynamic Programming) matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # delete all chars from s1
        for j in range(n + 1):
            dp[0][j] = j  # insert all chars from s2

        # Fill the DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # Characters match — no operation needed
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )

        return dp[m][n]


    def insert(self, word: str) -> None:
        """
        Inserts a word into the BK-Tree.
        Each node's children are keyed by edit distance from parent.
        """
        if not word or not word.strip():
            return

        if self.root is None:
            self.root = BKNode(word)
            self.size += 1
            return

        node = self.root
        while True:
            dist = self._levenshtein(word, node.word)

            # If word already exists
            if dist == 0: 
                return

            if dist not in node.children:
                # No child at this distance — insert here
                node.children[dist] = BKNode(word)
                self.size += 1
                return

            # Follow existing child at this distance
            node = node.children[dist]

    def build(self, indexer) -> None:
        """
        Builds BK-Tree from all terms in the inverted index vocabulary.
        """
        logger.info("Building BK-Tree from index vocabulary...")

        for term in indexer.index.keys():
            self.insert(term)

        logger.info(f"BK-Tree built — {self.size} words inserted")


    def search(
        self,
        query: str,
        max_distance: int = 2
    ) -> list[tuple[str, int]]:
        """
        Finds all words within max_distance edits of query.
        Uses BK-Tree property to prune search space efficiently.
        Returns list of (word, distance) tuples sorted by distance.
        O(log n) average case vs O(n) brute force.
        """
        if not query or not query.strip():
            return []

        if self.root is None:
            logger.warning("BK-Tree is empty.")
            return []

        query = query.lower().strip()
        results = []

        # Stack-based DFS — avoids recursion limit issues
        stack = [self.root]

        while stack:
            node = stack.pop()
            dist = self._levenshtein(query, node.word)

            if dist <= max_distance:
                results.append((node.word, dist))

            # BK-Tree pruning — only visit children within range
            # For any candidate child c with distance d from parent:
            # |dist - d| <= max_distance is required for c to be useful
            low = dist - max_distance
            high = dist + max_distance

            for child_dist, child_node in node.children.items():
                if low <= child_dist <= high:
                    stack.append(child_node)

        # Sort by distance — closest matches first
        results.sort(key=lambda x: x[1])

        logger.info(
            f"BK-Tree search '{query}' "
            f"(max_dist={max_distance}) → {results}"
        )
        return results

    def suggest(
        self,
        query: str,
        max_distance: int = 2,
        limit: int = 5
    ) -> list[str]:
        """
        Returns up to `limit` spelling suggestions for query.
        Wrapper around search that returns just the words.
        """
        results = self.search(query, max_distance)
        return [word for word, _ in results[:limit]]

    # Stat helpers
    def __len__(self) -> int:
        """Returns number of words in BK-Tree."""
        return self.size

    def __contains__(self, word: str) -> bool:
        """
        Returns True if exact word exists in BK-Tree.
        Uses search with max_distance=0.
        """
        results = self.search(word, max_distance=0)
        return len(results) > 0