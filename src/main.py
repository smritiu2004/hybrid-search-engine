"""
main.py
Command-line interface for the search tool.
Provides four core commands: build, load, print, find.
Additional commands: help, quit, explain.
"""

import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawler import Crawler
from src.indexer import Indexer
from src.storage import Storage
from src.search import SearchEngine
from src.semantic import SemanticSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Prints welcome banner on startup."""
    print("""
╔══════════════════════════════════════════════════════╗
║           Web Search Tool — COMP3011                 ║
║  Hybrid BM25+ | Semantic | RRF | Cross-Encoder       ║
╚══════════════════════════════════════════════════════╝
Type 'help' for available commands.
    """)


def print_help():
    """Prints available commands."""
    print("""
Available Commands:
───────────────────────────────────────────────────────
  build                  Crawl website and build index
  load                   Load index from disk
  print <word>           Print inverted index for word
  find <query>           Hybrid search (default)
  find --bm25 <query>    BM25+ keyword search only
  find --semantic <query> Semantic search only
  explain <query> <url>  Explain RRF score for a result
  help                   Show this help message
  quit                   Exit the search tool
───────────────────────────────────────────────────────
    """)


def print_results(results: list[dict], query: str) -> None:
    """
    Prints search results to terminal.
    Handles result, suggestion, and not_found types.
    """
    if not results:
        print(f"\n  No results found for '{query}'.\n")
        return

    # Check if result is a suggestion
    if results[0].get("type") == "suggestion":
        suggestion = results[0]
        print(f"\n  {suggestion['message']}")
        if suggestion["suggestions"]:
            print(f"  Did you mean: {', '.join(suggestion['suggestions'])}?")
        print()
        return

    # Print ranked results
    print(f"\n  Results for '{query}':\n")
    for i, result in enumerate(results, start=1):
        print(f"  {i}. {result['url']}")
        print(f"     Score: {result['score']}")
    print()


def print_index_entry(entry: dict) -> None:
    """
    Prints inverted index entry for a term.
    Handles found and not_found types.
    """
    if entry is None:
        print("\n  Search engine not ready.\n")
        return

    if entry["type"] == "not_found":
        print(f"\n  '{entry['term']}' not found in index.")
        if entry["suggestions"]:
            print(f"  Did you mean: {', '.join(entry['suggestions'])}?")
        print()
        return

    print(f"\n  Term: '{entry['term']}'")
    print(f"  Document frequency: {entry['document_frequency']}")
    print(f"  ─────────────────────────────────────────")

    for url, stats in entry["postings"].items():
        print(f"\n  URL:       {url}")
        print(f"  Frequency: {stats['frequency']}")
        print(f"  Positions: {stats['positions']}")
        print(f"  TF:        {stats['tf']}")
        print(f"  TF-IDF:    {stats['tf_idf']}")

    print()


def print_explanation(explanation: dict) -> None:
    """Prints RRF score breakdown for a result."""
    if not explanation:
        print("\n  Could not explain result.\n")
        return

    print(f"""
  RRF Score Breakdown:
  ─────────────────────────────────────────
  URL:                    {explanation['url']}
  BM25+ Rank:             {explanation['bm25_rank']}
  BM25+ Score:            {explanation['bm25_score']}
  BM25+ RRF Contribution: {explanation['bm25_rrf_contribution']}
  Semantic Rank:          {explanation['semantic_rank']}
  Semantic Score:         {explanation['semantic_score']}
  Semantic Contribution:  {explanation['semantic_rrf_contribution']}
  Final RRF Score:        {explanation['final_rrf_score']}
  ─────────────────────────────────────────
    """)


def handle_build(
    search_engine: SearchEngine,
    storage: Storage
) -> tuple[Indexer, dict]:
    """
    Handles build command.
    Crawls website, builds index, saves to disk.
    Returns (indexer, pages) tuple.
    """
    print("\n  Starting crawl — this will take several minutes")
    print("  (6 second politeness window between requests)\n")

    # Step 1 — crawl
    crawler = Crawler()
    pages = crawler.crawl()

    if not pages:
        print("\n  Crawl returned no pages. Check your connection.\n")
        return None, None

    print(f"\n  Crawl complete — {len(pages)} pages found\n")

    # Step 2 — index
    print("  Building inverted index...")
    indexer = Indexer()
    indexer.build(pages)
    print(f"  Index built — {len(indexer.index)} unique terms\n")

    # Step 3 — save
    print("  Saving index to disk...")
    storage.save(indexer)
    print("  Index saved.\n")

    # Step 4 — build search components
    print("  Building search engine components...")
    search_engine.build(indexer, pages)
    print("  Search engine ready.\n")

    # Step 5 — save FAISS index
    search_engine.semantic.save()
    print("  FAISS index saved.\n")

    return indexer, pages


def handle_load(
    search_engine: SearchEngine,
    storage: Storage
) -> tuple[Indexer, dict] | tuple[None, None]:
    """
    Handles load command.
    Loads index from disk and rebuilds search components.
    Returns (indexer, pages) tuple or (None, None) if failed.
    """
    print("\n  Loading index from disk...\n")

    indexer = Indexer()
    success = storage.load(indexer)

    if not success:
        print("  Failed to load index. Run 'build' first.\n")
        return None, None

    print(f"  Index loaded — {len(indexer.index)} terms, "
          f"{len(indexer.url_to_id)} pages\n")

    # Rebuild pages dict from url_to_id
    # Note: raw text not stored — semantic loaded from FAISS file
    pages = indexer.pages


    print("  Loading search components...")
    success = search_engine.load(indexer, pages)

    if not success:
        print("  Failed to load search components.\n")
        return None, None

    print("  Search engine ready.\n")
    return indexer, pages


def handle_find(
    parts: list[str],
    search_engine: SearchEngine
) -> None:
    """
    Handles find command.
    Supports --bm25 and --semantic flags.
    """
    if not search_engine.is_ready:
        print("\n  Index not loaded. Run 'build' or 'load' first.\n")
        return

    # Parse mode flag
    mode = "hybrid"
    if "--bm25" in parts:
        mode = "bm25"
        parts.remove("--bm25")
    elif "--semantic" in parts:
        mode = "semantic"
        parts.remove("--semantic")

    # Extract query
    query = " ".join(parts[1:]).strip()

    if not query:
        print("\n  Usage: find <query>\n")
        return

    print(f"\n  Searching ({mode}): '{query}'...\n")
    results = search_engine.find(query, mode=mode)
    print_results(results, query)


def handle_print(
    parts: list[str],
    search_engine: SearchEngine
) -> None:
    """
    Handles print command.
    Prints inverted index entry for a word.
    """
    if not search_engine.is_ready:
        print("\n  Index not loaded. Run 'build' or 'load' first.\n")
        return

    if len(parts) < 2:
        print("\n  Usage: print <word>\n")
        return

    term = parts[1].strip()
    entry = search_engine.print_term(term)
    print_index_entry(entry)


def handle_explain(
    parts: list[str],
    search_engine: SearchEngine
) -> None:
    """
    Handles explain command.
    Shows RRF score breakdown for a query/url pair.
    Usage: explain <query> <url>
    """
    if not search_engine.is_ready:
        print("\n  Index not loaded. Run 'build' or 'load' first.\n")
        return

    if len(parts) < 3:
        print("\n  Usage: explain <query> <url>\n")
        return

    # Last part is URL, everything in between is query
    url = parts[-1]
    query = " ".join(parts[1:-1]).strip()

    explanation = search_engine.explain(query, url)
    print_explanation(explanation)



def main():
    """
    Main CLI loop.
    Initialises components and listens for commands.
    """
    print_banner()

    # Initialise shared components
    storage = Storage()
    search_engine = SearchEngine()

    while True:
        try:
            # Read user input
            raw_input = input("  > ").strip()

            if not raw_input:
                continue

            # Split into parts
            parts = raw_input.split()
            command = parts[0].lower()

            # Route to handler
            if command == "build":
                handle_build(search_engine, storage)

            elif command == "load":
                handle_load(search_engine, storage)

            elif command == "print":
                handle_print(parts, search_engine)

            elif command == "find":
                handle_find(parts, search_engine)

            elif command == "explain":
                handle_explain(parts, search_engine)

            elif command == "help":
                print_help()

            elif command in ("quit", "exit", "q"):
                print("\n  Goodbye!\n")
                break

            else:
                print(f"\n  Unknown command: '{command}'. Type 'help'.\n")

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Type 'quit' to exit.\n")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n  Error: {e}\n")


if __name__ == "__main__":
    main()