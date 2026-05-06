# COMP3011 Web Search Tool

A command-line search tool that crawls [quotes.toscrape.com](https://quotes.toscrape.com), 
builds an inverted index, and provides hybrid search combining BM25+, 
semantic search, and Cross-Encoder reranking.

## Tech Stack

| Component | Technology |
|---|---|
| Crawler | requests + BeautifulSoup4 |
| Inverted Index | defaultdict + delta encoding |
| Keyword Ranking | BM25+ (rank-bm25) |
| Semantic Search | Bi-Encoder (all-MiniLM-L6-v2) + FAISS |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Query Suggestions | Trie + BK-Tree (Levenshtein) |
| Serialisation | MessagePack + SHA256 checksum |

## Project Structure

~~~ bash 
comp3011-web-search-tool/
├── src/
│   ├── crawler.py          # Web crawler with politeness window
│   ├── indexer.py          # Inverted index with delta encoding
│   ├── storage.py          # MessagePack serialisation + SHA256
│   ├── bm25_ranker.py      # BM25+ keyword ranking
│   ├── semantic.py         # Bi-Encoder + FAISS semantic search
│   ├── reranker.py         # Cross-Encoder reranking
│   ├── hybrid.py           # Reciprocal Rank Fusion
│   ├── query_processor.py  # spaCy query processing pipeline
│   ├── trie.py             # Prefix tree for autocomplete
│   ├── bktree.py           # BK-Tree for spell correction
│   ├── search.py           # SearchEngine facade
│   └── main.py             # CLI entry point
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_bm25.py
│   ├── test_semantic.py
│   ├── test_hybrid.py
│   ├── test_query_processor.py
│   └── test_search.py
├── data/
│   ├── index.msgpack       
│   └── faiss.index         
├── requirements.txt
└── README.md
~~~

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Smriti-Umesh/comp3011-web-search-tool.git
cd comp3011-web-search-tool
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download spaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

## Usage

**Start the search tool:**
```bash
python src/main.py
```

### Commands

**Build index** — crawls the website and builds the full index with 6-second politeness window.

~~~ bash 
build
~~~

**Load index** — loads previously built index from disk:

~~~ bash 
load
~~~

**Print index entry** — shows inverted index stats for a word
~~~ bash 
print love
print truth
~~~

**Find** — hybrid search (default):
~~~ bash 
find love
find good friends
~~~

**Find with BM25+ only:** 
~~~ bash 
find --bm25 truth
~~~

**Find with semantic search only:**
~~~ bash 
find --semantic world
~~~

**Explain** — shows RRF score breakdown for a result:
~~~ bash 
explain love https://quotes.toscrape.com/tag/love/page/1/
~~~

**Help:**
~~~ bash 
help
~~~

**Quit:**
~~~ bash 
quit
~~~

## Design Decisions

### Inverted Index
Uses a nested `defaultdict` for O(1) average lookup. Position lists 
are **delta-encoded** — storing gaps between positions rather than raw 
values, reducing storage size. URLs are mapped to integer IDs via a 
bidirectional dict to avoid redundant string storage throughout the index.

### BM25+
Chosen over standard BM25 as it adds a delta floor to the term frequency 
component, preventing over-penalisation of terms in longer documents.

### Hybrid Search Pipeline
BM25+ → fast keyword retrieval
Bi-Encoder → semantic similarity via FAISS IndexFlatIP
RRF → parameter-free fusion of ranked lists (k=60, Cormack et al. 2009)
Cross-Encoder → precision re-ranking of top 10 results

RRF was chosen over weighted score combination because BM25+ and cosine 
similarity scores are on incompatible scales. RRF uses only rank position, 
making it robust without manual weight tuning.

### FAISS IndexFlatIP
Vectors are L2-normalised before storage, making inner product equivalent 
to cosine similarity. `IndexFlatIP` was chosen over `IndexFlatL2` for 
this reason.

### MessagePack + SHA256
MessagePack binary serialisation produces a significantly smaller and 
faster file than JSON. A SHA256 checksum is stored alongside the index 
and verified on every load to detect corruption.

### Query Suggestions
- **Trie** — O(m) prefix autocomplete where m = query length
- **BK-Tree** — O(log n) spell correction using Levenshtein distance 
  with triangle inequality pruning

## Testing

**Run full test suite:**
```bash
pytest tests/ -v
```

**Run with coverage report:**
```bash
pytest --cov=src --cov-report=term-missing tests/
```

**Run a specific test file:**
```bash
pytest tests/test_crawler.py -v
```

## Dependencies

~~~ bash 
requests==2.31.0
beautifulsoup4==4.12.3
rank-bm25==0.2.2
sentence-transformers==2.7.0
faiss-cpu==1.8.0
msgpack==1.0.8
spacy==3.7.4
pytest==8.1.1
pytest-cov==5.0.0
~~~

## References

- Cormack, G.V., Clarke, C.L.A., Buettcher, S. (2009). 
  Reciprocal Rank Fusion outperforms Condorcet and individual 
  Rank Learning Methods. SIGIR 2009.
- Robertson, S., Zaragoza, H. (2009). 
  The Probabilistic Relevance Framework: BM25 and Beyond.
- Reimers, N., Gurevych, I. (2019). 
  Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

