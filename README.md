# Hybrid Semantic Search Engine

A command-line search tool, done as part of university coursework, that crawls [quotes.toscrape.com](https://quotes.toscrape.com), 
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

## Prerequisites
- Python 3.10+
- pip

## Environment Setup

**1. Create and activate virtual environment**
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download spaCy language model**
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

<details>
<summary>Click to expand rationale behind key technical choices</summary>

### Inverted Index
Uses a nested `defaultdict` for O(1) average lookup. Position lists 
are delta-encoded — storing gaps between positions rather than raw 
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
to cosine similarity. IndexFlatIP was chosen over IndexFlatL2 for 
this reason.

### MessagePack + SHA256
MessagePack binary serialisation produces a significantly smaller and 
faster file than JSON. A SHA256 checksum is stored alongside the index 
and verified on every load to detect corruption.

### Query Suggestions
- Trie — O(m) prefix autocomplete where m = query length
- BK-Tree — O(log n) spell correction using Levenshtein distance 
  with triangle inequality pruning

</details>


## References

Cormack, G.V., Clarke, C.L.A. and Buettcher, S. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval.

Reimers, N. and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

Robertson, S. 2010. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends® in Information Retrieval. 3(4), pp.333–389.

Bhatia, R. 2024. Cross-encoders vs Bi-encoders : A deep-dive into text encoding methods. Medium. [Online]. Available from: https://medium.com/@rbhatia46/cross-encoders-vs-bi-encoders-a-deep-dive-into-text-encoding-methods-d9aa890d6ca4.

