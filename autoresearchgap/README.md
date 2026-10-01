# 🔬 AutoResearchGap

**AI Framework for Automatic Research Gap Identification using LLMs and RAG**

AutoResearchGap ingests a corpus of research papers (titles + abstracts), retrieves
the literature most relevant to a topic you're investigating, and uses an LLM
(Claude) to synthesize a structured analysis of **what's already covered** vs.
**what research gaps remain** — grounded in the retrieved papers rather than
the model's memory.

## How it works (architecture)

```
                ┌─────────────────┐
   papers.csv → │  Data Loader     │  (CSV / JSON / live arXiv API)
   / arXiv      └────────┬─────────┘
                         │  title + abstract
                         ▼
                ┌─────────────────┐
                │  Embedder        │  TF-IDF (offline) or
                │                  │  sentence-transformers (semantic)
                └────────┬─────────┘
                         │  vectors
                         ▼
                ┌─────────────────┐
                │  FAISS Vector    │  cosine-similarity index
                │  Store           │
                └────────┬─────────┘
                         │  top-k relevant papers  (RETRIEVAL)
                         ▼
                ┌─────────────────┐
                │  Claude (LLM)    │  prompted with retrieved abstracts,
                │  Gap Analyzer    │  returns structured JSON            (GENERATION)
                └────────┬─────────┘
                         ▼
             covered themes + specific research gaps
             + justification + suggested future directions
```

This is a classic **RAG (Retrieval-Augmented Generation)** pipeline, applied to
the specific task of research-gap mining instead of generic Q&A.

## Project structure

```
autoresearchgap/
├── app.py                  # Streamlit web UI
├── cli.py                  # Command-line interface
├── requirements.txt
├── .env.example
├── sample_data/
│   └── papers.csv          # 10 sample NLP/RAG papers for demo & tests
├── src/
│   ├── data_loader.py       # CSV/JSON/arXiv loading, cleaning, de-duplication
│   ├── embeddings.py        # TF-IDF (offline) + sentence-transformers backends
│   ├── vector_store.py      # FAISS index: year filters, save/load
│   ├── llm_client.py        # Anthropic API wrapper (retries, model override)
│   └── rag_pipeline.py      # Retrieval + LLM + reference verification
└── tests/
    └── test_pipeline.py     # Offline tests (no API key needed; LLM is mocked)
```

## What's new in this version

- **Reference verification** – the LLM cites papers as `[P1]`, `[P2]`…; every citation is checked against the papers actually retrieved. Invented papers are dropped and reported as warnings.
- **Gap priority** – each gap gets `high / medium / low` and results are sorted by it.
- **Year filter** – restrict retrieval to a publication-year range (`--min-year`, `--max-year`, or the UI slider).
- **Save / load index** – `--save-index DIR` / `--load-index DIR`, no re-embedding on every run.
- **JSON + Markdown export** (CLI `--format json`, UI download buttons) and a similarity table for retrieved papers.
- **Robustness** – auto-retry on malformed LLM JSON, NaN/empty-row handling, de-duplication, case-insensitive CSV columns, Windows-safe uploads.
- **Smarter offline mode** – flags topic terms that the retrieved abstracts never mention.
- **Free LLM providers** – besides Claude you can use Google Gemini, Groq, OpenRouter or a local Ollama model (see below).
- Default Claude model is now `claude-sonnet-5` (override with `--model` or `LLM_MODEL`).

## Setup

Requires Python 3.9+.

```bash
cd autoresearchgap

# 1. (recommended) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add an API key (only needed for real LLM analysis - see "Free LLM options")
cp .env.example .env               # Windows: copy .env.example .env
# open .env, set LLM_PROVIDER and paste the matching key
```

## Free LLM options

Claude's API is paid. To run the full analysis for free, pick another provider
(`LLM_PROVIDER` in `.env`, `--provider` on the CLI, or the dropdown in the UI):

| Provider | Cost | Key | Notes |
|---|---|---|---|
| `gemini` | free tier | `GEMINI_API_KEY` from https://aistudio.google.com/apikey | easiest to start with |
| `groq` | free tier | `GROQ_API_KEY` from https://console.groq.com/keys | very fast, open models |
| `openrouter` | free `:free` models | `OPENROUTER_API_KEY` from https://openrouter.ai/keys | daily request limits |
| `ollama` | 100% free, local | none | install from https://ollama.com/download, then `ollama pull llama3.1` |

```bash
python cli.py --csv sample_data/papers.csv --topic "research gap detection" --provider gemini
python cli.py --csv sample_data/papers.csv --topic "research gap detection" --provider ollama --model llama3.1
```

Free-tier model names and limits change often. If you get a "model not found" or
quota error, pass a different name with `--model` (or set `LLM_MODEL`). Smaller
free models are less reliable at strict JSON and at grounding; the pipeline
retries bad JSON once and drops citations that were not actually retrieved.

## Usage

### 1. Web UI (recommended)

```bash
streamlit run app.py
```

Opens at http://localhost:8501. In the sidebar: load the sample data (or upload
CSV/JSON, or fetch from arXiv) → **Build / rebuild index** → type your topic →
**Analyze**. Tick "Offline demo mode" to try it without an API key.

### 2. Command line

```bash
# Offline demo, no API key needed:
python cli.py --csv sample_data/papers.csv \
  --topic "Automatic research gap identification using LLMs and RAG" --offline

# Full run with a real LLM call:
python cli.py --csv sample_data/papers.csv \
  --topic "Automatic research gap identification using LLMs and RAG"

# Live arXiv corpus, save the index, JSON report:
python cli.py --arxiv-query "retrieval augmented generation" --save-index my_index \
  --topic "RAG for scientific literature review" --format json --out report.json

# Re-use the saved index, only papers from 2022 onwards:
python cli.py --load-index my_index --topic "RAG evaluation" --min-year 2022
```

### 3. As a library

```python
from src.data_loader import load_from_csv
from src.rag_pipeline import AutoResearchGapPipeline

pipeline = AutoResearchGapPipeline(embedding_backend="tfidf")
pipeline.index_papers(load_from_csv("sample_data/papers.csv"))

result = pipeline.analyze("LLM-based literature review automation", top_k=5, min_year=2020)
print(result.to_markdown())   # or result.to_json()
```

## Bring your own data

Your own CSV needs at minimum `title` and `abstract` columns (`id`, `year`,
`authors`, `url` are optional; column names are case-insensitive, rows with an empty title/abstract are skipped):

```csv
id,title,year,authors,abstract
1,"My Paper Title",2024,"A. Author","Abstract text goes here..."
```

Or use `--arxiv-query "your search terms"` to pull live papers from arXiv
instead (requires internet access to `export.arxiv.org`).

## Design notes / choices

- **Two embedding backends** are supported so the project runs anywhere:
  - `tfidf` (default): pure scikit-learn, zero downloads, fully offline.
  - `sentence-transformers`: much stronger semantic embeddings, but downloads
    model weights from Hugging Face on first use (needs internet).
- **FAISS `IndexFlatIP`** is used with L2-normalized vectors, which is
  mathematically equivalent to cosine similarity — a good default for small-
  to-medium corpora. Swap in `IndexIVFFlat` for larger corpora (10k+ papers).
- **Structured JSON output** from the LLM (rather than free text) makes the
  result directly renderable in the UI and easy to post-process/export.
- **Offline mode** (`analyze_offline` / `--offline`) lets you exercise the
  entire retrieval pipeline without an API key — useful for demos, grading,
  or CI, and clearly labeled as a non-LLM placeholder so it's never mistaken
  for a real analysis.
- **Grounding / anti-hallucination**: the system prompt explicitly instructs
  the model to only reference the provided abstracts and to say so if the
  context is too thin to support a claim, rather than inventing papers.

## Limitations & future work

- Retrieval quality depends heavily on the embedding backend — TF-IDF is
  lexical (keyword) matching, not true semantic search. For serious use,
  enable `sentence-transformers` or swap in an API-based embedding model.
- Corpus size in this demo is small (10 papers). For a real literature review,
  ingest hundreds-to-thousands of papers (e.g., via the arXiv or Semantic
  Scholar APIs) and consider chunking full-text PDFs rather than abstracts only.
- No citation-graph analysis (who cites whom) is used — that's a natural
  complementary signal for gap detection that could be added.
- No clustering of near-identical gaps across multiple runs yet.
- TF-IDF vectors are stored densely, so very large corpora (10k+ papers) should use `sentence-transformers` (smaller vectors) or an IVF index.

## Testing

```bash
python -m pytest tests/ -v
```

All tests run fully offline (the LLM is replaced by a fake client).
