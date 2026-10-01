"""
Tests for AutoResearchGap. They run fully offline (no API key, no internet):
retrieval uses the TF-IDF embedder + bundled sample dataset, and the LLM step
is exercised with a fake client.

Run with:  python -m pytest tests/ -v
"""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import Paper, dedupe_papers, load_from_csv, load_from_json
from src.embeddings import get_embedder
from src.rag_pipeline import AutoResearchGapPipeline
from src.vector_store import VectorStore

SAMPLE_CSV = str(Path(__file__).resolve().parent.parent / "sample_data" / "papers.csv")


@pytest.fixture()
def pipeline():
    pipe = AutoResearchGapPipeline(embedding_backend="tfidf")
    pipe.index_papers(load_from_csv(SAMPLE_CSV))
    return pipe


class FakeLLM:
    """Stands in for LLMClient; returns queued replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user_prompt, max_tokens=3000):
        self.calls += 1
        return self.replies.pop(0)


# ------------------------------------------------------------- data loading
def test_load_from_csv():
    papers = load_from_csv(SAMPLE_CSV)
    assert len(papers) == 10
    assert all(isinstance(p, Paper) for p in papers)
    assert papers[0].title.startswith("Attention Is All You Need")


def test_csv_handles_case_missing_values_and_upload_buffer():
    csv = 'Title,Abstract,Year\n"Paper A","About A",2020\n"Paper B",,2021\n,"No title",2022\n'
    papers = load_from_csv(io.StringIO(csv))
    assert [p.title for p in papers] == ["Paper A"]  # rows with missing title/abstract are skipped
    assert papers[0].year == 2020
    assert "nan" not in papers[0].authors.lower()


def test_csv_missing_columns_raises():
    with pytest.raises(ValueError):
        load_from_csv(io.StringIO("foo,bar\n1,2\n"))


def test_json_loader_and_dedupe():
    data = [
        {"title": "Same Title", "abstract": "x", "year": 2021},
        {"title": "same  title!", "abstract": "y"},
        {"title": "Other", "abstract": "z", "authors": ["A", "B"]},
    ]
    papers = dedupe_papers(load_from_json(io.StringIO(json.dumps(data))))
    assert len(papers) == 2
    assert papers[1].authors == "A, B"


# ----------------------------------------------------- embeddings + retrieval
def test_tfidf_embedder_shapes():
    embedder = get_embedder("tfidf")
    vecs = embedder.fit_transform(["deep learning for nlp", "transformers for vision", "reinforcement learning"])
    assert vecs.shape == (3, embedder.dim)


def test_vector_store_search():
    embedder = get_embedder("tfidf")
    papers = load_from_csv(SAMPLE_CSV)
    vecs = embedder.fit_transform([p.as_text() for p in papers])
    store = VectorStore(dim=vecs.shape[1])
    store.build(papers, vecs)
    assert len(store) == len(papers)

    q = embedder.transform(["retrieval augmented generation for question answering"])[0]
    results = store.search(q, top_k=3)
    assert len(results) == 3
    assert "Retrieval-Augmented Generation" in results[0][0].title


def test_year_filter(pipeline):
    res = pipeline.retrieve("language models", top_k=10, min_year=2020)
    assert res and all(p.year >= 2020 for p, _ in res)
    assert pipeline.retrieve("language models", top_k=5, min_year=2100) == []


def test_save_and_load_roundtrip(pipeline, tmp_path):
    pipeline.save(str(tmp_path / "idx"))
    loaded = AutoResearchGapPipeline.load(str(tmp_path / "idx"))
    a = pipeline.retrieve("retrieval augmented generation", top_k=3)
    b = loaded.retrieve("retrieval augmented generation", top_k=3)
    assert [p.title for p, _ in a] == [p.title for p, _ in b]


# -------------------------------------------------------------- analysis
def test_pipeline_offline_analysis(pipeline):
    result = pipeline.analyze_offline("automatic research gap identification", top_k=3)
    assert len(result.retrieved_papers) == 3
    assert any("offline-mode" in g["gap"] for g in result.research_gaps)
    md = result.to_markdown()
    assert "Research Gap Analysis" in md and "[P1]" in md


def test_llm_analysis_verifies_references_and_sorts_by_priority(pipeline):
    reply = json.dumps(
        {
            "covered_themes": ["RAG basics"],
            "research_gaps": [
                {"gap": "Low one", "justification": "j", "related_papers": ["P1"], "suggested_direction": "d", "priority": "low"},
                {
                    "gap": "High one",
                    "justification": "j",
                    "related_papers": ["P2", "P99", "A Totally Made Up Paper Title Nobody Wrote"],
                    "suggested_direction": "d",
                    "priority": "HIGH",
                },
            ],
            "confidence_note": "ok",
        }
    )
    pipeline._llm = FakeLLM("```json\n" + reply + "\n```")
    result = pipeline.analyze("retrieval augmented generation", top_k=3)

    assert [g["priority"] for g in result.research_gaps] == ["high", "low"]  # sorted
    high = result.research_gaps[0]
    assert len(high["related_papers"]) == 1  # P2 kept, P99 + fake title dropped
    assert high["related_papers"][0] in [p.title for p in result.retrieved_papers]
    assert result.warnings, "hallucinated references must be reported"
    assert json.loads(result.to_json())["research_gaps"]


def test_invalid_json_is_retried_once(pipeline):
    good = json.dumps({"covered_themes": [], "research_gaps": [], "confidence_note": "n"})
    fake = FakeLLM("sorry, here is prose not json", good)
    pipeline._llm = fake
    result = pipeline.analyze("anything", top_k=3)
    assert fake.calls == 2 and result.confidence_note == "n"


def test_invalid_json_twice_raises(pipeline):
    pipeline._llm = FakeLLM("nope", "still nope")
    with pytest.raises(json.JSONDecodeError):
        pipeline.analyze("anything", top_k=3)


# ------------------------------------------------- free / OpenAI-compatible LLMs
def _serve(handler_cls):
    import threading
    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_openai_compatible_client_end_to_end(pipeline, monkeypatch):
    """Runs the whole analyze() flow against a fake local OpenAI-style server (like Ollama/Groq/Gemini)."""
    from http.server import BaseHTTPRequestHandler

    from src.llm_client import OpenAICompatClient

    seen = {}
    reply = json.dumps(
        {
            "covered_themes": ["t"],
            "research_gaps": [
                {"gap": "g", "justification": "j", "related_papers": ["P1"], "suggested_direction": "d", "priority": "high"}
            ],
            "confidence_note": "c",
        }
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"], seen["auth"], seen["model"] = self.path, self.headers.get("Authorization"), body["model"]
            out = json.dumps({"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = _serve(Handler)
    try:
        pipeline._llm = OpenAICompatClient(
            "ollama", model="tiny-model", base_url=f"http://127.0.0.1:{server.server_port}/v1"
        )
        result = pipeline.analyze("retrieval augmented generation", top_k=3)
    finally:
        server.shutdown()

    assert seen["path"] == "/v1/chat/completions" and seen["model"] == "tiny-model"
    assert seen["auth"] is None  # ollama needs no key
    assert result.research_gaps[0]["priority"] == "high" and result.research_gaps[0]["related_papers"]


def test_provider_factory_and_missing_key(monkeypatch):
    from src.llm_client import get_llm_client

    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_llm_client("gemini")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_llm_client("nope")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert get_llm_client("groq").base_url.startswith("https://api.groq.com")
    assert get_llm_client("ollama").api_key == ""
