"""
rag_pipeline.py
----------------
The heart of AutoResearchGap: combines the Retriever (vector_store.py) with
a Generator (llm_client.py) to produce a structured research-gap analysis
for a given research topic/query, grounded in retrieved literature (RAG).

Upgrades over v1:
  * papers are referenced by short IDs ([P1], [P2] ...) and every reference the
    LLM returns is VERIFIED against what was actually retrieved (anti-hallucination)
  * gaps carry a priority (high/medium/low) and are sorted by it
  * one automatic retry if the LLM returns malformed JSON
  * optional year filtering during retrieval
  * save()/load() so the index doesn't have to be rebuilt every run
  * smarter offline mode (topic-term coverage heuristic instead of a pure stub)
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .data_loader import Paper
from .embeddings import STEmbedder, TfidfEmbedder, get_embedder
from .llm_client import get_llm_client
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

SYSTEM_PROMPT = """You are AutoResearchGap, a research-intelligence assistant that helps \
researchers identify UNDER-EXPLORED research gaps in a scientific field.

You will be given:
1. A research topic/question the user is investigating.
2. A set of retrieved papers, each labelled with an ID like [P1], [P2] (title + abstract).

Your job:
- Summarize what the retrieved literature already covers (2-4 themes).
- Identify concrete, specific research gaps: things the retrieved papers do NOT \
address, only partially address, or explicitly call out as open problems.
- For each gap, justify it using the retrieved papers, cite them ONLY by their ID \
(e.g. "P2"), suggest a concrete future research direction, and give a priority \
("high", "medium" or "low") based on how important and under-explored it is.
- Be specific and grounded in the provided abstracts. Never invent papers, IDs or \
claims that are not supported by the given context. If the context is too thin to \
support a claim, say so in confidence_note rather than fabricating.

Respond ONLY with a valid JSON object (no markdown fences, no preamble) matching \
this schema:
{
  "covered_themes": [string, ...],
  "research_gaps": [
    {
      "gap": string,
      "justification": string,
      "related_papers": ["P1", "P3"],
      "suggested_direction": string,
      "priority": "high" | "medium" | "low"
    }
  ],
  "confidence_note": string
}
"""


@dataclass
class GapAnalysisResult:
    topic: str
    covered_themes: List[str]
    research_gaps: List[dict]
    confidence_note: str
    retrieved_papers: List[Paper] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "covered_themes": self.covered_themes,
            "research_gaps": self.research_gaps,
            "confidence_note": self.confidence_note,
            "warnings": self.warnings,
            "retrieved_papers": [
                {**p.to_dict(), "score": round(s, 4)}
                for p, s in zip(self.retrieved_papers, self.retrieved_scores or [0.0] * len(self.retrieved_papers))
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [f"# Research Gap Analysis: {self.topic}\n"]
        lines.append("## What the literature already covers\n")
        for theme in self.covered_themes:
            lines.append(f"- {theme}")
        lines.append("\n## Identified Research Gaps\n")
        for i, g in enumerate(self.research_gaps, 1):
            prio = g.get("priority", "")
            tag = f" [{prio.upper()}]" if prio else ""
            lines.append(f"### Gap {i}{tag}: {g.get('gap', '')}")
            lines.append(f"**Why:** {g.get('justification', '')}")
            related = ", ".join(g.get("related_papers", []))
            if related:
                lines.append(f"**Related papers:** {related}")
            lines.append(f"**Suggested direction:** {g.get('suggested_direction', '')}\n")
        if self.confidence_note:
            lines.append(f"> **Note:** {self.confidence_note}")
        for w in self.warnings:
            lines.append(f"> **Warning:** {w}")
        lines.append("\n## Retrieved Papers Used as Context\n")
        scores = self.retrieved_scores or [None] * len(self.retrieved_papers)
        for i, (p, s) in enumerate(zip(self.retrieved_papers, scores), 1):
            year = f" ({p.year})" if p.year else ""
            score = f" — similarity {s:.3f}" if s is not None else ""
            link = f" <{p.url}>" if p.url else ""
            lines.append(f"- [P{i}] {p.title}{year} — {p.authors}{score}{link}")
        return "\n".join(lines)


class AutoResearchGapPipeline:
    """
    End-to-end pipeline:
      papers -> embed -> build FAISS index -> retrieve top-k for a query
             -> prompt LLM with retrieved context -> verified structured gap analysis
    """

    def __init__(
        self,
        embedding_backend: str = "tfidf",
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.embedding_backend = embedding_backend
        self.embedder = get_embedder(embedding_backend)
        self.store: Optional[VectorStore] = None
        self._llm = None
        self._llm_model = llm_model
        self._llm_provider = llm_provider

    # ------------------------------------------------------------------ index
    def index_papers(self, papers: List[Paper]) -> None:
        if not papers:
            raise ValueError("Cannot index an empty list of papers.")
        texts = [p.as_text() for p in papers]
        embeddings = self.embedder.fit_transform(texts)
        self.store = VectorStore(dim=embeddings.shape[1])
        self.store.build(papers, embeddings)

    def save(self, directory: str) -> None:
        """Persist index + papers + embedder so later runs can skip re-embedding."""
        if self.store is None:
            raise RuntimeError("Nothing to save: call index_papers() first.")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.store.save(d)
        self.embedder.save(d)
        (d / "meta.json").write_text(json.dumps({"embedding_backend": self.embedding_backend}))

    @classmethod
    def load(
        cls, directory: str, llm_model: Optional[str] = None, llm_provider: Optional[str] = None
    ) -> "AutoResearchGapPipeline":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        backend = meta["embedding_backend"]
        pipe = cls.__new__(cls)
        pipe.embedding_backend = backend
        pipe.embedder = (
            STEmbedder.load(d) if backend.lower() in ("sentence-transformers", "st", "semantic") else TfidfEmbedder.load(d)
        )
        pipe.store = VectorStore.load(d)
        pipe._llm = None
        pipe._llm_model = llm_model
        pipe._llm_provider = llm_provider
        return pipe

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Tuple[Paper, float]]:
        if self.store is None:
            raise RuntimeError("Call index_papers() (or load()) before retrieve().")
        query_vec = self.embedder.transform([query])[0]
        return self.store.search(query_vec, top_k=top_k, min_year=min_year, max_year=max_year)

    # ------------------------------------------------------------- generation
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_client(provider=self._llm_provider, model=self._llm_model)
        return self._llm

    @staticmethod
    def _build_user_prompt(topic: str, retrieved: List[Tuple[Paper, float]]) -> str:
        blocks = []
        for i, (paper, score) in enumerate(retrieved, 1):
            year = f" ({paper.year})" if paper.year else ""
            blocks.append(
                f"[P{i}] {paper.title}{year}\nAuthors: {paper.authors}\n"
                f"Abstract: {paper.abstract}\n(similarity score: {score:.3f})"
            )
        context = "\n\n---\n\n".join(blocks)
        return (
            f"Research topic: {topic}\n\n"
            f"Retrieved literature context:\n\n{context}\n\n"
            "Analyze the above and return the JSON object described in your instructions."
        )

    @staticmethod
    def _parse_llm_json(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _complete_json(self, user_prompt: str) -> dict:
        """Call the LLM; if the JSON is malformed, ask once more, then give up."""
        llm = self._get_llm()
        raw = llm.complete(system=SYSTEM_PROMPT, user_prompt=user_prompt)
        try:
            return self._parse_llm_json(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, retrying once")
            retry_prompt = (
                user_prompt
                + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object."
            )
            raw = llm.complete(system=SYSTEM_PROMPT, user_prompt=retry_prompt)
            return self._parse_llm_json(raw)

    @staticmethod
    def _resolve_refs(refs, retrieved: List[Tuple[Paper, float]]) -> Tuple[List[str], List[str]]:
        """
        Map LLM-cited references ("P2", "[P2]", or a title) to real retrieved titles.
        Returns (resolved_titles, unresolved_refs). Anything unresolved is a
        potential hallucination and is reported instead of silently shown.
        """
        titles = [p.title for p, _ in retrieved]
        resolved, unresolved = [], []
        for ref in refs or []:
            ref_s = str(ref).strip()
            m = re.fullmatch(r"\[?P(\d+)\]?", ref_s, flags=re.IGNORECASE)
            if m and 1 <= int(m.group(1)) <= len(titles):
                title = titles[int(m.group(1)) - 1]
            else:
                close = difflib.get_close_matches(ref_s, titles, n=1, cutoff=0.6)
                title = close[0] if close else None
            if title is None:
                unresolved.append(ref_s)
            elif title not in resolved:
                resolved.append(title)
        return resolved, unresolved

    def _normalize_result(self, topic: str, parsed: dict, retrieved: List[Tuple[Paper, float]]) -> GapAnalysisResult:
        warnings: List[str] = []
        gaps = []
        for g in parsed.get("research_gaps", []) or []:
            if not isinstance(g, dict):
                continue
            related, bad = self._resolve_refs(g.get("related_papers", []), retrieved)
            if bad:
                warnings.append(
                    f"Gap '{str(g.get('gap', ''))[:60]}...' cited papers that were not retrieved and were dropped: {bad}"
                )
            prio = str(g.get("priority", "medium")).lower()
            gaps.append(
                {
                    "gap": str(g.get("gap", "")).strip(),
                    "justification": str(g.get("justification", "")).strip(),
                    "related_papers": related,
                    "suggested_direction": str(g.get("suggested_direction", "")).strip(),
                    "priority": prio if prio in PRIORITY_ORDER else "medium",
                }
            )
        gaps.sort(key=lambda g: PRIORITY_ORDER[g["priority"]])
        return GapAnalysisResult(
            topic=topic,
            covered_themes=[str(t) for t in parsed.get("covered_themes", []) or []],
            research_gaps=gaps,
            confidence_note=str(parsed.get("confidence_note", "")),
            retrieved_papers=[p for p, _ in retrieved],
            retrieved_scores=[s for _, s in retrieved],
            warnings=warnings,
        )

    def analyze(
        self,
        topic: str,
        top_k: int = 5,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> GapAnalysisResult:
        """Full pipeline: retrieve context for `topic`, call the LLM, return a verified structured result."""
        retrieved = self.retrieve(topic, top_k=top_k, min_year=min_year, max_year=max_year)
        if not retrieved:
            raise RuntimeError("No papers retrieved for this topic (check the year filter / corpus).")
        parsed = self._complete_json(self._build_user_prompt(topic, retrieved))
        return self._normalize_result(topic, parsed, retrieved)

    def analyze_offline(
        self,
        topic: str,
        top_k: int = 5,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> GapAnalysisResult:
        """
        Demo/offline mode: no API key required. Uses retrieval plus a simple
        coverage heuristic (which topic words never appear in the retrieved
        abstracts?) so the whole pipeline can be exercised without an LLM.
        """
        retrieved = self.retrieve(topic, top_k=top_k, min_year=min_year, max_year=max_year)
        themes = [f"Coverage found in: {p.title}" for p, _ in retrieved[:3]]
        titles = [p.title for p, _ in retrieved]

        gaps = [
            {
                "gap": (
                    "[offline-mode placeholder] Run with a real ANTHROPIC_API_KEY "
                    "to get an LLM-synthesized gap analysis instead of this heuristic stub."
                ),
                "justification": "No LLM call was made in offline mode.",
                "related_papers": titles,
                "suggested_direction": "Remove --offline / untick offline mode to get the full analysis.",
                "priority": "low",
            }
        ]

        # Heuristic: topic terms that none of the retrieved abstracts mention.
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            weak = ENGLISH_STOP_WORDS | {"using", "based", "towards", "toward", "approach", "study", "analysis", "novel"}
            topic_terms = {t for t in re.findall(r"[a-z]{4,}", topic.lower()) if t not in weak}
            corpus_text = " ".join(p.as_text().lower() for p, _ in retrieved)
            # crude stemming: 'llms' matches 'llm', 'models' matches 'model'
            missing = sorted(t for t in topic_terms if t.rstrip("s") not in corpus_text)
            if missing:
                gaps.insert(
                    0,
                    {
                        "gap": f"Topic aspects not mentioned in any retrieved abstract: {', '.join(missing)}",
                        "justification": "Keyword-coverage heuristic (not an LLM judgement): these terms from your topic never appear in the top-k papers.",
                        "related_papers": titles,
                        "suggested_direction": "Search for literature on these aspects specifically, or widen top-k / the corpus.",
                        "priority": "medium",
                    },
                )
        except Exception:  # noqa: BLE001 - heuristic must never break offline mode
            logger.debug("coverage heuristic failed", exc_info=True)

        return GapAnalysisResult(
            topic=topic,
            covered_themes=themes,
            research_gaps=gaps,
            confidence_note="Offline heuristic mode — no LLM used.",
            retrieved_papers=[p for p, _ in retrieved],
            retrieved_scores=[s for _, s in retrieved],
        )
