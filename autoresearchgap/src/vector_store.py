"""
vector_store.py
----------------
Thin wrapper around FAISS for building, querying and persisting a similarity
index over paper embeddings. This is the "R" (Retrieval) in RAG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from .data_loader import Paper


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.papers: List[Paper] = []

    def build(self, papers: List[Paper], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(papers):
            raise ValueError("Number of embeddings must match number of papers.")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} != index dim {self.dim}")
        self.papers = list(papers)
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Tuple[Paper, float]]:
        """Top-k most similar papers, optionally restricted to a year range."""
        if self.index.ntotal == 0:
            return []
        filtering = min_year is not None or max_year is not None
        # A flat index is exhaustive anyway, so for filtering we simply rank everything.
        k = self.index.ntotal if filtering else min(top_k, self.index.ntotal)
        query_vec = query_vec.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Paper, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            paper = self.papers[idx]
            if filtering:
                if paper.year is None:
                    continue
                if min_year is not None and paper.year < min_year:
                    continue
                if max_year is not None and paper.year > max_year:
                    continue
            results.append((paper, float(score)))
            if len(results) >= top_k:
                break
        return results

    # -- persistence -------------------------------------------------------- #
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "papers.json").write_text(
            json.dumps([p.to_dict() for p in self.papers], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        store = cls(dim=index.d)
        store.index = index
        raw = json.loads((directory / "papers.json").read_text(encoding="utf-8"))
        store.papers = [Paper.from_dict(d) for d in raw]
        return store

    def __len__(self) -> int:
        return len(self.papers)
