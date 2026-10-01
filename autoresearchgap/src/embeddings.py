"""
embeddings.py
-------------
Two interchangeable embedding backends:

1. TfidfEmbedder   - pure scikit-learn, works fully offline, no model download.
2. STEmbedder      - sentence-transformers (default 'all-MiniLM-L6-v2'), much
                     stronger semantic embeddings, but needs internet the first
                     time to download the model weights.

Both expose: fit_transform(texts), transform(texts), dim, save(dir), load(dir)
so the rest of the pipeline doesn't care which one is used.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEmbedder:
    """Offline TF-IDF based embedder. Deterministic, fast, no downloads."""

    name = "tfidf"

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return normalize(matrix.toarray().astype("float32"))

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first before transform().")
        matrix = self.vectorizer.transform(texts)
        return normalize(matrix.toarray().astype("float32"))

    @property
    def dim(self) -> int:
        return len(self.vectorizer.get_feature_names_out())

    def save(self, directory: Path) -> None:
        with open(Path(directory) / "embedder.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        # NOTE: pickle - only load indexes that you created yourself.
        obj = cls()
        with open(Path(directory) / "embedder.pkl", "rb") as f:
            obj.vectorizer = pickle.load(f)  # noqa: S301
        obj._fitted = True
        return obj


class STEmbedder:
    """
    Sentence-Transformers based embedder for higher quality semantic search.
    Requires: pip install sentence-transformers  (+ internet on first run).
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run:\n"
                "  pip install sentence-transformers\n"
                "or use the default 'tfidf' backend for a fully offline setup."
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)  # pretrained: "fit" is just encode

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    @property
    def dim(self) -> int:
        return self._dim

    def save(self, directory: Path) -> None:
        (Path(directory) / "embedder.json").write_text(json.dumps({"model_name": self.model_name}))

    @classmethod
    def load(cls, directory: Path) -> "STEmbedder":
        meta = json.loads((Path(directory) / "embedder.json").read_text())
        return cls(model_name=meta["model_name"])


def get_embedder(backend: str = "tfidf", **kwargs):
    """Factory: get_embedder('tfidf') or get_embedder('sentence-transformers')."""
    backend = backend.lower()
    if backend in ("tfidf", "offline", "default"):
        return TfidfEmbedder(**kwargs)
    if backend in ("sentence-transformers", "st", "semantic"):
        return STEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
