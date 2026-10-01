"""
data_loader.py
--------------
Loads a corpus of research papers (id, title, year, authors, abstract) from
local files (CSV/JSON), file-like objects (e.g. Streamlit uploads), or fetches
them live from the arXiv API.

This is the ingestion layer of AutoResearchGap.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PathOrBuffer = Union[str, Path, IO]


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    year: Optional[int] = None
    authors: str = ""
    source: str = "local"
    url: str = ""
    extra: dict = field(default_factory=dict)

    def as_text(self) -> str:
        """Concatenate title + abstract, the unit of text we embed & retrieve on."""
        return f"{self.title.strip()}. {self.abstract.strip()}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "source": self.source,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            id=str(d.get("id", "")),
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            year=d.get("year"),
            authors=d.get("authors", ""),
            source=d.get("source", "local"),
            url=d.get("url", ""),
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean(value) -> str:
    """Turn any cell into a clean string ('' for NaN/None, collapsed whitespace)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _to_year(value) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def dedupe_papers(papers: List[Paper]) -> List[Paper]:
    """Drop papers with an identical (normalized) title, keeping the first one."""
    seen, unique = set(), []
    for p in papers:
        key = _title_key(p.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    if len(unique) < len(papers):
        logger.info("Removed %d duplicate papers", len(papers) - len(unique))
    return unique


def _records_to_papers(records: List[dict], source: str) -> List[Paper]:
    papers, skipped = [], 0
    for i, rec in enumerate(records):
        rec = {str(k).strip().lower(): v for k, v in rec.items()}
        title, abstract = _clean(rec.get("title")), _clean(rec.get("abstract"))
        if not title or not abstract:
            skipped += 1
            continue
        authors = rec.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join(map(str, authors))
        papers.append(
            Paper(
                id=_clean(rec.get("id")) or str(i),
                title=title,
                abstract=abstract,
                year=_to_year(rec.get("year")),
                authors=_clean(authors),
                source=source,
                url=_clean(rec.get("url")),
            )
        )
    if skipped:
        logger.warning("Skipped %d rows with missing title/abstract", skipped)
    if not papers:
        raise ValueError("No valid papers found (each needs a non-empty title and abstract).")
    return papers


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_from_csv(path: PathOrBuffer) -> List[Paper]:
    """Load papers from a CSV (path or file-like) with columns: title, abstract [, id, year, authors, url]."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"title", "abstract"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return _records_to_papers(df.to_dict(orient="records"), source="local_csv")


def load_from_json(path: PathOrBuffer) -> List[Paper]:
    """Load papers from a JSON file (path or file-like): a list of objects with title/abstract/etc."""
    if hasattr(path, "read"):
        raw = path.read()
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):  # allow {"papers": [...]}
        data = data.get("papers", [])
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of paper objects (or {'papers': [...]}).")
    return _records_to_papers(data, source="local_json")


def fetch_from_arxiv(query: str, max_results: int = 25, retries: int = 3) -> List[Paper]:
    """
    Fetch papers live from the arXiv API (https://export.arxiv.org/api/query).

    NOTE: Requires outbound internet access to export.arxiv.org. If you're offline,
    use load_from_csv / load_from_json with the bundled sample_data instead.
    """
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            papers = _parse_arxiv_feed(resp.text)
            if not papers:
                raise ValueError("arXiv returned no results for this query.")
            return papers
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("arXiv attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Failed to fetch from arXiv after {retries} attempts: {last_err}. "
        "Check your internet connection, or use local CSV/JSON data instead."
    )


def _parse_arxiv_feed(xml_text: str) -> List[Paper]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.find("atom:id", ns).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        title = _clean(entry.find("atom:title", ns).text)
        abstract = _clean(entry.find("atom:summary", ns).text)
        published = entry.find("atom:published", ns).text
        authors = ", ".join(a.find("atom:name", ns).text for a in entry.findall("atom:author", ns))
        papers.append(
            Paper(
                id=arxiv_id,
                title=title,
                abstract=abstract,
                year=int(published[:4]) if published else None,
                authors=authors,
                source="arxiv",
                url=raw_id,
            )
        )
    return papers


def load_corpus(
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None,
    arxiv_query: Optional[str] = None,
    arxiv_max_results: int = 25,
) -> List[Paper]:
    """Convenience combinator: pull from whichever source(s) are provided (de-duplicated)."""
    papers: List[Paper] = []
    if csv_path:
        papers.extend(load_from_csv(csv_path))
    if json_path:
        papers.extend(load_from_json(json_path))
    if arxiv_query:
        papers.extend(fetch_from_arxiv(arxiv_query, max_results=arxiv_max_results))
    if not papers:
        raise ValueError("No data source provided to load_corpus().")
    return dedupe_papers(papers)
