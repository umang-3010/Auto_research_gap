#!/usr/bin/env python3
"""
AutoResearchGap CLI
--------------------
Examples:

  # Offline demo (no API key needed) using bundled sample data:
  python cli.py --csv sample_data/papers.csv --topic "LLM-based research gap detection" --offline

  # Full run with a real LLM call (needs ANTHROPIC_API_KEY in .env):
  python cli.py --csv sample_data/papers.csv --topic "LLM-based research gap detection"

  # Same, but with a FREE provider:
  python cli.py --csv sample_data/papers.csv --topic "..." --provider gemini      # GEMINI_API_KEY
  python cli.py --csv sample_data/papers.csv --topic "..." --provider groq        # GROQ_API_KEY
  python cli.py --csv sample_data/papers.csv --topic "..." --provider ollama      # local, no key

  # Live papers from arXiv, save the index for next time, write a JSON report:
  python cli.py --arxiv-query "retrieval augmented generation" --save-index my_index \\
      --topic "RAG for scientific literature review" --format json --out report.json

  # Re-use a saved index (no re-embedding, no corpus flags needed):
  python cli.py --load-index my_index --topic "RAG evaluation" --min-year 2022
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from src.data_loader import load_corpus
from src.rag_pipeline import AutoResearchGapPipeline


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="AutoResearchGap: identify research gaps using LLMs + RAG")
    p.add_argument("--csv", help="CSV of papers (title, abstract, [id, year, authors])")
    p.add_argument("--json", help="JSON file of papers")
    p.add_argument("--arxiv-query", help="Fetch papers live from arXiv matching this query")
    p.add_argument("--arxiv-max-results", type=int, default=25)
    p.add_argument("--load-index", metavar="DIR", help="Load a previously saved index instead of a corpus")
    p.add_argument("--save-index", metavar="DIR", help="Save the built index to this directory")
    p.add_argument("--topic", required=True, help="The research topic/question to analyze")
    p.add_argument("--top-k", type=int, default=5, help="Number of papers to retrieve as context")
    p.add_argument("--min-year", type=int, default=None, help="Only use papers published in/after this year")
    p.add_argument("--max-year", type=int, default=None, help="Only use papers published in/before this year")
    p.add_argument(
        "--embedding-backend",
        default="tfidf",
        choices=["tfidf", "sentence-transformers"],
        help="'tfidf' works offline; 'sentence-transformers' needs internet on first run.",
    )
    p.add_argument("--offline", action="store_true", help="Skip the LLM call (heuristic mode, no API key)")
    p.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "gemini", "groq", "openrouter", "ollama"],
        help="LLM provider (default: $LLM_PROVIDER or anthropic). gemini/groq/openrouter/ollama have free options.",
    )
    p.add_argument("--model", default=None, help="Override the model name for the chosen provider")
    p.add_argument("--format", choices=["md", "json"], default="md", help="Report format (default: md)")
    p.add_argument("--out", default=None, help="Write the report to this file instead of stdout")
    p.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    if not (args.load_index or args.csv or args.json or args.arxiv_query):
        p.error("Provide --load-index, or at least one of --csv / --json / --arxiv-query")

    try:
        if args.load_index:
            print(f"Loading saved index from {args.load_index} ...", file=sys.stderr)
            pipeline = AutoResearchGapPipeline.load(args.load_index, llm_model=args.model, llm_provider=args.provider)
        else:
            print("Loading corpus...", file=sys.stderr)
            papers = load_corpus(
                csv_path=args.csv,
                json_path=args.json,
                arxiv_query=args.arxiv_query,
                arxiv_max_results=args.arxiv_max_results,
            )
            print(f"Loaded {len(papers)} papers. Building index...", file=sys.stderr)
            pipeline = AutoResearchGapPipeline(
                embedding_backend=args.embedding_backend, llm_model=args.model, llm_provider=args.provider
            )
            pipeline.index_papers(papers)
            if args.save_index:
                pipeline.save(args.save_index)
                print(f"Index saved to {args.save_index}/", file=sys.stderr)

        print(f"Analyzing topic: {args.topic!r} (top_k={args.top_k})", file=sys.stderr)
        run = pipeline.analyze_offline if args.offline else pipeline.analyze
        result = run(args.topic, top_k=args.top_k, min_year=args.min_year, max_year=args.max_year)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    report = result.to_json() if args.format == "json" else result.to_markdown()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
