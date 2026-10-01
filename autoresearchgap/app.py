"""
AutoResearchGap — Streamlit UI

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import dedupe_papers, fetch_from_arxiv, load_from_csv, load_from_json
from src.llm_client import PROVIDERS, default_model
from src.rag_pipeline import AutoResearchGapPipeline

load_dotenv()

st.set_page_config(page_title="AutoResearchGap", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #18242b;
        --muted: #637177;
        --paper: #f5f1e9;
        --cream: #fffdf8;
        --line: #ddd7cc;
        --teal: #087f78;
        --coral: #e66c4c;
        --yellow: #f1c75b;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp > header { background: transparent; }
    [data-testid="stSidebar"] {
        background: #1d3033;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f3f0e8; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [role="radiogroup"] p { color: #f3f0e8 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #294447;
        border-color: #4c6364;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #e6c96a;
        border: 0;
        color: #1d3033;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #f1d87c; }
    [data-testid="stSidebar"] hr { border-color: #496062; }
    h1, h2, h3, p, label { font-family: 'Space Grotesk', sans-serif; }
    h1, h2, h3, [data-testid="stHeader"] { color: var(--ink) !important; }
    label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p { color: var(--ink); }
    h1 { letter-spacing: -1.5px; }
    .hero {
        display: flex;
        justify-content: space-between;
        gap: 24px;
        align-items: end;
        padding: 24px 0 30px;
        border-bottom: 1px solid var(--line);
    }
    .hero-kicker, .eyebrow {
        color: var(--teal);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 8px 0 6px;
        color: var(--ink);
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        font-weight: 700;
        line-height: 0.96;
    }
    .hero-copy { max-width: 570px; color: var(--muted); font-size: 1rem; }
    .hero-stamp {
        min-width: 154px;
        padding: 16px;
        border: 1px solid var(--ink);
        border-radius: 2px;
        background: var(--yellow);
        color: var(--ink);
        transform: rotate(2deg);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        line-height: 1.55;
    }
    .metric {
        min-height: 92px;
        padding: 16px 18px;
        border: 1px solid var(--line);
        border-radius: 3px;
        background: var(--cream);
    }
    .metric-label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { margin-top: 7px; color: var(--ink); font-size: 1.65rem; font-weight: 700; }
    .section-heading { margin: 30px 0 12px; color: var(--ink); font-size: 1.5rem; font-weight: 700; }
    .section-heading span { color: var(--coral); }
    .stTextInput > div > div, .stTextArea > div > div { background: var(--cream); border-color: var(--line); }
    .stTextInput input, .stTextArea textarea { color: var(--ink) !important; font-size: 1rem; }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #8a9495 !important; opacity: 1; }
    [data-testid="stSlider"] label, [data-testid="stCheckbox"] label { color: var(--ink) !important; }
    .stButton > button, .stDownloadButton > button { border-radius: 3px; font-weight: 600; }
    .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
    .stButton > button[kind="primary"]:hover { background: #056a65; border-color: #056a65; }
    [data-testid="stExpander"] { border-color: var(--line); background: rgba(255, 253, 248, 0.6); }
    .gap-card {
        margin: 12px 0;
        padding: 20px 22px;
        border-left: 5px solid var(--coral);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--cream);
    }
    .gap-card.high { border-left-color: #c94d43; }
    .gap-card.medium { border-left-color: var(--yellow); }
    .gap-card.low { border-left-color: var(--teal); }
    .gap-number { color: var(--coral); font-family: 'DM Mono', monospace; font-size: 0.72rem; text-transform: uppercase; }
    .gap-title { margin: 6px 0 12px; color: var(--ink); font-size: 1.12rem; font-weight: 700; }
    .gap-meta { color: var(--muted); font-size: 0.9rem; line-height: 1.55; }
    .direction { margin-top: 14px; padding: 12px 14px; background: #e9f1e8; color: #235d58; font-size: 0.9rem; }
    @media (max-width: 700px) {
        .hero { align-items: start; flex-direction: column; }
        .hero-stamp { transform: none; }
        .hero-title { font-size: 2.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-kicker">Literature intelligence / 01</div>
        <div class="hero-title">Find the edges<br>of what we know.</div>
        <div class="hero-copy">AutoResearchGap maps your literature, surfaces what is already covered, and turns the open space into grounded research directions.</div>
      </div>
      <div class="hero-stamp">RAG<br>RESEARCH<br>WORKSPACE<br><b>v1.0 / READY</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("pipeline", None)
st.session_state.setdefault("papers", None)
st.session_state.setdefault("result", None)

PRIORITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<div class='eyebrow'>Workspace setup</div>", unsafe_allow_html=True)
    st.header("01 / Literature corpus")
    source = st.radio("Data source", ["Sample dataset", "Upload CSV", "Upload JSON", "Fetch from arXiv"])

    papers = None
    try:
        if source == "Sample dataset":
            if st.button("Load sample_data/papers.csv"):
                papers = load_from_csv("sample_data/papers.csv")
        elif source == "Upload CSV":
            up = st.file_uploader("CSV with columns: title, abstract (+ id, year, authors)", type=["csv"])
            if up is not None:
                papers = load_from_csv(up)  # read straight from the upload (works on Windows too)
        elif source == "Upload JSON":
            up = st.file_uploader("JSON list of {title, abstract, year, authors}", type=["json"])
            if up is not None:
                papers = load_from_json(up)
        else:
            arxiv_q = st.text_input("arXiv search query", value="retrieval augmented generation")
            max_results = st.slider("Max results", 5, 100, 25)
            if st.button("Fetch from arXiv"):
                with st.spinner("Fetching from arXiv..."):
                    papers = fetch_from_arxiv(arxiv_q, max_results=max_results)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load data: {e}")

    if papers:
        st.session_state.papers = dedupe_papers(papers)
        st.session_state.pipeline = None  # corpus changed -> index must be rebuilt
        st.success(f"Loaded {len(st.session_state.papers)} papers.")

    st.divider()
    st.header("02 / Retrieval engine")
    backend = st.selectbox(
        "Backend",
        ["tfidf", "sentence-transformers"],
        help="tfidf works fully offline. sentence-transformers gives richer semantic search but needs internet the first time.",
    )
    if st.session_state.papers and st.button("Build / rebuild index", type="primary"):
        try:
            with st.spinner("Embedding papers and building FAISS index..."):
                pipeline = AutoResearchGapPipeline(embedding_backend=backend)
                pipeline.index_papers(st.session_state.papers)
                st.session_state.pipeline = pipeline
            st.success(f"Indexed {len(st.session_state.papers)} papers.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Indexing failed: {e}")

    st.divider()
    st.header("03 / Synthesis model")
    provider = st.selectbox(
        "Provider",
        list(PROVIDERS),
        index=list(PROVIDERS).index(os.environ.get("LLM_PROVIDER", "anthropic")) if os.environ.get("LLM_PROVIDER", "anthropic") in PROVIDERS else 0,
        help="gemini, groq and openrouter have free tiers; ollama runs locally (free, no key).",
    )
    key_env = PROVIDERS[provider]["key_env"]
    api_key = os.environ.get(key_env, "") if key_env else ""
    model = st.text_input(
        "Model",
        value=os.environ.get("LLM_MODEL") or default_model(provider),
        key=f"model_{provider}",
    )
    llm_ready = bool(api_key) or key_env is None

# --------------------------------------------------------------- main panel
papers_loaded = st.session_state.papers
metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown(f"<div class='metric'><div class='metric-label'>Corpus</div><div class='metric-value'>{len(papers_loaded or [])} papers</div></div>", unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown(f"<div class='metric'><div class='metric-label'>Index</div><div class='metric-value'>{'Ready' if st.session_state.pipeline else 'Waiting'}</div></div>", unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown(f"<div class='metric'><div class='metric-label'>Mode</div><div class='metric-value'>{'Offline' if not llm_ready else provider.title()}</div></div>", unsafe_allow_html=True)

if papers_loaded:
    with st.expander(f"📚 Loaded corpus ({len(papers_loaded)} papers)", expanded=False):
        for p in papers_loaded:
            year = f" ({p.year})" if p.year else ""
            st.markdown(f"**{p.title}{year}** — {p.authors}")
            st.caption(p.abstract)

st.markdown("<div class='section-heading'>Research question <span>/</span> retrieval brief</div>", unsafe_allow_html=True)
topic = st.text_input(
    "Research topic / question",
    placeholder="e.g. Automatic identification of research gaps using LLMs and retrieval-augmented generation",
)

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Number of papers to retrieve as context", 3, 20, 5)
with col2:
    years = [p.year for p in (papers_loaded or []) if p.year]
    if years and min(years) < max(years):
        year_range = st.slider("Publication year range", min(years), max(years), (min(years), max(years)))
    else:
        year_range = (None, None)

offline_mode = st.checkbox("Offline demo mode (skip LLM call, no API key needed)", value=not llm_ready)

if st.button("Run gap analysis  →", type="primary", disabled=not topic):
    pipeline = st.session_state.pipeline
    if pipeline is None:
        st.warning("Load a corpus and click 'Build / rebuild index' first (see sidebar).")
    else:
        pipeline._llm_provider = provider
        pipeline._llm_model = model or None
        pipeline._llm = None  # pick up a changed provider/key/model
        # Only apply the year filter if the user actually narrowed it.
        yr_min, yr_max = year_range if years and year_range != (min(years), max(years)) else (None, None)
        try:
            with st.spinner("Retrieving relevant literature and analyzing..."):
                run = pipeline.analyze_offline if offline_mode else pipeline.analyze
                st.session_state.result = run(topic, top_k=top_k, min_year=yr_min, max_year=yr_max)
        except Exception as e:  # noqa: BLE001
            st.session_state.result = None
            st.error(f"Analysis failed: {e}")

result = st.session_state.result
if result:
    st.markdown("<div class='section-heading'>What the field already knows <span>/</span> covered themes</div>", unsafe_allow_html=True)
    for theme in result.covered_themes:
        st.markdown(f"- {theme}")

    st.markdown("<div class='section-heading'>Where the field is still open <span>/</span> identified gaps</div>", unsafe_allow_html=True)
    for i, g in enumerate(result.research_gaps, 1):
        priority = g.get("priority", "medium")
        related = "; ".join(g.get("related_papers", [])) or "No direct references"
        st.markdown(
            f"""
            <div class="gap-card {priority}">
              <div class="gap-number">Gap {i} / {priority} priority</div>
              <div class="gap-title">{g.get('gap', '')}</div>
              <div class="gap-meta"><b>Why it matters:</b> {g.get('justification', '')}</div>
              <div class="gap-meta"><b>Grounded in:</b> {related}</div>
              <div class="direction"><b>Next direction →</b> {g.get('suggested_direction', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if result.confidence_note:
        st.info(result.confidence_note)
    for w in result.warnings:
        st.warning(w)

    with st.expander("🔎 Retrieved papers (context given to the LLM)"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"ID": f"P{i}", "Title": p.title, "Year": p.year, "Similarity": round(s, 3)}
                    for i, (p, s) in enumerate(zip(result.retrieved_papers, result.retrieved_scores), 1)
                ]
            ),
            hide_index=True,
        )

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Report (Markdown)", result.to_markdown(), "research_gap_report.md", "text/markdown")
    d2.download_button("⬇️ Report (JSON)", result.to_json(), "research_gap_report.json", "application/json")
else:
    st.caption("Load a corpus, build the index, enter a topic, then click Analyze.")
