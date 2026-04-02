import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
from utils import get_current_active_users
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def define_style():
    st.markdown(
        """
        <style>
            /* Expander header */
            .stExpander > .stButton > button {
                width: 100%;
                border: none;
                background-color: #f0f2f6;
                color: #333;
                text-align: left;
                padding: 15px;
                font-size: 16px;
                border-radius: 10px;
                margin-top: 5px;
            }
            .stExpander > .stExpanderContent {
                padding-left: 10px;
                padding-top: 10px;
            }
            a {
                color: #26557b;
                text-decoration: none;
            }
            /* Compact metric labels */
            [data-testid="stMetricLabel"] {
                font-size: 12px;
            }
            [data-testid="stMetricValue"] {
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo(last_date, db_size_bio, db_size_pubmed, db_size_med, db_size_arxiv):
    active_users = get_current_active_users()
    total_papers = int(db_size_pubmed) + int(db_size_bio) + int(db_size_med) + int(db_size_arxiv)

    LOGGER.info(
        f"Header: PubMed={db_size_pubmed}, BioRxiv={db_size_bio}, "
        f"medRxiv={db_size_med}, arXiv={db_size_arxiv}, users={active_users}"
    )

    pubmed_logo   = "https://upload.wikimedia.org/wikipedia/commons/f/fb/US-NLM-PubMed-Logo.svg"
    biorxiv_logo  = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    arxiv_logo    = "https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"

    sources = [
        (pubmed_logo,   "PubMed",  "https://pubmed.ncbi.nlm.nih.gov/", db_size_pubmed),
        (biorxiv_logo,  "BioRxiv", "https://www.biorxiv.org/",         db_size_bio),
        (medarxiv_logo, "medRxiv", "https://www.medrxiv.org/",         db_size_med),
        (arxiv_logo,    "arXiv",   "https://arxiv.org/",               db_size_arxiv),
    ]

    # Each source: logo + count + name, all on one horizontal line
    logo_cells = "".join(
        f"""<a href="{url}" target="_blank" class="mss-src" title="{name}">
              <img src="{logo}" alt="{name}">
              <span class="mss-src-info">
                <span class="mss-src-count">{int(size):,}</span>
                <span class="mss-src-name">{name}</span>
              </span>
            </a>"""
        for logo, name, url, size in sources
    )

    st.markdown(
        f"""
        <style>
          .mss-wrap {{
            text-align: center;
            padding: 20px 0 4px 0;
          }}
          .mss-title {{
            font-size: 3.8rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0 0 8px 0;
            line-height: 1.1;
            color: inherit;
          }}
          .mss-sub {{
            font-size: 0.85rem;
            font-weight: 400;
            color: #888;
            margin: 0 0 20px 0;
          }}
          .mss-logos {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 40px;
            flex-wrap: nowrap;
            margin-bottom: 14px;
          }}
          .mss-src {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            text-decoration: none !important;
            color: inherit !important;
            flex-shrink: 0;
          }}
          .mss-src img {{
            height: 52px;
            max-width: 130px;
            object-fit: contain;
            display: block;
            opacity: 0.9;
          }}
          .mss-src:hover img {{ opacity: 1; }}
          .mss-src-info {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
          }}
          .mss-src-count {{
            font-size: 0.85rem;
            font-weight: 600;
            color: inherit;
            line-height: 1;
          }}
          .mss-src-name {{
            font-size: 0.6rem;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            line-height: 1;
          }}
          .mss-foot {{
            font-size: 0.4rem;
            color: #bbb;
            margin: 0;
          }}
          .mss-foot b {{ color: #999; font-weight: 600; }}
        </style>

        <div class="mss-wrap">
          <p class="mss-title" style="font-size:2rem;font-weight:700;letter-spacing:-0.5px;line-height:1.1;margin:0 0 8px 0;color:inherit;">Manuscript Semantic Search</p>
          <p class="mss-sub">{total_papers:,} papers indexed &nbsp;·&nbsp;
            Updated <b style="color:#777">{last_date}</b> &nbsp;·&nbsp;
            {active_users} active user{'s' if active_users != 1 else ''}
          </p>
          <div class="mss-logos">{logo_cells}</div>
          <p class="mss-foot">
            Not affiliated with PubMed, BioRxiv, medRxiv, or arXiv.
            Logos are property of their respective owners.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def plot_score_vs_year(sorted_results: pd.DataFrame) -> go.Figure:
    """Interactive scatter: publication date × relevance score, sized by citations."""
    try:
        df = sorted_results.copy()

        df["Date_Parsed"] = pd.to_datetime(df["date"], errors="coerce")
        min_valid = df["Date_Parsed"].min()
        if pd.isnull(min_valid):
            min_valid = pd.Timestamp.now()
        df["Date_Plot"] = df["Date_Parsed"].fillna(min_valid)

        df["citations"] = pd.to_numeric(df.get("citations", 0), errors="coerce").fillna(0)
        df["marker_size"] = np.log1p(df["citations"]) * 5 + 5

        # Build hover text
        df["hover"] = df.apply(
            lambda r: (
                f"<b>{r['title']}</b><br>"
                f"Source: {r['source']}<br>"
                f"Score: {r['score']:.3f}<br>"
                f"Citations: {int(r['citations'])}<br>"
                f"DOI: {r['doi']}"
            ),
            axis=1,
        )

        fig = px.scatter(
            df,
            x="Date_Plot",
            y="score",
            size="marker_size",
            color="source",
            hover_name="title",
            hover_data={
                "Date_Plot": False,
                "score": ":.3f",
                "citations": True,
                "marker_size": False,
                "source": True,
            },
            labels={"score": "Relevance Score", "Date_Plot": "Publication Date"},
            title="Publication Date vs Relevance Score",
            color_discrete_map={
                "PubMed": "#3679ae",
                "BioRxiv": "#e07b39",
                "MedRxiv": "#3aad9c",
                "arXiv": "#b5412b",
            },
        )

        # Set per-trace hover so <extra> shows the source name, not a raw color string.
        # px.scatter creates one trace per source category, so we patch each individually.
        for trace in fig.data:
            trace.update(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Score: %{y:.3f}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Citations: %{customdata[0]}<br>"
                    f"<extra>{trace.name}</extra>"
                )
            )

        fig.update_layout(
            legend=dict(title="Source", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode="closest",
        )

        if df["Date_Parsed"].isnull().any():
            fig.add_annotation(
                text="Note: some points have inferred dates (metadata missing)",
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )

        return fig
    except Exception as e:
        st.error(f"Plot error: {e}")
        return go.Figure()


def generate_bibtex(df: pd.DataFrame) -> str:
    """Generates a BibTeX string from a results DataFrame."""
    entries = []

    for idx, row in df.iterrows():
        authors = str(row.get("authors", "Unknown")).replace(";", " and")
        authors = authors.replace("[", "").replace("]", "").replace("'", "")

        date_str = str(row.get("date", ""))
        match = re.search(r"\d{4}", date_str)
        year = match.group(0) if match else "n.d."

        first_author = re.sub(r"\W+", "", authors.split(" ")[0].split(",")[0].strip()) or "Anon"

        title = str(row.get("title", "No Title"))
        title_words = re.sub(r"[^\w\s]", "", title).split()
        first_word = title_words[0] if title_words else "Untitled"

        key = f"{first_author}{year}{first_word}_{idx}"

        source = str(row.get("source", "")).lower()
        journal = str(row.get("journal", ""))
        if journal.lower() in ("nan", ""):
            journal = source.capitalize()

        entry = f"@article{{{key},\n"
        entry += f"  author = {{{authors}}},\n"
        entry += f"  title = {{{title}}},\n"
        entry += f"  journal = {{{journal}}},\n"
        entry += f"  year = {{{year}}},\n"

        doi = row.get("doi")
        if doi and "10." in str(doi):
            clean = str(doi).strip()
            entry += f"  doi = {{{clean}}},\n"
            entry += f"  url = {{https://doi.org/{clean}}},\n"

        entry += "}\n"
        entries.append(entry)

    return "\n".join(entries)


def render_footer():
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    c1.markdown(
        """
        <div style='text-align: center;'>
            <b>[MSS] Developed by
            <a href="https://www.zylalab.org/" target="_blank">Dawid Zyla</a></b>
            &nbsp;|&nbsp;
            <a href="https://github.com/dzyla/pubmed_search" target="_blank">Source on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        """
        <div style="text-align: center; margin-top: 5px;">
            <a href="https://www.buymeacoffee.com/dzyla" target="_blank"
                style="
                    background-color: #3679ae;
                    color: #ffffff;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-family: Lato, sans-serif;
                    font-size: 16px;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                ">
                Buy me a coffee
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
