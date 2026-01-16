import os
import json
import time
import uuid
import gc
import sqlite3
import logging
import concurrent.futures
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import google.genai as genai
from sklearn.cluster import MiniBatchKMeans

# Import Modules
from data_handler import (
    create_session,
    get_citation_count_cached,
    get_link_info_cached,
    get_clean_doi,
    precalculate_full_text_links_parallel,
    report_dates_from_metadata,
    load_configs_and_db_sizes,
    get_citation_count
)
from search_engine import (
    combined_search,
    log_time
)
from ui_components import (
    define_style,
    logo,
    load_pumap_model_and_image
)

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# Session & DB
# =============================================================================
def get_current_active_users(db_path: str = "sessions_history.db", timeout: int = 300) -> int:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id
    current_time = int(time.time())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            last_seen INTEGER
        )
    """)
    conn.commit()
    cursor.execute("""
        INSERT INTO session_history (session_id, last_seen)
        VALUES (?, ?)
    """, (session_id, current_time))
    conn.commit()
    expiration_time = current_time - timeout
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT session_id, MAX(last_seen) AS last_seen
            FROM session_history
            GROUP BY session_id
            HAVING last_seen >= ?
        ) AS active_sessions
    """, (expiration_time,))
    active_count = cursor.fetchone()[0]
    conn.close()
    return active_count

def check_update_status():
    try:
        import requests
        response = requests.get("http://localhost:8001/status", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "update running":
                return True
    except Exception:
        return None
    return None

# =============================================================================
# AI Helpers
# =============================================================================
LLM_PROMPT_SUMMARY = """You are a research assistant tasked with summarizing a collection of abstracts extracted from a database of 39 million academic entries. Your goal is to synthesize a concise, clear, and insightful summary that captures the main themes, common findings, and noteworthy trends across the abstracts.

Instructions:
- Read the provided abstracts carefully.
- Shortly digest each abstract's content.
- Identify and list the central topics and findings.
- Highlight any recurring patterns or shared insights.
- Keep the summary succinct (1-3 short paragraphs) without using overly technical jargon.
- Do not include any external links or references.
- Format the response in markdown, using bullet points where appropriate.

Now, review the abstracts provided below and generate your summary.
"""

def summarize_abstract(abstracts, instructions, api_key, model_name="gemini-2.0-flash-lite-preview-02-05"):
    from google.genai import types
    if not api_key:
        return "API key not provided. Please obtain your own API key at https://aistudio.google.com/apikey"

    try:
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        formatted_text = "\n".join(f"{idx + 1}. {abstract}" for idx, abstract in enumerate(abstracts))
        prompt = f"{instructions}\n\n{formatted_text}"

        config = types.GenerateContentConfig(temperature=1, top_p=0.95, top_k=64, max_output_tokens=8192)
        response = client.models.generate_content(model=model_name, contents=types.Part.from_text(text=prompt), config=config)
        return response.text
    except Exception as e:
        return f"Google Flash model error: {e}"

def chat_with_papers(abstracts, user_question, api_key, model_name="gemini-2.0-flash-lite-preview-02-05"):
    from google.genai import types
    if not api_key:
        return "API key required."

    try:
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        context = "\n\n".join([f"Paper {i+1}: {a}" for i, a in enumerate(abstracts)])
        prompt = f"Based on the following academic abstracts, answer the user's question.\n\nContext:\n{context}\n\nQuestion: {user_question}"

        config = types.GenerateContentConfig(temperature=0.7, top_p=0.95, top_k=64, max_output_tokens=2048)
        response = client.models.generate_content(model=model_name, contents=types.Part.from_text(text=prompt), config=config)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# =============================================================================
# Clustering
# =============================================================================
def perform_clustering(df, n_clusters=5):
    if "embedding" not in df.columns or df.empty:
        return df
    try:
        embeddings = np.stack(df["embedding"].values)
        if len(embeddings) < n_clusters:
            n_clusters = len(embeddings)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        labels = kmeans.fit_predict(embeddings)
        df["cluster"] = labels
        return df
    except Exception as e:
        LOGGER.error(f"Clustering failed: {e}")
        df["cluster"] = -1
        return df

# =============================================================================
# Main App
# =============================================================================

st.set_page_config(page_title="MSS", page_icon="📜", layout="centered")

# Load configs
results = load_configs_and_db_sizes()
pubmed_config = results["pubmed_config"]
biorxiv_config = results["biorxiv_config"]
medrxiv_config = results["medrxiv_config"]
arxiv_config = results["arxiv_config"]
configs = results["configs"]
pubmed_db_size = results["pubmed_db_size"]
biorxiv_db_size = results["biorxiv_db_size"]
medrxiv_db_size = results["medrxiv_db_size"]
arxiv_db_size = results["arxiv_db_size"]

define_style()

try:
    last_biorxiv_date = report_dates_from_metadata(biorxiv_config["metadata_path"]).get("second_date_from_last_part", "N/A")
except:
    last_biorxiv_date = "N/A"

logo(last_biorxiv_date, biorxiv_db_size, pubmed_db_size, medrxiv_db_size, arxiv_db_size)

status = check_update_status()
if status:
    st.info("Database update in progress. Search might be slow...")

# --- Search Form ---
with st.form("search_form"):
    query = st.text_area("Enter your search query:", max_chars=8192, height=68)
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input("Number of results to show:", min_value=1, max_value=50, value=10)
    with col2:
        use_high_quality = st.toggle("Use high-quality search?", value=True, help="Enable for more accurate results (only entries with full abstracts and titles).")
        rerank_toggle = st.toggle("Use AI Reranking (Slower but better)", value=True, help="Re-scores top results using a Cross-Encoder for better relevance.")

    # Date Filter
    if st.session_state.get("date_filter_toggle", False):
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date_input = st.date_input("Start Date", value=datetime(2020, 1, 1))
        with col_date2:
            end_date_input = st.date_input("End Date", value=datetime.today())
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        end_date_str = end_date_input.strftime("%Y-%m-%d")
    else:
        start_date_str = None
        end_date_str = None

    col3 = st.container()
    with col3:
        if st.session_state.get("use_ai_checkbox"):
            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                ai_api_provided = st.text_input("Google AI API Key", value="", help="Obtain your own API key at https://aistudio.google.com/apikey", type="password")
            with ai_col2:
                ai_model_name = st.selectbox("AI Model", options=["gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"], index=0)
                ai_abstracts_count = st.number_input("Abstracts for Summary", min_value=1, max_value=20, value=9)
        else:
            ai_api_provided = None
            ai_model_name = "gemini-2.0-flash-lite-preview-02-05"
            ai_abstracts_count = 9

    submitted = st.form_submit_button("Search :material/search:", type="primary")

# --- Filters & Toggles ---
col_f1, col_f2 = st.columns(2)
use_ai = col_f1.toggle("Use AI generated summary?", key="use_ai_checkbox")
activate_date_filter = col_f2.toggle("Use Date Filter", value=False, key="date_filter_toggle")
STATUS = st.empty()

st.markdown("---")

if submitted and query:
    with st.spinner("Searching & Reranking..."):
        search_start_time = datetime.now()
        final_results = combined_search(query, configs, top_show=num_to_show, precision="ubinary",
                                        use_high_quality=use_high_quality,
                                        start_date=start_date_str, end_date=end_date_str,
                                        rerank=rerank_toggle)

        # Clustering
        if not final_results.empty:
            final_results = perform_clustering(final_results)

        total_time = datetime.now() - search_start_time
        st.markdown(f"<h6 style='text-align: center; color: #7882af;'>Search completed in {total_time.total_seconds():.2f} seconds</h6>", unsafe_allow_html=True)
        st.session_state["final_results"] = final_results
        st.session_state["search_query"] = query
        st.session_state["num_to_show"] = num_to_show
        st.session_state["use_ai"] = use_ai
        st.session_state["ai_api_provided"] = ai_api_provided
        st.session_state["use_high_quality"] = use_high_quality
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

if not final_results.empty:
    
    # Sort Controls
    col_sort1, col_sort2 = st.columns([2, 1])
    sort_option = col_sort1.segmented_control(
        "Sort results by:", 
        options=["Relevance", "Publication Date", "Citations"], 
        key="sort_option"
    )

    # Export Tools
    def to_bibtex(df):
        bib = ""
        for i, row in df.iterrows():
            bib += f"@article{{mss_{i},\n"
            bib += f"  title = {{{row.get('title', '')}}},\n"
            bib += f"  author = {{{row.get('authors', '')}}},\n"
            bib += f"  journal = {{{row.get('journal', '')}}},\n"
            bib += f"  year = {{{str(row.get('date', ''))[:4]}}},\n"
            bib += f"  doi = {{{row.get('doi', '')}}}\n"
            bib += "}\n\n"
        return bib

    csv = final_results.to_csv(index=False).encode('utf-8')
    bibtex = to_bibtex(final_results)

    with col_sort2:
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='mss_results.csv',
            mime='text/csv',
        )
        st.download_button(
            label="Download BibTeX",
            data=bibtex,
            file_name='mss_results.bib',
            mime='text/plain',
        )

    # Ensure sorted_results is defined
    sorted_results = st.session_state["final_results"].copy()

    # Clean DOIs upfront (fast)
    sorted_results["doi"] = sorted_results["doi"].apply(get_clean_doi)

    # Sorting Logic
    if sort_option == "Publication Date":
        sorted_results["date_parsed"] = pd.to_datetime(sorted_results["date"], errors="coerce")
        sorted_results = sorted_results.sort_values(by="date_parsed", ascending=False).reset_index(drop=True)
        sorted_results.drop(columns=["date_parsed"], inplace=True)
    elif sort_option == "Citations":
        if "citations" not in sorted_results.columns:
             with st.spinner("Fetching citations for sorting..."):
                 with ThreadPoolExecutor(max_workers=10) as executor:
                     citations_map = list(executor.map(get_citation_count_cached, sorted_results["doi"]))
                 sorted_results["citations"] = citations_map
        sorted_results = sorted_results.sort_values(by="citations", ascending=False).reset_index(drop=True)
    else:
        # Default relevance
        if "rerank_score" in sorted_results.columns:
            sorted_results = sorted_results.sort_values(by="rerank_score", ascending=False).reset_index(drop=True)
        else:
            sorted_results = sorted_results.sort_values(by="score", ascending=True).reset_index(drop=True)

    displayed_rows = sorted_results.to_dict('records')

    # Pre-calculate metadata (citations & links) in parallel BEFORE rendering loop
    # This ensures "Full Text Count" is accurate and UI is stable/numbered as requested.
    with st.spinner("Fetching metadata..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {}
            for i, row in enumerate(displayed_rows):
                future_cit = executor.submit(get_citation_count_cached, row['doi'])
                row_dict = {"source": row.get("source"), "version": row.get("version"), "doi": row.get("doi")}
                future_link = executor.submit(get_link_info_cached, row_dict)
                future_to_idx[future_cit] = (i, 'citations')
                future_to_idx[future_link] = (i, 'full_text_link')

            for future in concurrent.futures.as_completed(future_to_idx):
                idx, kind = future_to_idx[future]
                res = future.result()
                displayed_rows[idx][kind] = res

    # Count full text (Now accurate)
    ft_count = sum(1 for r in displayed_rows if r.get("full_text_link"))
    st.markdown(f":material/import_contacts: **{ft_count}** Full text documents available (free)")

    abstracts_for_summary = []

    # Render Results
    for idx, row in enumerate(displayed_rows):
        abstracts_for_summary.append(row["abstract"])

        citations = row.get("citations", 0)
        if citations is None: citations = 0

        doi_val = row["doi"]
        doi_link = f"https://doi.org/{doi_val}" if "arxiv.org" not in str(doi_val) else doi_val
        full_text_link = row.get("full_text_link")

        final_link = full_text_link if full_text_link is not None else doi_link
        full_text_notification = ":material/import_contacts:" if full_text_link is not None else ""
        has_full_text = full_text_link is not None

        cluster_id = row.get("cluster", -1)
        cluster_badge = f" [Cluster {cluster_id}]" if cluster_id != -1 else ""

        expander_title = (
            f"{idx + 1}\. {row['title']}{cluster_badge}\n\n"
            f"_(Score: {row.get('quality', 0):.2f} | Date: {row['date']} | Citations: {citations})_"
        )
        if full_text_notification:
            expander_title += f" | {full_text_notification}"

        with st.expander(expander_title):
            col_a, col_b, col_c = st.columns(3)

            score_text = f"**Relative Score:** {row.get('quality', 0):.2f}"
            if "rerank_score" in row and pd.notnull(row["rerank_score"]):
                score_text += f" | **Rerank:** {row['rerank_score']:.2f}"

            col_a.markdown(score_text)
            col_b.markdown(f"**Source:** {row['source']}")
            col_c.markdown(f"**Citations:** {citations}")
            st.markdown(f"**Authors:** {row['authors']}")
            col_d, col_e = st.columns(2)
            col_d.markdown(f"**Date:** {row['date']}")
            col_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")

            st.markdown(f"**Abstract:**\n{row['abstract']}")

            link_cols = st.columns(3)
            if has_full_text:
                link_cols[0].markdown(f"**[:material/import_contacts: Read Full Text]({final_link})**")
            link_cols[1].markdown(f"**[:material/link: View on Publisher Website]({doi_link})**")

    # Visualizations
    try:
        sorted_results["Date"] = pd.to_datetime(sorted_results["date"])
        # Ensure citations are in DF for plotting
        if "citations" not in sorted_results.columns:
             sorted_results["citations"] = [r.get("citations", 0) for r in displayed_rows]

        plot_data = sorted_results.copy()
        plot_data["marker_size"] = np.log1p(plot_data["citations"].fillna(0)) * 5 + 3

        if "cluster" in plot_data.columns:
            color_col = "cluster"
        else:
            color_col = "source"

        fig_scatter = px.scatter(
            plot_data,
            x="Date",
            y="quality",
            size="marker_size",
            hover_data={"title": True, "doi": True, "citations": True, "marker_size": False},
            color=color_col,
            title=f"Publication Dates and Relative Score (Colored by {color_col})",
        )
    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        fig_scatter = go.Figure()

    tabs = st.tabs(["Score vs Year", "Abstract Map", "References", "Chat with Papers"])
    with tabs[0]:
        st.plotly_chart(fig_scatter, width="stretch")
    with tabs[1]:
        pumap, hist_data = load_pumap_model_and_image("param_umap_model.h5", "hist2d.npz")
        try:
            hist = hist_data["hist"]
            xedges = hist_data["xedges"]
            yedges = hist_data["yedges"]
            x_min, x_max = float(xedges[0]), float(xedges[-1])
            y_min, y_max = float(yedges[0]), float(yedges[-1])
            
            heatmap = go.Heatmap(
                z=np.sqrt(hist.T),
                x=xedges,
                y=yedges,
                colorscale="dense",
                reversescale=False,
                showscale=False,
                opacity=1,
            )

            if "embedding" in sorted_results.columns:
                # Need to run UMAP transform
                # We need stored_raw_embeddings logic if we want to avoid re-run.
                # Simplification: Just run it if available.
                current_raw_embeddings = np.stack(sorted_results["embedding"].values)
                current_tensor = torch.from_numpy(current_raw_embeddings).float()
                # Run on CPU
                new_2d = pumap.transform(current_tensor)

                scatter_x = new_2d[:, 0]
                scatter_y = new_2d[:, 1]

                scatter = go.Scatter(
                    x=scatter_x,
                    y=scatter_y,
                    mode="markers",
                    marker=dict(
                        color=sorted_results["cluster"] if "cluster" in sorted_results.columns else "blue",
                        size=8,
                        line=dict(width=1, color="white"),
                        opacity=0.9,
                    ),
                    text=sorted_results["title"],
                    hoverinfo="text",
                    name="Found Abstracts",
                )
                fig_abstract_map = go.Figure(data=[heatmap, scatter])
            else:
                fig_abstract_map = go.Figure(data=[heatmap])

            fig_abstract_map.update_layout(
                title="Abstract Map",
                xaxis=dict(range=[x_min, x_max], showgrid=False),
                yaxis=dict(range=[y_min, y_max], showgrid=False),
                plot_bgcolor="white",
                height=600,
            )
            st.plotly_chart(fig_abstract_map, width="stretch")
        except Exception as e:
            st.error(f"Error loading map: {e}")
    
    with tabs[2]:
        st.markdown("#### References")
        for idx, row in enumerate(displayed_rows):
            st.markdown(f"**{idx + 1}\.** {row['authors']}<br>{row['title']} {row['date']} {row['journal']}<br>[{row['doi']}]({row.get('full_text_link', row['doi'])})", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("### Chat with these results")
        user_question = st.text_input("Ask a question about these papers:")
        if user_question and st.button("Ask AI"):
            if st.session_state.get("ai_api_provided"):
                # Use top 10 abstracts for context
                context_abstracts = abstracts_for_summary[:10]
                with st.spinner("AI is thinking..."):
                    answer = chat_with_papers(context_abstracts, user_question, st.session_state["ai_api_provided"], model_name=ai_model_name)
                st.markdown(answer)
            else:
                st.warning("Please provide an API Key in the search settings.")

    # Summary Generation (Bottom)
    if use_ai and abstracts_for_summary:
        st.markdown("---")
        with st.spinner("Generating AI summary..."):
            if st.session_state.get("ai_api_provided"):
                ai_count = st.session_state.get("ai_abstracts_count", 9)
                ai_model = st.session_state.get("ai_model_name", "gemini-2.0-flash-lite-preview-02-05")
                st.markdown(f"**AI Summary of top {ai_count} abstracts (Model: {ai_model}):**")
                summary_text = summarize_abstract(abstracts_for_summary[:ai_count], LLM_PROMPT_SUMMARY, st.session_state["ai_api_provided"], model_name=ai_model)
                st.markdown(summary_text)

elif submitted and final_results.empty:
    st.warning("#### No results found. Please try a different query.")

st.markdown("---")
c1, c2 = st.columns([2, 1])
c1.markdown("<div style='text-align: center;'><b>[MSS] Developed by <a href='https://www.dzyla.com/' target='_blank'>Dawid Zyla</a></b></div>", unsafe_allow_html=True)
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
                transition: background-color 0.2s ease;
            "
            onMouseOver="this.style.backgroundColor='#26557b'"
            onMouseOut="this.style.backgroundColor='#26557b'">
            Buy me a coffee
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
