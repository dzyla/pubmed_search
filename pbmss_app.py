import streamlit as st
import pandas as pd
import concurrent.futures
from datetime import datetime
import logging

import config_loader
import utils
import api_handler
import search_logic
import ui_components
import gemini_handler

LOGGER = logging.getLogger(__name__)

# --- Page Setup ---
st.set_page_config(page_title="MSS", page_icon="📜")
ui_components.define_style()

# --- Config path: CLI arg > default production path ---
# Run with:  streamlit run pbmss_app.py -- --config ./config_mss.yaml
import sys

_DEFAULT_CONFIG = "/root/pubmed_search/config_mss.yaml"
_config_path = _DEFAULT_CONFIG
_args = sys.argv[1:]  # Streamlit strips its own flags; remaining args are ours
for i, arg in enumerate(_args):
    if arg in ("--config", "-c") and i + 1 < len(_args):
        _config_path = _args[i + 1]
        break

# --- Load Config & State ---
config_data = config_loader.load_configs_and_db_sizes(_config_path)
configs = config_data["configs"]

# --- Startup: DB update check + background FAISS warm-up ---
if "updates_checked" not in st.session_state:
    with st.spinner("Checking for new manuscript embeddings…"):
        any_updates = search_logic.trigger_database_updates(configs)

        if any_updates:
            st.toast("New data detected! Database updated.", icon="🔄")
            config_loader.load_configs_and_db_sizes.clear()
            config_data = config_loader.load_configs_and_db_sizes()
            configs = config_data["configs"]

    # Kick off background FAISS index warm-up so the first search is faster.
    search_logic.warm_up_indexes(configs, background=True)
    st.session_state["updates_checked"] = True

# --- Status & Logo ---
last_biorxiv_date = utils.report_dates_from_metadata(config_data["biorxiv_config"])
ui_components.render_logo(
    last_biorxiv_date,
    config_data["biorxiv_db_size"],
    config_data["pubmed_db_size"],
    config_data["medrxiv_db_size"],
    config_data["arxiv_db_size"],
)


# ---------------------------------------------------------------------------
# Chat fragment (re-runs independently of main app)
# ---------------------------------------------------------------------------

@st.fragment
def render_chat_interface(sorted_results, ai_api_key):
    st.markdown("### 💬 Chat with Search Results")

    if st.session_state.get("ai_questions"):
        st.markdown("**Suggested Questions:**")
        q_cols = st.columns(len(st.session_state["ai_questions"]))
        for i, q in enumerate(st.session_state["ai_questions"]):
            if q_cols[i].button(q, key=f"sug_q_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun(scope="fragment")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about these papers…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response_text = gemini_handler.chat_with_context(
                    st.session_state.chat_history, prompt, sorted_results, ai_api_key
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})




# ---------------------------------------------------------------------------
# Search Form
# ---------------------------------------------------------------------------

with st.form("search_form"):
    query = st.text_area("Enter your search query:", max_chars=8192, height=128, help="Describe what you’re looking for. Semantic search performs best with full sentences or descriptive paragraphs. Avoid using isolated keywords; instead, try pasting an abstract or a detailed question to get the most relevant results.", placeholder="e.g. Structural insight into antibody-mediated neutralization of measles virus by a potent monoclonal antibody")
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input(
            "Number of results:", min_value=1, max_value=100, value=10
        )
    with col2:
        use_high_quality = st.toggle(
            "High-quality filter",
            value=True,
            help="Filter out entries with very short or missing abstracts.",
        )

    if st.session_state.get("date_filter_toggle", False):
        col_d1, col_d2 = st.columns(2)
        start_raw = col_d1.text_input("Start Date", value="2020-01-01", placeholder="YYYY-MM-DD")
        end_raw   = col_d2.text_input("End Date",   value=datetime.today().strftime("%Y-%m-%d"), placeholder="YYYY-MM-DD")
        try:
            datetime.strptime(start_raw, "%Y-%m-%d")
            datetime.strptime(end_raw,   "%Y-%m-%d")
            start_date_str, end_date_str = start_raw, end_raw
        except ValueError:
            st.warning("Dates must be in YYYY-MM-DD format (e.g. 2020-01-01).")
            start_date_str = end_date_str = None
    else:
        start_date_str = None
        end_date_str = None

    submitted = st.form_submit_button("Search :material/search:", type="primary")

# --- Toggles outside form ---
col_t1, col_t2 = st.columns(2)
use_ai = col_t1.toggle("AI Summary & Chat", key="use_ai_checkbox")
col_t2.toggle("Date Filter", value=False, key="date_filter_toggle")

if use_ai:
    ai_api_key = col_t1.text_input(
        "Google AI API Key",
        type="password",
        help="Get key at https://aistudio.google.com/apikey",
    )
else:
    ai_api_key = None

st.markdown("---")

col_a, col_b = st.columns(2)

# ---------------------------------------------------------------------------
# Phase 1: Search
# ---------------------------------------------------------------------------

if submitted and query:
    st.session_state["chat_history"] = []
    st.session_state["ai_summary"] = None
    st.session_state["ai_questions"] = []
    st.session_state["citations"] = None
    st.session_state["doi_list"] = None
    st.session_state["clean_doi"] = None
    st.session_state["full_text_links"] = None

    with st.status("Searching…", expanded=True) as status:
        t0 = datetime.now()

        st.write(":material/model_training: Encoding query…")
        query_packed = api_handler.get_query_embedding_packed(query)

        if query_packed is not None:
            st.write(":material/manage_search: Scanning vector indexes…")
            final_results = search_logic.combined_search_orchestrator(
                query_packed,
                configs,
                top_k=num_to_show,
                start_date=start_date_str,
                end_date=end_date_str,
                use_high_quality=use_high_quality,
            )
            elapsed = (datetime.now() - t0).total_seconds()
            status.update(
                label=f"Search complete — {elapsed:.2f}s  |  {len(final_results)} results",
                state="complete",
                expanded=False,
            )
            st.session_state["final_results"] = final_results
            st.session_state["search_query"] = query
            st.session_state["num_to_show"] = num_to_show
        else:
            status.update(label="Embedding failed.", state="error", expanded=False)
            final_results = pd.DataFrame()
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

# ---------------------------------------------------------------------------
# Phase 2: Display
# ---------------------------------------------------------------------------

if not final_results.empty:

    # Sort control
    sort_option = col_b.radio(
        "Sort by:",
        options=["Relevance", "Date", "Citations"],
        key="sort_option",
        horizontal=True,
    )

    sorted_results = st.session_state["final_results"].copy()
    all_doi = sorted_results["doi"].tolist()

    # Reset per-search cached metadata when DOI list changes
    if st.session_state.get("doi_list") != all_doi:
        st.session_state["doi_list"] = all_doi
        st.session_state["citations"] = None
        st.session_state["clean_doi"] = None
        st.session_state["full_text_links"] = None

    # Clean DOIs (fast, no network)
    if st.session_state.get("clean_doi") is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            clean_doi_list = list(executor.map(utils.get_clean_doi, all_doi))
        st.session_state["clean_doi"] = clean_doi_list
    sorted_results["doi"] = st.session_state["clean_doi"]

    # Full-text links (fast, no network)
    if st.session_state.get("full_text_links") is None:
        sorted_results = utils.precalculate_full_text_links_parallel(sorted_results)
        st.session_state["full_text_links"] = sorted_results["full_text_link"].tolist()
    else:
        sorted_results["full_text_link"] = st.session_state["full_text_links"]

    # Citations — use cached value or placeholder while fetching
    citations_ready = st.session_state.get("citations") is not None
    sorted_results["citations"] = (
        st.session_state["citations"] if citations_ready else [None] * len(sorted_results)
    )

    # Apply sort
    if sort_option == "Date":
        sorted_results["_date_parsed"] = pd.to_datetime(sorted_results["date"], errors="coerce")
        sorted_results = sorted_results.sort_values("_date_parsed", ascending=False).reset_index(drop=True)
        sorted_results.drop(columns=["_date_parsed"], inplace=True)
    elif sort_option == "Citations" and citations_ready:
        sorted_results = sorted_results.sort_values("citations", ascending=False).reset_index(drop=True)
    else:
        sorted_results = sorted_results.sort_values("score", ascending=False).reset_index(drop=True)

    full_text_link_n = sorted_results["full_text_link"].notnull().sum()
    col_a.markdown(
        f"#### Search results\n:material/import_contacts: {full_text_link_n} full-text available"
    )

    # --- Tabs ---
    tabs = st.tabs(["Results List", "Bibliography", "Chat with Papers"])

    with tabs[0]:
        for idx, row in sorted_results.iterrows():
            citations = row["citations"]
            citation_str = f"{citations:,}" if citations is not None else "…"
            doi_link = (
                f"https://doi.org/{row['doi']}"
                if "arxiv.org" not in str(row["doi"])
                else row["doi"]
            )
            full_text_link = row.get("full_text_link")
            final_link = full_text_link if full_text_link is not None else doi_link
            ft_icon = ":material/import_contacts:" if full_text_link else ""

            expander_title = (
                f"{idx + 1}\\. {row['title']}\n\n"
                f"_(Score: {row['score']:.2f} | Date: {row['date']} | Citations: {citation_str})_"
            )
            if ft_icon:
                expander_title += f" | {ft_icon}"

            with st.expander(expander_title):
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Score", f"{row['score']:.3f}")
                c_b.metric("Source", row["source"])
                c_c.metric("Citations", citation_str)
                st.markdown(f"**Authors:** {row['authors']}")
                c_d, c_e = st.columns(2)
                c_d.markdown(f"**Date:** {row['date']}")
                c_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")
                st.markdown(f"**Abstract:**\n{row['abstract']}")

                link_cols = st.columns(3)
                if full_text_link:
                    link_cols[0].markdown(f"**[:material/import_contacts: Full Text]({final_link})**")
                link_cols[1].markdown(f"**[:material/link: Publisher Site]({doi_link})**")

        st.markdown("---")
        fig_scatter = ui_components.plot_score_vs_year(sorted_results)
        st.plotly_chart(fig_scatter, width="stretch")

    with tabs[1]:
        st.markdown("### Export Bibliography")
        bibtex_str = ui_components.generate_bibtex(sorted_results)
        col_ex1, col_ex2 = st.columns([1, 2])
        with col_ex1:
            st.download_button(
                label="Download .bib file",
                data=bibtex_str,
                file_name=f"mss_search_{datetime.now().strftime('%Y%m%d_%H%M')}.bib",
                mime="text/x-bibtex",
                type="primary",
            )
        with st.expander("Preview BibTeX"):
            st.code(bibtex_str, language="latex")

    with tabs[2]:
        if not use_ai:
            st.info("Enable 'AI Summary & Chat' and provide an API key.")
        elif not ai_api_key:
            st.warning("Please provide a Google AI API key.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            render_chat_interface(sorted_results, ai_api_key)

    # --- Lazy citation fetch ---
    # Because Streamlit renders elements as it encounters them (top-to-bottom),
    # the expanders above are already visible when this spinner appears.
    # On first load: fetch and rerun so expander headers show actual counts.
    # On subsequent loads: session_state["citations"] is already set → no fetch.
    if not citations_ready:
        with st.spinner("Fetching citation counts…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                all_citations = list(executor.map(utils.get_citation_count, all_doi))
        st.session_state["citations"] = all_citations
        st.rerun()  # Update expander titles and sort (if Citations sort selected)

    # --- Phase 3: Async AI Summary ---
    if use_ai and ai_api_key and not sorted_results.empty:
        if st.session_state.get("ai_summary") is None:
            with st.status("🤖 Generating AI Analysis…", expanded=True) as status:
                st.write("Summarising abstracts…")
                summary = gemini_handler.summarize_search_results(sorted_results, ai_api_key)
                st.write("Generating suggested questions…")
                questions = gemini_handler.generate_example_questions(sorted_results, ai_api_key)
                st.session_state["ai_summary"] = summary
                st.session_state["ai_questions"] = questions
                status.update(label="AI Analysis Complete", state="complete", expanded=True)
                st.rerun()

        if st.session_state.get("ai_summary"):
            st.markdown("---")
            st.markdown("### 🤖 AI Summary")
            st.info(st.session_state["ai_summary"])

elif submitted and final_results.empty:
    st.warning("#### No results found. Try a different query.")

ui_components.render_footer()
