import logging
import sys
import streamlit as st
import pandas as pd
import concurrent.futures
from datetime import date, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import config_loader
import utils
from api_handler import get_query_embeddings, EmbeddingError
import search_logic
import ui_components
import gemini_handler
import paper_links

LOGGER = logging.getLogger(__name__)

# --- Page Setup ---
st.set_page_config(page_title="MSS", page_icon="📜")
ui_components.define_style()

# --- Config path: CLI arg > default production path ---
# Run with:  streamlit run pbmss_app.py -- --config ./config_mss.yaml
_DEFAULT_CONFIG = str(Path(__file__).parent / "config_mss.yaml")
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
            config_data = config_loader.load_configs_and_db_sizes(_config_path)
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

    # Answer the last user turn if it has no reply yet — covers both a typed
    # prompt and a suggested-question click (which only appends and reruns).
    history = st.session_state.chat_history
    if history and history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response_text = gemini_handler.chat_with_context(
                    history, history[-1]["content"], sorted_results, ai_api_key
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})




# ---------------------------------------------------------------------------
# Search Form
# ---------------------------------------------------------------------------

# A shared link (?q=…) pre-fills the query and runs the search once.
_url_query = st.query_params.get("q", "")
if "query_text" not in st.session_state:
    st.session_state["query_text"] = _url_query
auto_search = bool(_url_query) and not st.session_state.get("auto_searched")
st.session_state["auto_searched"] = True

with st.form("search_form"):
    query = st.text_area("Enter your search query:", key="query_text", max_chars=8192, height=128, help="Describe what you’re looking for. Semantic search performs best with full sentences or descriptive paragraphs. Avoid using isolated keywords; instead, try pasting an abstract or a detailed question to get the most relevant results.", placeholder="e.g. Structural insight into antibody-mediated neutralization of measles virus by a potent monoclonal antibody")
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input(
            "Number of results:", min_value=1, max_value=50, value=10
        )
    with col2:
        use_high_quality = st.toggle(
            "High-quality filter",
            value=True,
            help="Filter out entries with very short or missing abstracts.",
        )

    if st.session_state.get("date_filter_toggle", False):
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("From", value=date(2020, 1, 1), min_value=date(1800, 1, 1),
                                    max_value=date.today(), format="YYYY-MM-DD")
        end_d = col_d2.date_input("To", value=date.today(), min_value=date(1800, 1, 1),
                                  max_value=date.today(), format="YYYY-MM-DD")
        if start_d and end_d and start_d > end_d:
            start_d, end_d = end_d, start_d
        start_date_str = start_d.isoformat() if start_d else None
        end_date_str = end_d.isoformat() if end_d else None
    else:
        start_date_str = None
        end_date_str = None

    submitted = st.form_submit_button("Search :material/search:", type="primary") or auto_search

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

    # Shareable URL for this search (long pasted abstracts are left out of the URL)
    if len(query) <= 2000:
        st.query_params["q"] = query
    else:
        st.query_params.pop("q", None)

    if len(query) > 2000:
        st.info(
            "Long query: the embedding model reads only about the first 512 tokens "
            "(~2,000 characters), so text beyond that does not affect the results."
        )

    with st.status("Searching…", expanded=True) as status:
        t0 = datetime.now()

        st.write(":material/update: Checking for database updates…")
        any_updates = search_logic.trigger_database_updates(configs)
        if any_updates:
            st.write(":material/autorenew: Updates found! Updating database…")
            config_loader.load_configs_and_db_sizes.clear()
            config_data = config_loader.load_configs_and_db_sizes(_config_path)
            configs = config_data["configs"]

        st.write(":material/model_training: Encoding query…")
        try:
            query_packed, query_float = get_query_embeddings(query)
        except EmbeddingError as exc:
            st.error(str(exc))
            status.update(label="Embedding failed.", state="error", expanded=False)
            query_packed = None

        if query_packed is not None:
            st.write(":material/manage_search: Scanning vector indexes…")
            final_results = search_logic.combined_search_orchestrator(
                query_packed,
                configs,
                top_k=num_to_show,
                start_date=start_date_str,
                end_date=end_date_str,
                use_high_quality=use_high_quality,
                query_float=query_float,
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

    # Clean DOIs (fast, no network)
    if st.session_state.get("clean_doi") is None:
        st.session_state["clean_doi"] = [utils.get_clean_doi(d) for d in all_doi]
    sorted_results["doi"] = st.session_state["clean_doi"]
    sorted_results["rank"] = range(1, len(sorted_results) + 1)   # relevance rank

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

    by_source = sorted_results["source"].value_counts()
    col_a.markdown(
        f"#### {len(sorted_results)} results\n"
        + " · ".join(f"{name} {count}" for name, count in by_source.items())
    )

    # --- Tabs ---
    tabs = st.tabs(["Results List", "Bibliography", "Chat with Papers"])

    with tabs[0]:
        for idx, row in sorted_results.iterrows():
            citations = row["citations"]
            citation_str = f"{citations:,}" if citations is not None else "…"
            badges = paper_links.badges(row)
            retracted = paper_links.is_retracted(row)

            meta_parts = [str(row["date"] or "n.d."), row["source"], f"cited {citation_str}"]
            meta_parts += [b for b in badges if b != "Retracted"]
            expander_title = (
                f"{idx + 1}\\. {'⚠️ RETRACTED — ' if retracted else ''}{row['title']}\n\n"
                f"_{' · '.join(meta_parts)}_"
            )

            with st.expander(expander_title):
                if retracted:
                    st.error("This article has been retracted.", icon=":material/report:")
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Relevance rank", f"#{row['rank']}", help=f"Similarity {row['score']:.3f}")
                c_b.metric("Source", row["source"])
                c_c.metric("Citations", citation_str)
                st.markdown(f"**Authors:** {row['authors']}")
                c_d, c_e = st.columns(2)
                c_d.markdown(f"**Date:** {row['date']}")
                c_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")
                st.markdown(f"**Abstract:**\n{row['abstract']}")

                links = paper_links.build_links(row)
                if links:
                    st.markdown(" · ".join(f"**[{label}]({url})**" for label, url in links))

        st.markdown("---")
        fig_scatter = ui_components.plot_score_vs_year(sorted_results)
        st.plotly_chart(fig_scatter, width="stretch")

    with tabs[1]:
        st.markdown("### Export Bibliography")
        bibtex_str = ui_components.generate_bibtex(sorted_results)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        col_ex1.download_button(
            label="BibTeX (.bib)",
            data=bibtex_str,
            file_name=f"mss_search_{stamp}.bib",
            mime="text/x-bibtex",
            type="primary",
        )
        col_ex2.download_button(
            label="RIS (Zotero, EndNote)",
            data=ui_components.generate_ris(sorted_results),
            file_name=f"mss_search_{stamp}.ris",
            mime="application/x-research-info-systems",
        )
        col_ex3.download_button(
            label="CSV",
            data=ui_components.results_csv(sorted_results),
            file_name=f"mss_search_{stamp}.csv",
            mime="text/csv",
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
