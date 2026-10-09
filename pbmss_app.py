"""
Manuscript Search — Streamlit UI.

A thin client: every search goes to the backend (search_api.py) over HTTP, so
this process holds no model and no index. Run the backend first, then:

    streamlit run pbmss_app.py

Environment: MSS_BACKEND_URL (default http://127.0.0.1:8080), MSS_INTERNAL_KEY.
"""
import concurrent.futures
import logging
from datetime import date, datetime

import pandas as pd
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import backend_client
import gemini_handler
import ui_components
import ui_data

LOGGER = logging.getLogger(__name__)

SOURCE_OPTIONS = ["PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials"]
SOURCE_LABELS = ui_components.SOURCE_NAMES

st.set_page_config(page_title="Manuscript Search", page_icon="📜", layout="centered")
ui_components.apply_style()


@st.cache_data(ttl=300, show_spinner=False)
def corpus_stats() -> dict:
    return backend_client.stats()


ui_components.render_header(corpus_stats(), ui_data.get_current_active_users())


# ---------------------------------------------------------------------------
# Chat fragment (re-runs independently of the page)
# ---------------------------------------------------------------------------

@st.fragment
def render_chat_interface(results, ai_api_key):
    if st.session_state.get("ai_questions"):
        st.caption("Suggested questions")
        for i, q in enumerate(st.session_state["ai_questions"]):
            if st.button(q, key=f"sug_q_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun(scope="fragment")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about these papers"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Answer the last user turn if it has no reply yet — covers both a typed
    # prompt and a suggested-question click (which only appends and reruns).
    history = st.session_state.chat_history
    if history and history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Reading the abstracts…"):
                response_text = gemini_handler.chat_with_context(
                    history, history[-1]["content"], results, ai_api_key
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})


# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------

# A shared link (?q=…) pre-fills the query and runs the search once.
_url_query = st.query_params.get("q", "")
if "query_text" not in st.session_state:
    st.session_state["query_text"] = _url_query
auto_search = bool(_url_query) and not st.session_state.get("auto_searched")
st.session_state["auto_searched"] = True

with st.form("search_form", border=False):
    query = st.text_area(
        "What are you looking for?",
        key="query_text",
        max_chars=8192,
        height=120,
        placeholder="e.g. How do potent monoclonal antibodies neutralize measles virus, "
                    "and which epitopes on the fusion protein do they target?",
        help="Full sentences, research questions or a pasted abstract work best. "
             "Only about the first 2,000 characters are read.",
    )
    c1, c2, c3 = st.columns([1, 2, 1.3], vertical_alignment="bottom")
    num_to_show = c1.number_input("Results", min_value=1, max_value=50, value=10)
    sources = c2.pills(
        "Databases", SOURCE_OPTIONS, default=SOURCE_OPTIONS, selection_mode="multi",
        format_func=lambda s: SOURCE_LABELS[s],
    ) or []
    use_high_quality = c3.toggle(
        "Skip short abstracts", value=True,
        help="Leave out entries with very short or missing abstracts.",
    )

    if st.session_state.get("date_filter_toggle", False):
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("Published from", value=date(2020, 1, 1), min_value=date(1800, 1, 1),
                                    max_value=date.today(), format="YYYY-MM-DD")
        end_d = col_d2.date_input("Published until", value=date.today(), min_value=date(1800, 1, 1),
                                  max_value=date.today(), format="YYYY-MM-DD")
        if start_d and end_d and start_d > end_d:
            start_d, end_d = end_d, start_d
        start_date_str = start_d.isoformat() if start_d else None
        end_date_str = end_d.isoformat() if end_d else None
    else:
        start_date_str = end_date_str = None

    submitted = st.form_submit_button("Search", type="primary", icon=":material/search:") or auto_search

col_t1, col_t2 = st.columns(2)
col_t2.toggle("Filter by publication date", value=False, key="date_filter_toggle")
use_ai = col_t1.toggle("AI summary and chat", key="use_ai_checkbox",
                       help="Uses Google Gemini with your own API key.")
if use_ai:
    ai_api_key = col_t1.text_input(
        "Google AI Studio key", type="password",
        help="Create one at https://aistudio.google.com/apikey",
    )
    col_t1.caption("Titles and abstracts of your results are sent to Google Gemini. "
                   "Your key is used only for this session and is not stored.")
else:
    ai_api_key = None


# ---------------------------------------------------------------------------
# Phase 1: search (via the backend)
# ---------------------------------------------------------------------------

if submitted and query and not sources:
    st.warning("Choose at least one database to search.")
elif submitted and query:
    for key in ("ai_summary", "citations", "doi_list"):
        st.session_state[key] = None
    st.session_state["chat_history"] = []
    st.session_state["ai_questions"] = []

    # Shareable URL for this search (long pasted abstracts are left out of the URL)
    if len(query) <= 2000:
        st.query_params["q"] = query
    else:
        st.query_params.pop("q", None)
        st.info("Long query: only about the first 2,000 characters (512 tokens) are read, "
                "so text beyond that does not change the results.")

    with st.spinner("Searching…"):
        t0 = datetime.now()
        try:
            results, meta = backend_client.search(
                query, top_k=int(num_to_show), start_date=start_date_str, end_date=end_date_str,
                high_quality_only=use_high_quality,
                sources=None if set(sources) == set(SOURCE_OPTIONS) else sources,
            )
        except backend_client.BackendBusy as exc:
            st.warning(str(exc))
            results, meta = None, {}
        except backend_client.BackendError as exc:
            st.error(str(exc))
            results, meta = None, {}

    if results is not None:
        results["rank"] = range(1, len(results) + 1)
        st.session_state["final_results"] = results
        st.session_state["search_meta"] = {**meta, "elapsed": (datetime.now() - t0).total_seconds()}
        st.session_state["search_query"] = query
        if results.empty:
            st.info("Nothing matched. Try describing the topic in a full sentence, "
                    "widen the date range, or include more databases.")

final_results = st.session_state.get("final_results", pd.DataFrame())


# ---------------------------------------------------------------------------
# Phase 2: display
# ---------------------------------------------------------------------------

if not final_results.empty:
    meta = st.session_state.get("search_meta", {})
    results = final_results.copy()
    all_doi = results["doi"].tolist()

    if st.session_state.get("doi_list") != all_doi:
        st.session_state["doi_list"] = all_doi
        st.session_state["citations"] = None
    citations_ready = st.session_state.get("citations") is not None
    results["citations"] = st.session_state["citations"] if citations_ready else [None] * len(results)

    head_l, head_r = st.columns([1.4, 1], vertical_alignment="bottom")
    head_l.markdown(f"### {len(results)} results")
    sort_option = head_r.segmented_control(
        "Sort by", ["Relevance", "Newest", "Most cited"], default="Relevance",
        key="sort_option", label_visibility="collapsed",
    )
    if sort_option == "Newest":
        results = results.assign(_d=pd.to_datetime(results["date"], errors="coerce")) \
            .sort_values("_d", ascending=False).drop(columns="_d")
    elif sort_option == "Most cited" and citations_ready:
        results = results.sort_values("citations", ascending=False)
    results = results.reset_index(drop=True)

    if meta.get("query_bits"):
        ui_components.render_fingerprint(meta["query_bits"])

    if use_ai and ai_api_key and st.session_state.get("ai_summary"):
        with st.container(border=True):
            st.markdown("#### Summary of the top papers")
            st.markdown(st.session_state["ai_summary"])
            st.caption("Written by Gemini from the abstracts; numbers refer to the list below.")

    tab_results, tab_export, tab_chat = st.tabs(["Papers", "Export", "Ask the papers"])

    with tab_results:
        top_score = float(final_results["score"].max())
        for _, row in results.iterrows():
            ui_components.render_entry(row, int(row["rank"]), row["citations"], top_score)
        st.markdown("#### Relevance and publication date")
        st.plotly_chart(ui_components.plot_score_vs_year(results), width="stretch",
                        config={"displayModeBar": False})

    with tab_export:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        bibtex_str = ui_components.generate_bibtex(results)
        st.markdown("Download these results for your reference manager or a spreadsheet.")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        col_ex1.download_button("Download BibTeX", data=bibtex_str, file_name=f"mss_search_{stamp}.bib",
                                mime="text/x-bibtex", type="primary", width="stretch")
        col_ex2.download_button("Download RIS", data=ui_components.generate_ris(results),
                                file_name=f"mss_search_{stamp}.ris",
                                mime="application/x-research-info-systems", width="stretch",
                                help="For Zotero, EndNote and Mendeley.")
        col_ex3.download_button("Download CSV", data=ui_components.results_csv(results),
                                file_name=f"mss_search_{stamp}.csv", mime="text/csv", width="stretch")
        with st.expander("Preview BibTeX"):
            st.code(bibtex_str, language="latex")

    with tab_chat:
        if not use_ai:
            st.info("Turn on “AI summary and chat” above and add your Google AI Studio key to ask "
                    "questions about these papers.")
        elif not ai_api_key:
            st.info("Add your Google AI Studio key above to start.")
        else:
            st.session_state.setdefault("chat_history", [])
            render_chat_interface(final_results, ai_api_key)

    # --- Citation counts (Crossref), fetched after the list is on screen ---
    if not citations_ready:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            st.session_state["citations"] = list(executor.map(ui_data.get_citation_count, all_doi))
        st.rerun()

    # --- AI summary, after citations so the list renders first ---
    if use_ai and ai_api_key and st.session_state.get("ai_summary") is None:
        with st.spinner("Summarising the top papers with Gemini…"):
            summary, questions = gemini_handler.analyze_results(final_results, ai_api_key)
        st.session_state["ai_summary"] = summary
        st.session_state["ai_questions"] = questions
        st.rerun()

ui_components.render_footer()
