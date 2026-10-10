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

import re

import backend_client
import map_view
import ai_assistant
import ui_components
import ui_data

LOGGER = logging.getLogger(__name__)

SOURCE_OPTIONS = ["PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints", "OpenAlex", "Grants"]
DEFAULT_SOURCES = [s for s in SOURCE_OPTIONS if s != "Grants"]   # grants are opt-in
SOURCE_LABELS = ui_components.SOURCE_NAMES

st.set_page_config(page_title="Manuscript Search", page_icon="📜", layout="centered")
ui_components.apply_style()


@st.cache_data(ttl=300, show_spinner=False)
def corpus_stats() -> dict:
    return backend_client.stats()


@st.cache_data(ttl=600, show_spinner=False)
def cached_map_info():
    return backend_client.map_info()


ui_components.render_header(corpus_stats(), ui_data.get_current_active_users())


# ---------------------------------------------------------------------------
# Chat fragment (re-runs independently of the page)
# ---------------------------------------------------------------------------

_AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


@st.fragment
def render_chat_interface(results, ai):
    """A chat panel: the conversation scrolls in a fixed-height box, the question box sits below it."""
    history = st.session_state.chat_history
    label = ai_assistant.PROVIDERS[ai["provider"]].label

    if history:
        _, reset = st.columns([5, 1])
        if reset.button("New chat", type="tertiary", icon=":material/refresh:", key="chat_reset",
                        width="stretch"):
            history.clear()
            st.rerun(scope="fragment")

    with st.container(height=560 if history else "content", border=bool(history), key="chat_box"):
        if not history:
            st.markdown(f"Ask anything about the top {min(15, len(results))} results. Answers come from "
                        f"{label} ({ai['model']}), based only on these abstracts, and cite them by number.")
        for message in history:
            with st.chat_message(message["role"], avatar=_AVATARS.get(message["role"])):
                st.markdown(message["content"])
        # Answer the last question if it has no reply yet (typed or a suggested question)
        if history and history[-1]["role"] == "user":
            with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
                with st.spinner("Reading the abstracts…"):
                    answer = ai_assistant.chat_with_context(history, results, ai["provider"], ai["key"], ai["model"])
                st.markdown(answer)
            history.append({"role": "assistant", "content": answer})

    suggestions = [q for q in st.session_state.get("ai_questions") or []
                   if q not in {m["content"] for m in history}]
    if suggestions:
        picked = st.pills("Suggested questions", suggestions, key=f"chat_suggest_{len(history)}",
                          label_visibility="collapsed")
        if picked:
            history.append({"role": "user", "content": picked})
            st.rerun(scope="fragment")

    prompt = st.chat_input("Ask about these papers", key="chat_prompt")
    if prompt:
        history.append({"role": "user", "content": prompt})
        st.rerun(scope="fragment")


# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------

# A shared link (?q=… or ?similar=Source:id,…) re-runs that search once.
_url_query = st.query_params.get("q", "")
_url_similar = st.query_params.get("similar", "")
if "query_text" not in st.session_state:
    st.session_state["query_text"] = _url_query
first_load = not st.session_state.get("auto_searched")
auto_search = bool(_url_query) and first_load and not _url_similar
if _url_similar and first_load:
    st.session_state["similar_request"] = [r for r in _url_similar.split(",") if r]
st.session_state["auto_searched"] = True


def _clear_selection():
    for key in [k for k in st.session_state if k.startswith("sel::")]:
        del st.session_state[key]


def _store_results(results, meta, t0):
    results["rank"] = range(1, len(results) + 1)
    st.session_state["final_results"] = results
    st.session_state["search_meta"] = {**meta, "elapsed": (datetime.now() - t0).total_seconds()}
    for key in ("ai_summary", "citations", "doi_list"):
        st.session_state[key] = None
    st.session_state["chat_history"] = []
    st.session_state["ai_questions"] = []
    _clear_selection()

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
    sources = st.pills(
        "Databases", SOURCE_OPTIONS, default=DEFAULT_SOURCES, selection_mode="multi",
        format_func=lambda s: SOURCE_LABELS[s],
    ) or []

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

    c1, c2, c3 = st.columns([1, 1, 1.6], vertical_alignment="bottom")
    submitted = c1.form_submit_button("Search", type="primary", icon=":material/search:",
                                      width="stretch") or auto_search
    num_to_show = c2.number_input("Results", min_value=1, max_value=50, value=10)
    use_high_quality = c3.toggle(
        "Skip short abstracts", value=True,
        help="Leave out entries with very short or missing abstracts.",
    )

col_t1, col_t2 = st.columns(2)
col_t2.toggle("Filter by publication date", value=False, key="date_filter_toggle")
use_ai = col_t1.toggle("AI summary and chat", key="use_ai_checkbox",
                       help="Summarise the results and ask questions about them with an AI model, "
                            "using your own API key (Google, Anthropic, OpenAI, OpenRouter and others).")
ai = {"provider": "google", "key": "", "model": ""}
if use_ai:
    with st.container(border=True):
        a1, a2 = st.columns([1, 1.4])
        provider_id = a1.selectbox("Provider", list(ai_assistant.PROVIDERS), key="ai_provider",
                                   format_func=lambda k: ai_assistant.PROVIDERS[k].label)
        provider = ai_assistant.PROVIDERS[provider_id]
        api_key = a2.text_input(f"{provider.label} API key", type="password", key=f"ai_key_{provider_id}",
                                help=f"Create one at {provider.key_url}")
        # Models this key can use, fetched once per provider and key (kept in this session only)
        models = [provider.default_model]
        if api_key:
            cache = st.session_state.setdefault("ai_models", {})
            fingerprint = (provider_id, hash(api_key))
            if fingerprint not in cache:
                with st.spinner("Loading the models available to this key…"):
                    cache[fingerprint] = ai_assistant.list_models(provider_id, api_key)
            models = cache[fingerprint]
        model = st.selectbox("Model", models, key=f"ai_model_{provider_id}", accept_new_options=True,
                             help="Pick a model, or type the name of any model your key can use.")
        st.caption(f"{provider.note + ' ' if provider.note else ''}"
                   f"[Get a {provider.label} key]({provider.key_url}). Titles and abstracts of your "
                   f"results are sent to {provider.label}; your key is used only in this session and "
                   f"is not stored.")
    ai = {"provider": provider_id, "key": api_key, "model": model or provider.default_model}
ai_api_key = ai["key"] or None


# ---------------------------------------------------------------------------
# Phase 1: search (via the backend)
# ---------------------------------------------------------------------------

if submitted and query and not sources:
    st.warning("Choose at least one database to search.")
elif submitted and query:
    st.session_state["view"] = "search"
    st.query_params.pop("similar", None)

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
                sources=sources,
            )
        except backend_client.BackendBusy as exc:
            st.warning(str(exc))
            results, meta = None, {}
        except backend_client.BackendError as exc:
            st.error(str(exc))
            results, meta = None, {}

    if results is not None:
        _store_results(results, meta, t0)
        st.session_state["search_query"] = query
        if results.empty:
            st.info("Nothing matched. Try describing the topic in a full sentence, "
                    "widen the date range, or include more databases.")

# --- "More like this": papers similar to one or more example papers ---
similar_refs = st.session_state.pop("similar_request", None)
if similar_refs:
    if st.session_state.get("view") != "similar":
        st.session_state["search_backup"] = {
            k: st.session_state.get(k) for k in ("final_results", "search_meta", "search_query")}
    with st.spinner("Finding similar papers…"):
        t0 = datetime.now()
        try:
            results, meta = backend_client.similar(
                similar_refs, top_k=int(num_to_show), start_date=start_date_str, end_date=end_date_str,
                high_quality_only=use_high_quality,
                sources=sources,
            )
            _store_results(results, meta, t0)
            st.session_state["view"] = "similar"
            st.query_params["similar"] = ",".join(similar_refs)
        except backend_client.BackendBusy as exc:
            st.warning(str(exc))
        except backend_client.BackendError as exc:
            st.error(f"Could not find similar papers: {exc}")

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
    if st.session_state.get("view") == "similar":
        seeds = meta.get("seeds", [])
        head_l.markdown(f"### {len(results)} papers similar to "
                        f"{'this paper' if len(seeds) == 1 else f'these {len(seeds)} papers'}")
    else:
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

    if st.session_state.get("view") == "similar":
        ui_components.render_seeds(meta.get("seeds", []))
        if st.session_state.get("search_backup", {}).get("final_results") is not None:
            if st.button("Back to your search results", icon=":material/arrow_back:", type="tertiary"):
                for k, v in st.session_state.pop("search_backup").items():
                    st.session_state[k] = v
                for key in ("ai_summary", "citations", "doi_list"):
                    st.session_state[key] = None
                st.session_state["view"] = "search"
                st.query_params.pop("similar", None)
                _clear_selection()
                st.rerun()

    if meta.get("query_bits"):
        ui_components.render_fingerprint(meta["query_bits"], similar=st.session_state.get("view") == "similar")

    # Several papers selected -> one "more like these" search
    selected = [k[5:] for k, v in st.session_state.items() if k.startswith("sel::") and v]
    if selected:
        with st.container(border=True):
            b1, b2, b3 = st.columns([1.2, 2.2, 1], vertical_alignment="center")
            b1.markdown(f"**{len(selected)} selected**")
            if b2.button("Find papers like the selected", type="primary", icon=":material/hub:"):
                st.session_state["similar_request"] = selected
                st.rerun()
            if b3.button("Clear selection", type="tertiary"):
                _clear_selection()
                st.rerun()

    if use_ai and ai_api_key and st.session_state.get("ai_summary"):
        with st.container(border=True):
            st.markdown("#### Summary of the top papers")
            st.markdown(st.session_state["ai_summary"])
            st.caption(f"Written by {st.session_state.get('ai_summary_by', 'an AI model')} from the abstracts; "
                       "numbers refer to the list below.")

    tab_results, tab_map, tab_time, tab_export, tab_chat = st.tabs(
        ["Papers", "Map", "Timeline", "Export", "Ask the papers"])

    with tab_results:
        top_score = float(final_results["score"].max())
        # matched identifiers are reported normalized; show them as the user typed them
        term_display = {w.replace("-", "").lower(): w for w in
                        re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]", st.session_state.get("search_query") or "")}
        for _, row in results.iterrows():
            ui_components.render_entry(row, int(row["rank"]), row["citations"], top_score, term_display)
            ref = row.get("ref")
            if isinstance(ref, str) and ref:
                with st.container(key=f"actions_{row['rank']}"):
                    a1, a2, _ = st.columns([1, 1.7, 3], vertical_alignment="center")
                    a1.checkbox("Select", key=f"sel::{ref}",
                                help="Select several papers, then find papers like all of them.")
                    if a2.button("Similar papers", key=f"sim::{ref}", icon=":material/hub:", type="tertiary"):
                        st.session_state["similar_request"] = [ref]
                        st.rerun()

    with tab_map:
        map_info = cached_map_info()
        if not map_info:
            st.info("The map of all papers is not available right now. Your results are under Papers.")
        else:
            map_view.render_map(map_info, results, meta)
            st.caption("Every paper in the index, laid out so that papers on similar topics sit close "
                       "together. The star is your query and the numbered pins are your results. Click "
                       "anywhere to see the papers at that spot.")

    with tab_time:
        st.plotly_chart(ui_components.plot_score_vs_year(results), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Each dot is one of your results: how well it matches your query against when it "
                   "was published. Larger dots are cited more often.")

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
            st.info("Turn on “AI summary and chat” above and add an API key (Google, Anthropic, OpenAI, "
                    "OpenRouter, …) to ask questions about these papers.")
        elif not ai_api_key:
            st.info("Add your API key above to start.")
        else:
            st.session_state.setdefault("chat_history", [])
            render_chat_interface(final_results, ai)

    # --- Citation counts (Crossref), fetched after the list is on screen ---
    if not citations_ready:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            st.session_state["citations"] = list(executor.map(ui_data.get_citation_count, all_doi))
        st.rerun()

    # --- AI summary, after citations so the list renders first ---
    if use_ai and ai_api_key and st.session_state.get("ai_summary") is None:
        label = ai_assistant.PROVIDERS[ai["provider"]].label
        st.session_state["ai_summary_by"] = f"{label} ({ai['model']})"
        with st.spinner(f"Summarising the top papers with {label}…"):
            summary, questions = ai_assistant.analyze_results(final_results, ai["provider"], ai["key"],
                                                              ai["model"])
        st.session_state["ai_summary"] = summary
        st.session_state["ai_questions"] = questions
        st.rerun()

ui_components.render_footer(ui_components.freshness_line(corpus_stats()))
