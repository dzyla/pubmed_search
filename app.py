import streamlit as st
import pandas as pd
import concurrent.futures
from datetime import datetime
import logging

# Import modules
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

# --- Load Config & State ---
config_data = config_loader.load_configs_and_db_sizes()
configs = config_data["configs"]

# --- Automatic Update Check (Startup Only) ---
if "updates_checked" not in st.session_state:
    with st.spinner("Checking for new manuscript embeddings..."):
        # Trigger check/update in background
        any_updates = search_logic.trigger_database_updates(configs)
        
        if any_updates:
            st.toast("New data detected! Database updated.", icon="🔄")
            # Clear config cache so the UI numbers update immediately
            config_loader.load_configs_and_db_sizes.clear()
            # Reload fresh config data
            config_data = config_loader.load_configs_and_db_sizes()
            configs = config_data["configs"]
            
    st.session_state["updates_checked"] = True

# --- Status & Logo ---
last_biorxiv_date = utils.report_dates_from_metadata(config_data["biorxiv_config"])
ui_components.render_logo(
    last_biorxiv_date, 
    config_data["biorxiv_db_size"], 
    config_data["pubmed_db_size"], 
    config_data["medrxiv_db_size"], 
    config_data["arxiv_db_size"]
)

# update_status = api_handler.check_update_status()
# if update_status:
#     st.info("Database update in progress. Search might be slow...")

# --- Helper for Chat Callbacks ---
def on_question_click(question_text, df_context, api_key):
    """Callback to handle question button clicks."""
    st.session_state.chat_history.append({"role": "user", "content": question_text})
    # Generate response immediately to make it feel responsive
    response = gemini_handler.chat_with_context(
        st.session_state.chat_history, 
        question_text, 
        df_context, 
        api_key
    )
    st.session_state.chat_history.append({"role": "assistant", "content": response})

@st.fragment
def render_chat_interface(sorted_results, ai_api_key):
    st.markdown("### 💬 Chat with Search Results")
    
    # Display Suggested Questions
    if st.session_state.get("ai_questions"):
        st.markdown("**Suggested Questions:**")
        q_cols = st.columns(len(st.session_state["ai_questions"]))
        for i, q in enumerate(st.session_state["ai_questions"]):
            if q_cols[i].button(q, key=f"sug_q_{i}"):
                # Append to history directly
                st.session_state.chat_history.append({"role": "user", "content": q})
                # Force a rerun of this fragment to show the new message immediately
                st.rerun()

    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Ask about these papers..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = gemini_handler.chat_with_context(
                    st.session_state.chat_history, 
                    prompt, 
                    sorted_results, 
                    ai_api_key
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# --- Search Form ---
with st.form("search_form"):
    query = st.text_area("Enter your search query:", max_chars=8192, height=68)
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input("Number of results to show:", min_value=1, max_value=100, value=10)
    with col2:
        use_high_quality = st.toggle("Use high-quality search?", value=True,
                                   help="Enable for more accurate results (filters out entries with short/missing abstracts).")
        
    if st.session_state.get("date_filter_toggle", False):
        col_d1, col_d2 = st.columns(2)
        start_date = col_d1.date_input("Start Date", value=datetime(2020, 1, 1))
        end_date = col_d2.date_input("End Date", value=datetime.today())
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
    else:
        start_date_str = None
        end_date_str = None
        
    submitted = st.form_submit_button("Search :material/search:", type="primary")

# --- Toggles outside form ---
col_t1, col_t2 = st.columns(2)
use_ai = col_t1.toggle("Use AI features (Summary & Chat)?", key="use_ai_checkbox")
activate_date_filter = col_t2.toggle("Use Date Filter", value=False, key="date_filter_toggle")

# AI API Key Input
ai_api_key = None
if use_ai:
    ai_api_key = col_t1.text_input("Google AI API Key", type="password", help="Get key at https://aistudio.google.com/apikey")

STATUS = st.empty()

st.markdown("---")

col_a, col_b = st.columns(2)

# --- Execution Phase 1: Search ---
if submitted and query:
    # Reset states for new search
    st.session_state["chat_history"] = []
    st.session_state["ai_summary"] = None
    st.session_state["ai_questions"] = []
    
    with st.spinner("Searching..."):
        search_start_time = datetime.now()
        
        # 1. Get Packed Binary Embedding
        query_packed = api_handler.get_query_embedding_packed(query)
        
        if query_packed is not None:
            # 2. Execute Orchestrated Search
            final_results = search_logic.combined_search_orchestrator(
                query_packed, 
                configs, 
                top_k=num_to_show,
                start_date=start_date_str,
                end_date=end_date_str,
                use_high_quality=use_high_quality 
            )
            
            total_time = datetime.now() - search_start_time
            st.markdown(f"<h6 style='text-align: center; color: #7882af;'>Search completed in {total_time.total_seconds():.2f} seconds</h6>", unsafe_allow_html=True)
            
            st.session_state["final_results"] = final_results
            st.session_state["search_query"] = query
            st.session_state["num_to_show"] = num_to_show
        else:
            st.error("Failed to generate embedding.")
            final_results = pd.DataFrame() # FIX: Ensure final_results is defined even on API error
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

# --- Execution Phase 2: Display & Async AI ---
if not final_results.empty:
    
    sort_option = col_b.segmented_control(
        "Sort results by:", 
        options=["Relevance", "Publication Date", "Citations"], 
        key="sort_option"
    )
    
    sorted_results = st.session_state["final_results"].copy()
    all_doi = sorted_results["doi"].tolist()
    
    # --- Metadata fetching (Citations/Links) ---
    # We do this here to populate the table, it might block slightly but is necessary for the visual
    if st.session_state.get("doi_list") != all_doi:
        st.session_state["doi_list"] = all_doi
        st.session_state["citations"] = None
        st.session_state["clean_doi"] = None
        st.session_state["full_text_links"] = None

    if st.session_state.get("citations") is None:
        with st.spinner("Fetching citations..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                all_citations = list(executor.map(utils.get_citation_count, all_doi))
            st.session_state["citations"] = all_citations
    else:
        all_citations = st.session_state["citations"]

    if st.session_state.get("clean_doi") is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            clean_doi_list = list(executor.map(utils.get_clean_doi, all_doi))
        st.session_state["clean_doi"] = clean_doi_list
    else:
        clean_doi_list = st.session_state["clean_doi"]

    sorted_results["citations"] = all_citations
    sorted_results["doi"] = clean_doi_list

    # Sorting
    if sort_option == "Publication Date":
        sorted_results["date_parsed"] = pd.to_datetime(sorted_results["date"], errors="coerce")
        sorted_results = sorted_results.sort_values(by="date_parsed", ascending=False).reset_index(drop=True)
        sorted_results.drop(columns=["date_parsed"], inplace=True)
    elif sort_option == "Citations":
        sorted_results = sorted_results.sort_values(by="citations", ascending=False).reset_index(drop=True)
    else:
        sorted_results = sorted_results.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Full text links
    if st.session_state.get("full_text_links") is None:
        sorted_results = utils.precalculate_full_text_links_parallel(sorted_results)
        st.session_state["full_text_links"] = sorted_results["full_text_link"].tolist()
    else:
        sorted_results["full_text_link"] = st.session_state["full_text_links"]

    full_text_link_n = sorted_results["full_text_link"].notnull().sum()
    col_a.markdown(f"#### Search results\n:material/import_contacts: {full_text_link_n} Full, free text available")

    # --- Render Main Tabs ---
    tabs = st.tabs(["Results List", "Bibliography", "Chat with Papers"])
    
    with tabs[0]:
        # Render Expanders
        for idx, row in sorted_results.iterrows():
            citations = row["citations"]
            doi_link = f"https://doi.org/{row['doi']}" if "arxiv.org" not in str(row["doi"]) else row["doi"]
            full_text_link = row.get("full_text_link")
            final_link = full_text_link if full_text_link is not None else doi_link
            full_text_notification = ":material/import_contacts:" if full_text_link is not None else ""
            has_full_text = full_text_link is not None
            
            expander_title = (
                f"{idx + 1}\. {row['title']}\n\n" 
                f"_(Score: {row['score']:.2f} | Date: {row['date']} | Citations: {citations})_"
            )
            if full_text_notification:
                expander_title += f" | {full_text_notification}"

            with st.expander(expander_title):
                c_a, c_b, c_c = st.columns(3)
                c_a.markdown(f"**Score:** {row['score']:.2f}")
                c_b.markdown(f"**Source:** {row['source']}")
                c_c.markdown(f"**Citations:** {citations}")
                st.markdown(f"**Authors:** {row['authors']}")
                c_d, c_e = st.columns(2)
                c_d.markdown(f"**Date:** {row['date']}")
                c_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")
                st.markdown(f"**Abstract:**\n{row['abstract']}")
                
                link_cols = st.columns(3)
                if has_full_text:
                    link_cols[0].markdown(f"**[:material/import_contacts: Read Full Text]({final_link})**")
                link_cols[1].markdown(f"**[:material/link: View on Publisher Website]({doi_link})**")
        
        # Plot Score vs Year below results (Same tab)
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
                file_name=f"mss_search_results_{datetime.now().strftime('%Y%m%d_%H%M')}.bib",
                mime="text/x-bibtex",
                type="primary"
            )
        with st.expander("Preview BibTeX Data"):
            st.code(bibtex_str, language="latex")

    # --- Chat Tab ---
    with tabs[2]:
        if not use_ai:
            st.info("Please enable 'Use AI features' and provide an API Key.")
        elif not ai_api_key:
            st.warning("Please provide a Google AI API Key.")
        else:
            # Initialize history once
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            # Call the fragment
            render_chat_interface(sorted_results, ai_api_key)

    # --- Execution Phase 3: Async AI Summary ---
    # This runs AFTER everything else has rendered, preventing blockage.
    if use_ai and ai_api_key and not sorted_results.empty:
        # Check if we already have a summary for this specific query to avoid re-running on interactions
        if st.session_state.get("ai_summary") is None:
            summary_placeholder = st.container()
            with summary_placeholder:
                st.write("---")
                with st.status("🤖 Generating AI Analysis...", expanded=True) as status:
                    st.write("Summarizing abstracts...")
                    summary = gemini_handler.summarize_search_results(sorted_results, ai_api_key)
                    st.write("Generating suggested questions...")
                    questions = gemini_handler.generate_example_questions(sorted_results, ai_api_key)
                    
                    st.session_state["ai_summary"] = summary
                    st.session_state["ai_questions"] = questions
                    status.update(label="AI Analysis Complete", state="complete", expanded=True)
                    st.rerun() # Rerun to populate the top widgets and chat suggestions
        
        # If Summary Exists, Display it at the bottom (or wherever preferred)
        if st.session_state.get("ai_summary"):
            st.markdown("---")
            st.markdown("### 🤖 AI Summary of Findings")
            st.info(st.session_state["ai_summary"])

elif submitted and final_results.empty:
    st.warning("#### No results found. Please try a different query.")

ui_components.render_footer()
