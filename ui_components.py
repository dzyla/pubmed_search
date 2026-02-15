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
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

def define_style():
    st.markdown(
        """
        <style>
            .stExpander > .stButton > button {
                width: 100%;
                border: none;
                background-color: #f0f2f6;
                color: #333;
                text-align: left;
                padding: 15px;
                font-size: 18px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo(last_date, db_size_bio, db_size_pubmed, db_size_med, db_size_arxiv):
    active_users = get_current_active_users()
    pubmed_logo = "https://upload.wikimedia.org/wikipedia/commons/f/fb/US-NLM-PubMed-Logo.svg"
    biorxiv_logo = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    arxiv_logo = "https://upload.wikimedia.org/wikipedia/commons/7/7a/ArXiv_logo_2022.png"
    logging.info(f"Rendering logos with sizes: PubMed={db_size_pubmed}, BioRxiv={db_size_bio}, medRxiv={db_size_med}, arXiv={db_size_arxiv} and active users: {active_users}")
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
                <div style="text-align: center;">
                    <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">
                        <img src="{pubmed_logo}" alt="PubMed logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank" style="text-decoration: none; color: inherit;">PubMed</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://www.biorxiv.org/" target="_blank">
                        <img src="{biorxiv_logo}" alt="BioRxiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://www.biorxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">BioRxiv</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://www.medrxiv.org/" target="_blank">
                        <img src="{medarxiv_logo}" alt="medRxiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://www.medrxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">medRxiv</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://arxiv.org/" target="_blank">
                        <img src="{arxiv_logo}" alt="arXiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://arxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">arXiv</a>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <h3 style="color: black; margin: 0; font-weight: 400;">Manuscript Semantic Search [MSS]</h3>
                <p style="font-size: 16px; color: #555; margin: 5px 0 0 0;">
                    Last database update: {last_date} | Active users: <b>{active_users}</b><br>
                    Database size: PubMed: {int(db_size_pubmed):,} entries / BioRxiv: {int(db_size_bio):,} / MedRxiv: {int(db_size_med):,} / arXiv: {int(db_size_arxiv):,}
                </p>
                <p style="font-size: 9px; color: #777; margin-top: 15px;">
                    Disclaimer: This website is not affiliated with, endorsed by, or sponsored by PubMed, BioRxiv, medRxiv, or arXiv. The logos shown are the property of their respective owners and are used solely for informational purposes. All liability for any legal claims or issues arising from the display or use of these logos is expressly disclaimed.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def plot_score_vs_year(sorted_results):
    try:
        # 1. Handle Dates (Coerce to datetime)
        # We fill NaT dates with a "dummy" date (e.g., today) so they appear on the graph,
        # but we can rely on hover info to show truth. Or fill with min date in set.
        sorted_results["Date_Parsed"] = pd.to_datetime(sorted_results["date"], errors='coerce')
        
        # If date is totally missing, use current date or 1970 to force it to show
        # Using min() of valid dates usually keeps scale reasonable
        min_valid = sorted_results["Date_Parsed"].min()
        if pd.isnull(min_valid): min_valid = pd.Timestamp.now()
        
        # Fill missing dates so they get plotted
        sorted_results["Date_Plot"] = sorted_results["Date_Parsed"].fillna(min_valid)
        
        # 2. Handle Citations (Numeric, fill NaN with 0)
        if "citations" not in sorted_results.columns:
            sorted_results["citations"] = 0
        
        # Ensure numeric
        sorted_results["citations"] = pd.to_numeric(sorted_results["citations"], errors='coerce').fillna(0)
            
        plot_df = pd.DataFrame({
            "Date": sorted_results["Date_Plot"],
            "Title": sorted_results["title"],
            "Relative Score": sorted_results["score"],
            "DOI": sorted_results["doi"],
            "Source": sorted_results["source"],
            "citations": sorted_results["citations"]
        })
        
        # 3. Calculate marker size (log scale, ensuring min size)
        plot_df["marker_size"] = np.log1p(plot_df["citations"]) * 5 + 5
        
        fig = px.scatter(
            plot_df,
            x="Date",
            y="Relative Score",
            size="marker_size",
            hover_data={"Title": True, "DOI": True, "citations": True, "marker_size": False},
            color="Source",
            title="Publication Dates and Similarity Score"
        )
        
        # Optional: Add note about missing dates if any
        if sorted_results["Date_Parsed"].isnull().any():
            fig.add_annotation(
                text="Note: Some points have inferred dates (metadata missing)",
                xref="paper", yref="paper",
                x=0, y=1.1, showarrow=False, font=dict(size=10, color="gray")
            )
            
        fig.update_layout(legend=dict(title="Source"))
        return fig
    except Exception as e:
        st.error(f"Error in plotting Score vs Year: {str(e)}")
        return go.Figure()

def generate_bibtex(df):
    """
    Generates a BibTeX string from the results DataFrame.
    Includes robust key generation and field handling.
    """
    bibtex_entries = []
    
    for idx, row in df.iterrows():
        # 1. Parse Authors
        authors = str(row.get('authors', 'Unknown'))
        # Replace semicolons with 'and' for BibTeX standard
        authors = authors.replace(';', ' and')
        # Simple cleanup if it's a list string representation
        authors = authors.replace('[', '').replace(']', '').replace("'", "")
        
        # 2. Parse Year
        date_str = str(row.get('date', ''))
        year = "n.d."
        # Try to find 4 continuous digits for year
        match = re.search(r'\d{4}', date_str)
        if match:
            year = match.group(0)
            
        # 3. Generate Citation Key (FirstAuthor + Year + FirstWordTitle)
        # Extract first author surname
        first_author = authors.split(' ')[0].split(',')[0].strip()
        # Sanitize author name (remove non-alphanumeric)
        first_author_clean = re.sub(r'\W+', '', first_author)
        if not first_author_clean: first_author_clean = "Anon"
        
        # Extract first significant word of title
        title = str(row.get('title', 'No Title'))
        # Remove common stopwords for key generation if desired, or just take first alphanumeric chunk
        title_clean = re.sub(r'[^\w\s]', '', title)
        title_words = title_clean.split()
        first_word_title = title_words[0] if title_words else "Untitled"
        
        # Construct Key: AuthorYearWord_Index (Index ensures uniqueness)
        citation_key = f"{first_author_clean}{year}{first_word_title}_{idx}"
        
        # 4. Determine Entry Type & Journal
        source = str(row.get('source', '')).lower()
        journal = str(row.get('journal', ''))
        if journal.lower() == 'nan' or not journal:
            journal = source.capitalize() # Fallback to source name if journal is missing
            
        # Default to article
        entry_type = "article"
        
        # 5. Build Fields
        # Use {{}} for title to preserve capitalization in some BibTeX styles, though standard is {}
        entry = f"@{entry_type}{{{citation_key},\n"
        entry += f"  author = {{{authors}}},\n"
        entry += f"  title = {{{title}}},\n"
        entry += f"  journal = {{{journal}}},\n"
        entry += f"  year = {{{year}}},\n"
        
        doi = row.get('doi')
        if doi and "10." in str(doi):
            # clean DOI if necessary
            clean_doi = str(doi).strip()
            entry += f"  doi = {{{clean_doi}}},\n"
            entry += f"  url = {{https://doi.org/{clean_doi}}},\n"
            
        # Close entry
        entry += "}\n"
        bibtex_entries.append(entry)
        
    return "\n".join(bibtex_entries)

def render_footer():
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    c1.markdown(
        """
        <div style='text-align: center;'>
            <b>[MSS] Developed by <a href="https://www.zylalab.org/" target="_blank">Dawid Zyla</a></b>
            |
            <a href="https://github.com/dzyla/pubmed_search" target="_blank">Source code on GitHub</a>
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
