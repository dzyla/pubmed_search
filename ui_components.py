import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from umap_pytorch import load_pumap
import torch

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

def logo(db_update_date, db_size_bio, db_size_pubmed, db_size_med, db_size_arxiv):
    pubmed_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/720px-US-NLM-PubMed-Logo.svg.png?20080121063734"
    biorxiv_logo = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    arxiv_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/640px-ArXiv_logo_2022.svg.png"

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
                    Last database update: {db_update_date}<br>
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

@st.cache_resource
def load_pumap_model_and_image(model_path: str, image_path: str) -> tuple:
    model = load_pumap(model_path)
    if hasattr(model, "to"):
        model.to("cpu")
    image = np.load(image_path)
    return model, image

def render_results_ui(final_results, get_citation_count_cached, get_link_info_cached, get_clean_doi):
    # This function replaces the main loop in app.py
    # But Streamlit UI construction is usually better kept in main app to avoid passing st object or context issues.
    # We will keep main loop in app.py but import helpers.
    pass
