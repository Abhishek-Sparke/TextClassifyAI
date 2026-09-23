"""
TextClassify - AI-Powered Text Document Classification System
Production-grade Document Intelligence Interface with Automated Inference.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TextClassify | Document Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Premium Dark-First AI SaaS Theme (Linear / Perplexity / Vercel Aesthetic)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-dark: #07090e;
        --bg-card: rgba(14, 20, 31, 0.75);
        --bg-card-hover: rgba(22, 32, 50, 0.9);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.35);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Completely Remove / Hide Sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide Default Streamlit Chrome */
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}

    /* Base Body & Layout */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111827 0%, #07090e 75%, #030508 100%) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* Sticky Top Navigation Bar */
    .tc-navbar-wrapper {
        position: sticky;
        top: 0;
        z-index: 999;
        margin-bottom: 24px;
        background: rgba(11, 17, 27, 0.82);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-bottom: 1px solid var(--border-subtle);
        padding: 10px 0;
    }
    .tc-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        max-width: 1280px;
        margin: 0 auto;
        padding: 0 4px;
    }
    .tc-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
    }
    .tc-logo-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .tc-brand-title {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tc-brand-pill {
        font-size: 9.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 2px 7px;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }

    .tc-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        font-size: 11.5px;
        font-weight: 600;
        color: #34d399;
    }
    .tc-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: tc-pulse 2s infinite ease-in-out;
    }
    @keyframes tc-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Segmented Nav Custom Styling */
    div[data-testid="stSegmentedControl"] {
        background: rgba(14, 20, 31, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 3px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        padding: 5px 14px !important;
        color: #94a3b8 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: #6366f1 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
    }

    /* Glass Panels */
    .tc-card {
        background: var(--bg-card);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 20px;
    }
    .tc-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    .tc-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 12px;
    }
    .tc-card-title {
        font-size: 13px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tc-card-sub {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* Hero Banner */
    .tc-hero-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.85) 60%, rgba(10, 15, 26, 0.95) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 22px;
        padding: 36px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .tc-hero-card::after {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .tc-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 999px;
        padding: 5px 12px;
        margin-bottom: 14px;
    }
    .tc-hero-title {
        font-size: clamp(28px, 3.4vw, 42px);
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 10px;
        background: linear-gradient(135deg, #ffffff 40%, #e2e8f0 70%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tc-hero-sub {
        font-size: clamp(14px, 1.2vw, 16px);
        color: var(--text-secondary);
        max-width: 680px;
        line-height: 1.55;
        margin: 0 0 20px;
    }

    /* Stat Cards Grid */
    .tc-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .tc-stat-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 20px;
        position: relative;
        transition: all 0.22s ease;
    }
    .tc-stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.35);
        background: var(--bg-card-hover);
    }
    .tc-stat-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .tc-stat-val {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .tc-stat-sub {
        font-size: 11px;
        color: var(--accent-cyan);
        font-weight: 500;
        margin-top: 4px;
    }

    /* Prediction Display */
    .tc-pred-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(14, 20, 31, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .tc-pred-kicker {
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #818cf8;
        margin-bottom: 8px;
    }
    .tc-pred-cat {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .tc-pred-conf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 999px;
        padding: 4px 14px;
        color: #34d399;
        font-size: 13px;
        font-weight: 700;
        font-family: var(--font-mono);
        margin-bottom: 16px;
    }

    .tc-pred-info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 14px;
        text-align: left;
    }
    .tc-pred-info-item {
        background: rgba(0, 0, 0, 0.25);
        border-radius: 10px;
        padding: 8px 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .tc-pred-info-lbl {
        font-size: 9.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }
    .tc-pred-info-val {
        font-size: 13px;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 2px;
        font-family: var(--font-mono);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Horizontal Confidence Bars */
    .tc-conf-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .tc-conf-label {
        width: 140px;
        font-size: 12px;
        font-weight: 600;
        color: #cbd5e1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .tc-conf-track {
        flex: 1;
        height: 8px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .tc-conf-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        transition: width 0.3s ease;
    }
    .tc-conf-pct {
        width: 54px;
        font-size: 11.5px;
        font-weight: 700;
        color: #94a3b8;
        font-family: var(--font-mono);
        text-align: right;
    }

    /* Keyword Tags */
    .tc-tag-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tc-keyword-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 11px;
        background: rgba(99, 102, 241, 0.09);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        color: #c7d2fe;
    }
    .tc-keyword-wt {
        font-size: 10px;
        font-family: var(--font-mono);
        color: #818cf8;
        background: rgba(99, 102, 241, 0.2);
        padding: 1px 5px;
        border-radius: 4px;
    }

    /* Idle Ready Box */
    .tc-idle-box {
        text-align: center;
        padding: 44px 20px;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        background: rgba(0, 0, 0, 0.18);
    }
    .tc-idle-icon {
        font-size: 38px;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.3));
    }
    .tc-idle-title {
        font-size: 16px;
        font-weight: 750;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .tc-idle-desc {
        font-size: 12.5px;
        color: var(--text-muted);
        max-width: 320px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Model Performance Grid */
    .tc-model-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }
    .tc-model-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px;
        position: relative;
        transition: all 0.2s ease;
    }
    .tc-model-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }
    .tc-model-card.prod-badge-card {
        border-color: rgba(99, 102, 241, 0.45);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(14, 20, 31, 0.8) 100%);
    }
    .tc-model-name {
        font-size: 13.5px;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .tc-model-acc {
        font-size: 24px;
        font-weight: 800;
        color: #a5b4fc;
        letter-spacing: -0.02em;
    }
    .tc-model-acc-lbl {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .tc-model-metrics-row {
        display: flex;
        justify-content: space-between;
        font-size: 10.5px;
        color: #94a3b8;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 8px;
    }
    .tc-model-metrics-row b {
        color: #f1f5f9;
    }

    /* Pipeline Architecture Grid */
    .tc-pipeline-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-top: 14px;
    }
    .tc-pipeline-card {
        background: rgba(14, 20, 31, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 12px;
        text-align: center;
        position: relative;
        transition: all 0.22s ease;
    }
    .tc-pipeline-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(20, 28, 44, 0.8);
    }
    .tc-pipeline-num {
        font-family: var(--font-mono);
        font-size: 10px;
        color: #818cf8;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .tc-pipeline-title {
        font-size: 12.5px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .tc-pipeline-desc {
        font-size: 11px;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Meta Counter Bar */
    .tc-meta-counter {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11.5px;
        color: #64748b;
        font-family: var(--font-mono);
        padding: 4px 2px 10px;
    }

    /* Responsive */
    @media (max-width: 900px) {
        .tc-stat-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-model-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-pipeline-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-pred-info-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
        .tc-stat-grid { grid-template-columns: 1fr; }
        .tc-model-grid { grid-template-columns: 1fr; }
        .tc-pipeline-grid { grid-template-columns: 1fr; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Curated Realistic Example Documents
# -----------------------------------------------------------------------------
EXAMPLE_DOCUMENTS = {
    "Technology": (
        "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. "
        "The 3D rendering pipeline transforms polygon meshes and texture maps using Vulkan and DirectX, "
        "computing vertex lighting, anti-aliasing, reflections, and shadow maps at high refresh rates."
    ),
    "Sports": (
        "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings. "
        "In the bottom of the ninth, the clean-up hitter drove in two runs with a solid line drive over the outfield fence, "
        "securing the championship victory as the stadium erupted."
    ),
    "Business": (
        "The company reported increased revenue during the financial quarter. "
        "Investors are expecting stronger profits as the organization expands its products into international markets."
    ),
    "Politics": (
        "Congress held an extensive legislative debate on national fiscal reform, international trade treaties, and civil rights. "
        "Leaders presented constitutional arguments regarding government budget allocation, judicial oversight, and executive appointments."
    ),
    "Entertainment": (
        "The critically acclaimed feature film received multiple nominations at the international cinema awards. "
        "Critics praised the director's visionary storytelling, the orchestral musical score, and the lead actor's stirring theatrical performance."
    )
}


# -----------------------------------------------------------------------------
# Dynamic Category Icon Mapping
# -----------------------------------------------------------------------------
def get_category_icon(category_name: str, raw_class: str = "") -> str:
    combined = f"{category_name} {raw_class}".lower()
    if any(k in combined for k in ["business", "trade", "revenue", "profit", "finance", "investor", "market", "economy"]):
        return "💼"
    elif any(k in combined for k in ["world", "global", "international", "diplomacy"]):
        return "🌐"
    elif any(k in combined for k in ["entertain", "movie", "film", "cinema", "theatre", "actor", "music", "hollywood", "arts"]):
        return "🎬"
    elif any(k in combined for k in ["education", "academic", "university", "curriculum", "pedagogy", "school"]):
        return "🎓"
    elif any(k in combined for k in ["environment", "climate", "solar", "renewable", "ecology", "carbon"]):
        return "🌱"
    elif any(k in combined for k in ["wellness", "med", "health", "doctor", "clinical", "disease", "sci.med"]):
        return "🧘"
    elif any(k in combined for k in ["tech", "computer", "hardware", "software", "sys", "gpu", "compiler", "os"]):
        return "💻"
    elif any(k in combined for k in ["graphic", "rendering", "3d", "art", "design", "comp.graphics"]):
        return "🎨"
    elif any(k in combined for k in ["sport", "baseball", "soccer", "game", "hockey", "championship", "player"]):
        return "⚽"
    elif any(k in combined for k in ["politic", "congress", "law", "legislation", "constitution", "government", "senate"]):
        return "🏛️"
    elif any(k in combined for k in ["space", "nasa", "astronomy", "telescope", "orbit", "mars", "galaxy", "sci.space"]):
        return "🚀"
    elif any(k in combined for k in ["auto", "car", "engine", "vehicle", "transmission", "rec.autos"]):
        return "🚗"
    elif any(k in combined for k in ["sale", "forsale", "price", "offer", "discount"]):
        return "🏷️"
    elif any(k in combined for k in ["relig", "faith", "church", "god", "christian", "atheism"]):
        return "🕊️"
    return "📄"


# -----------------------------------------------------------------------------
# Artifact Loader with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    """
    Loads saved vectorizer, production best_model, all models, and metadata.
    """
    if not os.path.exists('models/best_model.joblib') or not os.path.exists('models/tfidf_vectorizer.joblib'):
        return None, None, None, None

    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    best_model = joblib.load('models/best_model.joblib')
    all_models = joblib.load('models/all_models.joblib')

    with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return vectorizer, best_model, all_models, metadata


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Session State Initialization
    if 'active_nav' not in st.session_state:
        st.session_state.active_nav = 'Dashboard'
    if 'doc_input' not in st.session_state:
        st.session_state.doc_input = ''
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'recent_classifications' not in st.session_state:
        st.session_state.recent_classifications = []
    if 'user_documents' not in st.session_state:
        st.session_state.user_documents = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None

    vectorizer, best_model, all_models, metadata = load_project_artifacts()

    if vectorizer is None:
        st.error("⚠️ Model artifacts not found in `models/`. Run `python train.py` to train and save models first.")
        st.stop()

    categories = metadata.get('categories', [])
    cat_display = metadata.get('category_display_names', {})
    metrics_table = pd.DataFrame(metadata.get('metrics_table', []))
    stats = metadata.get('dataset_statistics', {})
    best_model_name = metadata.get('best_model_name', 'Support Vector Machine')

    # -------------------------------------------------------------------------
    # STICKY TOP NAVIGATION BAR (Replaces Sidebar)
    # -------------------------------------------------------------------------
    st.markdown('<div class="tc-navbar-anchor"></div>', unsafe_allow_html=True)
    
    col_nav_brand, col_nav_tabs, col_nav_status = st.columns([2.3, 5.7, 2.0], vertical_alignment="center")
    
    with col_nav_brand:
        st.markdown('''
        <div class="tc-brand">
            <div class="tc-logo-icon">⚡</div>
            <div>
                <div class="tc-brand-title">TextClassify</div>
            </div>
            <div class="tc-brand-pill">PRODUCTION</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_nav_tabs:
        nav_options = ["Dashboard", "Classify", "Analytics", "Dataset", "How It Works", "About"]
        current_nav = st.session_state.active_nav if st.session_state.active_nav in nav_options else "Dashboard"
        selected_nav = st.segmented_control(
            "Navigation",
            options=nav_options,
            default=current_nav,
            label_visibility="collapsed",
            key="top_navbar_segmented"
        )
        if selected_nav and selected_nav != st.session_state.active_nav:
            st.session_state.active_nav = selected_nav
            st.rerun()

    with col_nav_status:
        st.markdown('''
        <div style="display:flex;justify-content:flex-end;align-items:center;">
            <div class="tc-status-pill">
                <span class="tc-status-dot"></span>
                <span>Model Online</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 1: DASHBOARD
    # -------------------------------------------------------------------------
    if st.session_state.active_nav == "Dashboard":
        st.markdown('''
        <section class="tc-hero-card">
            <div class="tc-hero-badge">
                <span>⚡</span> Document Intelligence Dashboard
            </div>
            <h1 class="tc-hero-title">TextClassify</h1>
            <p class="tc-hero-sub">
                Enterprise text classification powered by trained NLP transformers and regularized Machine Learning. 
                Documents are vectorized and classified automatically through the production engine.
            </p>
        </section>
        ''', unsafe_allow_html=True)

        c_cta1, c_cta2, c_cta3 = st.columns([1.5, 2, 1.5])
        with c_cta2:
            if st.button("Start Classification Workspace →", type="primary", use_container_width=True):
                st.session_state.active_nav = "Classify"
                st.rerun()

        st.write("")

        # 4 High-Level Summary Stat Cards
        total_docs = stats.get('total_documents', 3613)
        num_cats = stats.get('num_classes', len(categories))
        best_acc = 0.0
        if not metrics_table.empty and 'Accuracy' in metrics_table.columns:
            best_acc = metrics_table['Accuracy'].max() * 100

        st.markdown(f'''
        <div class="tc-stat-grid">
            <div class="tc-stat-card">
                <div class="tc-stat-label">Documents Processed</div>
                <div class="tc-stat-val">{total_docs:,}</div>
                <div class="tc-stat-sub">Cleaned & Tokenized</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-label">Categories</div>
                <div class="tc-stat-val">{num_cats}</div>
                <div class="tc-stat-sub">Target Semantic Domains</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-label">Model Accuracy</div>
                <div class="tc-stat-val">{best_acc:.2f}%</div>
                <div class="tc-stat-sub">Stratified Test Benchmark</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-label">Production Model</div>
                <div class="tc-stat-val" style="font-size:20px;padding-top:4px;">{best_model_name}</div>
                <div class="tc-stat-sub">Calibrated Linear SVM</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Recent Classifications Section
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">🕒 Recent Classifications</div>
                    <div class="tc-card-sub">Inference events recorded during the current operational session</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        if st.session_state.recent_classifications:
            rec_df = pd.DataFrame(st.session_state.recent_classifications)
            st.dataframe(
                rec_df,
                column_config={
                    "snippet": st.column_config.TextColumn("Document Snippet", width="large"),
                    "category": st.column_config.TextColumn("Predicted Category", width="medium"),
                    "confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "latency": st.column_config.TextColumn("Latency", width="small"),
                    "timestamp": st.column_config.TextColumn("Time", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.markdown('''
            <div style="text-align:center;padding:24px;color:#64748b;font-size:13px;">
                No documents classified yet this session. Click <b>Start Classification Workspace</b> above or choose <b>Classify</b> from the top navigation to analyze your first document.
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model Performance & Category Distribution
        col_dash_l, col_dash_r = st.columns([1, 1], gap="large")

        with col_dash_l:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">⚡ Model Performance Overview</div>
                        <div class="tc-card-sub">Cross-model benchmarking comparison</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            if not metrics_table.empty:
                chart_df = metrics_table[['Model', 'Accuracy']].copy()
                chart_df['Accuracy (%)'] = chart_df['Accuracy'] * 100
                st.bar_chart(chart_df.set_index('Model')['Accuracy (%)'], color="#6366f1")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_dash_r:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">📊 Category Distribution</div>
                        <div class="tc-card-sub">Corpus volume distribution across target topics</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            dist_data = stats.get('class_distribution', {})
            if dist_data:
                df_dist = pd.DataFrame(list(dist_data.items()), columns=['Category', 'Count'])
                df_dist['Label'] = df_dist['Category'].apply(lambda x: cat_display.get(x, x))
                st.bar_chart(df_dist.set_index('Label')['Count'], color="#06b6d4")
            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 2: CLASSIFICATION PAGE (Clean Two-Column Workspace)
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Classify":
        col_input, col_output = st.columns([1.1, 1], gap="large")

        # ---------------------------------------------------------------------
        # LEFT COLUMN: DOCUMENT INPUT
        # ---------------------------------------------------------------------
        with col_input:
            st.markdown('''
            <div class="tc-card" style="padding-bottom:18px;">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">📝 Document Input</div>
                        <div class="tc-card-sub">Paste the text you want to classify.</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Document Source Tabs: Standard Examples vs My Documents
            tab_src_1, tab_src_2 = st.tabs(["📚 Standard Presets", f"📁 My Documents ({len(st.session_state.user_documents)})"])

            with tab_src_1:
                st.markdown("<div style='font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>SELECT EXAMPLE</div>", unsafe_allow_html=True)
                ex_cols = st.columns(len(EXAMPLE_DOCUMENTS))
                for i, (ex_label, ex_text) in enumerate(EXAMPLE_DOCUMENTS.items()):
                    with ex_cols[i]:
                        if st.button(ex_label, key=f"btn_ex_{i}", use_container_width=True):
                            st.session_state.doc_input = ex_text
                            st.rerun()

            with tab_src_2:
                if st.session_state.user_documents:
                    st.markdown("<div style='font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>SAVED CUSTOM DOCUMENTS</div>", unsafe_allow_html=True)
                    for idx, udoc in enumerate(st.session_state.user_documents):
                        col_doc_btn, col_doc_del = st.columns([5, 1])
                        with col_doc_btn:
                            doc_title = udoc.get('title', f'Document {idx+1}')
                            doc_cat = udoc.get('category', 'Custom')
                            if st.button(f"📄 {doc_title} ({doc_cat})", key=f"btn_udoc_{idx}", use_container_width=True):
                                st.session_state.doc_input = udoc.get('text', '')
                                st.rerun()
                        with col_doc_del:
                            if st.button("🗑️", key=f"btn_del_udoc_{idx}", help="Delete document"):
                                st.session_state.user_documents.pop(idx)
                                st.rerun()
                else:
                    st.caption("No custom documents added yet. Use the 'Add Document' form below to save custom text.")

            # Expander 1: Upload Document File
            with st.expander("📂 Upload Document File (.txt, .md, .csv, .json)"):
                uploaded_doc = st.file_uploader(
                    "Choose a document file",
                    type=["txt", "md", "csv", "json"],
                    key="st_doc_file_uploader",
                    label_visibility="collapsed"
                )
                if uploaded_doc is not None:
                    try:
                        raw_bytes = uploaded_doc.read()
                        try:
                            file_str = raw_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            file_str = raw_bytes.decode('latin-1', errors='ignore')

                        fname = uploaded_doc.name
                        ext = fname.split('.')[-1].lower()

                        # Extract text from json/csv if appropriate
                        if ext == 'json':
                            try:
                                j_data = json.loads(file_str)
                                if isinstance(j_data, list):
                                    file_str = "\n\n".join([item.get('text', item.get('content', str(item))) for item in j_data if isinstance(item, dict)])
                                elif isinstance(j_data, dict):
                                    file_str = j_data.get('text', j_data.get('content', file_str))
                            except Exception:
                                pass
                        elif ext == 'csv':
                            try:
                                df_uploaded = pd.read_csv(pd.io.common.BytesIO(raw_bytes))
                                text_cols = [c for c in df_uploaded.columns if any(k in c.lower() for k in ['text', 'content', 'body', 'doc'])]
                                if text_cols:
                                    file_str = "\n\n".join(df_uploaded[text_cols[0]].dropna().astype(str).tolist())
                            except Exception:
                                pass

                        st.session_state.doc_input = file_str
                        st.success(f"✅ Loaded '{fname}' ({len(raw_bytes)/1024:.1f} KB, {len(file_str.split())} words)")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

            # Expander 2: Add to Custom Library (My Documents)
            with st.expander("➕ Save Current Document to My Documents"):
                save_title = st.text_input("Document Title", placeholder="e.g. James Webb Telescope Science Bulletin", key="input_save_title")
                save_cat = st.selectbox("Assign Category", ["Auto-detect"] + list(categories), key="select_save_cat")
                if st.button("Save to Library", key="btn_save_to_lib"):
                    if st.session_state.doc_input.strip():
                        new_item = {
                            "title": save_title.strip() or f"Doc {len(st.session_state.user_documents)+1}",
                            "text": st.session_state.doc_input.strip(),
                            "category": save_cat,
                            "time": time.strftime("%H:%M")
                        }
                        st.session_state.user_documents.append(new_item)
                        st.success(f"Saved '{new_item['title']}' to My Documents!")
                        st.rerun()
                    else:
                        st.warning("Please enter or upload document text first.")

            # Expander 3: Batch Document Classification
            with st.expander("⚡ Batch Process Multiple Documents"):
                st.caption("Upload multiple text files or a CSV with a 'text' column to classify in bulk.")
                batch_files = st.file_uploader(
                    "Upload files for batch processing",
                    type=["txt", "md", "csv"],
                    accept_multiple_files=True,
                    key="batch_file_uploader"
                )
                if batch_files and st.button("Run Batch Classification", key="btn_run_batch"):
                    batch_records = []
                    for b_file in batch_files:
                        b_bytes = b_file.read()
                        try:
                            b_text = b_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            b_text = b_bytes.decode('latin-1', errors='ignore')

                        if b_file.name.endswith('.csv'):
                            try:
                                df_b = pd.read_csv(pd.io.common.BytesIO(b_bytes))
                                text_c = [c for c in df_b.columns if any(k in c.lower() for k in ['text', 'content', 'body', 'doc'])]
                                col_name = text_c[0] if text_c else df_b.columns[0]
                                for r_idx, r_val in enumerate(df_b[col_name].dropna()):
                                    batch_records.append({"title": f"{b_file.name} (row {r_idx+1})", "text": str(r_val)})
                            except Exception:
                                batch_records.append({"title": b_file.name, "text": b_text})
                        else:
                            batch_records.append({"title": b_file.name, "text": b_text})

                    if batch_records:
                        with st.spinner(f"Classifying {len(batch_records)} documents..."):
                            b_results = []
                            for rec in batch_records:
                                c_clean = preprocess_document(rec["text"], apply_lemmatization=True)
                                vec = vectorizer.transform([c_clean if c_clean.strip() else rec["text"].lower()])
                                pred_idx = best_model.predict(vec)[0]
                                cat_id = categories[pred_idx] if pred_idx < len(categories) else str(pred_idx)
                                cat_name = cat_display.get(cat_id, cat_id)
                                probs = get_prediction_probabilities(best_model, vec)[0]
                                conf = round(float(np.max(probs)) * 100, 1) if len(probs) > 0 else 99.0
                                b_results.append({
                                    "Document Title": rec["title"],
                                    "Predicted Category": cat_name,
                                    "Confidence (%)": conf,
                                    "Word Count": len(rec["text"].split()),
                                    "Snippet": rec["text"][:120] + "..."
                                })
                            st.session_state.batch_results = b_results

                if st.session_state.batch_results:
                    df_batch_res = pd.DataFrame(st.session_state.batch_results)
                    st.dataframe(df_batch_res, use_container_width=True, hide_index=True)
                    csv_data = df_batch_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Batch Results (CSV)",
                        data=csv_data,
                        file_name=f"textclassify_batch_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # Large Textarea
            input_text = st.text_area(
                "Document Text Area",
                value=st.session_state.doc_input,
                height=220,
                placeholder="Paste the text you want to classify here, or use the file uploader above...",
                label_visibility="collapsed"
            )
            st.session_state.doc_input = input_text

            # Live Word Count and Character Count
            word_count = len(input_text.split()) if input_text.strip() else 0
            char_count = len(input_text)
            st.markdown(f'''
            <div class="tc-meta-counter">
                <span>{word_count:,} words &bull; {char_count:,} characters</span>
                <span style="color:#818cf8;font-weight:600;">Model: {best_model_name} (Auto)</span>
            </div>
            ''', unsafe_allow_html=True)

            # Action Buttons: Clear and Analyze Document
            btn_clear_col, btn_analyze_col = st.columns([1, 2])
            with btn_clear_col:
                if st.button("Clear", use_container_width=True):
                    st.session_state.doc_input = ""
                    st.session_state.last_prediction = None
                    st.rerun()

            with btn_analyze_col:
                analyze_clicked = st.button("Analyze Document →", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: AI PREDICTION
        # ---------------------------------------------------------------------
        with col_output:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">🤖 AI Prediction</div>
                        <div class="tc-card-sub">Automatic inference via configured production model</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Trigger Actual Model Inference
            if analyze_clicked and input_text.strip():
                with st.spinner("Executing preprocessing, TF-IDF vectorization, and inference..."):
                    start_t = time.perf_counter()

                    # 1. Preprocessing
                    clean_text = preprocess_document(input_text, apply_lemmatization=True)
                    if not clean_text.strip():
                        clean_text = input_text.lower()

                    # 2. TF-IDF Vectorization
                    input_tfidf = vectorizer.transform([clean_text])

                    # 3. Model Inference (Always using Production Best Model)
                    pred_raw = best_model.predict(input_tfidf)[0]

                    # 4. Probabilities
                    probs = get_prediction_probabilities(best_model, input_tfidf)[0]
                    elapsed = time.perf_counter() - start_t
                    latency_sec = max(elapsed, 0.005)

                    # Model Classes and Label Mapping Verification
                    model_classes = getattr(best_model, 'classes_', list(range(len(categories))))
                    
                    # Resolve predicted category
                    if isinstance(pred_raw, (int, np.integer)):
                        raw_class = categories[pred_raw] if pred_raw < len(categories) else f"class_{pred_raw}"
                    else:
                        raw_class = str(pred_raw)

                    disp_category = cat_display.get(raw_class, raw_class)
                    cat_icon = get_category_icon(disp_category, raw_class)

                    # Determine calibrated confidence percentage
                    if isinstance(pred_raw, (int, np.integer)) and pred_raw < len(probs):
                        confidence = float(probs[pred_raw]) * 100
                    else:
                        confidence = float(np.max(probs)) * 100

                    # Map probability distribution to categories
                    prob_breakdown = []
                    for i, p in enumerate(probs):
                        c_idx = model_classes[i] if i < len(model_classes) else i
                        c_name = categories[c_idx] if c_idx < len(categories) else f"Category {c_idx}"
                        c_disp = cat_display.get(c_name, c_name)
                        c_ico = get_category_icon(c_disp, c_name)
                        prob_breakdown.append({
                            "category": c_disp,
                            "raw_class": c_name,
                            "icon": c_ico,
                            "pct": float(p) * 100
                        })
                    prob_breakdown.sort(key=lambda x: x["pct"], reverse=True)

                    # 5. Extract top TF-IDF features
                    top_features = get_top_tfidf_terms_for_document(vectorizer, input_tfidf, top_n=6)

                    # Save prediction result in session state
                    st.session_state.last_prediction = {
                        "category": disp_category,
                        "raw_class": raw_class,
                        "icon": cat_icon,
                        "confidence": confidence,
                        "model_name": best_model_name,
                        "latency": latency_sec,
                        "prob_breakdown": prob_breakdown,
                        "top_features": top_features
                    }

                    # Append to recent classifications
                    snippet = (input_text[:65] + "...") if len(input_text) > 65 else input_text
                    st.session_state.recent_classifications.insert(0, {
                        "snippet": snippet,
                        "category": f"{cat_icon} {disp_category}",
                        "confidence": f"{confidence:.2f}%",
                        "latency": f"{latency_sec:.3f}s",
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    # Keep latest 10
                    st.session_state.recent_classifications = st.session_state.recent_classifications[:10]

            # Render Result or Idle State
            if st.session_state.last_prediction:
                res = st.session_state.last_prediction

                # Category & Confidence Banner
                st.markdown(f'''
                <div class="tc-pred-banner">
                    <div class="tc-pred-kicker">PREDICTED CATEGORY</div>
                    <div class="tc-pred-cat">
                        <span>{res['icon']}</span>
                        <span>{res['category']}</span>
                    </div>
                    <div>
                        <span class="tc-pred-conf-badge">&check; {res['confidence']:.2f}% confidence</span>
                    </div>
                    <div class="tc-pred-info-grid">
                        <div class="tc-pred-info-item">
                            <div class="tc-pred-info-lbl">MODEL USED</div>
                            <div class="tc-pred-info-val" title="{res['model_name']}">{res['model_name']}</div>
                        </div>
                        <div class="tc-pred-info-item">
                            <div class="tc-pred-info-lbl">LATENCY</div>
                            <div class="tc-pred-info-val">{res['latency']:.3f} sec</div>
                        </div>
                        <div class="tc-pred-info-item">
                            <div class="tc-pred-info-lbl">RAW CLASS</div>
                            <div class="tc-pred-info-val" title="{res['raw_class']}">{res['raw_class']}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # IMPORTANT TF-IDF FEATURES
                st.markdown("<div style='font-size:12px;font-weight:750;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>IMPORTANT TF-IDF FEATURES</div>", unsafe_allow_html=True)
                if res['top_features']:
                    tags_html = "".join([
                        f"<span class='tc-keyword-tag'>{term} <span class='tc-keyword-wt'>{weight:.3f}</span></span>"
                        for term, weight in res['top_features']
                    ])
                    st.markdown(f"<div class='tc-tag-wrap' style='margin-bottom:18px;'>{tags_html}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#64748b;font-size:12px;margin-bottom:16px;'>No significant vocabulary terms matched.</div>", unsafe_allow_html=True)

                # CLASSIFICATION CONFIDENCE BREAKDOWN
                st.markdown("<div style='font-size:12px;font-weight:750;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;'>CLASSIFICATION CONFIDENCE</div>", unsafe_allow_html=True)
                for item in res['prob_breakdown']:
                    pct_val = item['pct']
                    st.markdown(f'''
                    <div class="tc-conf-row">
                        <div class="tc-conf-label" title="{item['category']}">
                            <span>{item['icon']}</span>
                            <span>{item['category']}</span>
                        </div>
                        <div class="tc-conf-track">
                            <div class="tc-conf-fill" style="width: {pct_val:.1f}%;"></div>
                        </div>
                        <div class="tc-conf-pct">{pct_val:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

            else:
                # Before analysis: Idle placeholder state
                st.markdown('''
                <div class="tc-idle-box">
                    <div class="tc-idle-icon">🤖</div>
                    <div class="tc-idle-title">Ready to analyze</div>
                    <div class="tc-idle-desc">Enter a document and we'll automatically classify it.</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 3: ANALYTICS (Full Multi-Model Evaluation)
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Analytics":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📊 Multi-Model Performance Benchmarks</div>
                    <div class="tc-card-sub">Comprehensive evaluation across all 4 machine learning algorithms on identical stratified test partitions</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 4 Model Benchmark Cards
        st.markdown("<div class='tc-model-grid'>", unsafe_allow_html=True)
        cols_models = st.columns(4)
        for i, row in metrics_table.iterrows():
            m_name = row['Model']
            m_acc = row['Accuracy'] * 100
            m_prec = row.get('Precision (Weighted)', 0) * 100
            m_rec = row.get('Recall (Weighted)', 0) * 100
            m_f1 = row.get('F1-Score (Weighted)', 0) * 100
            is_prod = (m_name == best_model_name)

            with cols_models[i % 4]:
                st.markdown(f'''
                <div class="tc-model-card {'prod-badge-card' if is_prod else ''}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="tc-model-name">{m_name}</div>
                        {'<span style="font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;background:#6366f1;color:#fff;">PROD</span>' if is_prod else ''}
                    </div>
                    <div class="tc-model-acc">{m_acc:.1f}%</div>
                    <div class="tc-model-acc-lbl">Test Accuracy</div>
                    <div class="tc-model-metrics-row">
                        <span>Prec: <b>{m_prec:.1f}%</b></span>
                        <span>Rec: <b>{m_rec:.1f}%</b></span>
                        <span>F1: <b>{m_f1:.1f}%</b></span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Visualizations & Comparison Charts
        col_ch1, col_ch2 = st.columns([1, 1], gap="large")
        with col_ch1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Accuracy Comparison Across Classifiers</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/model_comparison.png"):
                st.image("visualizations/model_comparison.png", use_container_width=True)
            else:
                m_sorted = metrics_table.sort_values(by='Accuracy', ascending=True)
                st.bar_chart(m_sorted.set_index('Model')['Accuracy'] * 100, color="#6366f1")

        with col_ch2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Confusion Matrix Diagnostics</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/confusion_matrices.png"):
                st.image("visualizations/confusion_matrices.png", use_container_width=True)
            else:
                st.info("Confusion matrix visualization will be generated during training execution.")

        st.write("")

        # Comprehensive Metrics Comparison Table
        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Detailed Statistical Metrics Table</div>", unsafe_allow_html=True)
        st.dataframe(
            metrics_table.style.format({
                'Accuracy': '{:.2%}',
                'Precision (Weighted)': '{:.2%}',
                'Recall (Weighted)': '{:.2%}',
                'F1-Score (Weighted)': '{:.2%}',
                'F1-Score (Macro)': '{:.2%}',
                'Training Time (s)': '{:.3f}s'
            }),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 4: DATASET
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Dataset":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📁 Dataset & Feature Distribution</div>
                    <div class="tc-card-sub">Structural overview of the 20 Newsgroups corpus, splits, and TF-IDF feature space</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 5 Key Dataset Metrics
        d_c1, d_c2, d_c3, d_c4, d_c5 = st.columns(5)
        with d_c1:
            st.metric("Total Documents", f"{stats.get('total_documents', 3613):,}")
        with d_c2:
            st.metric("Training Samples", f"{int(stats.get('total_documents', 3613) * 0.8):,}")
        with d_c3:
            st.metric("Testing Samples", f"{int(stats.get('total_documents', 3613) * 0.2):,}")
        with d_c4:
            st.metric("Categories", f"{stats.get('num_classes', 4)}")
        with d_c5:
            vocab_len = len(getattr(vectorizer, 'vocabulary_', {})) if vectorizer else 5000
            st.metric("TF-IDF Features", f"{vocab_len:,}")

        st.write("")

        col_dt1, col_dt2 = st.columns([1, 1], gap="large")
        with col_dt1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Category Sample Distribution</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/class_distribution.png"):
                st.image("visualizations/class_distribution.png", use_container_width=True)
            else:
                dist_data = stats.get('class_distribution', {})
                df_dist = pd.DataFrame(list(dist_data.items()), columns=['Category', 'Count'])
                df_dist['Label'] = df_dist['Category'].apply(lambda x: cat_display.get(x, x))
                st.bar_chart(df_dist.set_index('Label')['Count'], color="#6366f1")

        with col_dt2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Top Discriminative Keywords per Category</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/top_keywords.png"):
                st.image("visualizations/top_keywords.png", use_container_width=True)
            else:
                st.info("Feature importance graphs saved in `visualizations/`.")

        st.markdown('''
        <div style="margin-top:20px;padding:16px;background:rgba(99, 102, 241, 0.06);border:1px solid rgba(99, 102, 241, 0.2);border-radius:12px;font-size:12.5px;color:#94a3b8;line-height:1.6;">
            <b style="color:#a5b4fc;">🛡️ Data Hygiene Guarantee:</b> All email headers, footers, sender signatures, and quotes were strictly removed prior to feature extraction to prevent artificial signal leakage. TF-IDF parameters were fit exclusively on the training partition.
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 5: HOW IT WORKS (Visual Pipeline)
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "How It Works":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">⚙️ How TextClassify Works</div>
                    <div class="tc-card-sub">Step-by-step visual machine learning pipeline from raw string to probabilistic class assignment</div>
                </div>
            </div>

            <div class="tc-pipeline-grid">
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 01</div>
                    <div class="tc-pipeline-title">DOCUMENT</div>
                    <div class="tc-pipeline-desc">Unstructured raw text is ingested from user input or automated payloads.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 02</div>
                    <div class="tc-pipeline-title">PREPROCESSING</div>
                    <div class="tc-pipeline-desc">Lowercasing, regex noise stripping, stop-word removal, and WordNet lemmatization.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 03</div>
                    <div class="tc-pipeline-title">TF-IDF</div>
                    <div class="tc-pipeline-desc">Unigram and bigram extraction with sublinear term frequency scaling and L2 normalization.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 04</div>
                    <div class="tc-pipeline-title">PRODUCTION MODEL</div>
                    <div class="tc-pipeline-desc">Automated inference through calibrated Linear Support Vector Machine.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 05</div>
                    <div class="tc-pipeline-title">CONFIDENCE</div>
                    <div class="tc-pipeline-desc">Platt scaling transforms geometric margins into calibrated class probabilities.</div>
                </div>
            </div>

            <div style="margin-top:28px;">
                <h4 style="color:#ffffff;font-size:16px;font-weight:750;margin-bottom:12px;">Mathematical Foundations</h4>
            </div>
        ''', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Sublinear Term Frequency & Inverse Document Frequency:**")
            st.latex(r"\text{TF}(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0")
            st.latex(r"\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")
        with col_m2:
            st.markdown("**L2 Unit Normalization & Support Vector Margin:**")
            st.latex(r"\mathbf{v}_{d} = \frac{\text{TF-IDF}(t, d)}{\|\text{TF-IDF}(t, d)\|_2}")
            st.latex(r"\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))")

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 6: ABOUT
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "About":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">ℹ️ About TextClassify</div>
                    <div class="tc-card-sub">Engineering specifications and architectural overview</div>
                </div>
            </div>

            <div style="font-size:14px;color:#cbd5e1;line-height:1.7;margin-bottom:20px;">
                <b>TextClassify</b> is an automated document intelligence platform designed to eliminate manual classification friction. 
                Rather than forcing users to guess which machine learning algorithm to select, the platform executes inference using a 
                rigorously validated, calibrated production model while providing full transparent confidence breakdowns and detected feature weights.
            </div>

            <div style="font-size:11px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">CORE TECHNOLOGIES</div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">
                <span class="tc-keyword-tag">Python 3.10+</span>
                <span class="tc-keyword-tag">Scikit-Learn</span>
                <span class="tc-keyword-tag">Linear Support Vector Machine</span>
                <span class="tc-keyword-tag">Platt Probability Calibration</span>
                <span class="tc-keyword-tag">Sublinear TF-IDF</span>
                <span class="tc-keyword-tag">Streamlit Enterprise</span>
                <span class="tc-keyword-tag">Joblib Serialization</span>
                <span class="tc-keyword-tag">NLTK & WordNet</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Global Footer
    st.markdown('''
    <div style="text-align:center;padding:28px 0 10px;font-size:12px;color:#475569;">
        ⚡ TextClassify &bull; Automated Document Intelligence Platform &bull; Production Inference Engine Online
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
