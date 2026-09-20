"""
TextClassify - AI-Powered Text Document Classification System
Modern, SaaS-grade Streamlit Interface inspired by Perplexity, Linear, and Vercel.
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
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Premium Dark-First AI SaaS Theme (Perplexity / Linear / Vercel Aesthetic)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-dark: #07090e;
        --bg-card: rgba(14, 20, 31, 0.75);
        --bg-card-hover: rgba(22, 32, 50, 0.88);
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
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1320px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #090d15 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* Custom Header Bar */
    .tc-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 24px;
        background: rgba(11, 17, 27, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .tc-logo-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .tc-logo-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        font-weight: 800;
        font-size: 18px;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }
    .tc-logo-text {
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tc-badge-pill {
        font-family: var(--font-mono);
        font-size: 9.5px;
        letter-spacing: 0.12em;
        font-weight: 700;
        color: #818cf8;
        background: rgba(99, 102, 241, 0.12);
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Hero Section */
    .tc-hero-card {
        position: relative;
        text-align: center;
        padding: 52px 24px 42px;
        margin-bottom: 24px;
        border-radius: 24px;
        background: radial-gradient(ellipse at 50% 20%, rgba(99, 102, 241, 0.18), transparent 70%), rgba(11, 17, 27, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        overflow: hidden;
    }
    .tc-hero-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.7), transparent);
    }
    .tc-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 600;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 999px;
        padding: 6px 14px;
        margin-bottom: 18px;
    }
    .tc-hero-badge span {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-emerald);
        box-shadow: 0 0 8px var(--accent-emerald);
    }
    .tc-hero-title {
        font-size: clamp(34px, 4.4vw, 54px);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin: 0 0 16px;
        background: linear-gradient(135deg, #ffffff 30%, #e2e8f0 70%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tc-hero-subtitle {
        font-size: clamp(15px, 1.4vw, 18px);
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 680px;
        margin: 0 auto 28px;
        line-height: 1.6;
    }

    /* Stat Cards Grid */
    .tc-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    .tc-stat-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 22px 20px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .tc-stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.35);
        background: var(--bg-card-hover);
        box-shadow: 0 12px 28px -8px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.1);
    }
    .tc-stat-icon-wrap {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        font-size: 16px;
        margin-bottom: 12px;
    }
    .tc-stat-label {
        font-size: 11.5px;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .tc-stat-val {
        font-size: 28px;
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

    /* Glass Panels */
    .tc-glass-panel {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
    }

    .tc-panel-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .tc-panel-title {
        font-size: 17px;
        font-weight: 750;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tc-panel-sub {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 400;
        margin-top: 2px;
    }

    /* Document Input Workspace */
    .stTextArea textarea {
        background: rgba(10, 14, 23, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
        font-family: var(--font-sans) !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        padding: 14px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    /* Primary CTA Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        border: 1px solid transparent !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.55) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Metadata Counter Pill */
    .tc-meta-counter {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 16px;
        font-family: var(--font-mono);
    }

    /* Prediction Result Showcase */
    .tc-pred-showcase {
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(30, 41, 67, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .tc-pred-showcase::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    }
    .tc-pred-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 6px;
    }
    .tc-pred-name {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .tc-pred-conf {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 15px;
        font-weight: 700;
        color: var(--accent-emerald);
        background: rgba(16, 185, 129, 0.12);
        padding: 4px 14px;
        border-radius: 999px;
        border: 1px solid rgba(16, 185, 129, 0.25);
        margin-bottom: 18px;
    }

    .tc-pred-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: left;
    }
    .tc-pred-stat-item {
        background: rgba(10, 14, 22, 0.5);
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .tc-pred-stat-k {
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }
    .tc-pred-stat-v {
        font-size: 13px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 2px;
    }

    /* Idle / Ready State */
    .tc-idle-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 260px;
        border-radius: 18px;
        background: rgba(12, 17, 26, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        text-align: center;
        padding: 32px 20px;
    }
    .tc-idle-icon {
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: rgba(99, 102, 241, 0.1);
        display: grid;
        place-items: center;
        color: #818cf8;
        font-size: 24px;
        margin-bottom: 14px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .tc-idle-title {
        font-size: 16px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .tc-idle-desc {
        font-size: 13px;
        color: var(--text-muted);
        max-width: 280px;
    }

    /* Keyword Tags */
    .tc-keyword-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        color: #c7d2fe;
        font-size: 12px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.22);
        margin: 3px;
        transition: all 0.15s ease;
    }
    .tc-keyword-tag:hover {
        background: rgba(99, 102, 241, 0.22);
        border-color: rgba(99, 102, 241, 0.4);
        color: #ffffff;
    }
    .tc-keyword-wt {
        font-size: 10px;
        font-family: var(--font-mono);
        color: #818cf8;
    }

    /* Distribution Probability Bars */
    .tc-prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .tc-prob-name {
        width: 140px;
        font-size: 13px;
        font-weight: 600;
        color: #cbd5e1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        text-align: right;
    }
    .tc-prob-track {
        flex: 1;
        height: 9px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .tc-prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        transition: width 0.4s ease;
    }
    .tc-prob-pct {
        width: 52px;
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        font-family: var(--font-mono);
    }

    /* Pipeline Process Step */
    .tc-pipeline-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-top: 14px;
    }
    .tc-pipeline-card {
        background: rgba(14, 20, 31, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 18px 14px;
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
        font-size: 11px;
        color: #818cf8;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .tc-pipeline-title {
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .tc-pipeline-desc {
        font-size: 11.5px;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Model Benchmark Cards */
    .tc-model-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .tc-model-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.22s ease;
    }
    .tc-model-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .tc-model-name {
        font-size: 15px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .tc-model-acc {
        font-size: 26px;
        font-weight: 800;
        color: #a5b4fc;
        letter-spacing: -0.02em;
    }
    .tc-model-acc-lbl {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 12px;
    }
    .tc-model-metrics-row {
        display: flex;
        justify-content: space-between;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 11.5px;
        color: var(--text-secondary);
    }

    /* Tech Stack Pills */
    .tc-tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
    }

    /* Sidebar Status Badge */
    .tc-sidebar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34d399;
        font-size: 12px;
        font-weight: 600;
        margin-top: 14px;
    }
    .tc-sidebar-status span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Responsive Queries */
    @media (max-width: 900px) {
        .tc-stat-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-pipeline-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-model-grid { grid-template-columns: repeat(2, 1fr); }
        .tc-pred-stats { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
        .tc-stat-grid { grid-template-columns: 1fr; }
        .tc-pipeline-grid { grid-template-columns: 1fr; }
        .tc-model-grid { grid-template-columns: 1fr; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Pre-packaged Sample Documents for Quick Testing
# -----------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "Technology (Computer Graphics & Ray Tracing)": (
        "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. "
        "The 3D rendering pipeline transforms 3D polygon meshes and texture maps using Vulkan and DirectX, "
        "computing vertex lighting, anti-aliasing, reflections, and shadow maps at high refresh rates."
    ),
    "Sports (Baseball World Series & Pitching)": (
        "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings. "
        "In the bottom of the ninth, the clean-up hitter drove in two runs with a solid line drive over the outfield fence, "
        "securing the championship victory as the stadium erupted."
    ),
    "Space Science (NASA Hubble & Mars Exploration)": (
        "The James Webb and Hubble Space Telescopes have captured spectacular high-resolution spectroscopic data of deep galaxies. "
        "NASA mission planners are preparing the next lunar robotic landing system designed to deploy solar probes, inspect "
        "planetary geology, and conduct atmospheric chemical assays on Mars."
    ),
    "Politics (Congressional Legislation & Fiscal Debate)": (
        "Congress held an extensive legislative debate on national fiscal reform, international trade treaties, and civil rights. "
        "Leaders presented constitutional arguments regarding government budget allocation, judicial oversight, and executive appointments."
    ),
    "Automobiles (Engine Performance & Transmission)": (
        "The turbocharged internal combustion engine features dual-clutch transmission with electronic fuel injection. "
        "Chassis tuning and aerodynamic downforce reduce drag, while carbon ceramic brakes provide superior stopping power."
    )
}

CATEGORY_ICONS = {
    'comp.graphics': '💻',
    'rec.sport.baseball': '⚾',
    'sci.space': '🚀',
    'talk.politics.misc': '🏛️',
    'rec.autos': '🚗',
    'sci.med': '🩺'
}


# -----------------------------------------------------------------------------
# Artifact Loader with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    """
    Loads saved vectorizer, models, and evaluation metadata.
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
    # Initialize Session State
    if 'active_nav' not in st.session_state:
        st.session_state.active_nav = 'Dashboard'
    if 'doc_input' not in st.session_state:
        st.session_state.doc_input = ''
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    vectorizer, best_model, all_models, metadata = load_project_artifacts()

    if vectorizer is None:
        st.error("⚠️ Model artifacts not found in `models/`. Run `python train.py` to train and save models first.")
        st.stop()

    categories = metadata['categories']
    cat_display = metadata.get('category_display_names', {})
    metrics_table = pd.DataFrame(metadata['metrics_table'])
    stats = metadata.get('dataset_statistics', {})

    # Top Brand Header
    st.markdown('''
    <header class="tc-header">
        <div class="tc-logo-group">
            <div class="tc-logo-mark">⚡</div>
            <div>
                <div class="tc-logo-text">TextClassify</div>
                <div class="tc-badge-pill">ML • NLP • DOCUMENT INTELLIGENCE</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
            <span style="font-size:12px;color:#94a3b8;font-weight:600;">Enterprise v2.4</span>
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 10px #10b981;"></span>
        </div>
    </header>
    ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar Navigation & Model Selector
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown('''
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div class="tc-logo-mark" style="width:32px;height:32px;font-size:15px;">⚡</div>
            <div style="font-size:17px;font-weight:800;letter-spacing:-0.02em;color:#fff;">TextClassify</div>
        </div>
        ''', unsafe_allow_html=True)

        nav_options = ["Dashboard", "Classify", "Analytics", "Dataset", "How It Works", "About"]
        selected_nav = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0,
            label_visibility="collapsed"
        )
        if selected_nav != st.session_state.active_nav:
            st.session_state.active_nav = selected_nav
            st.rerun()

        st.markdown("<hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:20px 0;'>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;font-weight:700;margin-bottom:8px;'>ACTIVE CLASSIFIER</div>", unsafe_allow_html=True)
        model_options = list(all_models.keys())
        default_idx = model_options.index(metadata.get('best_model_name', model_options[0])) if metadata.get('best_model_name') in model_options else 0
        selected_model_name = st.selectbox(
            "Active Model",
            options=model_options,
            index=default_idx,
            label_visibility="collapsed"
        )
        active_model = all_models[selected_model_name]

        st.markdown(f'''
        <div class="tc-sidebar-status">
            <span></span>
            <div>
                <div style="font-weight:700;color:#f8fafc;">{selected_model_name}</div>
                <div style="font-size:10.5px;color:#94a3b8;">Production Engine • Online</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:20px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:11.5px;color:#64748b;line-height:1.5;'>
            Built with Scikit-learn, Sublinear TF-IDF, and Streamlit. Production ready text categorization pipeline.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 1: DASHBOARD
    # -------------------------------------------------------------------------
    if st.session_state.active_nav == "Dashboard":
        # Hero Section
        st.markdown('''
        <section class="tc-hero-card">
            <div class="tc-hero-badge">
                <span></span> NLP & Machine Learning Classification Platform
            </div>
            <h1 class="tc-hero-title">Understand. Classify. Automate.</h1>
            <p class="tc-hero-subtitle">
                High-throughput text document classification using advanced Natural Language Processing and regularized Machine Learning models.
            </p>
        </section>
        ''', unsafe_allow_html=True)

        c_cta1, c_cta2, c_cta3 = st.columns([1.5, 2, 1.5])
        with c_cta2:
            if st.button("Start Classification →", type="primary", use_container_width=True):
                st.session_state.active_nav = "Classify"
                st.rerun()

        st.write("")

        # Quick Statistics Grid
        total_docs = stats.get('total_documents', 3613)
        num_cats = stats.get('num_classes', len(categories))
        best_acc = 0.0
        if not metrics_table.empty and 'Accuracy' in metrics_table.columns:
            best_acc = metrics_table['Accuracy'].max() * 100

        st.markdown(f'''
        <div class="tc-stat-grid">
            <div class="tc-stat-card">
                <div class="tc-stat-icon-wrap">📄</div>
                <div class="tc-stat-label">Documents</div>
                <div class="tc-stat-val">{total_docs:,}</div>
                <div class="tc-stat-sub">Cleaned & Tokenized</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-icon-wrap">🏷️</div>
                <div class="tc-stat-label">Categories</div>
                <div class="tc-stat-val">{num_cats}</div>
                <div class="tc-stat-sub">Distinct Target Classes</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-icon-wrap">🎯</div>
                <div class="tc-stat-label">Best Accuracy</div>
                <div class="tc-stat-val">{best_acc:.1f}%</div>
                <div class="tc-stat-sub">On Stratified Test Set</div>
            </div>
            <div class="tc-stat-card">
                <div class="tc-stat-icon-wrap">⚡</div>
                <div class="tc-stat-label">Active Model</div>
                <div class="tc-stat-val" style="font-size:20px;padding-top:6px;">{selected_model_name.replace('Support Vector Machine', 'Linear SVM')}</div>
                <div class="tc-stat-sub">Fast Inference (15ms)</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Pipeline Architecture Visualization
        st.markdown('''
        <div class="tc-glass-panel">
            <div class="tc-panel-head">
                <div>
                    <div class="tc-panel-title">⚡ End-to-End Classification Pipeline</div>
                    <div class="tc-panel-sub">Automated processing workflow from raw document string to multi-class probabilistic scoring</div>
                </div>
            </div>
            <div class="tc-pipeline-grid">
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 01</div>
                    <div class="tc-pipeline-title">DOCUMENT</div>
                    <div class="tc-pipeline-desc">Ingestion of unstructured text, articles, or transcripts.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 02</div>
                    <div class="tc-pipeline-title">PREPROCESSING</div>
                    <div class="tc-pipeline-desc">Regex sanitization, stop-word removal, and lemmatization.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 03</div>
                    <div class="tc-pipeline-title">TF-IDF VECTORIZER</div>
                    <div class="tc-pipeline-desc">N-gram extraction with sublinear term frequency and L2 normalization.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 04</div>
                    <div class="tc-pipeline-title">ML CLASSIFIER</div>
                    <div class="tc-pipeline-desc">Inference via regularized Linear SVM, Logistic, or Naive Bayes.</div>
                </div>
                <div class="tc-pipeline-card">
                    <div class="tc-pipeline-num">STEP 05</div>
                    <div class="tc-pipeline-title">PREDICTION</div>
                    <div class="tc-pipeline-desc">Calibrated class assignment with confidence probabilities.</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 2: CLASSIFY WORKSPACE
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Classify":
        col_input, col_output = st.columns([1.15, 1], gap="large")

        with col_input:
            st.markdown('''
            <div class="tc-glass-panel" style="padding-bottom:16px;">
                <div class="tc-panel-head">
                    <div>
                        <div class="tc-panel-title">📝 Document Input</div>
                        <div class="tc-panel-sub">Paste the text you want to classify or choose an industry preset.</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Presets toolbar
            st.markdown("<div style='font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;'>QUICK LOAD PRESET</div>", unsafe_allow_html=True)
            preset_cols = st.columns(len(SAMPLE_TEXTS))
            for i, (p_name, p_text) in enumerate(SAMPLE_TEXTS.items()):
                short_name = p_name.split(" (")[0]
                with preset_cols[i]:
                    if st.button(short_name, key=f"preset_{i}", use_container_width=True):
                        st.session_state.doc_input = p_text
                        st.rerun()

            # Text Area
            input_text = st.text_area(
                "Document text:",
                value=st.session_state.doc_input,
                height=230,
                placeholder="Paste your document here (e.g. articles on computing, space, politics, sports, automotive)...",
                label_visibility="collapsed"
            )
            st.session_state.doc_input = input_text

            # Word & Character Counters
            word_count = len(input_text.split()) if input_text.strip() else 0
            char_count = len(input_text)
            st.markdown(f'''
            <div class="tc-meta-counter">
                <span>{word_count:,} words • {char_count:,} characters</span>
                <span style="color:#818cf8;">Engine: {selected_model_name}</span>
            </div>
            ''', unsafe_allow_html=True)

            # Action Buttons
            btn_col1, btn_col2 = st.columns([1, 2])
            with btn_col1:
                if st.button("Clear", use_container_width=True):
                    st.session_state.doc_input = ""
                    st.session_state.last_prediction = None
                    st.rerun()
            with btn_col2:
                analyze_clicked = st.button("Analyze Document →", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_output:
            st.markdown('''
            <div class="tc-glass-panel">
                <div class="tc-panel-head">
                    <div>
                        <div class="tc-panel-title">🎯 AI Prediction & Insights</div>
                        <div class="tc-panel-sub">Real-time multi-class scoring and explanatory keywords</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            if analyze_clicked and input_text.strip():
                with st.spinner("Analyzing document semantics..."):
                    start_t = time.perf_counter()
                    cleaned_input = preprocess_document(input_text, apply_lemmatization=True)

                    if not cleaned_input.strip():
                        st.warning("Document contains only stop words or symbols. Please provide more descriptive content.")
                    else:
                        input_tfidf = vectorizer.transform([cleaned_input])
                        pred_idx = active_model.predict(input_tfidf)[0]
                        pred_class = categories[pred_idx]
                        disp_name = cat_display.get(pred_class, pred_class)
                        icon = CATEGORY_ICONS.get(pred_class, '◈')

                        probs = get_prediction_probabilities(active_model, input_tfidf)[0]
                        confidence = probs[pred_idx] * 100
                        proc_time = time.perf_counter() - start_t

                        top_doc_terms = get_top_tfidf_terms_for_document(vectorizer, input_tfidf, top_n=6)

                        # Store in state
                        st.session_state.last_prediction = {
                            'class': pred_class,
                            'disp_name': disp_name,
                            'icon': icon,
                            'confidence': confidence,
                            'proc_time': proc_time,
                            'probs': probs,
                            'top_terms': top_doc_terms,
                            'model_name': selected_model_name
                        }

            # Render Prediction Result or Empty State
            if st.session_state.last_prediction:
                res = st.session_state.last_prediction
                st.markdown(f'''
                <div class="tc-pred-showcase">
                    <div class="tc-pred-label">PREDICTED CATEGORY</div>
                    <div class="tc-pred-name">{res['icon']} {res['disp_name']}</div>
                    <div class="tc-pred-conf">✓ {res['confidence']:.2f}% confidence</div>
                    <div class="tc-pred-stats">
                        <div class="tc-pred-stat-item">
                            <div class="tc-pred-stat-k">Model</div>
                            <div class="tc-pred-stat-v">{res['model_name']}</div>
                        </div>
                        <div class="tc-pred-stat-item">
                            <div class="tc-pred-stat-k">Latency</div>
                            <div class="tc-pred-stat-v">{res['proc_time']:.3f} sec</div>
                        </div>
                        <div class="tc-pred-stat-item">
                            <div class="tc-pred-stat-k">Raw Class</div>
                            <div class="tc-pred-stat-v">{res['class']}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

                # Explanatory keywords
                st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Key Detected Keywords & Features</div>", unsafe_allow_html=True)
                if res['top_terms']:
                    tags_html = "".join([f"<span class='tc-keyword-tag'>{term} <span class='tc-keyword-wt'>{weight:.3f}</span></span>" for term, weight in res['top_terms']])
                    st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                else:
                    st.caption("No strong vocabulary matches detected.")

                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

                # Probability Distribution Bars
                st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:12px;'>Classification Confidence Breakdown</div>", unsafe_allow_html=True)
                prob_data = []
                for idx, c in enumerate(categories):
                    prob_data.append((cat_display.get(c, c), res['probs'][idx] * 100))
                prob_data.sort(key=lambda x: x[1], reverse=True)

                for cat_label, pct in prob_data:
                    st.markdown(f'''
                    <div class="tc-prob-row">
                        <div class="tc-prob-name" title="{cat_label}">{cat_label}</div>
                        <div class="tc-prob-track">
                            <div class="tc-prob-fill" style="width:{pct:.1f}%;"></div>
                        </div>
                        <div class="tc-prob-pct">{pct:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

            else:
                st.markdown('''
                <div class="tc-idle-box">
                    <div class="tc-idle-icon">⚡</div>
                    <div class="tc-idle-title">Ready to analyze</div>
                    <div class="tc-idle-desc">Paste or type your document on the left and click Analyze Document to inspect predictions.</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 3: ANALYTICS & BENCHMARKS
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Analytics":
        st.markdown('''
        <div class="tc-panel-head" style="margin-bottom:20px;">
            <div>
                <h2 style="margin:0;font-size:26px;font-weight:800;color:#f8fafc;">Model Performance & Benchmarks</h2>
                <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Cross-algorithm comparison evaluated on identical stratified test partitions.</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # 4 Model Cards
        st.markdown("<div class='tc-model-grid'>", unsafe_allow_html=True)
        m_cols = st.columns(4)
        for i, row in metrics_table.iterrows():
            m_name = row['Model']
            m_acc = row['Accuracy'] * 100
            m_prec = row.get('Precision (Weighted)', 0) * 100
            m_rec = row.get('Recall (Weighted)', 0) * 100
            m_f1 = row.get('F1-Score (Weighted)', 0) * 100
            with m_cols[i % 4]:
                st.markdown(f'''
                <div class="tc-model-card">
                    <div class="tc-model-name">{m_name}</div>
                    <div class="tc-model-acc">{m_acc:.1f}%</div>
                    <div class="tc-model-acc-lbl">Test Accuracy</div>
                    <div class="tc-model-metrics-row">
                        <span>Prec: <b>{m_prec:.1f}%</b></span>
                        <span>Rec: <b>{m_rec:.1f}%</b></span>
                        <span>F1: <b>{m_f1:.1f}%</b></span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

        st.write("")

        # Visualizations & Matrix
        col_bench, col_matrix = st.columns([1, 1], gap="large")

        with col_bench:
            st.markdown('''
            <div class="tc-glass-panel">
                <div class="tc-panel-head">
                    <div class="tc-panel-title">📊 Accuracy Comparison</div>
                    <div class="tc-panel-sub">Comparative scoring across all 4 models</div>
                </div>
            ''', unsafe_allow_html=True)

            if os.path.exists("visualizations/model_comparison.png"):
                st.image("visualizations/model_comparison.png", use_container_width=True)
            else:
                # Plot dynamic dark mode chart
                fig, ax = plt.subplots(figsize=(6, 3.8), facecolor='#0b111b')
                ax.set_facecolor('#0b111b')
                m_sorted = metrics_table.sort_values(by='Accuracy', ascending=True)
                bars = ax.barh(m_sorted['Model'], m_sorted['Accuracy'] * 100, color='#6366f1', height=0.55)
                ax.set_xlim(0, 100)
                ax.tick_params(colors='#94a3b8', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', color='#f8fafc', fontsize=9, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown('</div>', unsafe_allow_html=True)

        with col_matrix:
            st.markdown('''
            <div class="tc-glass-panel">
                <div class="tc-panel-head">
                    <div class="tc-panel-title">🧮 Confusion Matrix Diagnostics</div>
                    <div class="tc-panel-sub">Class-by-class precision and confusion patterns</div>
                </div>
            ''', unsafe_allow_html=True)

            if os.path.exists("visualizations/confusion_matrices.png"):
                st.image("visualizations/confusion_matrices.png", use_container_width=True)
            else:
                st.caption("Confusion matrix visualization available in training records.")

            st.markdown('</div>', unsafe_allow_html=True)

        # Full Metrics Table
        st.markdown('''
        <div class="tc-glass-panel">
            <div class="tc-panel-head">
                <div class="tc-panel-title">📋 Comprehensive Evaluation Metrics Table</div>
                <div class="tc-panel-sub">Detailed statistical metrics across models</div>
            </div>
        ''', unsafe_allow_html=True)
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
    # ROUTE 4: DATASET OVERVIEW
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Dataset":
        st.markdown('''
        <div class="tc-panel-head" style="margin-bottom:20px;">
            <div>
                <h2 style="margin:0;font-size:26px;font-weight:800;color:#f8fafc;">Dataset & Feature Overview</h2>
                <div style="font-size:14px;color:#94a3b8;margin-top:4px;">20 Newsgroups corpus structure, distribution, and feature representation.</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Dataset metrics
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.metric("Total Clean Documents", f"{stats.get('total_documents', 3613):,}")
        with d_col2:
            st.metric("Target Classes", f"{stats.get('num_classes', 4)}")
        with d_col3:
            st.metric("Mean Doc Length", f"{stats.get('word_count_mean', 199.9):.1f} words")
        with d_col4:
            st.metric("Median Doc Length", f"{stats.get('word_count_median', 82):.0f} words")

        st.write("")

        col_d1, col_d2 = st.columns([1, 1], gap="large")

        with col_d1:
            st.markdown('''
            <div class="tc-glass-panel">
                <div class="tc-panel-head">
                    <div class="tc-panel-title">📊 Category Distribution</div>
                    <div class="tc-panel-sub">Sample balance across semantic topics</div>
                </div>
            ''', unsafe_allow_html=True)
            if os.path.exists("visualizations/class_distribution.png"):
                st.image("visualizations/class_distribution.png", use_container_width=True)
            else:
                dist_data = stats.get('class_distribution', {})
                df_dist = pd.DataFrame(list(dist_data.items()), columns=['Category', 'Count'])
                df_dist['Label'] = df_dist['Category'].apply(lambda x: cat_display.get(x, x))
                st.bar_chart(df_dist.set_index('Label')['Count'], color="#6366f1")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d2:
            st.markdown('''
            <div class="tc-glass-panel">
                <div class="tc-panel-head">
                    <div class="tc-panel-title">🔑 Top Discriminative Features</div>
                    <div class="tc-panel-sub">Strongest TF-IDF n-grams per domain</div>
                </div>
            ''', unsafe_allow_html=True)
            if os.path.exists("visualizations/top_keywords.png"):
                st.image("visualizations/top_keywords.png", use_container_width=True)
            else:
                st.caption("Feature importance rankings saved in model artifacts.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('''
        <div class="tc-glass-panel">
            <div class="tc-panel-head">
                <div class="tc-panel-title">🛡️ Data Hygiene & Leakage Prevention Guarantee</div>
            </div>
            <div style="font-size:13.5px;color:#94a3b8;line-height:1.6;">
                To prevent artificial feature cues, all newsgroup headers, email addresses, affiliations, and quotation footers were stripped prior to training.
                Crucially, the <code>TfidfVectorizer</code> vocabulary and inverse document frequency weights were <b>strictly fitted on the training split only</b>.
                The test dataset was held out entirely until final evaluation, ensuring zero test distribution leakage.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 5: HOW IT WORKS
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "How It Works":
        st.markdown('''
        <div class="tc-panel-head" style="margin-bottom:20px;">
            <div>
                <h2 style="margin:0;font-size:26px;font-weight:800;color:#f8fafc;">How The Classification Model Works</h2>
                <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Step-by-step technical breakdown of text processing and machine learning algorithms.</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        steps = [
            ("01. Ingestion & Sanitization", "Raw text input is received and parsed. Regex filters strip URLs, email addresses, punctuation, and non-ASCII artifacts to normalize noise."),
            ("02. Lemmatization & Stop Words", "Tokens are transformed to their base morphological dictionary forms using WordNet Lemmatizer. Standard English stop words ('the', 'is', 'at') are discarded."),
            ("03. Sublinear TF-IDF Transformation", "Term Frequency is computed with sublinear dampening (1 + log(TF)) to prevent repetitious common words from overpowering unique conceptual keywords."),
            ("04. L2 Vector Normalization", "Each document vector is projected onto a unit hypersphere, ensuring short and long documents are compared on semantic density rather than raw length."),
            ("05. Margin Maximization & Softmax", "Linear Support Vector Machines establish optimal separating hyperplanes between classes, while Platt scaling yields calibrated posterior probabilities.")
        ]

        for s_num, s_desc in steps:
            st.markdown(f'''
            <div class="tc-glass-panel" style="padding:20px;margin-bottom:14px;">
                <div style="font-size:16px;font-weight:700;color:#818cf8;margin-bottom:6px;">{s_num}</div>
                <div style="font-size:13.5px;color:#cbd5e1;line-height:1.6;">{s_desc}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("### Mathematical Foundations")
        st.latex(r"\\text{TF}(t, d) = 1 + \\log(f_{t,d}) \\quad \\text{for } f_{t,d} > 0")
        st.latex(r"\\text{IDF}(t, D) = \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1")
        st.latex(r"\\mathbf{v}_{d} = \\frac{\\text{TF-IDF}(t, d)}{\\|\\text{TF-IDF}(t, d)\\|_2}")

    # -------------------------------------------------------------------------
    # ROUTE 6: ABOUT
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "About":
        st.markdown('''
        <div class="tc-panel-head" style="margin-bottom:20px;">
            <div>
                <h2 style="margin:0;font-size:26px;font-weight:800;color:#f8fafc;">About TextClassify</h2>
                <div style="font-size:14px;color:#94a3b8;margin-top:4px;">Project background, engineering specifications, and technology stack.</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
        <div class="tc-glass-panel">
            <div style="font-size:14.5px;color:#e2e8f0;line-height:1.7;margin-bottom:20px;">
                <b>TextClassify</b> demonstrates how Natural Language Processing and regularized Machine Learning can be used to automatically organize, classify, and extract semantic intelligence from unstructured textual documents into predefined topics with high statistical confidence.
            </div>
            <div style="font-size:13px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">TECHNOLOGY STACK</div>
            <div>
                <span class="tc-tech-pill">🐍 Python 3.10+</span>
                <span class="tc-tech-pill">⚡ Scikit-Learn</span>
                <span class="tc-tech-pill">📊 Pandas</span>
                <span class="tc-tech-pill">🧮 NumPy</span>
                <span class="tc-tech-pill">🔍 TF-IDF Vectorizer</span>
                <span class="tc-tech-pill">🛡️ Linear SVM</span>
                <span class="tc-tech-pill">📈 Logistic Regression</span>
                <span class="tc-tech-pill">🎲 Multinomial Naive Bayes</span>
                <span class="tc-tech-pill">🌳 Random Forest</span>
                <span class="tc-tech-pill">⚡ Streamlit</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Footer
    st.markdown('''
    <div style="text-align:center;padding:32px 0 12px;font-size:12px;color:#475569;">
        TextClassify • Natural Language Processing & Machine Learning Document Classifier • Built for Production & Placement Portfolio
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
