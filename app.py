"""
Streamlit Web Application: Classifying Text Documents Using Machine Learning

An interactive dashboard and prediction interface for real-time document classification,
model comparison, confusion matrix inspection, and TF-IDF feature analysis.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Document Classifier | Machine Learning",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pred-box {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .pred-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .pred-class {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .keyword-pill {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Pre-packaged Sample Documents for Quick Testing
# -----------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "Select a preset sample...": "",
    "Space Science (NASA Hubble & Mars Exploration)": (
        "The Hubble Space Telescope and the James Webb Space Telescope have captured "
        "spectacular high-resolution images of deep space galaxies, orbiting satellites, and planetary nebulae. "
        "NASA engineers are preparing the next lunar mission with an advanced rocket launch system "
        "designed to place solar probes into high earth orbit and conduct atmospheric spectroscopic analysis on Mars."
    ),
    "Baseball (MLB World Series & Pitching Stats)": (
        "The starting pitcher delivered a dazzling performance with nine strikeouts over seven scoreless innings. "
        "In the bottom of the ninth, the home run hitter drove in two runs with a solid line drive over the left field fence, "
        "securing the division championship for the team as the crowd cheered the walk-off victory."
    ),
    "Computer Graphics (3D Rendering & GPU Shaders)": (
        "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. "
        "The 3D rendering pipeline transforms 3D polygon meshes and texture maps using OpenGL or Vulkan, "
        "computing vertex lighting, anti-aliasing, and photorealistic reflections at 60 frames per second."
    ),
    "Politics (Congressional Debate & Foreign Policy)": (
        "Congress held an extensive legislative debate on national fiscal policy, international trade treaties, "
        "and civil liberties. Both bipartisan leaders presented arguments regarding government spending reforms, "
        "diplomatic negotiations, and the constitutional role of the senate in approving executive judicial appointments."
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
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">📄 Classifying Text Documents Using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An End-to-End NLP System for Automated Text Categorization, Model Benchmarking, and TF-IDF Analytics</div>', unsafe_allow_html=True)

    vectorizer, best_model, all_models, metadata = load_project_artifacts()

    if vectorizer is None:
        st.warning("⚠️ Model artifacts not found in `models/`. Please train the models first by running `python train.py` in your terminal.")
        st.stop()

    categories = metadata['categories']
    cat_display = metadata.get('category_display_names', {})
    metrics_table = pd.DataFrame(metadata['metrics_table'])

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    st.sidebar.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=600&auto=format&fit=crop", use_container_width=True)
    st.sidebar.title("Configuration & Models")

    selected_model_name = st.sidebar.selectbox(
        "Select Active Classifier:",
        options=list(all_models.keys()),
        index=list(all_models.keys()).index(metadata['best_model_name'])
    )

    active_model = all_models[selected_model_name]

    st.sidebar.divider()
    st.sidebar.markdown(f"**Best Model**: `{metadata['best_model_name']}`")
    st.sidebar.markdown(f"**Total Documents**: `{metadata['dataset_statistics']['total_documents']}`")
    st.sidebar.markdown(f"**Categories**: `{len(categories)}`")
    st.sidebar.markdown(f"**Vocabulary Features**: `{len(vectorizer.vocabulary_):,}`")

    st.sidebar.divider()
    st.sidebar.info(
        "💡 **Quick Tip**: Select a pre-loaded sample from the dropdown on the Live Classifier tab to immediately inspect predictions and top TF-IDF keywords."
    )

    # -------------------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Live Classifier",
        "📊 Model Comparison",
        "🔍 Feature & Keyword Analysis",
        "📁 Dataset Insights",
        "📖 Project Methodology & Report"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Live Classifier
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Interactive Document Classification")
        st.markdown(
            "Enter any text document, article, or paragraph below. The system cleans the text, extracts TF-IDF numerical features, "
            f"and classifies it using **{selected_model_name}**."
        )

        col_sample, col_clear = st.columns([4, 1])
        with col_sample:
            chosen_sample = st.selectbox("Choose a sample preset:", list(SAMPLE_TEXTS.keys()))

        initial_text = SAMPLE_TEXTS.get(chosen_sample, "")

        user_input = st.text_area(
            "Enter or paste text document:",
            value=initial_text,
            height=180,
            placeholder="Type or paste text here (e.g. an article on astronomy, baseball game summary, computer graphics, or politics)..."
        )

        classify_btn = st.button("🚀 Classify Document", type="primary", use_container_width=True)

        if classify_btn or (user_input.strip() and chosen_sample != "Select a preset sample..."):
            if not user_input.strip():
                st.error("Please enter some text before classifying.")
            else:
                with st.spinner("Preprocessing text and running classifier..."):
                    # 1. Preprocess input text
                    cleaned_input = preprocess_document(user_input, apply_lemmatization=True)

                    if not cleaned_input.strip():
                        st.warning("The input text contained only stop words or punctuation. Please provide more descriptive content.")
                    else:
                        # 2. Extract TF-IDF features
                        input_tfidf = vectorizer.transform([cleaned_input])

                        # 3. Predict class and probabilities
                        pred_idx = active_model.predict(input_tfidf)[0]
                        pred_class = categories[pred_idx]
                        disp_name = cat_display.get(pred_class, pred_class)
                        icon = CATEGORY_ICONS.get(pred_class, '📌')

                        # Get probabilities
                        probs = get_prediction_probabilities(active_model, input_tfidf)[0]
                        confidence = probs[pred_idx] * 100

                        # Get top TF-IDF words from this input
                        top_doc_terms = get_top_tfidf_terms_for_document(vectorizer, input_tfidf, top_n=8)

                        # Render Prediction Result Card
                        st.markdown(f"""
                        <div class="pred-box">
                            <div class="pred-title">Predicted Document Category</div>
                            <div class="pred-class">{icon} {disp_name} ({pred_class})</div>
                            <div style="margin-top: 8px; font-size: 1.15rem; font-weight: 500;">
                                Confidence: <strong>{confidence:.1f}%</strong> | Classifier: <em>{selected_model_name}</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.write("")

                        col_probs, col_terms = st.columns([3, 2])

                        with col_probs:
                            st.markdown("##### 📈 Prediction Probability Distribution")
                            prob_df = pd.DataFrame({
                                'Category': [cat_display.get(c, c) for c in categories],
                                'Probability (%)': [round(p * 100, 2) for p in probs]
                            }).sort_values(by='Probability (%)', ascending=True)

                            fig, ax = plt.subplots(figsize=(6, 2.8), dpi=200)
                            bars = ax.barh(prob_df['Category'], prob_df['Probability (%)'], color='#2563eb')
                            ax.set_xlim(0, 100)
                            ax.set_xlabel('Probability (%)', fontsize=9)
                            for bar in bars:
                                w = bar.get_width()
                                ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontsize=8, fontweight='bold')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        with col_terms:
                            st.markdown("##### 🔑 Key Informative Terms in Document")
                            st.markdown("Most influential TF-IDF terms extracted from this document:")
                            if top_doc_terms:
                                for term, weight in top_doc_terms:
                                    st.markdown(f'<span class="keyword-pill">{term} ({weight:.3f})</span>', unsafe_allow_html=True)
                            else:
                                st.write("No strong matching vocabulary terms detected.")

                            with st.expander("Show Preprocessed Text"):
                                st.code(cleaned_input)

    # -------------------------------------------------------------------------
    # TAB 2: Model Comparison
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Model Performance & Benchmarking")
        st.markdown(
            "We trained and evaluated four distinct classification algorithms on identical stratified test data. "
            "Below is the complete benchmark comparison across standard evaluation metrics."
        )

        st.dataframe(
            metrics_table.style.highlight_max(
                subset=['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
                color='#bbf7d0'
            ),
            use_container_width=True
        )

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("##### 📊 Comparative Performance Across Metrics")
            if os.path.exists("visualizations/model_comparison.png"):
                st.image("visualizations/model_comparison.png", use_container_width=True)
            else:
                st.info("Comparison chart saved during training.")

        with col_img2:
            st.markdown("##### 🧮 Confusion Matrix Heatmaps")
            if os.path.exists("visualizations/confusion_matrices.png"):
                st.image("visualizations/confusion_matrices.png", use_container_width=True)
            else:
                st.info("Confusion matrices saved during training.")

        st.markdown("---")
        st.markdown("#### 💡 Model Analysis & Discussion")
        st.markdown("""
        - **Multinomial Naive Bayes**: Highly efficient probabilistic baseline. Operates under the conditionally independent feature assumption $P(w_1, w_2, \\dots | C) = \\prod P(w_i | C)$. Extremely fast training and strong performance on sparse text features.
        - **Logistic Regression**: Linear multiclass model using the Softmax loss function with L2 regularization. Highly robust on high-dimensional text vectors and delivers high precision and recall.
        - **Support Vector Machine (Linear SVM)**: Searches for the maximum-margin hyperplane separating classes in the high-dimensional TF-IDF space. Typically excels in text categorization where feature spaces are sparse and linearly separable.
        - **Random Forest**: Ensemble of randomized decision trees. While effective on structured tabular data, it is computationally heavier on sparse high-dimensional n-gram spaces.
        """)

    # -------------------------------------------------------------------------
    # TAB 3: Feature & Keyword Analysis
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("TF-IDF Feature Representation & Top Category Keywords")
        st.markdown(
            "TF-IDF (Term Frequency–Inverse Document Frequency) measures how unique and informative words are for each category. "
            "Words with high TF-IDF appear frequently in specific documents but rarely across unrelated documents."
        )

        if os.path.exists("visualizations/top_keywords.png"):
            st.image("visualizations/top_keywords.png", caption="Top distinguishing TF-IDF terms per class", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📐 Mathematical Formulation of TF-IDF")
        st.latex(r"\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t', d}}")
        st.latex(r"\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")
        st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)")
        st.markdown(
            r"Our pipeline also applies **sublinear term frequency scaling** ($1 + \log(\text{TF})$) to dampen the influence of repetitious words, "
            r"along with **$L_2$ document normalization** so document length differences do not distort classification boundaries."
        )

        st.markdown("#### 🏆 Top Keywords Extracted per Class")
        top_terms = metadata.get('top_features_per_class', {})
        cols = st.columns(len(top_terms))
        for idx, (cat_name, words) in enumerate(top_terms.items()):
            with cols[idx % len(cols)]:
                disp = cat_display.get(cat_name, cat_name)
                st.markdown(f"**{disp}**")
                word_df = pd.DataFrame(words[:8], columns=["Keyword", "Mean TF-IDF"])
                st.dataframe(word_df, hide_index=True, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Dataset Insights
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("20 Newsgroups Dataset Overview")
        stats = metadata['dataset_statistics']

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Clean Documents", stats['total_documents'])
        col_s2.metric("Number of Classes", stats['num_classes'])
        col_s3.metric("Avg Document Length (words)", f"{stats['word_count_mean']:.1f}")
        col_s4.metric("Median Document Length", f"{stats['word_count_median']:.0f}")

        st.write("")
        if os.path.exists("visualizations/class_distribution.png"):
            st.image("visualizations/class_distribution.png", caption="Class distribution across dataset", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🛡️ Prevention of Data Leakage")
        st.info(
            "**Data Leakage Prevention Guarantee**: Metadata headers, signatures, quotes, and footers were removed "
            "prior to model training. The TF-IDF vectorizer was **strictly fit on the training split only** (`X_train`), "
            "preventing test distribution information from leaking into the feature vocabulary."
        )

    # -------------------------------------------------------------------------
    # TAB 5: Project Methodology & Report
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Project Methodology & College Report")
        st.markdown("""
        ### Project Summary: Classifying Text Documents Using Machine Learning

        #### 1. Abstract
        Automated text classification is a fundamental natural language processing (NLP) task with extensive applications
        in topic discovery, sentiment analysis, document archiving, and spam detection. This project demonstrates an end-to-end
        machine learning system capable of categorizing text documents into semantic domains with high accuracy.

        #### 2. Pipeline Architecture
        ```text
        [ Raw Text Document ]
                ↓
        [ Text Preprocessing ]
          - Lowercasing
          - URL & Email Removal
          - Punctuation & Non-ASCII Stripping
          - Regex Tokenization
          - NLTK Stop Word Filtering
          - WordNet Lemmatization
                ↓
        [ TF-IDF Feature Extraction ]
          - N-gram Range (1, 2)
          - Sublinear TF Scaling
          - L2 Vector Normalization
                ↓
        [ Model Training & Benchmarking ]
          - Multinomial Naive Bayes
          - Logistic Regression (Softmax)
          - Linear Support Vector Machine
          - Random Forest Classifier
                ↓
        [ Rigorous Evaluation ]
          - Accuracy, Precision, Recall, F1-Score
          - Confusion Matrix Diagnostics
                ↓
        [ Model Persistence & Streamlit UI ]
        ```

        #### 3. Limitations and Future Work
        - **Contextual Semantics**: Bag-of-words and TF-IDF ignore word order beyond defined n-grams. Future iterations could explore transformer models (e.g. BERT, RoBERTa) for deep contextual embeddings.
        - **Out-of-Vocabulary Terms**: Words absent from the training vocabulary receive zero weight; sub-word tokenization (Byte-Pair Encoding) can mitigate this.
        - **Domain Generalization**: Performance on domain-specific corpora (legal, clinical) can be enhanced through custom domain dictionaries.
        """)


if __name__ == "__main__":
    main()
