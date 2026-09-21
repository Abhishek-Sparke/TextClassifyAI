# Classifying Text Documents Using Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhishek-sparke-mlproj-app-cmlulp.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.64+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Web Application:** **[https://abhishek-sparke-mlproj-app-cmlulp.streamlit.app/](https://abhishek-sparke-mlproj-app-cmlulp.streamlit.app/)**  
> **College-Level Machine Learning Term Project**  
> An end-to-end Natural Language Processing (NLP) and Machine Learning system for automated multi-class text document categorization, rigorous model benchmarking, and real-time interactive inference.

---

## Table of Contents
1. [Project Overview & Abstract](#project-overview--abstract)
2. [Workflow Architecture](#workflow-architecture)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing & Leakage Prevention](#data-preprocessing--leakage-prevention)
5. [Feature Extraction: TF-IDF](#feature-extraction-tf-idf)
6. [Machine Learning Classifiers](#machine-learning-classifiers)
7. [Experimental Results & Evaluation](#experimental-results--evaluation)
8. [Streamlit Web Application](#streamlit-web-application)
9. [Project Directory Structure](#project-directory-structure)
10. [Installation & Setup](#installation--setup)
11. [How to Run the Project](#how-to-run-the-project)
12. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview & Abstract

With the exponential growth of unstructured digital text across news outlets, academic repositories, and enterprise knowledge bases, manual document classification is labor-intensive and unscalable.

This project delivers a complete, modular Machine Learning solution that automatically classifies raw text documents into predefined categories based on semantic content. The system processes raw text through a multi-stage NLP cleaning pipeline, extracts numerical vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)** with unigrams and bigrams, trains and evaluates **four supervised classification algorithms** (Multinomial Naive Bayes, Logistic Regression, Support Vector Machine, and Random Forest), and deploys the best-performing model into an interactive **Streamlit web application**.

---

## Workflow Architecture

```text
                  +-----------------------------------+
                  |      20 Newsgroups Dataset        |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         Data Preprocessing        |
                  |  - Lowercase Normalization        |
                  |  - Strip Headers, URLs, Emails    |
                  |  - Punctuation & Digit Removal    |
                  |  - Tokenization & Stopwords       |
                  |  - WordNet Lemmatization          |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Stratified Train/Test Split    |
                  |         (80% Train, 20% Test)     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     TF-IDF Feature Extraction     |
                  |  - Fit ONLY on Training Data      |
                  |  - Transform Train & Test Sets    |
                  |  - Sublinear TF & L2 Norm         |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
     +--------------+        +--------------+        +--------------+
     |  Multinomial |        |   Logistic   |        | Linear SVM / |
     |  Naive Bayes |        |  Regression  |        |  Calibrated  |
     +--------------+        +--------------+        +--------------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                             +--------------+
                             |    Random    |
                             |    Forest    |
                             +--------------+
                                    |
                                    v
                  +-----------------------------------+
                  |          Model Evaluation         |
                  |  - Accuracy, Precision, Recall, F1|
                  |  - Confusion Matrices Heatmaps    |
                  |  - Top Distinguishing Keywords    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Best Model Selection & Save    |
                  |  - Export via joblib to /models   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       Streamlit Prediction UI     |
                  |  - Live Document Classification   |
                  |  - Confidence & Probability Chart |
                  |  - Model Comparison Dashboard     |
                  +-----------------------------------+
```

---

## Dataset Description

The project uses the standard benchmark **20 Newsgroups Dataset** (available via `sklearn.datasets.fetch_20newsgroups`), comprising approximately 20,000 newsgroup postings partitioned across diverse subject areas.

For clear interpretability, swift convergence, and distinct thematic separation, the default configuration focuses on 4 representative semantic domains:
- **`comp.graphics`**: Computer Graphics, 3D Rendering, Algorithms, GPUs
- **`rec.sport.baseball`**: Baseball, Games, Teams, Scores, Pitchers
- **`sci.space`**: Astronomy, NASA, Spacecraft, Orbit, Planetary Missions
- **`talk.politics.misc`**: Governance, Policy, International Relations, Law

*(The pipeline can be extended to all 20 categories by adjusting the configuration in `src/dataset.py`).*

---

## Data Preprocessing & Leakage Prevention

### 1. Preprocessing Steps
Raw documents contain unstructured noise, internet formatting, and non-informative tokens. The cleaning pipeline implements:
1. **Handling Missing Values**: Null, non-string, or blank documents are filtered.
2. **Case Normalization**: All characters converted to lowercase.
3. **Artifact Removal**: URLs, emails, and newsgroup headers (`from:`, `subject:`, `lines:`) are removed via regular expressions.
4. **Punctuation & Noise Stripping**: All punctuation marks and non-ASCII glyphs are removed.
5. **Tokenization**: Regex-based tokenization extracting alphabetic tokens with length $\ge 2$.
6. **Stop Words Filtering**: Eliminates ubiquitous non-discriminative English words (e.g., "the", "and", "in") using NLTK English stopwords.
7. **Lemmatization**: Morphs inflected word forms to their canonical dictionary base form using NLTK `WordNetLemmatizer` (with fallback to `PorterStemmer`).

### 2. Prevention of Data Leakage
> [!IMPORTANT]
> **Strict Featurization Ordering**: To prevent data leakage, the dataset is first partitioned using a stratified 80/20 train/test split. The **TF-IDF vectorizer is fit strictly on the training set only**. The test set is solely transformed using the learned training vocabulary and IDF weights. Furthermore, newsgroup headers, footers, and quote tags are stripped (`remove=('headers', 'footers', 'quotes')`) so models do not exploit metadata clues.

---

## Feature Extraction: TF-IDF

Raw text cannot be fed directly into machine learning algorithms; it must be converted into numerical feature vectors. We utilize **Term Frequency–Inverse Document Frequency (TF-IDF)**:

### 1. Mathematical Formulation

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2. Enhancements in this Implementation
- **N-gram Range $(1, 2)$**: Captures both single words ("space", "graphics") and informative collocations ("space station", "ray tracing").
- **Sublinear Term Frequency**: Replaces raw $\text{TF}$ with $1 + \log(\text{TF})$ to dampen the disproportionate impact of words that repeat many times in a single long document.
- **$L_2$ Normalization**: Scales document vectors to unit norm ($\|v\|_2 = 1$), neutralizing the effect of varying document lengths.
- **Minimum Document Frequency (`min_df=2`)**: Filters rare typos and singletons.
- **`max_features=5000`**: Caps feature space to the most informative n-grams, reducing memory overhead.

---

## Machine Learning Classifiers

Four distinct algorithms representing different learning paradigms were implemented and benchmarked:

### 1. Multinomial Naive Bayes (`MultinomialNB`)
- **Type**: Generative Probabilistic Classifier.
- **Formulation**: Applies Bayes' Theorem under the feature independence assumption:
  $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$$
- **Smoothing**: Laplace smoothing ($\alpha = 0.1$) handles unseen terms in training.
- **Characteristics**: Fast training, minimal memory footprint, excellent baseline for sparse high-dimensional text.

### 2. Logistic Regression (`LogisticRegression`)
- **Type**: Linear Discriminative Model with Softmax / Multinomial Cross-Entropy Loss.
- **Formulation**: Estimates category probabilities using the multinomial sigmoid:
  $$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
- **Regularization**: $L_2$ penalty ($C=1.0$) prevents overfitting across thousands of features.

### 3. Support Vector Machine (`LinearSVC` + `CalibratedClassifierCV`)
- **Type**: Maximum-Margin Hyperplane Separator.
- **Formulation**: Solves the convex optimization problem:
  $$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Probability Calibration**: Uses 3-fold cross-validated Platt scaling to produce posterior class probabilities.
- **Characteristics**: Superior performance in high-dimensional text spaces where classes are linearly separable.

### 4. Random Forest (`RandomForestClassifier`)
- **Type**: Non-linear Bagging Ensemble.
- **Hyperparameters**: 150 randomized decision trees, `max_depth=30`.
- **Characteristics**: Resilient to outliers, non-parametric, captures complex feature interactions.

---

## Experimental Results & Evaluation

All models were evaluated on the held-out stratified test set (20% of data) using:
- **Accuracy**: Overall proportion of correctly categorized documents.
- **Precision (Weighted & Macro)**: Proportion of positive identifications that were correct.
- **Recall (Weighted & Macro)**: Proportion of actual positives correctly identified.
- **F1-Score (Weighted & Macro)**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Error distribution across actual vs. predicted categories.

### Performance Summary Table

| Machine Learning Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Score (Macro) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (Linear SVM)** | **89.74%** | **0.8975** | **0.8974** | **0.8974** | **0.8963** | 0.114s |
| **Logistic Regression** | **89.32%** | **0.8935** | **0.8932** | **0.8931** | **0.8925** | 0.144s |
| **Multinomial Naive Bayes** | **89.04%** | **0.8906** | **0.8904** | **0.8902** | **0.8903** | **0.004s** |
| **Random Forest** | **83.50%** | **0.8374** | **0.8350** | **0.8337** | **0.8323** | 0.369s |

*(Exact values are generated dynamically when running `train.py`).*

### Key Analytical Takeaways
1. **SVM & Logistic Regression dominate text classification**: Linear boundaries perform exceptionally well in high-dimensional sparse TF-IDF spaces.
2. **Naive Bayes provides the highest speed-to-performance ratio**: With an inference time under 1 millisecond, it is ideal for latency-sensitive deployments.
3. **Random Forest suffers slightly on sparse text**: High-dimensional sparse feature spaces are challenging for axis-aligned decision trees compared to margin-based linear hyperplanes.

---

## Streamlit Web Application

🌐 **Live Cloud Deployment**: **[https://abhishek-sparke-mlproj-app-cmlulp.streamlit.app/](https://abhishek-sparke-mlproj-app-cmlulp.streamlit.app/)**

The interactive web application (`app.py`) provides:
- **🔮 Live Document Classifier**:
  - Paste any text document or select one of 4 curated presets.
  - Predict the document class instantly.
  - View confidence percentage and a probability breakdown bar chart across all categories.
  - Inspect top TF-IDF keywords contributing to the decision.
- **📊 Model Comparison Tab**:
  - Side-by-side metric table with best scores highlighted.
  - Multi-metric comparative bar chart.
  - 2x2 confusion matrix heatmaps for error diagnosis.
- **🔍 Feature & Keyword Analysis**:
  - Top distinguishing keywords per class.
  - Mathematical breakdown of the TF-IDF formula.
- **📁 Dataset Insights**:
  - Document counts, average word counts, and class distribution charts.
- **📖 Methodology & Report**:
  - Complete project architecture and theoretical reference.

---

## Project Directory Structure

```text
mlproj/
├── app.py                     # Streamlit interactive web application
├── server.py                  # High-performance Python REST API server (Starlette/Uvicorn)
├── train.py                   # End-to-end training and evaluation script
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive project documentation
├── frontend/                  # Modern React + Vite + Tailwind CSS ML Studio
│   ├── src/                   # React components, benchmarks, and inference service
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py       # Cleaning, tokenization, lemmatization
│   ├── dataset.py             # Dataset loading, statistics, train/test split
│   ├── features.py            # TF-IDF vectorization & feature importance
│   ├── models.py              # Model initialization & training wrappers
│   └── evaluate.py            # Evaluation metrics & visualization plotting
├── models/                    # Saved artifacts (created after training)
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── all_models.joblib
│   └── model_metadata.json
└── visualizations/            # Generated charts (created after training)
    ├── class_distribution.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    └── top_keywords.png
```

---

## Installation & Setup

### 1. Clone or Open the Project
Ensure you are in the project root directory:
```bash
cd c:/Users/abhis/OneDrive/Apps/mlproj
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Step 1: Train Models and Generate Artifacts
Run the end-to-end training pipeline:
```bash
python train.py
```
This script will:
1. Fetch and clean the 20 Newsgroups dataset.
2. Preprocess text (normalization, stopwords, lemmatization).
3. Extract TF-IDF features.
4. Train all 4 classifiers.
5. Benchmark metrics and print the comparison table.
6. Generate 4 visualization plots in `visualizations/`.
7. Persist the best model and vectorizer in `models/`.

### Step 2: Launch the Web Applications

You have two interactive interface options:

#### Option A: Modern React + Vite ML Studio (Recommended)
A rich, modern responsive dashboard with dark mode, interactive Recharts benchmarks, 4x4 confusion matrix inspector, and real-time live classification playground:
```bash
# 1. (Optional) Start the Python ML API server for real-time model inference:
python server.py

# 2. In another terminal, start the Vite frontend:
cd frontend
npm run dev
```
Open `http://localhost:3000` in your browser. *(Note: The React app also includes an automated client-side inference engine fallback, so it works completely standalone without the Python server running).*

#### Option B: Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Limitations & Future Work

1. **Sub-word / Out-of-Vocabulary (OOV) Handling**:
   - *Limitation*: Traditional n-gram TF-IDF cannot assign weights to words unseen during training.
   - *Future Work*: Integrate Byte-Pair Encoding (BPE) or character n-grams.
2. **Context and Word Order**:
   - *Limitation*: Bag-of-words ignores word order beyond n-grams $(1, 2)$.
   - *Future Work*: Compare against contextual transformers like DistilBERT or RoBERTa.
3. **Multi-label Classification**:
   - *Limitation*: Currently assumes single-label document categorization.
   - *Future Work*: Extend to multi-label document tagging with binary relevance or classifier chains.

---

## Authors & Acknowledgments
- Developed for Machine Learning and Natural Language Processing coursework.
- Built with Python, Scikit-learn, and Streamlit.
