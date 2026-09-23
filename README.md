# TextClassify AI: Robust 4-Class NLP Text Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![FastAPI / Starlette](https://img.shields.io/badge/API-Starlette%20%7C%20FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![React / Vite](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

An end-to-end Machine Learning text classification system engineered to be robust on real-world inputs. Rather than assuming every input belongs perfectly to one of the four categories, the engine detects **Unknown / Out-of-Domain** text and **Ambiguous / Multi-Topic** text using calibrated model confidence and configurable decision thresholds.

---

## Complete Pipeline Architecture

```
Raw Text
   │
   ▼
Preprocessing (Lowercase, URL/Email removal, Noise cleaning, Lemmatization, 3D token preservation)
   │
   ▼
TF-IDF Vectorization (Unigrams + Bigrams, Sublinear TF, L2 Norm, Strict train-only fitting)
   │
   ▼
ML Model (Multinomial Naive Bayes, Logistic Regression, Calibrated Linear SVM, Random Forest)
   │
   ▼
Probability / Confidence (Calibrated probability distribution across all 4 target classes)
   │
   ▼
Decision Engine (Normal vs Ambiguous vs Unknown)
   ├── Unknown / Out-of-Domain: When max probability < threshold OR zero vocabulary overlap
   ├── Ambiguous / Multi-topic: When competing topics have strong signals and narrow margin
   └── Normal Prediction: Dominant in-domain class clearly identified
   │
   ▼
Delivery Interfaces (REST API / Streamlit UI / React Dashboard / CSV Batch Engine)
```

---

## Target Classes

The classifier focuses on four core categories:
1. `comp.graphics` — Computer Graphics (3D rendering, GPU shaders, polygon meshes, ray tracing)
2. `rec.sport.baseball` — Baseball (pitchers, home runs, innings, strikeouts, championship games)
3. `sci.space` — Space Science (NASA launches, orbiters, planetary astrophysics, Hubble telescope)
4. `talk.politics.misc` — Politics (legislation, federal government, senate debates, civil rights)

---

## Realistic Classification Strategy

Real-world text rarely fits cleanly into pre-determined buckets. This system defines three distinct operational states:

| Status | Prediction Output | Condition |
| :--- | :--- | :--- |
| **Normal** | Target Class (`sci.space`, etc.) | A single class clearly dominates ($P_1 \ge \text{threshold}$) and $P_1 - P_2 > \text{margin}$. |
| **Ambiguous** | `Ambiguous / Multi-topic` | Top-2 classes have close probabilities ($P_1 - P_2 \le 0.32$), both have substantial mass ($P_1 + P_2 \ge 0.68$), and multiple topics are detected. |
| **Unknown** | `Unknown / Out-of-Domain` | Model confidence is low ($P_1 < 0.60$) or the input has zero domain vocabulary overlap. |

### Example: Benchmark Multi-Topic Test Case
> **Input:** *"The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."*
- **Status:** `Ambiguous / Multi-topic`
- **Confidence:** `52.42%`
- **Detected Topics:** `Baseball` (Sports), `Politics`, `Space Science`
- **Probabilities:** Baseball: 52.42%, Politics: 21.52%, Space: 15.80%, Graphics: 10.27%
- **Top Keywords:** `['chief', 'minist', 'footbal', 'royal', 'mar', 'play']`

---

## Dataset & Augmentation

- **Source Corpus:** 20 Newsgroups filtered strictly to the 4 target classes (~2,729 clean documents).
- **Short-Text Augmentations:** Curated short phrases ("NASA launch", "baseball game", "new government law", "3D rendering software") are augmented into the training set to prevent length-underflow.
- **Data Hygiene:** Headers, footers, sender emails, and quote blocks are removed to prevent artificial leakage.
- **Stratified Split:** 80% Training (2,204 samples) and 20% Testing (552 samples).

---

## Text Preprocessing

- **Lowercase Normalization:** Eliminates casing discrepancies.
- **URL & Email Removal:** Strips `https?://` links, `www.*`, and email addresses.
- **Noise & Digit Cleaning:** Strips standalone digits while preserving alphanumeric terms like `3d`, `4k`, `cad`, `gpu`, and `nasa`.
- **Stopword Filtering:** Removes standard English stop words and conversational filler words while preserving key domain terms.
- **Lemmatization:** WordNet lemmatizer reduces inflected tokens to canonical roots (`innings` $\rightarrow$ `inning`, `shading` $\rightarrow$ `shade`).
- **Parity:** Exactly identical preprocessing pipeline shared across training, API inference, Streamlit, and batch processing.

---

## TF-IDF Feature Extraction

- **N-Gram Range:** `(1, 2)` captures both single words and compound phrases (e.g. `space`, `spacecraft`, `home run`, `ray tracing`).
- **Vocabulary Size:** `max_features = 5,000`.
- **Sublinear TF:** Uses $1 + \log(\text{TF})$ scaling to dampen repetitive words.
- **Normalization:** L2 unit norm prevents long documents from dominating over shorter texts.
- **Strict Ordering:** Fitted **only** on the training set to prevent data leakage. The test split is exclusively transformed.

---

## Machine Learning Models & Evaluation

Four supervised algorithms were trained and evaluated on identical stratified test partitions:

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Training Time | Latency (ms/doc) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **86.59%** | **0.8665** | **0.8659** | **0.8657** | **0.005s** | **0.002 ms** | 🏆 **Best Model** |
| **Logistic Regression** | 83.88% | 0.8432 | 0.8388 | 0.8391 | 0.222s | 0.001 ms | Production Ready |
| **Support Vector Machine (Linear SVM)** | 83.15% | 0.8373 | 0.8315 | 0.8326 | 0.150s | 0.010 ms | Calibrated CV |
| **Random Forest** | 77.54% | 0.7932 | 0.7754 | 0.7789 | 0.386s | 0.105 ms | Ensemble Baseline |

*Selection Metric: Primary selection based on Weighted F1-Score, secondary on Accuracy. Multinomial Naive Bayes achieved the highest F1-Score (86.57%) and lowest prediction latency.*

---

## REST API (FastAPI / Starlette)

### Endpoints:
- `GET /health`: Health status, active classes, models available, and threshold settings.
- `GET /models`: Complete benchmark metrics table across all 4 classifiers.
- `POST /predict`: Classify a single document with full confidence analysis.
- `POST /predict-batch`: Row-by-row batch classification.

### Example Request (`POST /predict`):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."}'
```

### Example Response:
```json
{
  "text": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.",
  "prediction": "Ambiguous / Multi-topic",
  "category": "Ambiguous / Multi-topic",
  "confidence": 0.5242,
  "probabilities": {
    "comp.graphics": 0.1027,
    "rec.sport.baseball": 0.5242,
    "sci.space": 0.158,
    "talk.politics.misc": 0.2152
  },
  "status": "ambiguous",
  "detected_topics": [
    "Baseball",
    "Politics",
    "Space Science"
  ],
  "top_keywords": [
    {"term": "chief", "weight": 0.521},
    {"term": "minist", "weight": 0.489},
    {"term": "footbal", "weight": 0.412},
    {"term": "mar", "weight": 0.395}
  ],
  "is_out_of_domain": false,
  "is_ambiguous": true,
  "reason": "Multiple competing topics detected (Baseball, Politics, Space Science): top-2 probability gap 30.90%",
  "processingTime": 0.003,
  "modelUsed": "Multinomial Naive Bayes"
}
```

### Error Handling:
- Empty text $\rightarrow$ `400 Bad Request` (`{"error": "Field 'text' cannot be empty or whitespace only"}`)
- Missing `text` field $\rightarrow$ `400 Bad Request` (`{"error": "Missing required field: 'text'"}`)
- Invalid JSON payload $\rightarrow$ `400 Bad Request` (`{"error": "Invalid JSON body in request"}`)
- Extremely long inputs (> 100,000 characters) $\rightarrow$ Safely truncated to 100,000 characters.

---

## Streamlit Web Application

Run locally with:
```bash
streamlit run app.py
```

### Features:
1. **Interactive Workspace:**
   - Textarea with live character and word counters.
   - Quick **Test Examples** selector (Pure Space, Baseball, Politics, Graphics, Short queries, Out-of-Domain, Ambiguous benchmark).
2. **AI Prediction Banner:**
   - Dynamic status badges: `✓ Normal Single-Topic` (green), `⚠️ Ambiguous / Multi-topic` (amber), or `❓ Unknown / Out-of-Domain` (slate).
   - Calibrated confidence percentage and execution latency.
3. **Detected Topics & Explanations:**
   - Displays all detected topics exceeding threshold.
   - Callout: *"Low-confidence predictions are marked as Unknown rather than being forced into a category."*
4. **Class Probability Distribution:**
   - Real-time horizontal bar visualization across the 4 classes.
5. **Important TF-IDF Features:**
   - Informative keyword tags and their numerical weights.

---

## CSV Batch Classification

In both the Streamlit UI and REST API:
1. Upload any `.csv` file.
2. The engine autodetects the text column (`text`, `document`, `content`, `message`, `sentence`, `body`, `doc`).
3. Every row is processed independently.
4. Summary counters: Total Rows, Normal Count, Ambiguous Count, Unknown Count.
5. Interactive results table and class distribution chart.
6. **Download Button:** Export classified results as `classified_results.csv`.

---

## React / Vite Dashboard

The React frontend (`frontend/`) provides a modern dark-first SaaS interface:
```bash
cd frontend
npm install
npm run dev
```

Connects to the local FastAPI backend (`http://localhost:8000`) or seamlessly activates zero-latency client-side ML inference when deployed statically on Vercel or GitHub Pages.

---

## How to Run Locally

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train and Evaluate Models
```bash
python train.py
```
*Trains all 4 classifiers, logs the comparison table, generates visualizations, and saves artifacts to `models/`.*

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```
*Validates preprocessing, model loading, 4-class inference, unknown detection, ambiguity detection, API endpoints, and CSV batch processing.*

### 4. Run Dedicated Stress Tests (45+ Scenarios)
```bash
python tests/stress_tests.py
```
*Tests pure topics, short sentences, mixed queries, out-of-domain sentences, and the multi-topic benchmark.*

### 5. Launch REST API Server
```bash
python server.py
# API runs on http://localhost:8000
```

### 6. Launch Streamlit UI
```bash
streamlit run app.py
# Streamlit runs on http://localhost:8501
```

### 7. Run with Docker
```bash
docker build -t textclassify-ai .
docker run -p 8501:8501 -p 8000:8000 textclassify-ai
```

---

## Project Structure

```
mlproj/
├── app.py                     # Streamlit Interactive Web Application
├── server.py                  # FastAPI / Starlette REST API Server
├── train.py                   # End-to-end Training & Evaluation Script
├── requirements.txt           # Python Dependencies
├── Dockerfile                 # Container Deployment Configuration
├── README.md                  # Comprehensive Documentation
├── models/
│   ├── best_model.joblib      # Production Best Model (Multinomial Naive Bayes)
│   ├── tfidf_vectorizer.joblib# Fitted TF-IDF Vectorizer
│   ├── all_models.joblib      # Serialized Dictionary of all 4 Fitted Models
│   └── model_metadata.json    # Complete Model Versioning & Threshold Metadata
├── src/
│   ├── __init__.py
│   ├── dataset.py             # 4-Class Dataset Loader & Short-Text Augmentations
│   ├── preprocessing.py       # Regex Cleaning, Tokenization, Lemmatization
│   ├── features.py            # TF-IDF Feature Extraction (Train-fit only)
│   ├── models.py              # 4 Classifiers & Probability Calibration
│   ├── evaluate.py            # Metrics, Confusion Matrices, Latency Benchmarking
│   └── inference_engine.py    # Centralized Decision & Confidence Threshold Engine
├── tests/
│   ├── test_pipeline.py       # Core Automated Test Suite (8 Tests)
│   └── stress_tests.py        # Dedicated 45+ Stress Test Scenarios
├── visualizations/            # Confusion Matrices & Comparison Charts
└── frontend/                  # React / Vite Modern Dashboard
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   ├── data/
    │   └── services/api.js
    └── package.json
```

---

## Key Takeaways

1. **Realistic Classification:** No text is blindly forced into an incorrect class; low-confidence inputs are accurately marked as **Unknown**.
2. **Ambiguity Awareness:** Inputs with competing signals (e.g., politics + space + sports) are recognized as **Ambiguous** with all detected topics reported.
3. **Calibrated Probabilities:** True probability distributions enable transparent explainability.
4. **Leakage-Free Feature Engineering:** TF-IDF is fitted strictly on training data.
5. **Production Viability:** Sub-millisecond latency (0.002 ms/doc) with multi-interface deployment (API, Streamlit, React, Docker).
