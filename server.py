"""
High-Performance Local Inference API Server
Classifying Text Documents Using Machine Learning

Provides REST API endpoints for real-time document classification,
model comparison, and TF-IDF feature explainability using Starlette and Uvicorn.
"""

import os
import json
import time
import joblib
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities

# -----------------------------------------------------------------------------
# Global Asset Pre-loading
# -----------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CATEGORY_ICONS = {
    "comp.graphics": "💻",
    "rec.sport.baseball": "⚾",
    "sci.space": "🚀",
    "talk.politics.misc": "🏛️"
}

CATEGORY_NAMES = {
    "comp.graphics": "Computer Graphics",
    "rec.sport.baseball": "Baseball",
    "sci.space": "Space Science",
    "talk.politics.misc": "Politics"
}

vectorizer = None
models_dict = {}
best_model = None
categories = []


def load_artifacts():
    global vectorizer, models_dict, best_model, categories
    
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if os.path.exists(vec_path):
        vectorizer = joblib.load(vec_path)

    all_models_path = os.path.join(MODELS_DIR, "all_models.joblib")
    if os.path.exists(all_models_path):
        models_dict = joblib.load(all_models_path)

    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            categories = meta.get("categories", [
                "comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.misc"
            ])
    else:
        categories = ["comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.misc"]


load_artifacts()

# -----------------------------------------------------------------------------
# Route Handlers
# -----------------------------------------------------------------------------
async def health_check(request):
    return JSONResponse({
        "status": "healthy",
        "service": "Text Document Classifier API",
        "categories": categories,
        "models_available": list(models_dict.keys()) if models_dict else ["Support Vector Machine"]
    })


async def classify_text(request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    raw_text = data.get("text", "").strip()
    model_id = data.get("model", "svm")

    if not raw_text:
        return JSONResponse({"error": "Text payload cannot be empty"}, status_code=400)

    start_time = time.time()

    # 1. NLP Preprocessing
    clean_text = preprocess_document(raw_text, apply_lemmatization=True)
    if not clean_text.strip():
        clean_text = raw_text.lower()

    # 2. Vectorize
    vector = vectorizer.transform([clean_text])

    # 3. Model selection
    # Map frontend model IDs to model dictionary keys
    id_mapping = {
        "svm": "Support Vector Machine",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
        "random_forest": "Random Forest"
    }
    model_key = id_mapping.get(model_id, "Support Vector Machine")
    model = models_dict.get(model_key, best_model)

    # 4. Predict
    prediction_idx = model.predict(vector)[0]
    category_id = categories[prediction_idx] if prediction_idx < len(categories) else str(prediction_idx)
    category_name = CATEGORY_NAMES.get(category_id, category_id)
    icon = CATEGORY_ICONS.get(category_id, "📄")

    # 5. Probabilities
    probs = get_prediction_probabilities(model, vector)[0]
    prob_distribution = []
    for i, p in enumerate(probs):
        c_id = categories[i] if i < len(categories) else f"cat_{i}"
        prob_distribution.append({
            "categoryId": c_id,
            "name": CATEGORY_NAMES.get(c_id, c_id),
            "icon": CATEGORY_ICONS.get(c_id, "📄"),
            "probability": round(float(p) * 100, 2)
        })
    prob_distribution.sort(key=lambda x: x["probability"], reverse=True)

    confidence = prob_distribution[0]["probability"] if prob_distribution else 99.0

    # 6. Top TF-IDF Keywords for Explainability
    keywords_raw = get_top_tfidf_terms_for_document(vectorizer, vector, top_n=8)
    top_keywords = [{"term": term, "weight": round(float(w), 3)} for term, w in keywords_raw]

    elapsed = round(time.time() - start_time, 3)

    return JSONResponse({
        "category": category_name,
        "categoryId": category_id,
        "icon": icon,
        "confidence": confidence,
        "probabilities": prob_distribution,
        "topKeywords": top_keywords,
        "processingTime": max(elapsed, 0.005),
        "modelUsed": model_key,
        "cleanText": clean_text[:400] + ("..." if len(clean_text) > 400 else "")
    })


# -----------------------------------------------------------------------------
# Starlette Application Setup with CORS
# -----------------------------------------------------------------------------
routes = [
    Route("/api/health", health_check, methods=["GET"]),
    Route("/api/classify", classify_text, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print(" Starting TextClassify AI API Server on http://localhost:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
