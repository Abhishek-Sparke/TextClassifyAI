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
    "comp.graphics": "🎨",
    "rec.sport.baseball": "⚾",
    "sci.space": "🚀",
    "talk.politics.misc": "🏛️",
    "rec.autos": "🚗",
    "sci.med": "🩺"
}

def get_category_icon(category_name: str, raw_class: str = "") -> str:
    combined = f"{category_name} {raw_class}".lower()
    if any(k in combined for k in ["business", "trade", "revenue", "profit", "finance", "investor", "market"]):
        return "💼"
    elif any(k in combined for k in ["tech", "computer", "hardware", "software", "sys", "gpu", "compiler"]):
        return "💻"
    elif any(k in combined for k in ["graphic", "rendering", "3d", "art", "design", "comp.graphics"]):
        return "🎨"
    elif any(k in combined for k in ["sport", "baseball", "soccer", "game", "hockey"]):
        return "⚽"
    elif any(k in combined for k in ["politic", "congress", "law", "government"]):
        return "🏛️"
    elif any(k in combined for k in ["entertain", "movie", "film", "cinema", "music"]):
        return "🎬"
    elif any(k in combined for k in ["space", "nasa", "astronomy", "telescope", "orbit", "sci.space"]):
        return "🚀"
    elif any(k in combined for k in ["auto", "car", "engine", "vehicle", "rec.autos"]):
        return "🚗"
    return CATEGORY_ICONS.get(raw_class, "📄")

CATEGORY_NAMES = {
    "comp.graphics": "Computer Graphics",
    "rec.sport.baseball": "Sports",
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
    icon = get_category_icon(category_name, category_id)

    # 5. Probabilities
    probs = get_prediction_probabilities(model, vector)[0]
    prob_distribution = []
    for i, p in enumerate(probs):
        c_id = categories[i] if i < len(categories) else f"cat_{i}"
        c_name = CATEGORY_NAMES.get(c_id, c_id)
        prob_distribution.append({
            "categoryId": c_id,
            "name": c_name,
            "icon": get_category_icon(c_name, c_id),
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


async def classify_batch(request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    docs = data.get("documents", [])
    model_id = data.get("model", "svm")

    if not isinstance(docs, list) or len(docs) == 0:
        return JSONResponse({"error": "documents must be a non-empty list"}, status_code=400)

    start_time = time.time()

    id_mapping = {
        "svm": "Support Vector Machine",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
        "random_forest": "Random Forest"
    }
    model_key = id_mapping.get(model_id, "Support Vector Machine")
    model = models_dict.get(model_key, best_model)

    results = []
    for doc in docs:
        doc_id = doc.get("id") or doc.get("title") or f"doc_{len(results)+1}"
        title = doc.get("title", f"Document {len(results)+1}")
        raw_text = doc.get("text", "").strip()

        if not raw_text:
            continue

        clean_text = preprocess_document(raw_text, apply_lemmatization=True)
        if not clean_text.strip():
            clean_text = raw_text.lower()

        vector = vectorizer.transform([clean_text])
        prediction_idx = model.predict(vector)[0]
        cat_id = categories[prediction_idx] if prediction_idx < len(categories) else str(prediction_idx)
        cat_name = CATEGORY_NAMES.get(cat_id, cat_id)
        icon = get_category_icon(cat_name, cat_id)

        probs = get_prediction_probabilities(model, vector)[0]
        confidence = round(float(np.max(probs)) * 100, 2) if len(probs) > 0 else 99.0

        keywords_raw = get_top_tfidf_terms_for_document(vectorizer, vector, top_n=5)
        top_keywords = [{"term": term, "weight": round(float(w), 3)} for term, w in keywords_raw]

        word_count = len(raw_text.split())

        results.append({
            "id": doc_id,
            "title": title,
            "category": cat_name,
            "categoryId": cat_id,
            "icon": icon,
            "confidence": confidence,
            "topKeywords": top_keywords,
            "wordCount": word_count,
            "snippet": raw_text[:180] + ("..." if len(raw_text) > 180 else "")
        })

    elapsed = round(time.time() - start_time, 3)

    return JSONResponse({
        "results": results,
        "total": len(results),
        "modelUsed": model_key,
        "processingTime": elapsed
    })


async def upload_document(request):
    try:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            return JSONResponse({"error": "No file uploaded in form data"}, status_code=400)

        contents = await uploaded_file.read()
        filename = uploaded_file.filename or "uploaded_document.txt"
        
        # Decode contents
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = contents.decode("latin-1")
            except Exception:
                text = str(contents)

        words = text.split()
        return JSONResponse({
            "filename": filename,
            "size": len(contents),
            "wordCount": len(words),
            "text": text
        })
    except Exception as e:
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)


# -----------------------------------------------------------------------------
# Starlette Application Setup with CORS
# -----------------------------------------------------------------------------
routes = [
    Route("/api/health", health_check, methods=["GET"]),
    Route("/api/classify", classify_text, methods=["POST"]),
    Route("/api/classify-batch", classify_batch, methods=["POST"]),
    Route("/api/upload", upload_document, methods=["POST"]),
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
