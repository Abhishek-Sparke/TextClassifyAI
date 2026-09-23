"""
High-Performance Local Inference API Server
Classifying Text Documents Using Machine Learning (4 Classes)

Provides REST API endpoints for document classification, model comparison,
out-of-domain / unknown detection, and multi-topic ambiguity analysis.
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

from src.inference_engine import (
    classify_document,
    InferenceConfig,
    DEFAULT_INFERENCE_CONFIG,
    TARGET_4_CLASSES,
    CATEGORY_DISPLAY_NAMES_4,
    CATEGORY_ICONS_4
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

vectorizer = None
models_dict = {}
best_model = None
metadata = {}
categories = TARGET_4_CLASSES
inference_config = DEFAULT_INFERENCE_CONFIG


def load_artifacts():
    global vectorizer, models_dict, best_model, metadata, categories, inference_config

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
            metadata = json.load(f)
            categories = metadata.get("categories", TARGET_4_CLASSES)
            thresh = metadata.get("confidence_thresholds", {})
            if thresh:
                inference_config = InferenceConfig(
                    confidence_threshold=thresh.get("confidence_threshold", 0.58),
                    min_active_tfidf=thresh.get("min_active_tfidf", 0.05),
                    ambiguity_margin=thresh.get("ambiguity_margin", 0.32),
                    topic_detection_threshold=thresh.get("topic_detection_threshold", 0.15)
                )


load_artifacts()


# -----------------------------------------------------------------------------
# Route Handlers
# -----------------------------------------------------------------------------
async def health_check(request):
    """
    GET /health and GET /api/health
    """
    return JSONResponse({
        "status": "healthy",
        "service": "Text Document Classifier API (4 Classes)",
        "classes": categories,
        "class_display_names": CATEGORY_DISPLAY_NAMES_4,
        "models_available": list(models_dict.keys()) if models_dict else ["Multinomial Naive Bayes"],
        "best_model": metadata.get("best_model_name", "Multinomial Naive Bayes"),
        "confidence_threshold": inference_config.confidence_threshold,
        "ambiguity_margin": inference_config.ambiguity_margin
    })


async def get_models(request):
    """
    GET /models and GET /api/models
    """
    return JSONResponse({
        "best_model": metadata.get("best_model_name", "Multinomial Naive Bayes"),
        "metrics_table": metadata.get("metrics_table", []),
        "selection_metric": metadata.get("selection_metric", "Weighted F1-score"),
        "categories": categories,
        "models": list(models_dict.keys()) if models_dict else []
    })


async def predict_text(request):
    """
    POST /predict and POST /api/classify
    Accepts: {"text": "...", "model": "..."}
    Returns:
    {
        "text": "...",
        "prediction": "...",
        "confidence": 0.0,
        "probabilities": {},
        "status": "normal | ambiguous | unknown",
        "detected_topics": [],
        "top_keywords": []
    }
    """
    # 1. Parse JSON body
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body in request"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

    # 2. Validate 'text' property
    if "text" not in data:
        return JSONResponse({"error": "Missing required field: 'text'"}, status_code=400)

    raw_text = data.get("text")
    if not isinstance(raw_text, str):
        return JSONResponse({"error": "Field 'text' must be a string"}, status_code=400)

    raw_text = raw_text.strip()
    if not raw_text:
        return JSONResponse({"error": "Field 'text' cannot be empty or whitespace only"}, status_code=400)

    # 3. Handle extremely long inputs safely (> 100,000 characters)
    max_len = 100000
    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    # 4. Model selection
    requested_model = data.get("model", "").strip()
    model_mapping = {
        "svm": "Support Vector Machine",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
        "random_forest": "Random Forest",
        "linear svc": "Support Vector Machine"
    }
    model_key = model_mapping.get(requested_model.lower(), requested_model)
    model = models_dict.get(model_key, best_model)

    start_time = time.perf_counter()

    # 5. Run Centralized Inference
    res = classify_document(
        raw_text=raw_text,
        model=model,
        vectorizer=vectorizer,
        categories=categories,
        config=inference_config
    )

    elapsed = round(time.perf_counter() - start_time, 4)
    model_used_name = model_key if model_key in models_dict else metadata.get("best_model_name", "Multinomial Naive Bayes")

    # Icon resolution
    if res["status"] == "unknown":
        icon = CATEGORY_ICONS_4["unknown"]
    elif res["status"] == "ambiguous":
        icon = CATEGORY_ICONS_4["ambiguous"]
    else:
        icon = CATEGORY_ICONS_4.get(res["raw_prediction"], "📄")

    category_disp = CATEGORY_DISPLAY_NAMES_4.get(res["prediction"], res["prediction"])

    # Probability distribution list for UI consumers
    prob_distribution = [
        {
            "categoryId": cat,
            "name": CATEGORY_DISPLAY_NAMES_4.get(cat, cat),
            "icon": CATEGORY_ICONS_4.get(cat, "📄"),
            "probability": round(p * 100, 2)
        }
        for cat, p in res["probabilities"].items()
    ]
    prob_distribution.sort(key=lambda x: x["probability"], reverse=True)

    return JSONResponse({
        "text": raw_text[:300] + ("..." if len(raw_text) > 300 else ""),
        "prediction": res["prediction"],
        "category": category_disp,
        "categoryId": res["raw_prediction"],
        "icon": icon,
        "confidence": round(res["confidence"], 4),
        "probabilities": res["probabilities"],
        "probability_distribution": prob_distribution,
        "status": res["status"],
        "detected_topics": res["detected_topics"],
        "top_keywords": res["top_keywords"],
        "topKeywords": res["top_keywords"],
        "is_out_of_domain": res["is_out_of_domain"],
        "is_ambiguous": res["is_ambiguous"],
        "reason": res["reason"],
        "processingTime": max(elapsed, 0.001),
        "modelUsed": model_used_name
    })


async def predict_batch(request):
    """
    POST /predict-batch and POST /api/classify-batch
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    docs = data.get("documents", [])
    if not isinstance(docs, list) or len(docs) == 0:
        return JSONResponse({"error": "'documents' must be a non-empty list of text objects"}, status_code=400)

    start_time = time.perf_counter()
    results = []

    for idx, item in enumerate(docs):
        if isinstance(item, str):
            row_text = item
            row_id = idx + 1
        elif isinstance(item, dict):
            row_text = item.get("text") or item.get("document") or item.get("content") or item.get("message") or item.get("sentence") or ""
            row_id = item.get("id", idx + 1)
        else:
            continue

        if not str(row_text).strip():
            results.append({
                "id": row_id,
                "text": "",
                "predicted_category": "Unknown / Out-of-Domain",
                "confidence": 0.0,
                "status": "unknown"
            })
            continue

        res = classify_document(str(row_text), best_model, vectorizer, categories, inference_config)
        results.append({
            "id": row_id,
            "text": str(row_text)[:120],
            "predicted_category": res["prediction"],
            "confidence": round(res["confidence"] * 100, 2),
            "status": res["status"],
            "detected_topics": res["detected_topics"]
        })

    elapsed = round(time.perf_counter() - start_time, 3)

    return JSONResponse({
        "results": results,
        "total": len(results),
        "successful_count": sum(1 for r in results if r["status"] == "normal"),
        "ambiguous_count": sum(1 for r in results if r["status"] == "ambiguous"),
        "unknown_count": sum(1 for r in results if r["status"] == "unknown"),
        "processingTime": elapsed
    })


async def upload_document(request):
    """
    POST /api/upload
    """
    try:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            return JSONResponse({"error": "No file uploaded in form data"}, status_code=400)

        contents = await uploaded_file.read()
        filename = uploaded_file.filename or "uploaded_document.txt"

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
# Routes Setup
# -----------------------------------------------------------------------------
routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/health", health_check, methods=["GET"]),
    Route("/models", get_models, methods=["GET"]),
    Route("/api/models", get_models, methods=["GET"]),
    Route("/predict", predict_text, methods=["POST"]),
    Route("/api/classify", predict_text, methods=["POST"]),
    Route("/predict-batch", predict_batch, methods=["POST"]),
    Route("/api/classify-batch", predict_batch, methods=["POST"]),
    Route("/api/upload", upload_document, methods=["POST"])
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
    print(" Supported routes: /health, /models, /predict, /predict-batch")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
