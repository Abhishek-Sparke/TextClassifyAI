"""
High-Performance Local Inference API Server
Classifying Text Documents Using Machine Learning (4 Classes)

Provides REST API endpoints for:
- /health & /api/health: System health and active models metadata
- /models & /api/models: Model metrics, cross-validation scores, hyperparameters
- /predict & /api/classify: Real-time single document classification
- /predict/batch & /api/classify-batch: High-throughput JSON batch prediction
- /explain & /api/explain: Explainability and word-level feature attributions
- /api/upload: Document and CSV upload processing
- /openapi.json: OpenAPI 3.0 specification
- /docs: Interactive Swagger UI
- /redoc: Interactive ReDoc documentation
"""

import os
import time
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
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
from src.explainability import explain_prediction
from src.model_validator import validate_and_load_artifacts, ArtifactValidationError

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Global state
vectorizer = None
models_dict = {}
best_model = None
metadata = {}
categories = TARGET_4_CLASSES
inference_config = DEFAULT_INFERENCE_CONFIG


def initialize_artifacts():
    global vectorizer, models_dict, best_model, metadata, categories, inference_config
    try:
        best_model, vectorizer, models_dict, metadata = validate_and_load_artifacts(MODELS_DIR)
        categories = metadata.get("categories", TARGET_4_CLASSES)
        thresh = metadata.get("confidence_thresholds", {})
        if thresh:
            inference_config = InferenceConfig(
                confidence_threshold=thresh.get("confidence_threshold", DEFAULT_INFERENCE_CONFIG.confidence_threshold),
                min_active_tfidf=thresh.get("min_active_tfidf", DEFAULT_INFERENCE_CONFIG.min_active_tfidf),
                ambiguity_margin=thresh.get("ambiguity_margin", DEFAULT_INFERENCE_CONFIG.ambiguity_margin),
                min_ambiguity_sum=thresh.get("min_ambiguity_sum", DEFAULT_INFERENCE_CONFIG.min_ambiguity_sum),
                min_ambiguity_p1=thresh.get("min_ambiguity_p1", DEFAULT_INFERENCE_CONFIG.min_ambiguity_p1),
                topic_detection_threshold=thresh.get("topic_detection_threshold", DEFAULT_INFERENCE_CONFIG.topic_detection_threshold),
                min_ambiguity_keywords=thresh.get("min_ambiguity_keywords", DEFAULT_INFERENCE_CONFIG.min_ambiguity_keywords)
            )
        print(f"[Server] Successfully loaded artifacts. Best Model: {metadata.get('best_model_name')}")
    except ArtifactValidationError as ave:
        print(f"[Server Warning] Artifact validation failed: {ave}")
    except Exception as e:
        print(f"[Server Warning] Error loading model artifacts: {e}")


initialize_artifacts()


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
        "version": metadata.get("model_version", "2.0.0"),
        "dataset_fingerprint": metadata.get("dataset_statistics", {}).get("fingerprint", "unknown"),
        "classes": categories,
        "class_display_names": CATEGORY_DISPLAY_NAMES_4,
        "models_available": list(models_dict.keys()) if models_dict else ["Logistic Regression"],
        "best_model": metadata.get("best_model_name", "Logistic Regression"),
        "confidence_threshold": inference_config.confidence_threshold,
        "ambiguity_margin": inference_config.ambiguity_margin
    })


async def get_models(request):
    """
    GET /models and GET /api/models
    """
    return JSONResponse({
        "best_model": metadata.get("best_model_name", "Logistic Regression"),
        "model_version": metadata.get("model_version", "2.0.0"),
        "metrics_table": metadata.get("metrics_table", []),
        "cross_validation_results": metadata.get("cross_validation_results", []),
        "selection_metric": metadata.get("selection_metric", "Weighted F1-score"),
        "categories": categories,
        "hyperparameters": metadata.get("hyperparameters", {}),
        "models": list(models_dict.keys()) if models_dict else []
    })


async def predict_text(request):
    """
    POST /predict and POST /api/classify
    Accepts: {"text": "...", "model": "..."}
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload in request"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

    if "text" not in data:
        return JSONResponse({"error": "Missing required field: 'text'"}, status_code=400)

    raw_text = data.get("text")
    if not isinstance(raw_text, str):
        return JSONResponse({"error": "Field 'text' must be a string"}, status_code=400)

    raw_text = raw_text.strip()
    if not raw_text:
        return JSONResponse({"error": "Field 'text' cannot be empty or whitespace only"}, status_code=400)

    # Security: Maximum text limit (100,000 characters)
    max_len = 100000
    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    # Model resolution
    requested_model = data.get("model", "").strip()
    model_mapping = {
        "svm": "Linear SVM",
        "linear svm": "Linear SVM",
        "logistic_regression": "Logistic Regression",
        "logistic regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
        "multinomial naive bayes": "Multinomial Naive Bayes",
        "complement naive bayes": "Complement Naive Bayes",
        "complement_nb": "Complement Naive Bayes",
        "sgd": "SGDClassifier",
        "sgdclassifier": "SGDClassifier",
        "ridge": "Ridge Classifier",
        "ridge classifier": "Ridge Classifier",
        "random_forest": "Random Forest",
        "random forest": "Random Forest"
    }
    model_key = model_mapping.get(requested_model.lower(), requested_model)
    model = models_dict.get(model_key, best_model)

    start_time = time.perf_counter()

    res = classify_document(
        raw_text=raw_text,
        model=model,
        vectorizer=vectorizer,
        categories=categories,
        config=inference_config
    )

    elapsed = round(time.perf_counter() - start_time, 4)
    model_used_name = model_key if model_key in models_dict else metadata.get("best_model_name", "Logistic Regression")

    icon = CATEGORY_ICONS_4.get(res["status"], CATEGORY_ICONS_4.get(res["raw_prediction"], "📄"))
    category_disp = CATEGORY_DISPLAY_NAMES_4.get(res["prediction"], res["prediction"])

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
        "competing_classes": res.get("competing_classes", []),
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
    POST /predict/batch, POST /predict-batch, and POST /api/classify-batch
    Accepts:
    {
        "documents": ["doc1", "doc2", ...],
        "model": "optional_model_name"
    }
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

    docs = data.get("documents", [])
    if not isinstance(docs, list) or len(docs) == 0:
        return JSONResponse({"error": "'documents' must be a non-empty list of text items"}, status_code=400)

    # Security: Maximum batch size (500 documents)
    max_batch = 500
    if len(docs) > max_batch:
        return JSONResponse({"error": f"Batch size exceeds maximum limit of {max_batch} documents"}, status_code=400)

    requested_model = data.get("model", "")
    model = models_dict.get(requested_model, best_model)

    start_time = time.perf_counter()
    results = []
    class_distribution = {}

    for idx, item in enumerate(docs):
        if isinstance(item, str):
            row_text = item
            row_id = idx + 1
        elif isinstance(item, dict):
            row_text = ""
            for field in ["text", "document", "content", "message", "sentence", "body", "doc"]:
                val = item.get(field)
                if val is not None:
                    if isinstance(val, float) and np.isnan(val):
                        continue
                    str_val = str(val).strip()
                    if str_val and str_val.lower() != "nan":
                        row_text = str_val
                        break
            row_id = item.get("id", idx + 1)
        else:
            continue

        if not str(row_text).strip():
            results.append({
                "id": row_id,
                "text": "",
                "predicted_category": "Unknown / Out-of-Domain",
                "confidence": 0.0,
                "status": "unknown",
                "top_keywords": []
            })
            continue

        res = classify_document(str(row_text), model, vectorizer, categories, inference_config)
        pred_label = CATEGORY_DISPLAY_NAMES_4.get(res["prediction"], res["prediction"])
        class_distribution[pred_label] = class_distribution.get(pred_label, 0) + 1

        results.append({
            "id": row_id,
            "text": str(row_text)[:140],
            "predicted_category": res["prediction"],
            "category_name": pred_label,
            "confidence": round(res["confidence"] * 100, 2),
            "status": res["status"],
            "detected_topics": res["detected_topics"],
            "top_keywords": [kw["term"] for kw in res["top_keywords"][:4]]
        })

    elapsed = round(time.perf_counter() - start_time, 3)

    return JSONResponse({
        "results": results,
        "total": len(results),
        "summary": {
            "normal_count": sum(1 for r in results if r["status"] == "normal"),
            "ambiguous_count": sum(1 for r in results if r["status"] == "ambiguous"),
            "unknown_count": sum(1 for r in results if r["status"] == "unknown"),
            "low_confidence_count": sum(1 for r in results if r["status"] == "low_confidence"),
            "class_distribution": class_distribution
        },
        "processingTime": elapsed,
        "modelUsed": metadata.get("best_model_name", "Logistic Regression")
    })


async def explain_text(request):
    """
    POST /explain and POST /api/explain
    Accepts: {"text": "..."}
    Returns detailed feature contribution analysis.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return JSONResponse({"error": "Field 'text' must be a non-empty string"}, status_code=400)

    model_name = data.get("model", "")
    model = models_dict.get(model_name, best_model)

    explanation = explain_prediction(text, model, vectorizer, categories)
    return JSONResponse(explanation)


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

        # Safe filename sanitization against path traversal
        clean_filename = os.path.basename(filename)

        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = contents.decode("latin-1")
            except Exception:
                text = str(contents)

        words = text.split()
        return JSONResponse({
            "filename": clean_filename,
            "size": len(contents),
            "wordCount": len(words),
            "text": text
        })
    except Exception as e:
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=500)


# -----------------------------------------------------------------------------
# OpenAPI Specification & Interactive Documentation Handlers
# -----------------------------------------------------------------------------
OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "TextClassify AI - Document Classification API",
        "description": "Production-grade text classification REST API classifying documents into 4 domains: Computer Graphics, Baseball, Space Science, and Politics. Supports real-time inference, batch prediction, out-of-domain detection, ambiguity analysis, and explainability.",
        "version": "2.0.0"
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Health Check & Model Metadata",
                "responses": {
                    "200": {"description": "Service is healthy and active"}
                }
            }
        },
        "/models": {
            "get": {
                "summary": "List Available Models & Evaluation Metrics",
                "responses": {
                    "200": {"description": "Returns model comparison metrics and hyperparameters"}
                }
            }
        },
        "/predict": {
            "post": {
                "summary": "Classify a Single Document",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "example": "NASA launched a spacecraft into orbit."},
                                    "model": {"type": "string", "example": "Logistic Regression"}
                                },
                                "required": ["text"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Classification result with confidence and status"},
                    "400": {"description": "Invalid input or empty text"}
                }
            }
        },
        "/predict/batch": {
            "post": {
                "summary": "Batch Classify Multiple Documents",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "documents": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "example": [
                                            "NASA launched a rocket.",
                                            "The baseball team scored five runs.",
                                            "The government passed a law."
                                        ]
                                    }
                                },
                                "required": ["documents"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "List of prediction records with summary statistics"},
                    "400": {"description": "Invalid input or batch size exceeded"}
                }
            }
        },
        "/explain": {
            "post": {
                "summary": "Explain Document Prediction",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "example": "The baseball pitcher struck out ten batters."}
                                },
                                "required": ["text"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Word-level feature attributions and active vocabulary weights"}
                }
            }
        }
    }
}


async def openapi_json(request):
    return JSONResponse(OPENAPI_SPEC)


async def swagger_ui(request):
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>TextClassify AI API Documentation</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
      </head>
      <body style="margin: 0; background: #fafafa;">
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
          window.onload = () => {
            window.ui = SwaggerUIBundle({
              url: '/openapi.json',
              dom_id: '#swagger-ui',
              deepLinking: true,
              presets: [SwaggerUIBundle.presets.apis],
            });
          };
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)


async def redoc_ui(request):
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>TextClassify AI ReDoc</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700">
      </head>
      <body style="margin: 0; padding: 0;">
        <redoc spec-url="/openapi.json"></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
      </body>
    </html>
    """
    return HTMLResponse(html)


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
    Route("/predict/batch", predict_batch, methods=["POST"]),
    Route("/predict-batch", predict_batch, methods=["POST"]),
    Route("/api/classify-batch", predict_batch, methods=["POST"]),
    Route("/explain", explain_text, methods=["POST"]),
    Route("/api/explain", explain_text, methods=["POST"]),
    Route("/api/upload", upload_document, methods=["POST"]),
    Route("/openapi.json", openapi_json, methods=["GET"]),
    Route("/docs", swagger_ui, methods=["GET"]),
    Route("/redoc", redoc_ui, methods=["GET"])
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
    print(" Documentation available at: http://localhost:8000/docs")
    print(" Supported routes: /health, /models, /predict, /predict/batch, /explain")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
