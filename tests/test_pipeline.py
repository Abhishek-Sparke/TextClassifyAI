"""
Automated Test Suite for Text Classification Pipeline (4 Classes)

Validates:
1. Text Preprocessing & Cleaning (None, empty, URLs, emails, numbers, 3D token preservation)
2. Model Artifacts Integrity (best_model, tfidf_vectorizer, all_models, model_metadata)
3. In-Domain Classification across all 4 classes
4. Unknown / Out-of-Domain Detection
5. Ambiguous / Multi-Topic Detection
6. REST API /predict and Error Handling (400 Bad Request, empty text, invalid JSON)
7. CSV Batch Processing row-by-row
"""

import os
import sys
import json
import asyncio
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities
from src.inference_engine import (
    classify_document,
    InferenceConfig,
    TARGET_4_CLASSES,
    CATEGORY_DISPLAY_NAMES_4
)
from server import predict_text, health_check, get_models


def test_clean_text():
    # Null and empty
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""

    # URLs, emails, numbers
    sample = "Visit https://space.nasa.gov or email contact@orbit.org with 9999 satellites!"
    cleaned = clean_text(sample)
    assert "https" not in cleaned
    assert "orbit.org" not in cleaned
    assert "9999" not in cleaned
    assert cleaned == cleaned.lower()

    # Preserves alphanumeric like 3d
    sample_3d = "Creating a 3D polygon model."
    cleaned_3d = clean_text(sample_3d)
    assert "3d" in cleaned_3d


def test_preprocess_document():
    raw_text = "The quick brown fox jumps over the lazy dog and renders a 3D scene!"
    processed = preprocess_document(raw_text, apply_lemmatization=True)
    assert isinstance(processed, str)
    assert len(processed) > 0

    tokens = processed.split()
    assert "the" not in tokens
    assert "and" not in tokens
    assert "3d" in tokens


def test_model_artifacts_exist():
    models_dir = "models"
    assert os.path.exists(os.path.join(models_dir, "best_model.joblib")), "best_model.joblib missing"
    assert os.path.exists(os.path.join(models_dir, "tfidf_vectorizer.joblib")), "tfidf_vectorizer.joblib missing"
    assert os.path.exists(os.path.join(models_dir, "all_models.joblib")), "all_models.joblib missing"
    assert os.path.exists(os.path.join(models_dir, "model_metadata.json")), "model_metadata.json missing"


def test_in_domain_classification():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    with open(os.path.join(models_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    categories = meta["categories"]
    assert len(categories) == 4
    assert set(categories) == set(TARGET_4_CLASSES)

    # Test pure Space
    res_space = classify_document("NASA launched a spacecraft into orbit.", model, vectorizer, categories)
    assert res_space["status"] == "normal"
    assert res_space["prediction"] == "sci.space"
    assert res_space["confidence"] > 0.70

    # Test pure Baseball
    res_base = classify_document("The starting pitcher struck out nine batters in the baseball game.", model, vectorizer, categories)
    assert res_base["status"] == "normal"
    assert res_base["prediction"] == "rec.sport.baseball"
    assert res_base["confidence"] > 0.70

    # Test pure Politics
    res_pol = classify_document("The government passed a new law regarding federal taxation.", model, vectorizer, categories)
    assert res_pol["status"] == "normal"
    assert res_pol["prediction"] == "talk.politics.misc"

    # Test pure Graphics
    res_graph = classify_document("3D rendering software with GPU ray tracing and polygon shading.", model, vectorizer, categories)
    assert res_graph["status"] == "normal"
    assert res_graph["prediction"] == "comp.graphics"


def test_unknown_detection():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    # Out of domain text
    res_pizza = classify_document("I ate pizza today.", model, vectorizer)
    assert res_pizza["status"] == "unknown"
    assert res_pizza["prediction"] == "Unknown / Out-of-Domain"
    assert res_pizza["is_out_of_domain"] is True

    # Empty text
    res_empty = classify_document("", model, vectorizer)
    assert res_empty["status"] == "unknown"
    assert res_empty["prediction"] == "Unknown / Out-of-Domain"


def test_ambiguous_detection():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    benchmark_text = "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
    res = classify_document(benchmark_text, model, vectorizer)

    assert res["status"] == "ambiguous", f"Expected 'ambiguous', got {res['status']}"
    assert res["prediction"] == "Ambiguous / Multi-topic"
    assert res["is_ambiguous"] is True
    assert len(res["detected_topics"]) >= 2
    # Verify probabilities are distributed
    probs = res["probabilities"]
    assert probs["rec.sport.baseball"] > 0.15
    assert probs["talk.politics.misc"] > 0.15


def test_api_endpoints_and_error_handling():
    class MockReq:
        def __init__(self, data):
            self._data = data
        async def json(self):
            return self._data

    async def run_api_tests():
        # Health check
        h_resp = await health_check(None)
        assert h_resp.status_code == 200
        h_data = json.loads(h_resp.body)
        assert h_data["status"] == "healthy"
        assert len(h_data["classes"]) == 4

        # Models list
        m_resp = await get_models(None)
        assert m_resp.status_code == 200
        m_data = json.loads(m_resp.body)
        assert len(m_data["models"]) == 4

        # Empty text -> 400 Bad Request
        p_empty = await predict_text(MockReq({"text": ""}))
        assert p_empty.status_code == 400

        # Missing text field -> 400 Bad Request
        p_missing = await predict_text(MockReq({}))
        assert p_missing.status_code == 400

        # Non-string text -> 400 Bad Request
        p_invalid = await predict_text(MockReq({"text": 12345}))
        assert p_invalid.status_code == 400

        # Valid predict
        p_valid = await predict_text(MockReq({"text": "NASA launched a spacecraft into orbit."}))
        assert p_valid.status_code == 200
        p_data = json.loads(p_valid.body)
        assert p_data["prediction"] == "sci.space"
        assert p_data["status"] == "normal"
        assert "probabilities" in p_data
        assert "detected_topics" in p_data
        assert "top_keywords" in p_data

        # Benchmark ambiguous prompt
        p_amb = await predict_text(MockReq({
            "text": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
        }))
        assert p_amb.status_code == 200
        amb_data = json.loads(p_amb.body)
        assert amb_data["status"] == "ambiguous"
        assert amb_data["prediction"] == "Ambiguous / Multi-topic"
        assert len(amb_data["detected_topics"]) >= 2

    asyncio.run(run_api_tests())


def test_csv_batch_processing():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    test_rows = [
        {"id": 1, "document": "NASA launched a spacecraft into orbit."},
        {"id": 2, "document": "The starting pitcher struck out nine batters in the baseball game."},
        {"id": 3, "document": "I ate pizza today."},
        {"id": 4, "document": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."}
    ]

    results = []
    for r in test_rows:
        res = classify_document(r["document"], model, vectorizer)
        results.append({
            "id": r["id"],
            "text": r["document"],
            "predicted_category": res["prediction"],
            "confidence": res["confidence"],
            "status": res["status"]
        })

    df_out = pd.DataFrame(results)
    assert len(df_out) == 4
    assert df_out.iloc[0]["status"] == "normal"
    assert df_out.iloc[1]["status"] == "normal"
    assert df_out.iloc[2]["status"] == "unknown"
    assert df_out.iloc[3]["status"] == "ambiguous"

    # Export to CSV check
    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    assert len(csv_bytes) > 0


if __name__ == "__main__":
    test_clean_text()
    print("[PASS] test_clean_text")
    test_preprocess_document()
    print("[PASS] test_preprocess_document")
    test_model_artifacts_exist()
    print("[PASS] test_model_artifacts_exist")
    test_in_domain_classification()
    print("[PASS] test_in_domain_classification")
    test_unknown_detection()
    print("[PASS] test_unknown_detection")
    test_ambiguous_detection()
    print("[PASS] test_ambiguous_detection")
    test_api_endpoints_and_error_handling()
    print("[PASS] test_api_endpoints_and_error_handling")
    test_csv_batch_processing()
    print("[PASS] test_csv_batch_processing")
    print("\n[SUCCESS] All 8 automated test suites passed successfully!")
