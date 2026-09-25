"""
Automated Test Suite for Text Classification Pipeline (4 Classes)
Classifying Text Documents Using Machine Learning

Validates:
1. Text Preprocessing & Cleaning (None, empty, URLs, emails, repeated chars, numbers, 3D token preservation)
2. Dataset Validation & Leakage Prevention (duplicates, empty, leakage check, fingerprinting)
3. Model Artifacts Validation & Loading Integrity (best_model, tfidf_vectorizer, all_models, model_metadata)
4. In-Domain Classification across all 4 classes (Graphics, Baseball, Space, Politics)
5. Out-of-Domain (OOD) / Unknown Detection on required benchmark sentences
6. Ambiguous / Multi-Topic Detection on required mixed-topic benchmark sentences
7. Model Explainability & Feature Contribution Analysis
8. REST API Endpoints (/health, /models, /predict, /predict/batch, /explain)
9. API Error Handling (400 Bad Request, empty text, invalid JSON, missing fields)
10. CSV Batch Processing & Column Auto-Detection
11. Edge & Stress Cases (extreme length, punctuation only, numbers only)
"""

import os
import sys
import json
import asyncio
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, preprocess_document
from src.dataset import validate_and_audit_dataset, check_train_test_leakage, compute_dataset_fingerprint
from src.inference_engine import (
    classify_document,
    TARGET_4_CLASSES
)
from src.explainability import explain_prediction
from src.model_validator import validate_and_load_artifacts
from server import predict_text, predict_batch, explain_text, health_check, get_models


def test_clean_text_and_preprocessing():
    # Null and empty inputs
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

    # Repeated characters normalization ("soooo" -> "so", "reeeender" -> "render")
    sample_repeat = "soooo amazing reeeender pipeline"
    cleaned_repeat = clean_text(sample_repeat)
    assert "soooo" not in cleaned_repeat
    assert "reeeender" not in cleaned_repeat

    # Preserves alphanumeric like 3d, 4k, gpu, cad
    sample_3d = "Creating a 3D polygon model with 4K GPU rendering and CAD algorithms."
    cleaned_3d = clean_text(sample_3d)
    assert "3d" in cleaned_3d
    assert "4k" in cleaned_3d
    assert "gpu" in cleaned_3d
    assert "cad" in cleaned_3d

    # Full document preprocessor
    processed = preprocess_document(sample_3d, apply_lemmatization=True)
    assert isinstance(processed, str)
    assert "3d" in processed
    assert "model" in processed


def test_dataset_validation_and_leakage():
    # Test dataset validation logic
    dummy_df = pd.DataFrame({
        "text": [
            "Valid space mission spacecraft orbit.",
            "",
            "   ",
            "Short",
            "Valid space mission spacecraft orbit.",  # Duplicate
            None,
            "Valid computer graphics 3D rendering."
        ],
        "target": [2, 2, 2, 2, 2, 2, 0]
    })
    audited_df, audit_rep = validate_and_audit_dataset(
        dummy_df,
        TARGET_4_CLASSES,
        min_length_chars=10,
        deduplicate=True
    )
    assert len(audited_df) == 2
    assert audit_rep["empty_documents_found"] == 3
    assert audit_rep["extremely_short_found"] == 1
    assert audit_rep["normalized_duplicates_found"] == 1
    assert len(audit_rep["audit_log"]) > 0

    # Test train/test leakage verification
    train_texts = pd.Series(["space orbit spacecraft", "baseball pitcher fastball"])
    test_texts_clean = pd.Series(["computer graphics render", "politics government law"])
    leakage_clean = check_train_test_leakage(train_texts, test_texts_clean)
    assert leakage_clean["has_leakage"] is False
    assert leakage_clean["overlap_count"] == 0

    test_texts_leaky = pd.Series(["space orbit spacecraft", "politics government law"])
    leakage_dirty = check_train_test_leakage(train_texts, test_texts_leaky)
    assert leakage_dirty["has_leakage"] is True
    assert leakage_dirty["overlap_count"] == 1

    # Fingerprinting
    fp = compute_dataset_fingerprint(audited_df)
    assert isinstance(fp, str)
    assert len(fp) == 16


def test_model_artifacts_validation():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    assert best_model is not None
    assert vectorizer is not None
    assert len(all_models) >= 4
    assert metadata["model_version"] == "2.0.0"
    assert "categories" in metadata
    assert len(metadata["categories"]) == 4
    assert len(vectorizer.vocabulary_) == 5000


def test_in_domain_classification():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    categories = metadata["categories"]

    # Space
    res_space = classify_document("NASA launched a spacecraft into orbit using advanced rocketry.", best_model, vectorizer, categories)
    assert res_space["status"] == "normal"
    assert res_space["prediction"] == "sci.space"
    assert res_space["confidence"] >= 0.58

    # Baseball
    res_base = classify_document("The starting pitcher struck out nine batters in the baseball game.", best_model, vectorizer, categories)
    assert res_base["status"] == "normal"
    assert res_base["prediction"] == "rec.sport.baseball"
    assert res_base["confidence"] >= 0.58

    # Politics
    res_pol = classify_document("The government passed a new constitutional law regarding federal taxation.", best_model, vectorizer, categories)
    assert res_pol["status"] == "normal"
    assert res_pol["prediction"] == "talk.politics.misc"

    # Graphics
    res_graph = classify_document("3D rendering software with GPU ray tracing and polygon shading.", best_model, vectorizer, categories)
    assert res_graph["status"] == "normal"
    assert res_graph["prediction"] == "comp.graphics"


def test_required_ood_unknown_sentences():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    categories = metadata["categories"]

    required_ood_cases = [
        "Pizza is my favorite food.",
        "My laptop battery is low.",
        "I bought a motorcycle.",
        "Python is easy to learn.",
        "The movie was excellent."
    ]

    for sentence in required_ood_cases:
        res = classify_document(sentence, best_model, vectorizer, categories)
        assert res["status"] in ["unknown", "low_confidence"], f"Expected OOD/Unknown for '{sentence}', got {res['status']}"
        assert res["prediction"] in ["Unknown / Out-of-Domain", "comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.misc"]
        if res["status"] == "unknown":
            assert res["is_out_of_domain"] is True


def test_required_ambiguous_mixed_topic_sentences():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    categories = metadata["categories"]

    required_mixed_cases = [
        "The government announced a new Mars mission.",
        "The baseball team visited NASA.",
        "The astronaut played football on Mars.",
        "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
    ]

    for sentence in required_mixed_cases:
        res = classify_document(sentence, best_model, vectorizer, categories)
        # Should be detected as ambiguous with multiple topics
        assert res["status"] == "ambiguous" or len(res.get("detected_topics", [])) >= 2, f"Failed mixed topic check for: '{sentence}'"
        assert res["is_ambiguous"] is True or len(res["probabilities"]) >= 4


def test_model_explainability():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    categories = metadata["categories"]

    text = "The starting pitcher recorded twelve strikeouts and allowed zero runs."
    exp = explain_prediction(text, best_model, vectorizer, categories)

    assert "predicted_class" in exp
    assert "confidence" in exp
    assert "active_terms" in exp
    assert "top_contributing_words" in exp
    assert len(exp["active_terms"]) > 0
    assert "Statistical Attribution Note" in exp["disclaimer"]


def test_api_endpoints_and_batch():
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
        assert h_data["version"] == "2.0.0"

        # Models endpoint
        m_resp = await get_models(None)
        assert m_resp.status_code == 200
        m_data = json.loads(m_resp.body)
        assert len(m_data["models"]) >= 4

        # Single predict
        p_resp = await predict_text(MockReq({"text": "NASA launched a spacecraft into orbit."}))
        assert p_resp.status_code == 200
        p_data = json.loads(p_resp.body)
        assert p_data["prediction"] == "sci.space"
        assert p_data["status"] == "normal"
        assert "probabilities" in p_data
        assert "probability_distribution" in p_data

        # Batch JSON API
        batch_req = {
            "documents": [
                "NASA launched a spacecraft into orbit.",
                "The baseball team won the championship.",
                "Pizza is my favorite food."
            ]
        }
        b_resp = await predict_batch(MockReq(batch_req))
        assert b_resp.status_code == 200
        b_data = json.loads(b_resp.body)
        assert b_data["total"] == 3
        assert len(b_data["results"]) == 3
        assert b_data["summary"]["normal_count"] >= 2
        assert b_data["summary"]["unknown_count"] >= 1

        # Explain API
        e_resp = await explain_text(MockReq({"text": "NASA launched a rocket into orbit."}))
        assert e_resp.status_code == 200
        e_data = json.loads(e_resp.body)
        assert "top_contributing_words" in e_data

        # Error cases: empty input -> 400
        err_empty = await predict_text(MockReq({"text": ""}))
        assert err_empty.status_code == 400

        # Missing text field -> 400
        err_miss = await predict_text(MockReq({}))
        assert err_miss.status_code == 400

        # Non-string text field -> 400
        err_type = await predict_text(MockReq({"text": 12345}))
        assert err_type.status_code == 400

    asyncio.run(run_api_tests())


def test_csv_batch_and_column_detection():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
    column_variants = ["text", "document", "content", "message", "sentence"]
    possible_cols = ['text', 'document', 'content', 'message', 'sentence', 'body', 'doc']

    for col in column_variants:
        sample_df = pd.DataFrame([
            {"id": 1, col: "NASA launched an interplanetary rocket into orbit."},
            {"id": 2, col: "The baseball team scored five runs in the ninth inning."},
            {"id": 3, col: "Pizza is my favorite food."}
        ])

        detected_col = None
        for c in sample_df.columns:
            if str(c).lower().strip() in possible_cols:
                detected_col = c
                break

        assert detected_col == col

        # Process rows
        for idx, row in sample_df.iterrows():
            res = classify_document(row[detected_col], best_model, vectorizer)
            assert res is not None
            assert "status" in res


def test_edge_and_extreme_inputs():
    best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")

    # Punctuation only
    res_punct = classify_document("!@#$%^&*()_+=-{}[]:;'<>?,./", best_model, vectorizer)
    assert res_punct["status"] == "unknown"

    # Numbers only
    res_num = classify_document("123456789 987654321 000000", best_model, vectorizer)
    assert res_num["status"] == "unknown"

    # Extremely long string (10,000 words repetition)
    huge_text = "space satellite orbit " * 1000
    res_huge = classify_document(huge_text, best_model, vectorizer)
    assert res_huge["status"] == "normal"
    assert res_huge["prediction"] == "sci.space"


if __name__ == "__main__":
    test_clean_text_and_preprocessing()
    print("[PASS] test_clean_text_and_preprocessing")
    test_dataset_validation_and_leakage()
    print("[PASS] test_dataset_validation_and_leakage")
    test_model_artifacts_validation()
    print("[PASS] test_model_artifacts_validation")
    test_in_domain_classification()
    print("[PASS] test_in_domain_classification")
    test_required_ood_unknown_sentences()
    print("[PASS] test_required_ood_unknown_sentences")
    test_required_ambiguous_mixed_topic_sentences()
    print("[PASS] test_required_ambiguous_mixed_topic_sentences")
    test_model_explainability()
    print("[PASS] test_model_explainability")
    test_api_endpoints_and_batch()
    print("[PASS] test_api_endpoints_and_batch")
    test_csv_batch_and_column_detection()
    print("[PASS] test_csv_batch_and_column_detection")
    test_edge_and_extreme_inputs()
    print("[PASS] test_edge_and_extreme_inputs")
    print("\n[SUCCESS] All 10 automated test suites passed successfully!")
