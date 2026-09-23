import os
import json
import numpy as np
import joblib

from src.preprocessing import clean_text, preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities


def test_clean_text():
    # Test handling None and empty string
    assert clean_text(None) == ""
    assert clean_text("") == ""

    # Test stripping URLs, emails, and numbers
    sample = "Contact us at info@example.com or visit https://textclassify.ai with 12345 users!"
    cleaned = clean_text(sample)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "12345" not in cleaned
    assert cleaned == cleaned.lower()


def test_preprocess_document():
    raw_text = "The quick brown fox jumps over the lazy dog and enjoys running fast!"
    processed = preprocess_document(raw_text, apply_lemmatization=True)
    assert isinstance(processed, str)
    assert len(processed) > 0
    # Stop words like 'the', 'and', 'over' should be removed
    tokens = processed.split()
    assert "the" not in tokens
    assert "and" not in tokens


def test_model_artifacts_exist():
    models_dir = "models"
    assert os.path.exists(os.path.join(models_dir, "best_model.joblib")), "best_model.joblib missing"
    assert os.path.exists(os.path.join(models_dir, "tfidf_vectorizer.joblib")), "tfidf_vectorizer.joblib missing"
    assert os.path.exists(os.path.join(models_dir, "model_metadata.json")), "model_metadata.json missing"


def test_model_inference_pipeline():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
        meta = json.load(f)

    categories = meta["categories"]

    # Sample sports text
    sample_text = (
        "The starting pitcher delivered nine strikeouts over seven scoreless innings "
        "to lead the team to victory in the baseball championship game."
    )

    clean = preprocess_document(sample_text, apply_lemmatization=True)
    vec = vectorizer.transform([clean])
    prediction_idx = model.predict(vec)[0]
    probabilities = get_prediction_probabilities(model, vec)[0]

    assert len(categories) == 20, f"Expected 20 categories, got {len(categories)}"
    assert "sci.crypt" in categories
    assert "soc.religion.christian" in categories
    assert "rec.motorcycles" in categories
    assert "comp.windows.x" in categories

    assert 0 <= prediction_idx < len(categories)
    predicted_category = categories[prediction_idx]
    assert isinstance(predicted_category, str)

    # Probabilities should sum to approximately 1.0
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-3)

    # Confidence for predicted category should be positive
    confidence = probabilities[prediction_idx]
    assert 0.0 < confidence <= 1.0

    # Top TF-IDF keywords should be extracted
    top_terms = get_top_tfidf_terms_for_document(vectorizer, vec, top_n=3)
    assert isinstance(top_terms, list)
    assert len(top_terms) > 0


def test_new_categories_inference():
    models_dir = "models"
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
        meta = json.load(f)
    categories = meta["categories"]

    assert len(categories) == 20

    # Test Cryptography classification
    crypt_text = "Public key cryptography and RSA encryption algorithms secure private data against cryptanalysis."
    clean_crypt = preprocess_document(crypt_text, apply_lemmatization=True)
    vec_crypt = vectorizer.transform([clean_crypt])
    pred_crypt = categories[model.predict(vec_crypt)[0]]
    assert pred_crypt == "sci.crypt"

    # Test Motorcycles classification
    moto_text = "The motorcycle rider replaced the helmet and rode the Harley bike down the highway."
    clean_moto = preprocess_document(moto_text, apply_lemmatization=True)
    vec_moto = vectorizer.transform([clean_moto])
    pred_moto = categories[model.predict(vec_moto)[0]]
    assert pred_moto == "rec.motorcycles"

    # Test Christianity classification
    relig_text = "The Christian church sermon addressed biblical scripture and gospel faith in Jesus Christ."
    clean_relig = preprocess_document(relig_text, apply_lemmatization=True)
    vec_relig = vectorizer.transform([clean_relig])
    pred_relig = categories[model.predict(vec_relig)[0]]
    assert pred_relig == "soc.religion.christian"

    # Test X Window System classification
    x_text = "The X11 server connects to client display windows and runs the xterm window manager terminal."
    clean_x = preprocess_document(x_text, apply_lemmatization=True)
    vec_x = vectorizer.transform([clean_x])
    pred_x = categories[model.predict(vec_x)[0]]
    assert pred_x == "comp.windows.x"

