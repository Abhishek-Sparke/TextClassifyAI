"""
Model Explainability Module for Text Document Classification
Classifying Text Documents Using Machine Learning (4 Classes)

Provides granular, interpretable explanations for individual model predictions:
- Predicted category & confidence score
- Class probability distribution
- Active TF-IDF vocabulary weights
- Feature contribution analysis via model coefficients / log-probabilities
- Distinct separation of statistical association vs human semantic reasoning
"""

from typing import Dict, Any, List, Optional
import numpy as np
from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities


def get_model_coefficients(model: Any) -> Optional[np.ndarray]:
    """
    Safely extracts linear weights / coefficients from diverse model structures,
    including calibrated ensemble wrappers (CalibratedClassifierCV).
    """
    if hasattr(model, "coef_"):
        return model.coef_

    # If wrapped in CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        coef_list = []
        for cc in model.calibrated_classifiers_:
            estimator = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if estimator is not None and hasattr(estimator, "coef_"):
                coef_list.append(estimator.coef_)
        if coef_list:
            return np.mean(coef_list, axis=0)

    # For Naive Bayes, feature_log_prob_ represents class conditional log-likelihood
    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_

    return None


def explain_prediction(
    raw_text: str,
    model: Any,
    vectorizer: Any,
    categories: List[str],
    top_n: int = 6
) -> Dict[str, Any]:
    """
    Generates a structured, mathematically grounded explanation of how
    the model reached its decision on a given input text.

    Parameters
    ----------
    raw_text : str
    model : fitted classifier
    vectorizer : fitted TfidfVectorizer
    categories : list of category strings
    top_n : int, number of top contributing terms to return

    Returns
    -------
    dict of explanation properties
    """
    if not raw_text or not raw_text.strip():
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {cat: 0.0 for cat in categories},
            "active_terms": [],
            "top_contributing_words": [],
            "competing_topic_words": [],
            "disclaimer": "No text provided to explain."
        }

    clean = preprocess_document(raw_text, apply_lemmatization=True)
    if not clean.strip():
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {cat: 0.0 for cat in categories},
            "active_terms": [],
            "top_contributing_words": [],
            "competing_topic_words": [],
            "disclaimer": "Input contained no recognizable vocabulary."
        }

    vec = vectorizer.transform([clean])
    probs = get_prediction_probabilities(model, vec)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = categories[pred_idx]
    confidence = float(probs[pred_idx])

    # Sorted indices for runner-up
    sorted_class_indices = np.argsort(probs)[::-1]
    second_idx = int(sorted_class_indices[1]) if len(sorted_class_indices) > 1 else pred_idx
    second_class = categories[second_idx]

    # Active TF-IDF terms
    active_terms_raw = get_top_tfidf_terms_for_document(vectorizer, vec, top_n=top_n)
    active_terms = [{"term": t, "tfidf_weight": round(w, 4)} for t, w in active_terms_raw]

    # Feature contribution via model coefficients
    coefs = get_model_coefficients(model)
    feature_names = np.array(vectorizer.get_feature_names_out())
    doc_array = vec.toarray().flatten()
    active_feature_mask = (doc_array > 0)
    active_feature_indices = np.where(active_feature_mask)[0]

    contributing_words = []
    competing_words = []

    if coefs is not None and len(active_feature_indices) > 0:
        # Check coefficient shape
        if coefs.ndim == 2 and coefs.shape[0] >= len(categories):
            pred_weights = coefs[pred_idx, active_feature_indices]
            feature_contributions = pred_weights * doc_array[active_feature_indices]

            top_contrib_order = np.argsort(feature_contributions)[::-1][:top_n]
            for idx in top_contrib_order:
                f_idx = active_feature_indices[idx]
                score = float(feature_contributions[idx])
                if score > 0:
                    contributing_words.append({
                        "word": str(feature_names[f_idx]),
                        "contribution_score": round(score, 4),
                        "direction": "supports_prediction"
                    })

            # Words supporting runner-up
            if second_idx != pred_idx:
                second_weights = coefs[second_idx, active_feature_indices]
                second_contributions = second_weights * doc_array[active_feature_indices]
                top_second_order = np.argsort(second_contributions)[::-1][:top_n]
                for idx in top_second_order:
                    f_idx = active_feature_indices[idx]
                    score = float(second_contributions[idx])
                    if score > 0:
                        competing_words.append({
                            "word": str(feature_names[f_idx]),
                            "contribution_score": round(score, 4),
                            "class": second_class
                        })

    # If model has no direct linear coefficients or if words were not found
    if not contributing_words and active_terms:
        for t in active_terms[:top_n]:
            contributing_words.append({
                "word": t["term"],
                "contribution_score": t["tfidf_weight"],
                "direction": "tfidf_weight"
            })

    prob_dict = {cat: round(float(probs[i]), 4) for i, cat in enumerate(categories)}

    disclaimer = (
        "Statistical Attribution Note: Word contribution weights are derived directly from "
        "model parameters and TF-IDF term weights. They quantify empirical statistical association "
        "within the trained corpus and do not imply causal understanding or human semantic reasoning."
    )

    return {
        "text": raw_text[:200] + ("..." if len(raw_text) > 200 else ""),
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "runner_up_class": second_class if second_idx != pred_idx else None,
        "probabilities": prob_dict,
        "active_terms": active_terms,
        "top_contributing_words": contributing_words,
        "competing_topic_words": competing_words,
        "disclaimer": disclaimer
    }
