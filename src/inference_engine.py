"""
Inference and Decision Engine Module
Classifying Text Documents Using Machine Learning (4 Classes)

Centralizes all classification decision logic, confidence calibration,
out-of-domain (UNKNOWN) detection, low-confidence warning (LOW_CONFIDENCE),
and multi-topic ambiguity analysis (AMBIGUOUS).

Supported Decision Statuses:
- NORMAL: High-confidence prediction belonging unambiguously to a single target category.
- LOW_CONFIDENCE: In-domain text with vocabulary overlap, but model confidence is marginal.
- UNKNOWN: Out-of-Domain text with zero or near-zero vocabulary overlap with target categories.
- AMBIGUOUS: Multi-topic or mixed-domain text with competing probability masses across multiple classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities


TARGET_4_CLASSES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc"
]

CATEGORY_DISPLAY_NAMES_4 = {
    "comp.graphics": "Computer Graphics",
    "rec.sport.baseball": "Baseball",
    "sci.space": "Space Science",
    "talk.politics.misc": "Politics"
}

CATEGORY_ICONS_4 = {
    "comp.graphics": "🎨",
    "rec.sport.baseball": "⚾",
    "sci.space": "🚀",
    "talk.politics.misc": "🏛️",
    "unknown": "❓",
    "ambiguous": "⚠️",
    "low_confidence": "⚡"
}


@dataclass
class InferenceConfig:
    """
    Central configuration for classification, out-of-domain, and ambiguity thresholds.
    Configurable in one single place.
    """
    # If top probability is below this, check for LOW_CONFIDENCE or UNKNOWN
    confidence_threshold: float = 0.58

    # Minimum sum of active TF-IDF weights to consider text in-domain
    min_active_tfidf: float = 0.05

    # Maximum probability difference between top 1 and top 2 for ambiguity
    ambiguity_margin: float = 0.60

    # Minimum sum of top-2 probabilities for a document to be considered multi-topic
    min_ambiguity_sum: float = 0.55

    # Minimum probability for top class in a multi-topic document
    min_ambiguity_p1: float = 0.28

    # Minimum probability for a secondary topic to be counted as detected
    topic_detection_threshold: float = 0.15

    # Minimum number of distinct domain keywords required for ambiguity
    min_ambiguity_keywords: int = 2

    # Display names mapping
    category_display_names: Dict[str, str] = field(
        default_factory=lambda: CATEGORY_DISPLAY_NAMES_4.copy()
    )


DEFAULT_INFERENCE_CONFIG = InferenceConfig()


def classify_document(
    raw_text: Optional[str],
    model: Any,
    vectorizer: Any,
    categories: Optional[List[str]] = None,
    config: Optional[InferenceConfig] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end inference on a single text document with
    unknown, low-confidence, and ambiguous detection heuristics.

    Parameters
    ----------
    raw_text : str or None
        Raw input string.
    model : fitted scikit-learn estimator
        Classification model.
    vectorizer : fitted TfidfVectorizer
        Feature extractor.
    categories : list of str, optional
        Target category labels (defaults to TARGET_4_CLASSES).
    config : InferenceConfig, optional
        Threshold configuration (defaults to DEFAULT_INFERENCE_CONFIG).

    Returns
    -------
    dict
        Structured prediction result.
    """
    if config is None:
        config = DEFAULT_INFERENCE_CONFIG

    if categories is None:
        categories = TARGET_4_CLASSES

    # 1. Input Validation
    if raw_text is None or not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "text": "" if raw_text is None else str(raw_text),
            "clean_text": "",
            "prediction": "Unknown / Out-of-Domain",
            "raw_prediction": categories[0] if categories else "comp.graphics",
            "confidence": 0.0,
            "probabilities": {cat: 0.0 for cat in categories},
            "status": "unknown",
            "detected_topics": [],
            "competing_classes": [],
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": "Empty or whitespace-only input",
            "active_tfidf_sum": 0.0
        }

    # 2. Text Preprocessing
    clean_text = preprocess_document(raw_text, apply_lemmatization=True)
    if not clean_text.strip():
        return {
            "text": raw_text,
            "clean_text": "",
            "prediction": "Unknown / Out-of-Domain",
            "raw_prediction": categories[0] if categories else "comp.graphics",
            "confidence": 0.0,
            "probabilities": {cat: 0.0 for cat in categories},
            "status": "unknown",
            "detected_topics": [],
            "competing_classes": [],
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": "Text contains no domain vocabulary or valid alphanumeric tokens",
            "active_tfidf_sum": 0.0
        }

    # 3. TF-IDF Vectorization
    vec = vectorizer.transform([clean_text])
    active_tfidf_weights = vec.toarray().flatten()
    active_tfidf_sum = float(np.sum(active_tfidf_weights))

    # Top keywords
    top_kw_raw = get_top_tfidf_terms_for_document(vectorizer, vec, top_n=6)
    top_keywords = [{"term": term, "weight": round(weight, 4)} for term, weight in top_kw_raw]

    # If zero vocabulary terms match the fitted vectorizer: UNKNOWN (Out-of-Domain)
    if active_tfidf_sum < config.min_active_tfidf or len(top_keywords) == 0:
        return {
            "text": raw_text,
            "clean_text": clean_text,
            "prediction": "Unknown / Out-of-Domain",
            "raw_prediction": categories[0] if categories else "comp.graphics",
            "confidence": 0.0,
            "probabilities": {cat: round(1.0 / len(categories), 4) for cat in categories},
            "status": "unknown",
            "detected_topics": [],
            "competing_classes": [],
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": f"Out-of-Domain: Zero vocabulary overlap with target domains (active TF-IDF: {active_tfidf_sum:.4f})",
            "active_tfidf_sum": round(active_tfidf_sum, 4)
        }

    # 4. Model Probabilities
    probs = get_prediction_probabilities(model, vec)[0]

    prob_dict = {}
    for idx, cat in enumerate(categories):
        p_val = float(probs[idx]) if idx < len(probs) else 0.0
        prob_dict[cat] = round(p_val, 4)

    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

    p1 = float(probs[top1_idx])
    p2 = float(probs[top2_idx]) if len(sorted_indices) > 1 else 0.0

    raw_predicted_cat = categories[top1_idx]

    # Candidate topics that cross the secondary detection threshold
    candidate_topics = []
    competing_classes_list = []
    for idx in sorted_indices:
        if probs[idx] >= config.topic_detection_threshold:
            cat_name = categories[idx]
            disp = config.category_display_names.get(cat_name, cat_name)
            candidate_topics.append(disp)
            competing_classes_list.append({
                "category": cat_name,
                "display_name": disp,
                "probability": round(float(probs[idx]), 4)
            })

    # 5. Apply Decision Rules: Out-of-Domain vs Ambiguous vs Low-Confidence vs Normal
    is_out_of_domain = False
    is_ambiguous = False

    # Check for near-uniform / high entropy out-of-domain distribution
    # If the highest probability is below 0.35, or the spread between highest and lowest class is tiny (< 0.12),
    # the classifier has no distinctive topical signal for any class (e.g. "Pizza is my favorite food", "low battery")
    if p1 < 0.35 or (p1 - float(np.min(probs))) < 0.12:
        is_out_of_domain = True
        status = "unknown"
        prediction = "Unknown / Out-of-Domain"
        final_detected_topics = []
        reason = f"Out-of-Domain: Model confidence is nearly uniform across all classes (max {p1:.1%}, spread {p1 - float(np.min(probs)):.1%})"

    # Multi-topic ambiguity condition:
    # Requires genuine competition between 2+ classes with multiple domain keywords
    elif (
        len(top_keywords) >= config.min_ambiguity_keywords
        and p2 >= config.topic_detection_threshold
        and (p1 - p2) <= config.ambiguity_margin
        and p1 < 0.82
        and len(candidate_topics) >= 2
    ):
        is_ambiguous = True
        status = "ambiguous"
        prediction = "Ambiguous / Multi-topic"
        final_detected_topics = candidate_topics
        reason = f"Multiple competing topics detected ({', '.join(candidate_topics)}): top-2 probability gap {abs(p1 - p2):.2%}"

    elif p1 < config.confidence_threshold:
        # If active TF-IDF is barely above threshold or keywords are minimal
        if active_tfidf_sum < 0.15 or len(top_keywords) < 2:
            is_out_of_domain = True
            status = "unknown"
            prediction = "Unknown / Out-of-Domain"
            final_detected_topics = []
            reason = f"Out-of-Domain: Low topical signal (active TF-IDF {active_tfidf_sum:.3f}, confidence {p1:.1%})"
        else:
            status = "low_confidence"
            prediction = raw_predicted_cat
            final_detected_topics = [config.category_display_names.get(raw_predicted_cat, raw_predicted_cat)]
            reason = f"Low confidence prediction ({p1:.1%}) below standard threshold ({config.confidence_threshold:.1%})"

    else:
        status = "normal"
        prediction = raw_predicted_cat
        final_detected_topics = [config.category_display_names.get(raw_predicted_cat, raw_predicted_cat)]
        reason = f"Dominant topic clearly identified: {config.category_display_names.get(raw_predicted_cat, raw_predicted_cat)} ({p1:.1%})"

    return {
        "text": raw_text,
        "clean_text": clean_text,
        "prediction": prediction,
        "raw_prediction": raw_predicted_cat,
        "confidence": round(p1, 4),
        "probabilities": prob_dict,
        "status": status,
        "detected_topics": final_detected_topics,
        "competing_classes": competing_classes_list,
        "top_keywords": top_keywords,
        "is_out_of_domain": is_out_of_domain,
        "is_ambiguous": is_ambiguous,
        "reason": reason,
        "active_tfidf_sum": round(active_tfidf_sum, 4)
    }
