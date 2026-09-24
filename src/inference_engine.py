"""
Inference and Decision Engine Module

Centralizes all classification decision logic, confidence scoring,
out-of-domain (Unknown) detection, and multi-topic (Ambiguous) detection.
All thresholds are centrally defined and configurable here.
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
    "ambiguous": "⚠️"
}


@dataclass
class InferenceConfig:
    """
    Central configuration for classification and confidence thresholds.
    Configurable in one single place.
    """
    # If top probability is below this, classify as "Unknown / Out-of-Domain"
    confidence_threshold: float = 0.58

    # Minimum sum of active TF-IDF weights to consider text in-domain.
    min_active_tfidf: float = 0.05

    # Maximum probability difference between top 1 and top 2 for ambiguity
    ambiguity_margin: float = 0.38

    # Minimum sum of top-2 probabilities for a document to be considered multi-topic
    min_ambiguity_sum: float = 0.60

    # Minimum probability for top class in a multi-topic document
    min_ambiguity_p1: float = 0.30

    # Minimum probability for a secondary topic to be counted as detected
    topic_detection_threshold: float = 0.16

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
    unknown and ambiguous detection heuristics.

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
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": "Empty or whitespace-only input"
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
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": "Text contains no domain vocabulary"
        }

    # 3. TF-IDF Vectorization
    vec = vectorizer.transform([clean_text])
    active_tfidf_weights = vec.toarray().flatten()
    active_tfidf_sum = float(np.sum(active_tfidf_weights))

    # Top keywords
    top_kw_raw = get_top_tfidf_terms_for_document(vectorizer, vec, top_n=6)
    top_keywords = [{"term": term, "weight": round(weight, 4)} for term, weight in top_kw_raw]

    # If zero vocabulary terms match the fitted vectorizer
    if active_tfidf_sum < config.min_active_tfidf or len(top_keywords) == 0:
        return {
            "text": raw_text,
            "clean_text": clean_text,
            "prediction": "Unknown / Out-of-Domain",
            "raw_prediction": categories[0] if categories else "comp.graphics",
            "confidence": 0.0,
            "probabilities": {cat: 0.25 for cat in categories},
            "status": "unknown",
            "detected_topics": [],
            "top_keywords": [],
            "is_out_of_domain": True,
            "is_ambiguous": False,
            "reason": f"Zero vocabulary overlap with domain (active TF-IDF: {active_tfidf_sum:.4f})",
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
    for idx in sorted_indices:
        if probs[idx] >= config.topic_detection_threshold:
            cat_name = categories[idx]
            disp = config.category_display_names.get(cat_name, cat_name)
            candidate_topics.append(disp)

    # 5. Apply Decision Rules: Out-of-Domain vs Ambiguous vs Normal
    is_out_of_domain = False
    is_ambiguous = False

    # Multi-topic ambiguity condition:
    # Requires genuine competition between 2+ classes:
    # 1. Document has at least 2 distinct domain keywords
    # 2. Top class has sufficient signal (p1 >= min_ambiguity_p1)
    # 3. Top 2 classes together account for significant mass (p1 + p2 >= min_ambiguity_sum)
    # 4. Gap between top 1 and top 2 is within the ambiguity margin
    # 5. Secondary topic has substantial confidence (p2 >= topic_detection_threshold)
    # 6. Exclude pure single topics with very high confidence (p1 < 0.78)
    # 7. At least 2 candidate topics detected
    ambiguity_condition = (
        (len(top_keywords) >= config.min_ambiguity_keywords)
        and (p1 >= config.min_ambiguity_p1)
        and (p1 + p2 >= config.min_ambiguity_sum)
        and (p1 - p2 <= config.ambiguity_margin)
        and (p2 >= config.topic_detection_threshold)
        and (p1 < 0.78)
        and (len(candidate_topics) >= 2)
    )

    if ambiguity_condition:
        is_ambiguous = True
        status = "ambiguous"
        prediction = "Ambiguous / Multi-topic"
        final_detected_topics = candidate_topics
        reason = f"Multiple competing topics detected ({', '.join(candidate_topics)}): top-2 probability gap {abs(p1 - p2):.2%}"

    elif p1 < config.confidence_threshold:
        is_out_of_domain = True
        status = "unknown"
        prediction = "Unknown / Out-of-Domain"
        final_detected_topics = []
        reason = f"Model confidence ({p1:.2%}) below confidence threshold ({config.confidence_threshold:.2%})"

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
        "top_keywords": top_keywords,
        "is_out_of_domain": is_out_of_domain,
        "is_ambiguous": is_ambiguous,
        "reason": reason,
        "active_tfidf_sum": round(active_tfidf_sum, 4)
    }
