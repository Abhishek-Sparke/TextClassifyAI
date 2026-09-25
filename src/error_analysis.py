"""
Error Analysis System for Text Document Classification
Classifying Text Documents Using Machine Learning (4 Classes)

Performs deep-dive forensic analysis on model misclassifications:
- False positives and false negatives per class
- Frequently confused class pairs (off-diagonal confusion matrix ranking)
- Slice-based error analysis by document length (short, medium, long)
- Confidence stratification (low-confidence errors vs overconfident errors)
- Concrete misclassified examples with actual class, predicted class, confidence, and text
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities


def run_error_analysis(
    model: Any,
    vectorizer: Any,
    X_test_raw: pd.Series,
    y_test: pd.Series,
    target_names: List[str],
    short_words_threshold: int = 25,
    long_words_threshold: int = 120
) -> Dict[str, Any]:
    """
    Executes comprehensive error analysis on the test set predictions.

    Parameters
    ----------
    model : fitted classifier
    vectorizer : fitted TfidfVectorizer
    X_test_raw : pd.Series of raw or preprocessed test texts
    y_test : pd.Series of true target integer labels
    target_names : list of class names
    short_words_threshold : word count boundary for short texts
    long_words_threshold : word count boundary for long texts

    Returns
    -------
    dict containing:
        - total_test_samples
        - total_errors
        - overall_error_rate
        - confused_class_pairs
        - length_slice_metrics
        - confidence_breakdown
        - sample_errors
    """
    X_test_list = list(X_test_raw)
    y_true_list = list(y_test)

    # Vectorize and get predictions & probabilities
    X_vec = vectorizer.transform(X_test_list)
    probs = get_prediction_probabilities(model, X_vec)
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    total_samples = len(y_true_list)
    misclassified_indices = [i for i in range(total_samples) if y_true_list[i] != y_pred[i]]
    total_errors = len(misclassified_indices)
    overall_error_rate = round(total_errors / max(total_samples, 1), 4)

    # 1. Frequently Confused Class Pairs
    cm = confusion_matrix(y_true_list, y_pred, labels=list(range(len(target_names))))
    confused_pairs = []
    for actual_idx in range(len(target_names)):
        for pred_idx in range(len(target_names)):
            if actual_idx != pred_idx and cm[actual_idx, pred_idx] > 0:
                confused_pairs.append({
                    "actual_class": target_names[actual_idx],
                    "predicted_class": target_names[pred_idx],
                    "error_count": int(cm[actual_idx, pred_idx])
                })
    confused_pairs.sort(key=lambda x: x["error_count"], reverse=True)

    # 2. Slice-based analysis by length
    length_buckets = {
        "short (< 25 words)": {"total": 0, "errors": 0},
        "medium (25-120 words)": {"total": 0, "errors": 0},
        "long (> 120 words)": {"total": 0, "errors": 0}
    }

    sample_errors = []

    for i in range(total_samples):
        text = str(X_test_list[i])
        word_count = len(text.split())
        is_error = (i in misclassified_indices)

        if word_count < short_words_threshold:
            bucket_key = "short (< 25 words)"
        elif word_count <= long_words_threshold:
            bucket_key = "medium (25-120 words)"
        else:
            bucket_key = "long (> 120 words)"

        length_buckets[bucket_key]["total"] += 1
        if is_error:
            length_buckets[bucket_key]["errors"] += 1

            # Extract sample details for errors
            actual_cat = target_names[y_true_list[i]]
            pred_cat = target_names[y_pred[i]]
            conf = float(confs[i])

            # Get active TF-IDF terms
            doc_vec = X_vec[i]
            top_kw = get_top_tfidf_terms_for_document(vectorizer, doc_vec, top_n=5)

            sample_errors.append({
                "test_index": i,
                "actual_class": actual_cat,
                "predicted_class": pred_cat,
                "confidence": round(conf, 4),
                "word_count": word_count,
                "length_category": bucket_key.split()[0],
                "top_keywords": [kw[0] for kw in top_kw],
                "text_snippet": text[:180] + ("..." if len(text) > 180 else "")
            })

    # Compute error rates per length slice
    slice_results = {}
    for bucket_key, counts in length_buckets.items():
        tot = counts["total"]
        errs = counts["errors"]
        rate = round(errs / max(tot, 1), 4) if tot > 0 else 0.0
        slice_results[bucket_key] = {
            "total_documents": tot,
            "error_count": errs,
            "error_rate": rate
        }

    # 3. Confidence breakdown of errors
    low_conf_errors = sum(1 for e in sample_errors if e["confidence"] < 0.60)
    high_conf_errors = sum(1 for e in sample_errors if e["confidence"] >= 0.80)

    # Sort sample errors: most overconfident errors first
    sample_errors.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "total_test_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
        "accuracy": round(1.0 - overall_error_rate, 4),
        "confused_class_pairs": confused_pairs,
        "length_slice_metrics": slice_results,
        "confidence_breakdown": {
            "low_confidence_errors_under_60pct": low_conf_errors,
            "overconfident_errors_over_80pct": high_conf_errors,
            "medium_confidence_errors_60_to_80pct": total_errors - low_conf_errors - high_conf_errors
        },
        "sample_errors": sample_errors[:15]  # Top 15 representative error cases
    }
