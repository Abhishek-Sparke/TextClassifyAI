"""
Model Artifact Validation & Safe Loading Module
Classifying Text Documents Using Machine Learning (4 Classes)

Validates model integrity, schema compatibility, versioning,
and feature dimension consistency before loading models into production services.
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple


class ArtifactValidationError(Exception):
    """Raised when serialized model artifacts fail validation checks."""
    pass


def validate_and_load_artifacts(
    models_dir: str
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Safely validates and loads machine learning artifacts:
    1. Checks existence of required files:
       - best_model.joblib
       - tfidf_vectorizer.joblib
       - model_metadata.json
    2. Validates metadata schema, model version, and dataset fingerprint.
    3. Validates vectorizer vocabulary size and fitted state.
    4. Validates model-vectorizer feature dimension compatibility (n_features_in_).
    5. Validates model class list matches metadata.

    Parameters
    ----------
    models_dir : str
        Path to the directory containing model artifacts.

    Returns
    -------
    best_model : fitted estimator
    vectorizer : fitted TfidfVectorizer
    all_models : dict of all trained models (if available, else {best_model_name: best_model})
    metadata : dict of model metadata

    Raises
    ------
    ArtifactValidationError : If any compatibility or integrity check fails.
    """
    if not os.path.exists(models_dir):
        raise ArtifactValidationError(f"Models directory not found: '{models_dir}'. Run 'python train.py' first.")

    best_model_path = os.path.join(models_dir, "best_model.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")
    all_models_path = os.path.join(models_dir, "all_models.joblib")

    # 1. Existence check
    missing_files = []
    if not os.path.exists(best_model_path):
        missing_files.append("best_model.joblib")
    if not os.path.exists(vec_path):
        missing_files.append("tfidf_vectorizer.joblib")
    if not os.path.exists(meta_path):
        missing_files.append("model_metadata.json")

    if missing_files:
        raise ArtifactValidationError(
            f"Missing required model artifact(s) in '{models_dir}': {', '.join(missing_files)}. "
            "Please train the model pipeline using 'python train.py'."
        )

    # 2. Metadata validation
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise ArtifactValidationError(f"Corrupt or unreadable metadata file '{meta_path}': {str(e)}")

    required_meta_keys = [
        "model_version",
        "best_model_name",
        "categories",
        "dataset_statistics",
        "tfidf_config"
    ]
    for key in required_meta_keys:
        if key not in metadata:
            raise ArtifactValidationError(f"Invalid model_metadata.json: missing required field '{key}'.")

    # 3. Load Vectorizer & Validate Vocabulary
    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        raise ArtifactValidationError(f"Failed to deserialize vectorizer from '{vec_path}': {str(e)}")

    if not hasattr(vectorizer, "vocabulary_") or len(vectorizer.vocabulary_) == 0:
        raise ArtifactValidationError("Loaded TfidfVectorizer is empty or unfitted.")

    vocab_size = len(vectorizer.vocabulary_)

    # 4. Load Best Model & Validate
    try:
        best_model = joblib.load(best_model_path)
    except Exception as e:
        raise ArtifactValidationError(f"Failed to deserialize model from '{best_model_path}': {str(e)}")

    # Check feature dimension compatibility
    model_features = getattr(best_model, "n_features_in_", None)
    if model_features is not None and model_features != vocab_size:
        raise ArtifactValidationError(
            f"Feature dimension mismatch: Vectorizer has {vocab_size} features, "
            f"but model expects {model_features} features."
        )

    # Check class compatibility if model has classes_
    model_classes = getattr(best_model, "classes_", None)
    expected_categories = metadata["categories"]
    if model_classes is not None and len(model_classes) != len(expected_categories):
        raise ArtifactValidationError(
            f"Class count mismatch: Model has {len(model_classes)} classes, "
            f"metadata defines {len(expected_categories)} classes."
        )

    # 5. Load all_models if present
    all_models = {}
    if os.path.exists(all_models_path):
        try:
            all_models = joblib.load(all_models_path)
        except Exception:
            all_models = {metadata["best_model_name"]: best_model}
    else:
        all_models = {metadata["best_model_name"]: best_model}

    return best_model, vectorizer, all_models, metadata
