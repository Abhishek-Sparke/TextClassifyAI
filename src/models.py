"""
Machine Learning Models Module

Defines and initializes the 4 classification algorithms:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (Linear SVM with probability calibration)
4. Random Forest Classifier
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes and returns the 4 classification models with balanced,
    production-ready hyperparameters.

    Parameters
    ----------
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    dict
        Dictionary mapping model names to scikit-learn estimator instances.
    """
    # 1. Multinomial Naive Bayes
    # Fast probabilistic classifier; alpha=0.1 provides optimal smoothing for sparse TF-IDF text
    nb = MultinomialNB(alpha=0.1)

    # 2. Logistic Regression
    # Softmax / multiclass regression with L2 regularization
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state
    )

    # 3. Support Vector Machine (SVM)
    # CalibratedClassifierCV wraps LinearSVC with Platt scaling / isotonic regression
    # to provide calibrated class probabilities (predict_proba) for the UI
    base_svm = LinearSVC(C=1.0, random_state=random_state, dual='auto')
    svm = CalibratedClassifierCV(estimator=base_svm, cv=3)

    # 4. Random Forest Classifier
    # Non-linear ensemble model with 150 estimators
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        random_state=random_state,
        n_jobs=-1
    )

    return {
        "Multinomial Naive Bayes": nb,
        "Logistic Regression": lr,
        "Support Vector Machine": svm,
        "Random Forest": rf
    }


def train_single_model(model, X_train, y_train) -> Tuple[Any, float]:
    """
    Fits a single model and records elapsed training time in seconds.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    return model, elapsed_time


def train_all_models(
    models: Dict[str, Any],
    X_train,
    y_train
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Trains all models in the dictionary.

    Parameters
    ----------
    models : dict
        Model name -> estimator.
    X_train : sparse matrix or ndarray
    y_train : pd.Series or ndarray

    Returns
    -------
    trained_models : dict
        Model name -> fitted estimator.
    training_times : dict
        Model name -> elapsed seconds.
    """
    trained_models = {}
    training_times = {}

    for name, model in models.items():
        print(f"  --> Training {name}...")
        fitted_model, duration = train_single_model(model, X_train, y_train)
        trained_models[name] = fitted_model
        training_times[name] = round(duration, 3)
        print(f"      Finished in {duration:.3f} seconds.")

    return trained_models, training_times


def get_prediction_probabilities(model, X_vector) -> np.ndarray:
    """
    Extracts class probability distribution for an input vector.
    Falls back to softmax over decision_function if predict_proba is unavailable.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_vector)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X_vector)
        # Apply numerically stable softmax
        exp_d = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)
    else:
        # One-hot fallback
        preds = model.predict(X_vector)
        probs = np.zeros((len(preds), len(model.classes_)))
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        return probs
