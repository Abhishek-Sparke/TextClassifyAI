"""
Machine Learning Models Module

Defines, initializes, and provides probability inference for 4 classifiers:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (Linear SVM with CalibratedClassifierCV)
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
    production-grade hyperparameters.

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
    nb = MultinomialNB(alpha=0.1)

    # 2. Logistic Regression with balanced class weights
    lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=400,
        solver='lbfgs',
        random_state=random_state,
        n_jobs=-1
    )

    # 3. Support Vector Machine (SVM)
    # CalibratedClassifierCV wraps LinearSVC with Platt scaling
    # to yield calibrated class probabilities for out-of-domain/ambiguity detection
    base_svm = LinearSVC(C=1.0, class_weight='balanced', random_state=random_state, dual='auto')
    svm = CalibratedClassifierCV(estimator=base_svm, cv=3)

    # 4. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        class_weight='balanced',
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
    Falls back to numerically stable softmax over decision_function if predict_proba is unavailable.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_vector)
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(X_vector)
        # Apply numerically stable softmax
        exp_d = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)
    else:
        preds = model.predict(X_vector)
        num_classes = len(getattr(model, "classes_", [0, 1, 2, 3]))
        probs = np.zeros((len(preds), num_classes))
        for i, p in enumerate(preds):
            probs[i, int(p)] = 1.0
        return probs
