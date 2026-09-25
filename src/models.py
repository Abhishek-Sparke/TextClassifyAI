"""
Machine Learning Models & Controlled Hyperparameter Tuning Module
Classifying Text Documents Using Machine Learning (4 Classes)

Defines, tunes, and trains 7 text classifiers optimized for sparse TF-IDF feature spaces:
1. Multinomial Naive Bayes (MultinomialNB)
2. Complement Naive Bayes (ComplementNB)
3. Logistic Regression (LogisticRegression)
4. Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV)
5. Stochastic Gradient Descent Classifier (SGDClassifier with log_loss)
6. Ridge Classifier (RidgeClassifier with CalibratedClassifierCV)
7. Random Forest Classifier (RandomForestClassifier)

Includes controlled GridSearchCV hyperparameter tuning on training data only.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes and returns 7 classification models with balanced,
    production-grade default hyperparameters.
    """
    # 1. Multinomial Naive Bayes
    nb = MultinomialNB(alpha=0.1)

    # 2. Complement Naive Bayes (specifically designed for text classification)
    cnb = ComplementNB(alpha=0.1, norm=False)

    # 3. Logistic Regression
    lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=500,
        solver='lbfgs',
        random_state=random_state
    )

    # 4. Linear Support Vector Machine with Platt scaling
    base_svm = LinearSVC(C=1.0, class_weight='balanced', random_state=random_state, dual='auto')
    svm = CalibratedClassifierCV(estimator=base_svm, cv=3)

    # 5. SGDClassifier (Logistic loss for high-speed sparse gradient descent)
    sgd = SGDClassifier(
        loss='log_loss',
        alpha=1e-4,
        penalty='l2',
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state
    )

    # 6. Ridge Classifier with probability calibration
    base_ridge = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=random_state)
    ridge = CalibratedClassifierCV(estimator=base_ridge, cv=3)

    # 7. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )

    return {
        "Multinomial Naive Bayes": nb,
        "Complement Naive Bayes": cnb,
        "Logistic Regression": lr,
        "Linear SVM": svm,
        "SGDClassifier": sgd,
        "Ridge Classifier": ridge,
        "Random Forest": rf
    }


def get_hyperparameter_grids(random_state: int = 42) -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Returns base estimators and parameter search grids for controlled hyperparameter tuning.
    Focuses only on high-impact parameters suitable for sparse text features.
    """
    grids = {
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
        ),
        "Complement Naive Bayes": (
            ComplementNB(),
            {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0], "norm": [False, True]}
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=500, class_weight='balanced', random_state=random_state, solver='lbfgs'),
            {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
        ),
        "SGDClassifier": (
            SGDClassifier(loss='log_loss', max_iter=1000, class_weight='balanced', random_state=random_state),
            {"alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], "penalty": ["l2", "elasticnet"]}
        ),
        "Ridge Classifier": (
            RidgeClassifier(class_weight='balanced', random_state=random_state),
            {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
        ),
        "Linear SVM": (
            LinearSVC(class_weight='balanced', random_state=random_state, dual='auto'),
            {"C": [0.1, 0.5, 1.0, 2.0]}
        )
    }
    return grids


def tune_and_train_models(
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
    enable_tuning: bool = True
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Performs controlled hyperparameter tuning on training data using Stratified 5-Fold
    GridSearchCV, then wraps non-probabilistic models with calibration and fits them.

    Never peeks at the test set.

    Returns
    -------
    trained_models : dict of fitted models
    training_times : dict of elapsed training/tuning time in seconds
    best_params : dict of discovered optimal parameters
    """
    trained_models = {}
    training_times = {}
    best_params = {}

    if not enable_tuning:
        models = get_default_models(random_state=random_state)
        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            dur = time.time() - t0
            trained_models[name] = model
            training_times[name] = round(dur, 3)
            best_params[name] = "Default"
        return trained_models, training_times, best_params

    grids = get_hyperparameter_grids(random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, (base_est, param_grid) in grids.items():
        print(f"  --> Tuning {name} with {cv}-Fold Stratified CV...")
        t0 = time.time()

        grid_search = GridSearchCV(
            estimator=base_est,
            param_grid=param_grid,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train, y_train)
        best_est = grid_search.best_estimator_
        best_p = grid_search.best_params_
        best_params[name] = best_p
        print(f"      Best Parameters: {best_p} (CV F1: {grid_search.best_score_:.4f})")

        # For Linear SVM and Ridge, wrap tuned estimator with CalibratedClassifierCV
        # to ensure well-calibrated class probabilities for OOD and ambiguity detection
        if name == "Linear SVM":
            calibrated_svm = CalibratedClassifierCV(estimator=best_est, cv=3)
            calibrated_svm.fit(X_train, y_train)
            trained_models[name] = calibrated_svm
        elif name == "Ridge Classifier":
            calibrated_ridge = CalibratedClassifierCV(estimator=best_est, cv=3)
            calibrated_ridge.fit(X_train, y_train)
            trained_models[name] = calibrated_ridge
        else:
            trained_models[name] = best_est

        dur = time.time() - t0
        training_times[name] = round(dur, 3)
        print(f"      Tuning & Fitting completed in {dur:.2f}s.")

    # Train Random Forest (baseline ensemble)
    print("  --> Training Random Forest Baseline...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_dur = time.time() - t0
    trained_models["Random Forest"] = rf
    training_times["Random Forest"] = round(rf_dur, 3)
    best_params["Random Forest"] = {"n_estimators": 100, "max_depth": 25}
    print(f"      Random Forest completed in {rf_dur:.2f}s.")

    return trained_models, training_times, best_params


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Backwards compatibility alias for get_default_models.
    """
    return get_default_models(random_state=random_state)


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
        exp_d = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)
    else:
        preds = model.predict(X_vector)
        num_classes = len(getattr(model, "classes_", [0, 1, 2, 3]))
        probs = np.zeros((len(preds), num_classes))
        for i, p in enumerate(preds):
            probs[i, int(p)] = 1.0
        return probs
