"""
Feature Extraction and Engineering Module (TF-IDF)
Classifying Text Documents Using Machine Learning (4 Classes)

Implements TF-IDF vectorization with strict featurization ordering
(fit on training data only) to strictly prevent data leakage.
Supports:
- Unigram and bigram tokenization
- Configurable vocabulary sizing (max_features)
- Frequency thresholding (min_df, max_df)
- Sublinear term-frequency dampening (1 + log(tf))
- L1/L2 vector normalization
- Cross-validated feature engineering parameter experimentation on training set
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: bool = True,
    norm: str = 'l2'
) -> TfidfVectorizer:
    """
    Constructs a configured TfidfVectorizer.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum vocabulary size to retain the most frequent informative n-grams.
    ngram_range : tuple of (int, int), default=(1, 2)
        Extract unigrams and/or bigrams.
    min_df : int or float, default=2
        Ignore terms that appear in fewer than min_df documents (filters rare noise).
    max_df : float, default=1.0
        Ignore terms that appear in more than max_df portion of documents (corpus-wide stops).
    sublinear_tf : bool, default=True
        Apply sublinear scaling 1 + log(tf) to dampen the effect of very frequent words.
    norm : {'l1', 'l2', None}, default='l2'
        Vector norm.

    Returns
    -------
    TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm
    )


def extract_features(
    vectorizer: TfidfVectorizer,
    X_train_raw: pd.Series,
    X_test_raw: pd.Series
) -> Tuple[Any, Any, TfidfVectorizer]:
    """
    Extracts TF-IDF features strictly adhering to ML best practices:
    - Fits the vectorizer ONLY on the training corpus (X_train_raw).
    - Transforms testing corpus without fitting to avoid data leakage.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
    X_train_raw : pd.Series of preprocessed training texts.
    X_test_raw : pd.Series of preprocessed testing texts.

    Returns
    -------
    X_train_tfidf, X_test_tfidf, vectorizer
    """
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    return X_train_tfidf, X_test_tfidf, vectorizer


def experiment_tfidf_configurations(
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Systematically benchmarks multiple TF-IDF configurations using stratified
    cross-validation on the training data ONLY. Never peeks at the test set.

    Evaluates:
    - Unigram vs Unigram+Bigram
    - Vocabulary size variations (3000, 5000, 10000)
    - Sublinear TF vs standard linear TF
    - Min-DF filtering (1, 2, 5)

    Returns
    -------
    pd.DataFrame containing CV mean score, std dev, and parameter settings.
    """
    experiments = [
        {
            "name": "Unigram Baseline",
            "ngram_range": (1, 1),
            "max_features": 5000,
            "min_df": 2,
            "sublinear_tf": True,
            "norm": "l2"
        },
        {
            "name": "Unigram + Bigram (Optimal 5K)",
            "ngram_range": (1, 2),
            "max_features": 5000,
            "min_df": 2,
            "sublinear_tf": True,
            "norm": "l2"
        },
        {
            "name": "High-Capacity Bigram (10K)",
            "ngram_range": (1, 2),
            "max_features": 10000,
            "min_df": 2,
            "sublinear_tf": True,
            "norm": "l2"
        },
        {
            "name": "Compact Bigram (3K)",
            "ngram_range": (1, 2),
            "max_features": 3000,
            "min_df": 2,
            "sublinear_tf": True,
            "norm": "l2"
        },
        {
            "name": "Linear TF (No Sublinear Scaling)",
            "ngram_range": (1, 2),
            "max_features": 5000,
            "min_df": 2,
            "sublinear_tf": False,
            "norm": "l2"
        },
        {
            "name": "High Min-DF Filter (min_df=5)",
            "ngram_range": (1, 2),
            "max_features": 5000,
            "min_df": 5,
            "sublinear_tf": True,
            "norm": "l2"
        }
    ]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for exp in experiments:
        vec = build_tfidf_vectorizer(
            max_features=exp["max_features"],
            ngram_range=exp["ngram_range"],
            min_df=exp["min_df"],
            sublinear_tf=exp["sublinear_tf"],
            norm=exp["norm"]
        )

        fold_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            X_tr_tfidf = vec.fit_transform(X_tr_fold)
            X_val_tfidf = vec.transform(X_val_fold)

            clf = MultinomialNB(alpha=0.1)
            clf.fit(X_tr_tfidf, y_tr_fold)
            score = clf.score(X_val_tfidf, y_val_fold)
            fold_scores.append(score)

        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))

        results.append({
            "Configuration": exp["name"],
            "N-Gram Range": str(exp["ngram_range"]),
            "Max Features": exp["max_features"],
            "Min DF": exp["min_df"],
            "Sublinear TF": exp["sublinear_tf"],
            "CV Mean Accuracy": round(mean_acc, 4),
            "CV Std Dev": round(std_acc, 4)
        })

    df_results = pd.DataFrame(results).sort_values(by="CV Mean Accuracy", ascending=False).reset_index(drop=True)
    return df_results


def get_top_tfidf_terms_for_document(
    vectorizer: TfidfVectorizer,
    text_tfidf_vector,
    top_n: int = 8
) -> List[Tuple[str, float]]:
    """
    Returns the top N terms with the highest TF-IDF weights for a given document vector.
    """
    if not hasattr(vectorizer, "vocabulary_") or len(vectorizer.vocabulary_) == 0:
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(text_tfidf_vector, "toarray"):
        row = text_tfidf_vector.toarray().flatten()
    else:
        row = np.array(text_tfidf_vector).flatten()

    if len(row) == 0:
        return []

    top_indices = np.argsort(row)[::-1][:top_n]
    results = []
    for idx in top_indices:
        weight = float(row[idx])
        if weight > 0:
            results.append((str(feature_names[idx]), round(weight, 4)))
    return results


def get_top_features_per_category(
    vectorizer: TfidfVectorizer,
    X_tfidf,
    y: pd.Series,
    target_names: List[str],
    top_n: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes average TF-IDF weights across all documents in each category
    and extracts the top distinguishing terms.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    y_array = np.array(y)
    results = {}

    for class_idx, class_name in enumerate(target_names):
        mask = (y_array == class_idx)
        if not np.any(mask):
            continue
        class_docs = X_tfidf[mask]
        mean_weights = np.asarray(class_docs.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_weights)[::-1][:top_n]

        top_terms = [
            (str(feature_names[idx]), round(float(mean_weights[idx]), 4))
            for idx in top_indices
        ]
        results[class_name] = top_terms

    return results


def explain_tfidf() -> str:
    """
    Returns a comprehensive mathematical and conceptual explanation of TF-IDF.
    """
    return """
### Understanding TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF is a statistical numerical statistic designed to reflect how important a word is to a document in a collection or corpus.

1. **Term Frequency (TF)**:
   Measures how frequently a term $t$ occurs in document $d$:
   $$\\text{TF}(t, d) = \\frac{\\text{count of } t \\text{ in } d}{\\text{total terms in } d}$$
   With sublinear scaling: $\\text{TF}_{\\text{sublinear}} = 1 + \\log(\\text{TF}(t, d))$ for $\\text{TF} > 0$.

2. **Inverse Document Frequency (IDF)**:
   Measures the informational value of the word across the entire corpus of $N$ documents:
   $$\\text{IDF}(t) = \\log\\left(\\frac{1 + N}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
   Common words appear across all documents and receive low IDF. Rare domain-specific terms (e.g., "orbit", "pitcher", "gpu") receive high IDF.

3. **TF-IDF Weight**:
   $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$
   Normalized using the $L_2$ norm:
   $$v_{\\text{norm}} = \\frac{v}{\\|v\\|_2}$$
   This balances document lengths and yields dense, highly separable features for linear and probabilistic classifiers.
"""
