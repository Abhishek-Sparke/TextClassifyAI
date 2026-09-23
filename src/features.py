"""
Feature Extraction Module (TF-IDF)

Implements TF-IDF vectorization with strict featurization ordering
(fit on training data only) to strictly prevent data leakage.
Supports unigrams, bigrams, sublinear TF scaling, configurable max_features,
and L2 normalization.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Constructs a configured TfidfVectorizer.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum vocabulary size to retain the most frequent informative n-grams.
    ngram_range : tuple of (int, int), default=(1, 2)
        Extract both unigrams and bigrams (e.g., 'space' and 'space station').
    min_df : int, default=2
        Ignore terms that appear in fewer than min_df documents (removes rare noise).
    sublinear_tf : bool, default=True
        Apply sublinear scaling 1 + log(tf) to dampen the effect of very frequent words.

    Returns
    -------
    TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        norm='l2'
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


def get_top_tfidf_terms_for_document(
    vectorizer: TfidfVectorizer,
    text_tfidf_vector,
    top_n: int = 8
) -> List[Tuple[str, float]]:
    """
    Returns the top N terms with the highest TF-IDF weights for a given document vector.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer.
    text_tfidf_vector : sparse matrix or 1D array
        TF-IDF vector for a single document.
    top_n : int, default=8
        Number of top terms to return.

    Returns
    -------
    list of (term, weight) tuples
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
   Common words (e.g., "the", "is", "article") appear across all documents and receive low IDF. Rare domain-specific terms (e.g., "orbit", "pitcher", "gpu") receive high IDF.

3. **TF-IDF Weight**:
   $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$
   Normalized using the $L_2$ norm:
   $$v_{\\text{norm}} = \\frac{v}{\\|v\\|_2}$$
   This balances document lengths and yields dense, highly separable features for linear and probabilistic classifiers.
"""
