"""
Text Preprocessing Module

Provides text normalization, noise cleaning, stop-word removal,
tokenization, and lemmatization/stemming for text classification.
"""

import re
import string
from typing import List, Optional

# Attempt to load NLTK resources with graceful fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    # Verify stopwords availability locally
    try:
        nltk.data.find('corpora/stopwords')
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP_WORDS = set(ENGLISH_STOP_WORDS)

    # Verify WordNet availability locally, otherwise fallback to PorterStemmer
    try:
        nltk.data.find('corpora/wordnet')
        LEMMATIZER = WordNetLemmatizer()
    except LookupError:
        LEMMATIZER = None

    STEMMER = PorterStemmer()
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
    LEMMATIZER = None
    STEMMER = None


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text by:
    1. Handling null/missing or non-string inputs.
    2. Converting text to lowercase.
    3. Stripping URLs, emails, and header artifacts.
    4. Removing punctuation, digits, and special characters.
    5. Normalizing multiple whitespace characters to single spaces.

    Parameters
    ----------
    text : str or None
        Raw input text document.

    Returns
    -------
    str
        Cleaned text string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove email header signatures like 'writes:', 'subject:', etc.
    text = re.sub(r"(from|subject|re|lines|organization|writes):", " ", text)

    # 5. Remove numbers/digits (optional, keeps words focused on semantic content)
    text = re.sub(r"\b\d+\b", " ", text)

    # 6. Remove punctuation and non-alphanumeric characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 7. Remove non-ascii characters / symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 8. Collapse whitespace and strip borders
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str, apply_lemmatization: bool = True) -> str:
    """
    Tokenizes text, filters stop words and short tokens, and applies
    lemmatization (or stemming as fallback).

    Parameters
    ----------
    text : str
        Pre-cleaned text string.
    apply_lemmatization : bool
        Whether to apply lemmatization on tokens.

    Returns
    -------
    str
        Processed string of space-separated normalized lemmas.
    """
    if not text:
        return ""

    # Tokenize using regex word boundaries (fast, robust, avoids tokenizer crashes)
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    filtered_tokens: List[str] = []
    for token in tokens:
        # Filter stop words
        if token in STOP_WORDS:
            continue

        # Lemmatize or stem
        if apply_lemmatization and LEMMATIZER is not None:
            try:
                token = LEMMATIZER.lemmatize(token)
            except Exception:
                pass
        elif apply_lemmatization and STEMMER is not None:
            try:
                token = STEMMER.stem(token)
            except Exception:
                pass

        filtered_tokens.append(token)

    return " ".join(filtered_tokens)


def preprocess_document(text: Optional[str], apply_lemmatization: bool = True) -> str:
    """
    Full text preprocessing pipeline for an individual document:
    Handling nulls -> Lowercase -> Cleaning -> Tokenizing -> Stopwords -> Lemmatization.

    Parameters
    ----------
    text : str or None
        Raw input document text.
    apply_lemmatization : bool
        Whether to apply lemmatization.

    Returns
    -------
    str
        Processed clean text ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    processed = tokenize_and_lemmatize(cleaned, apply_lemmatization=apply_lemmatization)
    return processed


def preprocess_corpus(texts: List[str], apply_lemmatization: bool = True) -> List[str]:
    """
    Applies the preprocessing pipeline to a collection (list) of documents.
    Utilizes multi-threaded batch parallelization for large corpora (e.g. 100,000 docs).

    Parameters
    ----------
    texts : list of str
        List of raw document strings.
    apply_lemmatization : bool
        Whether to apply lemmatization.

    Returns
    -------
    list of str
        List of preprocessed text documents.
    """
    if len(texts) > 5000:
        try:
            from joblib import Parallel, delayed
            return Parallel(n_jobs=-1, batch_size=250, prefer="threads")(
                delayed(preprocess_document)(doc, apply_lemmatization) for doc in texts
            )
        except Exception:
            pass
    return [preprocess_document(doc, apply_lemmatization=apply_lemmatization) for doc in texts]
