"""
Text Preprocessing Module

Provides text normalization, noise cleaning, stop-word removal,
tokenization, and lemmatization/stemming for text classification.
Preserves meaningful technical and domain tokens like '3d', 'cad', 'gpu', 'nasa'.
"""

import re
import string
from typing import List, Optional

# Attempt to load NLTK resources with graceful fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    try:
        nltk.data.find('corpora/stopwords')
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP_WORDS = set(ENGLISH_STOP_WORDS)

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

# Ensure domain-critical words are never accidentally filtered
PRESERVED_WORDS = {"3d", "2d", "4k", "cad", "gpu", "cpu", "nasa", "mars", "law", "hit", "run", "win"}
STOP_WORDS = STOP_WORDS - PRESERVED_WORDS

# Add standard non-topical conversational words that carry zero domain signal
EXTRA_STOP_WORDS = {
    "today", "yesterday", "tomorrow", "day", "week", "month", "year",
    "went", "got", "ate", "eat", "eating", "bought", "buy", "buying",
    "shopping", "shop", "shopped", "pizza", "favorite", "amazing",
    "good", "bad", "thing", "things", "like", "really", "going", "goes"
}
STOP_WORDS = STOP_WORDS | EXTRA_STOP_WORDS


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text by:
    1. Handling null/missing or non-string inputs.
    2. Converting text to lowercase.
    3. Stripping URLs, emails, and header artifacts.
    4. Removing punctuation, standalone numbers, and non-ascii noise.
    5. Normalizing multiple whitespace characters.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove email header signatures
    text = re.sub(r"\b(from|subject|re|lines|organization|writes|article):", " ", text)

    # 5. Remove standalone digits while preserving '3d', '4k'
    text = re.sub(r"\b\d+\b", " ", text)

    # 6. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 7. Remove non-ascii characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 8. Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str, apply_lemmatization: bool = True) -> str:
    """
    Tokenizes text, filters stop words and noise, and applies lemmatization.
    """
    if not text:
        return ""

    tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text)

    filtered_tokens: List[str] = []
    for token in tokens:
        if token.isdigit():
            continue

        if token in STOP_WORDS and token not in PRESERVED_WORDS:
            continue

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
    Full text preprocessing pipeline for an individual document.
    """
    cleaned = clean_text(text)
    processed = tokenize_and_lemmatize(cleaned, apply_lemmatization=apply_lemmatization)
    return processed


def preprocess_corpus(texts: List[str], apply_lemmatization: bool = True) -> List[str]:
    """
    Applies the preprocessing pipeline to a collection of documents.
    """
    return [preprocess_document(doc, apply_lemmatization=apply_lemmatization) for doc in texts]
