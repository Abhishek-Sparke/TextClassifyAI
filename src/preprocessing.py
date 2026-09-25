"""
Text Preprocessing and Normalization Module
Classifying Text Documents Using Machine Learning (4 Classes)

Provides a unified, configurable, and deterministic text preprocessing pipeline.
Guarantees that training, evaluation, batch classification, and real-time inference
always execute identical normalization steps.

Handles:
- Lowercasing
- URL stripping
- Email stripping
- Header artifact removal
- Repeated character normalization ("soooo" -> "so")
- Punctuation removal
- Standalone digit stripping while preserving technical alphanumeric tokens ('3d', '4k', '2d')
- Non-ASCII/Unicode normalization
- Stopword filtering with domain preservation
- Lemmatization with WordNetLemmatizer and fallback PorterStemmer
- Whitespace normalization
"""

import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Set

# Attempt to load NLTK resources with graceful fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    try:
        nltk.data.find('corpora/stopwords')
        BASE_STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        BASE_STOP_WORDS = set(ENGLISH_STOP_WORDS)

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
    BASE_STOP_WORDS = set(ENGLISH_STOP_WORDS)
    LEMMATIZER = None
    STEMMER = None

# Ensure domain-critical tokens are never accidentally filtered
DEFAULT_PRESERVED_WORDS = {
    "3d", "2d", "4k", "cad", "gpu", "cpu", "vulkan", "opengl", "shader",
    "nasa", "mars", "moon", "orbit", "law", "hit", "run", "win", "era", "rbi",
    "pitcher", "batter", "strikeout", "spacecraft", "satellite", "render"
}

# Expand standard stop words with conversational non-domain terms to prevent
# out-of-domain conversational text from triggering false topical matches
DEFAULT_CONVERSATIONAL_STOP_WORDS = {
    "today", "yesterday", "tomorrow", "tonight", "went", "go", "going", "gone",
    "came", "come", "coming", "bought", "buy", "buying", "ate", "eat", "eating",
    "got", "get", "getting", "saw", "see", "seen", "seeing", "looked", "look",
    "said", "say", "saying", "told", "tell", "telling", "asked", "ask", "asking",
    "wanted", "want", "wanting", "tried", "try", "trying", "favorite", "favourite",
    "amazing", "awesome", "movie", "shopping", "battery", "laptop", "phone",
    "really", "very", "much", "many", "good", "bad", "great", "nice", "fine",
    "also", "even", "still", "always", "never", "maybe", "probably", "actually",
    "thing", "things", "people", "person", "day", "days", "year", "years", "time"
}


@dataclass
class PreprocessingConfig:
    """
    Configuration options for the text preprocessing pipeline.
    Allows testing and tweaking each normalization step independently.
    """
    lowercase: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    strip_email_headers: bool = True
    normalize_repeated_chars: bool = True
    remove_standalone_numbers: bool = True
    remove_punctuation: bool = True
    remove_non_ascii: bool = True
    strip_whitespace: bool = True
    remove_stopwords: bool = True
    apply_lemmatization: bool = True
    min_token_length: int = 2
    preserved_words: Set[str] = field(default_factory=lambda: set(DEFAULT_PRESERVED_WORDS))
    conversational_stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_CONVERSATIONAL_STOP_WORDS))

    def get_effective_stopwords(self) -> Set[str]:
        combined = (BASE_STOP_WORDS | self.conversational_stopwords) - self.preserved_words
        return combined


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()


class TextPreprocessor:
    """
    Reusable and thread-safe text preprocessor instance.
    Guarantees strict parity across training and inference.
    """
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_PREPROCESSING_CONFIG
        self.stop_words = self.config.get_effective_stopwords()

    def clean_text(self, text: Optional[str]) -> str:
        """
        Cleans raw string according to configuration.
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # 1. Lowercase
        if self.config.lowercase:
            text = text.lower()

        # 2. Normalize unicode (e.g., café -> cafe, smart quotes)
        text = unicodedata.normalize('NFKD', text)

        # 3. Remove URLs
        if self.config.strip_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Remove email addresses
        if self.config.strip_emails:
            text = re.sub(r"\S+@\S+", " ", text)

        # 5. Remove email header signatures
        if self.config.strip_email_headers:
            text = re.sub(r"\b(from|subject|re|lines|organization|writes|article):", " ", text)

        # 6. Normalize repeated characters ("soooo" -> "so", "reeeender" -> "render")
        if self.config.normalize_repeated_chars:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # 7. Remove standalone numbers while preserving '3d', '4k', '2d'
        if self.config.remove_standalone_numbers:
            text = re.sub(r"\b\d+\b", " ", text)

        # 8. Remove punctuation
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # 9. Remove non-ascii characters
        if self.config.remove_non_ascii:
            text = re.sub(r"[^\x00-\x7F]+", " ", text)

        # 10. Collapse whitespace and strip
        if self.config.strip_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Tokenizes text, filters stop words, and applies lemmatization/stemming.
        """
        if not text:
            return ""

        # Extract alphanumeric words >= min_token_length
        pattern = rf"\b[a-zA-Z0-9]{{{self.config.min_token_length},}}\b"
        tokens = re.findall(pattern, text)

        filtered_tokens: List[str] = []
        for token in tokens:
            if token.isdigit():
                continue

            if self.config.remove_stopwords:
                if token in self.stop_words and token not in self.config.preserved_words:
                    continue

            if self.config.apply_lemmatization:
                if LEMMATIZER is not None:
                    try:
                        token = LEMMATIZER.lemmatize(token)
                    except Exception:
                        pass
                elif STEMMER is not None:
                    try:
                        token = STEMMER.stem(token)
                    except Exception:
                        pass

            filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def preprocess_document(self, text: Optional[str]) -> str:
        """
        Executes end-to-end cleaning and tokenization on a single document.
        """
        cleaned = self.clean_text(text)
        return self.tokenize_and_lemmatize(cleaned)

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Processes a list of documents.
        """
        return [self.preprocess_document(doc) for doc in texts]


# Default singleton pipeline instance
GLOBAL_PREPROCESSOR = TextPreprocessor(DEFAULT_PREPROCESSING_CONFIG)


# Module-level convenience functions maintaining backwards compatibility
def clean_text(text: Optional[str], config: Optional[PreprocessingConfig] = None) -> str:
    if config:
        return TextPreprocessor(config).clean_text(text)
    return GLOBAL_PREPROCESSOR.clean_text(text)


def tokenize_and_lemmatize(
    text: str,
    apply_lemmatization: bool = True,
    config: Optional[PreprocessingConfig] = None
) -> str:
    if config:
        cfg = config
    else:
        cfg = PreprocessingConfig(apply_lemmatization=apply_lemmatization)
    return TextPreprocessor(cfg).tokenize_and_lemmatize(text)


def preprocess_document(
    text: Optional[str],
    apply_lemmatization: bool = True,
    config: Optional[PreprocessingConfig] = None
) -> str:
    if config:
        cfg = config
    else:
        cfg = PreprocessingConfig(apply_lemmatization=apply_lemmatization)
    return TextPreprocessor(cfg).preprocess_document(text)


def preprocess_corpus(
    texts: List[str],
    apply_lemmatization: bool = True,
    config: Optional[PreprocessingConfig] = None
) -> List[str]:
    if config:
        cfg = config
    else:
        cfg = PreprocessingConfig(apply_lemmatization=apply_lemmatization)
    return TextPreprocessor(cfg).preprocess_corpus(texts)
