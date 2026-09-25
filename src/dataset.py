"""
Dataset Loading, Validation, Auditing, and Splitting Module
Classifying Text Documents Using Machine Learning (4 Target Classes)

Handles loading the 4 target classes from 20 Newsgroups:
1. comp.graphics (Computer Graphics)
2. rec.sport.baseball (Baseball)
3. sci.space (Space Science)
4. talk.politics.misc (Politics)

Incorporates:
- Comprehensive dataset validation (empty, short, long, nulls, invalid labels)
- Exact and normalized duplicate detection and audited removal
- Train/Test data leakage verification
- Cryptographic dataset fingerprinting (SHA-256)
- Statistical profiling & vocabulary measurement
- Curated short-text, noisy, and domain augmentations
"""

import hashlib
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_header,
    strip_newsgroup_footer,
    strip_newsgroup_quoting
)
from sklearn.model_selection import train_test_split

DEFAULT_CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc"
]

CATEGORY_DISPLAY_NAMES = {
    "comp.graphics": "Computer Graphics",
    "rec.sport.baseball": "Baseball",
    "sci.space": "Space Science",
    "talk.politics.misc": "Politics"
}

CATEGORY_ICONS = {
    "comp.graphics": "🎨",
    "rec.sport.baseball": "⚾",
    "sci.space": "🚀",
    "talk.politics.misc": "🏛️"
}

# Curated short-text, noisy, and domain augmentations to make the classifier
# highly responsive to short inputs and noisy real-world text
SHORT_TEXT_AUGMENTATIONS = [
    # comp.graphics
    ("comp.graphics", "3D rendering software"),
    ("comp.graphics", "The computer generated a 3D image."),
    ("comp.graphics", "GPU accelerated ray tracing and polygon shading"),
    ("comp.graphics", "computer graphics rendering engine"),
    ("comp.graphics", "Vulkan DirectX OpenGL shader pipeline"),
    ("comp.graphics", "3D mesh modeling and texture mapping"),
    ("comp.graphics", "rasterization anti-aliasing frame buffer"),
    ("comp.graphics", "digital image processing algorithms and CAD rendering"),
    ("comp.graphics", "virtual reality rendering and 3D graphics card"),
    ("comp.graphics", "rendering photo-realistic 3D scenes"),
    ("comp.graphics", "bitmap vector graphics resolution render"),
    ("comp.graphics", "3D 3D 3D render render!! email questions to support@3dgraphics.org or call 1-800-555-0199 🎨"),
    ("comp.graphics", "GPU 3D graphic rendering 4K resolution!! https://graphics.org #3DRender 🎨"),
    ("comp.graphics", "My graphics card is overheating while rendering 3D shaders."),
    ("comp.graphics", "CAD computer graphics pipeline rendering polygon vertices and ray tracing"),

    # rec.sport.baseball
    ("rec.sport.baseball", "baseball game"),
    ("rec.sport.baseball", "The baseball team scored five runs."),
    ("rec.sport.baseball", "starting pitcher struck out nine batters"),
    ("rec.sport.baseball", "home run over the outfield fence"),
    ("rec.sport.baseball", "baseball league World Series championship"),
    ("rec.sport.baseball", "baseball inning bullpen fastball strikeout"),
    ("rec.sport.baseball", "catcher thrown out runner at second base"),
    ("rec.sport.baseball", "baseball batting average and pitching earned run average"),
    ("rec.sport.baseball", "Major League Baseball playoffs and home runs"),
    ("rec.sport.baseball", "grand slam bottom of ninth inning win"),
    ("rec.sport.baseball", "baseball stadium umpire ball strike count"),
    ("rec.sport.baseball", "THE PITCHER STRUCK OUT 10 BATTERS IN INNING 9!!! BASEBALL CHAMPIONSHIP WON ⚾⚾"),
    ("rec.sport.baseball", "baseball game score: 5-4 in bottom of 9th inning! http://mlb-live.com #Baseball"),
    ("rec.sport.baseball", "starting pitcher recorded 12 strikeouts and allowed zero walks in the baseball game"),

    # sci.space
    ("sci.space", "NASA launch"),
    ("sci.space", "NASA launched a spacecraft into orbit."),
    ("sci.space", "spacecraft rocket into planetary orbit"),
    ("sci.space", "Hubble space telescope cosmic exploration"),
    ("sci.space", "Mars rover planetary mission astrophysics"),
    ("sci.space", "lunar landing astronaut moon mission"),
    ("sci.space", "satellite orbital mechanics and zero gravity"),
    ("sci.space", "deep space propulsion and interplanetary probe"),
    ("sci.space", "space station orbital trajectory launch vehicle"),
    ("sci.space", "solar system astronomy and galaxy observation"),
    ("sci.space", "NASA astronaut spacewalk space shuttle mission"),
    ("sci.space", "NASA launched a probe!!! Check it out at https://nasa.gov/mission?id=99283 #SpaceExploration 🚀🚀"),
    ("sci.space", "Can NASA launch the spacecraft tomorrow into orbital trajectory?"),
    ("sci.space", "Mars rover discovered signs of water on the planetary surface"),

    # talk.politics.misc
    ("talk.politics.misc", "new government law"),
    ("talk.politics.misc", "The government passed a new law."),
    ("talk.politics.misc", "presidential senate congressional election legislation"),
    ("talk.politics.misc", "supreme court constitutional rights debate"),
    ("talk.politics.misc", "federal government policy and public taxation"),
    ("talk.politics.misc", "political party democracy freedom civil rights"),
    ("talk.politics.misc", "congressional hearing vote and political administration"),
    ("talk.politics.misc", "foreign diplomacy bilateral treaty government summit"),
    ("talk.politics.misc", "government bureaucracy executive branch legislation"),
    ("talk.politics.misc", "politicians campaign reform constitutional amendment"),
    ("talk.politics.misc", "parliament prime minister democratic election vote"),
    ("talk.politics.misc", "BREAKING: Congressional debate over government legislation & civil rights!! https://gov.org/law 🏛️"),
    ("talk.politics.misc", "federal government passed a new constitutional law regarding taxation and voting rights"),
    ("talk.politics.misc", "political debate over presidential administration and foreign policy")
]


def clean_document_metadata(raw_text: str) -> str:
    """
    Strips email headers, footers, and quote blocks from a newsgroup document
    to prevent metadata leakage.
    """
    if not isinstance(raw_text, str):
        return ""
    text = strip_newsgroup_header(raw_text)
    text = strip_newsgroup_footer(text)
    text = strip_newsgroup_quoting(text)
    return text


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Generates a deterministic SHA-256 cryptographic hash of the dataset
    based on normalized text strings and target labels.
    Ensures verifiable data provenance and version tracking.
    """
    hasher = hashlib.sha256()
    sorted_df = df.sort_values(by=['text', 'target']).reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        sample_str = f"{str(row['text']).strip()}||{row['target']}\n"
        hasher.update(sample_str.encode('utf-8', errors='ignore'))
    return hasher.hexdigest()[:16]


def validate_and_audit_dataset(
    df: pd.DataFrame,
    target_names: List[str],
    min_length_chars: int = 15,
    max_length_chars: int = 50000,
    deduplicate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data pipeline validation and quality audit.
    Identifies and logs:
    - Missing / null labels
    - Invalid target labels
    - Empty and whitespace-only documents
    - Extremely short documents (< min_length_chars)
    - Extremely long documents (> max_length_chars)
    - Exact duplicate documents
    - Normalized duplicate documents

    Parameters
    ----------
    df : pd.DataFrame with 'text' and 'target'
    target_names : list of valid category names
    min_length_chars : minimum acceptable length
    max_length_chars : maximum acceptable length before flagging/trimming
    deduplicate : whether to remove duplicate documents

    Returns
    -------
    audited_df : pd.DataFrame
    audit_report : dict
    """
    initial_total = len(df)
    audit_log = []

    # 1. Missing Label Detection
    null_labels_mask = df['target'].isna()
    null_labels_count = int(null_labels_mask.sum())
    if null_labels_count > 0:
        audit_log.append(f"Removed {null_labels_count} documents with missing (null/NaN) target labels.")
        df = df[~null_labels_mask].copy()

    # 2. Invalid Label Detection
    valid_targets = set(range(len(target_names)))
    invalid_labels_mask = ~df['target'].isin(valid_targets)
    invalid_labels_count = int(invalid_labels_mask.sum())
    if invalid_labels_count > 0:
        audit_log.append(f"Removed {invalid_labels_count} documents with invalid target indices outside range [0, {len(target_names)-1}].")
        df = df[~invalid_labels_mask].copy()

    # 3. Empty & Missing Text Detection
    null_text_mask = df['text'].isna()
    empty_text_mask = df['text'].astype(str).str.strip().str.len() == 0
    empty_total_mask = null_text_mask | empty_text_mask
    empty_count = int(empty_total_mask.sum())
    if empty_count > 0:
        audit_log.append(f"Removed {empty_count} documents with empty, null, or whitespace-only content.")
        df = df[~empty_total_mask].copy()

    # 4. Extremely Short Documents
    short_mask = df['text'].astype(str).str.strip().str.len() < min_length_chars
    short_count = int(short_mask.sum())
    if short_count > 0:
        audit_log.append(f"Removed {short_count} extremely short documents (< {min_length_chars} characters) lacking semantic domain signal.")
        df = df[~short_mask].copy()

    # 5. Extremely Long Documents (flagged and safely capped to max_length_chars)
    long_mask = df['text'].astype(str).str.len() > max_length_chars
    long_count = int(long_mask.sum())
    if long_count > 0:
        audit_log.append(f"Detected {long_count} extremely long documents (> {max_length_chars} characters). Truncated to {max_length_chars} chars to prevent denial of service and memory spikes.")
        df.loc[long_mask, 'text'] = df.loc[long_mask, 'text'].astype(str).str.slice(0, max_length_chars)

    # 6. Duplicate Detection
    exact_duplicates_count = int(df.duplicated(subset=['text']).sum())
    normalized_texts = df['text'].astype(str).str.lower().str.strip()
    normalized_duplicates_count = int(normalized_texts.duplicated().sum())

    if deduplicate and normalized_duplicates_count > 0:
        audit_log.append(f"Removed {normalized_duplicates_count} duplicate documents (retaining first occurrence).")
        df = df.loc[~normalized_texts.duplicated()].copy()

    df = df.reset_index(drop=True)
    final_total = len(df)

    audit_report = {
        "initial_documents": initial_total,
        "final_documents": final_total,
        "documents_removed": initial_total - final_total,
        "null_labels_found": null_labels_count,
        "invalid_labels_found": invalid_labels_count,
        "empty_documents_found": empty_count,
        "extremely_short_found": short_count,
        "extremely_long_found": long_count,
        "exact_duplicates_found": exact_duplicates_count,
        "normalized_duplicates_found": normalized_duplicates_count,
        "audit_log": audit_log
    }

    return df, audit_report


def check_train_test_leakage(
    X_train: pd.Series,
    X_test: pd.Series
) -> Dict[str, Any]:
    """
    Checks for exact and normalized document leakage between train and test splits.
    Ensures strict validation integrity.
    """
    train_set = set(X_train.astype(str).str.strip().str.lower())
    test_set = set(X_test.astype(str).str.strip().str.lower())
    overlap = train_set.intersection(test_set)

    return {
        "has_leakage": len(overlap) > 0,
        "overlap_count": len(overlap),
        "overlap_sample": list(overlap)[:5],
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


def load_newsgroup_dataset(
    categories: Optional[List[str]] = None,
    remove_metadata: bool = True,
    subset: str = 'all',
    include_augmentations: bool = True,
    deduplicate: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Fetches the 20 Newsgroups corpus filtered to target classes,
    performs quality validation, strips headers/footers/quotes to prevent leakage,
    augments with curated short texts, and computes provenance fingerprint.

    Returns
    -------
    df : pd.DataFrame with 'text', 'target', 'category_name'
    target_names : list of category names
    audit_report : dict of data cleaning audit findings
    """
    selected_cats = categories if categories is not None else DEFAULT_CATEGORIES

    print(f"[Dataset] Loading 20 Newsgroups for classes: {selected_cats}...")
    remove = ('headers', 'footers', 'quotes') if remove_metadata else ()
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=selected_cats,
        shuffle=True,
        random_state=42,
        remove=remove
    )

    all_texts = bunch.data
    all_targets = bunch.target
    target_names = list(bunch.target_names)

    cleaned_docs = [clean_document_metadata(doc) for doc in all_texts]

    raw_df = pd.DataFrame({
        'text': cleaned_docs,
        'target': all_targets
    })

    # Validate and audit raw corpus
    audited_df, audit_report = validate_and_audit_dataset(
        raw_df,
        target_names,
        min_length_chars=15,
        max_length_chars=50000,
        deduplicate=deduplicate
    )

    # Append short-text and domain augmentations
    if include_augmentations:
        cat_to_target = {cat: idx for idx, cat in enumerate(target_names)}
        aug_rows = []
        for cat, aug_text in SHORT_TEXT_AUGMENTATIONS:
            if cat in cat_to_target:
                aug_rows.append({
                    'text': aug_text,
                    'target': cat_to_target[cat]
                })
        if aug_rows:
            df_aug = pd.DataFrame(aug_rows)
            audited_df = pd.concat([audited_df, df_aug], ignore_index=True)
            audit_report["augmentations_added"] = len(df_aug)
            audit_report["audit_log"].append(f"Appended {len(df_aug)} short-text and domain augmentations to enhance real-world responsiveness.")
            print(f"[Dataset] Appended {len(df_aug)} short-text domain augmentations.")

    audited_df['category_name'] = audited_df['target'].map(lambda idx: target_names[idx])
    audited_df['fingerprint'] = compute_dataset_fingerprint(audited_df)

    return audited_df, target_names, audit_report


def get_dataset_statistics(df: pd.DataFrame, target_names: List[str]) -> Dict[str, Any]:
    """
    Computes summary and profiling statistics for the dataset:
    - total documents
    - unique documents
    - duplicate count
    - documents per class
    - character and word length statistics (mean, min, max, median, std)
    - approximate vocabulary size
    - dataset SHA-256 fingerprint
    """
    char_lengths = df['text'].astype(str).str.len()
    word_counts = df['text'].apply(lambda x: len(str(x).split()))
    class_counts = df['category_name'].value_counts().to_dict()

    # Compute total unique vocabulary terms (simple whitespace split)
    unique_words = set()
    for doc in df['text'].astype(str):
        for token in doc.lower().split():
            clean_tok = "".join(ch for ch in token if ch.isalnum())
            if len(clean_tok) > 1:
                unique_words.add(clean_tok)

    fingerprint = df['fingerprint'].iloc[0] if 'fingerprint' in df.columns else compute_dataset_fingerprint(df)
    unique_docs = int(df['text'].nunique())
    duplicate_docs = len(df) - unique_docs

    stats = {
        'total_documents': len(df),
        'unique_documents': unique_docs,
        'duplicate_count': duplicate_docs,
        'num_classes': len(target_names),
        'classes': target_names,
        'class_distribution': class_counts,
        'fingerprint': fingerprint,
        'vocabulary_size': len(unique_words),
        'char_length_min': int(char_lengths.min()) if len(char_lengths) > 0 else 0,
        'char_length_max': int(char_lengths.max()) if len(char_lengths) > 0 else 0,
        'char_length_mean': round(float(char_lengths.mean()), 1) if len(char_lengths) > 0 else 0.0,
        'word_count_min': int(word_counts.min()) if len(word_counts) > 0 else 0,
        'word_count_max': int(word_counts.max()) if len(word_counts) > 0 else 0,
        'word_count_mean': round(float(word_counts.mean()), 1) if len(word_counts) > 0 else 0.0,
        'word_count_median': float(word_counts.median()) if len(word_counts) > 0 else 0.0,
        'word_count_std': round(float(word_counts.std()), 1) if len(word_counts) > 0 else 0.0
    }
    return stats


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    enforce_deduplication: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits documents and targets into stratified train and test sets.
    Verifies that no test documents leak into training data.
    """
    col = 'clean_text' if 'clean_text' in df.columns else 'text'
    work_df = df
    if enforce_deduplication and work_df.duplicated(subset=[col]).any():
        work_df = work_df.drop_duplicates(subset=[col]).reset_index(drop=True)

    X = work_df[col]
    y = work_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    leakage_info = check_train_test_leakage(X_train, X_test)
    if leakage_info["has_leakage"]:
        raise ValueError(f"CRITICAL: Train/Test leakage detected! {leakage_info['overlap_count']} overlapping documents found.")

    return X_train, X_test, y_train, y_test
