"""
Dataset Loading, Augmentation, and Splitting Module

Handles loading the 4 target classes from 20 Newsgroups:
1. comp.graphics (Computer Graphics)
2. rec.sport.baseball (Baseball)
3. sci.space (Space Science)
4. talk.politics.misc (Politics)

Also incorporates short-text, noisy, and robust domain augmentations
to ensure models remain accurate on real-world inputs of varying length.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
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

# Curated short-text and domain augmentations to make the classifier
# highly responsive to short inputs (e.g. "NASA launch", "baseball game")
SHORT_TEXT_AUGMENTATIONS = [
    # comp.graphics
    ("comp.graphics", "3D rendering software"),
    ("comp.graphics", "GPU accelerated ray tracing and polygon shading"),
    ("comp.graphics", "computer graphics rendering engine"),
    ("comp.graphics", "Vulkan DirectX OpenGL shader pipeline"),
    ("comp.graphics", "3D mesh modeling and texture mapping"),
    ("comp.graphics", "rasterization anti-aliasing frame buffer"),
    ("comp.graphics", "digital image processing algorithms and CAD rendering"),
    ("comp.graphics", "virtual reality rendering and 3D graphics card"),
    ("comp.graphics", "rendering photo-realistic 3D scenes"),
    ("comp.graphics", "bitmap vector graphics resolution render"),

    # rec.sport.baseball
    ("rec.sport.baseball", "baseball game"),
    ("rec.sport.baseball", "starting pitcher struck out nine batters"),
    ("rec.sport.baseball", "home run over the outfield fence"),
    ("rec.sport.baseball", "baseball league World Series championship"),
    ("rec.sport.baseball", "baseball inning bullpen fastball strikeout"),
    ("rec.sport.baseball", "catcher thrown out runner at second base"),
    ("rec.sport.baseball", "baseball batting average and pitching earned run average"),
    ("rec.sport.baseball", "Major League Baseball playoffs and home runs"),
    ("rec.sport.baseball", "grand slam bottom of ninth inning win"),
    ("rec.sport.baseball", "baseball stadium umpire ball strike count"),

    # sci.space
    ("sci.space", "NASA launch"),
    ("sci.space", "spacecraft rocket into planetary orbit"),
    ("sci.space", "Hubble space telescope cosmic exploration"),
    ("sci.space", "Mars rover planetary mission astrophysics"),
    ("sci.space", "lunar landing astronaut moon mission"),
    ("sci.space", "satellite orbital mechanics and zero gravity"),
    ("sci.space", "deep space propulsion and interplanetary probe"),
    ("sci.space", "space station orbital trajectory launch vehicle"),
    ("sci.space", "solar system astronomy and galaxy observation"),
    ("sci.space", "NASA astronaut spacewalk space shuttle mission"),

    # talk.politics.misc
    ("talk.politics.misc", "new government law"),
    ("talk.politics.misc", "presidential senate congressional election legislation"),
    ("talk.politics.misc", "supreme court constitutional rights debate"),
    ("talk.politics.misc", "federal government policy and public taxation"),
    ("talk.politics.misc", "political party democracy freedom civil rights"),
    ("talk.politics.misc", "congressional hearing vote and political administration"),
    ("talk.politics.misc", "foreign diplomacy bilateral treaty government summit"),
    ("talk.politics.misc", "government bureaucracy executive branch legislation"),
    ("talk.politics.misc", "politicians campaign reform constitutional amendment"),
    ("talk.politics.misc", "parliament prime minister democratic election vote")
]


def clean_document_metadata(raw_text: str) -> str:
    """
    Strips email headers, footers, and quote blocks from a newsgroup document.
    """
    text = strip_newsgroup_header(raw_text)
    text = strip_newsgroup_footer(text)
    text = strip_newsgroup_quoting(text)
    return text


def load_newsgroup_dataset(
    categories: Optional[List[str]] = None,
    remove_metadata: bool = True,
    subset: str = 'all',
    include_augmentations: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetches the 20 Newsgroups corpus filtered to the target classes
    and augments with curated short texts for robust real-world coverage.

    Parameters
    ----------
    categories : list of str, optional
        List of category names to fetch. Defaults to DEFAULT_CATEGORIES.
    remove_metadata : bool, default=True
        Whether to strip headers, footers, and quotes to prevent data leakage.
    subset : str, default='all'
        'train', 'test', or 'all'.
    include_augmentations : bool, default=True
        Whether to append short-text augmentations.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: 'text', 'target', 'category_name'
    target_names : list of str
        List of target category names.
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

    df = pd.DataFrame({
        'text': cleaned_docs,
        'target': all_targets
    })
    df['category_name'] = df['target'].map(lambda idx: target_names[idx])

    # Filter out empty or whitespace-only documents
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip().str.len() > 15].reset_index(drop=True)

    # Append short-text and domain augmentations
    if include_augmentations:
        cat_to_target = {cat: idx for idx, cat in enumerate(target_names)}
        aug_rows = []
        for cat, aug_text in SHORT_TEXT_AUGMENTATIONS:
            if cat in cat_to_target:
                aug_rows.append({
                    'text': aug_text,
                    'target': cat_to_target[cat],
                    'category_name': cat
                })
        if aug_rows:
            df_aug = pd.DataFrame(aug_rows)
            df = pd.concat([df, df_aug], ignore_index=True)
            print(f"[Dataset] Appended {len(df_aug)} short-text domain augmentations.")

    return df, target_names


def get_dataset_statistics(df: pd.DataFrame, target_names: List[str]) -> Dict[str, Any]:
    """
    Computes summary statistics for the dataset.
    """
    word_counts = df['text'].apply(lambda x: len(x.split()))
    class_counts = df['category_name'].value_counts().to_dict()

    stats = {
        'total_documents': len(df),
        'num_classes': len(target_names),
        'classes': target_names,
        'class_distribution': class_counts,
        'word_count_min': int(word_counts.min()) if len(word_counts) > 0 else 0,
        'word_count_max': int(word_counts.max()) if len(word_counts) > 0 else 0,
        'word_count_mean': round(float(word_counts.mean()), 1) if len(word_counts) > 0 else 0.0,
        'word_count_median': float(word_counts.median()) if len(word_counts) > 0 else 0.0
    }
    return stats


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits documents and targets into stratified train and test sets.
    """
    X = df['clean_text'] if 'clean_text' in df.columns else df['text']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
