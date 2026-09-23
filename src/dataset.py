"""
Dataset Loading and Splitting Module

Handles loading the 20 Newsgroups dataset, cleaning metadata artifacts,
computing dataset summary statistics, and performing stratified train-test splits.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_files, get_data_home
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_header,
    strip_newsgroup_footer,
    strip_newsgroup_quoting
)
from sklearn.model_selection import train_test_split

from src.data_loader_100k import (
    ALL_26_CATEGORIES,
    CATEGORY_DISPLAY_NAMES_26,
    build_and_cache_100k_dataset
)

# 26 Categories: 20 Newsgroups + Business/Finance, World News, Entertainment, Health, Education, Environment
DEFAULT_CATEGORIES = ALL_26_CATEGORIES
CATEGORY_DISPLAY_NAMES = CATEGORY_DISPLAY_NAMES_26


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
    subset: str = 'all'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetches or loads the multi-domain text classification dataset (100,000 documents, 26 classes).

    Parameters
    ----------
    categories : list of str, optional
        List of category names to fetch. If None, uses DEFAULT_CATEGORIES (26 categories).
    remove_metadata : bool, default=True
        Whether to strip headers, footers, and quotes to prevent data leakage.
    subset : str, default='all'
        'train', 'test', or 'all' to load complete data for customized splitting.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: 'text', 'target', 'category_name'
    target_names : list of str
        List of target class names.
    """
    selected_cats = categories if categories is not None else DEFAULT_CATEGORIES

    # Check if multi-domain 100k dataset covers the requested categories
    if set(selected_cats).issubset(set(ALL_26_CATEGORIES)):
        print(f"[Dataset] Loading multi-domain 100k dataset for {len(selected_cats)} categories...")
        df_100k = build_and_cache_100k_dataset()
        if len(selected_cats) < len(ALL_26_CATEGORIES):
            df_100k = df_100k[df_100k['category_name'].isin(selected_cats)].reset_index(drop=True)
            # Reindex targets
            cat_map = {cat: idx for idx, cat in enumerate(selected_cats)}
            df_100k['target'] = df_100k['category_name'].map(cat_map)
        return df_100k, selected_cats

    # Check for fast local directory
    data_home = get_data_home()
    train_dir = os.path.join(data_home, '20news_home', '20news-bydate-train')
    test_dir = os.path.join(data_home, '20news_home', '20news-bydate-test')

    all_texts = []
    all_targets = []
    target_names = selected_cats

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"[Dataset] Loading '{len(selected_cats)}' categories from local repository cache...")
        if subset in ('train', 'all'):
            b_train = load_files(train_dir, categories=selected_cats, encoding='latin1')
            all_texts.extend(b_train.data)
            all_targets.extend(b_train.target)
            target_names = list(b_train.target_names)

        if subset in ('test', 'all'):
            b_test = load_files(test_dir, categories=selected_cats, encoding='latin1')
            all_texts.extend(b_test.data)
            all_targets.extend(b_test.target)
            target_names = list(b_test.target_names)
    else:
        print("[Dataset] Fetching 20 Newsgroups via scikit-learn...")
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

    # Strip metadata if requested and loaded from files
    if remove_metadata:
        cleaned_docs = [clean_document_metadata(doc) for doc in all_texts]
    else:
        cleaned_docs = all_texts

    df = pd.DataFrame({
        'text': cleaned_docs,
        'target': all_targets
    })

    # Map target integer to string category name
    df['category_name'] = df['target'].map(lambda idx: target_names[idx])

    # Filter out empty or whitespace-only documents that appear after stripping metadata
    df['text'] = df['text'].astype(str)
    initial_len = len(df)
    df = df[df['text'].str.strip().str.len() > 10].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[Dataset] Filtered {dropped} blank/header-only documents after stripping metadata.")

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
        'word_count_min': int(word_counts.min()),
        'word_count_max': int(word_counts.max()),
        'word_count_mean': float(word_counts.mean()),
        'word_count_median': float(word_counts.median())
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
    # Use preprocessed text if available to ensure vocabulary contains meaningful terms
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
