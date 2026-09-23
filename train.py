"""
End-to-End Training and Evaluation Pipeline
Classifying Text Documents Using Machine Learning (4 Classes)

Executes dataset loading, robust text preprocessing, TF-IDF feature extraction,
training of 4 classifiers, evaluation, latency measurement, visualization generation,
and model persistence with joblib.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np

from src.dataset import (
    load_newsgroup_dataset,
    get_dataset_statistics,
    split_data,
    DEFAULT_CATEGORIES,
    CATEGORY_DISPLAY_NAMES
)
from src.preprocessing import preprocess_corpus
from src.features import (
    build_tfidf_vectorizer,
    extract_features,
    get_top_features_per_category
)
from src.models import get_models, train_all_models
from src.evaluate import (
    evaluate_all_models,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_class_distribution,
    plot_top_keywords
)
from src.inference_engine import DEFAULT_INFERENCE_CONFIG


def run_pipeline():
    print("=" * 80)
    print(" PROJECT: Classifying Text Documents Using Machine Learning (4 Classes)")
    print(" Classes: comp.graphics, rec.sport.baseball, sci.space, talk.politics.misc")
    print("=" * 80)

    # 1. Dataset Loading
    print("\n[Step 1/7] Loading Dataset (4 Core Classes from 20 Newsgroups)...")
    df, target_names = load_newsgroup_dataset(
        categories=DEFAULT_CATEGORIES,
        remove_metadata=True,
        include_augmentations=True
    )
    stats = get_dataset_statistics(df, target_names)

    print(f" -> Total Documents: {stats['total_documents']}")
    print(f" -> Number of Classes: {stats['num_classes']}")
    print(f" -> Classes: {stats['classes']}")
    print("\n -> Class Breakdown:")
    for cat, count in stats['class_distribution'].items():
        disp = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        print(f"    * {cat} ({disp}): {count} documents")

    # 2. Text Preprocessing
    print("\n[Step 2/7] Preprocessing Text Data...")
    print(" -> Steps: Lowercasing, Noise/URL/Email Removal, Alphanumeric Preservation (3D, GPU), Lemmatization")
    start_prep = time.time()
    df['clean_text'] = preprocess_corpus(df['text'].tolist(), apply_lemmatization=True)
    prep_duration = time.time() - start_prep
    print(f" -> Preprocessing completed in {prep_duration:.2f} seconds.")

    non_empty_mask = df['clean_text'].str.strip().str.len() > 0
    df = df[non_empty_mask].reset_index(drop=True)
    print(f" -> Retained {len(df)} documents with non-empty vocabulary.")

    # 3. Stratified Train / Test Split
    print("\n[Step 3/7] Splitting Dataset (Stratified 80% Train, 20% Test)...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df,
        test_size=0.2,
        random_state=42
    )
    print(f" -> Training Samples: {len(X_train_raw)}")
    print(f" -> Testing Samples:  {len(X_test_raw)}")

    # 4. TF-IDF Feature Extraction
    print("\n[Step 4/7] TF-IDF Feature Extraction...")
    print(" -> Strict Featurization: Fitting vectorizer ONLY on training set to prevent data leakage.")
    tfidf_config = {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": True,
        "norm": "l2"
    }
    vectorizer = build_tfidf_vectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        sublinear_tf=tfidf_config["sublinear_tf"]
    )
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(vectorizer, X_train_raw, X_test_raw)
    vocab_size = len(vectorizer.vocabulary_)
    print(f" -> Vocabulary Size (Unigrams & Bigrams): {vocab_size} features")
    print(f" -> X_train_tfidf Shape: {X_train_tfidf.shape}")
    print(f" -> X_test_tfidf Shape:  {X_test_tfidf.shape}")

    top_features_per_class = get_top_features_per_category(
        vectorizer,
        X_train_tfidf,
        y_train,
        target_names,
        top_n=10
    )

    # 5. Machine Learning Models Training
    print("\n[Step 5/7] Initializing and Training 4 Classifiers...")
    models = get_models(random_state=42)
    trained_models, training_times = train_all_models(models, X_train_tfidf, y_train)

    # 6. Model Evaluation & Comparison
    print("\n[Step 6/7] Evaluating Classifiers on Test Set...")
    comparison_df, eval_results, best_model_name = evaluate_all_models(
        trained_models,
        X_test_tfidf,
        y_test,
        target_names,
        training_times
    )

    print("\n" + "=" * 80)
    print(" MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    print(f"\n >>> BEST PERFORMING MODEL: {best_model_name} <<<")
    best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    print(f"     Accuracy: {best_row['Accuracy']:.4f} | F1-Score (Weighted): {best_row['F1-Score (Weighted)']:.4f} | Latency: {best_row['Latency (ms/doc)']:.3f} ms/doc")

    # 7. Generate Visualizations & Save Artifacts
    print("\n[Step 7/7] Saving Visualizations and Model Artifacts...")
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    plot_class_distribution(df, target_names, 'visualizations/class_distribution.png')
    plot_model_comparison(comparison_df, 'visualizations/model_comparison.png')
    plot_confusion_matrices(eval_results, target_names, 'visualizations/confusion_matrices.png')
    plot_top_keywords(top_features_per_class, 'visualizations/top_keywords.png')
    print(" -> Saved 4 visualization figures in 'visualizations/' directory.")

    joblib.dump(trained_models[best_model_name], 'models/best_model.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(trained_models, 'models/all_models.joblib')

    metadata = {
        'best_model_name': best_model_name,
        'selection_metric': 'F1-Score (Weighted) primary, Accuracy secondary',
        'categories': target_names,
        'category_display_names': CATEGORY_DISPLAY_NAMES,
        'dataset_statistics': stats,
        'tfidf_config': tfidf_config,
        'preprocessing_config': {
            'lowercase': True,
            'strip_urls': True,
            'strip_emails': True,
            'remove_standalone_digits': True,
            'preserve_alphanumeric': ['3d', 'gpu', 'nasa', 'cad', 'law'],
            'lemmatization': True
        },
        'confidence_thresholds': {
            'confidence_threshold': DEFAULT_INFERENCE_CONFIG.confidence_threshold,
            'min_active_tfidf': DEFAULT_INFERENCE_CONFIG.min_active_tfidf,
            'ambiguity_margin': DEFAULT_INFERENCE_CONFIG.ambiguity_margin,
            'min_ambiguity_sum': DEFAULT_INFERENCE_CONFIG.min_ambiguity_sum,
            'topic_detection_threshold': DEFAULT_INFERENCE_CONFIG.topic_detection_threshold
        },
        'metrics_table': comparison_df.to_dict(orient='records'),
        'top_features_per_class': top_features_per_class,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(" -> Saved model artifacts in 'models/' directory:")
    print("    * models/best_model.joblib")
    print("    * models/tfidf_vectorizer.joblib")
    print("    * models/all_models.joblib")
    print("    * models/model_metadata.json")

    print("\n" + "=" * 80)
    print(" TRAINING & EVALUATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
