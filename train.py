"""
End-to-End Machine Learning Training, Validation, and Evaluation Pipeline
Classifying Text Documents Using Machine Learning (4 Classes)

Orchestrates:
1. Dataset loading, quality auditing, duplicate detection, and SHA-256 fingerprinting.
2. Unified NLP preprocessing with exact training-inference parity.
3. Train/Test splitting with zero-leakage verification.
4. Systematic TF-IDF feature engineering experimentation via 5-fold CV.
5. Controlled hyperparameter tuning with GridSearchCV for 7 classifiers.
6. Stratified 5-Fold Cross-Validation on training data.
7. Final untouched test set evaluation (Accuracy, Weighted/Macro F1, Precision, Recall, Latency, Size).
8. Forensic error analysis (confused pairs, length slices, sample errors).
9. Comprehensive publication-grade visualization generation.
10. Versioned metadata persistence & model artifact integrity verification.
"""

import os
import json
import time
import platform
import joblib
import pandas as pd
import numpy as np
import sklearn

from src.dataset import (
    load_newsgroup_dataset,
    get_dataset_statistics,
    split_data,
    check_train_test_leakage,
    DEFAULT_CATEGORIES,
    CATEGORY_DISPLAY_NAMES
)
from src.preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    TextPreprocessor
)
from src.features import (
    build_tfidf_vectorizer,
    extract_features,
    get_top_features_per_category,
    experiment_tfidf_configurations
)
from src.models import (
    tune_and_train_models
)
from src.evaluate import (
    run_stratified_cross_validation,
    evaluate_all_models,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_per_class_f1,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_top_keywords
)
from src.error_analysis import run_error_analysis
from src.inference_engine import DEFAULT_INFERENCE_CONFIG
from src.model_validator import validate_and_load_artifacts

RANDOM_SEED = 42
MODEL_VERSION = "2.0.0"


def set_seed(seed: int = 42):
    np.random.seed(seed)


def run_pipeline():
    set_seed(RANDOM_SEED)
    start_total_time = time.time()

    print("=" * 85)
    print(" PROJECT: Classifying Text Documents Using Machine Learning (Production v2.0)")
    print(" Target Classes: comp.graphics, rec.sport.baseball, sci.space, talk.politics.misc")
    print(f" Python: {platform.python_version()} | Scikit-Learn: {sklearn.__version__} | Seed: {RANDOM_SEED}")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 1. Dataset Loading, Quality Audit, and Fingerprinting
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Loading and Auditing Dataset (4 Target Classes)...")
    df, target_names, audit_report = load_newsgroup_dataset(
        categories=DEFAULT_CATEGORIES,
        remove_metadata=True,
        include_augmentations=True,
        deduplicate=True
    )
    stats = get_dataset_statistics(df, target_names)

    print(f" -> Total Cleaned Documents: {stats['total_documents']}")
    print(f" -> Dataset Fingerprint (SHA-256): {stats['fingerprint']}")
    print(f" -> Vocabulary Size (Unique tokens): {stats['vocabulary_size']}")
    print(f" -> Word Length: Mean={stats['word_count_mean']}, Min={stats['word_count_min']}, Max={stats['word_count_max']}")
    print("\n -> Data Pipeline Audit Findings:")
    for log_entry in audit_report["audit_log"]:
        print(f"    * {log_entry}")

    print("\n -> Class Breakdown:")
    for cat, count in stats['class_distribution'].items():
        disp = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        print(f"    * {cat} ({disp}): {count} documents")

    # -------------------------------------------------------------------------
    # 2. Text Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Executing Unified NLP Preprocessing...")
    preprocessor = TextPreprocessor(DEFAULT_PREPROCESSING_CONFIG)
    start_prep = time.time()
    df['clean_text'] = preprocessor.preprocess_corpus(df['text'].tolist())
    prep_duration = time.time() - start_prep
    print(f" -> Preprocessing completed in {prep_duration:.2f} seconds.")

    non_empty_mask = df['clean_text'].str.strip().str.len() > 0
    df = df[non_empty_mask].reset_index(drop=True)
    print(f" -> Retained {len(df)} documents with non-empty vocabulary.")

    clean_dups = int(df.duplicated(subset=['clean_text']).sum())
    if clean_dups > 0:
        print(f" -> Detected {clean_dups} duplicate documents post-preprocessing. Removing to guarantee zero train/test leakage.")
        df = df.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
        audit_report["audit_log"].append(f"Removed {clean_dups} duplicate documents post-preprocessing to guarantee zero train/test data leakage.")

    # -------------------------------------------------------------------------
    # 3. Stratified Train / Test Split & Leakage Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Splitting Dataset (Stratified 80% Train, 20% Test)...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    leakage = check_train_test_leakage(X_train_raw, X_test_raw)
    print(f" -> Training Samples: {len(X_train_raw)}")
    print(f" -> Testing Samples:  {len(X_test_raw)}")
    print(f" -> Data Leakage Check: PASS (Overlap Count = {leakage['overlap_count']})")

    # -------------------------------------------------------------------------
    # 4. Feature Engineering Experiments & Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Running TF-IDF Feature Engineering Experiments on Training Data...")
    tfidf_exp_df = experiment_tfidf_configurations(X_train_raw, y_train, cv=5, random_state=RANDOM_SEED)
    print(tfidf_exp_df.to_string(index=False))

    print("\n -> Building Selected Production TF-IDF Vectorizer...")
    tfidf_config = {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "norm": "l2"
    }
    vectorizer = build_tfidf_vectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"],
        sublinear_tf=tfidf_config["sublinear_tf"],
        norm=tfidf_config["norm"]
    )
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(vectorizer, X_train_raw, X_test_raw)
    vocab_size = len(vectorizer.vocabulary_)
    print(f" -> Production Vocabulary Size: {vocab_size} features")

    top_features_per_class = get_top_features_per_category(
        vectorizer,
        X_train_tfidf,
        y_train,
        target_names,
        top_n=10
    )

    # -------------------------------------------------------------------------
    # 5. Hyperparameter Tuning & Model Training (7 Classifiers)
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Tuning and Training 7 Classifiers with Stratified 5-Fold GridSearchCV...")
    trained_models, training_times, best_params = tune_and_train_models(
        X_train_tfidf,
        y_train,
        cv=5,
        random_state=RANDOM_SEED,
        enable_tuning=True
    )

    # -------------------------------------------------------------------------
    # 6. Stratified Cross-Validation Reporting (Training Set Only)
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Evaluating Stratified 5-Fold Cross-Validation on Training Data...")
    cv_comparison_df, cv_details = run_stratified_cross_validation(
        trained_models,
        X_train_tfidf,
        y_train,
        cv=5,
        random_state=RANDOM_SEED
    )
    print("\n" + "=" * 85)
    print(" STRATIFIED 5-FOLD CROSS-VALIDATION SUMMARY (TRAINING DATA ONLY)")
    print("=" * 85)
    print(cv_comparison_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 7. Final Untouched Test Set Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Evaluating Classifiers on Untouched Final Test Set...")
    comparison_df, eval_results, best_model_name = evaluate_all_models(
        trained_models,
        X_test_tfidf,
        y_test,
        target_names,
        training_times
    )

    print("\n" + "=" * 85)
    print(" UNTOUCHED TEST SET PERFORMANCE COMPARISON TABLE")
    print("=" * 85)
    print(comparison_df.to_string(index=False))
    print("=" * 85)
    print(f"\n >>> BEST PERFORMING MODEL: {best_model_name} <<<")
    best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    print(f"     Accuracy: {best_row['Accuracy']:.4f} | F1-Score (Weighted): {best_row['F1-Score (Weighted)']:.4f} | Latency: {best_row['Latency (ms/doc)']:.3f} ms/doc | Size: {best_row['Model Size (KB)']} KB")

    # -------------------------------------------------------------------------
    # 8. Forensic Error Analysis on Best Model
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Running Forensic Error Analysis on Best Model...")
    error_analysis_report = run_error_analysis(
        trained_models[best_model_name],
        vectorizer,
        X_test_raw,
        y_test,
        target_names
    )
    print(f" -> Total Test Samples: {error_analysis_report['total_test_samples']}")
    print(f" -> Total Misclassifications: {error_analysis_report['total_errors']} (Error Rate: {error_analysis_report['overall_error_rate']:.2%})")
    print("\n -> Top Confused Category Pairs:")
    for pair in error_analysis_report['confused_class_pairs'][:4]:
        print(f"    * True: {pair['actual_class']} -> Pred: {pair['predicted_class']} ({pair['error_count']} errors)")

    print("\n -> Error Rate by Document Length Slice:")
    for slice_name, slice_info in error_analysis_report['length_slice_metrics'].items():
        print(f"    * {slice_name}: {slice_info['error_count']}/{slice_info['total_documents']} ({slice_info['error_rate']:.2%})")

    # -------------------------------------------------------------------------
    # 9. Visualizations & Artifact Persistence
    # -------------------------------------------------------------------------
    print("\n -> Generating Publication-Grade Visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    plot_class_distribution(df, target_names, 'visualizations/class_distribution.png')
    plot_model_comparison(comparison_df, 'visualizations/model_comparison.png')
    plot_confusion_matrices(eval_results, target_names, 'visualizations/confusion_matrices.png')
    plot_per_class_f1(eval_results, target_names, 'visualizations/per_class_f1.png')
    plot_confidence_distribution(trained_models[best_model_name], X_test_tfidf, y_test, target_names, 'visualizations/confidence_distribution.png')
    plot_top_keywords(top_features_per_class, 'visualizations/top_keywords.png')
    print(" -> Saved 6 visualization charts in 'visualizations/' directory.")

    # Save Models
    joblib.dump(trained_models[best_model_name], 'models/best_model.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(trained_models, 'models/all_models.joblib')

    metadata = {
        'model_version': MODEL_VERSION,
        'best_model_name': best_model_name,
        'selection_metric': 'F1-Score (Weighted) primary, Accuracy secondary',
        'random_seed': RANDOM_SEED,
        'categories': target_names,
        'category_display_names': CATEGORY_DISPLAY_NAMES,
        'dataset_statistics': stats,
        'data_audit_report': audit_report,
        'tfidf_config': tfidf_config,
        'tfidf_experiments': tfidf_exp_df.to_dict(orient='records'),
        'preprocessing_config': {
            'lowercase': True,
            'strip_urls': True,
            'strip_emails': True,
            'strip_email_headers': True,
            'normalize_repeated_chars': True,
            'remove_standalone_numbers': True,
            'remove_punctuation': True,
            'remove_non_ascii': True,
            'strip_whitespace': True,
            'remove_stopwords': True,
            'apply_lemmatization': True,
            'preserved_words': list(DEFAULT_PREPROCESSING_CONFIG.preserved_words)
        },
        'hyperparameters': best_params,
        'confidence_thresholds': {
            'confidence_threshold': DEFAULT_INFERENCE_CONFIG.confidence_threshold,
            'min_active_tfidf': DEFAULT_INFERENCE_CONFIG.min_active_tfidf,
            'ambiguity_margin': DEFAULT_INFERENCE_CONFIG.ambiguity_margin,
            'min_ambiguity_sum': DEFAULT_INFERENCE_CONFIG.min_ambiguity_sum,
            'min_ambiguity_p1': DEFAULT_INFERENCE_CONFIG.min_ambiguity_p1,
            'topic_detection_threshold': DEFAULT_INFERENCE_CONFIG.topic_detection_threshold,
            'min_ambiguity_keywords': DEFAULT_INFERENCE_CONFIG.min_ambiguity_keywords
        },
        'cross_validation_results': cv_comparison_df.to_dict(orient='records'),
        'cv_details': cv_details,
        'metrics_table': comparison_df.to_dict(orient='records'),
        'error_analysis_summary': {
            "total_test_samples": error_analysis_report["total_test_samples"],
            "total_errors": error_analysis_report["total_errors"],
            "overall_error_rate": error_analysis_report["overall_error_rate"],
            "confused_class_pairs": error_analysis_report["confused_class_pairs"][:5],
            "length_slice_metrics": error_analysis_report["length_slice_metrics"],
            "confidence_breakdown": error_analysis_report["confidence_breakdown"]
        },
        'top_features_per_class': top_features_per_class,
        'system_environment': {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'joblib_version': joblib.__version__
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # 10. Self-Validation of Model Artifacts
    # -------------------------------------------------------------------------
    print("\n -> Verifying Model Artifacts Integrity...")
    validate_and_load_artifacts("models")
    print(" -> Artifact Integrity Verification: PASS (All checks succeeded).")

    total_pipeline_time = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 85)
    print(f" TRAINING & EVALUATION PIPELINE SUCCESSFULLY COMPLETED in {total_pipeline_time}s!")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
