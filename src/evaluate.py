"""
Model Evaluation and Visualization Module

Computes performance metrics (Accuracy, Precision, Recall, Weighted F1, Macro F1),
prediction latency, training duration, confusion matrices, and comparative plots.
"""

from typing import Dict, Any, List, Tuple
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def evaluate_single_model(
    model,
    X_test,
    y_test,
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Computes all standard classification evaluation metrics and per-sample latency.

    Parameters
    ----------
    model : estimator
    X_test : feature matrix
    y_test : ground truth labels
    target_names : list of class names

    Returns
    -------
    dict of metrics
    """
    # Measure prediction latency across test set
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_duration = time.perf_counter() - start_pred
    per_doc_latency_ms = round((pred_duration / max(len(y_test), 1)) * 1000.0, 3)

    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    return {
        'accuracy': float(acc),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'latency_ms_per_doc': per_doc_latency_ms,
        'predictions': y_pred.tolist()
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_test,
    y_test,
    target_names: List[str],
    training_times: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], str]:
    """
    Evaluates all trained models, produces a comparison DataFrame, and identifies
    the best performing model based on Weighted F1-score (breaking ties with Accuracy).

    Returns
    -------
    comparison_df : pd.DataFrame
    evaluation_results : dict
    best_model_name : str
    """
    results = {}
    rows = []

    for name, model in models.items():
        metrics = evaluate_single_model(model, X_test, y_test, target_names)
        metrics['training_time'] = training_times.get(name, 0.0)
        results[name] = metrics

        rows.append({
            'Model': name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision (Weighted)': round(metrics['precision_weighted'], 4),
            'Recall (Weighted)': round(metrics['recall_weighted'], 4),
            'F1-Score (Weighted)': round(metrics['f1_weighted'], 4),
            'F1-Score (Macro)': round(metrics['f1_macro'], 4),
            'Training Time (s)': metrics['training_time'],
            'Latency (ms/doc)': metrics['latency_ms_per_doc']
        })

    comparison_df = pd.DataFrame(rows)
    # Selection rule: Primary sort by Weighted F1-Score, secondary by Accuracy
    comparison_df = comparison_df.sort_values(
        by=['F1-Score (Weighted)', 'Accuracy'],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = comparison_df.iloc[0]['Model']

    return comparison_df, results, best_model_name


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """
    Creates and saves a multi-metric comparison bar plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    melted = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
        var_name='Metric',
        value_name='Score'
    )

    plt.figure(figsize=(10, 5.5), dpi=300)
    ax = sns.barplot(
        data=melted,
        x='Model',
        y='Score',
        hue='Metric',
        palette='viridis'
    )
    plt.title('Classifier Performance Comparison across Evaluation Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Machine Learning Algorithm', fontsize=11, fontweight='semibold')
    plt.ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='semibold')
    plt.ylim(0.0, 1.08)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.xticks(rotation=10, ha='right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=8, rotation=0, xytext=(0, 2),
                textcoords='offset points'
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(
    evaluation_results: Dict[str, Dict[str, Any]],
    target_names: List[str],
    output_path: str
):
    """
    Plots a 2x2 grid of confusion matrices for the 4 classifiers.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_names = list(evaluation_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    short_labels = ['Graphics', 'Baseball', 'Space', 'Politics']

    for idx, name in enumerate(model_names[:4]):
        cm = np.array(evaluation_results[name]['confusion_matrix'])
        acc = evaluation_results[name]['accuracy']

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=axes[idx],
            annot_kws={"size": 11, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}\n(Accuracy: {acc*100:.1f}%)", fontsize=11, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=9.5, fontweight='semibold')
        axes[idx].set_ylabel('True Label', fontsize=9.5, fontweight='semibold')
        axes[idx].tick_params(axis='x', rotation=15, labelsize=9)
        axes[idx].tick_params(axis='y', rotation=0, labelsize=9)

    plt.suptitle('Confusion Matrix Heatmaps for 4 Classifiers', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_class_distribution(df: pd.DataFrame, target_names: List[str], output_path: str):
    """
    Plots distribution of documents across the 4 classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counts = df['category_name'].value_counts()
    plt.figure(figsize=(8, 4.5), dpi=300)
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='crest', legend=False)

    plt.title('Dataset Class Distribution (4 Target Categories)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Document Category', fontsize=10.5, fontweight='semibold')
    plt.ylabel('Document Count', fontsize=10.5, fontweight='semibold')
    plt.xticks(rotation=15, ha='right', fontsize=9.5)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=9, xytext=(0, 3),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_keywords(top_features_dict: Dict[str, List[Tuple[str, float]]], output_path: str):
    """
    Plots top informative TF-IDF keywords per class.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)
    axes = axes.flatten()

    colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b']

    for idx, (class_name, words) in enumerate(top_features_dict.items()):
        if idx >= len(axes):
            break
        words = words[:8]
        terms = [w[0] for w in words][::-1]
        scores = [w[1] for w in words][::-1]

        axes[idx].barh(terms, scores, color=colors[idx % len(colors)], edgecolor='none')
        axes[idx].set_title(f"{class_name}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Mean TF-IDF Score', fontsize=9)
        axes[idx].tick_params(labelsize=8.5)

    plt.suptitle('Top Distinguishing TF-IDF Terms per Category', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
