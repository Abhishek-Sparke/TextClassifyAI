"""
Model Evaluation and Visualization Module

Computes performance metrics (Accuracy, Precision, Recall, F1-score),
generates comparison tables, plots confusion matrices and comparative charts.
"""

from typing import Dict, Any, List, Tuple
import os
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

# Apply sleek styling for charts
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
    Computes all standard classification evaluation metrics for a single model.

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
    y_pred = model.predict(X_test)

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
    the best performing model.

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
            'Training Time (s)': metrics['training_time']
        })

    comparison_df = pd.DataFrame(rows)
    # Sort by F1-Score (Weighted) then Accuracy
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
    plt.ylim(0.0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.xticks(rotation=15, ha='right')

    # Value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=7.5, rotation=0, xytext=(0, 2),
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
    Adapts dynamic cell formatting and label resolution for up to 20 classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_names = list(evaluation_results.keys())
    num_classes = len(target_names)

    fig_size = (18, 16) if num_classes > 10 else (13, 11)
    annot_size = 5.5 if num_classes > 10 else 11
    tick_size = 7 if num_classes > 10 else 9

    fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=300)
    axes = axes.flatten()

    display_map = {
        'alt.atheism': 'Atheism',
        'comp.graphics': 'Graphics',
        'comp.os.ms-windows.misc': 'MS Win',
        'comp.sys.ibm.pc.hardware': 'IBM PC',
        'comp.sys.mac.hardware': 'Mac HW',
        'comp.windows.x': 'Win X',
        'misc.forsale': 'Sale',
        'rec.autos': 'Autos',
        'rec.motorcycles': 'Mcycles',
        'rec.sport.baseball': 'Baseball',
        'rec.sport.hockey': 'Hockey',
        'sci.crypt': 'Crypt',
        'sci.electronics': 'Electronics',
        'sci.med': 'Medicine',
        'sci.space': 'Space',
        'soc.religion.christian': 'Christian',
        'talk.politics.guns': 'Guns',
        'talk.politics.mideast': 'Mideast',
        'talk.politics.misc': 'Politics',
        'talk.religion.misc': 'Religion'
    }
    short_labels = [display_map.get(name, name.split('.')[-1].capitalize()) for name in target_names]

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
            annot_kws={"size": annot_size, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}\n(Accuracy: {acc*100:.1f}%)", fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10, fontweight='semibold')
        axes[idx].set_ylabel('True Label', fontsize=10, fontweight='semibold')
        axes[idx].tick_params(axis='x', rotation=45 if num_classes > 10 else 25, labelsize=tick_size)
        axes[idx].tick_params(axis='y', rotation=0, labelsize=tick_size)

    plt.suptitle('Confusion Matrix Heatmaps for All Classification Models', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_class_distribution(df: pd.DataFrame, target_names: List[str], output_path: str):
    """
    Plots distribution of documents across classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_classes = len(target_names)

    counts = df['category_name'].value_counts()
    fig_width = max(10, num_classes * 0.65)
    plt.figure(figsize=(fig_width, 5.0), dpi=300)
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='crest', legend=False)

    plt.title('Dataset Class Distribution (Number of Documents per Category)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Document Category', fontsize=11, fontweight='semibold')
    plt.ylabel('Document Count', fontsize=11, fontweight='semibold')
    plt.xticks(rotation=40 if num_classes > 8 else 20, ha='right', fontsize=8.5 if num_classes > 10 else 9.5)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=7.5 if num_classes > 10 else 9, xytext=(0, 3),
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
    num_classes = len(top_features_dict)
    cols = 4 if num_classes >= 12 else 2
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16 if cols == 4 else 12, 3.2 * rows), dpi=300)
    axes = np.array(axes).flatten()

    for idx, (class_name, words) in enumerate(top_features_dict.items()):
        if idx >= len(axes):
            break
        # Take top 8 words
        words = words[:8]
        terms = [w[0] for w in words][::-1]
        scores = [w[1] for w in words][::-1]

        axes[idx].barh(terms, scores, color='#3b82f6', edgecolor='none')
        axes[idx].set_title(f"{class_name}", fontsize=9.5, fontweight='bold')
        axes[idx].set_xlabel('Mean TF-IDF Score', fontsize=8)
        axes[idx].tick_params(labelsize=8)

    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Top Distinguishing TF-IDF Terms per Category', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
