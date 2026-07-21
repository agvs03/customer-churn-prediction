"""Plotting & evaluation helpers (EDA + model diagnostics)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------- EDA plots ------------------------------------
def plot_class_distribution(df, target, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts().sort_index()
    labels = ["Retained (0)", "Churned (1)"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Class Distribution (Churn)")
    ax.set_ylabel("Customers")
    return _save(fig, out_dir, "class_distribution.png")


def plot_histograms(df, out_dir, cols=("tenure", "MonthlyCharges", "TotalCharges")):
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(data=df, x=col, hue="Churn", bins=30, kde=True, ax=ax,
                     palette={0: "#4C72B0", 1: "#C44E52"}, alpha=0.6)
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Numeric Feature Distributions by Churn", y=1.02)
    return _save(fig, out_dir, "histograms.png")


def plot_correlation_heatmap(df, out_dir):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(min(0.6 * len(corr) + 2, 18), min(0.55 * len(corr) + 2, 16)))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title("Correlation Heatmap (numeric + engineered features)")
    return _save(fig, out_dir, "correlation_heatmap.png")


# --------------------------- Model diagnostics ----------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, out_dir, name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - XGBoost")
    return _save(fig, out_dir, name)


def plot_roc_curve(y_true, y_proba, out_dir, name="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, name)


def plot_feature_importance(model, feature_names, out_dir, top_n=20, name="feature_importance.png"):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1], color="#55A868")
    ax.set_title(f"Top {top_n} Feature Importances - XGBoost")
    ax.set_xlabel("Importance (gain)")
    return _save(fig, out_dir, name)


def plot_model_comparison(results_df, out_dir, name="model_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    results_df = results_df.sort_values("cv_roc_auc")
    ax.barh(results_df["model"], results_df["cv_roc_auc"], color="#4C72B0",
            xerr=results_df["cv_std"], capsize=4)
    ax.set_xlabel("Cross-validated ROC-AUC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("5-Model Comparison (5-fold CV)")
    for i, (v, s) in enumerate(zip(results_df["cv_roc_auc"], results_df["cv_std"])):
        ax.text(v, i, f" {v:.3f}", va="center")
    return _save(fig, out_dir, name)
