"""
Model evaluation with business metrics for the Churn Prediction Pipeline.

Beyond standard ML metrics, we translate model performance into
business impact: how many churners do we catch, and at what cost?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {"No": "#C9B8D4", "Yes": "#F2C4CE"}
COLORS = ["#C9B8D4", "#F2C4CE"]
SINGLE_COLOR = "#C9B8D4"
BACKGROUND = "#FAF8FB"
PLOTS_DIR = Path("data/processed")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Full model evaluation: ML metrics + business metrics + plots.

    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name for plot titles

    Returns:
        Dictionary with all metrics
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "model_name": model_name,
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # Plot all evaluations
    plot_roc_curve(y_test, y_prob, model_name)
    plot_precision_recall(y_test, y_prob, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_threshold_analysis(y_test, y_prob, model_name)
    plot_business_impact(y_test, y_prob, model_name)

    return metrics


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, model_name: str):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(fpr, tpr, color="#9B89B4", linewidth=2.5,
            label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#DDDDDD", linewidth=1.5,
            linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=SINGLE_COLOR)
    ax.set_title(f"{model_name} — ROC Curve", fontsize=14,
                 fontweight="bold", color="#4A4A4A")
    ax.set_xlabel("False Positive Rate", color="#4A4A4A")
    ax.set_ylabel("True Positive Rate", color="#4A4A4A")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDDDDD")
    ax.tick_params(colors="#4A4A4A")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"09_roc_curve_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()


def plot_precision_recall(y_test, y_prob, model_name: str):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(recall, precision, color="#9B89B4", linewidth=2.5)
    ax.axhline(y=baseline, color="#DDDDDD", linewidth=1.5,
               linestyle="--", label=f"Baseline ({baseline:.2%})")
    ax.fill_between(recall, precision, alpha=0.1, color=SINGLE_COLOR)
    ax.set_title(f"{model_name} — Precision-Recall Curve",
                 fontsize=14, fontweight="bold", color="#4A4A4A")
    ax.set_xlabel("Recall", color="#4A4A4A")
    ax.set_ylabel("Precision", color="#4A4A4A")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDDDDD")
    ax.tick_params(colors="#4A4A4A")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"10_pr_curve_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    cmap = LinearSegmentedColormap.from_list("cm", ["#FAF8FB", "#C9B8D4", "#6D5A8E"])
    sns.heatmap(cm_pct, annot=False, fmt=".2%", cmap=cmap,
                ax=ax, linewidths=2, linecolor="white",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.4, f"{cm[i,j]:,}",
                    ha='center', fontsize=14, fontweight='bold', color="#4A4A4A")
            ax.text(j + 0.5, i + 0.6, f"({cm_pct[i,j]:.1%})",
                    ha='center', fontsize=11, color="#4A4A4A")

    ax.set_title(f"{model_name} — Confusion Matrix",
                 fontsize=14, fontweight="bold", color="#4A4A4A", pad=15)
    ax.set_ylabel("Actual", color="#4A4A4A")
    ax.set_xlabel("Predicted", color="#4A4A4A")
    ax.tick_params(colors="#4A4A4A")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"11_confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()


def plot_threshold_analysis(y_test, y_prob, model_name: str):
    """Plot precision, recall and F1 across thresholds."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if y_pred_t.sum() == 0:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
        else:
            from sklearn.metrics import precision_score, recall_score
            precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
            f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(thresholds, precisions, color="#C9B8D4", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, color="#F2C4CE", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, color="#9B89B4", linewidth=2.5, label="F1")
    ax.axvline(x=best_t, color="#6D5A8E", linewidth=1.5,
               linestyle="--", label=f"Best threshold ({best_t:.2f})")
    ax.set_title(f"{model_name} — Threshold Analysis",
                 fontsize=14, fontweight="bold", color="#4A4A4A")
    ax.set_xlabel("Threshold", color="#4A4A4A")
    ax.set_ylabel("Score", color="#4A4A4A")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDDDDD")
    ax.tick_params(colors="#4A4A4A")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"12_threshold_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()

    print(f"\nBest threshold by F1: {best_t:.2f}")
    print(f"  Precision: {precisions[np.argmax(f1s)]:.4f}")
    print(f"  Recall:    {recalls[np.argmax(f1s)]:.4f}")
    print(f"  F1:        {max(f1s):.4f}")


def plot_business_impact(y_test, y_prob, model_name: str,
                          retention_cost: float = 50,
                          churn_cost: float = 500):
    """
    Translate model performance into business value.

    Args:
        retention_cost: Cost of contacting a customer (€)
        churn_cost: Revenue lost per churned customer (€)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    net_savings = []

    total_churners = y_test.sum()

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        # Savings: churners we catch × churn cost avoided
        # Cost: everyone we contact × retention cost
        savings = tp * churn_cost
        cost = (tp + fp) * retention_cost
        net_savings.append(savings - cost)

    best_t = thresholds[np.argmax(net_savings)]
    best_saving = max(net_savings)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(thresholds, net_savings, color="#9B89B4", linewidth=2.5)
    ax.fill_between(thresholds, net_savings, 0,
                    where=[s > 0 for s in net_savings],
                    alpha=0.2, color=SINGLE_COLOR, label="Net positive")
    ax.axvline(x=best_t, color="#6D5A8E", linewidth=1.5,
               linestyle="--", label=f"Optimal threshold ({best_t:.2f})")
    ax.axhline(y=0, color="#DDDDDD", linewidth=1)
    ax.set_title(f"{model_name} — Business Impact\n"
                 f"(retention cost=€{retention_cost}/customer, churn cost=€{churn_cost}/customer)",
                 fontsize=13, fontweight="bold", color="#4A4A4A")
    ax.set_xlabel("Threshold", color="#4A4A4A")
    ax.set_ylabel("Net Savings (€)", color="#4A4A4A")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#DDDDDD")
    ax.tick_params(colors="#4A4A4A")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"13_business_impact_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()

    print(f"\nBusiness Impact Analysis:")
    print(f"  Optimal threshold: {best_t:.2f}")
    print(f"  Max net savings per {len(y_test):,} customers: €{best_saving:,.0f}")
    print(f"  Assumptions: retention=€{retention_cost}, churn cost=€{churn_cost}")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(".").resolve()))

    from src.churn_pipeline.data.loader import load_raw_data
    from src.churn_pipeline.data.preprocessing import prepare_data
    from src.churn_pipeline.features.engineering import FeatureEngineer