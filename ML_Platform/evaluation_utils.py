"""
Evaluation utilities: metrics & visualizations for classification and regression.

Public API
----------
- evaluate_model(model, X_test, y_test, task_type)
- plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None)
- plot_regression_results(y_true, y_pred)
- plot_training_curves(history)
- get_metrics_report(task_type, metrics_dict)

All plotting functions return matplotlib Figure objects (ready for Streamlit).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics as skm


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def _safe_predict_proba(model, X):
    """Return predicted probabilities or decision function if available, else None."""
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Convert decision scores to 2-column proba-like for binary tasks
            if scores.ndim == 1:
                proba = np.vstack([1 - _sigmoid(scores), _sigmoid(scores)]).T
            else:
                proba = _softmax(scores)
    except Exception:
        proba = None
    return proba


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate_model(model, X_test, y_test, task_type: str) -> Dict[str, Any]:
    """Compute metrics and produce predictions for the given task type.

    Returns a dict with at least:
      - y_pred
      - metrics (dict)
      - proba (optional, for classification)
      - auc_roc (optional)
    """
    task = (task_type or "").lower()

    # Keras models compatibility
    is_keras = hasattr(model, "predict") and "keras" in str(type(model)).lower()

    if task == "classification":
        if is_keras:
            # Keras classification: output shape can be (n, 1) for binary or (n, C)
            raw = model.predict(X_test, verbose=0)
            if raw.ndim == 2 and raw.shape[1] > 1:
                y_proba = raw
                y_pred = np.argmax(raw, axis=1)
            else:
                y_proba = np.hstack([1 - raw, raw])
                y_pred = (raw.ravel() >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = _safe_predict_proba(model, X_test)

        acc = skm.accuracy_score(y_test, y_pred)
        precision = skm.precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = skm.recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = skm.f1_score(y_test, y_pred, average="macro", zero_division=0)

        result = {
            "y_pred": y_pred,
            "metrics": {
                "accuracy": float(acc),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
            },
        }

        # AUC-ROC (binary only)
        try:
            if y_proba is not None:
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    # probabilities for class 1
                    auc = skm.roc_auc_score(y_test, y_proba.ravel())
                elif y_proba.shape[1] == 2:
                    auc = skm.roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # multiclass average
                    auc = skm.roc_auc_score(y_test, y_proba, multi_class="ovr")
                result["metrics"]["auc_roc"] = float(auc)
                result["proba"] = y_proba
        except Exception:
            pass

        return result

    elif task == "regression":
        y_pred = model.predict(X_test)
        mae = skm.mean_absolute_error(y_test, y_pred)
        mse = skm.mean_squared_error(y_test, y_pred)
        r2 = skm.r2_score(y_test, y_pred)
        return {
            "y_pred": y_pred,
            "metrics": {"mae": float(mae), "mse": float(mse), "r2": float(r2)},
        }

    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels: Optional[list] = None, normalize: Optional[str] = None):
    """Draw a confusion matrix and return a Matplotlib figure.

    normalize ∈ {None, 'true', 'pred', 'all'} aligns with sklearn's API.
    """
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion" + (" (normalisée)" if normalize else ""))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    return fig


def _scatter_diag(ax, x, y, title: str, xlabel: str, ylabel: str):
    ax.scatter(x, y, alpha=0.6)
    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_regression_results(y_true, y_pred) -> Tuple[plt.Figure, plt.Figure]:
    """Return two figures: (y_true vs y_pred scatter, residuals distribution)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Scatter
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    _scatter_diag(ax1, y_true, y_pred, "Vrai vs Prédit", "Vrai", "Prédit")
    fig1.tight_layout()

    # Residuals
    residuals = y_true - y_pred
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(residuals, bins=30, kde=True, ax=ax2)
    ax2.set_title("Résidus (Vrai - Prédit)")
    ax2.set_xlabel("Résidu")
    fig2.tight_layout()

    return fig1, fig2


def plot_training_curves(history) -> Optional[plt.Figure]:
    """Plot training/validation curves from a Keras History object.

    Returns a single figure with loss and (if present) accuracy/MAE.
    """
    if history is None or not hasattr(history, "history"):
        return None

    h = history.history
    fig, ax = plt.subplots(figsize=(6, 4))

    # Loss
    if "loss" in h:
        ax.plot(h["loss"], label="loss")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="val_loss")

    # Accuracy or MAE depending on model
    aux_keys = [k for k in ("accuracy", "val_accuracy", "mae", "val_mae") if k in h]
    for k in aux_keys:
        ax.plot(h[k], label=k)

    ax.set_title("Courbes d'entraînement")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valeur")
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def get_metrics_report(task_type: str, metrics_dict: Dict[str, float]) -> Dict[str, Any]:
    """Return a lightweight, serializable report dict.

    Can be dumped to JSON/CSV by the caller if desired.
    """
    return {
        "task_type": task_type,
        "metrics": {k: float(v) for k, v in (metrics_dict or {}).items()},
    }
