"""Reusable evaluation metrics for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import PRECISION_AT_K_RATIO


# def compute_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
#     """Compute F1-score with safe handling for empty/degenerate inputs."""
#     if len(y_true) == 0:
#         return 0.0
#     return float(f1_score(y_true.astype(int), y_pred.astype(int), zero_division=0))


def compute_precision_at_k(
    y_true: pd.Series,
    scores: pd.Series,
    k_ratio: float = PRECISION_AT_K_RATIO,
) -> float:
    """Compute Precision@K from top-k highest risk scores."""
    if len(y_true) == 0:
        return 0.0

    k_count = max(1, int(np.ceil(len(scores) * k_ratio)))
    top_indices = scores.nlargest(k_count).index

    true_top = y_true.loc[top_indices].astype(int)
    return float(true_top.mean()) if len(true_top) > 0 else 0.0


def compute_recall_at_k(
    y_true: pd.Series,
    scores: pd.Series,
    k_ratio: float = PRECISION_AT_K_RATIO,
) -> float:
    """Compute Recall@K from top-k highest risk scores."""
    if len(y_true) == 0:
        return 0.0

    y_true_int = y_true.astype(int)
    positives = int(y_true_int.sum())
    if positives == 0:
        return 0.0

    k_count = max(1, int(np.ceil(len(scores) * k_ratio)))
    top_indices = scores.nlargest(k_count).index
    true_positive_at_k = int(y_true_int.loc[top_indices].sum())
    return float(true_positive_at_k / positives)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: pd.Series,
    k_ratio: float = PRECISION_AT_K_RATIO,
) -> dict[str, float]:
    """Return a dictionary of evaluation metrics."""
    return {
        # "f1_score": compute_f1(y_true=y_true, y_pred=y_pred),
        "precision_at_k": compute_precision_at_k(y_true=y_true, scores=scores, k_ratio=k_ratio),
        "recall_at_k": compute_recall_at_k(y_true=y_true, scores=scores, k_ratio=k_ratio),
    }
