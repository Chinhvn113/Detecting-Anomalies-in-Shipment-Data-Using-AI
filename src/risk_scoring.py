"""Risk score normalization, combination, and flagging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FLAG_COL,
    IQR_SCORE_COL,
    RARITY_SCORE_COL,
    RESIDUAL_SCORE_COL,
    RISK_SCORE_COL,
    RISK_WEIGHTS,
    TOP_K_RATIO,
    Z_SCORE_COL,
)


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Deprecated helper kept for backward compatibility."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


def rank_based_normalize(series: pd.Series) -> pd.Series:
    """Normalize a numeric series using percentile ranks in [0, 1]."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

    ranked = values.rank(pct=True, method="average")
    return ranked.fillna(0.0).astype(float)


def normalize_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all base anomaly signals and append _norm columns."""
    out = df.copy()
    out["rank_score"] = rank_based_normalize(out[RESIDUAL_SCORE_COL])

    for col in [RESIDUAL_SCORE_COL, Z_SCORE_COL, IQR_SCORE_COL, RARITY_SCORE_COL]:
        out[f"{col}_norm"] = rank_based_normalize(out[col])
    return out


def compute_risk_score(df: pd.DataFrame, weights: dict[str, float] = RISK_WEIGHTS) -> pd.DataFrame:
    """Combine normalized anomaly signals into a weighted final risk score."""
    out = normalize_signal_columns(df)

    out[RISK_SCORE_COL] = (
        weights[RESIDUAL_SCORE_COL] * out[f"{RESIDUAL_SCORE_COL}_norm"]
        + weights[Z_SCORE_COL] * out[f"{Z_SCORE_COL}_norm"]
        + weights[IQR_SCORE_COL] * out[f"{IQR_SCORE_COL}_norm"]
        + weights[RARITY_SCORE_COL] * out[f"{RARITY_SCORE_COL}_norm"]
    )
    return out


def flag_top_k_percent(
    df: pd.DataFrame,
    k_ratio: float = TOP_K_RATIO,
    score_col: str = RISK_SCORE_COL,
) -> pd.DataFrame:
    """Flag top-k fraction by risk score instead of fixed threshold."""
    out = df.copy()
    n_rows = len(out)
    if n_rows == 0:
        out[FLAG_COL] = pd.Series(dtype=bool)
        return out

    k_count = max(1, int(np.ceil(n_rows * k_ratio)))
    ranked_indices = out[score_col].nlargest(k_count).index
    out[FLAG_COL] = False
    out.loc[ranked_indices, FLAG_COL] = True
    return out
