"""Human-readable explanation generation for anomaly flags."""

from __future__ import annotations

import pandas as pd

from .config import (
    EXPLANATION_COL,
    FLAG_COL,
    IQR_SCORE_COL,
    RARITY_SCORE_COL,
    RESIDUAL_SCORE_COL,
    Z_SCORE_COL,
)


def _build_single_explanation(row: pd.Series) -> str:
    """Create a row-level explanation string from score contributions."""
    reasons: list[str] = []

    if row.get(f"{Z_SCORE_COL}_norm", 0.0) >= 0.6:
        reasons.append("Unit price significantly deviates from peer average")

    if row.get(f"{IQR_SCORE_COL}_norm", 0.0) >= 0.6:
        reasons.append("Unit price lies outside robust peer IQR range")

    if row.get(f"{RESIDUAL_SCORE_COL}_norm", 0.0) >= 0.6:
        reasons.append("Value inconsistent with expected based on shipment characteristics")

    if row.get(f"{RARITY_SCORE_COL}_norm", 0.0) >= 0.6:
        reasons.append("Rare product-origin combination")

    if not reasons:
        reasons.append("Combined signals indicate elevated anomaly risk")

    return "; ".join(reasons)


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """Add explanation text for flagged rows."""
    out = df.copy()
    out[EXPLANATION_COL] = ""

    flagged_mask = out[FLAG_COL].fillna(False)
    out.loc[flagged_mask, EXPLANATION_COL] = out.loc[flagged_mask].apply(
        _build_single_explanation,
        axis=1,
    )

    return out
