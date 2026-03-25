"""Lightweight Gaussian assumption checks based on skewness."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    GAUSSIAN_MIN_OK_RATIO,
    GAUSSIAN_MIN_PEER_COUNT,
    GAUSSIAN_SKEW_THRESHOLD,
    LOG_PRICE_PER_KG_COL,
    PEER_COUNT_COL,
    PEER_GAUSSIAN_OK_COL,
    PEER_KEYS,
    PEER_SKEW_COL,
)


def compute_peer_skewness(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    peer_keys: list[str] = PEER_KEYS,
) -> pd.DataFrame:
    """Compute skewness and Gaussian-validity flag per peer group.

    The optional sampling parameter can be used to inspect only a subset of peers.
    """
    grouped = (
        df.groupby(peer_keys, dropna=False)
        .agg(
            **{
                PEER_SKEW_COL: (LOG_PRICE_PER_KG_COL, lambda s: float(s.skew()) if len(s) > 2 else np.nan),
                PEER_COUNT_COL: (LOG_PRICE_PER_KG_COL, "count"),
            }
        )
        .reset_index()
    )

    grouped[PEER_SKEW_COL] = grouped[PEER_SKEW_COL].fillna(0.0)
    grouped[PEER_GAUSSIAN_OK_COL] = (
        grouped[PEER_COUNT_COL] >= GAUSSIAN_MIN_PEER_COUNT
    ) & (grouped[PEER_SKEW_COL].abs() <= GAUSSIAN_SKEW_THRESHOLD)

    if sample_size is not None and sample_size > 0 and len(grouped) > sample_size:
        grouped = grouped.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return grouped


def is_gaussian_reasonable(
    peer_skewness_df: pd.DataFrame,
    min_ok_ratio: float = GAUSSIAN_MIN_OK_RATIO,
) -> bool:
    """Return global Gaussian-assumption flag based on share of valid peers."""
    if peer_skewness_df.empty:
        return False
    ok_ratio = float(peer_skewness_df[PEER_GAUSSIAN_OK_COL].mean())
    return ok_ratio >= min_ok_ratio
