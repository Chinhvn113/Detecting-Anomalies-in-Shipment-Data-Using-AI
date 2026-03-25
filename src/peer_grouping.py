"""Peer group aggregation utilities."""

from __future__ import annotations

import pandas as pd

from .config import (
    LOG_PRICE_PER_KG_COL,
    PEER_COUNT_COL,
    PEER_KEYS,
    PEER_MEAN_COL,
    PEER_MEDIAN_COL,
    PEER_Q1_COL,
    PEER_Q3_COL,
    PEER_STD_COL,
    PRICE_PER_KG_COL,
)


def compute_peer_statistics(df: pd.DataFrame, peer_keys: list[str] = PEER_KEYS) -> pd.DataFrame:
    """Compute peer-level statistics used by anomaly signals."""
    peer_stats = (
        df.groupby(peer_keys, dropna=False)
        .agg(
            **{
                PEER_MEAN_COL: (LOG_PRICE_PER_KG_COL, "mean"),
                PEER_STD_COL: (LOG_PRICE_PER_KG_COL, "std"),
                PEER_MEDIAN_COL: (PRICE_PER_KG_COL, "median"),
                PEER_Q1_COL: (PRICE_PER_KG_COL, lambda s: s.quantile(0.25)),
                PEER_Q3_COL: (PRICE_PER_KG_COL, lambda s: s.quantile(0.75)),
                PEER_COUNT_COL: (PRICE_PER_KG_COL, "count"),
            }
        )
        .reset_index()
    )

    peer_stats[PEER_STD_COL] = peer_stats[PEER_STD_COL].fillna(0.0)
    return peer_stats


def attach_peer_statistics(
    df: pd.DataFrame,
    peer_stats: pd.DataFrame,
    peer_keys: list[str] = PEER_KEYS,
) -> pd.DataFrame:
    """Merge peer statistics back onto row-level data."""
    return df.merge(peer_stats, on=peer_keys, how="left")
