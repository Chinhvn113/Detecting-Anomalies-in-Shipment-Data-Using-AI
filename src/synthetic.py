"""Synthetic anomaly generation for evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BORDERLINE_IQR_FACTOR,
    GROUND_TRUTH_COL,
    HS6_COL,
    ITEM_PRICE_COL,
    NET_MASS_COL,
    NOISE_STD,
    ORIGIN_COL,
    PRICE_REDUCTION_FACTOR,
    RARE_HS6_TOKEN,
    RARE_ORIGIN_TOKEN,
    RANDOM_SEED,
    SYNTHETIC_ANOMALY_RATIO,
)


def _inject_borderline_anomalies(
    out: pd.DataFrame,
    idx_borderline: np.ndarray,
    iqr_factor: float,
) -> None:
    """Inject subtle undervaluation anomalies using peer median and IQR.

    Target unit price is set near the lower distribution tail:
    median(unit_price) - iqr_factor * IQR(unit_price)
    """
    if len(idx_borderline) == 0:
        return

    item_price = pd.to_numeric(out[ITEM_PRICE_COL], errors="coerce")
    net_mass = pd.to_numeric(out[NET_MASS_COL], errors="coerce")
    valid_mass = net_mass > 0

    unit_price = pd.Series(np.nan, index=out.index, dtype=float)
    unit_price.loc[valid_mass] = item_price.loc[valid_mass] / net_mass.loc[valid_mass]

    peer_stats = (
        out.assign(_unit_price=unit_price)
        .groupby([HS6_COL, ORIGIN_COL], dropna=False)["_unit_price"]
        .agg(
            peer_median="median",
            peer_q1=lambda s: s.quantile(0.25),
            peer_q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    peer_stats["peer_iqr"] = (peer_stats["peer_q3"] - peer_stats["peer_q1"]).clip(lower=0)

    global_median = float(unit_price.median(skipna=True))
    global_q1 = float(unit_price.quantile(0.25))
    global_q3 = float(unit_price.quantile(0.75))
    global_iqr = max(global_q3 - global_q1, 0.0)

    target = out.loc[idx_borderline, [HS6_COL, ORIGIN_COL]].merge(
        peer_stats[[HS6_COL, ORIGIN_COL, "peer_median", "peer_iqr"]],
        on=[HS6_COL, ORIGIN_COL],
        how="left",
    )

    target_unit_price = (
        target["peer_median"].fillna(global_median)
        - iqr_factor * target["peer_iqr"].fillna(global_iqr)
    ).clip(lower=0.0)

    target_unit_series = pd.Series(target_unit_price.to_numpy(), index=idx_borderline)
    valid_idx = [idx for idx in idx_borderline if valid_mass.loc[idx]]
    if len(valid_idx) == 0:
        return

    out.loc[valid_idx, ITEM_PRICE_COL] = (
        target_unit_series.loc[valid_idx] * net_mass.loc[valid_idx]
    ).astype(float)


def generate_synthetic_dataset(
    df: pd.DataFrame,
    anomaly_ratio: float = SYNTHETIC_ANOMALY_RATIO,
    random_state: int = RANDOM_SEED,
    borderline_iqr_factor: float = BORDERLINE_IQR_FACTOR,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Inject synthetic anomalies and add ground-truth labels.

    Strategies:
    - price reduction for undervaluation-like anomalies
    - multiplicative noise for inconsistent pricing
    - rare product-origin combination injection
    - borderline subtle anomalies using median - factor * IQR
    """
    rng = np.random.default_rng(random_state)
    
    # Limit to max_samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)
    
    out = df.copy()
    n_rows = len(out)
    out[GROUND_TRUTH_COL] = 0

    if n_rows == 0:
        return out

    anomaly_count = max(1, int(np.ceil(n_rows * anomaly_ratio)))
    anomaly_indices = rng.choice(out.index.to_numpy(), size=anomaly_count, replace=False)

    idx_price, idx_noise, idx_rare, idx_borderline = np.array_split(anomaly_indices, 4)

    if len(idx_price) > 0:
        reduction_factors = rng.uniform(0.3, 0.7, size=len(idx_price))
        out.loc[idx_price, ITEM_PRICE_COL] = (
            pd.to_numeric(out.loc[idx_price, ITEM_PRICE_COL], errors="coerce").fillna(0.0)
            * reduction_factors
        )

    if len(idx_noise) > 0:
        noise = rng.normal(loc=1.0, scale=NOISE_STD, size=len(idx_noise))
        noise = np.clip(noise, 0.1, None)
        out.loc[idx_noise, ITEM_PRICE_COL] = (
            pd.to_numeric(out.loc[idx_noise, ITEM_PRICE_COL], errors="coerce").fillna(0.0)
            * noise
        )

    if len(idx_rare) > 0:
        # Ensure dtype compatibility before assigning rare string tokens.
        out[ORIGIN_COL] = out[ORIGIN_COL].astype("string")
        out[HS6_COL] = out[HS6_COL].astype("string")
        out.loc[idx_rare, ORIGIN_COL] = RARE_ORIGIN_TOKEN
        out.loc[idx_rare, HS6_COL] = RARE_HS6_TOKEN

    _inject_borderline_anomalies(
        out=out,
        idx_borderline=idx_borderline,
        iqr_factor=borderline_iqr_factor,
    )

    out.loc[anomaly_indices, GROUND_TRUTH_COL] = 1
    return out
