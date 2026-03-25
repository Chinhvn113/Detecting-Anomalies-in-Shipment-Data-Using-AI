"""Feature engineering for shipment anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ITEM_PRICE_COL,
    LOG_MASS_COL,
    LOG_PRICE_PER_KG_COL,
    NET_MASS_COL,
    PRICE_PER_KG_COL,
)


def add_price_mass_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price/mass derived features with safe numeric operations."""
    out = df.copy()

    item_price = pd.to_numeric(out[ITEM_PRICE_COL], errors="coerce")
    net_mass = pd.to_numeric(out[NET_MASS_COL], errors="coerce")

    valid_mass = net_mass > 0
    out[PRICE_PER_KG_COL] = np.where(valid_mass, item_price / net_mass, np.nan)

    non_negative_price_per_kg = out[PRICE_PER_KG_COL].clip(lower=0).fillna(0)
    non_negative_mass = net_mass.clip(lower=0).fillna(0)

    out[LOG_PRICE_PER_KG_COL] = np.log1p(non_negative_price_per_kg)
    out[LOG_MASS_COL] = np.log1p(non_negative_mass)
    return out
