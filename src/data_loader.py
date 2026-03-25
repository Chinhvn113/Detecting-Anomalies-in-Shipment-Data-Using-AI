"""Data loading and cleaning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import (
    DECLARATION_ID_COL,
    HS6_COL,
    ITEM_PRICE_COL,
    NET_MASS_COL,
    ORIGIN_COL,
)


REQUIRED_COLUMNS: Sequence[str] = [
    DECLARATION_ID_COL,
    HS6_COL,
    ORIGIN_COL,
    NET_MASS_COL,
    ITEM_PRICE_COL,
]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load shipment dataset from CSV."""
    return pd.read_csv(file_path)


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str] = REQUIRED_COLUMNS) -> None:
    """Validate that all required columns are present."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and edge-case handling.

    - Drops rows missing peer-defining keys.
    - Coerces numeric fields to numeric and imputes with median.
    - Removes invalid rows with non-positive mass or negative price.
    """
    validate_required_columns(df)

    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=[HS6_COL, ORIGIN_COL])
    cleaned[HS6_COL] = cleaned[HS6_COL].astype("string").fillna("UNKNOWN").astype(str)
    cleaned[ORIGIN_COL] = cleaned[ORIGIN_COL].astype("string").fillna("UNKNOWN").astype(str)

    numeric_cols = [NET_MASS_COL, ITEM_PRICE_COL]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        median_value = cleaned[col].median(skipna=True)
        if pd.isna(median_value):
            median_value = 0.0
        cleaned[col] = cleaned[col].fillna(median_value)

    cleaned = cleaned[(cleaned[NET_MASS_COL] > 0) & (cleaned[ITEM_PRICE_COL] >= 0)].copy()

    object_cols = cleaned.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        cleaned[col] = cleaned[col].fillna("UNKNOWN")

    cleaned = cleaned.reset_index(drop=True)
    return cleaned
