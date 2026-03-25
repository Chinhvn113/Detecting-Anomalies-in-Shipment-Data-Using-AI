"""Anomaly signal computation module."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import (
    DATE_COL,
    DEPARTURE_COL,
    EPSILON,
    HS6_COL,
    IQR_SCORE_COL,
    IMPORT_TYPE_COL,
    IMPORT_USE_COL,
    LOG_MASS_COL,
    LOG_PRICE_PER_KG_COL,
    MODE_TRANSPORT_COL,
    NET_MASS_COL,
    OFFICE_ID_COL,
    ORIGIN_COL,
    ORIGIN_INDICATOR_COL,
    PEER_COUNT_COL,
    PEER_GAUSSIAN_OK_COL,
    PEER_KEYS,
    PEER_MEAN_COL,
    PEER_Q1_COL,
    PEER_Q3_COL,
    PEER_STD_COL,
    PAYMENT_TYPE_COL,
    PRICE_PER_KG_COL,
    PROCESS_TYPE_COL,
    RARITY_SCORE_COL,
    RESIDUAL_COL,
    RESIDUAL_SCORE_COL,
    TAX_RATE_COL,
    TAX_TYPE_COL,
    Z_SCORE_COL,
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series and return zero where division is invalid."""
    den = denominator.replace(0, np.nan)
    result = numerator / den
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_z_score(df: pd.DataFrame, gaussian_valid: bool) -> pd.Series:
    """Compute absolute z-score from peer mean/std on log scale."""
    if not gaussian_valid:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    raw_z = _safe_divide(df[LOG_PRICE_PER_KG_COL] - df[PEER_MEAN_COL], df[PEER_STD_COL])
    z_score = raw_z.abs()

    if PEER_GAUSSIAN_OK_COL in df.columns:
        z_score = z_score.where(df[PEER_GAUSSIAN_OK_COL], 0.0)

    return z_score.fillna(0.0)


def compute_iqr_score(df: pd.DataFrame) -> pd.Series:
    """Compute robust IQR outlier score based on peer quartiles."""
    q1 = df[PEER_Q1_COL]
    q3 = df[PEER_Q3_COL]
    iqr = (q3 - q1).clip(lower=0)

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    value = df[PRICE_PER_KG_COL]

    lower_excess = (lower_bound - value).clip(lower=0)
    upper_excess = (value - upper_bound).clip(lower=0)
    distance = lower_excess + upper_excess

    return _safe_divide(distance, iqr + EPSILON).fillna(0.0)


def _build_residual_model(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> Pipeline:
    """Create sklearn pipeline for residual-based anomaly signal."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_residual_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit a regression model and return residual and absolute residual scores."""
    model_df = df.copy()

    if DATE_COL in model_df.columns:
        parsed_date = pd.to_datetime(model_df[DATE_COL], errors="coerce")
        model_df["decl_year"] = parsed_date.dt.year
        model_df["decl_month"] = parsed_date.dt.month
        model_df["decl_dayofweek"] = parsed_date.dt.dayofweek

    candidate_numeric = [
        NET_MASS_COL,
        TAX_RATE_COL,
        LOG_MASS_COL,
        PEER_MEAN_COL,
        PEER_STD_COL,
        PEER_COUNT_COL,
        "decl_year",
        "decl_month",
        "decl_dayofweek",
    ]
    candidate_categorical = [
        HS6_COL,
        ORIGIN_COL,
    ]

    numeric_features = [col for col in candidate_numeric if col in df.columns]
    categorical_features = [col for col in candidate_categorical if col in df.columns]

    if not numeric_features and not categorical_features:
        zeros = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        return zeros, zeros

    y = pd.to_numeric(model_df[LOG_PRICE_PER_KG_COL], errors="coerce")
    y_filled = y.fillna(y.median() if not np.isnan(y.median()) else 0.0)

    train_mask = y.notna()
    if train_mask.sum() < 10:
        zeros = pd.Series(np.zeros(len(model_df)), index=model_df.index, dtype=float)
        return zeros, zeros

    model = _build_residual_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X = model_df[numeric_features + categorical_features].copy()
    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("UNKNOWN").astype(str)

    try:
        model.fit(X.loc[train_mask], y_filled.loc[train_mask])
    except Exception:
        fallback_model = Pipeline(
            steps=[
                ("preprocessor", model.named_steps["preprocessor"]),
                ("model", LinearRegression()),
            ]
        )
        fallback_model.fit(X.loc[train_mask], y_filled.loc[train_mask])
        model = fallback_model

    predictions = pd.Series(model.predict(X), index=df.index)
    residual = y_filled - predictions
    residual_score = residual.abs()

    return residual, residual_score


def compute_rarity_score(df: pd.DataFrame) -> pd.Series:
    """Compute rarity score from inverse peer group size."""
    peer_count = pd.to_numeric(df[PEER_COUNT_COL], errors="coerce").fillna(0)
    return 1.0 / (peer_count + 1.0)


def compute_anomaly_signals(
    df: pd.DataFrame,
    gaussian_valid: bool,
    peer_keys: list[str] = PEER_KEYS,
) -> pd.DataFrame:
    """Compute all anomaly signals and return enriched dataframe."""
    _ = peer_keys
    out = df.copy()

    out[Z_SCORE_COL] = compute_z_score(out, gaussian_valid=gaussian_valid)
    out[IQR_SCORE_COL] = compute_iqr_score(out)

    residual, residual_score = compute_residual_score(out)
    out[RESIDUAL_COL] = residual
    out[RESIDUAL_SCORE_COL] = residual_score

    out[RARITY_SCORE_COL] = compute_rarity_score(out)

    for col in [Z_SCORE_COL, IQR_SCORE_COL, RESIDUAL_SCORE_COL, RARITY_SCORE_COL]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
