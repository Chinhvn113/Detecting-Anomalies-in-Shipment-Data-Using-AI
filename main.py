"""Main script for shipment anomaly detection pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.anomaly_detection import compute_anomaly_signals
from src.config import (
    DEFAULT_DATA_PATH,
    FLAG_COL,
    GROUND_TRUTH_COL,
    PEER_GAUSSIAN_OK_COL,
    PEER_KEYS,
    PRECISION_AT_K_RATIO,
    RISK_SCORE_COL,
    TOP_K_RATIO,
)
from src.data_loader import clean_dataset, load_dataset
from src.evaluation import evaluate_predictions
from src.explanation import add_explanations
from src.feature_engineering import add_price_mass_features
from src.gaussian_check import compute_peer_skewness, is_gaussian_reasonable
from src.peer_grouping import attach_peer_statistics, compute_peer_statistics
from src.risk_scoring import compute_risk_score, flag_top_k_percent
from src.synthetic import generate_synthetic_dataset


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Run the full anomaly detection pipeline on input data."""
    engineered = add_price_mass_features(df)

    peer_stats = compute_peer_statistics(engineered, peer_keys=PEER_KEYS)
    enriched = attach_peer_statistics(engineered, peer_stats, peer_keys=PEER_KEYS)

    peer_skew = compute_peer_skewness(engineered, peer_keys=PEER_KEYS)
    gaussian_valid = is_gaussian_reasonable(peer_skew)

    enriched = enriched.merge(
        peer_skew[PEER_KEYS + [PEER_GAUSSIAN_OK_COL]],
        on=PEER_KEYS,
        how="left",
    )
    enriched[PEER_GAUSSIAN_OK_COL] = enriched[PEER_GAUSSIAN_OK_COL].fillna(False)

    scored = compute_anomaly_signals(enriched, gaussian_valid=gaussian_valid)
    scored = compute_risk_score(scored)
    scored = flag_top_k_percent(scored, k_ratio=TOP_K_RATIO, score_col=RISK_SCORE_COL)
    scored = add_explanations(scored)

    return scored, gaussian_valid


def run_main(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Execute pipeline for production scoring and synthetic evaluation."""
    raw_df = load_dataset(data_path)
    clean_df = clean_dataset(raw_df)

    scored_df, gaussian_valid = run_pipeline(clean_df)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    production_output_path = output_dir / "scored_shipments.csv"
    production_anomalies_df = scored_df[scored_df[FLAG_COL]].copy()
    production_anomalies_df.to_csv(production_output_path, index=False)

    synthetic_df = generate_synthetic_dataset(clean_df)
    synthetic_scored_df, synthetic_gaussian_valid = run_pipeline(synthetic_df)

    if GROUND_TRUTH_COL in synthetic_scored_df.columns:
        metrics = evaluate_predictions(
            y_true=synthetic_scored_df[GROUND_TRUTH_COL],
            y_pred=synthetic_scored_df[FLAG_COL],
            scores=synthetic_scored_df[RISK_SCORE_COL],
            k_ratio=PRECISION_AT_K_RATIO,
        )
    else:
        metrics = {"f1_score": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

    synthetic_output_path = output_dir / "synthetic_scored_shipments.csv"
    synthetic_anomalies_df = synthetic_scored_df[synthetic_scored_df[FLAG_COL]].copy()
    synthetic_anomalies_df.to_csv(synthetic_output_path, index=False)

    print("Pipeline completed.")
    print(f"Rows scored: {len(scored_df)}")
    print(f"Production anomalies saved: {len(production_anomalies_df)}")
    print(f"Gaussian assumption valid (production): {gaussian_valid}")
    print(f"Gaussian assumption valid (synthetic): {synthetic_gaussian_valid}")
    print(f"Synthetic anomalies saved: {len(synthetic_anomalies_df)}")
    # print(f"F1-score on synthetic data: {metrics['f1_score']:.4f}")
    print(f"Precision@K on synthetic data: {metrics['precision_at_k']:.4f}")
    print(f"Recall@K on synthetic data: {metrics['recall_at_k']:.4f}")
    print(f"Production output: {production_output_path.resolve()}")
    print(f"Synthetic output: {synthetic_output_path.resolve()}")


if __name__ == "__main__":
    run_main()
