# Shipment Anomaly Detection Pipeline

This project provides a modular anomaly detection pipeline for customs shipment declarations.

## Structure

- `src/config.py`: constants, thresholds, and score weights
- `src/data_loader.py`: loading and cleaning
- `src/feature_engineering.py`: derived price and log features
- `src/peer_grouping.py`: peer-level statistics by HS6 + origin
- `src/gaussian_check.py`: skewness-based Gaussian reasonableness check
- `src/anomaly_detection.py`: z-score, IQR, residual, rarity signals
- `src/risk_scoring.py`: normalization, weighted risk score, top-k flagging
- `src/explanation.py`: human-readable explanations
- `src/synthetic.py`: synthetic anomaly generation with labels
- `src/evaluation.py`: Precision@K, and Recall@K
- `main.py`: end-to-end execution script

## Feature Engineering

The following table describes the engineered features created during the data preprocessing phase:

| Feature Name | Column Name | Description |
|---|---|---|
| Price per Kilogram | `price_per_kg` | Derived feature calculated as Item Price divided by Net Mass. Represents the unit price per kilogram of the shipment. |
| Log Price Per Kilogram | `log_price_per_kg` | Log-transformed version of price_per_kg using log1p transformation. Helps normalize the distribution of price_per_kg values for anomaly detection. |
| Log Mass | `log_mass` | Log-transformed version of Net Mass using log1p transformation. Helps normalize the distribution of shipment mass values for anomaly detection. |
| IQR Score | `iqr_score` | Outlier score based on distance from the peer-group interquartile range (IQR). Higher values indicate stronger deviation from the typical peer-group price range. |
| Rarity Score | `rarity_score` | Rarity-based score derived from how uncommon a shipment's peer-group characteristics are in the dataset. Higher values indicate rarer, potentially anomalous declarations. |

## Data Path

Default input path is configured as:

`data\df_syn_eng.csv`

## Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Execute:

```bash
python main.py
```

## Output

Generated files are written to `outputs/`:

- `scored_shipments.csv`
- `synthetic_scored_shipments.csv`

The scored output includes:

- individual scores: `z_score`, `iqr_score`, `residual_score`, `rarity_score`
- normalized scores with `_norm` suffix
- `risk_score`
- `flag` (top 10%)
- `explanation`

## Notes

- Edge cases are handled for missing values, small peer groups, and safe division.
- Z-score is effectively disabled when Gaussian assumption is globally not reasonable.
- Evaluation is performed on synthetic anomalies with configurable Precision@K and Recall@K.
