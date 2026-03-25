"""Configuration constants for the anomaly detection pipeline."""

from __future__ import annotations

from pathlib import Path

# Column names in the source dataset
DECLARATION_ID_COL = "Declaration ID"
HS6_COL = "HS6 Code"
ORIGIN_COL = "Country of Origin"
DEPARTURE_COL = "Country of Departure"
NET_MASS_COL = "Net Mass"
ITEM_PRICE_COL = "Item Price"
TAX_RATE_COL = "Tax Rate"
OFFICE_ID_COL = "Office ID"
MODE_TRANSPORT_COL = "Mode of Transport"
IMPORT_TYPE_COL = "Import Type"
TAX_TYPE_COL = "Tax Type"
DATE_COL = "Date"
PROCESS_TYPE_COL = "Process Type"
IMPORT_USE_COL = "Import Use"
PAYMENT_TYPE_COL = "Payment Type"
ORIGIN_INDICATOR_COL = "Country of Origin Indicator"

# Engineered columns
PRICE_PER_KG_COL = "price_per_kg"
LOG_PRICE_PER_KG_COL = "log_price_per_kg"
LOG_MASS_COL = "log_mass"

# Peer grouping
PEER_KEYS = [HS6_COL, ORIGIN_COL]

# Peer statistics columns
PEER_MEAN_COL = "peer_mean_log_price"
PEER_STD_COL = "peer_std_log_price"
PEER_MEDIAN_COL = "peer_median_price"
PEER_Q1_COL = "peer_q1_price"
PEER_Q3_COL = "peer_q3_price"
PEER_COUNT_COL = "peer_count"
PEER_SKEW_COL = "peer_skewness"
PEER_GAUSSIAN_OK_COL = "peer_gaussian_ok"

# Score columns
Z_SCORE_COL = "z_score"
IQR_SCORE_COL = "iqr_score"
RESIDUAL_COL = "residual"
RESIDUAL_SCORE_COL = "residual_score"
RARITY_SCORE_COL = "rarity_score"
RISK_SCORE_COL = "risk_score"
FLAG_COL = "flag"
EXPLANATION_COL = "explanation"

# Evaluation labels
GROUND_TRUTH_COL = "ground_truth"

# Default pipeline settings
TOP_K_RATIO = 0.05
PRECISION_AT_K_RATIO = 0.1
MIN_GROUP_SIZE = 3
EPSILON = 1e-9
RANDOM_SEED = 42

# Gaussian assumption settings
GAUSSIAN_SKEW_THRESHOLD = 1.0
GAUSSIAN_MIN_PEER_COUNT = 5
GAUSSIAN_MIN_OK_RATIO = 0.7

# Score aggregation weights
RISK_WEIGHTS = {
    RESIDUAL_SCORE_COL: 0.4,
    Z_SCORE_COL: 0.0, # Gaussian z-score is not used in final risk score due to violation of Gaussian assumption
    IQR_SCORE_COL: 0.4,
    RARITY_SCORE_COL: 0.2,
}

# Synthetic anomaly generation
SYNTHETIC_ANOMALY_RATIO = 0.12
PRICE_REDUCTION_FACTOR = 0.5
NOISE_STD = 0.35
BORDERLINE_IQR_FACTOR = 1.2
RARE_ORIGIN_TOKEN = "ZZ"
RARE_HS6_TOKEN = "999999"

# Dataset path
DEFAULT_DATA_PATH = Path(
    "./data/df_syn_eng.csv"
)
