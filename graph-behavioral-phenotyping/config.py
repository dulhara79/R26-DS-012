"""
config.py
=========
Central configuration for the Graph-Based Spatio-Temporal Behavioral
Phenotyping pipeline. Edit the PATH variables to match your environment,
or override them via environment variables before running.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
# Set these via env-vars so collaborators can point to their own dataset
# without editing source code:
#   export DATASET_PATH=/path/to/studentlife/dataset/
#   export OUTPUT_DIR=/path/to/outputs/
#   export MODELS_DIR=/path/to/models/

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/content/drive/MyDrive/Anxiety/dataset/archive (1)/dataset/"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/content/drive/MyDrive/Anxiety/outputs/"
)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    "/content/drive/MyDrive/Anxiety/models/"
)

# ── Time windows for feature bucketing ───────────────────────────────────────
TIME_WINDOWS = {
    'morning'  : (6,  12),
    'afternoon': (12, 18),
    'evening'  : (18, 23),
    'night'    : (23, 30),   # 23–06 wraps
}
WINDOW_ORDER = ['morning', 'afternoon', 'evening', 'night']

# ── PHQ-9 response mapping ────────────────────────────────────────────────────
PHQ_MAP = {
    'Not at all'              : 0,
    'Several days'            : 1,
    'More than half the days' : 2,
    'Nearly every day'        : 3,
    'nan'                     : 0,
}
SCORE_COLS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_CONFIG = {
    'node_feat': 9,
    'hidden'   : 64,
    'heads'    : 4,
    'drop'     : 0.3,
}

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    'epochs'        : 200,
    'lr'            : 0.001,
    'weight_decay'  : 1e-4,
    'batch_size'    : 4,
    'patience'      : 40,
    'lr_patience'   : 15,
    'lr_factor'     : 0.5,
    'cv_folds'      : 5,
    'random_state'  : 42,
    'final_epochs'  : 80,
}

# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    'CRITICAL' : 0.7,
    'HIGH'     : 0.5,
    'MODERATE' : 0.3,
}

STRESS_THRESHOLD     = 3.0   # stress level >= this → high-stress
VULNERABILITY_CUTOFF = 0.5   # vulnerability score >= this → high vulnerability

# ── GPS cleaning ─────────────────────────────────────────────────────────────
GPS_ACCURACY_LIMIT  = 100    # metres
GPS_MAX_SPEED_MPS   = 55     # metres/second (~200 km/h)
STAY_POINT_RADIUS_M = 50     # metres
STAY_POINT_MIN_PTS  = 5      # DBSCAN min_samples
MAX_EDGE_GAP_MIN    = 240    # minutes; transitions > this are dropped

# ── Phenotyping ──────────────────────────────────────────────────────────────
PHENOTYPE_LABELS = {
    2: {
        0: 'Low Vulnerability',
        1: 'High Vulnerability',
    },
    3: {
        0: 'Social-Spatial Withdrawal',
        1: 'Circadian Disruption',
        2: 'Hypervigilant Mobility',
    },
    4: {
        0: 'Social-Spatial Withdrawal',
        1: 'Circadian Disruption',
        2: 'Hypervigilant Mobility',
        3: 'Irregular Patterns',
    },
}
