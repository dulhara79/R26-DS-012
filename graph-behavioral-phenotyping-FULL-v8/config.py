"""Configuration for Component 2 consolidated v8."""
from pathlib import Path
import os

SEED = 42

COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
COHORT_YEAR = {
    "INS-W_1": 2018,
    "INS-W_2": 2019,
    "INS-W_3": 2020,
    "INS-W_4": 2021,
}

DEV_COHORT = "INS-W_1"
FINAL_COHORTS = ["INS-W_2", "INS-W_3", "INS-W_4"]

GLOBEM_ROOT = Path(os.environ.get("GLOBEM_ROOT", "./globem")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs_v8")).expanduser()

ANXIETY_COL = "anx_weekly_subscale"
LABEL_FILE = "dep_weekly.csv"

USE_SEGMENTS = ["morning", "afternoon", "evening", "night"]
WINDOW_DAYS = 28

MAX_BASES = 48
MAX_BASES_PER_SENSOR = 10
MIN_BASE_COVERAGE = 0.35
MIN_DAYS = 14
MIN_NODES = 12

N_SPLITS = 5
N_REPEATS = 3
INNER_VAL_FRAC = 0.20
EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 64

CALIBRATION = "auto"
CALIB_MIN_ISOTONIC = 400

N_BOOT = 2000
N_PERM = 50
PERM_EPOCHS = 60
PERM_REPEATS = 1
SEARCH_REPEATS = 5

MODEL_CONFIG = {
    "hidden": 32,
    "heads": 2,
    "drop": 0.50,
}

FUSION_WEIGHT = 0.0
