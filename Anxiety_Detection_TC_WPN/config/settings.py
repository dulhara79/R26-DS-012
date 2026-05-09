"""
settings.py
TC-WPN Project Configuration
Updated for multi-source pipeline (v4):
  - MIMIC-IV paths (unchanged)
  - MIMIC-III paths (NEW)
  - MIMIC-III-Ext-Notes path (NEW)

Add these to your .env file:
  MIMIC_IV_DATASET_PATH=/path/to/mimic-iv/3.1
  MIMIC_IV_NOTE_DATASET_PATH=/path/to/mimic-iv-note/2.2
  MIMIC_III_DATASET_PATH=/path/to/mimic-iii/1.4          # NEW
  MIMIC_III_EXT_NOTES_PATH=/path/to/mimic-iii-ext-notes  # NEW
  MIMIC_PROCESSED_BASE_DIR=/path/to/output/processed
  MIMIC_PROCESSED_PKL_DIR=/path/to/output/processed/pkl

On Kaggle: set these as Dataset paths or environment variables in Cell 1.
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Project Root
PROJECT_ROOT = Path(__file__).parents[1].resolve()
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# MIMIC-IV PATHS
# =============================================================================
MIMIC_IV_DATASET_PATH = os.getenv("MIMIC_IV_DATASET_PATH")
MIMIC_IV_NOTE_DATASET_PATH = os.getenv("MIMIC_IV_NOTE_DATASET_PATH")

# =============================================================================
# MIMIC-III PATHS  (NEW v4)
# PhysioNet: https://physionet.org/content/mimiciii/1.4/
# Contains: NOTEEVENTS.csv.gz, DIAGNOSES_ICD.csv.gz,
#           PRESCRIPTIONS.csv.gz, PATIENTS.csv.gz, ADMISSIONS.csv.gz
# =============================================================================
MIMIC_III_DATASET_PATH = os.getenv("MIMIC_III_DATASET_PATH", "")

# =============================================================================
# MIMIC-III-EXT-NOTES PATH  (NEW v4)
# PhysioNet: https://physionet.org/content/mimic-iii-ext-notes/1.0.0/
# Contains: labels.csv, notes.csv
# labels.csv columns: row_id, trigger_word, concept, semtypes,
#                     start, end, detection, encounter, negation
# =============================================================================
MIMIC_III_EXT_NOTES_PATH = os.getenv("MIMIC_III_EXT_NOTES_PATH", "")

# =============================================================================
# PROCESSED DATA PATHS
# =============================================================================
MIMIC_PROCESSED_BASE_DIR = os.getenv(
    "MIMIC_PROCESSED_BASE_DIR",
    str(PROJECT_ROOT / "data" / "processed"),
)
MIMIC_PROCESSED_PKL_DIR = os.getenv(
    "MIMIC_PROCESSED_PKL_DIR",
    str(PROJECT_ROOT / "data" / "processed" / "pkl"),
)
MIMIC_ANALYSIS_PATH = os.getenv(
    "MIMIC_ANALYSIS_PATH",
    str(PROJECT_ROOT / "data" / "processed" / "mimic_analysis"),
)

# =============================================================================
# SPLIT FILE PATHS
# =============================================================================
MIMIC_TRAIN_BALANCED_PATH = os.getenv(
    "MIMIC_TRAIN_BALANCED_PATH",
    str(Path(MIMIC_PROCESSED_BASE_DIR) / "mimic_anxiety_train_balanced.csv"),
)
MIMIC_TRAIN_HIGH_CONF_PATH = os.getenv(
    "MIMIC_TRAIN_HIGH_CONF_PATH",
    str(Path(MIMIC_PROCESSED_BASE_DIR) / "mimic_anxiety_train_high_conf.csv"),
)
MIMIC_VAL_PATH = os.getenv(
    "MIMIC_VAL_PATH",
    str(Path(MIMIC_PROCESSED_BASE_DIR) / "mimic_anxiety_val_real_world.csv"),
)
MIMIC_TEST_PATH = os.getenv(
    "MIMIC_TEST_PATH",
    str(Path(MIMIC_PROCESSED_BASE_DIR) / "mimic_anxiety_test_real_world.csv"),
)
MIMIC_TEST_HIGH_CONF_PATH = os.getenv(
    "MIMIC_TEST_HIGH_CONF_PATH",
    str(Path(MIMIC_PROCESSED_BASE_DIR) / "mimic_anxiety_test_high_conf.csv"),
)

# =============================================================================
# VALIDATION WARNINGS
# =============================================================================
if not MIMIC_IV_DATASET_PATH:
    warnings.warn("MIMIC_IV_DATASET_PATH not set.")
if not MIMIC_IV_NOTE_DATASET_PATH:
    warnings.warn("MIMIC_IV_NOTE_DATASET_PATH not set.")
if not MIMIC_III_DATASET_PATH:
    warnings.warn(
        "MIMIC_III_DATASET_PATH not set — MIMIC-III notes will be skipped. "
        "Set this to significantly improve training corpus size and AUROC."
    )
if not MIMIC_III_EXT_NOTES_PATH:
    warnings.warn(
        "MIMIC_III_EXT_NOTES_PATH not set — Ext-Notes negation patterns "
        "will be skipped. Default regex negation will be used."
    )
