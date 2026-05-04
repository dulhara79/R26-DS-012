"""C3 singleton artifact loader.

All ML artifacts (XGBoost, calibrator, conformal predictor, SHAP explainer,
FAISS index, seed case base, NL templates) are loaded ONCE at startup and
cached in the ModelRegistry singleton.

If optional artifacts (FAISS index, counterfactual engine, LIME results) are
missing, the registry logs a warning and provides graceful fallback.
"""
from __future__ import annotations

import json
import logging
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.config import ARTIFACTS_DIR, FEATURE_COLS, SHAP_WEIGHTS

logger = logging.getLogger("c3.model_loader")


# ---------------------------------------------------------------------------
# File paths — relative to ARTIFACTS_DIR
# ---------------------------------------------------------------------------
_PATHS = {
    "xgboost":               "xgboost_smotenc.pkl",
    "calibrator":            "probability_calibrator.pkl",
    "conformal":             "conformal_predictor.pkl",
    "shap_explainer":        "xai_shap_explainer.pkl",
    "counterfactual_engine": "xai_counterfactual_engine.pkl",
    "nl_template":           "xai_nl_template.json",
    "lime_results":          "xai_lime_results.json",
    "seed_case_base":        "seed_case_base.csv",
    "feature_cols":          "feature_cols.json",
    "faiss_index":           "faiss_rawspace.index",
    "faiss_metadata":        "faiss_metadata.json",
}

# Required — startup fails if any of these are missing
_REQUIRED = {"xgboost", "calibrator", "conformal", "seed_case_base"}


class ModelRegistry:
    """Singleton that holds all loaded artifacts."""

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return
        # XGBoost classifier (calibrated pipeline)
        self.xgboost: Any = None
        # Isotonic probability calibrator
        self.calibrator: Any = None
        # Conformal predictor (APS)
        self.conformal: Any = None
        # SHAP TreeExplainer
        self.shap_explainer: Any = None
        # DiCE counterfactual engine (optional)
        self.counterfactual_engine: Any = None
        # NL explanation templates (optional, falls back to default strings)
        self.nl_template: dict = {}
        # LIME results cache (optional)
        self.lime_results: dict = {}
        # Seed case base — always available (required)
        self.seed_case_base: pd.DataFrame | None = None
        # FAISS index (optional — fallback to euclidean L2)
        self.faiss_index: Any = None
        self.faiss_metadata: list[dict] = []
        self.faiss_available: bool = False
        # Feature column order
        self.feature_cols: list[str] = list(FEATURE_COLS)

        self._initialised = True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_all(self, artifacts_dir: Path | None = None) -> None:
        """Load every artifact. Required artifacts fail fast; optional ones warn."""
        art = Path(artifacts_dir or ARTIFACTS_DIR)
        if not art.exists():
            raise RuntimeError(f"Artifacts directory missing: {art}")

        logger.info(f"Loading artifacts from {art}")
        missing_required = [
            k for k in _REQUIRED if not (art / _PATHS[k]).exists()
        ]
        if missing_required:
            raise RuntimeError(
                f"Required artifacts missing: {missing_required}. "
                f"Cannot start API without these."
            )

        # --- required ---
        raw = self._load_pickle(art / _PATHS["xgboost"]); self.xgboost = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        self.calibrator = self._load_pickle(art / _PATHS["calibrator"])
        self.conformal = self._load_pickle(art / _PATHS["conformal"])
        self.seed_case_base = pd.read_csv(art / _PATHS["seed_case_base"])
        logger.info(f"  seed_case_base: {len(self.seed_case_base)} rows")

        # --- optional: feature columns override ---
        if (art / _PATHS["feature_cols"]).exists():
            with open(art / _PATHS["feature_cols"]) as f:
                raw = json.load(f); self.feature_cols = raw["feature_cols"] if isinstance(raw, dict) and "feature_cols" in raw else raw
                logger.info(f"  feature_cols: {self.feature_cols}")

        # --- optional: SHAP explainer ---
        if (art / _PATHS["shap_explainer"]).exists():
            try:
                self.shap_explainer = self._load_pickle(art / _PATHS["shap_explainer"])
                logger.info("  shap_explainer: loaded")
            except Exception as e:
                logger.warning(f"  shap_explainer: load failed ({e}), using fallback")

        # --- optional: DiCE counterfactuals ---
        if (art / _PATHS["counterfactual_engine"]).exists():
            try:
                self.counterfactual_engine = self._load_pickle(
                    art / _PATHS["counterfactual_engine"]
                )
                logger.info("  counterfactual_engine: loaded")
            except Exception as e:
                logger.warning(f"  counterfactual_engine: load failed ({e})")

        # --- optional: NL templates ---
        if (art / _PATHS["nl_template"]).exists():
            try:
                with open(art / _PATHS["nl_template"]) as f:
                    self.nl_template = json.load(f)
                logger.info("  nl_template: loaded")
            except Exception as e:
                logger.warning(f"  nl_template: load failed ({e})")

        # --- optional: LIME cache ---
        if (art / _PATHS["lime_results"]).exists():
            try:
                with open(art / _PATHS["lime_results"]) as f:
                    self.lime_results = json.load(f)
                logger.info("  lime_results: loaded")
            except Exception as e:
                logger.warning(f"  lime_results: load failed ({e})")

        # --- optional: FAISS index ---
        faiss_path = art / _PATHS["faiss_index"]
        faiss_meta_path = art / _PATHS["faiss_metadata"]
        if faiss_path.exists() and faiss_meta_path.exists():
            try:
                import faiss  # noqa: WPS433 (lazy import — faiss is heavy)
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(faiss_meta_path) as f:
                    self.faiss_metadata = json.load(f)
                self.faiss_available = True
                logger.info(f"  faiss_index: {self.faiss_index.ntotal} vectors loaded")
            except Exception as e:
                logger.warning(
                    f"  faiss_index: load failed ({e}), using euclidean fallback"
                )
                self.faiss_available = False
        else:
            logger.warning(
                "  faiss_index: not present, using euclidean L2 fallback on seed_case_base"
            )

        logger.info("All artifacts loaded.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with open(path, "rb") as f:
            return joblib.load(f)

    def get_shap_weight_vector(self) -> np.ndarray:
        """Return SHAP weights as a numpy array aligned with feature_cols."""
        return np.array(
            [SHAP_WEIGHTS.get(c, 0.0) for c in self.feature_cols],
            dtype=np.float32,
        )


# Module-level singleton accessor
registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """FastAPI dependency — returns the singleton."""
    return registry





