"""C3 FastAPI application entry point.

Startup sequence (via lifespan context manager):
  1. Load all ML artifacts via ModelRegistry singleton
  2. Register routers
  3. Enable CORS for Flutter development

Production command:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import clinician, recommend, risk, session, user
from app.services.model_loader import get_registry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("c3")


# ---------------------------------------------------------------------------
# Lifespan — loads all artifacts ONCE at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("C3 API starting — loading artifacts ...")
    registry = get_registry()
    registry.load_all()
    logger.info("C3 API ready.")
    yield
    logger.info("C3 API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="C3 — Continuously Learning Personalised Anxiety Intervention",
    description=(
        "Phase 3 FastAPI backend for R26-DS-012. "
        "Wraps the XGBoost + SHAP + FAISS pipeline behind a REST API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permissive for development. Tighten before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(risk.router)
app.include_router(recommend.router)
app.include_router(session.router)
app.include_router(user.router)
app.include_router(clinician.router)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "C3 API",
        "version": "1.0.0",
        "status":  "ok",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check — reports which artifacts loaded."""
    r = get_registry()
    return {
        "status":                "ok",
        "xgboost_loaded":        r.xgboost is not None,
        "calibrator_loaded":     r.calibrator is not None,
        "conformal_loaded":      r.conformal is not None,
        "shap_loaded":           r.shap_explainer is not None,
        "faiss_available":       r.faiss_available,
        "seed_case_base_rows":   0 if r.seed_case_base is None else len(r.seed_case_base),
    }
