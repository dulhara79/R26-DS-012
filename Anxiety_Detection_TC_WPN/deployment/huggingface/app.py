"""
app.py — TC-WPN deployment wrapper (Phase 3).

This is a DEPLOYMENT LAYER, not a copy of the research repository (§14).
It answers exactly one question (§12):

    "Given this note text and this support set, what does TC-WPN predict?"

It does NOT answer "what is the final multimodal anxiety risk?" — that belongs
to the central backend. There is no fusion logic in this file and none should
be added to it.

STRUCTURE OF THE CHANGES, KEYED TO THE REVIEW
=============================================
KEPT (§15): FastAPI, authentication, /predict, /health, structured support_set,
note_date, temporal metadata, strict checkpoint loading, Gradio demo, model
version metadata, uncertainty/entropy output.

CHANGED (§15):
  1. Silent default support sets removed from the production API. The demo UI
     keeps them. A missing support class now returns the §6 error envelope.
  2. Explicit model/checkpoint version — model_version comes from
     deployment_config.json, not a hardcoded "TC-WPN v1.0".
  3. preprocessing_version returned on every prediction.
  4. inference_configuration returned on every prediction.
  5. Support-set composition validated: >= 1 anxiety AND >= 1 control.
  6. Explicit startup failure when the checkpoint's cfg disagrees with
     deployment_config.json.
  7. Research metrics separated from inference output — AUROC appears only in
     /health, never in a /predict response.
  8. Attention spans renamed `attention_based_highlighted_spans` and returned
     with an explicit non-faithfulness note. They are not called explanations.

  Additionally, per §8: the four-band LOW/MODERATE/HIGH/VERY HIGH output is
  returned as `application_risk_band`, clearly labelled as an application-layer
  interpretation, because the model was evaluated as a binary classifier and
  those four bands were never validated as clinical classes.

  Per §7: AUROC (threshold-free discrimination) and the operating threshold
  (selected on validation) are reported as separate, differently-named things.

MODEL SOURCE (Phase 1 / Phase 3)
================================
This wrapper imports `tc_wpn.model` — the VENDORED copy of
tcwpn_test/src/tcwpn/model.py, the final validated implementation. It does NOT
import the older tc_wpn/models/core.py architecture that the previous Space
used. Run vendor.sh to populate it.

One consequence worth knowing: the final model.py removed the BiGRU
TemporalEncoder that the old core.py applied to support embeddings. The old
Space wrapper never called that encoder, so its prototypes were built in a
different representation space from its queries. Building this wrapper on the
final model.py removes that failure mode structurally — there is no encoder
left to forget to call. `build_prototype` and `classify` are public forward-only
methods and are called directly here rather than re-implemented, which is the
only way to guarantee Phase 7 parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, login

import gradio as gr
from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tc_wpn.model import build_model                      # vendored, Phase 3
from tc_wpn.preprocessing import pack_notes               # vendored, §3
from accounts import router as auth_router, verify_bearer  # unchanged

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

# =============================================================================
# DEPLOYMENT CONFIG — the single versioned unit (§3, §17)
# =============================================================================
with open(os.path.join(HERE, "deployment_config.json")) as fh:
    CFG: Dict[str, Any] = json.load(fh)

PROV = CFG["provenance"]
PRE = CFG["preprocessing"]
INF = CFG["inference_configuration"]
OP = CFG["operating_point"]
RESEARCH = CFG["research_metrics"]
API = CFG["api"]

MODEL_VERSION = CFG["model_version"]
PREPROCESSING_VERSION = CFG["preprocessing_version"]
CONTRACT_VERSION = API["contract_version"]

MODEL_REPO = PROV["hf_model_repo"]
MODEL_FILE = PROV["checkpoint_filename"]
MODEL_REVISION = PROV.get("hf_model_repo_revision") or None

REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "true").lower() != "false"
STRICT_STARTUP = os.environ.get("STRICT_STARTUP", "true").lower() != "false"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

_model = None
_tokenizer = None
_startup: Dict[str, Any] = {"ready": False, "errors": [], "checks": {}}


class DeploymentIntegrityError(RuntimeError):
    """§15.6 — raised when the checkpoint is not the one this config describes."""


def _unfilled(value) -> bool:
    return isinstance(value, str) and value.strip().startswith("<FILL")


# =============================================================================
# §15.6 — CHECKPOINT / CONFIG MATCH
# =============================================================================
def _verify_config_complete():
    """Phase 2. A deployment with unfilled provenance is not verifiable, so it
    does not start. Guessing any of these is exactly the failure §16 warns
    about: everything runs, but it is not the model you evaluated."""
    missing = []
    for key in ("tcwpn_git_commit", "checkpoint_filename", "checkpoint_sha256"):
        if _unfilled(PROV.get(key)):
            missing.append(f"provenance.{key}")
    if _unfilled(MODEL_VERSION):
        missing.append("model_version")
    if _unfilled(OP.get("threshold")):
        missing.append("operating_point.threshold")
    if missing:
        raise DeploymentIntegrityError(
            "deployment_config.json still contains <FILL> placeholders: "
            + ", ".join(missing)
            + ". Complete Phase 1 and Phase 2 before deploying."
        )


def _verify_checkpoint_hash(path: str):
    """§17 — the uploaded artefact must be the artefact the config names."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    expected = PROV["checkpoint_sha256"]
    _startup["checks"]["checkpoint_sha256"] = digest
    if digest != expected:
        raise DeploymentIntegrityError(
            f"Checkpoint hash mismatch. deployment_config.json expects "
            f"{expected}, the file downloaded from {MODEL_REPO}/{MODEL_FILE} "
            f"hashes to {digest}. Either the model repo was updated without "
            f"updating this config, or the wrong file is being served."
        )


def _verify_architecture(ckpt: dict):
    """§15.6 + §3. scripts/train.py saves {"model", "cfg", "k", "seed", "step"}.
    `cfg` is the training config, so the checkpoint carries the architecture it
    was trained with and we can compare rather than assume."""
    train_cfg = (ckpt.get("cfg") or {}).get("model")
    if not isinstance(train_cfg, dict):
        raise DeploymentIntegrityError(
            "Checkpoint has no cfg.model block. It was not produced by "
            "scripts/train.py at the pinned commit, so its architecture cannot "
            "be verified against deployment_config.json."
        )

    kmap = INF["_keyword_map"]
    # supervisor-facing name -> (value we expect, key as it appears in cfg.model)
    expected = {
        "preset": (INF["preset"], "preset"),
        "encoder_name": (INF["encoder_name"], "encoder_name"),
        "projection_dim": (INF["projection_dim"], "projection_dim"),
        "lambda_decay": (INF["lambda_decay"], kmap["lambda_decay"]),
        "beta": (INF["beta"], kmap["beta"]),
        "init_temperature": (INF["init_temperature"], "init_temperature"),
        "consistency_passes": (INF["consistency_passes"], "consistency_passes"),
    }

    mismatches = []
    for label, (want, cfg_key) in expected.items():
        got = train_cfg.get(cfg_key)
        if got is None:
            continue  # key absent from the training config; preset supplies it
        if isinstance(want, float) or isinstance(got, float):
            ok = abs(float(got) - float(want)) < 1e-9
        else:
            ok = got == want
        if not ok:
            mismatches.append(f"{label}: config says {want!r}, checkpoint says {got!r}")

    _startup["checks"]["checkpoint_cfg"] = train_cfg
    _startup["checks"]["k_shot"] = ckpt.get("k")
    _startup["checks"]["seed"] = ckpt.get("seed")

    if mismatches:
        raise DeploymentIntegrityError(
            "Checkpoint architecture does not match deployment_config.json:\n  - "
            + "\n  - ".join(mismatches)
        )


# =============================================================================
# LOAD
# =============================================================================
def load_model():
    """Strict checkpoint loading is KEPT (§15). A key mismatch means the
    vendored model.py has drifted from the checkpoint, and that must fail
    loudly rather than serve a partially initialised model that still returns
    confident-looking probabilities."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _verify_config_complete()

    _tokenizer = AutoTokenizer.from_pretrained(PRE["tokenizer_name"])

    path = hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE,
        revision=MODEL_REVISION, token=hf_token,
    )
    _verify_checkpoint_hash(path)

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    _verify_architecture(ckpt)

    kmap = INF["_keyword_map"]
    model = build_model({
        "preset": INF["preset"],
        "encoder_name": INF["encoder_name"],
        "projection_dim": INF["projection_dim"],
        "freeze_bert": INF["freeze_bert"],
        "init_temperature": INF["init_temperature"],
        kmap["lambda_decay"]: INF["lambda_decay"],
        kmap["beta"]: INF["beta"],
        "consistency_passes": INF["consistency_passes"],
    }).to(DEVICE)

    model.load_state_dict(ckpt["model"], strict=True)   # KEPT: strict
    model.eval()

    _model = model
    _startup["ready"] = True
    return _model, _tokenizer


# =============================================================================
# TEMPORAL METADATA (§5 — kept, and made to mean what it meant in training)
# =============================================================================
def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    except Exception:
        return None


def compute_days_before_last(support_dates: List[Optional[datetime]],
                             query_date: Optional[datetime]):
    """
    The model's temporal input is `days_before_patient_last_note`
    (src/tcwpn/collate.py) — how far a note sits before the LAST note observed
    for that SAME patient. It is a within-patient quantity, not "days before
    today".

    Reference point = the most recent date among the support notes and the
    query note. A note with no date cannot be placed on that axis; rather than
    inventing a position for it, it takes the median of the dated notes and is
    counted in `undated_support_notes` on the response, so the caller can see
    the input was incomplete.

    Returns (days [K] float list, n_undated int, reference ISO string or None).
    """
    dated = [d for d in support_dates if d is not None]
    if query_date is not None:
        dated_with_query = dated + [query_date]
    else:
        dated_with_query = dated

    if not dated_with_query:
        return [0.0] * len(support_dates), len(support_dates), None

    reference = max(dated_with_query)
    known = [max(0.0, (reference - d).total_seconds() / 86400.0)
             for d in support_dates if d is not None]
    fill = sorted(known)[len(known) // 2] if known else 0.0

    days, n_undated = [], 0
    for d in support_dates:
        if d is None:
            days.append(fill)
            n_undated += 1
        else:
            days.append(max(0.0, (reference - d).total_seconds() / 86400.0))
    return days, n_undated, reference.isoformat()


# =============================================================================
# INFERENCE — calls the vendored model's own methods (Phase 7 parity)
# =============================================================================
@torch.no_grad()
def run_inference(note_text: str,
                  support_texts: List[str],
                  support_labels: List[int],
                  support_days: List[float],
                  n_undated: int,
                  reference_date: Optional[str],
                  used_default_support_set: bool,
                  return_attention: bool = False,
                  return_support_contributions: bool = False) -> dict:
    model, tok = load_model()

    sup_ids, sup_mask, sup_idx = pack_notes(
        support_texts, tok, PRE["max_len"], PRE["max_chunks"], PRE["stride"], DEVICE)
    qry_ids, qry_mask, qry_idx = pack_notes(
        [note_text], tok, PRE["max_len"], PRE["max_chunks"], PRE["stride"], DEVICE)

    sup_emb = model.embedder(sup_ids, sup_mask, sup_idx)
    qry_emb = model.embedder(qry_ids, qry_mask, qry_idx)

    labels = torch.tensor(support_labels, dtype=torch.long, device=DEVICE)
    days = torch.tensor(support_days, dtype=torch.float32, device=DEVICE)

    # Class order is derived exactly as PrototypicalModel.forward derives it:
    # sorted(set(support labels)). With labels {0,1} that is [0, 1], so column 1
    # is the anxiety class. Derived, not assumed.
    classes = sorted(set(support_labels))
    prototypes, weights = [], {}
    for c in classes:
        mask = labels == c
        proto, w = model.build_prototype(sup_emb[mask], days[mask])
        prototypes.append(proto)
        weights[c] = w

    logits = model.classify(qry_emb, prototypes)
    probs = F.softmax(logits, dim=-1)
    positive_col = {c: i for i, c in enumerate(classes)}.get(1, logits.size(1) - 1)
    probability = float(probs[0, positive_col].item())

    threshold = float(OP["threshold"])
    prediction = "ANXIETY" if probability >= threshold else "NO ANXIETY"
    confidence = probability if probability >= threshold else 1.0 - probability

    p = min(max(probability, 1e-9), 1 - 1e-9)
    entropy = float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))

    # §8 — an application-layer interpretation, explicitly named as one. The
    # model was evaluated as a binary classifier; these four bands were never
    # validated as clinical classes.
    if probability >= 0.85:
        band = "VERY HIGH"
    elif probability >= 0.70:
        band = "HIGH"
    elif probability >= threshold:
        band = "MODERATE"
    else:
        band = "LOW"

    n_anx = sum(1 for l in support_labels if l == 1)
    n_ctrl = sum(1 for l in support_labels if l == 0)

    resp = {
        # ---- §11 contract ----------------------------------------------------
        "model": "TC-WPN",
        "model_version": MODEL_VERSION,                       # §15.2
        "prediction": prediction,
        "probability": round(probability, 6),
        "threshold": threshold,
        "confidence": round(float(confidence), 6),
        "entropy": round(entropy, 6),
        "support_count": {"anxiety": n_anx, "control": n_ctrl},
        "temporal_weighting_used": bool(model.use_temporal_weight),
        "used_default_support_set": bool(used_default_support_set),

        # ---- §15.3 / §15.4 ---------------------------------------------------
        "preprocessing_version": PREPROCESSING_VERSION,
        "inference_configuration": {
            "projection_dim": INF["projection_dim"],
            "lambda_decay": INF["lambda_decay"],
            "beta": INF["beta"],
            "aux_weight": INF["aux_weight"],
        },

        # ---- provenance & interpretation ------------------------------------
        "contract_version": CONTRACT_VERSION,
        "tcwpn_git_commit": PROV["tcwpn_git_commit"],
        "prototype_consistency_weighting_used": bool(model.use_pcw),
        "temperature": round(float(torch.exp(model.log_temperature).item()), 4),
        "undated_support_notes": n_undated,
        "temporal_reference_note_date": reference_date,
        "probability_semantics": (
            "softmax over cosine-distance prototype logits. Uncalibrated: no "
            "calibrator is fitted in this deployment."
        ),
        # §8 — named as application layer, not a validated model output.
        "application_risk_band": band,
        "application_risk_band_note": (
            "Application-layer UI band, not a validated model output. TC-WPN "
            "was evaluated as a binary classifier; LOW/MODERATE/HIGH/VERY HIGH "
            "were not validated as clinical classes."
        ),
    }

    if return_support_contributions:
        contribs = []
        for c in classes:
            idx = [i for i, l in enumerate(support_labels) if l == c]
            for i, w in zip(idx, weights[c].tolist()):
                contribs.append({
                    "support_index": i,
                    "label": "anxiety" if c == 1 else "control",
                    "excerpt": support_texts[i][:120],
                    "weight": round(float(w), 6),
                    "days_before_last_note": round(support_days[i], 3),
                })
        resp["support_contributions"] = contribs

    if return_attention:
        # §9 — NOT called an explanation, and not described as the features
        # responsible for the prediction.
        resp["attention_based_highlighted_spans"] = _highlighted_spans(model, tok, note_text)
        resp["attention_note"] = (
            "Attention-derived textual cues. Attention weights are not "
            "automatically faithful feature attribution; no faithfulness "
            "analysis has been performed. Do not present as the features "
            "responsible for the prediction."
        )

    return resp


def _highlighted_spans(model, tokenizer, text: str, top_k: int = 8):
    """§9 — visualisation aid only. Kept because it is useful in the UI, named
    so that it cannot be mistaken for attribution."""
    try:
        enc = tokenizer(text, max_length=PRE["max_len"], truncation=True,
                        return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        with torch.no_grad():
            out = model.embedder.bert(
                input_ids=ids, attention_mask=enc["attention_mask"].to(DEVICE),
                output_attentions=True)
        scores = out.attentions[-1][0].mean(dim=0).mean(dim=0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(ids[0])

        spans, cur, acc, n = [], [], 0.0, 0
        for tok_, sc in zip(tokens, scores):
            if tok_ in ("[CLS]", "[SEP]", "[PAD]"):
                if cur:
                    spans.append((" ".join(cur), acc / max(n, 1)))
                    cur, acc, n = [], 0.0, 0
                continue
            if tok_.startswith("##"):
                cur.append(tok_[2:]); acc += float(sc); n += 1
            else:
                if cur:
                    spans.append((" ".join(cur), acc / max(n, 1)))
                cur, acc, n = [tok_], float(sc), 1
        if cur:
            spans.append((" ".join(cur), acc / max(n, 1)))

        spans = [(s.replace(" ##", ""), v) for s, v in spans if len(s) > 3]
        spans.sort(key=lambda x: x[1], reverse=True)
        top = spans[:top_k]
        total = sum(v for _, v in top) or 1.0
        return [{"text": s, "attention_share": round(v / total, 6)} for s, v in top]
    except Exception:
        return []


# =============================================================================
# FASTAPI
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        print(f"[startup] ready: {MODEL_VERSION}")
    except DeploymentIntegrityError as e:
        _startup["errors"].append(str(e))
        print(f"[startup] DEPLOYMENT INTEGRITY FAILURE\n{e}")
        if STRICT_STARTUP:
            # §15.6 / §16 — refuse to serve a model that cannot be shown to be
            # the model that was evaluated.
            raise
    except Exception as e:
        _startup["errors"].append(str(e))
        print(f"[startup] model load failed: {e}")
    yield


app = FastAPI(title="TC-WPN Clinical NLP Service", version=CONTRACT_VERSION,
              docs_url="/docs", redoc_url=None, lifespan=lifespan)

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"],
                   allow_headers=["Authorization", "Content-Type"])
app.include_router(auth_router)   # KEPT: authentication


def auth(authorization: str = Header(default=None)) -> dict:
    return verify_bearer(authorization) if REQUIRE_AUTH else {"sub": "anonymous"}


# ---- §11 request contract ---------------------------------------------------
class SupportNote(BaseModel):
    id: Optional[str] = None
    text: str
    label: str                      # "anxiety" | "control"
    note_date: Optional[str] = None


class PredictRequest(BaseModel):
    patient_id: Optional[str] = None
    note_text: str
    note_type: Optional[str] = Field(default="Discharge summary")
    note_date: Optional[str] = None
    visit_count: Optional[int] = 1
    support_set: Optional[List[SupportNote]] = None
    return_attention: Optional[bool] = False
    return_support_contributions: Optional[bool] = False


def _error(code: str, message: str, status: int = 400, **extra):
    """§6 error envelope, used unchanged everywhere."""
    body = {"status": "error", "error_code": code, "message": message}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


@app.get("/health")
async def health():
    """§15.7 — research metrics live HERE and only here.

    §7 — discrimination and operating point are reported as two separate
    things. AUROC does not depend on the threshold; the threshold is an
    operating point selected on validation.
    """
    body = {
        "status": "ok" if _startup["ready"] else "not_ready",
        "model": "TC-WPN",
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "contract_version": CONTRACT_VERSION,
        "model_loaded": _model is not None,
        "startup_errors": _startup["errors"],
        "provenance": {
            "tcwpn_git_commit": PROV["tcwpn_git_commit"],
            "training_config": PROV["training_config"],
            "run_name": PROV["run_name"],
            "checkpoint_filename": PROV["checkpoint_filename"],
            "checkpoint_sha256_expected": PROV["checkpoint_sha256"],
            "checkpoint_sha256_observed": _startup["checks"].get("checkpoint_sha256"),
            "hf_model_repo": PROV["hf_model_repo"],
            "hf_model_repo_revision": PROV["hf_model_repo_revision"],
            "space_commit": os.environ.get("SPACE_COMMIT", "unknown"),
        },
        "inference_configuration": INF,
        "operating_point": {
            "threshold": OP["threshold"],
            "threshold_source": OP["threshold_source"],
            "note": ("This is an operating point selected on validation. It is "
                     "not derived from AUROC — AUROC is threshold-free."),
        },
        "scope": CFG["scope"],
    }

    if RESEARCH.get("metrics_verified"):
        body["research_metrics"] = {
            "discrimination": RESEARCH["discrimination"],
            "at_locked_threshold": RESEARCH["at_locked_threshold"],
            "calibration": RESEARCH["calibration"],
            "evaluation_context": RESEARCH["evaluation_context"],
            "caveat": RESEARCH["caveat"],
            "note": ("Discrimination (AUROC/PR-AUC) is threshold-free. The "
                     "figures under at_locked_threshold depend on the operating "
                     "point above."),
        }
    else:
        body["research_metrics"] = None
        body["research_metrics_withheld"] = (
            "metrics_verified is false in deployment_config.json. A metric is "
            "published only once it has been confirmed to have been measured on "
            "the checkpoint named in provenance (§17)."
        )
    return body


@app.post("/predict")
async def predict(req: PredictRequest, claims: dict = Depends(auth)):
    if not _startup["ready"]:
        return _error("SERVICE_NOT_READY",
                      "Model is not loaded. See /health startup_errors.", 503)

    if not req.note_text or not req.note_text.strip():
        return _error("MISSING_NOTE_TEXT", "note_text is required and must be non-empty.")

    if not req.support_set:
        # §6 / §15.1 — no silent fallback to demo notes in the API.
        return _error(
            "MISSING_SUPPORT_SET",
            "Both anxiety and control support examples are required for TC-WPN inference.",
            422, required={"anxiety": API["min_anxiety_support"],
                           "control": API["min_control_support"]},
        )

    texts, labels, dates = [], [], []
    for n in req.support_set:
        if not n.text or not n.text.strip():
            continue
        if n.label == "anxiety":
            labels.append(1)
        elif n.label == "control":
            labels.append(0)
        else:
            return _error("INVALID_SUPPORT_LABEL",
                          f"support_set label must be 'anxiety' or 'control', got {n.label!r}.",
                          422)
        texts.append(n.text.strip())
        dates.append(_parse_iso(n.note_date))

    # §15.5 — validate composition.
    n_anx, n_ctrl = labels.count(1), labels.count(0)
    if n_anx < API["min_anxiety_support"] or n_ctrl < API["min_control_support"]:
        return _error(
            "MISSING_SUPPORT_SET",
            "Both anxiety and control support examples are required for TC-WPN inference.",
            422,
            provided={"anxiety": n_anx, "control": n_ctrl},
            required={"anxiety": API["min_anxiety_support"],
                      "control": API["min_control_support"]},
        )

    days, n_undated, reference = compute_days_before_last(dates, _parse_iso(req.note_date))

    try:
        result = run_inference(
            req.note_text, texts, labels, days, n_undated, reference,
            used_default_support_set=False,
            return_attention=bool(req.return_attention),
            return_support_contributions=bool(req.return_support_contributions),
        )
    except Exception as e:
        return _error("INFERENCE_FAILED", str(e), 500)

    result["status"] = "ok"
    result["patient_id"] = req.patient_id
    result["analysed_by"] = claims.get("sub")
    return result


@app.post("/api/predict")
async def api_predict(req: PredictRequest, claims: dict = Depends(auth)):
    return await predict(req, claims)


# =============================================================================
# GRADIO DEMO UI — KEPT (§15). Demo defaults live here and nowhere else (§6).
# =============================================================================
DEMO_ANXIETY_NOTE = (
    "Patient is a 26-year-old female presenting with persistent and excessive worry "
    "about work, health, and finances for the past 8 months. Difficulty controlling "
    "the worry, present most days. Associated fatigue, poor concentration, irritability, "
    "muscle tension, disturbed sleep. PHQ-9 score 16. GAD-7 score 14. "
    "Currently prescribed sertraline 100mg. Referred for CBT. "
    "Diagnosis: Generalized anxiety disorder F41.1."
)
DEMO_CONTROL_NOTE = (
    "Patient admitted for elective laparoscopic cholecystectomy. "
    "Presenting complaint: recurrent right upper quadrant pain after fatty meals. "
    "Ultrasound confirmed cholelithiasis. No psychiatric history. "
    "Procedure completed without complication. Discharged day 1 post-op."
)


def g_predict(note_text, anx_blob, ctrl_blob):
    if not note_text.strip():
        return "Enter a clinical note.", ""
    anx = [t.strip() for t in anx_blob.split("\n---\n") if t.strip()]
    ctrl = [t.strip() for t in ctrl_blob.split("\n---\n") if t.strip()]
    used_default = not (anx and ctrl)
    anx = anx or [DEMO_ANXIETY_NOTE]
    ctrl = ctrl or [DEMO_CONTROL_NOTE]

    texts = anx + ctrl
    labels = [1] * len(anx) + [0] * len(ctrl)
    days = [0.0] * len(texts)

    r = run_inference(note_text, texts, labels, days, len(texts), None,
                      used_default_support_set=used_default)
    warn = ("\n[DEMO SUPPORT SET IN USE — the model is unadapted. The API "
            "refuses this input; only this UI permits it.]\n" if used_default else "\n")
    out = (
        f"{'='*58}{warn}"
        f"PREDICTION            : {r['prediction']}\n"
        f"PROBABILITY           : {r['probability']:.4f}   (uncalibrated)\n"
        f"OPERATING THRESHOLD   : {r['threshold']}   (locked on validation)\n"
        f"CONFIDENCE            : {r['confidence']:.4f}\n"
        f"ENTROPY               : {r['entropy']:.4f}\n"
        f"APPLICATION RISK BAND : {r['application_risk_band']}  "
        f"(UI band, not a validated model output)\n"
        f"{'='*58}\n"
        f"model_version         : {r['model_version']}\n"
        f"preprocessing_version : {r['preprocessing_version']}\n"
        f"{'='*58}\n"
        f"CLINICAL DECISION SUPPORT ONLY — not a diagnostic device.\n"
        f"{'='*58}"
    )
    return out, f"support — anxiety: {len(anx)} | control: {len(ctrl)}"


with gr.Blocks(title="TC-WPN") as gradio_app:
    gr.Markdown(
        "# TC-WPN — clinical anxiety detection\n"
        "Research demo. REST API at `/predict`, docs at `/docs`, provenance at `/health`.\n\n"
        "This is a public Space. Do not paste identifiable patient text."
    )
    with gr.Row():
        with gr.Column():
            ni = gr.Textbox(label="Clinical note (query)", lines=10)
            gr.Dropdown(label="Load example",
                        choices=["GAD — active", "Control — surgical"]
                        ).change(lambda c: DEMO_ANXIETY_NOTE if c == "GAD — active"
                                 else DEMO_CONTROL_NOTE, outputs=ni)
        with gr.Column():
            ab = gr.Textbox(label="ANXIETY support notes (separate with a line: ---)", lines=6)
            cb = gr.Textbox(label="CONTROL support notes (separate with a line: ---)", lines=6)
    ro = gr.Textbox(label="Result", lines=15, interactive=False)
    io_ = gr.Textbox(label="Support set", lines=1, interactive=False)
    gr.Button("Analyse", variant="primary").click(g_predict, [ni, ab, cb], [ro, io_])


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")


app = gr.mount_gradio_app(app, gradio_app, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
