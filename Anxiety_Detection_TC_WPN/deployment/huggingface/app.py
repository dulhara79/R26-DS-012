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

CONTRACT v1.1.0 — ALIGNMENT WITH R26-DS-012_service_contracts.md
================================================================
v1.0.0 answered the model question correctly but did not speak the central
backend's envelope. v1.1.0 adds it, and corrects one genuine inference bug.

ADDED (contract §1 common envelope — every model service returns these):
  subject_id, modality, score, status, captured_at, computed_at, latency_ms.
  `score` and `probability` are deliberately the same float. `score` is the
  envelope field fusion reads; `probability` is C4's native name. Duplicating
  one float is cheaper than an argument about which field was meant.

ADDED (auditability):
  prototype_distance_anxiety / prototype_distance_control — restored. The
  squashed [0,1] number alone makes a stored reading unauditable later. These
  are REPORT-ONLY: they are recomputed from the same normalised embeddings for
  display and take no part in producing `score`, which still comes from
  model.classify(). Phase 7 parity is therefore unaffected.
  support_set_version — echoed. Change the site's support bank and yesterday's
  scores are no longer reproducible, so the version must reach the audit log.
  Support note `id` is echoed in support_contributions.

FIXED — the temporal axis (this was a real bug, not a naming issue):
  v1.0.0 computed one shared reference point, max(support dates + query date),
  and measured every support note against it. That is not the training
  quantity. scripts/apply_index_time.py sets
      days_before_patient_last_note := days_before_index = (t_index − charttime)
  where t_index is the index time of THAT NOTE'S OWN PATIENT. Support notes
  come from K different patients (sampler.py guarantees it), so there are K
  reference points, not one — and under the old code a support note more
  recent than the query silently inflated every other note's delta.
  Only the backend knows each support note's patient anchor, so the backend
  now supplies `days_before_index` per note. `temporal_axis` on the response
  states which path was taken; nothing is silently approximated.

REMOVED FROM THE CONTRACT DOC (not implementable / not true):
  status "no_support_set" — a prototypical network has no stored prototypes.
    build_prototype() runs per request; the checkpoint holds no centroid. With
    K=0 there is nothing to classify against. 422 MISSING_SUPPORT_SET stands.
    (tcwpn_full does carry an aux_head over the query embedding alone, so a
    support-free score is technically computable — but evaluation.py scores
    out["p_anxiety"], the PROTOTYPE path. Every metric you have describes that
    path. The aux head is an unmeasured code path and is not served.)
  calibrated_probability — no calibrator is fitted anywhere in this pipeline
    (deployment_config: calibrator_fitted false; seed-42 ECE 0.0849, Brier
    0.209). The field is replaced by `probability` + calibration_status.
  ece in a /predict response — a research metric measured over 813 episodes.
    It does not describe one note. /health only (§15.7).
  temporal_weight / confidence_weight split in support_contributions —
    build_prototype returns ONE normalised weight vector; w^T and w^C are
    multiplied inside the loop and never surfaced separately. Splitting them
    would require changing the model's public API and would break Phase 7.

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
import time
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


def resolve_temporal_deltas(support_notes: List["SupportNote"],
                            query_date: Optional[datetime]):
    """
    Produce the [K] vector that TemporalRecencyWeight consumes, and say
    honestly which of three paths produced it.

    WHAT THE MODEL WAS TRAINED ON
    -----------------------------
    src/tcwpn/collate.py reads `days_before_patient_last_note` off each record.
    scripts/apply_index_time.py overwrites that column:

        days_before_patient_last_note := days_before_index
        days_before_index              = (t_index − charttime), clipped at 0

    t_index is the index time of the patient THAT NOTE belongs to. Because
    sampler.py guarantees one support slot = one distinct patient, a K-shot
    support set carries K independent reference points. There is no single
    shared "most recent note" axis, and constructing one — as contract v1.0.0
    did with max(support ∪ query) — is not the training quantity. It also
    misbehaves: a support note more recent than the query moved the reference
    forward and inflated every other note's delta.

    (The docstring on TemporalRecencyWeight in the vendored model.py still
    describes the pre-index-time semantics. It is stale; indexing.py §TEMPORAL
    FEATURE is authoritative. See NOTES_FOR_SRC_TCWPN_MODEL.md.)

    THE THREE PATHS
    ---------------
    "backend_supplied"  — every note carried days_before_index. Correct, and
                          the only path that reproduces training semantics.
                          Only the backend can compute it: it is the one party
                          that knows each support note's patient anchor.
    "approximated"      — at least one note fell back to (query_date − note_date).
                          This treats the QUERY note as the index time, which is
                          right for the prediction point but wrong for the
                          support note's own patient. Reported so a caller can
                          see the reading is approximate rather than assume it.
    "unavailable"       — no usable dates at all. All deltas are 0.0, which
                          makes w^T = exp(0) = 1 for every note, i.e. temporal
                          weighting is inert. Reported rather than hidden,
                          because "all weights equal" and "recency applied" are
                          very different claims to put in a clinician's record.

    Returns (days [K] float, n_supplied int, n_approximated int,
             n_undated int, axis str, reference ISO str or None).
    """
    days: List[float] = []
    n_supplied = n_approximated = n_undated = 0

    for n in support_notes:
        if n.days_before_index is not None:
            days.append(max(0.0, float(n.days_before_index)))
            n_supplied += 1
            continue

        d = _parse_iso(n.note_date)
        if d is not None and query_date is not None:
            days.append(max(0.0, (query_date - d).total_seconds() / 86400.0))
            n_approximated += 1
        else:
            # No inventing a position on an axis we cannot place the note on.
            days.append(0.0)
            n_undated += 1

    if n_approximated == 0 and n_undated == 0 and days:
        axis = "backend_supplied"
    elif n_supplied == 0 and n_approximated == 0:
        axis = "unavailable"
    else:
        axis = "approximated"

    reference = query_date.isoformat() if query_date is not None else None
    return days, n_supplied, n_approximated, n_undated, axis, reference


# =============================================================================
# INFERENCE — calls the vendored model's own methods (Phase 7 parity)
# =============================================================================
@torch.no_grad()
def run_inference(note_text: str,
                  support_texts: List[str],
                  support_labels: List[int],
                  support_days: List[float],
                  used_default_support_set: bool,
                  support_ids: Optional[List[Optional[str]]] = None,
                  temporal: Optional[dict] = None,
                  support_set_version: Optional[str] = None,
                  return_attention: bool = False,
                  return_support_contributions: bool = False) -> dict:
    t0 = time.perf_counter()
    model, tok = load_model()

    temporal = temporal or {
        "axis": "unavailable", "reference": None,
        "n_supplied": 0, "n_approximated": 0, "n_undated": len(support_texts),
    }
    support_ids = support_ids or [None] * len(support_texts)

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

    # REPORT-ONLY. Recomputed here purely so a stored reading stays auditable —
    # the squashed [0,1] number alone cannot be checked later. `probability`
    # above already came from model.classify(); nothing below feeds back into
    # it, so Phase 7 parity is untouched. Squared euclidean on the unit sphere
    # is 2 − 2·cos, which is exactly what classify() takes cdist of.
    _q = F.normalize(qry_emb, dim=-1)[0]
    _dist = {c: float(((_q - prototypes[i]) ** 2).sum().item())
             for i, c in enumerate(classes)}

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
        # ---- contract §1 common envelope -------------------------------------
        # subject_id, modality, score, status, captured_at and computed_at are
        # filled by the /predict handler, which is the layer that has the
        # request. `score` is set there to exactly `probability`.
        "model": "TC-WPN",
        "model_version": MODEL_VERSION,                       # §15.2

        # ---- the prediction --------------------------------------------------
        "prediction": prediction,
        "probability": round(probability, 6),
        "threshold": threshold,
        "confidence": round(float(confidence), 6),
        "entropy": round(entropy, 6),

        # Uncalibrated, and named so. deployment_config: calibrator_fitted
        # false; the seed-42 clean benchmark measures ECE 0.0849, Brier 0.209.
        # Fusion averages this with C1 and C2 as if the three were comparable
        # probabilities — that is a modelling assumption, and this field is
        # where C4 declares its side of it.
        "calibration_status": "uncalibrated",
        "probability_semantics": (
            "softmax over cosine-distance prototype logits. Uncalibrated: no "
            "calibrator is fitted in this deployment."
        ),

        # ---- auditability ----------------------------------------------------
        "prototype_distance_anxiety": round(_dist.get(1, float("nan")), 6),
        "prototype_distance_control": round(_dist.get(0, float("nan")), 6),
        "temperature": round(float(torch.exp(model.log_temperature).item()), 4),

        # ---- the support set that built the prototypes ------------------------
        "support_count": {"anxiety": n_anx, "control": n_ctrl,
                          "k": n_anx + n_ctrl},
        "support_set_version": support_set_version,
        "evaluated_k": API.get("evaluated_k"),
        "temporal_axis": temporal["axis"],
        "temporal_reference": temporal["reference"],
        "support_notes_with_supplied_delta": temporal["n_supplied"],
        "support_notes_with_approximated_delta": temporal["n_approximated"],
        "undated_support_notes": temporal["n_undated"],
        "used_default_support_set": bool(used_default_support_set),

        # ---- mechanisms — switched on, not shown to work ----------------------
        "temporal_weighting_used": bool(model.use_temporal_weight),
        "prototype_consistency_weighting_used": bool(model.use_pcw),
        "mechanism_note": (
            "temporal_weighting_used and prototype_consistency_weighting_used "
            "report which mechanisms are ENABLED in this build. Neither shows a "
            "statistically detectable effect on AUROC once the auxiliary CE "
            "head is held constant (paired over 5 seeds against aux_only: "
            "tcwpn_full +0.0006, p=0.886; temporal_aux +0.0006, p=0.906; "
            "pcw_aux -0.0080, p=0.150). Do not attribute this score to either "
            "mechanism."
        ),

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
        # §8 — named as application layer, not a validated model output.
        "application_risk_band": band,
        "application_risk_band_note": (
            "Application-layer UI band, not a validated model output. TC-WPN "
            "was evaluated as a binary classifier; LOW/MODERATE/HIGH/VERY HIGH "
            "were not validated as clinical classes."
        ),

        "latency_ms": None,   # set at the end of this function
    }

    if return_support_contributions:
        # ONE weight per note, not a w^T / w^C split. build_prototype()
        # multiplies the two inside its loop and returns only the final
        # normalised vector; separating them would mean changing the model's
        # public API, which is exactly what Phase 7 parity forbids.
        contribs = []
        for c in classes:
            idx = [i for i, l in enumerate(support_labels) if l == c]
            for i, w in zip(idx, weights[c].tolist()):
                contribs.append({
                    "id": support_ids[i],
                    "support_index": i,
                    "label": "anxiety" if c == 1 else "control",
                    "excerpt": support_texts[i][:120],
                    "weight": round(float(w), 6),
                    "days_before_index": round(support_days[i], 3),
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

    resp["latency_ms"] = int((time.perf_counter() - t0) * 1000)
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
    """One entry in the site's labelled reference bank.

    NOT a previous note of the patient being scored. sampler.py guarantees
    support ∩ query patients = ∅ in every training episode, and collate.py
    raises on overlap. Nothing raises at serving time, so the BACKEND must
    enforce the same exclusion when it selects these notes — otherwise it
    reproduces the exact leakage the clean pipeline was rebuilt to remove.
    """
    id: Optional[str] = None
    text: str
    label: str                              # "anxiety" | "control"
    note_date: Optional[str] = None         # ISO-8601, fallback only
    days_before_index: Optional[float] = None
    """(t_index − charttime) in days for THIS note's own patient, >= 0.

    Preferred. This is the exact quantity the model was trained on
    (scripts/apply_index_time.py). Only the backend can compute it, because
    only the backend knows each support note's patient anchor. Omit it and the
    service falls back to note_date and reports temporal_axis "approximated".
    """


class PredictRequest(BaseModel):
    # subject_id is the canonical backend ID (contract §1). patient_id is
    # accepted as a deprecated alias so the existing ClinAnx Flutter build
    # keeps working through the migration; the response always returns
    # subject_id.
    subject_id: Optional[str] = None
    patient_id: Optional[str] = None

    note_text: str
    note_type: Optional[str] = Field(default="Discharge summary")
    note_date: Optional[str] = None
    visit_count: Optional[int] = 1

    support_set: Optional[List[SupportNote]] = None
    support_set_version: Optional[str] = None
    """Identifier of the site reference bank these notes came from. Stamped on
    the response and required in the audit log: change the bank and yesterday's
    scores are no longer reproducible."""

    return_attention: Optional[bool] = False
    return_support_contributions: Optional[bool] = False

    def resolved_subject_id(self) -> Optional[str]:
        return self.subject_id or self.patient_id


def _error(code: str, message: str, status: int = 400, **extra):
    """§6 error envelope.

    `modality` and `score: null` are carried so the backend can store the
    failure as a typed reading and show the clinician a gap, rather than
    silently dropping a modality that was due (contract §8).
    """
    body = {"status": "error", "error_code": code, "message": message,
            "modality": "c4_clinical_nlp", "score": None,
            "model_version": MODEL_VERSION,
            "computed_at": datetime.now(timezone.utc)
                                   .isoformat().replace("+00:00", "Z")}
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
        "support_set_contract": {
            "required": API["requires_support_set"],
            "min_anxiety": API["min_anxiety_support"],
            "min_control": API["min_control_support"],
            "evaluated_k": API.get("evaluated_k"),
            "note": ("The support set is a bank of labelled notes from OTHER "
                     "patients — it is the classifier, not the subject's own "
                     "history. Prototypes are built per request and discarded; "
                     "no prototype is stored in the checkpoint. The backend "
                     "must exclude the queried subject_id from the bank."),
            "temporal_field": ("days_before_index per note, = (t_index - "
                               "charttime) for that note's own patient. Supply "
                               "it; note_date is an approximation."),
        },
        "calibration_status": "uncalibrated",
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
            # Published beside the headline number, not behind it. The blinded
            # result is the honest one and the mechanism finding is what the
            # service must never claim credit for.
            "blinded": RESEARCH.get("blinded"),
            "mechanism_finding": RESEARCH.get("mechanism_finding"),
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
    subject_id = req.resolved_subject_id()

    if not _startup["ready"]:
        return _error("SERVICE_NOT_READY",
                      "Model is not loaded. See /health startup_errors.", 503,
                      subject_id=subject_id)

    if not req.note_text or not req.note_text.strip():
        return _error("MISSING_NOTE_TEXT",
                      "note_text is required and must be non-empty.",
                      subject_id=subject_id)

    if not req.support_set:
        # §6 / §15.1 — no silent fallback to demo notes in the API.
        # And no "no_support_set" status: there is nothing to run. Prototypes
        # are built per request from these notes; the checkpoint holds none.
        return _error(
            "MISSING_SUPPORT_SET",
            "Both anxiety and control support examples are required for TC-WPN inference.",
            422, subject_id=subject_id,
            required={"anxiety": API["min_anxiety_support"],
                      "control": API["min_control_support"]},
        )

    notes, texts, labels, ids = [], [], [], []
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
                          422, subject_id=subject_id)
        notes.append(n)
        texts.append(n.text.strip())
        ids.append(n.id)

    # §15.5 — validate composition.
    n_anx, n_ctrl = labels.count(1), labels.count(0)
    if n_anx < API["min_anxiety_support"] or n_ctrl < API["min_control_support"]:
        return _error(
            "MISSING_SUPPORT_SET",
            "Both anxiety and control support examples are required for TC-WPN inference.",
            422, subject_id=subject_id,
            provided={"anxiety": n_anx, "control": n_ctrl},
            required={"anxiety": API["min_anxiety_support"],
                      "control": API["min_control_support"]},
        )

    days, n_sup, n_approx, n_undated, axis, reference = resolve_temporal_deltas(
        notes, _parse_iso(req.note_date))

    try:
        result = run_inference(
            req.note_text, texts, labels, days,
            used_default_support_set=False,
            support_ids=ids,
            temporal={"axis": axis, "reference": reference,
                      "n_supplied": n_sup, "n_approximated": n_approx,
                      "n_undated": n_undated},
            support_set_version=req.support_set_version,
            return_attention=bool(req.return_attention),
            return_support_contributions=bool(req.return_support_contributions),
        )
    except Exception as e:
        # Stored as a gap in the clinician timeline rather than dropped —
        # contract §8, "the modality was due and did not arrive".
        return _error("INFERENCE_FAILED", str(e), 500, subject_id=subject_id)

    # ---- contract §1 common envelope, applied last so it cannot be shadowed --
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result["subject_id"] = subject_id
    result["modality"] = "c4_clinical_nlp"
    # Same float as `probability`, under the name fusion reads. Higher always
    # means more risk, for every modality (contract §1).
    result["score"] = result["probability"]
    result["status"] = "ok"
    result["captured_at"] = req.note_date or now   # when the note was written
    result["computed_at"] = now                    # when inference ran
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

    # The UI has no index times, so every delta is 0 and w^T = exp(0) = 1 for
    # every note. temporal_axis says "unavailable" rather than letting the
    # screen imply recency weighting ran.
    r = run_inference(
        note_text, texts, labels, days,
        used_default_support_set=used_default,
        temporal={"axis": "unavailable", "reference": None,
                  "n_supplied": 0, "n_approximated": 0, "n_undated": len(texts)},
    )
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
        f"TEMPORAL AXIS         : {r['temporal_axis']}\n"
        f"CALIBRATION           : {r['calibration_status']}\n"
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