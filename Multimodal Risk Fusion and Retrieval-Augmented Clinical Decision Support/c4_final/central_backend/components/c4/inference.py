"""Frozen, dependency-free DCAR calibrated Elastic-Net inference.

The five-fold coefficients are exported from the original sklearn joblib model;
no training data or unsafe deserialization is needed inside the backend.
"""
from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("model.json")

class UnsupportedProfile(ValueError):
    """Incomplete or unsupported demographic profile."""

@lru_cache(maxsize=1)
def load_model():
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if m.get("component_id") != "c4_demographic" or m.get("target") != "GAD-7 >= 10" or len(m.get("folds", [])) != 5:
        raise ValueError("invalid C4 model metadata")
    if not 0 < m["reference_prevalence"] < 1 or set(m["categories"]) != {"gender", "edu", "smoke", "drink"}:
        raise ValueError("invalid C4 model metadata")
    n = 1 + sum(len(v)-1 for v in m["categories"].values())
    for f in m["folds"]:
        if len(f["coef"]) != n or f["age_scale"] <= 0:
            raise ValueError("invalid C4 model dimensions")
        vals = f["coef"] + [f[k] for k in ("age_mean","age_scale","intercept","sigmoid_a","sigmoid_b")]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("invalid C4 numeric coefficients")
    return m

def model_available():
    try:
        load_model()
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False

def _sigmoid(x):
    if x >= 0:
        z=math.exp(-x)
        return 1/(1+z)
    z=math.exp(x)
    return z/(1+z)

def _logit(p):
    p=min(max(p,1e-12),1-1e-12)
    return math.log(p/(1-p))

def predict_demographics(raw):
    m=load_model()
    if not isinstance(raw,dict):
        raise UnsupportedProfile("demographics must be an object")
    try:
        if isinstance(raw.get("age"),bool) or raw.get("age") is None:
            raise ValueError()
        age=float(raw["age"])
    except (ValueError, TypeError, OverflowError):
        raise UnsupportedProfile("age is missing or invalid") from None
    if not math.isfinite(age):
        raise UnsupportedProfile("age must be finite")
    if not m["age_training_range"][0] <= age <= m["age_training_range"][1]:
        raise UnsupportedProfile("age outside training range")
    if not m["age_supported_range"][0] <= age <= m["age_supported_range"][1]:
        raise UnsupportedProfile("age outside well-represented range 18-27")
    encoded=[]
    for key,levels in m["categories"].items():
        value=raw.get(key)
        if not isinstance(value,str):
            raise UnsupportedProfile(key+" is required")
        value=value.strip().casefold()
        if value not in levels:
            raise UnsupportedProfile("unsupported "+key+" category")
        encoded.extend(1.0 if value==category else 0.0 for category in levels[1:])
    fold_p=[]
    for f in m["folds"]:
        x=[(age-f["age_mean"])/f["age_scale"],*encoded]
        decision=f["intercept"]+sum(w*v for w,v in zip(f["coef"],x))
        fold_p.append(_sigmoid(-(f["sigmoid_a"]*decision+f["sigmoid_b"])))
    p=sum(fold_p)/len(fold_p)
    baseline=m["reference_prevalence"]
    evidence=_sigmoid(_logit(p)-_logit(baseline))
    return {
        "demographic_probability":p,
        "demographic_probability_percent":p*100,
        "demographic_fusion_score":evidence,
        "reference_prevalence":baseline,
        "target":m["target"],
        "model":m["model_type"],
        "model_version":m["model_version"],
        "score_semantics":"calibrated_probability_gad7_ge10",
        "fusion_score_semantics":"relative_evidence_not_probability"
    }
