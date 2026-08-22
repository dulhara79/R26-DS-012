"""
Stage 1 of fusion: SCORE HARMONISATION.

The four components emit numbers on four incompatible scales:

    C1 physiological   reconstruction-error derived anomaly value
    C2 behavioural     GATv2 logit
    C3 clinical notes  prototype-distance probability, threshold locked at 0.4036
    C4 demographic     calibrated P(GAD-7 >= 10), base rate ~0.02

Averaging those directly is meaningless — a 0.6 from C3 and a 0.6 from C4 do not
describe the same amount of risk. Every score is therefore mapped to its
PERCENTILE RANK within that modality's own frozen reference distribution, which
puts all four on one interpretable axis: "how high is this patient compared with
everyone else this component has scored".

The reference distributions are FROZEN artefacts. They are computed once, saved,
and never recomputed at inference. If a teammate redeploys their Space with a
retrained model, their score distribution shifts and the drift monitor below
flags it — rather than silently corrupting every composite from then on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REFERENCE_DIR = Path(__file__).parent / "reference"


@dataclass
class Harmonised:
    value: float            # percentile in [0, 1] — the fusion input
    raw: float              # what the component actually sent
    drift: bool             # reference distribution no longer looks right
    note: Optional[str] = None


class Harmoniser:
    """Percentile mapping against a frozen per-modality reference distribution."""

    # If a modality's recent scores drift this far (in mean percentile) from the
    # expected 0.5, the reference is probably stale. 0.25 is deliberately loose —
    # a real ward population WILL sit higher than a reference cohort, so this is
    # a "look at this" signal, not an automatic exclusion.
    DRIFT_TOLERANCE = 0.25

    def __init__(self, reference_dir: Path = REFERENCE_DIR):
        self.reference_dir = reference_dir
        self.refs: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, dict] = {}
        self._recent: Dict[str, list] = {}
        self._load()

    def _load(self):
        if not self.reference_dir.exists():
            return
        for f in sorted(self.reference_dir.glob("*.json")):
            blob = json.loads(f.read_text())
            m = blob["modality"]
            self.refs[m] = np.sort(np.asarray(blob["scores"], dtype=float))
            self.meta[m] = {k: v for k, v in blob.items() if k != "scores"}

    def available(self) -> Dict[str, int]:
        return {m: int(v.size) for m, v in self.refs.items()}

    def harmonise(self, modality: str, raw: Optional[float]) -> Optional[Harmonised]:
        if raw is None:
            return None
        raw = float(raw)

        ref = self.refs.get(modality)
        if ref is None or ref.size < 30:
            # No usable reference yet. Pass the raw value through, clipped, and say so.
            # This is honest degradation: the composite is still computed, but the
            # response tells the clinician the scales were not harmonised.
            return Harmonised(
                value=float(np.clip(raw, 0.0, 1.0)), raw=raw, drift=False,
                note=f"no reference distribution for {modality} — raw score passed through, "
                     f"cross-modality comparison is NOT valid until one is built",
            )

        pct = float(np.searchsorted(ref, raw, side="right") / ref.size)
        pct = float(np.clip(pct, 0.0, 1.0))

        # rolling drift check over the last 200 observations
        buf = self._recent.setdefault(modality, [])
        buf.append(pct)
        if len(buf) > 200:
            buf.pop(0)
        drift = len(buf) >= 50 and abs(float(np.mean(buf)) - 0.5) > self.DRIFT_TOLERANCE

        return Harmonised(
            value=pct, raw=raw, drift=drift,
            note=(f"score distribution has drifted from the frozen reference "
                  f"(mean percentile {np.mean(buf):.2f}) — check whether "
                  f"{modality} was redeployed") if drift else None,
        )


def save_reference(modality: str, scores, source: str, model_version: str,
                   reference_dir: Path = REFERENCE_DIR) -> Path:
    """Freeze a reference distribution. Run once per modality, then leave it alone.

    For C4 use the array your notebook saved as dcar_reference_scores.npy.
    For C1/C2/C3 ask each teammate for the score vector from THEIR held-out
    evaluation set — not from training data, or the reference inherits their
    overfitting.
    """
    reference_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size < 30:
        raise ValueError(f"{modality}: need at least 30 scores, got {scores.size}")

    path = reference_dir / f"{modality}.json"
    path.write_text(json.dumps({
        "modality": modality,
        "source": source,
        "model_version": model_version,
        "n": int(scores.size),
        "min": float(scores.min()), "max": float(scores.max()),
        "mean": float(scores.mean()),
        "quartiles": [float(q) for q in np.percentile(scores, [25, 50, 75])],
        "scores": [round(float(s), 6) for s in np.sort(scores)],
    }, indent=2))
    return path
