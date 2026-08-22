"""
CARE-AnxRAG client — calls the RAG service as a separate HTTP service, per the
integration contract:

    Flutter / Doctor UI -> central_backend -> RAG HTTP client -> CARE-AnxRAG
    -> retrieval + local Ollama -> structured answer/citations -> central_backend

CARE-AnxRAG is NOT imported into this process. It is a standalone FastAPI
service (~2GB of models/vector store/venv) that the backend calls over HTTP,
exactly like C1/C3/C4. This keeps central_backend small and independently
deployable, and means the RAG service can be redeployed, restarted, or scaled
without touching this code at all.

CONTRACT (as documented by CARE-AnxRAG's own author):
    GET  /health
    POST /v1/ask   {"question": "..."}   ->
        {
          "answer": str, "citations": [...], "confidence": float,
          "conflict_score": float, "abstained": bool,
          "abstention_reason": str | None,
          "safety_level": str, "safety_message": str | None,
          "latest_evidence_at": str | None,
          "knowledge_base_last_sync_at": str | None, "retrieval": ... | None
        }

CURRENT SCOPE (deliberate, per the integration note): the RAG does not take
subject_id, fusion scores, or clinical notes — question + its own evidence
corpus only. central_backend retains subject_id for auth/audit but does not
forward patient data into the RAG call. If patient-specific evidence synthesis
is added later, it should be a separate, explicit, structured field — never by
silently piping raw clinical note text into a question string.

TWO LAYERS OF SAFETY, deliberately not just one:
  1. CARE-AnxRAG has its own `safety_level` / `safety_message` fields — trust
     and surface them, but its exact vocabulary isn't fully documented yet and
     its author has explicitly flagged the current build as pending a full
     validation re-run.
  2. A local, dependency-free crisis pre-screen runs in THIS process before
     the network call is even made. Not a replacement for CARE-AnxRAG's own
     handling — a backstop that doesn't depend on a third-party service being
     reachable, healthy, or correctly configured to catch it.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

RAG_URL = os.getenv("RAG_URL", "http://127.0.0.1:8000").rstrip("/")
RAG_TOKEN = os.getenv("RAG_TOKEN", "")
# Local generation via Ollama can genuinely take tens of seconds, unlike the
# HF Spaces (which are usually fast once awake). Default timeout reflects that.
RAG_TIMEOUT_S = float(os.getenv("RAG_TIMEOUT_S", "60"))

# Minimal, local, dependency-free — deliberately not importing the removed
# rag.py (that module is superseded entirely by CARE-AnxRAG). This is a
# backstop, not the primary safety mechanism; CARE-AnxRAG's own safety_level
# is the primary one.
_CRISIS_PATTERNS = [
    r"suicid", r"self[\s-]?harm", r"kill (him|her|them|my)self",
    r"end (his|her|their|my) life", r"overdose", r"\bod\b",
    r"hurt (him|her|them|my)self",
]


def local_crisis_prescreen(question: Optional[str]) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(re.search(p, q) for p in _CRISIS_PATTERNS)


@dataclass
class Citation:
    citation_id: str
    title: Optional[str] = None
    source_name: Optional[str] = None
    excerpt: Optional[str] = None
    url: Optional[str] = None
    evidence_level: Optional[str] = None


@dataclass
class RagResult:
    available: bool                        # False = the service call itself failed
    answer: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    confidence: Optional[float] = None
    conflict_score: Optional[float] = None
    abstained: bool = False
    abstention_reason: Optional[str] = None
    safety_level: str = "unknown"
    safety_message: Optional[str] = None
    local_crisis_bypass: bool = False      # True = our pre-screen fired, RAG never called
    error: Optional[str] = None
    knowledge_base_last_sync_at: Optional[str] = None
    latency_ms: Optional[int] = None

    def to_wire(self) -> dict:
        return {
            "available": self.available,
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "confidence": self.confidence,
            "conflict_score": self.conflict_score,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "safety_level": self.safety_level,
            "safety_message": self.safety_message,
            "local_crisis_bypass": self.local_crisis_bypass,
            "error": self.error,
            "knowledge_base_last_sync_at": self.knowledge_base_last_sync_at,
            "latency_ms": self.latency_ms,
        }


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if RAG_TOKEN:
        h["Authorization"] = f"Bearer {RAG_TOKEN}"
    return h


def call_rag(question: str, client: Optional[httpx.Client] = None) -> RagResult:
    """POST /v1/ask. A failed call is `available=False`, never a fabricated
    answer — same rule as every other component in this system: a service
    that didn't respond is missing, not a source of invented content."""
    if local_crisis_prescreen(question):
        return RagResult(available=True, answer=None, abstained=True,
                         abstention_reason="local crisis pre-screen matched before the RAG "
                                           "service was called",
                         safety_level="crisis", local_crisis_bypass=True,
                         safety_message=("Possible self-harm or suicide-related content "
                                        "detected. Follow the ward's crisis protocol "
                                        "immediately. This decision-support layer does not "
                                        "respond to crisis situations."))

    if not RAG_URL:
        return RagResult(available=False, error="RAG_URL not configured")

    own = client is None
    client = client or httpx.Client()
    t0 = dt.datetime.now(dt.timezone.utc)
    try:
        r = client.post(f"{RAG_URL}/v1/ask", headers=_headers(),
                        json={"question": question}, timeout=RAG_TIMEOUT_S)
        latency = int((dt.datetime.now(dt.timezone.utc) - t0).total_seconds() * 1000)
        if r.status_code != 200:
            return RagResult(available=False, error=f"HTTP {r.status_code}", latency_ms=latency)
        body = r.json()
        citations = [Citation(citation_id=c.get("citation_id", "?"), title=c.get("title"),
                              source_name=c.get("source_name"), excerpt=c.get("excerpt"),
                              url=c.get("url"), evidence_level=c.get("evidence_level"))
                    for c in body.get("citations") or []]
        return RagResult(
            available=True, answer=body.get("answer"), citations=citations,
            confidence=body.get("confidence"), conflict_score=body.get("conflict_score"),
            abstained=bool(body.get("abstained", False)),
            abstention_reason=body.get("abstention_reason"),
            safety_level=body.get("safety_level", "unknown"),
            safety_message=body.get("safety_message"),
            knowledge_base_last_sync_at=body.get("knowledge_base_last_sync_at"),
            latency_ms=latency)
    except httpx.TimeoutException:
        return RagResult(available=False, error=f"timeout after {RAG_TIMEOUT_S}s "
                                                 "(local generation can be slow — "
                                                 "check the RAG service is warm)")
    except Exception as exc:                                # noqa: BLE001
        return RagResult(available=False, error=f"{type(exc).__name__}: {exc}"[:160])
    finally:
        if own:
            client.close()


def check_rag_health(client: Optional[httpx.Client] = None) -> dict:
    if not RAG_URL:
        return {"configured": False}
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.get(f"{RAG_URL}/health", headers=_headers(), timeout=10)
        return {"configured": True, "reachable": r.status_code == 200,
               "detail": r.json() if r.status_code == 200 else f"HTTP {r.status_code}"}
    except Exception as exc:                                # noqa: BLE001
        return {"configured": True, "reachable": False, "detail": str(exc)[:160]}
    finally:
        if own:
            client.close()
