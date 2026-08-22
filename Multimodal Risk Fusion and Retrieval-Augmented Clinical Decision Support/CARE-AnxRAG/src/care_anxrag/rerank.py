from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .models import SearchHit
from .util import clamp, content_tokens, sigmoid


class Reranker(Protocol):
    def score(self, query: str, hits: Sequence[SearchHit]) -> list[float]: ...


class HeuristicReranker:
    def score(self, query: str, hits: Sequence[SearchHit]) -> list[float]:
        query_tokens = set(content_tokens(query))
        scores: list[float] = []
        for hit in hits:
            document_tokens = set(content_tokens(f"{hit.chunk.title} {hit.chunk.section_heading} {hit.chunk.text}"))
            if not query_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & document_tokens) / len(query_tokens)
            title_tokens = set(content_tokens(hit.chunk.title))
            title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
            phrase_bonus = 0.15 if query.lower() in hit.chunk.text.lower() else 0.0
            scores.append(clamp((0.68 * overlap) + (0.17 * title_overlap) + phrase_bonus))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the 'production' extra or "
                "set CARE_RERANKER_PROVIDER=heuristic."
            ) from exc
        self.model_name = model_name
        # Force raw logits so score calibration is deterministic across
        # sentence-transformers releases; sigmoid is applied exactly once below.
        self.model = CrossEncoder(model_name, activation_fn=nn.Identity())

    def score(self, query: str, hits: Sequence[SearchHit]) -> list[float]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        raw = np.asarray(
            self.model.predict(pairs, show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)
        if raw.size != len(hits):
            raise RuntimeError(
                "CrossEncoder returned an unexpected number of scores: "
                f"expected {len(hits)}, received {raw.size}"
            )
        return [clamp(sigmoid(float(value))) for value in raw]
