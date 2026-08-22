from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

from .models import EvidenceRelation, RelationLabel, SearchHit
from .util import clamp, content_tokens


class NliClassifier(Protocol):
    def classify(self, pairs: Sequence[tuple[SearchHit, SearchHit]]) -> list[EvidenceRelation]: ...


class HeuristicNliClassifier:
    """Deterministic offline NLI used only for tests and smoke checks.

    It compares sentence-level propositions rather than whole chunks. This prevents an
    unrelated safety sentence such as "must not diagnose" from masking a contradiction
    between "CBT is recommended" and "CBT is not recommended".
    """

    _sentence_splitter = re.compile(r"(?<=[.!?])\s+|\n+")
    _positive_patterns = (
        re.compile(r"\brecommended\b", re.I),
        re.compile(r"\beffective\b", re.I),
        re.compile(r"\bbeneficial\b", re.I),
        re.compile(r"\bsupports?\b", re.I),
        re.compile(r"\bimproves?\b", re.I),
        re.compile(r"\bshould\s+(?:be\s+)?(?:offered|used|considered)\b", re.I),
    )
    _negative_patterns = (
        re.compile(r"\bnot\s+recommended\b", re.I),
        re.compile(r"\bnot\s+effective\b", re.I),
        re.compile(r"\bineffective\b", re.I),
        re.compile(r"\bcontraindicated\b", re.I),
        re.compile(r"\bshould\s+not\s+(?:be\s+)?(?:offered|used|considered)\b", re.I),
        re.compile(r"\bdoes\s+not\s+(?:work|help|improve|reduce)\b", re.I),
        re.compile(r"\bno\s+(?:clear\s+|sufficient\s+)?evidence\b", re.I),
    )

    @classmethod
    def _sentences(cls, text: str) -> list[str]:
        return [part.strip() for part in cls._sentence_splitter.split(text) if part.strip()]

    @classmethod
    def _polarity(cls, sentence: str) -> int:
        # Check explicit negative constructions first because they often contain a
        # positive anchor token ("not recommended", "not effective").
        if any(pattern.search(sentence) for pattern in cls._negative_patterns):
            return -1
        if any(pattern.search(sentence) for pattern in cls._positive_patterns):
            return 1
        return 0

    @staticmethod
    def _overlap(left: str, right: str) -> float:
        left_tokens = set(content_tokens(left))
        right_tokens = set(content_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))

    def classify(self, pairs: Sequence[tuple[SearchHit, SearchHit]]) -> list[EvidenceRelation]:
        relations: list[EvidenceRelation] = []
        for left, right in pairs:
            best_contradiction = 0.0
            best_entailment = 0.0
            for left_sentence in self._sentences(left.chunk.text):
                left_polarity = self._polarity(left_sentence)
                for right_sentence in self._sentences(right.chunk.text):
                    overlap = self._overlap(left_sentence, right_sentence)
                    if overlap < 0.20:
                        continue
                    right_polarity = self._polarity(right_sentence)
                    if left_polarity and right_polarity and left_polarity != right_polarity:
                        best_contradiction = max(
                            best_contradiction,
                            clamp(0.58 + 0.50 * overlap),
                        )
                    elif overlap >= 0.42 and (
                        left_polarity == right_polarity or left_polarity == 0 or right_polarity == 0
                    ):
                        best_entailment = max(best_entailment, clamp(0.45 + 0.50 * overlap))

            if best_contradiction >= best_entailment and best_contradiction > 0.0:
                label = RelationLabel.CONTRADICTION
                confidence = best_contradiction
            elif best_entailment > 0.0:
                label = RelationLabel.ENTAILMENT
                confidence = best_entailment
            else:
                chunk_overlap = self._overlap(left.chunk.text, right.chunk.text)
                label = RelationLabel.NEUTRAL
                confidence = clamp(0.50 + (1.0 - chunk_overlap) * 0.15)

            relations.append(
                EvidenceRelation(
                    left_chunk_id=left.chunk.chunk_id,
                    right_chunk_id=right.chunk.chunk_id,
                    label=label,
                    confidence=confidence,
                )
            )
        return relations


class CrossEncoderNliClassifier:
    KNOWN_LABEL_ORDERS = {
        "cross-encoder/nli-deberta-v3-base": [
            RelationLabel.CONTRADICTION,
            RelationLabel.ENTAILMENT,
            RelationLabel.NEUTRAL,
        ]
    }

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the 'production' extra or "
                "set CARE_NLI_PROVIDER=heuristic."
            ) from exc
        self.model_name = model_name
        # NLI requires the three unnormalized class logits. Applying softmax is
        # intentionally handled in classify() so label probabilities are explicit.
        self.model = CrossEncoder(model_name, activation_fn=nn.Identity())
        self.labels = self._resolve_label_order()

    def _resolve_label_order(self) -> list[RelationLabel]:
        config = getattr(getattr(self.model, "model", None), "config", None)
        id2label = getattr(config, "id2label", None)
        if isinstance(id2label, dict):
            resolved: list[RelationLabel] = []
            for index in range(3):
                raw = id2label.get(index, id2label.get(str(index)))
                normalized = str(raw or "").strip().lower()
                try:
                    resolved.append(RelationLabel(normalized))
                except ValueError:
                    resolved = []
                    break
            if len(resolved) == 3 and len(set(resolved)) == 3:
                return resolved
        known = self.KNOWN_LABEL_ORDERS.get(self.model_name.lower())
        if known is not None:
            return list(known)
        raise RuntimeError(
            f"Cannot determine NLI label order for model {self.model_name!r}. "
            "Use a model whose id2label maps to contradiction, entailment, and neutral, "
            "or use the validated default model."
        )

    def classify(self, pairs: Sequence[tuple[SearchHit, SearchHit]]) -> list[EvidenceRelation]:
        if not pairs:
            return []
        text_pairs = [(left.chunk.text, right.chunk.text) for left, right in pairs]
        logits = np.asarray(
            self.model.predict(text_pairs, show_progress_bar=False),
            dtype=np.float64,
        )
        if logits.ndim == 1 and len(pairs) == 1 and logits.size == 3:
            logits = logits.reshape(1, 3)
        if logits.ndim != 2 or logits.shape != (len(pairs), 3):
            raise RuntimeError(
                "NLI model must return one three-logit row per evidence pair; "
                f"expected {(len(pairs), 3)}, received {logits.shape}"
            )
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        relations: list[EvidenceRelation] = []
        for (left, right), probability in zip(pairs, probabilities, strict=True):
            index = int(probability.argmax())
            relations.append(
                EvidenceRelation(
                    left_chunk_id=left.chunk.chunk_id,
                    right_chunk_id=right.chunk.chunk_id,
                    label=self.labels[index],
                    confidence=clamp(float(probability[index])),
                )
            )
        return relations
