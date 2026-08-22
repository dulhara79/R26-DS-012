from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import httpx
import numpy as np

from .db import Database
from .util import content_tokens, normalize_vector


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def ping(self) -> bool: ...


class HashEmbedder:
    """Deterministic feature-hashing embedder for tests and offline smoke checks."""

    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"hash:{self.dimensions}:v2"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            text_tokens = content_tokens(text)
            for index, token in enumerate(text_tokens):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                position = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vector[position] += sign
                if index + 1 < len(text_tokens):
                    bigram = f"{token}_{text_tokens[index + 1]}"
                    digest2 = hashlib.blake2b(bigram.encode("utf-8"), digest_size=16).digest()
                    position2 = int.from_bytes(digest2[:8], "big") % self.dimensions
                    sign2 = 1.0 if digest2[8] % 2 == 0 else -1.0
                    vector[position2] += 0.5 * sign2
            vectors.append(normalize_vector(vector.tolist()))
        return vectors

    def ping(self) -> bool:
        return True


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        dimensions: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        dimension = "native" if self.dimensions is None else str(self.dimensions)
        return f"ollama:{self.model}:{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
            "truncate": True,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return [normalize_vector(vector) for vector in embeddings]

    def ping(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the 'production' extra."
            ) from exc
        self.model_name = model
        self.model = SentenceTransformer(model)

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        values = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [np.asarray(value, dtype=np.float32).tolist() for value in values]

    def ping(self) -> bool:
        return True


class CachedEmbedder:
    def __init__(self, base: Embedder, database: Database):
        self.base = base
        self.database = database

    @property
    def model_id(self) -> str:
        return self.base.model_id

    def embed_with_hashes(
        self, texts: Sequence[str], text_hashes: Sequence[str]
    ) -> list[list[float]]:
        if len(texts) != len(text_hashes):
            raise ValueError("texts and text_hashes must have the same length")
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text_hash in enumerate(text_hashes):
            cached = self.database.get_cached_embedding(self.model_id, text_hash)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(texts[index])
            else:
                results[index] = cached

        if missing_texts:
            generated = self.base.embed(missing_texts)
            if len(generated) != len(missing_indices):
                raise RuntimeError("Embedding provider returned the wrong number of vectors")
            for index, vector in zip(missing_indices, generated, strict=True):
                results[index] = vector
                self.database.save_cached_embedding(self.model_id, text_hashes[index], vector)

        return [vector for vector in results if vector is not None]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.base.embed(texts)

    def ping(self) -> bool:
        return self.base.ping()
