from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .db import Database
from .embeddings import (
    CachedEmbedder,
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
)
from .generation import Generator, OllamaGenerator, RuleBasedGenerator
from .ingestion import IngestionService
from .models import HealthStatus, SourceConfig
from .nli import CrossEncoderNliClassifier, HeuristicNliClassifier, NliClassifier
from .rag import CareAnxRag
from .registry import load_source_registry
from .rerank import CrossEncoderReranker, HeuristicReranker, Reranker
from .retrieval import CareRetriever
from .vector_store import ChromaVectorStore, SQLiteVectorStore, VectorStore


@dataclass(slots=True)
class Runtime:
    settings: Settings
    database: Database
    sources: list[SourceConfig]
    embedder: Embedder
    cached_embedder: CachedEmbedder
    vector_store: VectorStore
    reranker: Reranker
    nli: NliClassifier
    generator: Generator
    ingestion: IngestionService
    retriever: CareRetriever
    rag: CareAnxRag

    def health(self) -> HealthStatus:
        database_ok = self.database.ping() and self.database.integrity_check() == "ok"
        vector_ok = self.vector_store.ping()
        ollama_required = (
            self.settings.embedding_provider == "ollama"
            or self.settings.generator_provider == "ollama"
        )
        ollama_ok: bool | None = None
        if ollama_required:
            checks: list[bool] = []
            if self.settings.embedding_provider == "ollama":
                checks.append(self.embedder.ping())
            if self.settings.generator_provider == "ollama":
                checks.append(self.generator.ping())
            ollama_ok = all(checks)
        stored_embedding_identity = self.database.get_embedding_identity()
        active_or_staged_chunks = self.database.count_chunks(status=None)
        embedding_identity_ok = (
            active_or_staged_chunks == 0
            or stored_embedding_identity == self.embedder.model_id
        )
        all_ok = (
            database_ok
            and vector_ok
            and embedding_identity_ok
            and (ollama_ok is not False)
        )
        status = "ok" if all_ok else "degraded"
        details: dict[str, Any] = {
            "database_path": str(self.settings.database_path),
            "vector_backend": self.settings.vector_backend,
            "embedding_provider": self.settings.embedding_provider,
            "generator_provider": self.settings.generator_provider,
            "reranker_provider": self.settings.reranker_provider,
            "nli_provider": self.settings.nli_provider,
            "runtime_embedding_identity": self.embedder.model_id,
            "stored_embedding_identity": stored_embedding_identity,
            "embedding_identity_ok": embedding_identity_ok,
        }
        try:
            details["clinical_vectors"] = self.vector_store.count(
                self.settings.clinical_collection
            )
            details["research_vectors"] = self.vector_store.count(
                self.settings.research_collection
            )
        except Exception as exc:
            details["vector_count_error"] = str(exc)
        return HealthStatus(
            status=status,
            database=database_ok,
            vector_store=vector_ok,
            ollama=ollama_ok,
            details=details,
        )


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    sources = load_source_registry(settings.source_registry_path)
    database.upsert_sources(sources)

    embedder = _build_embedder(settings)
    cached_embedder = CachedEmbedder(embedder, database)
    vector_store = _build_vector_store(settings)
    reranker = _build_reranker(settings)
    nli = _build_nli(settings)
    generator = _build_generator(settings)
    ingestion = IngestionService(
        settings,
        database,
        sources,
        cached_embedder,
        vector_store,
    )
    retriever = CareRetriever(
        settings,
        database,
        vector_store,
        embedder,
        reranker,
        nli,
    )
    rag = CareAnxRag(settings, retriever, generator)
    return Runtime(
        settings=settings,
        database=database,
        sources=sources,
        embedder=embedder,
        cached_embedder=cached_embedder,
        vector_store=vector_store,
        reranker=reranker,
        nli=nli,
        generator=generator,
        ingestion=ingestion,
        retriever=retriever,
        rag=rag,
    )


def _build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder(settings.embedding_dimensions)
    if settings.embedding_provider == "ollama":
        dimensions = settings.embedding_dimensions if settings.embedding_dimensions > 0 else None
        return OllamaEmbedder(
            settings.ollama_base_url,
            settings.embedding_model,
            settings.request_timeout_seconds,
            dimensions,
        )
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def _build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "sqlite":
        return SQLiteVectorStore(settings.database_path)
    if settings.vector_backend == "chroma":
        return ChromaVectorStore(settings.vector_path)
    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")


def _build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "heuristic":
        return HeuristicReranker()
    return CrossEncoderReranker(settings.reranker_model)


def _build_nli(settings: Settings) -> NliClassifier:
    if settings.nli_provider == "heuristic":
        return HeuristicNliClassifier()
    return CrossEncoderNliClassifier(settings.nli_model)


def _build_generator(settings: Settings) -> Generator:
    if settings.generator_provider == "rule":
        return RuleBasedGenerator()
    return OllamaGenerator(
        settings.ollama_base_url,
        settings.generation_model,
        max(120.0, settings.request_timeout_seconds),
    )
