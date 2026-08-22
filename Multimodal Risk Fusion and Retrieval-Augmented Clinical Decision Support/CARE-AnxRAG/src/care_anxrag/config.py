from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value {value!r}; expected one of "
        "1/0, true/false, yes/no, or on/off"
    )


def _as_float(value: str | None, default: float) -> float:
    return default if value is None else float(value)


def _as_int(value: str | None, default: int) -> int:
    return default if value is None else int(value)


def _resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve paths predictably, treating relative configuration as project-relative."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_dotenv(path: Path) -> None:
    """Load a minimal .env file without overriding existing environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class RetrievalWeights:
    semantic: float = 0.20
    lexical: float = 0.08
    rrf: float = 0.07
    rerank: float = 0.25
    authority: float = 0.14
    evidence: float = 0.13
    freshness: float = 0.06
    applicability: float = 0.07

    def validate(self) -> None:
        values = [
            self.semantic,
            self.lexical,
            self.rrf,
            self.rerank,
            self.authority,
            self.evidence,
            self.freshness,
            self.applicability,
        ]
        if any(value < 0 for value in values):
            raise ValueError("Retrieval weights must be non-negative")
        total = sum(values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Retrieval weights must sum to 1.0; got {total:.6f}")


@dataclass(slots=True)
class Settings:
    project_root: Path
    care_home: Path
    database_path: Path
    vector_path: Path
    source_registry_path: Path

    vector_backend: str = "chroma"
    embedding_provider: str = "ollama"
    embedding_model: str = "embeddinggemma"
    embedding_dimensions: int = 256
    generator_provider: str = "ollama"
    generation_model: str = "gemma3:4b"
    ollama_base_url: str = "http://localhost:11434"

    reranker_provider: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    nli_provider: str = "cross_encoder"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    dense_candidates: int = 40
    lexical_candidates: int = 40
    fused_candidates: int = 30
    rerank_candidates: int = 20
    final_context_chunks: int = 6
    rrf_k: int = 60

    minimum_care_score: float = 0.43
    minimum_relevance_score: float = 0.24
    minimum_confidence: float = 0.50
    contradiction_threshold: float = 0.72
    unresolved_conflict_threshold: float = 0.32
    dominance_margin: float = 0.15
    min_distinct_sources: int = 1

    clinical_half_life_days: int = 3650
    research_half_life_days: int = 1095
    request_timeout_seconds: float = 60.0
    source_user_agent: str = "CARE-AnxRAG/0.1 research-contact-required"
    admin_key: str = ""
    crisis_resource_text: str = (
        "If there is immediate danger or you may act on thoughts of self-harm, "
        "contact local emergency services now or go to the nearest emergency department. "
        "Contact a local crisis line or a trusted person who can stay with you."
    )
    allow_network_sync: bool = True
    auto_promote_research: bool = False
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)

    @classmethod
    def from_env(
        cls,
        project_root: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        root = Path(project_root or os.getenv("CARE_PROJECT_ROOT", Path.cwd())).resolve()
        load_dotenv(root / ".env")
        env = dict(os.environ if environ is None else environ)
        care_home = _resolve_path(env.get("CARE_HOME", root / "var"), root)
        settings = cls(
            project_root=root,
            care_home=care_home,
            database_path=_resolve_path(
                env.get("CARE_DATABASE_PATH", care_home / "care_anxrag.sqlite3"), root
            ),
            vector_path=_resolve_path(env.get("CARE_VECTOR_PATH", care_home / "chroma"), root),
            source_registry_path=_resolve_path(
                env.get("CARE_SOURCE_REGISTRY", root / "config" / "sources.yaml"), root
            ),
            vector_backend=env.get("CARE_VECTOR_BACKEND", "chroma").lower(),
            embedding_provider=env.get("CARE_EMBEDDING_PROVIDER", "ollama").lower(),
            embedding_model=env.get("CARE_EMBEDDING_MODEL", "embeddinggemma"),
            embedding_dimensions=_as_int(env.get("CARE_EMBEDDING_DIMENSIONS"), 256),
            generator_provider=env.get("CARE_GENERATOR_PROVIDER", "ollama").lower(),
            generation_model=env.get("CARE_GENERATION_MODEL", "gemma3:4b"),
            ollama_base_url=env.get("CARE_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            reranker_provider=env.get("CARE_RERANKER_PROVIDER", "cross_encoder").lower(),
            reranker_model=env.get(
                "CARE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
            ),
            nli_provider=env.get("CARE_NLI_PROVIDER", "cross_encoder").lower(),
            nli_model=env.get("CARE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"),
            dense_candidates=_as_int(env.get("CARE_DENSE_CANDIDATES"), 40),
            lexical_candidates=_as_int(env.get("CARE_LEXICAL_CANDIDATES"), 40),
            fused_candidates=_as_int(env.get("CARE_FUSED_CANDIDATES"), 30),
            rerank_candidates=_as_int(env.get("CARE_RERANK_CANDIDATES"), 20),
            final_context_chunks=_as_int(env.get("CARE_FINAL_CONTEXT_CHUNKS"), 6),
            rrf_k=_as_int(env.get("CARE_RRF_K"), 60),
            minimum_care_score=_as_float(env.get("CARE_MINIMUM_CARE_SCORE"), 0.43),
            minimum_relevance_score=_as_float(
                env.get("CARE_MINIMUM_RELEVANCE_SCORE"), 0.24
            ),
            minimum_confidence=_as_float(env.get("CARE_MINIMUM_CONFIDENCE"), 0.50),
            contradiction_threshold=_as_float(
                env.get("CARE_CONTRADICTION_THRESHOLD"), 0.72
            ),
            unresolved_conflict_threshold=_as_float(
                env.get("CARE_UNRESOLVED_CONFLICT_THRESHOLD"), 0.32
            ),
            dominance_margin=_as_float(env.get("CARE_DOMINANCE_MARGIN"), 0.15),
            min_distinct_sources=_as_int(env.get("CARE_MIN_DISTINCT_SOURCES"), 1),
            clinical_half_life_days=_as_int(
                env.get("CARE_CLINICAL_HALF_LIFE_DAYS"), 3650
            ),
            research_half_life_days=_as_int(
                env.get("CARE_RESEARCH_HALF_LIFE_DAYS"), 1095
            ),
            request_timeout_seconds=_as_float(
                env.get("CARE_REQUEST_TIMEOUT_SECONDS"), 60.0
            ),
            source_user_agent=env.get(
                "CARE_SOURCE_USER_AGENT",
                "CARE-AnxRAG/0.1 research-contact-required",
            ),
            admin_key=env.get("CARE_ADMIN_KEY", ""),
            crisis_resource_text=env.get(
                "CARE_CRISIS_RESOURCE_TEXT",
                cls.__dataclass_fields__["crisis_resource_text"].default,
            ),
            allow_network_sync=_as_bool(env.get("CARE_ALLOW_NETWORK_SYNC"), True),
            auto_promote_research=_as_bool(env.get("CARE_AUTO_PROMOTE_RESEARCH"), False),
            weights=RetrievalWeights(
                semantic=_as_float(env.get("CARE_WEIGHT_SEMANTIC"), 0.20),
                lexical=_as_float(env.get("CARE_WEIGHT_LEXICAL"), 0.08),
                rrf=_as_float(env.get("CARE_WEIGHT_RRF"), 0.07),
                rerank=_as_float(env.get("CARE_WEIGHT_RERANK"), 0.25),
                authority=_as_float(env.get("CARE_WEIGHT_AUTHORITY"), 0.14),
                evidence=_as_float(env.get("CARE_WEIGHT_EVIDENCE"), 0.13),
                freshness=_as_float(env.get("CARE_WEIGHT_FRESHNESS"), 0.06),
                applicability=_as_float(env.get("CARE_WEIGHT_APPLICABILITY"), 0.07),
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        self.weights.validate()
        if self.vector_backend not in {"chroma", "sqlite"}:
            raise ValueError("CARE_VECTOR_BACKEND must be 'chroma' or 'sqlite'")
        if self.embedding_provider not in {"ollama", "sentence_transformers", "hash"}:
            raise ValueError("Unsupported embedding provider")
        if self.generator_provider not in {"ollama", "rule"}:
            raise ValueError("Unsupported generator provider")
        if self.reranker_provider not in {"cross_encoder", "heuristic"}:
            raise ValueError("Unsupported reranker provider")
        if self.nli_provider not in {"cross_encoder", "heuristic"}:
            raise ValueError("Unsupported NLI provider")
        positive_ints = [
            self.dense_candidates,
            self.lexical_candidates,
            self.fused_candidates,
            self.rerank_candidates,
            self.final_context_chunks,
            self.rrf_k,
        ]
        if any(value <= 0 for value in positive_ints):
            raise ValueError("Candidate counts and rrf_k must be positive")
        if self.embedding_dimensions < 0:
            raise ValueError("CARE_EMBEDDING_DIMENSIONS cannot be negative")
        if self.embedding_provider == "hash" and self.embedding_dimensions <= 0:
            raise ValueError("Hash embeddings require CARE_EMBEDDING_DIMENSIONS > 0")
        if (
            self.embedding_provider == "ollama"
            and self.embedding_model.split(":", 1)[0].lower() == "embeddinggemma"
            and self.embedding_dimensions not in {0, 128, 256, 512, 768}
        ):
            raise ValueError(
                "embeddinggemma supports CARE_EMBEDDING_DIMENSIONS values "
                "0 (native), 128, 256, 512, or 768"
            )
        if self.min_distinct_sources <= 0:
            raise ValueError("CARE_MIN_DISTINCT_SOURCES must be positive")
        if self.clinical_half_life_days <= 0 or self.research_half_life_days <= 0:
            raise ValueError("Evidence half-life values must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("CARE_REQUEST_TIMEOUT_SECONDS must be positive")
        if self.rerank_candidates > self.fused_candidates:
            raise ValueError("CARE_RERANK_CANDIDATES cannot exceed CARE_FUSED_CANDIDATES")
        if self.final_context_chunks > self.rerank_candidates:
            raise ValueError("CARE_FINAL_CONTEXT_CHUNKS cannot exceed CARE_RERANK_CANDIDATES")
        for value in [
            self.minimum_care_score,
            self.minimum_relevance_score,
            self.minimum_confidence,
            self.contradiction_threshold,
            self.unresolved_conflict_threshold,
            self.dominance_margin,
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError("Thresholds must be between 0 and 1")

    def ensure_directories(self) -> None:
        self.care_home.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)

    @property
    def clinical_collection(self) -> str:
        return "care_clinical_core"

    @property
    def research_collection(self) -> str:
        return "care_research_frontier"
