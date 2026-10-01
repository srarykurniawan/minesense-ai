"""Konfigurasi terpusat. Semua nilai bisa di-override via environment variable (prefix MINESENSE_) atau file .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Mode editable/dev: root repo = 2 level di atas paket. Mode terinstal (site-packages): pakai working directory.
_PKG_ROOT = Path(__file__).resolve().parents[2]
ROOT = _PKG_ROOT if (_PKG_ROOT / "data").exists() else Path.cwd()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINESENSE_", env_file=".env", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"

    # --- Security ---
    api_keys: str = ""  # comma-separated; kosong = auth nonaktif (hanya untuk dev)
    rate_limit_per_minute: int = 60
    cors_origins: str = "*"

    # --- Embeddings ---
    embedding_backend: Literal["hashing", "sentence-transformers", "openai"] = "hashing"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 768  # untuk hashing backend

    # --- Vector store ---
    vector_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "minesense"

    # --- Retrieval ---
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Gate relevansi (out-of-domain): lolos jika skor BM25 teratas >= min_bm25 ATAU cosine dense >= min_dense.
    # Kalibrasi ulang min_dense bila ganti embedder (e5/bge memberi cosine lebih tinggi untuk teks tak relevan).
    min_bm25_score: float = 2.0
    min_dense_score: float = 0.45
    min_query_coverage: float = 0.51

    # --- Classifier ---
    classifier_backend: Literal["sklearn", "transformers"] = "sklearn"

    # --- LLM ---
    llm_provider: Literal["extractive", "anthropic", "openai", "ollama"] = "extractive"
    llm_model: str = "claude-sonnet-4-5"
    llm_base_url: str = ""  # untuk openai-compatible (vLLM, Ollama /v1, LM Studio)
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 900
    llm_timeout_s: float = 60.0

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def classifier_path(self) -> Path:
        return self.artifacts_dir / "models" / "incident_classifier.joblib"

    @property
    def feedback_db(self) -> Path:
        return self.artifacts_dir / "feedback" / "feedback.sqlite3"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
