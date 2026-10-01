"""Embedding backends (pluggable).

- hashing: offline, tanpa download model, deterministik. Cocok untuk CI/dev & CPU kecil.
- sentence-transformers: multilingual-e5 / bge-m3 (kualitas semantik jauh lebih baik).
- openai: endpoint /embeddings yang kompatibel OpenAI (OpenAI, vLLM, Ollama, TEI).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from minesense.config import Settings
from minesense.rag.text import normalize


class Embedder(ABC):
    name: str = "base"
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class HashingEmbedder(Embedder):
    """Char n-gram (3–5) + word unigram/bigram di-hash ke ruang berdimensi tetap.
    Robust terhadap imbuhan Bahasa Indonesia (mem-, -kan, -an) karena n-gram karakter."""

    name = "hashing"

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=dim,
                                       alternate_sign=False, norm="l2", preprocessor=normalize)
        self._word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=dim,
                                       alternate_sign=False, norm="l2", preprocessor=normalize)

    def _embed(self, texts: list[str]) -> np.ndarray:
        a = self._char.transform(texts).toarray()
        b = self._word.transform(texts).toarray()
        return _l2(0.6 * a + 0.4 * b)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover - butuh download model
    name = "sentence-transformers"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._e5 = "e5" in model.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        q = f"query: {text}" if self._e5 else text
        return self.model.encode([q], normalize_embeddings=True)[0].astype(np.float32)


class OpenAICompatEmbedder(Embedder):  # pragma: no cover - butuh endpoint
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str):
        import httpx

        self.model = model
        self.client = httpx.Client(base_url=base_url or "https://api.openai.com/v1",
                                   headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        self.dim = len(self.embed_query("tes"))

    def _call(self, texts: list[str]) -> np.ndarray:
        r = self.client.post("/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        return _l2(np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = [self._call(texts[i:i + 64]) for i in range(0, len(texts), 64)]
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(s.embedding_model)
    if s.embedding_backend == "openai":
        return OpenAICompatEmbedder(s.embedding_model, s.llm_base_url, s.llm_api_key)
    return HashingEmbedder(s.embedding_dim)
