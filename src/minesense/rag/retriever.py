"""Hybrid retriever: dense + BM25 → Reciprocal Rank Fusion → (opsional) cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass

from minesense.config import Settings, get_settings
from minesense.rag.chunking import Chunk
from minesense.rag.index import HybridIndex


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float = 0.0
    sparse_score: float = 0.0


def rrf_fuse(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + w / (k + r + 1)
    return fused


class CrossEncoderReranker:  # pragma: no cover - butuh download model
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, max_length=512)

    def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        scores = self.model.predict([(query, it.chunk.text) for it in items])
        for it, s in zip(items, scores):
            it.score = float(s)
        return sorted(items, key=lambda x: -x.score)


class FastEmbedReranker:  # pragma: no cover - butuh download model
    """Cross-encoder multilingual via ONNX (tanpa torch), mis. jinaai/jina-reranker-v2-base-multilingual."""

    def __init__(self, model: str, cache_dir: str | None = None):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:
            raise RuntimeError("Install: pip install fastembed") from e
        self.model = TextCrossEncoder(model_name=model, cache_dir=cache_dir)

    def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        scores = list(self.model.rerank(query, [f"{it.chunk.title} | {it.chunk.section}\n{it.chunk.text}"
                                                for it in items]))
        for it, sc in zip(items, scores):
            it.score = float(sc)
        return sorted(items, key=lambda x: -x.score)


def build_reranker(s: Settings):
    if s.reranker == "fastembed":
        return FastEmbedReranker(s.reranker_model, cache_dir=str(s.model_cache_dir))
    if s.reranker == "cross-encoder":
        return CrossEncoderReranker(s.reranker_model)
    return None


class HybridRetriever:
    def __init__(self, index: HybridIndex, settings: Settings | None = None):
        self.index = index
        self.s = settings or get_settings()
        self.reranker = build_reranker(self.s)

    def retrieve(self, query: str, top_k: int | None = None, filters: dict | None = None,
                 mode: str = "hybrid") -> list[Retrieved]:
        """mode: hybrid (default) | dense | sparse — dense/sparse dipakai untuk benchmark/ablation."""
        k = top_k or self.s.top_k
        dense = self.index.dense_search(query, self.s.candidate_k, filters) if mode != "sparse" else []
        sparse = self.index.sparse_search(query, self.s.candidate_k, filters) if mode != "dense" else []
        d_ids, s_ids = [c for c, _ in dense], [c for c, _ in sparse]
        fused = rrf_fuse([d_ids, s_ids], k=self.s.rrf_k)
        d_rank = {c: i for i, c in enumerate(d_ids)}
        s_rank = {c: i for i, c in enumerate(s_ids)}
        d_sc, s_sc = dict(dense), dict(sparse)
        items = [Retrieved(self.index.chunks[c], sc, d_rank.get(c), s_rank.get(c), d_sc.get(c, 0.0), s_sc.get(c, 0.0))
                 for c, sc in sorted(fused.items(), key=lambda x: -x[1]) if c in self.index.chunks]
        if self.reranker and items:
            items = self.reranker.rerank(query, items[: self.s.rerank_candidates]) + items[self.s.rerank_candidates:]
        return items[:k]
