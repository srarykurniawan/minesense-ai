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


class HybridRetriever:
    def __init__(self, index: HybridIndex, settings: Settings | None = None):
        self.index = index
        self.s = settings or get_settings()
        self.reranker = CrossEncoderReranker(self.s.reranker_model) if self.s.reranker == "cross-encoder" else None

    def retrieve(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[Retrieved]:
        k = top_k or self.s.top_k
        dense = self.index.dense_search(query, self.s.candidate_k, filters)
        sparse = self.index.sparse_search(query, self.s.candidate_k, filters)
        d_ids, s_ids = [c for c, _ in dense], [c for c, _ in sparse]
        fused = rrf_fuse([d_ids, s_ids], k=self.s.rrf_k)
        d_rank = {c: i for i, c in enumerate(d_ids)}
        s_rank = {c: i for i, c in enumerate(s_ids)}
        d_sc, s_sc = dict(dense), dict(sparse)
        items = [Retrieved(self.index.chunks[c], sc, d_rank.get(c), s_rank.get(c), d_sc.get(c, 0.0), s_sc.get(c, 0.0))
                 for c, sc in sorted(fused.items(), key=lambda x: -x[1]) if c in self.index.chunks]
        if self.reranker and items:
            items = self.reranker.rerank(query, items[: self.s.candidate_k])
        return items[:k]
