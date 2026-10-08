"""RAG pipeline end-to-end: guard input → retrieve → gate relevansi → generate → validasi output."""
from __future__ import annotations

import time

from minesense.config import Settings, get_settings
from minesense.guardrails import (
    SAFETY_FOOTER,
    check_input,
    grounding_score,
    is_safety_critical,
    validate_citations,
)
from minesense.llm.prompts import RAG_SYSTEM, RAG_USER
from minesense.llm.providers import LLM, ExtractiveLLM, LLMError, build_llm
from minesense.logging_utils import get_logger
from minesense.rag.calibration import resolve_dense_threshold
from minesense.rag.index import HybridIndex
from minesense.rag.retriever import HybridRetriever, Retrieved
from minesense.rag.text import light_stem, tokenize
from minesense.schemas import ChatResponse, Citation

log = get_logger(__name__)

NO_ANSWER = ("Informasi tersebut tidak ditemukan dalam dokumen yang tersedia. "
             "Silakan hubungi pengawas area, tim K3, atau KTT untuk konfirmasi.")


def format_context(items: list[Retrieved]) -> str:
    return "\n\n".join(f"[{i}] {it.chunk.title} — {it.chunk.section} ({it.chunk.doc_id})\n{it.chunk.text}"
                       for i, it in enumerate(items, 1))


def to_citations(items: list[Retrieved]) -> list[Citation]:
    return [Citation(doc_id=it.chunk.doc_id, title=it.chunk.title, chunk_id=it.chunk.chunk_id,
                     score=round(it.score, 4), snippet=it.chunk.text[:280]) for it in items]


class RAGPipeline:
    def __init__(self, index: HybridIndex, llm: LLM | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.index = index
        self.retriever = HybridRetriever(index, self.s)
        self.llm = llm or build_llm(self.s)
        self.fallback = ExtractiveLLM()
        self.dense_threshold = resolve_dense_threshold(self.s, index.embedder.name)

    def is_relevant(self, query: str, items: list[Retrieved]) -> bool:
        """Gate out-of-domain: skor retrieval cukup DAN cakupan leksikal kata kunci query di konteks memadai.
        BM25 saja rawan lolos karena 1 kata langka (mis. 'terbaru'), maka dicek coverage token query."""
        if not items:
            return False
        if max(i.dense_score for i in items) >= self.dense_threshold:
            return True
        if max(i.sparse_score for i in items) < self.s.min_bm25_score:
            return False
        q = {light_stem(t) for t in tokenize(query, remove_stop=True)}
        if not q:
            return False
        ctx = {light_stem(t) for t in tokenize(" ".join(f"{i.chunk.title} {i.chunk.section} {i.chunk.text}"
                                                        for i in items))}
        return len(q & ctx) / len(q) >= self.s.min_query_coverage

    def answer(self, question: str, top_k: int | None = None, filters: dict | None = None,
               intent: str = "qa_sop", system: str = RAG_SYSTEM) -> ChatResponse:
        t0 = time.perf_counter()
        guard = check_input(question)
        warnings = list(guard.warnings)
        if guard.blocked:
            return ChatResponse(answer="Permintaan tidak dapat diproses.", citations=[], intent="blocked",
                                grounded=False, warnings=warnings, latency_ms=0.0, provider="guardrail")

        items = self.retriever.retrieve(guard.text, top_k, filters)
        if not self.is_relevant(guard.text, items):
            return ChatResponse(answer=NO_ANSWER, citations=[], intent="out_of_scope", grounded=True,
                                warnings=warnings, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                                provider="gate")

        user = RAG_USER.format(context=format_context(items), question=guard.text)
        provider = self.llm.name
        try:
            text = self.llm.generate(system, user)
        except LLMError as e:
            log.warning(f"LLM error, fallback extractive: {e}")
            warnings.append("LLM utama tidak tersedia; jawaban disusun secara ekstraktif dari dokumen.")
            text, provider = self.fallback.generate(system, user), "extractive-fallback"

        valid, invalid = validate_citations(text, len(items))
        if invalid:
            warnings.append(f"Sitasi tidak valid dihapus: {invalid}")
            for c in invalid:
                text = text.replace(f"[{c}]", "")
        g = grounding_score(text, [it.chunk.text for it in items])
        grounded = g >= 0.6 and (bool(valid) or "tidak ditemukan" in text.lower())
        if not grounded:
            warnings.append(f"Skor grounding rendah ({g:.2f}); verifikasi jawaban dengan dokumen sumber.")
        if is_safety_critical(question) and SAFETY_FOOTER not in text:
            text = f"{text}\n\n{SAFETY_FOOTER}"

        # hanya kembalikan sumber yang benar-benar disitasi (bila ada), agar UI ringkas
        cited_items = [items[c - 1] for c in valid] or items
        cited_items = list({it.chunk.chunk_id: it for it in cited_items}.values())
        return ChatResponse(answer=text, citations=to_citations(cited_items), intent=intent, grounded=grounded,
                            warnings=warnings, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                            provider=provider)
