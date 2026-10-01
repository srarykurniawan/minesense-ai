"""Index hybrid: dense (vector store) + sparse (BM25). Build, persist, load."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from minesense.config import Settings, get_settings
from minesense.logging_utils import get_logger
from minesense.rag.chunking import Chunk, Document, chunk_document, load_corpus
from minesense.rag.embeddings import Embedder, build_embedder
from minesense.rag.text import tokenize
from minesense.rag.vectorstore import InMemoryVectorStore, QdrantVectorStore, VectorStore

log = get_logger(__name__)


class HybridIndex:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.s = settings or get_settings()
        self.embedder = embedder or build_embedder(self.s)
        self.chunks: dict[str, Chunk] = {}
        self.store: VectorStore = self._new_store()
        self.bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []

    def _new_store(self) -> VectorStore:
        if self.s.vector_backend == "qdrant":
            return QdrantVectorStore(self.s.qdrant_url, self.s.qdrant_collection, self.embedder.dim)
        return InMemoryVectorStore()

    # ---------------- build ----------------
    def add_documents(self, docs: list[Document]) -> int:
        new_chunks: list[Chunk] = []
        for d in docs:
            # re-ingest dokumen yang sama → ganti chunk lama
            for cid in [c for c, ch in self.chunks.items() if ch.doc_id == d.doc_id]:
                del self.chunks[cid]
            new_chunks += chunk_document(d, self.s.chunk_size, self.s.chunk_overlap)
        if not new_chunks:
            return 0
        t0 = time.time()
        vecs = self.embedder.embed_documents([c.embed_text for c in new_chunks])
        payloads = [{"doc_id": c.doc_id, **{k: v for k, v in c.metadata.items() if k in ("doc_type", "equipment", "department")}}
                    for c in new_chunks]
        self.store.upsert([c.chunk_id for c in new_chunks], vecs, payloads)
        for c in new_chunks:
            self.chunks[c.chunk_id] = c
        self._rebuild_bm25()
        log.info(f"indexed {len(new_chunks)} chunks from {len(docs)} docs in {time.time() - t0:.2f}s")
        return len(new_chunks)

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self.chunks)
        corpus = [tokenize(self.chunks[i].embed_text) for i in self._bm25_ids]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def build_from_dir(self, directory: Path) -> int:
        return self.add_documents(load_corpus(directory))

    # ---------------- persist ----------------
    def save(self, path: Path | None = None) -> Path:
        path = path or self.s.index_dir
        path.mkdir(parents=True, exist_ok=True)
        self.store.save(path)
        (path / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks.values()], ensure_ascii=False))
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump((self._bm25_ids, self.bm25), f)
        (path / "manifest.json").write_text(json.dumps({
            "embedder": self.embedder.name, "dim": self.embedder.dim, "n_chunks": len(self.chunks),
            "n_docs": len({c.doc_id for c in self.chunks.values()}), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        return path

    @classmethod
    def load(cls, settings: Settings | None = None, path: Path | None = None) -> "HybridIndex":
        idx = cls(settings)
        path = path or idx.s.index_dir
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["embedder"] != idx.embedder.name:
            raise RuntimeError(f"Index dibangun dengan embedder '{manifest['embedder']}', "
                               f"config saat ini '{idx.embedder.name}'. Jalankan ulang `minesense ingest`.")
        idx.chunks = {d["chunk_id"]: Chunk(**d) for d in json.loads((path / "chunks.json").read_text())}
        if idx.s.vector_backend == "memory":
            idx.store = InMemoryVectorStore.load(path)
        with open(path / "bm25.pkl", "rb") as f:
            idx._bm25_ids, idx.bm25 = pickle.load(f)
        return idx

    # ---------------- query primitives ----------------
    def dense_search(self, query: str, k: int, filters=None) -> list[tuple[str, float]]:
        return self.store.search(self.embedder.embed_query(query), k, filters)

    def sparse_search(self, query: str, k: int, filters=None) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1]
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            cid = self._bm25_ids[i]
            ch = self.chunks[cid]
            if filters and not all(str({**ch.metadata, "doc_id": ch.doc_id}.get(a)) == str(b) for a, b in filters.items()):
                continue
            out.append((cid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def stats(self) -> dict:
        return {"n_chunks": len(self.chunks), "n_docs": len({c.doc_id for c in self.chunks.values()}),
                "embedder": self.embedder.name, "vector_backend": self.s.vector_backend}
