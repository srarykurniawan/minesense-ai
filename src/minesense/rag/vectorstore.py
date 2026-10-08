"""Vector store: in-memory (numpy, persist ke disk) dan Qdrant (produksi)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def _match(meta: dict, filters: dict[str, str] | None) -> bool:
    return not filters or all(str(meta.get(k)) == str(v) for k, v in filters.items())


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int, filters: dict[str, str] | None = None) -> list[tuple[str, float]]: ...

    def save(self, path: Path) -> None:  # noqa: B027
        pass

    def count(self) -> int:
        return 0


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.payloads: list[dict] = []

    def upsert(self, ids, vectors, payloads):
        pos = {i: n for n, i in enumerate(self.ids)}
        if self.vectors.size == 0:
            self.vectors = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        new_v, new_ids, new_p = [], [], []
        for i, v, p in zip(ids, vectors, payloads):
            if i in pos:
                self.vectors[pos[i]] = v
                self.payloads[pos[i]] = p
            else:
                new_ids.append(i)
                new_v.append(v)
                new_p.append(p)
        if new_ids:
            self.vectors = np.vstack([self.vectors, np.asarray(new_v, dtype=np.float32)])
            self.ids += new_ids
            self.payloads += new_p

    def search(self, vector, k, filters=None):
        if not self.ids:
            return []
        scores = self.vectors @ vector.astype(np.float32)
        order = np.argsort(-scores)
        out = []
        for idx in order:
            if _match(self.payloads[idx], filters):
                out.append((self.ids[idx], float(scores[idx])))
                if len(out) >= k:
                    break
        return out

    def count(self):
        return len(self.ids)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "vector_ids.json").write_text(json.dumps({"ids": self.ids, "payloads": self.payloads}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "InMemoryVectorStore":
        st = cls()
        st.vectors = np.load(path / "vectors.npy")
        d = json.loads((path / "vector_ids.json").read_text(encoding="utf-8"))
        st.ids, st.payloads = d["ids"], d["payloads"]
        return st


class QdrantVectorStore(VectorStore):  # pragma: no cover - butuh server Qdrant
    def __init__(self, url: str, collection: str, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    @staticmethod
    def _pid(s: str) -> str:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))

    def upsert(self, ids, vectors, payloads):
        from qdrant_client.models import PointStruct

        pts = [PointStruct(id=self._pid(i), vector=v.tolist(), payload={**p, "_id": i})
               for i, v, p in zip(ids, vectors, payloads)]
        for b in range(0, len(pts), 256):
            self.client.upsert(self.collection, points=pts[b:b + 256])

    def search(self, vector, k, filters=None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = None
        if filters:
            flt = Filter(must=[FieldCondition(key=k_, match=MatchValue(value=v)) for k_, v in filters.items()])
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k, query_filter=flt).points
        return [(r.payload["_id"], float(r.score)) for r in res]

    def count(self):
        return self.client.count(self.collection).count
