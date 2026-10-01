"""Service container: memuat semua komponen sekali saat startup dan menyediakan operasi bisnis."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from minesense.agent.router import route
from minesense.classifier.model import IncidentClassifier
from minesense.config import Settings, get_settings
from minesense.extraction.shift_report import extract
from minesense.logging_utils import get_logger
from minesense.maintenance.diagnose import MaintenanceAssistant
from minesense.rag.chunking import load_document
from minesense.rag.index import HybridIndex
from minesense.rag.pipeline import RAGPipeline
from minesense.schemas import ChatResponse, MaintenanceRequest

log = get_logger(__name__)


class FeedbackStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
            session_id TEXT, question TEXT, answer TEXT, rating INTEGER, comment TEXT)""")
        self.conn.commit()

    def add(self, **kw) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO feedback (ts, session_id, question, answer, rating, comment) VALUES (?,?,?,?,?,?)",
                (time.time(), kw.get("session_id"), kw["question"], kw["answer"], kw["rating"], kw.get("comment")))
            self.conn.commit()
            return int(cur.lastrowid)

    def summary(self) -> dict:
        row = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(rating=1),0), COALESCE(SUM(rating=-1),0) FROM feedback").fetchone()
        return {"total": row[0], "positive": row[1], "negative": row[2]}


class Services:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.started = time.time()
        self.index = self._load_or_build_index()
        self.rag = RAGPipeline(self.index, settings=self.s)
        self.classifier = self._load_classifier()
        self.maintenance = MaintenanceAssistant(self.s.data_dir, self.rag)
        self.feedback = FeedbackStore(self.s.feedback_db)
        self._ingest_lock = threading.Lock()
        self._manifest_mtime = self._mtime()
        self._last_check = time.time()
        log.info(f"services ready: {self.index.stats()} llm={self.rag.llm.name}")

    # ---------------- loading ----------------
    def _load_or_build_index(self) -> HybridIndex:
        manifest = self.s.index_dir / "manifest.json"
        if manifest.exists():
            try:
                return HybridIndex.load(self.s)
            except Exception as e:  # index usang/beda embedder → rebuild
                log.warning(f"gagal memuat index ({e}); membangun ulang")
        idx = HybridIndex(self.s)
        idx.build_from_dir(self.s.data_dir / "corpus")
        if (self.s.data_dir / "uploads").exists():  # dokumen unggahan pengguna ikut diindeks ulang
            idx.build_from_dir(self.s.data_dir / "uploads")
        idx.save()
        return idx

    def _load_classifier(self):
        if self.s.classifier_backend == "transformers":
            from minesense.classifier.hf_backend import TransformerIncidentClassifier
            return TransformerIncidentClassifier(self.s.artifacts_dir / "models")
        if not self.s.classifier_path.exists():
            log.warning("model classifier belum ada; melatih baseline sekarang")
            from minesense.classifier.train import main as train_main
            train_main([])
        return IncidentClassifier.load(self.s.classifier_path)

    # ---------------- multi-worker consistency ----------------
    def _mtime(self) -> float:
        m = self.s.index_dir / "manifest.json"
        return m.stat().st_mtime if m.exists() else 0.0

    def refresh_if_stale(self, every_s: float = 10.0) -> None:
        """Worker lain (uvicorn --workers N) mungkin meng-ingest dokumen baru → muat ulang index bila manifest berubah."""
        if self.s.vector_backend != "memory" or time.time() - self._last_check < every_s:
            return
        self._last_check = time.time()
        mt = self._mtime()
        if mt > self._manifest_mtime:
            with self._ingest_lock:
                self.index = HybridIndex.load(self.s)
                self.rag.index = self.index
                self.rag.retriever.index = self.index
                self._manifest_mtime = mt
            log.info("index dimuat ulang (berubah oleh worker lain)")

    # ---------------- operations ----------------
    def ingest_file(self, path: Path) -> dict:
        doc = load_document(path)
        with self._ingest_lock:
            n = self.index.add_documents([doc])
            self.index.save()
            self._manifest_mtime = self._mtime()
        return {"doc_id": doc.doc_id, "title": doc.title, "chunks": n}

    def chat(self, message: str, top_k: int | None = None, filters: dict | None = None) -> ChatResponse:
        intent = route(message)
        t0 = time.perf_counter()
        if intent == "incident_report":
            pred = self.classifier.predict(message)
            sop = self.rag.answer(f"Tindakan dan prosedur untuk insiden {pred.category.replace('_', ' ')}: {message}",
                                  top_k=3, intent=intent)
            text = (f"**Klasifikasi insiden:** {pred.category} (keyakinan {pred.category_confidence:.0%}) — "
                    f"risiko **{pred.risk_level.upper()}**" + (" → WAJIB ESKALASI" if pred.requires_escalation else "")
                    + "\n\n**Tindakan segera yang disarankan:**\n" + "\n".join(f"- {a}" for a in pred.recommended_actions)
                    + "\n\n**Rujukan SOP:**\n" + sop.answer)
            return sop.model_copy(update={"answer": text, "intent": intent, "data": pred.model_dump(),
                                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        if intent == "shift_report":
            res, warns = extract(message, self.rag.llm, use_llm=self.rag.llm.name != "extractive")
            res.warnings = warns
            d = res.data
            text = (f"Laporan terbaca ({res.method}): {d.pit or '-'} shift {d.shift or '-'} {d.tanggal or ''} — "
                    f"OB {d.ob_removal_bcm or '-'} BCM, coal {d.coal_getting_ton or '-'} ton, SR {d.stripping_ratio or '-'}."
                    + (f"\nField belum terisi: {', '.join(res.missing_fields)}" if res.missing_fields else ""))
            return ChatResponse(answer=text, citations=[], intent=intent, grounded=True, warnings=warns,
                                latency_ms=round((time.perf_counter() - t0) * 1000, 1), provider=res.method,
                                data=res.model_dump())
        if intent == "maintenance":
            import re
            unit = re.search(r"\b(?:DT|EX|BD|GR)-\d{3,5}\b", message, re.I)
            code = re.search(r"\b[A-Z]-[A-Z]{2,3}-\d{3}\b", message)
            eq = unit.group(0).upper() if unit else (self.maintenance.faults[code.group(0)]["model"]
                                                    if code and code.group(0) in self.maintenance.faults else "unknown")
            r = self.maintenance.diagnose(MaintenanceRequest(equipment=eq, symptom=message,
                                                             fault_code=code.group(0) if code else None))
            return ChatResponse(answer=r.answer, citations=r.citations, intent=intent, grounded=True, warnings=[],
                                latency_ms=round((time.perf_counter() - t0) * 1000, 1), provider=self.rag.llm.name,
                                data=r.model_dump(exclude={"citations", "answer"}))
        return self.rag.answer(message, top_k, filters, intent="qa_sop")
