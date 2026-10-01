"""FastAPI service MineSense. Jalankan: uvicorn minesense.api.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from minesense import __version__
from minesense.config import get_settings
from minesense.extraction.shift_report import extract
from minesense.llm.prompts import PROMPT_VERSION
from minesense.logging_utils import get_logger, setup_logging
from minesense.schemas import (
    ChatRequest,
    ChatResponse,
    ExtractRequest,
    ExtractResponse,
    FeedbackRequest,
    IncidentPrediction,
    IncidentRequest,
    MaintenanceRequest,
    MaintenanceResponse,
    SearchRequest,
)
from minesense.services import Services

setup_logging()
log = get_logger("minesense.api")

REQS = Counter("minesense_requests_total", "Total HTTP requests", ["path", "method", "status"])
LAT = Histogram("minesense_request_latency_seconds", "Request latency", ["path"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
INTENTS = Counter("minesense_chat_intent_total", "Chat intents", ["intent", "grounded"])
FEEDBACK = Counter("minesense_feedback_total", "User feedback", ["rating"])

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    STATE["svc"] = await run_in_threadpool(Services)
    log.info(f"startup selesai dalam {time.time() - t0:.1f}s")
    yield
    STATE.clear()


s = get_settings()
app = FastAPI(title="MineSense AI API", version=__version__, lifespan=lifespan,
              description="RAG K3/SOP, klasifikasi insiden, ekstraksi laporan shift, dan diagnosa maintenance tambang.")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        resp.headers["x-request-id"] = rid
        return resp
    finally:
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        REQS.labels(path, request.method, str(status)).inc()
        LAT.labels(path).observe(dt)
        if path not in ("/health", "/metrics"):
            log.info("request", extra={"request_id": rid, "path": path, "status": status,
                                       "latency_ms": round(dt * 1000, 1)})


# ---------------- security ----------------
_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
_hits: dict[str, deque] = defaultdict(deque)


def auth(request: Request, key: str | None = Depends(_api_key)) -> str:
    st = get_settings()
    if st.api_key_set and key not in st.api_key_set:
        raise HTTPException(401, "API key tidak valid atau tidak ada (header X-API-Key)")
    ident = key or (request.client.host if request.client else "anon")
    # rate limit sliding window per key/IP (per proses; gunakan Redis/gateway untuk multi-replika)
    q, now = _hits[ident], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= st.rate_limit_per_minute:
        raise HTTPException(429, "Terlalu banyak permintaan, coba lagi sebentar")
    q.append(now)
    return ident


def svc() -> Services:
    if "svc" not in STATE:
        raise HTTPException(503, "Service belum siap")
    STATE["svc"].refresh_if_stale()
    return STATE["svc"]


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Terjadi kesalahan internal"})


# ---------------- ops endpoints ----------------
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    if "svc" not in STATE:
        raise HTTPException(503, "not ready")
    return {"status": "ready", **STATE["svc"].index.stats()}


@app.get("/metrics", tags=["ops"])
def metrics():
    # uvicorn --workers N: agregasi metrik lintas proses bila PROMETHEUS_MULTIPROC_DIR di-set
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import CollectorRegistry, multiprocess
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/info", tags=["ops"], dependencies=[Depends(auth)])
def info(sv: Services = Depends(svc)):
    return {"version": __version__, "prompt_version": PROMPT_VERSION, "env": sv.s.env, "llm": sv.rag.llm.name,
            "llm_model": sv.s.llm_model if sv.rag.llm.name != "extractive" else None, "index": sv.index.stats(),
            "classifier": sv.classifier.meta, "feedback": sv.feedback.summary(),
            "uptime_s": round(time.time() - sv.started)}


# ---------------- AI endpoints ----------------
@app.post("/v1/chat", response_model=ChatResponse, tags=["ai"])
async def chat(req: ChatRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    res = await run_in_threadpool(sv.chat, req.question, req.top_k, req.filters)
    INTENTS.labels(res.intent, str(res.grounded)).inc()
    return res


@app.post("/v1/search", tags=["ai"])
async def search(req: SearchRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    items = await run_in_threadpool(sv.rag.retriever.retrieve, req.query, req.top_k, req.filters)
    return [{"chunk_id": i.chunk.chunk_id, "doc_id": i.chunk.doc_id, "title": i.chunk.title,
             "section": i.chunk.section, "score": round(i.score, 4), "text": i.chunk.text} for i in items]


@app.post("/v1/incidents/classify", response_model=IncidentPrediction, tags=["ai"])
def classify(req: IncidentRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    return sv.classifier.predict(req.description)


@app.post("/v1/reports/extract", response_model=ExtractResponse, tags=["ai"])
async def extract_report(req: ExtractRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    res, warns = await run_in_threadpool(extract, req.text, sv.rag.llm, req.use_llm)
    res.warnings = warns
    return res


@app.post("/v1/maintenance/diagnose", response_model=MaintenanceResponse, tags=["ai"])
async def diagnose(req: MaintenanceRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    return await run_in_threadpool(sv.maintenance.diagnose, req)


@app.post("/v1/documents", tags=["admin"])
async def upload_document(file: UploadFile = File(...), sv: Services = Depends(svc), _: str = Depends(auth)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".md", ".txt", ".pdf"):
        raise HTTPException(415, "Format didukung: .md, .txt, .pdf")
    dest_dir = sv.s.data_dir / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dest_dir) as tmp:
        shutil.copyfileobj(file.file, tmp)
    final = dest_dir / safe_name
    Path(tmp.name).replace(final)
    return await run_in_threadpool(sv.ingest_file, final)


@app.post("/v1/feedback", tags=["ai"])
def feedback(req: FeedbackRequest, sv: Services = Depends(svc), _: str = Depends(auth)):
    FEEDBACK.labels(str(req.rating)).inc()
    return {"id": sv.feedback.add(**req.model_dump())}
