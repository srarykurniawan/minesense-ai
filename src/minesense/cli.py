"""CLI: minesense <command>

  synth    — generate data sintetis
  ingest   — bangun index RAG dari data/corpus (+ data/uploads)
  train    — latih classifier insiden (baseline)
  sft      — bangun dataset fine-tuning LLM
  eval     — evaluasi menyeluruh + quality gates (+ set v2 jujur, regression gate, LLM-judge)
  benchmark— bandingkan embedder × mode retrieval × reranker (set v2)
  calibrate— kalibrasi ambang relevansi dense untuk embedder aktif
  bootstrap— synth + ingest + train + sft (setup awal)
  serve    — jalankan API
  ask      — tanya cepat dari terminal
"""
from __future__ import annotations

import argparse
import json
import sys


def _utf8_stdio() -> None:
    """Windows (cp1252) tidak bisa mencetak karakter seperti → atau ≥; paksa UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    _utf8_stdio()
    ap = argparse.ArgumentParser(prog="minesense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("synth", "ingest", "train", "sft", "bootstrap", "calibrate"):
        sub.add_parser(c)
    ev = sub.add_parser("eval")
    ev.add_argument("--update-baseline", action="store_true")
    ev.add_argument("--no-judge", action="store_true")
    bm = sub.add_parser("benchmark")
    bm.add_argument("--full", action="store_true")
    bm.add_argument("--configs", default="")
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--workers", type=int, default=1)
    ak = sub.add_parser("ask")
    ak.add_argument("question")
    a = ap.parse_args()

    from minesense.config import get_settings
    s = get_settings()

    def synth():
        from minesense.data.synth import write_all
        print(write_all())

    def ingest():
        from minesense.rag.index import HybridIndex
        idx = HybridIndex(s)
        n = idx.build_from_dir(s.data_dir / "corpus")
        if n == 0:
            sys.exit(f"Tidak ada dokumen di {s.data_dir / 'corpus'} — set MINESENSE_DATA_DIR dengan benar")
        if (s.data_dir / "uploads").exists():
            n += idx.build_from_dir(s.data_dir / "uploads")
        print(f"{n} chunks → {idx.save()}")

    def train():
        from minesense.classifier.train import main as m
        m([])

    def sft():
        from minesense.finetune.build_sft_dataset import build
        build()

    if a.cmd == "synth":
        synth()
    elif a.cmd == "ingest":
        ingest()
    elif a.cmd == "train":
        train()
    elif a.cmd == "sft":
        sft()
    elif a.cmd == "bootstrap":
        synth(), ingest(), train(), sft()
    elif a.cmd == "eval":
        from minesense.eval.run_all import main as ev
        sys.exit(ev((["--update-baseline"] if a.update_baseline else []) + (["--no-judge"] if a.no_judge else [])))
    elif a.cmd == "benchmark":
        from minesense.eval.benchmark import main as bench
        bench((["--full"] if a.full else []) + (["--configs", a.configs] if a.configs else []))
    elif a.cmd == "calibrate":
        from minesense.rag.calibration import calibrate
        from minesense.rag.index import HybridIndex
        idx = HybridIndex(s)
        idx.build_from_dir(s.data_dir / "corpus")
        v1 = [json.loads(x)["question"] for x in open(s.data_dir / "eval" / "rag_qa.jsonl", encoding="utf-8")]
        print(json.dumps({idx.embedder.name: calibrate(idx, s, v1)}, indent=2))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("minesense.api.main:app", host=a.host, port=a.port, workers=a.workers)
    elif a.cmd == "ask":
        from minesense.services import Services
        r = Services(s).chat(a.question)
        print(r.answer)
        print(json.dumps([c.doc_id for c in r.citations]), r.intent, r.warnings)


if __name__ == "__main__":
    main()
