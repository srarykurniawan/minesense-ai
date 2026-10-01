"""CLI: minesense <command>

  synth    — generate data sintetis
  ingest   — bangun index RAG dari data/corpus (+ data/uploads)
  train    — latih classifier insiden (baseline)
  sft      — bangun dataset fine-tuning LLM
  eval     — evaluasi menyeluruh + quality gates
  bootstrap— synth + ingest + train + sft (setup awal)
  serve    — jalankan API
  ask      — tanya cepat dari terminal
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="minesense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("synth", "ingest", "train", "sft", "eval", "bootstrap"):
        sub.add_parser(c)
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
        sys.exit(ev([]))
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
