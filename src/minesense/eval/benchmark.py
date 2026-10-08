"""Benchmark embedder × mode retrieval × reranker pada set evaluasi v2 → artifacts/reports/benchmark.{json,md}

    minesense benchmark            # hashing (2 varian) + MiniLM multilingual + MiniLM+reranker
    minesense benchmark --full     # + mpnet multilingual + multilingual-e5-large (unduh ±3,5 GB)
    minesense benchmark --configs hashing,minilm

Butuh `pip install fastembed` dan akses internet ke HuggingFace untuk konfigurasi neural (model di-cache di
artifacts/model_cache). Konfigurasi yang gagal dimuat dilewati dengan alasan tercatat — tidak menggagalkan run.
Gate relevansi tiap embedder dikalibrasi dulu pada set TERPISAH (v1 + OOS kalibrasi), bukan pada set v2.
"""
from __future__ import annotations

import argparse
import json
import time

from minesense.config import get_settings
from minesense.eval.rag_eval import evaluate, failures_markdown, headline, load_set
from minesense.logging_utils import get_logger
from minesense.rag.calibration import calibrate
from minesense.rag.index import HybridIndex
from minesense.rag.pipeline import RAGPipeline

log = get_logger(__name__)

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
E5L = "intfloat/multilingual-e5-large"
JINA = "jinaai/jina-reranker-v2-base-multilingual"

CONFIGS = {
    "hashing": {"label": "Hashing n-gram (ambang lama manual 0,45)", "embedding_backend": "hashing",
                "min_dense_score": 0.45, "calibrate": False},
    "hashing-cal": {"label": "Hashing n-gram + ambang terkalibrasi (default)", "embedding_backend": "hashing"},
    "minilm": {"label": "MiniLM-L12 multilingual (0,22 GB)", "embedding_backend": "fastembed", "embedding_model": MINILM},
    "minilm-rerank": {"label": "MiniLM-L12 + reranker Jina v2 multilingual", "embedding_backend": "fastembed",
                      "embedding_model": MINILM, "reranker": "fastembed", "reranker_model": JINA},
    "mpnet": {"label": "MPNet multilingual (1,0 GB)", "embedding_backend": "fastembed", "embedding_model": MPNET,
              "full": True},
    "e5-large": {"label": "multilingual-e5-large (2,2 GB)", "embedding_backend": "fastembed", "embedding_model": E5L,
                 "full": True},
    "e5-large-rerank": {"label": "e5-large + reranker Jina v2", "embedding_backend": "fastembed",
                        "embedding_model": E5L, "reranker": "fastembed", "reranker_model": JINA, "full": True},
}
_SETTING_KEYS = ("embedding_backend", "embedding_model", "reranker", "reranker_model", "min_dense_score")


def run_config(key: str, cfg: dict, items: list[dict], v1_questions: list[str]) -> dict:
    base = get_settings()
    s = base.model_copy(update={k: cfg[k] for k in _SETTING_KEYS if k in cfg} | {"vector_backend": "memory"})
    row = {"config": key, "label": cfg["label"]}
    t0 = time.perf_counter()
    idx = HybridIndex(s)
    idx.build_from_dir(s.data_dir / "corpus")
    row["index_build_s"] = round(time.perf_counter() - t0, 2)
    if cfg.get("calibrate", True):
        row["calibration"] = calibrate(idx, s, v1_questions)
    rag = RAGPipeline(idx, settings=s)
    row["dense_threshold"] = rag.dense_threshold
    row["modes"] = {}
    for mode in ("dense", "sparse", "hybrid"):
        if mode == "dense" and key.endswith("rerank"):
            continue
        rep = evaluate(rag, items, mode=mode, answers=(mode == "hybrid"))
        row["modes"][mode] = headline(rep)
        if mode == "hybrid":
            row["hybrid_report"] = rep
    return row


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="minesense benchmark")
    ap.add_argument("--full", action="store_true", help="sertakan model besar (mpnet, e5-large)")
    ap.add_argument("--configs", default="", help="daftar kunci dipisah koma: " + ",".join(CONFIGS))
    a = ap.parse_args(argv)
    s = get_settings()
    items = load_set(s.data_dir / "eval" / "rag_qa_v2.jsonl")
    v1 = [json.loads(line)["question"] for line in open(s.data_dir / "eval" / "rag_qa.jsonl", encoding="utf-8")]
    keys = [k.strip() for k in a.configs.split(",") if k.strip()] or \
        [k for k, c in CONFIGS.items() if a.full or not c.get("full")]

    rows, skipped = [], []
    for k in keys:
        log.info(f"benchmark: {k}")
        try:
            rows.append(run_config(k, CONFIGS[k], items, v1))
        except Exception as e:  # model tidak bisa diunduh / paket belum terpasang
            skipped.append({"config": k, "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            log.warning(f"lewati {k}: {e}")

    out = s.artifacts_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "hybrid_report"} for r in rows]
    (out / "benchmark.json").write_text(json.dumps({"rows": slim, "skipped": skipped,
                                                    "llm": s.llm_provider, "n_items": len(items)},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
    md = _markdown(rows, skipped, s.llm_provider, len(items))
    (out / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:40]))
    return {"rows": slim, "skipped": skipped}


def _f(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def _markdown(rows, skipped, llm, n) -> list[str]:
    md = ["# Benchmark Retrieval & RAG — set evaluasi v2", "",
          f"_Dibuat {time.strftime('%Y-%m-%d %H:%M')} · {n} pertanyaan (parafrase, kolokial, skenario, "
          f"tak-terjawab, di luar topik) · LLM: `{llm}`_", "",
          "## Retrieval (pertanyaan yang jawabannya ada di dokumen)", "",
          "| Konfigurasi | Mode | Hit@1 | Hit@5 | MRR | Evidence@5 |", "|---|---|---|---|---|---|"]
    for r in rows:
        for mode, h in r["modes"].items():
            md.append(f"| {r['label']} | {mode} | {_f(h['hit@1'])} | {_f(h['hit@5'])} | {_f(h['mrr'])} | "
                      f"{_f(h['evidence@5'])} |")
    md += ["", "## End-to-end (mode hybrid, termasuk gate relevansi & LLM)", "",
           "| Konfigurasi | Jawaban benar | Penolakan keliru | Menolak yg tak terjawab/OOS | Ambang dense | "
           "Build index (dtk) | p50 latensi (ms) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rep = r["hybrid_report"]
        a, ref = rep["answerable"], rep["refusal"]
        ci = lambda d, m: f"{d[m]['mean']:.2f} [{d[m]['ci95'][0]:.2f}–{d[m]['ci95'][1]:.2f}]"  # noqa: E731
        md.append(f"| {r['label']} | {ci(a, 'answer_correct')} | {ci(a, 'false_refusal')} | "
                  f"{ci(ref, 'refusal_rate')} | {r['dense_threshold']:.2f} | {r['index_build_s']} | "
                  f"{rep.get('latency_ms', {}).get('p50', '-')} |")
    if rows:
        best_ret = max(rows, key=lambda r: (r["modes"]["hybrid"]["evidence@5"] or 0, r["modes"]["hybrid"]["mrr"] or 0))
        e2e = lambda r: (r["modes"]["hybrid"]["answer_correct"] or 0) + 0.5 * (r["modes"]["hybrid"]["refusal_rate"] or 0)  # noqa: E731
        best = max(rows, key=e2e)
        md += ["", "## Rekomendasi", "",
               f"- Retrieval terbaik (Evidence@5 hybrid): **{best_ret['label']}**",
               f"- End-to-end terbaik (jawaban benar + 0,5 × penolakan tepat): **{best['label']}**", "",
               "Angka dalam [ ] adalah interval kepercayaan 95% (bootstrap). Bila interval dua konfigurasi banyak "
               "tumpang-tindih, perbedaannya belum meyakinkan — perbesar set evaluasi sebelum memutuskan. "
               "Metrik penolakan sangat dipengaruhi LLM: mode `extractive` tidak bisa menolak pertanyaan yang "
               "jawabannya tidak ada; jalankan ulang dengan LLM aktif untuk angka penolakan yang representatif.",
               "", f"### 15 kegagalan pertama — {best['label']}", ""] + failures_markdown(best["hybrid_report"])
    if skipped:
        md += ["", "## Konfigurasi dilewati", ""] + [f"- `{x['config']}`: {x['reason']}" for x in skipped]
    return md


if __name__ == "__main__":
    main()
