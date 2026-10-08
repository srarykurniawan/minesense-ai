"""Evaluasi RAG v2 yang jujur: set parafrase/kolokial/skenario/tak-terjawab, CI bootstrap, analisis kegagalan.

Metrik (per tipe pertanyaan + keseluruhan):
  Retrieval (pertanyaan terjawab):  hit@1, hit@5, mrr, evidence@5 (fakta acuan ada di 5 chunk teratas)
  Jawaban   (pertanyaan terjawab):  answer_correct (semua fakta acuan muncul di jawaban),
                                     cited_correct_doc, false_refusal (menolak padahal jawabannya ada)
  Penolakan (tak terjawab / OOS):   refusal_rate (harus mendekati 1; sisanya = potensi halusinasi)
  Opsional: LLM-as-judge (faithfulness, relevance, correctness) bila provider juri aktif.
Setiap metrik disertai interval kepercayaan 95% (bootstrap) — dengan n≈20/tipe, selisih <10 poin
seringkali BUKAN perbedaan nyata.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from minesense.rag.pipeline import RAGPipeline

ANSWERABLE = ("paraphrase", "colloquial", "scenario")
REFUSE_TYPES = ("unanswerable", "out_of_scope")


def norm(text: str) -> str:
    t = text.lower().replace("**", "").replace("–", "-").replace("—", "-")
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)
    return re.sub(r"\s+", "", t)


def contains_all(text: str, keywords: list[str]) -> bool:
    nt = norm(text)
    return all(norm(k) in nt for k in keywords)


def is_refusal(resp) -> bool:
    return resp.intent in ("out_of_scope", "blocked") or "tidak ditemukan" in resp.answer.lower()


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (round(sum(values) / n, 4), round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4))


def load_set(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def evaluate(rag: RAGPipeline, items: list[dict], *, mode: str = "hybrid", answers: bool = True,
             judge=None, judge_max: int = 60, k: int = 5) -> dict:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures, latencies, judged = [], [], 0
    for it in items:
        t, q = it["type"], it["question"]
        rec = per[t]
        if t in ANSWERABLE:
            ret = rag.retriever.retrieve(q, k, mode=mode)
            docs = [r.chunk.doc_id for r in ret]
            rec["hit@1"].append(float(docs[:1] == [it["doc_id"]]))
            rec["hit@5"].append(float(it["doc_id"] in docs))
            rec["mrr"].append(next((1 / (i + 1) for i, d in enumerate(docs) if d == it["doc_id"]), 0.0))
            rec["evidence@5"].append(float(contains_all(" ".join(r.chunk.text for r in ret), it["keywords"])))
        if not answers:
            continue
        t0 = time.perf_counter()
        resp = rag.answer(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        refused = is_refusal(resp)
        fail_reason = None
        if t in ANSWERABLE:
            ok = (not refused) and contains_all(resp.answer, it["keywords"])
            rec["answer_correct"].append(float(ok))
            rec["false_refusal"].append(float(refused))
            rec["cited_correct_doc"].append(float(any(c.doc_id == it["doc_id"] for c in resp.citations)))
            if not ok:
                fail_reason = "ditolak (false refusal)" if refused else "fakta acuan tidak ada di jawaban"
        else:
            rec["refusal_rate"].append(float(refused))
            if not refused:
                fail_reason = "tidak menolak (potensi halusinasi)"
        if judge is not None and judged < judge_max and t != "out_of_scope":
            from minesense.eval.judge import judge_one
            ctx = [c.snippet for c in resp.citations] or [r.chunk.text for r in rag.retriever.retrieve(q, 3)]
            j = judge_one(judge, q, ctx, resp.answer, it["keywords"])
            judged += 1
            for key in ("faithfulness", "relevance", "correctness"):
                if key in j:
                    rec[f"judge_{key}"].append(j[key])
        if fail_reason:
            failures.append({"type": t, "question": q, "expected_doc": it["doc_id"], "expected": it["keywords"],
                             "reason": fail_reason, "intent": resp.intent,
                             "cited": [c.doc_id for c in resp.citations][:3], "answer": resp.answer[:240]})

    def summarize(groups: list[str]) -> dict:
        merged: dict[str, list[float]] = defaultdict(list)
        for g in groups:
            for m, v in per[g].items():
                merged[m] += v
        return {m: {"mean": ci[0], "ci95": [ci[1], ci[2]], "n": len(v)}
                for m, v in sorted(merged.items()) if (ci := bootstrap_ci(v))}

    report = {
        "by_type": {t: summarize([t]) for t in per},
        "answerable": summarize([t for t in ANSWERABLE if t in per]),
        "refusal": summarize([t for t in REFUSE_TYPES if t in per]),
        "failures": failures,
        "mode": mode, "embedder": rag.index.embedder.name,
        "reranker": rag.s.reranker if rag.retriever.reranker else "none", "llm": rag.llm.name,
        "judge": getattr(judge, "name", None),
    }
    if latencies:
        lat = sorted(latencies)
        report["latency_ms"] = {"p50": round(statistics.median(lat), 1), "p95": round(lat[int(0.95 * (len(lat) - 1))], 1)}
    return report


def headline(rep: dict) -> dict:
    """Ringkasan satu baris untuk tabel perbandingan & regression gate."""
    a, r = rep["answerable"], rep["refusal"]
    g = lambda d, k: d.get(k, {}).get("mean")  # noqa: E731
    return {"hit@1": g(a, "hit@1"), "hit@5": g(a, "hit@5"), "mrr": g(a, "mrr"), "evidence@5": g(a, "evidence@5"),
            "answer_correct": g(a, "answer_correct"), "false_refusal": g(a, "false_refusal"),
            "refusal_rate": g(r, "refusal_rate"), "judge_faithfulness": g(a, "judge_faithfulness"),
            "judge_correctness": g(a, "judge_correctness")}


def failures_markdown(rep: dict, limit: int = 15) -> list[str]:
    lines = ["| Tipe | Pertanyaan | Harapan | Masalah | Sumber disitasi |", "|---|---|---|---|---|"]
    for f in rep["failures"][:limit]:
        exp = f"{f['expected_doc'] or '-'}: {', '.join(f['expected'])}" if f["expected"] else "menolak"
        lines.append(f"| {f['type']} | {f['question'][:80]} | {exp} | {f['reason']} | {', '.join(f['cited']) or '-'} |")
    return lines
