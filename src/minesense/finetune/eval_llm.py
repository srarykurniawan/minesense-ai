"""Evaluasi LLM (base vs fine-tuned) pada data/sft/val.jsonl melalui provider apa pun (Ollama/vLLM/API).

    MINESENSE_LLM_PROVIDER=ollama MINESENSE_LLM_MODEL=qwen2.5:1.5b-instruct python -m minesense.finetune.eval_llm
    MINESENSE_LLM_PROVIDER=ollama MINESENSE_LLM_MODEL=minesense             python -m minesense.finetune.eval_llm

Metrik per tugas:
  extraction  : json_valid, field_accuracy
  incident    : json_valid, kategori_acc, risiko_acc
  grounded_qa : citation_rate, keyword_overlap (token jawaban emas yang muncul)
  refusal     : refusal_acc (model menolak saat konteks salah)
  maintenance : code_acc (kode gangguan yang benar disebut)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from minesense.config import get_settings
from minesense.llm.providers import build_llm, parse_json
from minesense.rag.text import tokenize


def score(task: str, gold: str, pred: str) -> dict[str, float]:
    if task in ("extraction", "incident"):
        try:
            p = parse_json(pred)
        except Exception:
            return {"json_valid": 0.0}
        g = json.loads(gold)
        if task == "incident":
            return {"json_valid": 1.0, "kategori_acc": float(p.get("kategori") == g["kategori"]),
                    "risiko_acc": float(p.get("tingkat_risiko") == g["tingkat_risiko"])}
        keys = [k for k in g if k != "catatan"]
        ok = sum(str(p.get(k)).lower() == str(g[k]).lower() or _num_eq(p.get(k), g[k]) for k in keys)
        return {"json_valid": 1.0, "field_accuracy": ok / len(keys)}
    if task == "refusal":
        return {"refusal_acc": float("tidak ditemukan" in pred.lower())}
    if task == "grounded_qa":
        gt = set(tokenize(gold.replace("[", " ").replace("]", " ")))
        return {"citation_rate": float(bool(re.search(r"\[\d+\]", pred))),
                "keyword_overlap": len(gt & set(tokenize(pred))) / max(1, len(gt)),
                "false_refusal": float("tidak ditemukan" in pred.lower())}
    if task == "maintenance":
        code = re.search(r"[A-Z]-[A-Z]{2,3}-\d{3}", gold)
        return {"code_acc": float(bool(code) and code.group(0) in pred)}
    return {}


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    s = get_settings()
    llm = build_llm(s)
    rows = [json.loads(line) for line in open(s.data_dir / "sft" / "val.jsonl", encoding="utf-8")]
    rows = rows[: a.limit] if a.limit else rows
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sys_m, user_m, gold_m = r["messages"]
        pred = llm.generate(sys_m["content"], user_m["content"], json_mode=r["task"] in ("extraction", "incident"))
        for k, v in score(r["task"], gold_m["content"], pred).items():
            agg[r["task"]][k].append(v)
    report = {t: {k: round(sum(v) / len(v), 4) for k, v in m.items()} | {"n": len(next(iter(m.values())))}
              for t, m in agg.items()}
    report["_model"] = f"{s.llm_provider}:{s.llm_model}"
    out = s.artifacts_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"llm_eval_{s.llm_provider}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
