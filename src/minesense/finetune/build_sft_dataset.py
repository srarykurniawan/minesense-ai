"""Bangun dataset instruction-tuning (format chat messages JSONL) untuk fine-tuning LLM dengan LoRA.

Tugas yang diajarkan ke model:
  1. grounded_qa   : menjawab dari KONTEKS dengan sitasi [n] (dari RAG_QA + retrieval nyata)
  2. refusal       : menolak bila konteks tidak memuat jawaban (mencegah halusinasi)
  3. extraction    : laporan shift → JSON terstruktur
  4. incident      : laporan insiden → JSON {kategori, risiko, tindakan}
  5. maintenance   : gejala + kode gangguan → diagnosa terstruktur

Output: data/sft/train.jsonl, data/sft/val.jsonl  (format {"messages":[{role,content},...], "task": ...})
Jalankan: python -m minesense.finetune.build_sft_dataset
"""
from __future__ import annotations

import csv
import json
import random

from minesense.classifier.model import RECOMMENDED_ACTIONS
from minesense.config import get_settings
from minesense.llm.prompts import EXTRACT_SYSTEM, MAINT_SYSTEM, RAG_SYSTEM, RAG_USER
from minesense.rag.index import HybridIndex
from minesense.rag.pipeline import format_context
from minesense.rag.retriever import HybridRetriever
from minesense.rag.text import split_sentences

INCIDENT_SYSTEM = ("Klasifikasikan laporan insiden tambang. Keluarkan JSON: "
                   '{"kategori": ..., "tingkat_risiko": "rendah|sedang|tinggi|kritis", "tindakan_segera": [...]}')

PARAPHRASE_PREFIX = ["", "Mohon info, ", "Tolong jelaskan: ", "Saya pengawas baru. ", "Pertanyaan cepat: "]


def _msg(system: str, user: str, assistant: str, task: str) -> dict:
    return {"task": task, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                       {"role": "assistant", "content": assistant}]}


def build(seed: int = 7) -> dict[str, int]:
    s = get_settings()
    rng = random.Random(seed)
    idx = HybridIndex(s)
    idx.build_from_dir(s.data_dir / "corpus")
    ret = HybridRetriever(idx, s)
    ex: list[dict] = []

    # 1 & 2) grounded QA + refusal
    qa = [json.loads(line) for line in open(s.data_dir / "eval" / "rag_qa.jsonl", encoding="utf-8")]
    for item in qa:
        for prefix in rng.sample(PARAPHRASE_PREFIX, 2):
            q = prefix + item["question"]
            items = ret.retrieve(q, 4)
            # cari kalimat yang memuat kata kunci jawaban → jawaban emas berbasis teks sumber
            ans_lines = []
            for n, it in enumerate(items, 1):
                for sent in split_sentences(it.chunk.text):
                    if any(k.lower() in sent.lower() for k in item["keywords"]) and len(ans_lines) < 3:
                        ans_lines.append(f"- {sent.rstrip('.')}. [{n}]")
            if not ans_lines:
                continue
            answer = "\n".join(ans_lines)
            ex.append(_msg(RAG_SYSTEM, RAG_USER.format(context=format_context(items), question=q), answer, "grounded_qa"))
            # refusal: konteks dari dokumen yang SALAH
            wrong = [it for it in ret.retrieve(rng.choice(qa)["question"], 6) if it.chunk.doc_id != item["doc_id"]][:3]
            if wrong:
                ex.append(_msg(RAG_SYSTEM, RAG_USER.format(context=format_context(wrong), question=q),
                               "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia. "
                               "Silakan konfirmasi ke pengawas area, tim K3, atau KTT.", "refusal"))

    # 3) extraction
    for line in open(s.data_dir / "shift_reports.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ex.append(_msg(EXTRACT_SYSTEM, r["text"], json.dumps(r["gold"], ensure_ascii=False), "extraction"))

    # 4) incident
    with open(s.data_dir / "incidents.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rng.sample(rows, min(500, len(rows))):
        out = {"kategori": r["category"], "tingkat_risiko": r["risk_level"],
               "tindakan_segera": RECOMMENDED_ACTIONS[r["category"]][:2]}
        ex.append(_msg(INCIDENT_SYSTEM, r["description"], json.dumps(out, ensure_ascii=False), "incident"))

    # 5) maintenance
    faults = json.load(open(s.data_dir / "fault_codes.json", encoding="utf-8"))
    with open(s.data_dir / "work_orders.csv", encoding="utf-8") as f:
        wos = list(csv.DictReader(f))
    for wo in rng.sample(wos, min(300, len(wos))):
        fc = faults.get(wo["fault_code"]) or next(v for v in faults.values() if v["model"] == wo["model"]
                                                  and wo["root_cause"] in v["causes"])
        user = f"Unit {wo['unit']} ({wo['model']}), gejala: {wo['symptom']}. Kode: {wo['fault_code'] or 'tidak tercatat'}."
        ans = (f"Kode terkait: {fc['code']} — {fc['title']}.\n1) Kemungkinan penyebab:\n"
               + "\n".join(f"   - {c}" for c in fc["causes"])
               + "\n2) Langkah pemeriksaan & perbaikan:\n" + "\n".join(f"   - {a}" for a in fc["actions"])
               + "\n3) Keselamatan: terapkan LOTO (SOP-K3-002) sebelum pekerjaan perawatan."
               + ("\n⚠️ Item kritis: unit tidak boleh beroperasi sampai diperbaiki." if fc.get("critical") else ""))
        ex.append(_msg(MAINT_SYSTEM, user, ans, "maintenance"))

    rng.shuffle(ex)
    n_val = max(20, int(0.08 * len(ex)))
    out = s.data_dir / "sft"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", ex[:n_val]), ("train", ex[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats: dict[str, int] = {}
    for e in ex:
        stats[e["task"]] = stats.get(e["task"], 0) + 1
    stats["train"], stats["val"] = len(ex) - n_val, n_val
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    build()
