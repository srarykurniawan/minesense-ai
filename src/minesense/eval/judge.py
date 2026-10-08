"""LLM-as-judge: menilai jawaban RAG secara semantik (melengkapi metrik kata kunci yang kaku).

Dimensi (skala 0–1):
  faithfulness : proporsi klaim dalam jawaban yang DIDUKUNG konteks (anti-halusinasi)
  relevance    : seberapa langsung jawaban menjawab pertanyaan
  correctness  : apakah jawaban memuat fakta acuan (reference) — 1 benar, 0.5 sebagian, 0 salah/tidak ada
Konfigurasi juri terpisah dari LLM aplikasi (disarankan model lebih kuat / berbeda untuk mengurangi bias):
  MINESENSE_JUDGE_PROVIDER, MINESENSE_JUDGE_MODEL, MINESENSE_JUDGE_API_KEY, MINESENSE_JUDGE_BASE_URL
Keterbatasan: juri LLM bisa bias/inkonsisten — validasi manual ±20 sampel sebelum memercayai angkanya.
"""
from __future__ import annotations

import json

from minesense.config import Settings
from minesense.llm.providers import LLM, LLMError, build_llm, parse_json

JUDGE_SYSTEM = """Anda adalah evaluator ketat untuk sistem tanya-jawab keselamatan tambang.
Nilai JAWABAN terhadap PERTANYAAN, KONTEKS (dokumen yang diberikan ke sistem), dan FAKTA ACUAN.
Keluarkan HANYA JSON:
{"faithfulness": 0-1, "relevance": 0-1, "correctness": 0|0.5|1, "unsupported_claims": ["..."], "reason": "..."}
- faithfulness: bagi jawaban menjadi klaim; skor = klaim yang didukung KONTEKS / total klaim. Penolakan
  ("tidak ditemukan") dihitung faithfulness 1.
- relevance: 1 bila langsung menjawab inti pertanyaan, 0 bila tidak nyambung.
- correctness: 1 bila jawaban memuat semua FAKTA ACUAN (boleh beda redaksi/format angka), 0.5 sebagian, 0 tidak.
  Bila FAKTA ACUAN kosong (pertanyaan tak terjawab dari dokumen): 1 jika jawaban menolak/menyatakan tidak
  ditemukan, 0 jika mengarang jawaban.
Jangan memberi nilai berdasarkan gaya bahasa."""


def build_judge(s: Settings) -> LLM | None:
    provider = s.judge_provider or s.llm_provider
    if provider == "extractive":
        return None
    js = s.model_copy(update={
        "llm_provider": provider, "llm_model": s.judge_model or s.llm_model,
        "llm_api_key": s.judge_api_key or s.llm_api_key, "llm_base_url": s.judge_base_url or s.llm_base_url,
        "llm_temperature": 0.0})
    return build_llm(js)


def judge_one(judge: LLM, question: str, contexts: list[str], answer: str, reference: list[str]) -> dict:
    user = (f"PERTANYAAN: {question}\n\nKONTEKS:\n" + "\n---\n".join(contexts[:5])
            + f"\n\nFAKTA ACUAN: {json.dumps(reference, ensure_ascii=False)}\n\nJAWABAN:\n{answer}")
    try:
        out = parse_json(judge.generate(JUDGE_SYSTEM, user, json_mode=True, temperature=0, max_tokens=400))
        return {"faithfulness": float(out.get("faithfulness", 0)), "relevance": float(out.get("relevance", 0)),
                "correctness": float(out.get("correctness", 0)),
                "unsupported_claims": out.get("unsupported_claims", [])[:5], "reason": str(out.get("reason", ""))[:300]}
    except (LLMError, ValueError, TypeError) as e:
        return {"error": str(e)[:200]}
