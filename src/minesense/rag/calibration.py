"""Kalibrasi ambang relevansi dense (gate out-of-scope) per embedder.

Masalah: cosine similarity berbeda skala antar model — hashing ≈0,25 untuk teks tak relevan, sedangkan
e5 bisa ≈0,75. Ambang tetap (0,45) akan meloloskan semua pertanyaan di e5 atau menolak semua di model lain.

Solusi: cari ambang yang memisahkan pertanyaan in-domain vs out-of-domain dengan balanced accuracy terbaik.
PENTING (anti-overfitting): kalibrasi memakai set terpisah dari set evaluasi v2:
  - in-domain : data/eval/rag_qa.jsonl (v1)
  - out-domain: CALIBRATION_OOS di bawah (berbeda dari OUT_OF_SCOPE di set v2)
Hasil disimpan di artifacts/calibration.json, dimuat otomatis oleh pipeline untuk embedder yang sama.
"""
from __future__ import annotations

import json
import time

import numpy as np

from minesense.config import Settings

CALIBRATION_OOS = [
    "Bagaimana cara membuat kue bolu?", "Siapa presiden Amerika Serikat pertama?", "Rekomendasi tempat wisata di Bali",
    "Bagaimana cara install Python di Windows?", "Berapa kurs dolar hari ini?", "Lirik lagu kebangsaan Indonesia",
    "Cara menanam cabai di pot", "Hasil pertandingan liga Inggris tadi malam", "Apa itu fotosintesis?",
    "Tips wawancara kerja di bank", "Bagaimana cara memperpanjang SIM di Jakarta?", "Resep es teh manis",
    "Siapa pemain sepak bola terbaik dunia?", "Cara mengatasi jerawat", "Apa ibu kota Australia?",
    "Bagaimana cara bermain catur?", "Rekomendasi laptop untuk mahasiswa", "Sejarah candi Borobudur",
]

# Hasil `minesense calibrate` untuk hashing pada set kalibrasi terpisah (2026-10-01): 0,3497 → dibulatkan 0,35.
# Nilai lama 0,45 (ditebak manual) menyebabkan 29% penolakan keliru pada set v2 — lihat docs/EVALUATION.md.
_DEFAULT_HASHING = 0.35
_DISABLED = 1.01


def resolve_dense_threshold(s: Settings, embedder_name: str) -> float:
    if s.min_dense_score is not None:
        return s.min_dense_score
    try:
        cal = json.loads(s.calibration_path.read_text(encoding="utf-8"))
        if embedder_name in cal:
            return float(cal[embedder_name]["min_dense_score"])
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return _DEFAULT_HASHING if embedder_name == "hashing" else _DISABLED


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Ambang dengan balanced accuracy tertinggi; bila seri, pilih ambang tertinggi (lebih konservatif)."""
    cands = sorted(set(np.round(pos + neg, 4)))
    best_t, best_acc = _DISABLED, 0.0
    for t in cands:
        tpr = np.mean([p >= t for p in pos])
        tnr = np.mean([n < t for n in neg])
        acc = (tpr + tnr) / 2
        if acc > best_acc or (acc == best_acc and t > best_t):
            best_t, best_acc = float(t), float(acc)
    # beri margin: taruh ambang di tengah antara skor OOS tertinggi di bawahnya dan ambang terpilih
    below = [n for n in neg if n < best_t]
    if below and best_t < _DISABLED:
        best_t = (max(below) + best_t) / 2
    return best_t, best_acc


def calibrate(index, s: Settings, in_domain: list[str]) -> dict:
    top = lambda q: (index.dense_search(q, 1) or [("", 0.0)])[0][1]  # noqa: E731
    pos = [top(q) for q in in_domain]
    neg = [top(q) for q in CALIBRATION_OOS]
    t, acc = best_threshold(pos, neg)
    # dense gate hanya berguna bila pemisahan tegas; bila tidak, matikan (andalkan BM25 + coverage)
    if acc < 0.8:
        t = _DISABLED
    entry = {"min_dense_score": round(t, 4), "balanced_accuracy": round(acc, 4),
             "in_domain_min": round(min(pos), 4), "in_domain_median": round(float(np.median(pos)), 4),
             "oos_max": round(max(neg), 4), "oos_median": round(float(np.median(neg)), 4),
             "n_pos": len(pos), "n_neg": len(neg), "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        cal = json.loads(s.calibration_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        cal = {}
    cal[index.embedder.name] = entry
    s.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    s.calibration_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return entry
