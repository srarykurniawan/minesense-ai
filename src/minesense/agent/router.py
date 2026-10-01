"""Router intent untuk endpoint chat terpadu: mengarahkan pesan ke tool yang tepat.

Intent:
  incident_report  → classifier insiden (+ jawaban SOP terkait)
  shift_report     → ekstraksi laporan produksi
  maintenance      → asisten diagnosa alat berat
  qa_sop           → RAG Q&A dokumen (default)
"""
from __future__ import annotations

import re

_FAULT = re.compile(r"\b[A-Z]-[A-Z]{2,3}-\d{3}\b")
_UNIT = re.compile(r"\b(DT|EX|BD|GR)-\d{3,5}\b", re.I)

_MAINT_WORDS = ("overheat", "bocor", "rusak", "error", "kode", "fault", "lambat", "tidak mau", "susah starter",
                "bunyi", "tekanan oli", "alarm", "breakdown", "troubleshoot", "diagnosa", "macet", "panas")
_INCIDENT_WORDS = ("terjadi", "menabrak", "terguling", "tersengat", "terjatuh", "tumpah", "korban", "cedera",
                   "longsor", "nyaris", "near miss", "insiden", "kecelakaan", "misfire ditemukan", "flyrock terlempar")
_SHIFT_WORDS = ("ob removal", "overburden", "coal getting", "bcm", "stripping ratio", "laporan produksi", " coal ",
                "batubara", " ob ", "shift ")


def route(text: str) -> str:
    low = f" {text.lower()} "
    if sum(w in low for w in _SHIFT_WORDS) >= 2 and re.search(r"\d[\d.]{3,}", text):
        return "shift_report"
    low = low.strip()
    is_question = low.endswith("?") or low.startswith(("apa", "berapa", "bagaimana", "kapan", "siapa",
                                                                  "apakah", "mengapa", "kenapa", "bolehkah"))
    if _FAULT.search(text) or (_UNIT.search(text) and any(w in low for w in _MAINT_WORDS)):
        return "maintenance"
    if not is_question and any(w in low for w in _INCIDENT_WORDS) and len(low.split()) >= 8:
        return "incident_report"
    return "qa_sop"
