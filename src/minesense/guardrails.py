"""Guardrails input & output: PII redaction, prompt-injection, grounding, peringatan safety-critical."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from minesense.rag.text import split_sentences, tokenize

_INJECTION = [
    r"abaikan (semua )?(instruksi|perintah|aturan)", r"ignore (all |previous |the )*(instructions|rules)",
    r"system prompt", r"you are now", r"kamu sekarang adalah", r"jailbreak", r"developer mode",
    r"tampilkan (prompt|instruksi) (sistem|awal)",
]
_PII = {
    "NIK": re.compile(r"\b\d{16}\b"),
    "TELEPON": re.compile(r"(?:\+62|\b0)8\d{2}[-\s]?\d{3,4}[-\s]?\d{3,5}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}
_SAFETY_TERMS = {"peledakan", "misfire", "flyrock", "loto", "isolasi", "ketinggian", "longsor", "lereng", "listrik",
                 "tersengat", "rem", "terguling", "petir", "darurat", "kebakaran", "evakuasi", "b3", "harness"}


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    warnings: list[str] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, rx in _PII.items():
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return text, found


def check_input(text: str) -> GuardResult:
    res = GuardResult(text=text)
    low = text.lower()
    if any(re.search(p, low) for p in _INJECTION):
        res.blocked = True
        res.warnings.append("Terdeteksi upaya prompt injection; permintaan ditolak.")
        return res
    res.text, pii = redact_pii(text)
    if pii:
        res.warnings.append(f"Data pribadi disamarkan: {', '.join(pii)}")
    return res


def is_safety_critical(text: str) -> bool:
    return bool(_SAFETY_TERMS & set(tokenize(text)))


def grounding_score(answer: str, contexts: list[str]) -> float:
    """Proporsi kalimat jawaban yang token kontennya ≥50% muncul di konteks (proxy faithfulness murah)."""
    ctx = set(tokenize(" ".join(contexts)))
    sents = [s for s in split_sentences(answer) if len(tokenize(s)) >= 3]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        toks = set(tokenize(s))
        if toks and len(toks & ctx) / len(toks) >= 0.5:
            ok += 1
    return ok / len(sents)


def validate_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    invalid = [c for c in cited if c < 1 or c > n_sources]
    return [c for c in cited if c not in invalid], invalid


SAFETY_FOOTER = ("⚠️ Catatan keselamatan: jawaban ini adalah ringkasan dokumen. Selalu ikuti SOP resmi, izin kerja, "
                 "dan instruksi pengawas/KTT di lapangan.")
