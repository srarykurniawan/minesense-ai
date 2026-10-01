"""Ekstraksi laporan produksi shift (teks bebas / WhatsApp / tabel) → JSON terstruktur.

Strategi hybrid:
  1. Parser berbasis aturan (cepat, deterministik, tanpa biaya).
  2. LLM (opsional) dengan JSON mode → divalidasi Pydantic; field kosong diisi dari hasil aturan.
  3. Validasi konsistensi bisnis (SR ≈ OB/coal, rentang wajar).
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from minesense.llm.prompts import EXTRACT_SYSTEM
from minesense.llm.providers import LLM, parse_json
from minesense.schemas import ExtractResponse, ShiftReport

_NUM = r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?|\d+\.\d+)"
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5, "jun": 6, "jul": 7, "agu": 8, "agt": 8,
           "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12}


def parse_id_number(s: str) -> float:
    """'25.000' → 25000; '5,25' → 5.25; '1.234,5' → 1234.5; '3.5' (desimal gaya EN) → 3.5"""
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        return float(s.replace(".", "").replace(",", "."))
    if "," in s:
        return float(s.replace(",", "."))
    return float(s)


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.I)
    return m.group(1) if m else None


def _find_num(patterns: list[str], text: str) -> float | None:
    for p in patterns:
        v = _find(p, text)
        if v is not None:
            return parse_id_number(v)
    return None


def parse_rules(text: str) -> ShiftReport:
    t = text.replace(" ", " ")
    data: dict = {}
    # tanggal: 15/04/2026, 15-04-2026, 2026-04-15, 15 April 2026
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        data["tanggal"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    elif m := re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t):
        data["tanggal"] = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    elif m := re.search(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})\b", t):
        mon = _MONTHS.get(m.group(2).lower()[:3])
        if mon:
            data["tanggal"] = f"{m.group(3)}-{mon:02d}-{int(m.group(1)):02d}"
    if m := re.search(r"shift\s*:?\s*(pagi|malam|siang|1|2|i{1,2})\b", t, re.I):
        v = m.group(1).lower()
        data["shift"] = {"1": "pagi", "i": "pagi", "2": "malam", "ii": "malam"}.get(v, v)
    if m := re.search(r"\b(Pit\s+[A-Z][a-zA-Z0-9]*)", t):
        data["pit"] = m.group(1)
    data["ob_removal_bcm"] = _find_num([rf"(?:OB(?:\s*Removal)?|Overburden)[^0-9\n]{{0,25}}?{_NUM}"], t)
    data["coal_getting_ton"] = _find_num([rf"(?:Coal(?:\s*Getting)?|Batubara)[^0-9\n]{{0,25}}?{_NUM}"], t)
    data["stripping_ratio"] = _find_num([rf"(?:Stripping\s*Ratio|\bSR)\b[^0-9\n]{{0,10}}?{_NUM}"], t)
    data["hujan_jam"] = _find_num([rf"(?:Rain|Hujan|licin)[^0-9\n]{{0,30}}?{_NUM}\s*(?:jam|\|)"], t)
    data["jumlah_insiden"] = (int(v) if (v := _find_num([rf"(?:Insiden|Jumlah insiden)[^0-9\n]{{0,25}}?{_NUM}"], t))
                              is not None else None)
    bd_line = _find(r"(?:Unit\s*Breakdown|Unit\s*BD|Breakdown)\s*[:|]\s*([^\n|]*?)(?:\.\s|\||\n|$)", t)
    if bd_line is not None:
        data["unit_breakdown"] = sorted(set(re.findall(r"\b[A-Z]{2}-\d{3,5}\b", bd_line)))
    data["catatan"] = (_find(r"(?:Catatan|Keterangan)\s*:\s*(.+)", t) or "").strip() or None
    if data.get("catatan") is None:
        # gaya pesan bebas: kalimat setelah informasi insiden
        m = re.search(r"kejadian\.\s*(.+?)(?:\s*Terima kasih\.?)?$", t, re.S)
        data["catatan"] = m.group(1).strip() if m else None
    return ShiftReport(**{k: v for k, v in data.items() if v is not None})


def consistency_warnings(r: ShiftReport) -> list[str]:
    w = []
    if r.ob_removal_bcm and r.coal_getting_ton and r.stripping_ratio:
        sr = r.ob_removal_bcm / r.coal_getting_ton
        if abs(sr - r.stripping_ratio) > 0.05 * max(sr, 1):
            w.append(f"Stripping ratio dilaporkan {r.stripping_ratio} tidak sesuai OB/coal = {sr:.2f}")
    if r.hujan_jam is not None and not 0 <= r.hujan_jam <= 12:
        w.append("Jam hujan di luar rentang shift 0–12 jam")
    if r.ob_removal_bcm is not None and r.ob_removal_bcm > 500_000:
        w.append("Volume OB tidak wajar untuk satu shift (> 500.000 BCM)")
    return w


REQUIRED = ["tanggal", "shift", "pit", "ob_removal_bcm", "coal_getting_ton", "stripping_ratio", "hujan_jam",
            "jumlah_insiden"]


def extract(text: str, llm: LLM | None = None, use_llm: bool = False) -> tuple[ExtractResponse, list[str]]:
    rules = parse_rules(text)
    result, method = rules, "rules"
    if use_llm and llm is not None and llm.name != "extractive":
        try:
            raw = parse_json(llm.generate(EXTRACT_SYSTEM, text, json_mode=True, temperature=0))
            llm_rep = ShiftReport(**{k: v for k, v in raw.items() if v is not None and k in ShiftReport.model_fields})
            merged = {**rules.model_dump(exclude_none=True), **llm_rep.model_dump(exclude_none=True)}
            if not llm_rep.unit_breakdown and rules.unit_breakdown:
                merged["unit_breakdown"] = rules.unit_breakdown
            result, method = ShiftReport(**merged), "llm+rules"
        except (ValueError, ValidationError) as e:
            return ExtractResponse(data=rules, method="rules", missing_fields=_missing(rules)), [f"LLM gagal: {e}"]
    if result.stripping_ratio is None and result.ob_removal_bcm and result.coal_getting_ton:
        result.stripping_ratio = round(result.ob_removal_bcm / result.coal_getting_ton, 2)
    return ExtractResponse(data=result, method=method, missing_fields=_missing(result)), consistency_warnings(result)


def _missing(r: ShiftReport) -> list[str]:
    return [f for f in REQUIRED if getattr(r, f) is None]
