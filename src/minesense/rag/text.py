"""Utilitas teks Bahasa Indonesia: tokenisasi & stopword ringan."""
from __future__ import annotations

import re

STOPWORDS = set("""
yang dan di ke dari untuk pada dengan dalam adalah ini itu atau oleh sebagai juga akan dapat bila jika saat
apa apakah berapa bagaimana kapan siapa mengapa kenapa harus wajib boleh tidak ada sudah belum karena agar
serta para tersebut setiap seluruh hanya lebih kurang the a of to is are be minimal minimum maksimum
gimana gmn seberapa yg tolong mohon saya kami kita aku nya sih dong ya lagi cara membuat bisa mau benar
info jelaskan pertanyaan cepat
""".split())

_PREFIXES = ("meng", "meny", "mem", "men", "me", "peng", "pem", "pen", "per", "pe", "ber", "ter", "di", "ke")
_SUFFIXES = ("kan", "nya", "an", "i")


def light_stem(tok: str) -> str:
    """Stemmer ringan Bahasa Indonesia (cukup untuk pencocokan coverage, bukan linguistik penuh)."""
    if len(tok) <= 5 or not tok.isalpha():
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            break
    for pre in _PREFIXES:
        if tok.startswith(pre) and len(tok) - len(pre) >= 4:
            tok = tok[len(pre):]
            break
    return tok

_TOKEN = re.compile(r"[a-z0-9]+(?:[-–/,.][a-z0-9]+)*", re.I)


def normalize(text: str) -> str:
    return text.lower().replace("–", "-")


def tokenize(text: str, remove_stop: bool = True) -> list[str]:
    toks = _TOKEN.findall(normalize(text))
    out = []
    for t in toks:
        t = t.strip(".,")
        if not t or (remove_stop and t in STOPWORDS):
            continue
        out.append(t)
        # pecah token gabungan (E-ENG-101 -> e, eng, 101) agar match parsial
        if "-" in t or "/" in t:
            out.extend(p for p in re.split(r"[-/]", t) if p and p not in STOPWORDS)
    return out


def split_sentences(text: str) -> list[str]:
    text = text.replace("**", "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    sep = re.compile(r"[|\-: ]+")
    for i, p in enumerate(parts):
        p = p.strip(" -*#>")
        nxt = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if sep.fullmatch(p) or len(p.strip(" |")) <= 3 or (p.startswith("|") and nxt and sep.fullmatch(nxt)):
            continue  # baris pemisah / header tabel markdown
        if p.startswith("|"):
            p = " — ".join(c.strip() for c in p.strip("|").split("|") if c.strip())
        out.append(p)
    return out
