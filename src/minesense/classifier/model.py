"""Klasifikasi laporan insiden: kategori hazard + tingkat risiko.

Backend:
  - sklearn (default, CPU, <1 detik/latih ulang): TF-IDF word+char n-gram + Logistic Regression.
  - transformers: model hasil fine-tune IndoBERT/XLM-R (lihat train_transformer.py), dimuat dari direktori.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from minesense.rag.text import normalize
from minesense.schemas import IncidentPrediction

RECOMMENDED_ACTIONS: dict[str, list[str]] = {
    "kendaraan_alat_berat": [
        "Hentikan unit terlibat dan amankan lokasi; lakukan tes alkohol & fatigue check operator (SOP-K3-008).",
        "Periksa kepatuhan batas kecepatan, jarak beriring, dan tanggul pengaman (SOP-K3-001).",
        "Verifikasi hasil P2H unit, terutama rem, steering, dan alarm mundur (SOP-OPS-001).",
    ],
    "longsor_geoteknik": [
        "Tutup akses dan evakuasi zona di bawah lereng; terapkan TARP sesuai level pergerakan (SOP-K3-005).",
        "Minta inspeksi geoteknik segera dan tingkatkan frekuensi pemantauan prisma/radar.",
        "Periksa kondisi drainase dan genangan air di crest/jenjang.",
    ],
    "peledakan": [
        "Barikade area, larang pengeboran dalam jarak < 1 m dari lubang misfire (SOP-K3-004).",
        "Hanya juru ledak & pengawas peledakan yang boleh mendekat setelah waktu tunggu terpenuhi.",
        "Laporkan ke KTT dan lakukan investigasi tingkat tinggi/kritis (SOP-K3-008).",
    ],
    "listrik": [
        "Isolasi sumber energi dan terapkan LOTO lengkap termasuk zero energy test (SOP-K3-002).",
        "Berikan pertolongan pertama hanya setelah sumber listrik dipastikan mati; panggil ERT.",
        "Inspeksi kabel trailing/panel dan pasang proteksi mekanis.",
    ],
    "bekerja_di_ketinggian": [
        "Hentikan pekerjaan; pastikan izin kerja ketinggian dan full body harness 100% tie-off (SOP-K3-003).",
        "Periksa akses tangga (tiga titik kontak) dan pengamanan alat dari jatuh.",
        "Tinjau rencana penyelamatan (evakuasi ≤ 15 menit).",
    ],
    "lingkungan": [
        "Hentikan sumber kebocoran, lokalisasi dengan spill kit dan tanggul (SOP-ENV-001).",
        "Kirim tanah terkontaminasi ke TPS limbah B3 dalam 1 x 24 jam.",
        "Untuk penyimpangan kualitas air: lakukan pengolahan (kapur/flokulan) dan lapor departemen lingkungan (SOP-ENV-002).",
    ],
    "kesehatan_kerja": [
        "Bebastugaskan operator dari mengoperasikan unit dan lakukan fit-to-work assessment (SOP-K3-006).",
        "Berikan istirahat/power nap minimal 20 menit atau ganti operator.",
        "Tinjau pola shift, jam tidur, dan pengendalian paparan (debu/kebisingan).",
    ],
}


# Leksikon domain (weak supervision): membantu generalisasi ke kalimat/pola yang belum pernah dilihat model.
DOMAIN_LEXICON: dict[str, list[str]] = {
    "kendaraan_alat_berat": ["menabrak", "tabrak", "terguling", "tergelincir", "senggol", "mundur", "rem", "kecepatan",
                             "dump truck", "light vehicle", "lv ", "blind spot", "jalur", "haul road", "beriring"],
    "longsor_geoteknik": ["longsor", "lereng", "retakan", "crest", "toe", "rockfall", "batu jatuh", "amblesan",
                          "radar", "prisma", "rembesan", "jenjang", "penggembungan"],
    "peledakan": ["peledakan", "ledak", "misfire", "flyrock", "sirine", "charging", "detonator", "juru ledak", "getaran"],
    "listrik": ["listrik", "tersengat", "panel", "mcc", "breaker", "kabel", "arc flash", "tegangan", "isolator",
                "alternator", "percikan"],
    "bekerja_di_ketinggian": ["ketinggian", "terjatuh", "tangga", "harness", "scaffolding", "deck", "terpeleset",
                              "titik kontak", "atas vessel"],
    "lingkungan": ["tumpah", "tumpahan", "oli", "solar", "coolant", "limbah", "b3", "ph ", "tss", "kolam",
                   "drainase", "bocor"],
    "kesehatan_kerja": ["microsleep", "mengantuk", "tidur", "lelah", "tekanan darah", "pusing", "debu", "kebisingan",
                        "db ", "dehidrasi", "pingsan", "fit to work"],
}


class LexiconFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            low = f" {normalize(t)} "
            rows.append([sum(k in low for k in kws) for kws in DOMAIN_LEXICON.values()])
        return csr_matrix(np.asarray(rows, dtype=np.float32) * 0.6)


def _build_pipeline(C: float = 6.0) -> Pipeline:
    feats = FeatureUnion([
        ("lex", LexiconFeatures()),
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=60000)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


class IncidentClassifier:
    def __init__(self):
        self.cat_model: Pipeline | None = None
        self.risk_model: Pipeline | None = None
        self.meta: dict = {}

    # ---------- training ----------
    def fit(self, texts: list[str], categories: list[str], risks: list[str]) -> "IncidentClassifier":
        t0 = time.time()
        self.cat_model = _build_pipeline().fit(texts, categories)
        self.risk_model = _build_pipeline(C=4.0).fit(texts, risks)
        self.meta = {"backend": "sklearn-tfidf-logreg", "n_train": len(texts),
                     "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_seconds": round(time.time() - t0, 2)}
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cat": self.cat_model, "risk": self.risk_model, "meta": self.meta}, path)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "IncidentClassifier":
        obj = cls()
        d = joblib.load(path)
        obj.cat_model, obj.risk_model, obj.meta = d["cat"], d["risk"], d["meta"]
        return obj

    # ---------- inference ----------
    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return self.cat_model.predict_proba(texts), self.risk_model.predict_proba(texts)

    def predict(self, text: str) -> IncidentPrediction:
        pc, pr = self.predict_proba([text])
        pc, pr = pc[0], pr[0]
        cats, risks = self.cat_model.classes_, self.risk_model.classes_
        ci, ri = int(np.argmax(pc)), int(np.argmax(pr))
        cat, risk = str(cats[ci]), str(risks[ri])
        risk = _apply_safety_overrides(text, risk)
        top = sorted(zip(cats, pc), key=lambda x: -x[1])[:3]
        return IncidentPrediction(
            category=cat, category_confidence=round(float(pc[ci]), 4), risk_level=risk,
            risk_confidence=round(float(pr[ri]), 4), top_categories=[(str(c), round(float(p), 4)) for c, p in top],
            recommended_actions=RECOMMENDED_ACTIONS.get(cat, []),
            requires_escalation=risk in ("tinggi", "kritis") or float(pc[ci]) < 0.5,
        )


# Aturan keselamatan deterministik di atas model ML: model tidak boleh menurunkan risiko kejadian fatal.
_CRITICAL_TERMS = ("meninggal", "fatal", "cacat permanen", "terjepit", "tertimbun", "menimbun", "korban kritis",
                   "keluar dari radius aman")  # flyrock di luar radius aman = kritis (SOP-K3-004 §6)
_HIGH_TERMS = ("patah tulang", "lti", "misfire", "flyrock", "rumah sakit", "high potential")
_ORDER = ["rendah", "sedang", "tinggi", "kritis"]


def _apply_safety_overrides(text: str, risk: str) -> str:
    low = text.lower()
    floor = "rendah"
    if any(t in low for t in _HIGH_TERMS):
        floor = "tinggi"
    if any(t in low for t in _CRITICAL_TERMS):
        floor = "kritis"
    return risk if _ORDER.index(risk) >= _ORDER.index(floor) else floor
