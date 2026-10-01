"""Skema domain & API (Pydantic v2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------- Taksonomi insiden ----------
HAZARD_CATEGORIES = [
    "kendaraan_alat_berat",   # tabrakan/terguling HD, LV, dozer
    "longsor_geoteknik",      # kegagalan lereng, runtuhan
    "peledakan",              # misfire, flyrock
    "listrik",                # sengatan, arc flash
    "bekerja_di_ketinggian",  # jatuh dari ketinggian
    "lingkungan",             # tumpahan oli, air asam tambang
    "kesehatan_kerja",        # debu, kebisingan, kelelahan
]
RISK_LEVELS = ["rendah", "sedang", "tinggi", "kritis"]


class Citation(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    score: float
    snippet: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    filters: dict[str, str] | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    intent: str
    grounded: bool
    warnings: list[str] = []
    latency_ms: float
    provider: str
    data: dict | None = None  # payload terstruktur untuk intent non-QA (klasifikasi, ekstraksi, diagnosa)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    filters: dict[str, str] | None = None


class IncidentRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=5000)


class IncidentPrediction(BaseModel):
    category: str
    category_confidence: float
    risk_level: str
    risk_confidence: float
    top_categories: list[tuple[str, float]]
    recommended_actions: list[str]
    requires_escalation: bool


class ShiftReport(BaseModel):
    """Struktur target ekstraksi laporan produksi shift."""
    tanggal: str | None = None
    shift: Literal["pagi", "malam", "siang"] | None = None
    pit: str | None = None
    ob_removal_bcm: float | None = Field(None, description="Overburden removal (BCM)")
    coal_getting_ton: float | None = None
    stripping_ratio: float | None = None
    hujan_jam: float | None = Field(None, description="Jam hilang akibat hujan/slippery")
    unit_breakdown: list[str] = []
    jumlah_insiden: int | None = None
    catatan: str | None = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)
    use_llm: bool = False


class ExtractResponse(BaseModel):
    data: ShiftReport
    method: Literal["rules", "llm", "llm+rules"]
    missing_fields: list[str]
    warnings: list[str] = []


class MaintenanceRequest(BaseModel):
    equipment: str = Field(..., examples=["DT-1043", "EX-2011", "DT-100"])
    symptom: str = Field(..., min_length=5, max_length=2000)
    fault_code: str | None = None
    hour_meter: float | None = None


class MaintenanceResponse(BaseModel):
    equipment: str
    probable_causes: list[str]
    recommended_steps: list[str]
    similar_work_orders: list[dict]
    fault_code_info: dict | None
    citations: list[Citation]
    safety_note: str
    answer: str


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str
    answer: str
    rating: int = Field(..., ge=-1, le=1)
    comment: str | None = None
