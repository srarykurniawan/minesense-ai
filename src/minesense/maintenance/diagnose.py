"""Asisten diagnosa maintenance alat berat: kode gangguan + histori work order (similarity) + RAG manual."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from minesense.llm.prompts import MAINT_SYSTEM
from minesense.llm.providers import LLMError
from minesense.rag.pipeline import RAGPipeline, format_context, to_citations
from minesense.rag.text import normalize
from minesense.schemas import MaintenanceRequest, MaintenanceResponse

PREFIX_TO_MODEL = {"DT": "DT-100", "EX": "EX-2000", "BD": "BD-375"}
SAFETY = ("Terapkan LOTO (SOP-K3-002) sebelum perawatan: matikan unit, isolasi baterai, buang tekanan hidrolik, "
          "turunkan/topang attachment dan vessel, lalu lakukan zero energy test.")


def resolve_model(equipment: str) -> str | None:
    eq = equipment.upper().strip()
    if eq in PREFIX_TO_MODEL.values():
        return eq
    return PREFIX_TO_MODEL.get(eq.split("-")[0])


class MaintenanceAssistant:
    def __init__(self, data_dir: Path, rag: RAGPipeline):
        self.rag = rag
        self.faults: dict = json.loads((data_dir / "fault_codes.json").read_text(encoding="utf-8"))
        with open(data_dir / "work_orders.csv", encoding="utf-8") as f:
            self.wos = list(csv.DictReader(f))
        self.vec = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.wo_matrix = self.vec.fit_transform([f"{w['symptom']} {w['fault_code']}" for w in self.wos])
        sym_rows = [(code, f"{f['title']} {' '.join(f['symptoms'])}") for code, f in self.faults.items()]
        self.fault_codes = [c for c, _ in sym_rows]
        self.fault_matrix = self.vec.transform([t for _, t in sym_rows])

    def similar_work_orders(self, model: str | None, symptom: str, k: int = 3) -> list[dict]:
        sims = cosine_similarity(self.vec.transform([symptom]), self.wo_matrix)[0]
        order = sims.argsort()[::-1]
        out = []
        for i in order:
            w = self.wos[i]
            if model and w["model"] != model:
                continue
            out.append({**{k_: w[k_] for k_ in ("wo_id", "unit", "fault_code", "symptom", "root_cause", "action",
                                               "downtime_hours")}, "similarity": round(float(sims[i]), 3)})
            if len(out) >= k:
                break
        return out

    def infer_fault(self, model: str | None, symptom: str, similar: list[dict]) -> str | None:
        sims = cosine_similarity(self.vec.transform([symptom]), self.fault_matrix)[0]
        scores: Counter = Counter()
        for code, s in zip(self.fault_codes, sims):
            if not model or self.faults[code]["model"] == model:
                scores[code] += float(s)
        for w in similar:  # voting dari histori WO
            if w["fault_code"]:
                scores[w["fault_code"]] += 0.5 * w["similarity"]
        return scores.most_common(1)[0][0] if scores else None

    def diagnose(self, req: MaintenanceRequest) -> MaintenanceResponse:
        model = resolve_model(req.equipment)
        similar = self.similar_work_orders(model, req.symptom)
        code = (req.fault_code or "").upper().strip() or self.infer_fault(model, req.symptom, similar)
        fault = self.faults.get(code) if code else None

        query = f"{req.equipment} {model or ''} {code or ''} {req.symptom}"
        filters = {"doc_type": "manual"}
        items = self.rag.retriever.retrieve(query, 4, filters)

        causes = list(fault["causes"]) if fault else []
        for w in similar:  # tambahkan akar masalah dari histori bila belum ada
            if w["root_cause"] not in causes:
                causes.append(w["root_cause"])
        steps = list(fault["actions"]) if fault else []
        if fault and fault.get("critical"):
            steps.insert(0, "ITEM KRITIS: pasang tag DO NOT OPERATE, unit tidak boleh beroperasi sampai diperbaiki.")

        hist = "\n".join(f"- {w['wo_id']} {w['unit']} [{w['fault_code'] or '-'}] {w['symptom']} → "
                         f"penyebab: {w['root_cause']}; tindakan: {w['action']}" for w in similar)
        user = (f"KONTEKS:\n{format_context(items)}\n\nHISTORI WORK ORDER SERUPA:\n{hist}\n\n"
                f"PERTANYAAN: Unit {req.equipment} (model {model}), gejala: {req.symptom}. "
                f"Kode: {code or 'tidak ada'}. Hour meter: {req.hour_meter or '-'}.")
        try:
            answer = self.rag.llm.generate(MAINT_SYSTEM, user)
        except LLMError:
            answer = self.rag.fallback.generate(MAINT_SYSTEM, user)
        if self.rag.llm.name == "extractive":
            answer = self._template_answer(req, model, code, fault, causes, steps)

        return MaintenanceResponse(
            equipment=req.equipment, probable_causes=causes[:6], recommended_steps=steps, similar_work_orders=similar,
            fault_code_info={"code": code, "title": fault["title"], "model": fault["model"],
                             "critical": fault.get("critical", False), "inferred": not req.fault_code} if fault else None,
            citations=to_citations(items), safety_note=SAFETY, answer=answer)

    @staticmethod
    def _template_answer(req, model, code, fault, causes, steps) -> str:
        head = (f"Diagnosa untuk {req.equipment} ({model or 'model tidak dikenal'}). "
                + (f"Kode terkait: {code} — {fault['title']}" + (" (diinferensi dari gejala)." if not req.fault_code else ".")
                   if fault else "Kode gangguan tidak dapat dipastikan; lakukan pemeriksaan umum."))
        return (head + "\n1) Kemungkinan penyebab:\n" + "\n".join(f"   - {c}" for c in causes[:6])
                + "\n2) Langkah pemeriksaan & perbaikan:\n" + "\n".join(f"   - {s}" for s in steps)
                + f"\n3) Keselamatan: {SAFETY}")
