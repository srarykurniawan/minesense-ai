import json

import pytest

from minesense.agent.router import route
from minesense.config import get_settings
from minesense.extraction.shift_report import extract, parse_id_number
from minesense.llm.providers import parse_json


@pytest.fixture(scope="module")
def clf():
    from minesense.classifier.train import main
    main([])
    from minesense.classifier.model import IncidentClassifier
    return IncidentClassifier.load(get_settings().classifier_path)


def test_classifier_basic(clf):
    p = clf.predict("Mekanik tersengat listrik saat memperbaiki panel MCC, LOTO tidak diterapkan.")
    assert p.category == "listrik"
    assert p.recommended_actions


def test_safety_override_never_downgrades_fatal(clf):
    p = clf.predict("Operator terpeleset dari tangga excavator dan meninggal dunia.")
    assert p.risk_level == "kritis" and p.requires_escalation


def test_classifier_quality(clf):
    rep = json.loads((get_settings().artifacts_dir / "reports" / "classifier_eval.json").read_text())
    assert rep["random_split"]["category_macro_f1"] >= 0.9
    assert rep["random_split"]["risk_kritis_recall"] >= 0.85


@pytest.mark.parametrize("s,v", [("25.000", 25000), ("5,25", 5.25), ("1.234,5", 1234.5), ("3.5", 3.5), ("12", 12)])
def test_parse_id_number(s, v):
    assert parse_id_number(s) == v


def test_extraction_hard_set():
    rows = [json.loads(line) for line in open(get_settings().data_dir / "eval" / "shift_reports_hard.jsonl")]
    for r in rows:
        d = extract(r["text"])[0].data.model_dump()
        for k, g in r["gold"].items():
            assert (sorted(d[k]) == sorted(g)) if isinstance(g, list) else d[k] == g, (k, g, d[k])


def test_extraction_consistency_warning():
    res, warns = extract("Shift pagi 01/02/2026 Pit Barat. OB 30.000 BCM, coal 5.000 ton, SR 9,00. Hujan 1 jam. Insiden 0")
    assert res.data.ob_removal_bcm == 30000 and any("Stripping ratio" in w for w in warns)


def test_parse_json_tolerant():
    assert parse_json('Berikut:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


@pytest.mark.parametrize("text,intent", [
    ("Berapa radius aman peledakan?", "qa_sop"),
    ("DT-1043 overheat di tanjakan", "maintenance"),
    ("Apa arti kode E-BRK-301?", "maintenance"),
    ("Dump truck DT-1201 menabrak tanggul di KM 7, operator luka ringan dan dibawa ke klinik", "incident_report"),
    ("Shift malam Pit Barat OB 30.000 BCM coal 5.000 ton", "shift_report"),
    ("Apa itu stripping ratio?", "qa_sop"),
])
def test_router(text, intent):
    assert route(text) == intent


def test_maintenance_infers_fault(rag):
    from minesense.maintenance.diagnose import MaintenanceAssistant
    from minesense.schemas import MaintenanceRequest
    m = MaintenanceAssistant(get_settings().data_dir, rag)
    r = m.diagnose(MaintenanceRequest(equipment="EX-2011", symptom="semua gerakan lambat dan joystick tidak responsif"))
    assert r.fault_code_info["code"] == "X-HYD-120"
    r2 = m.diagnose(MaintenanceRequest(equipment="DT-1001", symptom="rem terasa dalam", fault_code="E-BRK-301"))
    assert r2.recommended_steps[0].startswith("ITEM KRITIS")
