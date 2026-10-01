import pytest
from fastapi.testclient import TestClient

H = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    from minesense.api.main import app
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["n_docs"] >= 10


def test_auth_required(client):
    assert client.post("/v1/chat", json={"question": "radius aman peledakan?"}).status_code == 401


def test_chat_qa(client):
    r = client.post("/v1/chat", json={"question": "Berapa radius aman peledakan untuk manusia?"}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "qa_sop" and "500 meter" in body["answer"] and body["citations"]
    assert "x-request-id" in r.headers


def test_chat_routes_incident(client):
    r = client.post("/v1/chat", headers=H, json={"question": "Dump truck DT-1201 terguling di tepi dumping, "
                                                              "operator patah tulang dan dirujuk ke rumah sakit"}).json()
    assert r["intent"] == "incident_report" and r["data"]["risk_level"] in ("tinggi", "kritis")


def test_classify_extract_diagnose(client):
    r = client.post("/v1/incidents/classify", headers=H,
                    json={"description": "Ditemukan 2 lubang misfire setelah peledakan di Pit Barat."}).json()
    assert r["category"] == "peledakan"
    r = client.post("/v1/reports/extract", headers=H, json={
        "text": "Shift pagi 2026-07-21 Pit Utara OB: 18.900 bcm Coal: 6.300 ton Insiden: 0"}).json()
    assert r["data"]["ob_removal_bcm"] == 18900 and r["data"]["stripping_ratio"] == 3.0
    r = client.post("/v1/maintenance/diagnose", headers=H,
                    json={"equipment": "DT-1043", "symptom": "vessel lambat naik saat dumping"}).json()
    assert r["fault_code_info"]["code"] == "E-HYD-601"


def test_upload_document_then_answer(client):
    md = b"---\ndoc_id: SOP-TEST-9\ntitle: Prosedur Drone Survey\n---\n# Drone\nTinggi terbang drone survey maksimum 120 meter di atas pit."
    r = client.post("/v1/documents", headers=H, files={"file": ("drone.md", md, "text/markdown")})
    assert r.status_code == 200 and r.json()["doc_id"] == "SOP-TEST-9"
    a = client.post("/v1/chat", headers=H, json={"question": "Berapa tinggi terbang maksimum drone survey?"}).json()
    assert "120 meter" in a["answer"]


def test_upload_rejects_bad_type(client):
    r = client.post("/v1/documents", headers=H, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415


def test_feedback_metrics_info(client):
    assert client.post("/v1/feedback", headers=H, json={"question": "q", "answer": "a", "rating": 1}).status_code == 200
    assert "minesense_requests_total" in client.get("/metrics").text
    info = client.get("/v1/info", headers=H).json()
    assert info["feedback"]["positive"] >= 1


def test_validation_error(client):
    assert client.post("/v1/chat", headers=H, json={"question": "a"}).status_code == 422
