"""MineSense UI (Streamlit).

Dua mode:
  - remote   : UI memanggil API terpisah di MINESENSE_API_URL (Docker/VPS).  `streamlit run ui/app.py`
  - embedded : MINESENSE_API_URL=embedded → API dijalankan di dalam proses Streamlit (Streamlit Community Cloud,
               satu proses saja). Entry point: `streamlit run streamlit_app.py`
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("MINESENSE_API_URL", "http://localhost:8000")
KEY = os.getenv("MINESENSE_UI_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}
EMBEDDED = API.strip().lower() == "embedded"
ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="MineSense AI", page_icon="⛏️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

/* hero */
.ms-hero { background: linear-gradient(120deg, #0f1b2d 0%, #16365e 60%, #1d5ea8 100%); border-radius: 22px;
  padding: 30px 34px; color: #fff; margin-bottom: 18px; position: relative; overflow: hidden; }
.ms-hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 50%; background: rgba(255,255,255,.06); }
.ms-hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -.02em; color: #fff; }
.ms-hero p { font-size: 1.02rem; color: #c5d1e3; margin: 0; max-width: 720px; }
.ms-badges { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.ms-badge { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2); border-radius: 999px;
  padding: 4px 12px; font-size: .8rem; color: #e6edf7; }

/* KPI cards */
.ms-kpi { background: #fff; border: 1px solid #e6e5e1; border-radius: 16px; padding: 16px 18px; height: 100%; }
.ms-kpi .v { font-size: 1.7rem; font-weight: 800; color: #0b0b0b; line-height: 1.1; }
.ms-kpi .l { font-size: .82rem; color: #52514e; margin-top: 4px; }

/* tabs as pills */
.stTabs [role="tablist"] { gap: 8px; border-bottom: none !important; flex-wrap: wrap; }
.stTabs [role="tab"] { background: #f1f0ec !important; border-radius: 999px !important; padding: 8px 18px !important;
  height: auto !important; border: none !important; }
.stTabs [role="tab"][aria-selected="true"] { background: #2a78d6 !important; }
.stTabs [role="tab"][aria-selected="true"] p { color: #fff !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* generic cards & chips */
.ms-card { background: #fff; border: 1px solid #e6e5e1; border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }
.ms-card h4 { margin: 0 0 8px 0; font-size: 1rem; }
.ms-chip { display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: .78rem; font-weight: 600;
  margin-right: 6px; background: #eaf2fc; color: #1d5ea8; }
.ms-risk { display: inline-flex; align-items: center; gap: 8px; border-radius: 12px; padding: 10px 16px;
  font-weight: 800; font-size: 1.15rem; color: #fff; }
.ms-muted { color: #8a8984; font-size: .82rem; }
.ms-section { font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #2a78d6;
  margin: 6px 0 8px 0; }

/* suggestion buttons */
div[data-testid="stButton"] button[kind="secondary"] { border-radius: 14px; border: 1px solid #e6e5e1;
  background: #fff; text-align: left; justify-content: flex-start; min-height: 64px; }
div[data-testid="stButton"] button[kind="secondary"]:hover { border-color: #2a78d6; color: #1d5ea8; }
div[data-testid="stButton"] button[kind="primary"] { border-radius: 12px; background: #2a78d6; border-color: #2a78d6; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0f1b2d; }
section[data-testid="stSidebar"] * { color: #e6edf7; }
section[data-testid="stSidebar"] .ms-side-title { font-size: 1.35rem; font-weight: 800; margin: 4px 0 2px 0; }
section[data-testid="stSidebar"] .ms-side-sub { color: #8ea3c2 !important; font-size: .85rem; }
.ms-status { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1f3150;
  font-size: .86rem; }
.ms-status span:last-child { font-weight: 600; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button { background: transparent !important;
  border: 1px solid #2c3f5c !important; min-height: 40px; justify-content: center; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover { border-color: #7fb3f0 !important; }
.ms-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #1baf7a; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- backend
@st.cache_resource(show_spinner="Memuat index RAG & model (±20–60 detik saat pertama kali)...")
def _embedded_client():
    """Menjalankan FastAPI app di dalam proses (tanpa server terpisah) — semua logika endpoint dipakai ulang."""
    from fastapi.testclient import TestClient

    from minesense.api.main import app

    client = TestClient(app)
    client.__enter__()  # jalankan lifespan (load index + classifier)
    return client


def call(method: str, path: str, **kw):
    if EMBEDDED:
        r = _embedded_client().request(method, path, headers=HEADERS, **kw)
        if r.status_code >= 400:
            st.error(f"Error {r.status_code}: {r.text[:300]}")
            return None
        return r.json()
    try:
        r = httpx.request(method, f"{API}{path}", headers=HEADERS, timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except httpx.HTTPError as e:
        st.error(f"Tidak dapat menghubungi API di {API}: {e}")
    return None


def esc(x) -> str:
    return html.escape(str(x))


RISK_STYLE = {"rendah": ("#0e8a4b", "●"), "sedang": ("#c98500", "▲"), "tinggi": ("#e0601f", "◆"), "kritis": ("#d23b3b", "■")}
INTENT_LABEL = {"qa_sop": "Tanya SOP/K3", "incident_report": "Laporan insiden", "shift_report": "Laporan shift",
                "maintenance": "Maintenance", "out_of_scope": "Di luar topik", "blocked": "Ditolak"}


def show_citations(cits):
    if cits:
        with st.expander(f"📚 Sumber dokumen ({len(cits)})"):
            for i, c in enumerate(cits, 1):
                st.markdown(f"**[{i}] {c['title']}**  \n`{c['doc_id']}` · relevansi {c['score']}")
                st.caption(c["snippet"])


# ----------------------------------------------------------------------------- sidebar
info = call("GET", "/v1/info")
with st.sidebar:
    logo = ASSETS / "logo.png"
    if logo.exists():
        b64 = base64.b64encode(logo.read_bytes()).decode()
        st.markdown(f'<div style="text-align:center;margin:-1rem 0 .8rem 0;background:#fff;border-radius:16px;'
                    f'padding:12px"><img src="data:image/png;base64,{b64}" style="width:88%;max-width:210px"></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="ms-side-title">⛏️ MineSense AI</div>'
                '<div class="ms-side-sub">Asisten AI K3, operasi & maintenance tambang</div>',
                unsafe_allow_html=True)
    st.write("")
    if info:
        rows = [("Status", '<span class="ms-dot"></span>Online'),
                ("Dokumen terindeks", info["index"]["n_docs"]),
                ("Mesin jawaban", "LLM" if info["llm"] != "extractive" else "Ekstraktif"),
                ("Pencarian", "BM25 + vektor"),
                ("Versi", info["version"])]
        st.markdown("".join(f'<div class="ms-status"><span>{k}</span><span>{v}</span></div>' for k, v in rows),
                    unsafe_allow_html=True)
    st.write("")
    with st.expander("ℹ️ Tentang aplikasi"):
        st.markdown("Jawaban disusun **hanya dari dokumen** dan selalu mencantumkan sumber. "
                    "Untuk keputusan keselamatan, tetap ikuti SOP resmi dan instruksi pengawas/KTT.")
    if st.button("🗑️ Hapus percakapan", width="stretch"):
        st.session_state.history = []
        st.rerun()
    st.markdown('<div class="ms-side-sub" style="margin-top:18px">Demo dengan data sintetis · perusahaan fiktif</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- hero + KPI
st.markdown("""
<div class="ms-hero">
  <h1>Asisten AI untuk Tambang</h1>
  <p>Tanya SOP & regulasi K3, nilai risiko insiden, baca laporan shift, dan diagnosa kerusakan alat berat —
     setiap jawaban disertai sumber dokumennya.</p>
  <div class="ms-badges"><span class="ms-badge">RAG bersitasi</span><span class="ms-badge">Klasifikasi risiko</span>
  <span class="ms-badge">Ekstraksi laporan</span><span class="ms-badge">Diagnosa alat berat</span></div>
</div>""", unsafe_allow_html=True)

k = st.columns(4)
kpis = [(info["index"]["n_docs"] if info else "–", "Dokumen SOP & manual"), ("4", "Modul AI"),
        ("96%", "Dokumen tepat ditemukan*"), ("75%", "Jawaban benar*")]
for col, (v, lbl) in zip(k, kpis):
    col.markdown(f'<div class="ms-kpi"><div class="v">{v}</div><div class="l">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown('<div class="ms-muted" style="margin:6px 0 14px 2px">*diukur pada 115 pertanyaan bergaya lapangan '
            '(data uji sintetis)</div>', unsafe_allow_html=True)

tab_chat, tab_inc, tab_ext, tab_mt, tab_doc = st.tabs(
    ["💬 Asisten", "🚨 Klasifikasi Insiden", "📊 Laporan Shift", "🔧 Diagnosa Maintenance", "📁 Dokumen"])

# ----------------------------------------------------------------------------- chat
SUGGESTIONS = [
    ("🧨", "Berapa radius aman peledakan untuk manusia?"),
    ("🔒", "Siapa yang boleh melepas gembok LOTO jika pemiliknya tidak ada?"),
    ("🚛", "Dump truck DT-1201 terguling di tepi dumping, operator patah tulang dan dirujuk ke rumah sakit"),
    ("🔧", "EX-2011 semua gerakan lambat dan joystick tidak responsif"),
]

with tab_chat:
    st.session_state.setdefault("history", [])
    pending = st.session_state.pop("pending_q", None)

    if not st.session_state.history and not pending:
        st.markdown('<div class="ms-section">Coba salah satu contoh</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, s) in enumerate(SUGGESTIONS):
            if cols[i % 2].button(f"{icon}  {s}", key=f"sg{i}", width="stretch"):
                st.session_state.pending_q = s
                st.rerun()

    msgs = st.container()  # riwayat di atas, kotak input selalu di bawahnya
    with msgs:
        for h in st.session_state.history:
            with st.chat_message(h["role"], avatar="🧑‍🏭" if h["role"] == "user" else "⛏️"):
                st.markdown(h["content"])
                if h.get("meta"):
                    st.markdown(h["meta"], unsafe_allow_html=True)
                if h.get("citations"):
                    show_citations(h["citations"])

    q = st.chat_input("Tulis pertanyaan, laporan insiden, laporan shift, atau gejala kerusakan…") or pending
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with msgs:
            with st.chat_message("user", avatar="🧑‍🏭"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="⛏️"):
                with st.spinner("Mencari di dokumen…"):
                    r = call("POST", "/v1/chat", json={"question": q})
                if r:
                    st.markdown(r["answer"])
                    ok = r["grounded"]
                    meta = (f'<span class="ms-chip">{esc(INTENT_LABEL.get(r["intent"], r["intent"]))}</span>'
                            f'<span class="ms-chip" style="background:{"#e6f6ee" if ok else "#fdeee7"};'
                            f'color:{"#0e7a43" if ok else "#b8461c"}">{"Sesuai dokumen" if ok else "Perlu verifikasi"}</span>'
                            f'<span class="ms-muted">{r["latency_ms"]} ms</span>')
                    st.markdown(meta, unsafe_allow_html=True)
                    for w in r["warnings"]:
                        st.warning(w)
                    show_citations(r["citations"])
                    n = len(st.session_state.history)
                    c1, c2, _ = st.columns([1, 1, 10])
                    if c1.button("👍", key=f"up{n}"):
                        call("POST", "/v1/feedback", json={"question": q, "answer": r["answer"], "rating": 1})
                        st.toast("Terima kasih atas masukannya!")
                    if c2.button("👎", key=f"dn{n}"):
                        call("POST", "/v1/feedback", json={"question": q, "answer": r["answer"], "rating": -1})
                        st.toast("Masukan dicatat untuk perbaikan.")
                    st.session_state.history.append({"role": "assistant", "content": r["answer"], "meta": meta,
                                                     "citations": r["citations"]})

# ----------------------------------------------------------------------------- incident
with tab_inc:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="ms-section">Laporan kejadian</div>', unsafe_allow_html=True)
        desc = st.text_area("Deskripsi insiden", height=180, label_visibility="collapsed", value=(
            "Dump truck DT-1043 menabrak tanggul di haul road KM 7 Pit Barat pukul 03.40 saat jalan basah. "
            "Operator mengalami luka robek dan dijahit di klinik."))
        go = st.button("Nilai risiko", type="primary", width="stretch")
    with right:
        st.markdown('<div class="ms-section">Hasil penilaian</div>', unsafe_allow_html=True)
        if go:
            r = call("POST", "/v1/incidents/classify", json={"description": desc})
            if r:
                color, icon = RISK_STYLE.get(r["risk_level"], ("#52514e", "●"))
                st.markdown(
                    f'<div class="ms-card"><div class="ms-muted">Tingkat risiko</div>'
                    f'<div class="ms-risk" style="background:{color};margin:6px 0 12px">{icon} {r["risk_level"].upper()}</div>'
                    f'<div class="ms-muted">Kategori bahaya</div><h4>{esc(r["category"].replace("_", " ").title())}</h4>'
                    f'<div>{"🚨 <b>Wajib eskalasi</b> ke pengawas/KTT" if r["requires_escalation"] else "✅ Tidak perlu eskalasi"}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown("**Tindakan segera**")
                for a in r["recommended_actions"]:
                    st.markdown(f"- {a}")
                with st.expander("Keyakinan model per kategori"):
                    for cat, p in r["top_categories"]:
                        st.progress(float(p), text=f"{cat.replace('_', ' ')} · {p:.0%}")
        else:
            st.info("Tulis atau tempel laporan kejadian, lalu klik **Nilai risiko**.")

# ----------------------------------------------------------------------------- shift report
with tab_ext:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="ms-section">Tempel laporan (format bebas / WhatsApp)</div>', unsafe_allow_html=True)
        txt = st.text_area("Laporan shift", height=200, label_visibility="collapsed", value=(
            "Lapor ndan, shift 2 tanggal 3 Mei 2026 di Pit Kenari. Overburden 41.350 BCM, batubara 8.270 ton, SR 5,00. "
            "Hujan 2,5 jam. Breakdown: DT-1123, EX-2045. Insiden 1. Catatan: ramp B2 perlu disiram."))
        use_llm = st.checkbox("Gunakan LLM (bila aktif)")
        go = st.button("Baca laporan", type="primary", width="stretch")
    with right:
        st.markdown('<div class="ms-section">Data terstruktur</div>', unsafe_allow_html=True)
        if go:
            r = call("POST", "/v1/reports/extract", json={"text": txt, "use_llm": use_llm})
            if r:
                d = r["data"]
                fmt = lambda v, s="": "–" if v is None else f"{v:,.0f}{s}".replace(",", ".")  # noqa: E731
                st.markdown(f'<div class="ms-card"><span class="ms-chip">{esc(d.get("pit") or "-")}</span>'
                            f'<span class="ms-chip">Shift {esc(d.get("shift") or "-")}</span>'
                            f'<span class="ms-chip">{esc(d.get("tanggal") or "-")}</span></div>', unsafe_allow_html=True)
                m = st.columns(2)
                m[0].metric("Overburden", fmt(d.get("ob_removal_bcm"), " BCM"))
                m[1].metric("Batubara", fmt(d.get("coal_getting_ton"), " ton"))
                m = st.columns(3)
                m[0].metric("Stripping ratio", d.get("stripping_ratio") or "–")
                m[1].metric("Jam hujan", d.get("hujan_jam") if d.get("hujan_jam") is not None else "–")
                m[2].metric("Insiden", d.get("jumlah_insiden") if d.get("jumlah_insiden") is not None else "–")
                st.markdown("**Unit breakdown:** " + (", ".join(f"`{u}`" for u in d.get("unit_breakdown") or []) or "nihil"))
                if r["missing_fields"]:
                    st.warning(f"Belum terbaca: {', '.join(r['missing_fields'])}")
                for w in r["warnings"]:
                    st.warning(w)
                with st.expander("Lihat JSON"):
                    st.json(d)
        else:
            st.info("Tempel laporan shift, lalu klik **Baca laporan**.")

# ----------------------------------------------------------------------------- maintenance
with tab_mt:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="ms-section">Informasi unit</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        eq = c1.text_input("Unit / model", "DT-1043")
        code = c2.text_input("Kode gangguan (opsional)", "")
        sym = st.text_area("Gejala", "Suhu mesin naik terus saat menanjak, indikator temperatur merah", height=120)
        go = st.button("Diagnosa", type="primary", width="stretch")
    with right:
        st.markdown('<div class="ms-section">Hasil diagnosa</div>', unsafe_allow_html=True)
        if go:
            r = call("POST", "/v1/maintenance/diagnose",
                     json={"equipment": eq, "symptom": sym, "fault_code": code or None})
            if r:
                fc = r["fault_code_info"]
                if fc:
                    st.markdown(f'<div class="ms-card"><div class="ms-muted">Kode gangguan'
                                f'{" (diperkirakan dari gejala)" if fc["inferred"] else ""}</div>'
                                f'<h4>{esc(fc["code"])} — {esc(fc["title"])}</h4>'
                                f'{"<span class=ms-chip style=background:#fde8e8;color:#b02a2a>ITEM KRITIS</span>" if fc["critical"] else ""}'
                                f'</div>', unsafe_allow_html=True)
                st.markdown("**Kemungkinan penyebab**")
                for c in r["probable_causes"][:5]:
                    st.markdown(f"- {c}")
                st.markdown("**Langkah pemeriksaan**")
                for i, s in enumerate(r["recommended_steps"], 1):
                    st.markdown(f"{i}. {s}")
                st.warning(f"🔒 {r['safety_note']}")
        else:
            st.info("Isi unit dan gejala, lalu klik **Diagnosa**.")
    if go and r:
        st.markdown('<div class="ms-section" style="margin-top:12px">Work order serupa dari histori</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(r["similar_work_orders"]), width="stretch", hide_index=True)
        show_citations(r["citations"])

# ----------------------------------------------------------------------------- documents
with tab_doc:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="ms-section">Tambah dokumen</div>', unsafe_allow_html=True)
        up = st.file_uploader("Unggah SOP/manual (.md, .txt, .pdf)", type=["md", "txt", "pdf"])
        if up and st.button("Indeks dokumen", type="primary"):
            r = call("POST", "/v1/documents", files={"file": (up.name, up.getvalue())})
            if r:
                st.success(f"Terindeks: {r['title']} ({r['chunks']} bagian)")
    with right:
        st.markdown('<div class="ms-section">Cari di dokumen</div>', unsafe_allow_html=True)
        qd = st.text_input("Kata kunci", placeholder="mis. jarak aman peledakan")
        if qd:
            res = call("POST", "/v1/search", json={"query": qd, "top_k": 5})
            for x in res or []:
                st.markdown(f'<div class="ms-card"><span class="ms-chip">{esc(x["doc_id"])}</span>'
                            f'<b>{esc(x["section"])}</b><div class="ms-muted" style="margin-top:6px">'
                            f'{esc(x["text"][:260])}…</div></div>', unsafe_allow_html=True)
