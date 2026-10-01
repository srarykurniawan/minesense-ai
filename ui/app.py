"""MineSense UI (Streamlit). Jalankan: streamlit run ui/app.py  (butuh API berjalan di MINESENSE_API_URL)."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("MINESENSE_API_URL", "http://localhost:8000")
KEY = os.getenv("MINESENSE_UI_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}

st.set_page_config(page_title="MineSense AI", page_icon="⛏️", layout="wide")


def call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", headers=HEADERS, timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except httpx.HTTPError as e:
        st.error(f"Tidak dapat menghubungi API di {API}: {e}")
    return None


def show_citations(cits):
    if cits:
        with st.expander(f"📚 Sumber ({len(cits)})"):
            for i, c in enumerate(cits, 1):
                st.markdown(f"**[{i}] {c['title']}** · `{c['doc_id']}` · skor {c['score']}")
                st.caption(c["snippet"])


st.sidebar.title("⛏️ MineSense AI")
st.sidebar.caption("Asisten AI K3, operasi & maintenance tambang")
info = call("GET", "/v1/info")
if info:
    st.sidebar.metric("Dokumen terindeks", info["index"]["n_docs"])
    st.sidebar.write(f"LLM: `{info['llm']}` {info.get('llm_model') or ''}")
    st.sidebar.write(f"Classifier: `{info['classifier'].get('backend')}`")
    st.sidebar.write(f"Versi: {info['version']} · prompt {info['prompt_version']}")

tab_chat, tab_inc, tab_ext, tab_mt, tab_doc = st.tabs(
    ["💬 Asisten", "🚨 Klasifikasi Insiden", "📊 Ekstraksi Laporan Shift", "🔧 Diagnosa Maintenance", "📁 Dokumen"])

with tab_chat:
    st.caption("Tanya SOP/K3/regulasi, laporkan insiden, tempel laporan shift, atau jelaskan gejala kerusakan unit.")
    if "history" not in st.session_state:
        st.session_state.history = []
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])
            if h.get("citations"):
                show_citations(h["citations"])
    if q := st.chat_input("Contoh: Berapa radius aman peledakan untuk manusia?"):
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Mencari di dokumen..."):
                r = call("POST", "/v1/chat", json={"question": q})
            if r:
                st.markdown(r["answer"])
                st.caption(f"intent: `{r['intent']}` · grounded: {'✅' if r['grounded'] else '⚠️'} · "
                           f"{r['latency_ms']} ms · {r['provider']}")
                for w in r["warnings"]:
                    st.warning(w)
                show_citations(r["citations"])
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up{len(st.session_state.history)}"):
                    call("POST", "/v1/feedback", json={"question": q, "answer": r["answer"], "rating": 1})
                if c2.button("👎", key=f"dn{len(st.session_state.history)}"):
                    call("POST", "/v1/feedback", json={"question": q, "answer": r["answer"], "rating": -1})
                st.session_state.history.append({"role": "assistant", "content": r["answer"],
                                                 "citations": r["citations"]})

with tab_inc:
    desc = st.text_area("Deskripsi insiden", height=140, value=(
        "Dump truck DT-1043 menabrak tanggul di haul road KM 7 Pit Barat pukul 03.40 saat jalan basah. "
        "Operator mengalami luka robek dan dijahit di klinik."))
    if st.button("Klasifikasikan", type="primary"):
        r = call("POST", "/v1/incidents/classify", json={"description": desc})
        if r:
            c1, c2, c3 = st.columns(3)
            c1.metric("Kategori", r["category"].replace("_", " "), f"{r['category_confidence']:.0%}")
            c2.metric("Tingkat risiko", r["risk_level"].upper(), f"{r['risk_confidence']:.0%}")
            c3.metric("Eskalasi", "YA" if r["requires_escalation"] else "Tidak")
            st.subheader("Tindakan segera")
            for a in r["recommended_actions"]:
                st.markdown(f"- {a}")
            st.bar_chart(pd.DataFrame(r["top_categories"], columns=["kategori", "prob"]).set_index("kategori"))

with tab_ext:
    txt = st.text_area("Tempel laporan shift (format bebas)", height=200, value=(
        "Lapor ndan, shift 2 tanggal 3 Mei 2026 di Pit Kenari. Overburden 41.350 BCM, batubara 8.270 ton, SR 5,00. "
        "Hujan 2,5 jam. Breakdown: DT-1123, EX-2045. Insiden 1. Catatan: ramp B2 perlu disiram."))
    use_llm = st.checkbox("Gunakan LLM (bila provider aktif)")
    if st.button("Ekstrak", type="primary"):
        r = call("POST", "/v1/reports/extract", json={"text": txt, "use_llm": use_llm})
        if r:
            st.json(r["data"])
            st.caption(f"metode: {r['method']}")
            if r["missing_fields"]:
                st.warning(f"Field kosong: {', '.join(r['missing_fields'])}")
            for w in r["warnings"]:
                st.warning(w)

with tab_mt:
    c1, c2 = st.columns(2)
    eq = c1.text_input("Unit / model", "DT-1043")
    code = c2.text_input("Kode gangguan (opsional)", "")
    sym = st.text_area("Gejala", "Suhu mesin naik terus saat menanjak, indikator temperatur merah")
    if st.button("Diagnosa", type="primary"):
        r = call("POST", "/v1/maintenance/diagnose", json={"equipment": eq, "symptom": sym, "fault_code": code or None})
        if r:
            st.markdown(r["answer"])
            if r["fault_code_info"]:
                st.info(f"Kode: {r['fault_code_info']['code']} — {r['fault_code_info']['title']}"
                        + (" (diinferensi)" if r["fault_code_info"]["inferred"] else ""))
            st.subheader("Work order serupa")
            st.dataframe(pd.DataFrame(r["similar_work_orders"]), width="stretch")
            show_citations(r["citations"])

with tab_doc:
    up = st.file_uploader("Unggah SOP/manual (.md, .txt, .pdf) untuk diindeks", type=["md", "txt", "pdf"])
    if up and st.button("Indeks dokumen"):
        r = call("POST", "/v1/documents", files={"file": (up.name, up.getvalue())})
        if r:
            st.success(f"Terindeks: {r['title']} ({r['chunks']} chunk)")
    q = st.text_input("Cari di dokumen (debug retrieval)")
    if q:
        res = call("POST", "/v1/search", json={"query": q, "top_k": 5})
        if res:
            st.dataframe(pd.DataFrame(res)[["doc_id", "section", "score", "text"]], width="stretch")
