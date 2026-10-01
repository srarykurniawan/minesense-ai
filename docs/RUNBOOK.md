# Runbook Operasional

## Mengganti data sintetis dengan data asli
| Data | Lokasi | Format | Setelah diganti |
|---|---|---|---|
| SOP / manual / regulasi | `data/corpus/*.md\|.txt\|.pdf` atau upload via UI/`POST /v1/documents` | Markdown dengan front-matter `doc_id`, `title`, `doc_type` (sop/manual/regulation_summary/guideline), `equipment` | `minesense ingest` (atau otomatis saat upload) |
| Laporan insiden historis | `data/incidents.csv` | kolom `incident_id, description, category, risk_level[, template_id]` | `minesense train` lalu `minesense eval` |
| Work order maintenance | `data/work_orders.csv` | `wo_id, unit, model, fault_code, symptom, root_cause, action, downtime_hours, hour_meter` | restart API |
| Tabel kode gangguan | `data/fault_codes.json` | `{code: {code, model, title, causes[], actions[], symptoms[], critical}}` | restart API; sesuaikan `PREFIX_TO_MODEL` di `maintenance/diagnose.py` |
| Eval set RAG | `data/eval/rag_qa.jsonl` | `{question, doc_id, keywords[]}` — **tulis oleh pengguna nyata (pengawas/K3), dengan parafrase** | `minesense eval` |
| Laporan shift | `data/shift_reports.jsonl`, `data/eval/shift_reports_hard.jsonl` | `{text, gold{...}}` | `minesense eval` |

Tips: hapus/arsipkan dokumen sintetis dari `data/corpus` sebelum ingest data asli agar jawaban tidak tercampur. Jika kategori hazard perusahaan berbeda, ubah `HAZARD_CATEGORIES`, `RECOMMENDED_ACTIONS`, dan `DOMAIN_LEXICON`.

## Siklus perbaikan berkelanjutan
1. Pantau dashboard: grounded rate, out_of_scope rate, feedback 👎.
2. Tarik feedback negatif: `sqlite3 artifacts/feedback/feedback.sqlite3 "select question, answer, comment from feedback where rating=-1"`.
3. Klasifikasikan penyebab: dokumen tidak ada (→ tambah dokumen), retrieval salah (→ tambah ke eval set, coba embedder neural/reranker), jawaban salah (→ perbaiki prompt / tambah contoh SFT).
4. Tambah kasus ke `data/eval/*`, jalankan `minesense eval`, naikkan ambang quality gate bila sudah stabil.
5. Fine-tune ulang LLM per kuartal bila data SFT bertambah signifikan.

## Troubleshooting
| Gejala | Penyebab umum | Tindakan |
|---|---|---|
| `/ready` 503 lama setelah start | Index dibangun ulang / classifier dilatih | Tunggu; cek `docker compose logs api` |
| "Index dibangun dengan embedder ..." | Ganti `MINESENSE_EMBEDDING_BACKEND` | Otomatis rebuild saat start; atau `minesense ingest` |
| Banyak jawaban "tidak ditemukan" | Gate terlalu ketat untuk embedder baru | Kalibrasi `MINESENSE_MIN_DENSE_SCORE`, `MINESENSE_MIN_QUERY_COVERAGE` |
| Warning "LLM utama tidak tersedia" | API key salah / Ollama mati / timeout | Cek `.env`, `docker compose ps ollama`, naikkan `MINESENSE_LLM_TIMEOUT_S` |
| 401 | Header `X-API-Key` hilang/salah | Pakai key dari `.env` |
| 429 | Rate limit | Naikkan `MINESENSE_RATE_LIMIT_PER_MINUTE` |
| Sertifikat HTTPS gagal | DNS belum mengarah / port 80 tertutup | Cek A record & firewall, `docker compose logs caddy` |
