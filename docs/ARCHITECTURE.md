# Arsitektur & Keputusan Desain

## Alur permintaan `/v1/chat`
1. **Auth & rate limit** (X-API-Key, sliding window 60 dtk).
2. **Router intent** (`agent/router.py`): `qa_sop` | `incident_report` | `shift_report` | `maintenance`. Rule-based — deterministik, murah, mudah diaudit. Dapat diganti classifier/LLM tool-calling bila pola pesan makin beragam.
3. **Guardrail input**: blok prompt-injection, samarkan PII (NIK 16 digit, telepon, email).
4. **Hybrid retrieval**: dense (vector store) + BM25, masing-masing `candidate_k=20`, digabung **Reciprocal Rank Fusion** (k=60), opsional cross-encoder reranker (`bge-reranker-v2-m3`).
5. **Gate relevansi (out-of-scope)**: lolos bila cosine dense ≥ ambang ATAU (BM25 ≥ ambang DAN ≥51% token query — setelah stemming ringan — muncul di konteks). Mencegah jawaban asal untuk pertanyaan di luar domain.
6. **Generasi** dengan prompt bersitasi `[n]`; konteks dokumen diperlakukan sebagai data (mitigasi indirect prompt injection).
7. **Validasi output**: sitasi di luar rentang dihapus, skor grounding (proporsi kalimat yang didukung konteks), footer keselamatan untuk topik safety-critical.
8. **Fallback**: bila LLM gagal (timeout/5xx, retry eksponensial 3x), jawaban disusun ekstraktif → layanan tetap tersedia (graceful degradation).

## Keputusan desain utama
| Keputusan | Alasan | Alternatif |
|---|---|---|
| Hybrid BM25 + dense | Dokumen tambang penuh kode/angka (E-BRK-301, 22 kN, 1827 K/30/MEM/2018) — BM25 unggul untuk exact match; dense untuk parafrase. Eval: hybrid MRR 0,97 vs dense 0,90 | Dense saja |
| Embedder `hashing` default | Jalan offline di CPU kecil, tanpa download model, deterministik untuk CI | `multilingual-e5-small/base`, `bge-m3` (set `MINESENSE_EMBEDDING_BACKEND=sentence-transformers`) — disarankan untuk produksi dengan data nyata |
| Vector store in-memory + persist | Korpus SOP perusahaan biasanya < 50 ribu chunk → numpy cukup cepat (<5 ms) | Qdrant (`MINESENSE_VECTOR_BACKEND=qdrant`) untuk korpus besar / multi-server |
| Chunking per heading + sliding window | SOP terstruktur per bagian; chunk mempertahankan judul & section → sitasi informatif | Fixed-size murni |
| Classifier sklearn + leksikon domain | Latih ulang < 10 dtk di CPU; leksikon menaikkan macro-F1 pada pola kalimat baru dari 0,30 → 0,95 | IndoBERT fine-tune (notebook 02) |
| Safety override deterministik | Model ML tidak boleh menurunkan risiko kejadian fatal/LTI (kata "meninggal", "patah tulang", "misfire", ...) | Murni model |
| Ekstraksi rules + LLM | Rules: gratis, deterministik, akurat untuk format umum; LLM menangani format liar; validasi konsistensi SR | LLM saja |
| LLM provider pluggable | Data sensitif → self-hosted (Ollama/vLLM); kualitas maksimum → API komersial | Terikat satu vendor |
| Fine-tune QLoRA model kecil | Muat di GPU gratis T4, bisa diserve di CPU via GGUF; mengajarkan format sitasi, penolakan, JSON | Full fine-tune (mahal), hanya prompting |

## Konsistensi multi-worker
Uvicorn `--workers 2`: setiap worker memegang index sendiri. Saat dokumen diunggah, worker penerima menyimpan index; worker lain mendeteksi perubahan `manifest.json` (cek tiap 10 dtk) dan memuat ulang. Metrik Prometheus diagregasi via `PROMETHEUS_MULTIPROC_DIR`.

## Skalabilitas ke depan
- >1 server: Qdrant + BM25 di Qdrant sparse vectors / OpenSearch, rate limit di gateway/Redis, feedback ke PostgreSQL.
- Antrian ingest (Celery/RQ) untuk PDF besar + OCR.
- Tracing LLM (Langfuse/OpenTelemetry) untuk audit percakapan.
