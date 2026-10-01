# ⛏️ MineSense AI

Platform **AI Engineering end-to-end untuk industri pertambangan**: RAG atas SOP/K3/regulasi/manual alat berat, klasifikasi laporan insiden, ekstraksi laporan produksi shift, diagnosa maintenance, serta pipeline **fine-tuning** (QLoRA LLM + IndoBERT), evaluasi otomatis dengan *quality gates*, dan deployment produksi (Docker, HTTPS, CI/CD, monitoring).

> **Data di repo ini SINTETIS** (perusahaan fiktif "PT Nusantara Mineral Sejahtera", model alat fiktif DT-100/EX-2000/BD-375). Arsitektur dan pipeline siap pakai; ganti data dengan dokumen dan histori asli perusahaan Anda (lihat [docs/RUNBOOK.md](docs/RUNBOOK.md#mengganti-data-sintetis-dengan-data-asli)).

## Kemampuan

| Use case | Endpoint | Teknik |
|---|---|---|
| Asisten K3 & SOP (tanya-jawab bersitasi) | `POST /v1/chat` | Hybrid retrieval (dense + BM25 → RRF), gate out-of-scope, guardrails, sitasi tervalidasi |
| Klasifikasi insiden (kategori hazard + risiko) | `POST /v1/incidents/classify` | TF-IDF + leksikon domain + LogReg (CPU) **atau** IndoBERT fine-tuned; *safety override* deterministik |
| Ekstraksi laporan produksi shift → JSON | `POST /v1/reports/extract` | Parser aturan + LLM JSON-mode, validasi Pydantic & konsistensi bisnis (SR = OB/coal) |
| Diagnosa maintenance alat berat | `POST /v1/maintenance/diagnose` | Inferensi kode gangguan + similarity work order + RAG manual |
| Router intent (satu kotak chat untuk semua) | `POST /v1/chat` | Rule-based router → tool yang sesuai |
| Ingest dokumen baru (.md/.txt/.pdf) | `POST /v1/documents` | Chunking sadar heading, re-index inkremental, hot-reload lintas worker |
| Fine-tuning LLM domain | `notebooks/01_*.ipynb` | QLoRA (TRL) Qwen2.5-1.5B di Colab T4 gratis → merge → GGUF → Ollama |

## Arsitektur

```
             ┌──────────── Streamlit UI (8501) ─────────────┐
 Pengguna ──▶│ Caddy (HTTPS, basic auth UI, /api → API)     │
             └───────────────────────┬──────────────────────┘
                                     ▼
 ┌──────────────────────── FastAPI (8000) ───────────────────────────┐
 │ auth X-API-Key · rate limit · request-id · Prometheus /metrics     │
 │  Router intent ─┬─ RAG pipeline: guard input → hybrid retrieve →   │
 │                 │   gate relevansi → LLM → validasi sitasi/grounding│
 │                 ├─ Incident classifier (sklearn | transformers)     │
 │                 ├─ Shift-report extractor (rules + LLM)             │
 │                 └─ Maintenance assistant (fault code + WO + manual) │
 └───────┬───────────────────┬──────────────────────┬────────────────┘
         ▼                   ▼                      ▼
  Index (memori/Qdrant)   LLM provider pluggable   SQLite feedback
  + BM25                  extractive | Claude | OpenAI-compat | Ollama (model fine-tune)
```
Detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Mulai cepat (lokal, CPU, tanpa API key)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,ui,pdf]"
minesense bootstrap        # data sintetis + index RAG + classifier + dataset SFT
pytest -q                  # 40 tes
minesense eval             # laporan evaluasi + quality gates → artifacts/reports/
make api                   # http://localhost:8000/docs
make ui                    # http://localhost:8501   (terminal lain)
minesense ask "Berapa radius aman peledakan untuk manusia?"
```

Tanpa API key, LLM berjalan dalam mode **extractive** (jawaban disusun dari kalimat dokumen + sitasi). Aktifkan LLM generatif di `.env`:

```bash
MINESENSE_LLM_PROVIDER=anthropic   MINESENSE_LLM_API_KEY=sk-ant-...   MINESENSE_LLM_MODEL=claude-sonnet-4-5
# atau model fine-tune Anda sendiri via Ollama:
MINESENSE_LLM_PROVIDER=ollama      MINESENSE_LLM_MODEL=minesense
```

## Docker

```bash
cp .env.example .env               # isi API key dll.
docker compose up -d --build       # api:8000 + ui:8501
docker compose --profile ollama --profile monitoring up -d   # + Ollama, Prometheus, Grafana
```

## Deploy produksi
Tiga jalur, lihat [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md):
1. **VPS + Docker Compose + Caddy (HTTPS otomatis)** — `scripts/deploy_vps.sh <repo> <domain>` (rekomendasi).
2. **CI/CD GitHub Actions** — lint → tes → evaluasi (quality gates) → build image ke GHCR → deploy SSH ke VPS.
3. **Hugging Face Spaces (gratis, demo)** — `docker/Dockerfile.spaces`.

## Fine-tuning
| Model | Dimana | Waktu | Perintah |
|---|---|---|---|
| Classifier baseline | CPU lokal | ~6 dtk | `minesense train` |
| IndoBERT classifier | Colab T4 | ~10 mnt | `notebooks/02_finetune_incident_classifier_colab.ipynb` |
| LLM QLoRA (Qwen2.5-1.5B) | Colab/Kaggle T4 | ~45–60 mnt | `notebooks/01_finetune_llm_qlora_colab.ipynb` |

Evaluasi base vs fine-tuned: `python -m minesense.finetune.eval_llm` (metrik JSON-validity, akurasi kategori/risiko, field accuracy, citation rate, refusal accuracy). Lihat [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Hasil evaluasi (data sintetis, mode extractive, embedder hashing)

| Komponen | Metrik | Nilai |
|---|---|---|
| Retrieval hybrid | Hit@1 / Hit@5 / MRR | 0,95 / 1,00 / 0,97 |
| Retrieval dense saja | Hit@1 / Hit@5 | 0,85 / 0,98 |
| RAG | keyword recall · sitasi benar · tolak out-of-scope · blok injection | 0,93 · 0,98 · 1,00 · 1,00 |
| Classifier (pola kalimat tak terlihat / group split) | macro-F1 kategori · macro-F1 risiko · recall "kritis" | 0,95 · 0,90 · 1,00 |
| Ekstraksi laporan shift (hard set) | field accuracy | 1,00 |

⚠️ **Baca dengan kritis:** angka setinggi ini wajar untuk data sintetis yang berbasis template dan eval set yang kecil (41 pertanyaan, 4 laporan "hard"). Pada data nyata, angka **akan turun** — itulah alasan harness evaluasi & quality gates disertakan: jalankan ulang pada data asli dan kalibrasi ambang. Lihat bagian "Keterbatasan" di MODEL_CARD.

## Struktur repo
```
src/minesense/
  rag/           chunking, embeddings, vectorstore, index (hybrid), retriever (RRF), pipeline
  llm/           providers (extractive/anthropic/openai/ollama), prompts
  classifier/    model sklearn, train, train_transformer (IndoBERT), hf_backend
  extraction/    shift_report (rules + LLM)
  maintenance/   diagnose (fault code + WO similarity + manual RAG)
  agent/         router intent
  finetune/      build_sft_dataset, train_lora (QLoRA), merge_and_export (GGUF), eval_llm
  eval/          run_all (metrik + quality gates)
  data/          synth (generator data sintetis)
  api/           FastAPI
  guardrails.py  services.py  config.py  cli.py
data/corpus/     17 dokumen SOP/manual/regulasi (sintetis)
ui/              Streamlit
docker/ deploy/ scripts/ .github/workflows/ notebooks/ tests/ docs/
```
