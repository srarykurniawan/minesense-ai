# Panduan Step-by-Step: dari Laptop sampai Produksi

Urutan ini dirancang supaya setiap fase **terverifikasi dulu** sebelum lanjut. Jangan lompat fase. Perintah ditulis untuk **Windows PowerShell** (laptop) dan **bash** (VPS Linux).

| Fase | Tujuan | Di mana | Estimasi |
|---|---|---|---|
| 0 | Siapkan tools | Laptop | 30 mnt |
| 1 | Jalankan & uji lokal (tanpa Docker) | Laptop | 20 mnt |
| 2 | Simpan ke GitHub + CI hijau | Laptop + GitHub | 20 mnt |
| 3 | Uji stack Docker lokal | Laptop | 20 mnt |
| 4 | Fine-tuning model (opsional, disarankan) | Google Colab | 1–2 jam |
| 5 | Siapkan VPS + domain | Penyedia VPS + DNS | 30–60 mnt |
| 6 | Deploy produksi (HTTPS) | VPS | 20 mnt |
| 7 | Verifikasi produksi | Laptop/browser | 15 mnt |
| 8 | Aktifkan LLM generatif | VPS | 15 mnt |
| 9 | Monitoring & backup | VPS | 30 mnt |
| 10 | CI/CD otomatis ke VPS | GitHub | 30 mnt |
| 11 | Ganti data sintetis → data asli, lalu go-live | Laptop + VPS | berkelanjutan |

---

## Fase 0 — Prasyarat (laptop)
1. **Python 3.11** — python.org (centang *Add python.exe to PATH*).
2. **Git** — git-scm.com.
3. **Docker Desktop** — docker.com (aktifkan WSL2 bila diminta, restart).
4. Akun **GitHub** dan **Google** (untuk Colab).

Cek:
```powershell
python --version ; git --version ; docker --version ; docker compose version
```
✅ Selesai bila keempat perintah menampilkan versi.

## Fase 1 — Jalankan & uji lokal
```powershell
cd "D:\PROJECT DATA ENGINEERING FINAL\minesense"
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # bila diblokir: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev,ui,pdf]"
minesense bootstrap                    # data sintetis + index RAG + classifier + dataset SFT
pytest -q                              # harus: 40 passed
minesense eval                         # harus: QUALITY GATES: PASS
```
Jalankan aplikasi (2 terminal, keduanya sudah `Activate.ps1`):
```powershell
# Terminal 1
uvicorn minesense.api.main:app --port 8000
# Terminal 2
streamlit run ui/app.py
```
Buka http://localhost:8000/docs dan http://localhost:8501. Coba tanya: *"Berapa radius aman peledakan untuk manusia?"*

✅ Selesai bila: 40 tes lulus, quality gates PASS, UI menjawab dengan sitasi.

## Fase 2 — GitHub + CI
1. Di github.com buat repo **private** bernama `minesense` (tanpa README).
2. Push:
```powershell
git add -A
git commit -m "Setup awal MineSense"
git remote add origin https://github.com/<username>/minesense.git
git push -u origin main
```
3. Buka tab **Actions** → workflow *CI/CD* berjalan: `test` (lint, tes, evaluasi) lalu `build` (image ke GHCR).

✅ Selesai bila job `test` dan `build` hijau. Image terlihat di profil GitHub → *Packages*.

## Fase 3 — Uji Docker lokal
```powershell
copy .env.example .env
python -c "import secrets;print(secrets.token_urlsafe(32))"   # salin hasilnya
```
Edit `.env`: isi `MINESENSE_API_KEYS` dan `MINESENSE_UI_API_KEY` dengan key tadi, ubah `MINESENSE_ENV=dev`, `MINESENSE_CORS_ORIGINS=*`.
```powershell
docker compose up -d --build
docker compose ps                      # api harus (healthy)
docker compose logs -f api             # Ctrl+C untuk keluar
```
Buka http://localhost:8501. Matikan: `docker compose down`.

✅ Selesai bila `api` *healthy* dan UI menjawab. Masalah build di sini akan sama di VPS — perbaiki di sini dulu.

## Fase 4 — Fine-tuning di Colab (opsional, disarankan)
LLM QLoRA:
1. Buka colab.research.google.com → *Upload* `notebooks/01_finetune_llm_qlora_colab.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. Repo private → paling mudah: zip folder project (tanpa `.venv`), upload sebagai `/content/minesense.zip` (panel Files kiri). Notebook otomatis memakainya.
4. *Runtime → Run all* (±45–60 mnt). Hasil: `artifacts/gguf/minesense-qwen15b-q4_k_m.gguf` + `Modelfile`, tersalin ke Google Drive `MyDrive/minesense/`.
5. Unduh kedua file tersebut ke folder `models/` di project.

Classifier IndoBERT (opsional): jalankan `notebooks/02_finetune_incident_classifier_colab.ipynb` dengan cara yang sama; bandingkan `eval.json` dengan baseline.

✅ Selesai bila file `.gguf` dan `Modelfile` ada di `models/`, dan training loss/eval loss menurun.

## Fase 5 — VPS + domain
1. Sewa VPS **Ubuntu 22.04/24.04**: minimal 2 vCPU / 4 GB RAM / 40 GB (pakai Claude API); **4 vCPU / 8 GB** bila menjalankan model fine-tune via Ollama.
2. Punya domain → di panel DNS buat **A record**: `minesense.domainanda.com` → IP VPS. Cek dari laptop: `nslookup minesense.domainanda.com` (harus menampilkan IP VPS; bisa perlu beberapa menit–jam).
3. Login & buat user non-root:
```bash
ssh root@IP_VPS
adduser deploy && usermod -aG sudo deploy
# (disarankan) salin SSH key Anda lalu nonaktifkan login password
exit
ssh deploy@IP_VPS
```
4. Akses repo private dari VPS — pilih satu:
   - **Deploy key** (rekomendasi): di VPS `ssh-keygen -t ed25519 -C vps` → isi `~/.ssh/id_ed25519.pub` ke GitHub repo → *Settings → Deploy keys* (read-only). Repo URL: `git@github.com:<username>/minesense.git`.
   - Atau Personal Access Token (fine-grained, read-only contents).

✅ Selesai bila `nslookup` mengarah ke VPS dan `ssh -T git@github.com` dari VPS berhasil.

## Fase 6 — Deploy produksi
Di VPS:
```bash
git clone git@github.com:<username>/minesense.git ~/minesense
cd ~/minesense
bash scripts/deploy_vps.sh git@github.com:<username>/minesense.git minesense.domainanda.com
```
- Run pertama memasang Docker lalu berhenti → **logout, login lagi**, jalankan perintah yang sama sekali lagi.
- Skrip mencetak **API key** dan **password UI** satu kali → **simpan di password manager**.
- Skrip membuka firewall 22/80/443, build image, menjalankan api + ui + Caddy (HTTPS otomatis), dan smoke test.

✅ Selesai bila skrip mencetak `OK: https://minesense.domainanda.com`.

## Fase 7 — Verifikasi produksi
```powershell
$KEY="<api key dari fase 6>"
curl.exe https://minesense.domainanda.com/api/health
curl.exe -X POST https://minesense.domainanda.com/api/v1/chat -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{\"question\":\"Berapa radius aman peledakan untuk manusia?\"}'
curl.exe -o NUL -w "%{http_code}" https://minesense.domainanda.com/api/metrics   # harus 404
```
Checklist:
- [ ] `https://` dengan gembok valid
- [ ] UI meminta login (basic auth), lalu bisa chat, klasifikasi, ekstraksi, diagnosa
- [ ] `/api/docs` terbuka; request tanpa `X-API-Key` → 401
- [ ] Upload satu dokumen di tab *Dokumen* lalu tanyakan isinya → terjawab
- [ ] `docker compose ps` di VPS: semua *Up/healthy*

## Fase 8 — Aktifkan LLM generatif
Edit `~/minesense/.env` di VPS, pilih **satu**:

**A. Claude API** (kualitas terbaik, data dikirim ke API):
```
MINESENSE_LLM_PROVIDER=anthropic
MINESENSE_LLM_API_KEY=sk-ant-...
MINESENSE_LLM_MODEL=claude-sonnet-4-5
```
**B. Model fine-tune Anda via Ollama** (data tetap di server):
```bash
scp models/minesense-qwen15b-q4_k_m.gguf models/Modelfile deploy@IP_VPS:~/minesense/models/   # dari laptop
cd ~/minesense
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile ollama up -d
docker compose exec ollama ollama create minesense -f /models/Modelfile
```
`.env`:
```
MINESENSE_LLM_PROVIDER=ollama
MINESENSE_LLM_MODEL=minesense
MINESENSE_LLM_BASE_URL=http://ollama:11434/v1
```
Terapkan: `docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile ollama up -d`
Cek di UI sidebar: *LLM: anthropic/ollama*. Bandingkan kualitas: `docker compose exec api python -m minesense.finetune.eval_llm --limit 50` (butuh `data/sft`; jalankan `minesense sft` dulu bila belum ada).

✅ Selesai bila jawaban berbentuk kalimat natural dengan sitasi `[n]` dan tanpa warning fallback.

## Fase 9 — Monitoring & backup
```bash
echo "GRAFANA_PASSWORD=$(openssl rand -base64 16)" >> .env
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```
Dari laptop: `ssh -L 3000:localhost:3000 deploy@IP_VPS` → buka http://localhost:3000 (admin / password di atas) → dashboard *MineSense AI*.

Backup harian (crontab -e di VPS):
```
0 2 * * * cd ~/minesense && mkdir -p backup && docker run --rm -v minesense_uploads:/d -v $PWD/backup:/b alpine tar czf /b/uploads-$(date +\%F).tgz -C /d . && docker run --rm -v minesense_feedback:/d -v $PWD/backup:/b alpine tar czf /b/feedback-$(date +\%F).tgz -C /d .
```
Salin folder `backup/` ke object storage/Drive secara berkala.

## Fase 10 — CI/CD otomatis ke VPS
GitHub repo → *Settings*:
1. *Secrets and variables → Actions → Secrets*: `VPS_HOST` (IP), `VPS_USER` (`deploy`), `VPS_SSH_KEY` (private key yang boleh login ke VPS), `GHCR_READ_TOKEN` (PAT classic dengan scope `read:packages`).
2. *Variables*: `DEPLOY_ENABLED` = `true`.
3. *Environments* → buat `production` → aktifkan *Required reviewers* (deploy menunggu persetujuan Anda).

Alur setelahnya: `git push` → tes + evaluasi (gagal = berhenti) → build image → (setujui) → VPS menarik image ber-tag commit → health check.
Rollback: di VPS ubah `IMAGE_TAG=<sha lama>` di `.env` → `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-build`.

## Fase 11 — Data asli & go-live
1. Kumpulkan SOP/manual/regulasi resmi → `data/corpus/` (format front-matter di `docs/RUNBOOK.md`); arsipkan dokumen sintetis.
2. Ganti `incidents.csv`, `work_orders.csv`, `fault_codes.json` dengan data historis (anonimkan nama pekerja).
3. Minta pengawas/tim K3 menulis 50–100 pertanyaan nyata + dokumen acuannya → `data/eval/rag_qa.jsonl`.
4. `minesense ingest ; minesense train ; minesense eval` → analisis angka, kalibrasi ambang (`MINESENSE_MIN_DENSE_SCORE`, `MINESENSE_MIN_QUERY_COVERAGE`), pertimbangkan embedder `multilingual-e5`.
5. Fine-tune ulang LLM (Fase 4) dengan data SFT dari data asli.
6. Pilot 2–4 minggu dengan sekelompok kecil pengawas → pantau feedback 👎 dan grounded rate → perbaiki → rilis luas.

> Sistem ini alat bantu keputusan. Keputusan keselamatan (stop work, peledakan, evakuasi) tetap pada pengawas/KTT yang berwenang.
