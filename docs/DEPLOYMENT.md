# Panduan Deployment MineSense AI

## Pilihan 1 — VPS + Docker Compose + Caddy (rekomendasi produksi)

**Spesifikasi minimum:** Ubuntu 22.04/24.04, 2 vCPU, 4 GB RAM, 30 GB disk (mode extractive/Claude API).
Bila menjalankan model fine-tune lokal via Ollama di CPU: 4 vCPU, 8 GB RAM (Qwen2.5-1.5B Q4 ≈ 1,1 GB RAM, ~5–15 token/detik di CPU).

### Langkah
1. **DNS**: buat A record `minesense.perusahaan.co.id` → IP publik VPS.
2. **Push repo** ke GitHub (privat disarankan).
3. Di VPS:
   ```bash
   git clone https://github.com/<user>/minesense.git && cd minesense
   bash scripts/deploy_vps.sh https://github.com/<user>/minesense.git minesense.perusahaan.co.id
   ```
   Skrip: memasang Docker, firewall (22/80/443), membuat `.env` dengan API key acak + password UI (dicetak sekali — simpan), build image, menjalankan stack, dan smoke test HTTPS.
4. Akses:
   - UI: `https://<domain>/` (basic auth `admin` / password dari skrip)
   - API: `https://<domain>/api/docs` — header `X-API-Key: <key>`
   ```bash
   curl -s https://<domain>/api/v1/chat -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
        -d '{"question":"Berapa radius aman peledakan untuk manusia?"}'
   ```

### Mengaktifkan LLM
Edit `.env` lalu `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`:
- **Claude API**: `MINESENSE_LLM_PROVIDER=anthropic`, `MINESENSE_LLM_API_KEY=...`, `MINESENSE_LLM_MODEL=<model>`.
- **Model fine-tune Anda (Ollama)**:
  ```bash
  cp minesense-qwen15b-q4_k_m.gguf Modelfile models/        # hasil notebook 01
  docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile ollama up -d
  docker compose exec ollama ollama create minesense -f /models/Modelfile
  # .env: MINESENSE_LLM_PROVIDER=ollama  MINESENSE_LLM_MODEL=minesense  MINESENSE_LLM_BASE_URL=http://ollama:11434/v1
  ```
- **vLLM (server GPU)**: jalankan `vllm serve artifacts/merged/minesense-qwen15b --served-model-name minesense`, set `MINESENSE_LLM_PROVIDER=openai`, `MINESENSE_LLM_BASE_URL=http://<host>:8000/v1`.

> Data sensitif (laporan insiden, nama pekerja) tidak boleh keluar perusahaan? Gunakan Ollama/vLLM self-hosted — semua inferensi tetap di server Anda.

### Monitoring
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 user@vps     # jangan ekspos Grafana ke publik
```
Dashboard "MineSense AI" otomatis ter-provision (request rate, p95 latency, error 5xx, intent, grounded rate, feedback). Alert rules di `deploy/alerts.yml` (error rate > 5%, p95 chat > 10 dtk, grounded rate < 80%, lonjakan feedback negatif). Hubungkan Alertmanager/Grafana alerting ke email/Telegram/Slack sesuai kebutuhan.

### Backup
Volume penting: `minesense_uploads` (dokumen unggahan), `minesense_feedback` (SQLite feedback), `minesense_index`.
```bash
docker run --rm -v minesense_uploads:/d -v $PWD/backup:/b alpine tar czf /b/uploads-$(date +%F).tgz -C /d .
docker run --rm -v minesense_feedback:/d -v $PWD/backup:/b alpine tar czf /b/feedback-$(date +%F).tgz -C /d .
```
Jadwalkan via cron harian dan salin ke object storage.

### Update versi
```bash
git pull && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```
Index dibangun ulang otomatis bila embedder berubah. Bila korpus di `data/corpus` berubah, hapus volume index agar dibangun ulang dari image baru: `docker volume rm minesense_index` (setelah `down`).

## Pilihan 2 — CI/CD GitHub Actions → GHCR → VPS
`.github/workflows/ci.yml`:
1. `test`: ruff, pytest, `minesense eval` (**gagal bila quality gate tidak lolos** — mencegah regresi kualitas AI masuk produksi). Laporan evaluasi diunggah sebagai artifact.
2. `build`: image `api` & `ui` → `ghcr.io/<owner>/<repo>-{api,ui}:<sha>`.
3. `deploy` (opsional, aktif bila repo variable `DEPLOY_ENABLED=true`): SSH ke VPS, pull image ber-tag SHA, `up -d`, health check.

Secrets yang diperlukan: `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `GHCR_READ_TOKEN` (PAT dengan scope `read:packages`). Buat environment `production` dengan *required reviewers* agar deploy butuh persetujuan.

Rollback: set `IMAGE_TAG=<sha sebelumnya>` di `.env` VPS lalu `docker compose ... up -d --no-build`.

## Pilihan 3 — Hugging Face Spaces (demo gratis)
1. Buat Space baru → SDK **Docker** → hardware CPU basic.
2. Push isi repo ke Space, dan salin `docker/Dockerfile.spaces` menjadi `Dockerfile` di root Space.
3. Tambahkan header berikut di awal `README.md` Space:
   ```yaml
   ---
   title: MineSense AI
   sdk: docker
   app_port: 7860
   ---
   ```
4. (Opsional) Settings → Secrets: `MINESENSE_LLM_PROVIDER`, `MINESENSE_LLM_API_KEY`.
Catatan: storage Space bersifat sementara (dokumen unggahan hilang saat restart) — hanya untuk demo, jangan untuk data perusahaan.

## Checklist keamanan produksi
- [ ] `MINESENSE_API_KEYS` diisi (tanpa ini auth nonaktif) dan dirotasi berkala.
- [ ] `MINESENSE_CORS_ORIGINS` dibatasi ke domain Anda.
- [ ] UI di balik basic auth / SSO (mis. Cloudflare Access, oauth2-proxy).
- [ ] `/metrics`, Grafana, Prometheus tidak terekspos publik (sudah diblok di Caddyfile).
- [ ] Backup volume terjadwal.
- [ ] Bila memakai LLM API eksternal: pastikan kebijakan data perusahaan mengizinkan; PII (NIK, telepon, email) sudah disamarkan otomatis oleh guardrail sebelum dikirim.
- [ ] Rate limit per proses — untuk >1 server, pindahkan ke API gateway/Redis.
