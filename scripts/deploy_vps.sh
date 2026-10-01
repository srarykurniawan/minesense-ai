#!/usr/bin/env bash
# Deploy MineSense ke VPS Ubuntu 22.04/24.04 (min. 2 vCPU, 4 GB RAM; 8 GB bila memakai Ollama).
# Pemakaian (di VPS, sebagai user dengan sudo):
#   curl -fsSL https://raw.githubusercontent.com/<user>/minesense/main/scripts/deploy_vps.sh | bash -s -- <repo_url> <domain>
#   atau: bash scripts/deploy_vps.sh https://github.com/<user>/minesense.git minesense.perusahaan.co.id
set -euo pipefail
REPO_URL="${1:?repo url}"
DOMAIN="${2:?domain}"
APP_DIR="$HOME/minesense"

echo "==> 1. Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "Docker terpasang. Logout/login lalu jalankan ulang skrip ini."; exit 0
fi

echo "==> 2. Firewall (22, 80, 443)"
if command -v ufw >/dev/null; then
  sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw --force enable
fi

echo "==> 3. Kode"
if [ -d "$APP_DIR/.git" ]; then git -C "$APP_DIR" pull --ff-only; else git clone "$REPO_URL" "$APP_DIR"; fi
cd "$APP_DIR"

echo "==> 4. .env"
if [ ! -f .env ]; then
  cp .env.example .env
  KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
  UIPASS=$(python3 -c "import secrets;print(secrets.token_urlsafe(12))")
  HASH=$(docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "$UIPASS" | sed 's/\$/$$/g')
  sed -i "s|^MINESENSE_API_KEYS=.*|MINESENSE_API_KEYS=$KEY|; s|^MINESENSE_UI_API_KEY=.*|MINESENSE_UI_API_KEY=$KEY|" .env
  sed -i "s|^DOMAIN=.*|DOMAIN=$DOMAIN|; s|^MINESENSE_CORS_ORIGINS=.*|MINESENSE_CORS_ORIGINS=https://$DOMAIN|" .env
  sed -i "s|^UI_BASIC_AUTH_HASH=.*|UI_BASIC_AUTH_HASH=$HASH|" .env
  echo "------------------------------------------------------------"
  echo " API key       : $KEY"
  echo " UI login      : admin / $UIPASS"
  echo " SIMPAN kredensial ini. Edit .env untuk mengaktifkan LLM (anthropic/ollama)."
  echo "------------------------------------------------------------"
fi

echo "==> 5. Build & jalankan (pastikan DNS A record $DOMAIN → IP VPS ini)"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

echo "==> 6. Smoke test"
for i in $(seq 1 30); do
  if curl -fsS "https://$DOMAIN/api/health" >/dev/null 2>&1; then echo "OK: https://$DOMAIN"; exit 0; fi
  sleep 5
done
echo "Health check belum lolos — cek: docker compose logs --tail=100 api caddy"; exit 1
