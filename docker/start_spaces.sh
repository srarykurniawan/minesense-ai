#!/usr/bin/env bash
# API di background (localhost), Streamlit di port publik 7860.
set -e
uvicorn minesense.api.main:app --host 127.0.0.1 --port 8000 &
for i in $(seq 1 60); do curl -fs http://127.0.0.1:8000/ready >/dev/null && break; sleep 1; done
exec streamlit run ui/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true \
  --browser.gatherUsageStats=false --client.toolbarMode=minimal
