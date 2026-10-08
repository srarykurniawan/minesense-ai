"""Entry point Streamlit Community Cloud (mode embedded: API + UI dalam satu proses).

Di share.streamlit.io pilih file ini sebagai "Main file path". Lokal: `streamlit run streamlit_app.py`.
Secrets (Settings → Secrets di Streamlit Cloud) dipetakan ke environment variable, contoh:
    MINESENSE_LLM_PROVIDER = "anthropic"
    MINESENSE_LLM_API_KEY  = "sk-ant-..."
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # tanpa perlu `pip install -e .`

try:  # secrets → env (harus sebelum minesense diimpor karena config di-cache)
    import streamlit as st

    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))
except Exception:  # tidak ada secrets.toml → abaikan
    pass

os.environ.setdefault("MINESENSE_API_URL", "embedded")
os.environ.setdefault("MINESENSE_DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("MINESENSE_ARTIFACTS_DIR", str(ROOT / "artifacts"))
os.environ.setdefault("MINESENSE_RATE_LIMIT_PER_MINUTE", "600")

runpy.run_path(str(ROOT / "ui" / "app.py"), run_name="__main__")
