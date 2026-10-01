"""Merge adapter LoRA ke base model, lalu (opsional) konversi ke GGUF untuk Ollama / llama.cpp (CPU-friendly).

    python -m minesense.finetune.merge_and_export --adapter artifacts/lora/<run>/adapter --gguf q4_k_m

Hasil:
  artifacts/merged/<name>/            — model HF (safetensors) → bisa diserve vLLM:  vllm serve artifacts/merged/<name>
  artifacts/gguf/<name>-<q>.gguf      — untuk Ollama/llama.cpp di CPU/VPS
  artifacts/gguf/Modelfile            — `ollama create minesense -f Modelfile`
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from minesense.config import get_settings

MODELFILE = """FROM ./{gguf}
TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
"""


def main() -> None:  # pragma: no cover - butuh torch + model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--name", default="minesense-llm")
    ap.add_argument("--gguf", default=None, help="kuantisasi GGUF, mis. q4_k_m / q8_0")
    ap.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"), help="path repo llama.cpp")
    a = ap.parse_args()

    s = get_settings()
    meta = json.loads((a.adapter.parent / "train_meta.json").read_text())
    base = meta["base"]
    merged_dir = s.artifacts_dir / "merged" / a.name
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("merged ->", merged_dir)

    if a.gguf:
        gdir = s.artifacts_dir / "gguf"
        gdir.mkdir(parents=True, exist_ok=True)
        f16 = gdir / f"{a.name}-f16.gguf"
        final = gdir / f"{a.name}-{a.gguf}.gguf"
        subprocess.run([sys.executable, str(a.llama_cpp / "convert_hf_to_gguf.py"), str(merged_dir),
                        "--outfile", str(f16), "--outtype", "f16"], check=True)
        quant_bin = next((p for p in [a.llama_cpp / "build/bin/llama-quantize", a.llama_cpp / "llama-quantize"]
                          if p.exists()), None)
        if quant_bin:
            subprocess.run([str(quant_bin), str(f16), str(final), a.gguf.upper()], check=True)
        else:
            final = f16
            print("llama-quantize tidak ditemukan; memakai GGUF f16")
        (gdir / "Modelfile").write_text(MODELFILE.format(gguf=final.name))
        print(f"GGUF -> {final}\nLanjutkan: cd {gdir} && ollama create minesense -f Modelfile")


if __name__ == "__main__":
    main()
