"""Fine-tuning LLM dengan QLoRA (TRL SFTTrainer). Target: GPU gratis Colab/Kaggle (T4 16GB).

    pip install -e ".[train]"
    python -m minesense.finetune.build_sft_dataset
    python -m minesense.finetune.train_lora --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2

Pilihan base model (multilingual, lisensi permisif, muat di T4):
  - Qwen/Qwen2.5-1.5B-Instruct  (default, ~25 menit/epoch di T4 untuk ±1.200 contoh)
  - Qwen/Qwen2.5-3B-Instruct    (lebih baik, butuh 4-bit)
  - google/gemma-2-2b-it, meta-llama/Llama-3.2-3B-Instruct (cek lisensi)
Output: artifacts/lora/<run>/adapter  → lanjut ke merge_and_export.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path

from minesense.config import get_settings

UPSAMPLE = {"grounded_qa": 3, "refusal": 2}  # tugas kecil tapi krusial di-upsample


def load_jsonl(p: Path, upsample: bool) -> list[dict]:
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    if upsample:
        rows = [r for r in rows for _ in range(UPSAMPLE.get(r["task"], 1))]
        random.Random(0).shuffle(rows)
    return rows


def _filtered(cls, **kw):
    """Kompatibilitas lintas versi TRL/transformers: buang argumen yang tidak dikenal."""
    params = inspect.signature(cls.__init__).parameters
    if "max_seq_length" in kw and "max_seq_length" not in params and "max_length" in params:
        kw["max_length"] = kw.pop("max_seq_length")
    return cls(**{k: v for k, v in kw.items() if k in params})


def main() -> None:  # pragma: no cover - butuh GPU
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M"))
    ap.add_argument("--push-to-hub", default=None, help="mis. username/minesense-qwen2.5-1.5b-lora")
    a = ap.parse_args()

    s = get_settings()
    out = s.artifacts_dir / "lora" / a.run_name
    train = Dataset.from_list([{"messages": r["messages"]} for r in load_jsonl(s.data_dir / "sft" / "train.jsonl", True)])
    val = Dataset.from_list([{"messages": r["messages"]} for r in load_jsonl(s.data_dir / "sft" / "val.jsonl", False)])

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    quant = None if a.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=quant, device_map="auto",
                                                 torch_dtype=torch.bfloat16 if bf16 else torch.float16)
    model.config.use_cache = False

    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = _filtered(
        SFTConfig, output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(), gradient_checkpointing=True,
        max_seq_length=a.max_len, packing=False, report_to="none", seed=42,
        assistant_only_loss=False,  # set True di TRL ≥0.20 bila chat template mendukung {% generation %}
    )
    trainer = _filtered(SFTTrainer, model=model, args=cfg, train_dataset=train, eval_dataset=val,
                        peft_config=lora, processing_class=tok, tokenizer=tok)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    (out / "train_meta.json").write_text(json.dumps({"base": a.base, "args": vars(a), "eval": metrics}, indent=2))
    print("eval:", metrics, "\nadapter:", out / "adapter")
    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub)
        tok.push_to_hub(a.push_to_hub)


if __name__ == "__main__":
    main()
