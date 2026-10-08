"""Fine-tuning transformer (IndoBERT / XLM-R) untuk klasifikasi insiden. Jalankan di GPU (Colab/Kaggle T4).

    pip install -e ".[train]"
    python -m minesense.classifier.train_transformer --target category --model indobenchmark/indobert-base-p1
    python -m minesense.classifier.train_transformer --target risk_level

Output: artifacts/models/incident-<target>/ (bisa di-push ke HF Hub dengan --push-to-hub user/repo).
Model ini dipakai API bila MINESENSE_CLASSIFIER_BACKEND=transformers (lihat classifier/hf_backend.py).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from minesense.config import get_settings


def main() -> None:  # pragma: no cover - butuh GPU & HF
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["category", "risk_level"], default="category")
    ap.add_argument("--model", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--push-to-hub", default=None)
    a = ap.parse_args()

    s = get_settings()
    rows = list(csv.DictReader(open(s.data_dir / "incidents.csv", encoding="utf-8")))
    labels = sorted({r[a.target] for r in rows})
    l2i = {lab: i for i, lab in enumerate(labels)}
    groups = [r.get("template_id", r["incident_id"]) for r in rows]
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(rows, groups=groups))

    tok = AutoTokenizer.from_pretrained(a.model)

    def to_ds(idx):
        ds = Dataset.from_dict({"text": [rows[i]["description"] for i in idx],
                                "label": [l2i[rows[i][a.target]] for i in idx]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

    train_ds, eval_ds = to_ds(tr_i), to_ds(te_i)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"), "acc": float((pred == p.label_ids).mean())}

    out_dir = s.artifacts_dir / "models" / f"incident-{a.target}"
    args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=32, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(), logging_steps=20, report_to="none", save_total_limit=1, seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    res = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(res)
    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub)
        tok.push_to_hub(a.push_to_hub)


if __name__ == "__main__":
    main()
