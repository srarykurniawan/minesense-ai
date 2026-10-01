"""Backend inferensi classifier memakai model transformer hasil fine-tune (train_transformer.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from minesense.classifier.model import RECOMMENDED_ACTIONS, _apply_safety_overrides
from minesense.schemas import IncidentPrediction


class _Head:  # pragma: no cover - butuh torch
    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.classes_ = np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            return self.torch.softmax(self.model(**enc).logits, -1).numpy()


class TransformerIncidentClassifier:  # pragma: no cover
    def __init__(self, models_dir: Path):
        self.cat = _Head(models_dir / "incident-category")
        self.risk = _Head(models_dir / "incident-risk_level")
        self.meta = {"backend": "transformers", "path": str(models_dir)}

    def predict(self, text: str) -> IncidentPrediction:
        pc, pr = self.cat.predict_proba([text])[0], self.risk.predict_proba([text])[0]
        ci, ri = int(pc.argmax()), int(pr.argmax())
        cat = str(self.cat.classes_[ci])
        risk = _apply_safety_overrides(text, str(self.risk.classes_[ri]))
        top = sorted(zip(self.cat.classes_, pc), key=lambda x: -x[1])[:3]
        return IncidentPrediction(category=cat, category_confidence=round(float(pc[ci]), 4), risk_level=risk,
                                  risk_confidence=round(float(pr[ri]), 4),
                                  top_categories=[(str(c), round(float(p), 4)) for c, p in top],
                                  recommended_actions=RECOMMENDED_ACTIONS.get(cat, []),
                                  requires_escalation=risk in ("tinggi", "kritis") or float(pc[ci]) < 0.5)
