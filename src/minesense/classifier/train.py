"""Training + evaluasi classifier insiden (sklearn baseline). Jalankan: python -m minesense.classifier.train"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from minesense.classifier.model import IncidentClassifier, _apply_safety_overrides
from minesense.config import get_settings


def load_incidents(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _eval(clf: IncidentClassifier, rows: list[dict]) -> dict:
    texts = [r["description"] for r in rows]
    pc, pr = clf.predict_proba(texts)
    cat_pred = clf.cat_model.classes_[pc.argmax(1)]
    risk_pred = [_apply_safety_overrides(t, r) for t, r in zip(texts, clf.risk_model.classes_[pr.argmax(1)])]
    y_c, y_r = [r["category"] for r in rows], [r["risk_level"] for r in rows]
    # recall kelas kritis paling penting secara bisnis (false negative = bahaya)
    crit = [i for i, y in enumerate(y_r) if y == "kritis"]
    return {
        "category_macro_f1": round(f1_score(y_c, cat_pred, average="macro"), 4),
        "risk_macro_f1": round(f1_score(y_r, risk_pred, average="macro"), 4),
        "risk_kritis_recall": round(sum(risk_pred[i] == "kritis" for i in crit) / max(1, len(crit)), 4),
        "n_test": len(rows),
        "category_report": classification_report(y_c, cat_pred, output_dict=True, zero_division=0),
        "risk_report": classification_report(y_r, risk_pred, output_dict=True, zero_division=0),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    s = get_settings()
    rows = load_incidents(args.data or s.data_dir / "incidents.csv")

    # 1) split acak (stratified)
    tr, te = train_test_split(rows, test_size=0.2, random_state=args.seed, stratify=[r["category"] for r in rows])
    clf = IncidentClassifier().fit([r["description"] for r in tr], [r["category"] for r in tr], [r["risk_level"] for r in tr])
    random_split = _eval(clf, te)

    # 2) split per template (uji generalisasi ke pola kalimat yang tidak pernah dilihat)
    groups = [r.get("template_id", r["incident_id"]) for r in rows]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    tr_i, te_i = next(gss.split(rows, groups=groups))
    clf_g = IncidentClassifier().fit([rows[i]["description"] for i in tr_i], [rows[i]["category"] for i in tr_i],
                                     [rows[i]["risk_level"] for i in tr_i])
    group_split = _eval(clf_g, [rows[i] for i in te_i])

    # 3) model final dilatih pada seluruh data
    final = IncidentClassifier().fit([r["description"] for r in rows], [r["category"] for r in rows],
                                     [r["risk_level"] for r in rows])
    final.meta["eval_random_split"] = {k: v for k, v in random_split.items() if not k.endswith("report")}
    final.meta["eval_group_split"] = {k: v for k, v in group_split.items() if not k.endswith("report")}
    final.save(s.classifier_path)

    report = {"random_split": random_split, "group_split": group_split, "model_path": str(s.classifier_path)}
    out = s.artifacts_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"random_split": final.meta["eval_random_split"], "group_split": final.meta["eval_group_split"]}
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    main()
