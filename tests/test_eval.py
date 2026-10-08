import json

from minesense.config import get_settings
from minesense.eval.judge import build_judge, judge_one
from minesense.eval.rag_eval import bootstrap_ci, contains_all, evaluate, headline, load_set, norm
from minesense.llm.providers import LLM
from minesense.rag.calibration import best_threshold, calibrate, resolve_dense_threshold


def test_norm_handles_indonesian_number_and_dash_formats():
    assert contains_all("Tekanan pilot normal 30-35 bar.", ["30–35 bar"])
    assert contains_all("minimal 1.8 m dari tanah", ["1,8"])
    assert contains_all("**Body prop** wajib", ["body prop"])
    assert norm("50 °C") == norm("50°c")


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert lo <= mean <= hi and 0 <= lo and hi <= 1


def test_best_threshold_separates():
    t, acc = best_threshold([0.6, 0.7, 0.8], [0.2, 0.3, 0.4])
    assert 0.4 < t <= 0.6 and acc == 1.0


def test_eval_set_v2_is_valid():
    items = load_set(get_settings().data_dir / "eval" / "rag_qa_v2.jsonl")
    types = {i["type"] for i in items}
    assert {"paraphrase", "colloquial", "scenario", "unanswerable", "out_of_scope"} <= types
    assert len(items) >= 100
    assert all(i["keywords"] for i in items if i["type"] in ("paraphrase", "colloquial", "scenario"))


def test_evaluate_produces_metrics_with_ci(rag):
    items = load_set(get_settings().data_dir / "eval" / "rag_qa_v2.jsonl")
    sample = [i for i in items if i["type"] == "paraphrase"][:8] + [i for i in items if i["type"] == "out_of_scope"][:3]
    rep = evaluate(rag, sample)
    h = headline(rep)
    assert 0 <= h["hit@5"] <= 1 and h["refusal_rate"] == 1.0
    assert rep["answerable"]["hit@5"]["ci95"][0] <= rep["answerable"]["hit@5"]["mean"]


def test_calibration_written_and_resolved(index):
    s = get_settings()
    v1 = [json.loads(x)["question"] for x in open(s.data_dir / "eval" / "rag_qa.jsonl", encoding="utf-8")]
    entry = calibrate(index, s, v1)
    assert entry["balanced_accuracy"] > 0.8
    assert resolve_dense_threshold(s, "hashing") == entry["min_dense_score"]
    assert resolve_dense_threshold(s, "fastembed:unknown-model") == 1.01  # belum dikalibrasi → nonaktif
    s.calibration_path.unlink()


class FakeJudge(LLM):
    name = "fake-judge"

    def generate(self, *a, **k):
        return '```json\n{"faithfulness": 1, "relevance": 0.9, "correctness": 1, "unsupported_claims": []}\n```'


def test_judge_parsing_and_disabled_for_extractive():
    j = judge_one(FakeJudge(), "q", ["ctx"], "jawaban", ["500 meter"])
    assert j["faithfulness"] == 1.0 and j["correctness"] == 1.0
    assert build_judge(get_settings()) is None  # provider extractive → juri nonaktif
