from fastapi.testclient import TestClient

from app.main import app
from app.metrics import compare_runs, drift_score, summarize
from app.runner import load_golden, run_evaluation
import os

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "golden_dataset.jsonl")


def test_golden_dataset_evaluates():
    report = run_evaluation(load_golden(GOLDEN))
    assert report["count"] == 6
    assert report["mean"] > 0.3
    assert report["judge_mode"] == "heuristic"


def test_regression_comparison_flags_drop():
    base = {"mean": 0.9, "per_case": {"a": 0.9, "b": 0.8}}
    cand = {"mean": 0.6, "per_case": {"a": 0.5, "b": 0.8}}
    out = compare_runs(base, cand)
    assert out["regressed"] is True
    assert out["regressed_cases"] == ["a"]


def test_drift_low_for_similar_high_for_shifted():
    assert drift_score([0.8, 0.85, 0.9, 0.82], [0.81, 0.86, 0.88, 0.83]) < 0.2
    assert drift_score([0.9, 0.92, 0.88, 0.91], [0.1, 0.2, 0.15, 0.05]) > 0.2


def test_api_evaluate():
    client = TestClient(app)
    cases = load_golden(GOLDEN)[:2]
    resp = client.post("/evaluate", json={"cases": cases})
    assert resp.status_code == 200 and resp.json()["count"] == 2
    assert summarize([])["count"] == 0
