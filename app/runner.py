"""Run an evaluation over a golden dataset and produce a report."""
import json

from .judge import LLMJudge
from .metrics import failed_cases, summarize


def load_golden(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(cases: list[dict], judge: LLMJudge | None = None,
                   responder=None) -> dict:
    """`responder` maps a prompt to a model response; defaults to the case's
    canned `response` field so golden files are self-contained."""
    judge = judge or LLMJudge()
    per_case = {}
    for case in cases:
        response = responder(case["prompt"]) if responder else case.get("response", "")
        per_case[case["id"]] = judge.score(case["prompt"], response, case["reference"])
    report = summarize(list(per_case.values()))
    report["per_case"] = per_case
    report["failed_cases"] = failed_cases(per_case)
    report["judge_mode"] = judge.mode
    return report
