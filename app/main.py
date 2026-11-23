from fastapi import FastAPI
from pydantic import BaseModel

from .metrics import compare_runs, drift_score
from .runner import run_evaluation

app = FastAPI(title="LLM Evaluation Pipeline")


class EvalRequest(BaseModel):
    cases: list[dict]


class CompareRequest(BaseModel):
    baseline: dict
    candidate: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    return run_evaluation(req.cases)


@app.post("/compare")
def compare(req: CompareRequest):
    return compare_runs(req.baseline, req.candidate)


@app.post("/drift")
def drift(payload: dict):
    return {"psi": drift_score(payload["baseline_scores"], payload["current_scores"])}
