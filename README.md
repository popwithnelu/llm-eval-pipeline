# LLM Evaluation Pipeline

An evaluation service for LLM-powered applications: run a **golden dataset**
through an **LLM-as-judge** scorer, produce a summary report, compare runs for
**regressions**, and monitor **drift** with a Population Stability Index (PSI)
over score distributions.

## How it works

```
golden_dataset.jsonl -> runner -> LLMJudge (live model or offline heuristic)
                     -> summarize (mean / pass rate)
                     -> compare_runs (regression report vs a baseline run)
                     -> drift_score (PSI: <0.1 stable, 0.1–0.2 moderate, >0.2 major drift)
```

Results are designed to be stored per run in PostgreSQL (optionally with
pgvector for embedding-based analysis); the service itself is stateless.

## Tech stack
Python, FastAPI, Pydantic; optional OpenAI API for live judging; optional PostgreSQL/pgvector storage.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST localhost:8000/evaluate -H 'Content-Type: application/json' \
  -d '{"cases": [{"id": "x", "prompt": "What is RAG?", "response": "...", "reference": "..."}]}'
```

## Tests

```bash
python -m pytest   # offline: heuristic judge, sample golden dataset included
```

## Project structure

```
app/ (judge, metrics, runner, main) · data/golden_dataset.jsonl · tests/test_eval.py
```

MIT License — see LICENSE.
