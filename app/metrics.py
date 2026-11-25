"""Aggregate metrics, regression comparison and drift detection."""
import math
import statistics


def summarize(scores: list[float], threshold: float = 0.6) -> dict:
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "stddev": 0.0,
                "pass_rate": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(statistics.median(scores), 4),
        "stddev": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "pass_rate": round(sum(1 for s in scores if s >= threshold) / len(scores), 4),
        "min": min(scores),
        "max": max(scores),
    }


def failed_cases(per_case: dict[str, float], threshold: float = 0.6) -> list[str]:
    """Case ids scoring below `threshold`, weakest first — the triage list
    you read first when a run regresses."""
    return [cid for cid, s in sorted(per_case.items(), key=lambda kv: kv[1])
            if s < threshold]


def compare_runs(baseline: dict, candidate: dict, tolerance: float = 0.02) -> dict:
    """Regression report: is the candidate run worse than the baseline?"""
    delta = round(candidate["mean"] - baseline["mean"], 4)
    regressed_cases = [
        cid for cid, s in candidate.get("per_case", {}).items()
        if cid in baseline.get("per_case", {}) and s < baseline["per_case"][cid] - tolerance
    ]
    return {
        "mean_delta": delta,
        "regressed": delta < -tolerance,
        "regressed_cases": regressed_cases,
        "baseline_mean": baseline["mean"],
        "candidate_mean": candidate["mean"],
    }


def drift_score(baseline_scores: list[float], current_scores: list[float]) -> float:
    """Population Stability Index (PSI) over score histograms; >0.2 = major drift."""
    def hist(xs):
        bins = [0] * 5
        for x in xs:
            bins[min(4, int(x * 5))] += 1
        total = max(len(xs), 1)
        return [(b + 0.5) / (total + 2.5) for b in bins]  # smoothed

    base, cur = hist(baseline_scores), hist(current_scores)
    return round(sum((c - b) * math.log(c / b) for b, c in zip(base, cur)), 4)
