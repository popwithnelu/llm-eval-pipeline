"""LLM-as-judge scorer.

Live mode (optional): if OPENAI_API_KEY is set and the SDK is installed, the
judge prompts a model. Offline mode uses a transparent heuristic scorer
(reference keyword coverage + length sanity) so the whole pipeline and its
tests run without network access.
"""
import os
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class LLMJudge:
    def __init__(self):
        self.mode = "heuristic"
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai  # type: ignore

                self._client = openai.OpenAI()
                self.mode = "openai"
            except Exception:
                self.mode = "heuristic"

    def score(self, prompt: str, response: str, reference: str) -> float:
        """Return a score in [0, 1]."""
        if self.mode == "openai":
            return self._score_with_model(prompt, response, reference)
        ref = _tokens(reference)
        if not ref:
            return 0.0
        coverage = len(ref & _tokens(response)) / len(ref)
        length_penalty = 1.0 if len(response.strip()) >= 10 else 0.5
        return round(min(1.0, coverage * length_penalty), 4)

    def _score_with_model(self, prompt, response, reference) -> float:
        msg = (
            "Score the response against the reference from 0 to 1. "
            f"Prompt: {prompt}\nResponse: {response}\nReference: {reference}\n"
            "Reply with only the number."
        )
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": msg}]
        )
        try:
            return max(0.0, min(1.0, float(resp.choices[0].message.content.strip())))
        except Exception:
            return 0.0
