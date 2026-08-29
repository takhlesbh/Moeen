"""The one judge contract, shared by every eval surface.

Single rule: **an unparseable judge reply is an infrastructure error, never a
score.**

Both eval surfaces previously parsed judge replies themselves and returned
``{"overall": 0}`` on failure. That silently converted "the harness could not
obtain a verdict" into "the model scored zero" — a product regression that
never happened, recorded as if it had. On the HTTP path the run could still be
marked COMPLETED with that fabricated zero inside it.

This module lives inside the package (not in the standalone ``evals/`` tree)
because the FastAPI process cannot import that tree. The CLI re-exports from
here — see ``evals/judges/base.py`` — so the parser exists exactly once.
"""
from __future__ import annotations

import json
from typing import Any

# The judge model. Single source of truth: every caller resolves its provider
# from this value and labels its request with it, so the model a provider is
# resolved *for* can never drift from the model a request is labelled *with*.
JUDGE_MODEL = "claude-opus-4-7"

# Judges answer with a small fixed-shape JSON verdict; 500 tokens is ample and
# bounds the cost of a judge that starts rambling instead of scoring.
JUDGE_MAX_TOKENS = 500


class JudgeError(RuntimeError):
    """The judge did not return a usable verdict.

    Carries the complete raw response so the failure can be diagnosed from the
    persisted evidence alone, without re-running anything.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


async def invoke_judge(provider: Any, judge_prompt: str) -> dict[str, Any]:
    """Call the judge and return its parsed verdict, or raise ``JudgeError``.

    Scoring semantics for a VALID verdict are unchanged: the parsed object is
    returned as-is, with ``overall`` coerced to float and the raw text attached
    as ``_raw`` so callers can persist exactly what the judge said alongside
    what was parsed out of it.
    """
    message = await provider.messages_create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    blocks = [b for b in message.content if getattr(b, "type", None) == "text"]
    if not blocks:
        raise JudgeError("judge returned no text block", raw="")
    text = blocks[0].text

    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise JudgeError("no JSON object found in judge response", raw=text)
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge JSON is malformed: {exc}", raw=text) from exc
    if not isinstance(parsed, dict):
        raise JudgeError("judge JSON is not an object", raw=text)
    if "overall" not in parsed:
        raise JudgeError("judge verdict has no 'overall' field", raw=text)
    try:
        parsed["overall"] = float(parsed["overall"])
    except (TypeError, ValueError) as exc:
        raise JudgeError(
            f"judge 'overall' is not numeric: {parsed['overall']!r}", raw=text
        ) from exc

    parsed["_raw"] = text
    return parsed
