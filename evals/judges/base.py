"""Judge contract for the standalone CLI harness.

Deliberately a re-export, not an implementation: the canonical parser lives at
``openexecutive.evals.judge_contract`` so the CLI and the HTTP eval path share
exactly one copy. Duplicating it is how the two surfaces drifted in the first
place — the CLI was fixed to raise on an unparseable verdict while the HTTP
path kept returning ``{"overall": 0}``.

``run_evals.py`` puts ``packages/core`` on ``sys.path`` before importing this
module, so the package import below resolves for the CLI.
"""
from __future__ import annotations

from openexecutive.evals.judge_contract import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JudgeError,
    invoke_judge,
)

__all__ = ["JUDGE_MAX_TOKENS", "JUDGE_MODEL", "JudgeError", "invoke_judge"]
