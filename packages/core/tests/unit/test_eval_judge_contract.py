"""The judge contract on the HTTP eval path.

The invariant under test: **the eval harness never converts an
infrastructure/judge failure into a model score.**

Before this, all three judges in ``openexecutive/evals/judges.py`` caught every
parse failure and returned ``{"overall": 0}``. The runner then scored that as
an ordinary failed evaluation (0 < 3.5), and ``/evals/runs`` still marked the
run COMPLETED — so a broken judge was indistinguishable from a bad model, and
the API reported success while doing it.

Every test here mocks the provider. No network, no paid call.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

import pytest

from openexecutive.evals import judges
from openexecutive.evals.judge_contract import JudgeError, invoke_judge

# One reply shape per way a judge can be unusable. Each of these previously
# became {"overall": 0} and was recorded as a legitimate score.
MALFORMED_REPLIES = {
    "no JSON at all": "The response seemed reasonable to me.",
    "truncated JSON": '{"overall": 4, "notes": "cut off',
    "JSON but not an object": "[1, 2, 3]",
    "missing overall": '{"notes": "forgot to score"}',
    "non-numeric overall": '{"overall": "excellent"}',
}


def _provider(text: str) -> Any:
    class _P:
        async def messages_create(self, **_: Any) -> Any:
            return NS(content=[NS(type="text", text=text)])

    return _P()


def _no_text_provider() -> Any:
    class _P:
        async def messages_create(self, **_: Any) -> Any:
            return NS(content=[NS(type="tool_use", id="x", name="y", input={})])

    return _P()


CHAT_SCENARIO = {
    "id": "s1",
    "domain": "finance",
    "description": "d",
    "query": "q",
    "expected_topics": ["a"],
    "quality_criteria": {},
    "workflow": "w",
    "expected_artifact_sections": [],
    "event": {},
    "context": {},
    "expected_decision": {},
}


# ---------------------------------------------------------------------------
# The shared contract itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(MALFORMED_REPLIES))
def test_invoke_judge_raises_on_every_malformed_reply(label: str) -> None:
    text = MALFORMED_REPLIES[label]
    with pytest.raises(JudgeError) as exc:
        asyncio.run(invoke_judge(_provider(text), "prompt"))
    # The complete raw reply must survive — it is the only way to diagnose a
    # judge outage after the fact.
    assert exc.value.raw == text


def test_invoke_judge_raises_when_there_is_no_text_block() -> None:
    with pytest.raises(JudgeError):
        asyncio.run(invoke_judge(_no_text_provider(), "prompt"))


def test_invoke_judge_preserves_valid_scoring_semantics() -> None:
    """Valid verdicts are unchanged — this slice must not move any score."""
    verdict = asyncio.run(
        invoke_judge(
            _provider('{"overall": 4.2, "persona_coherence": 5, "notes": "good"}'),
            "prompt",
        )
    )
    assert verdict["overall"] == 4.2
    assert verdict["persona_coherence"] == 5
    assert verdict["notes"] == "good"
    assert verdict["_raw"]


def test_invoke_judge_coerces_numeric_string_overall() -> None:
    """A judge that quotes its number is still usable — don't fail closed on
    something that parses cleanly to a float."""
    verdict = asyncio.run(invoke_judge(_provider('{"overall": "4"}'), "prompt"))
    assert verdict["overall"] == 4.0


# ---------------------------------------------------------------------------
# The three HTTP-path judges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "judge_name,args",
    [
        ("judge_chat", (CHAT_SCENARIO, "some response")),
        ("judge_workflow", (CHAT_SCENARIO, "some artifact")),
        ("judge_triage", (CHAT_SCENARIO, {"alert": True, "severity": "high"})),
    ],
)
@pytest.mark.parametrize("label", sorted(MALFORMED_REPLIES))
def test_http_judges_no_longer_fail_open(
    monkeypatch: pytest.MonkeyPatch, judge_name: str, args: tuple, label: str
) -> None:
    """Regression: each of these returned {"overall": 0} for these replies."""
    text = MALFORMED_REPLIES[label]
    monkeypatch.setattr(judges, "get_provider", lambda _model: _provider(text))
    with pytest.raises(JudgeError) as exc:
        asyncio.run(getattr(judges, judge_name)(*args))
    assert exc.value.raw == text


@pytest.mark.parametrize(
    "judge_name,args",
    [
        ("judge_chat", (CHAT_SCENARIO, "some response")),
        ("judge_workflow", (CHAT_SCENARIO, "some artifact")),
        ("judge_triage", (CHAT_SCENARIO, {"alert": True})),
    ],
)
def test_http_judges_still_score_valid_verdicts(
    monkeypatch: pytest.MonkeyPatch, judge_name: str, args: tuple
) -> None:
    monkeypatch.setattr(
        judges, "get_provider", lambda _model: _provider('{"overall": 4.5}')
    )
    verdict = asyncio.run(getattr(judges, judge_name)(*args))
    assert verdict["overall"] == 4.5


def test_no_fail_open_literal_remains_in_judges_module() -> None:
    """Belt and braces: the literal that caused this cannot creep back in."""
    from pathlib import Path

    src = Path(judges.__file__).read_text()
    assert '"overall": 0' not in src


def test_judge_model_has_exactly_one_definition() -> None:
    """The model a provider is resolved FOR must equal the model a request is
    labelled WITH; two constants is how those drift."""
    from openexecutive.evals import judge_contract

    assert judges._JUDGE_MODEL is judge_contract.JUDGE_MODEL


# ---------------------------------------------------------------------------
# The runner turns a JudgeError into an honest scenario_error
# ---------------------------------------------------------------------------


def test_runner_emits_scenario_error_not_a_zero_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge outage must surface as scenario_error carrying the raw reply —
    never as a scenario_done whose scores say the model earned a zero."""
    from openexecutive.evals import runner

    async def _boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise JudgeError("no JSON object found in judge response", raw="I refuse.")

    class _FakeAgent:
        async def triage(self, *_a: Any, **_k: Any) -> Any:
            return NS(model_dump=lambda mode=None: {"alert": True})

    monkeypatch.setattr(runner, "judge_triage", _boom)
    import openexecutive.agents.triage as triage_mod

    monkeypatch.setattr(triage_mod, "TriageAgent", lambda *a, **k: _FakeAgent())

    # A VALID AlertEvent — the point is to reach the judge and fail there, not
    # to trip scenario construction on the way in.
    scenario = {
        "id": "triage_x",
        "domain": "triage",
        "description": "d",
        "event": {"source": "email", "external_id": "e1", "body": "b"},
    }
    monkeypatch.setattr(
        runner, "load_scenarios", lambda **_k: [dict(scenario, _kind="triage")]
    )

    async def _collect() -> list[dict[str, Any]]:
        return [ev async for ev in runner.run_scenarios(kind="triage")]

    events = asyncio.run(_collect())
    by_type = {e["type"]: e for e in events}

    assert "scenario_error" in by_type, f"expected scenario_error, got {list(by_type)}"
    err = by_type["scenario_error"]
    assert err["error_kind"] == "judge_unparseable"
    assert err["judge_raw"] == "I refuse."
    # And crucially: no scenario_done carrying a fabricated score.
    assert "scenario_done" not in by_type
    assert by_type["suite_done"]["passed"] == 0
