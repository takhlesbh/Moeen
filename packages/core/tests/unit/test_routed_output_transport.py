"""The envelope is the asyncio task result — and only that.

Phase 3B2's first design carried the CFO's calculations back on a module
ContextVar. ``route_parallel`` gathers one child task per specialist, and a
ContextVar set in a gathered child is invisible to the awaiting parent (each
task runs in a copy of the context) — worse, the parent keeps whatever the var
held before, so a turn that computed nothing reads as the previous turn. These
tests drive the REAL ``route_parallel`` and assert on its return value, and
the first one locks the ContextVar defect out permanently.

Failure semantics are pinned as they were before the envelope existed:
``gather`` without ``return_exceptions``, no handler in ``call_one`` — one
failing specialist still fails the batch, and cancellation propagates.
"""
from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from openexecutive.orchestrator.router import SPECIALIST_REGISTRY, route_parallel
from openexecutive.specialists.calculation_gateway import SpecialistCalculations
from openexecutive.specialists.routed_output import (
    CALC_CLAIM_SET_UNSAFE,
    CALC_CONTEXT_UNAVAILABLE,
    CALC_DISABLED,
    CALC_GATEWAY_RAISED,
    CALC_NO_PROPOSALS,
    CALC_SKIPPED_STRUCTURE_LOST,
    CALC_TOO_MANY_PROPOSALS,
    DIAGNOSTIC_CODES,
    RoutedSpecialistOutput,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "openexecutive"


def _envelope(narrative: str, **kw: Any) -> RoutedSpecialistOutput:
    return RoutedSpecialistOutput(specialist="cfo", narrative=narrative, **kw)


def _parallel(calls: list[dict[str, str]]) -> list[RoutedSpecialistOutput]:
    return asyncio.run(route_parallel(calls=calls, retrieved_knowledge_map={}))


# ---------------------------------------------------------------------------
# G7: the real parent receives the envelope; a ContextVar would not have worked
# ---------------------------------------------------------------------------


def test_route_parallel_returns_envelopes_in_call_order() -> None:
    calcs = SpecialistCalculations(specialist="cfo")
    with (
        patch.object(SPECIALIST_REGISTRY["cso"], "analyze", AsyncMock(return_value="cso-text")),
        patch.object(
            SPECIALIST_REGISTRY["cfo"], "analyze_structured",
            AsyncMock(return_value=_envelope("cfo-text", calculations=calcs)),
        ),
    ):
        out = _parallel([
            {"specialist": "cso", "query": "a"},
            {"specialist": "cfo", "query": "b"},
        ])
    assert [o.specialist for o in out] == ["cso", "cfo"]
    assert [o.narrative for o in out] == ["cso-text", "cfo-text"]
    assert out[0].calculations is None and out[0].specialist_result is None
    assert out[1].calculations is calcs  # the very object the child produced


def test_a_contextvar_set_in_the_child_never_reaches_the_parent() -> None:
    """Anti-regression for the rejected design. Only the return value crosses."""
    leak: contextvars.ContextVar[str | None] = contextvars.ContextVar("leak", default=None)

    async def structured(**_: Any) -> RoutedSpecialistOutput:
        leak.set("set-in-child")
        return _envelope("cfo-text", diagnostics=(CALC_NO_PROPOSALS,))

    async def parent() -> tuple[str | None, list[RoutedSpecialistOutput]]:
        leak.set(None)
        with patch.object(SPECIALIST_REGISTRY["cfo"], "analyze_structured", structured):
            out = await route_parallel(
                calls=[{"specialist": "cfo", "query": "q"}], retrieved_knowledge_map={}
            )
        return leak.get(), out

    seen, out = asyncio.run(parent())
    assert seen is None                      # the ContextVar channel is write-only
    assert out[0].narrative == "cfo-text"    # the task result is not
    assert out[0].diagnostics == (CALC_NO_PROPOSALS,)


# ---------------------------------------------------------------------------
# G13: nothing stale crosses turns or concurrent tasks
# ---------------------------------------------------------------------------


def test_sequential_turns_do_not_inherit_a_previous_envelope() -> None:
    populated = _envelope("t1", calculations=SpecialistCalculations(specialist="cfo"))
    empty = _envelope("t2", diagnostics=(CALC_DISABLED,))

    async def both() -> tuple[list[RoutedSpecialistOutput], list[RoutedSpecialistOutput]]:
        with patch.object(SPECIALIST_REGISTRY["cfo"], "analyze_structured",
                          AsyncMock(return_value=populated)):
            first = await route_parallel(calls=[{"specialist": "cfo", "query": "q"}],
                                         retrieved_knowledge_map={})
        with patch.object(SPECIALIST_REGISTRY["cfo"], "analyze_structured",
                          AsyncMock(return_value=empty)):
            second = await route_parallel(calls=[{"specialist": "cfo", "query": "q"}],
                                          retrieved_knowledge_map={})
        return first, second

    first, second = asyncio.run(both())
    assert first[0].calculations is not None
    assert second[0].calculations is None
    assert second[0].diagnostics == (CALC_DISABLED,)


def test_concurrent_route_parallel_calls_do_not_cross() -> None:
    async def structured(query: str, **_: Any) -> RoutedSpecialistOutput:
        await asyncio.sleep(0.01 if query == "A" else 0)
        return _envelope(f"answer-{query}")

    async def both() -> list[list[RoutedSpecialistOutput]]:
        with patch.object(SPECIALIST_REGISTRY["cfo"], "analyze_structured", structured):
            return list(await asyncio.gather(
                route_parallel(calls=[{"specialist": "cfo", "query": "A"}],
                               retrieved_knowledge_map={}),
                route_parallel(calls=[{"specialist": "cfo", "query": "B"}],
                               retrieved_knowledge_map={}),
            ))

    a, b = asyncio.run(both())
    assert a[0].narrative == "answer-A" and b[0].narrative == "answer-B"


# ---------------------------------------------------------------------------
# G8: existing failure semantics are unchanged
# ---------------------------------------------------------------------------


def test_one_raising_specialist_still_fails_the_batch() -> None:
    with (
        patch.object(SPECIALIST_REGISTRY["cso"], "analyze",
                     AsyncMock(side_effect=RuntimeError("provider down"))),
        patch.object(SPECIALIST_REGISTRY["cfo"], "analyze_structured",
                     AsyncMock(return_value=_envelope("fine"))),
        pytest.raises(RuntimeError, match="provider down"),
    ):
        _parallel([{"specialist": "cso", "query": "a"}, {"specialist": "cfo", "query": "b"}])


def test_cancellation_propagates_through_route_parallel() -> None:
    async def hang(**_: Any) -> str:
        await asyncio.sleep(3600)
        return "never"

    async def run() -> None:
        with patch.object(SPECIALIST_REGISTRY["cso"], "analyze", hang):
            task = asyncio.ensure_future(route_parallel(
                calls=[{"specialist": "cso", "query": "a"}], retrieved_knowledge_map={}
            ))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_legacy_specialists_and_unknown_names_are_wrapped_verbatim() -> None:
    with patch.object(SPECIALIST_REGISTRY["cso"], "analyze", AsyncMock(return_value="")):
        out = _parallel([{"specialist": "cso", "query": "a"}, {"specialist": "nobody", "query": "b"}])
    assert out[0].narrative == ""                                # empty string preserved
    assert out[1].narrative == "Unknown specialist: nobody"      # legacy bytes
    assert all(o.specialist_result is None and o.calculations is None for o in out)


# ---------------------------------------------------------------------------
# Diagnostics are a closed, distinguishable vocabulary
# ---------------------------------------------------------------------------


def test_diagnostics_are_seven_distinct_literals() -> None:
    codes = (CALC_DISABLED, CALC_NO_PROPOSALS, CALC_SKIPPED_STRUCTURE_LOST,
             CALC_TOO_MANY_PROPOSALS, CALC_CONTEXT_UNAVAILABLE, CALC_CLAIM_SET_UNSAFE,
             CALC_GATEWAY_RAISED)
    assert len(set(codes)) == 7
    assert frozenset(codes) == DIAGNOSTIC_CODES
    assert all(c == c.strip() and c.isprintable() and c.islower() for c in codes)


# ---------------------------------------------------------------------------
# G9: nothing but narrative reaches the Executive prompt, Committee, or audit
# ---------------------------------------------------------------------------


def test_executive_stores_only_the_narrative_from_an_envelope() -> None:
    """Source-level: every sink in the agent loop reads ``.narrative``.

    The loop cannot be driven end to end without a fake streaming provider,
    so the property is pinned where it lives: the three assignments that feed
    the model's tool_result, the Committee's specialist_outputs and the audit
    row all take ``narrative``; no attribute of the envelope other than
    ``narrative`` is read anywhere in the Executive or the Committee.
    """
    executive = (PACKAGE_ROOT / "orchestrator" / "executive.py").read_text(encoding="utf-8")
    assert 'results_by_id[tu["id"]] = out.narrative' in executive
    assert 'specialist_outputs_out[call["specialist"]] = out.narrative' in executive
    assert '"response": spec_out.narrative' in executive
    for module in ("orchestrator/executive.py", "orchestrator/committee.py",
                   "orchestrator/committee_reviewers.py"):
        source = (PACKAGE_ROOT / module).read_text(encoding="utf-8")
        for attribute in (".calculations", ".specialist_result", ".frame", ".diagnostics"):
            assert attribute not in source, f"{module} reads {attribute}"


def test_committee_still_takes_plain_strings() -> None:
    import inspect

    from openexecutive.orchestrator.committee import Committee

    annotation = inspect.signature(Committee.review).parameters["specialist_outputs"].annotation
    assert annotation == "dict[str, str] | None"


# ---------------------------------------------------------------------------
# G9 — end to end through the REAL Executive agent loop
# ---------------------------------------------------------------------------
#
# Everything on the path is real: Executive._stream_agent_loop → route_parallel
# → route_to_specialist_structured → FinanceAgent.analyze_structured →
# parse_specialist_result → execute_proposals → execute_batch → the envelope
# back in the parent loop. Only the two model providers are scripted, and they
# are patched at their SEPARATE bindings: the Executive's stream call is bound
# in orchestrator/executive.py, the CFO's messages_create call in
# agents/base.py (inherited by FinanceAgent), and the CFO's thinking-support
# probe resolves openexecutive.providers.get_provider at call time. Patching
# one does not patch another.
#
# What this test does NOT execute: Committee review. _stream_agent_loop does
# not call the Committee; stream_chat_with_committee does, at
# executive.py `self._committee.review(..., specialist_outputs=specialist_outputs)`,
# consuming the same `specialist_outputs` mapping this test proves is
# dict[str, str]. Committee runtime coverage is out of scope here and is not
# claimed.

ENGINE_ONLY_FIELDS = (
    "arithmetic_status", "fingerprint", "authority_version",
    "normalized_operands", "exact_result", "expression_executed",
)
RICH_TYPE_NAMES = ("RoutedSpecialistOutput", "SpecialistCalculations", "CalculationResult")


class _CfoProvider:
    """The CFO's provider: one scripted ``messages_create`` reply, calls recorded."""

    def __init__(self, message: Any) -> None:
        self._message = message
        self.calls: list[dict[str, Any]] = []

    async def messages_create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._message


def test_g9_calculations_reach_the_parent_loop_and_only_narrative_reaches_the_sinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from decimal import Decimal
    from types import SimpleNamespace

    from openexecutive.audit import AuditLogger, set_audit_logger
    from openexecutive.audit.context import set_turn
    from openexecutive.calc.contract import CalculationResult
    from openexecutive.orchestrator.debug_events import DebugCollector
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.specialists.calculation_gateway import mint_request_id
    from openexecutive.specialists.result_contract import EMIT_SPECIALIST_RESULT_TOOL
    from tests.unit.test_executive_form_patch import (
        _FinalMsg,
        _ScriptedProvider,
        _TextBlock,
        _ToolUseBlock,
    )

    session_id, turn_id = "sess-g9", "t-g9"
    narrative = "Sources exceed capex by TND 11,860,000, leaving real headroom."
    payload = {
        "narrative": narrative,
        "claims": [{
            "claim_id": "c1",
            "text": "Financing sources exceed capex by TND 11,860,000.",
            "claim_type": "derived_calculation",
            "attribution": "specialist_judgement",
            "calculation": {"inputs": ["42,000,000", "30,140,000"],
                            "method": "sources - capex",
                            "model_stated_result": "11,860,000"},
        }],
        "calculation_requests": [{
            "operation": "subtract",
            "operands": [
                {"operand_id": "sources", "label": "sources", "value": "42000000",
                 "unit": {"code": "currency:TND"}, "basis": "applicant_stated"},
                {"operand_id": "capex", "label": "capex", "value": "30140000",
                 "unit": {"code": "currency:TND"}, "basis": "applicant_stated"},
            ],
            "target_unit": {"code": "currency:TND"},
            "purpose": "headroom",
            "claim_ref": "c1",
        }],
    }
    cfo_message = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="prose block"),
        SimpleNamespace(type="tool_use", id="tu-emit",
                        name=EMIT_SPECIALIST_RESULT_TOOL["name"], input=payload),
    ])
    cfo_provider = _CfoProvider(cfo_message)
    exec_provider = _ScriptedProvider([
        _FinalMsg([_ToolUseBlock("tu-cfo", "consult_specialist",
                                 {"specialist": "cfo", "query": "Reconcile headroom.",
                                  "context": "GreenHarvest"})], stop_reason="tool_use"),
        _FinalMsg([_TextBlock("Final synthesis.")], stop_reason="end_turn"),
    ])

    # Flag ON; no ChromaDB; no Honcho (disabled by default → "" prefetch).
    monkeypatch.setenv("CALC_CFO_STRUCTURED_ENABLED", "true")
    monkeypatch.setattr("openexecutive.knowledge.retriever.retrieve_structured",
                        lambda **_: ("", None))
    monkeypatch.setattr("openexecutive.knowledge.retriever.retrieve", lambda **_: "")
    monkeypatch.setattr("openexecutive.knowledge.retriever.retrieve_failures", lambda **_: "")
    audit = AuditLogger(tmp_path / "audit.db")
    set_audit_logger(audit)

    calc_outcomes_out: list[RoutedSpecialistOutput] = []
    specialist_outputs_out: dict[str, str] = {}
    consulted_out: list[str] = []
    collector = DebugCollector(turn_id=turn_id)
    emitted: list[Any] = []

    async def drive() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider",
                  return_value=exec_provider),
            patch("openexecutive.agents.base.get_provider", return_value=cfo_provider),
            patch("openexecutive.providers.get_provider", return_value=cfo_provider),
            set_turn(session_id=session_id, turn_id=turn_id),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[],
                messages=[{"role": "user", "content": "How much headroom?"}],
                model="claude-test",
                debug_collector=collector,
                consulted_out=consulted_out,
                specialist_outputs_out=specialist_outputs_out,
                turn_id=turn_id,
                calc_outcomes_out=calc_outcomes_out,
            ):
                emitted.append(item)

    try:
        asyncio.run(drive())
    finally:
        set_audit_logger(None)

    # Both scripted providers were actually driven — the real path ran.
    assert len(exec_provider.calls) == 2
    assert len(cfo_provider.calls) == 1
    assert consulted_out == ["cfo"]

    # 1–2. Exactly one envelope, for the CFO, arrived in the PARENT loop.
    assert len(calc_outcomes_out) == 1
    outcome = calc_outcomes_out[0]
    assert isinstance(outcome, RoutedSpecialistOutput)
    assert outcome.specialist == "cfo"
    assert outcome.diagnostics == ()

    # 3. Real gateway/engine records, checked against independent arithmetic.
    calcs = outcome.calculations
    assert isinstance(calcs, SpecialistCalculations)
    assert len(calcs.requests) == 1 and len(calcs.results) == 1 and calcs.dropped == ()
    record = calcs.results[0]
    assert isinstance(record, CalculationResult)
    assert record.arithmetic_status == "ARITHMETIC_VERIFIED", record.errors
    assert record.result_value == "11860000.00"
    assert Decimal(record.result_value) == Decimal("42000000") - Decimal("30140000")
    assert record.authority.authority_version == "0.2.0-engine"
    assert record.evidence.status == "EVIDENCE_UNAVAILABLE"
    assert record.is_verified_evidence() is False
    assert outcome.specialist_result is not None
    proposal = outcome.specialist_result.calculation_requests[0]
    content = json.dumps(proposal.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    assert calcs.requests[0].request_id == mint_request_id(
        case_id=session_id, run_id=turn_id, specialist="cfo", position=0,
        claim_ref="c1", content=content,
    )
    assert calcs.requests[0].correlation.case_id == session_id
    assert calcs.requests[0].correlation.run_id == turn_id
    assert calcs.requests[0].correlation.claim_id == "c1"
    assert outcome.frame is not None and outcome.frame.run_id == turn_id

    # 4. Model-owned and gateway-owned parts are separate objects; the claim's
    #    provenance did not move because the arithmetic succeeded.
    assert outcome.specialist_result is not calcs
    claim = outcome.specialist_result.claims[0]
    assert claim.calculation is not None
    assert claim.calculation.verification_status == "unverified"
    assert claim.calculation.verified_result is None

    # 5. The Committee-facing mapping holds the narrative string and nothing else.
    assert set(specialist_outputs_out) == {"cfo"}
    committee_input = specialist_outputs_out["cfo"]
    assert type(committee_input) is str
    assert committee_input == outcome.narrative == narrative
    assert not isinstance(committee_input, RoutedSpecialistOutput | SpecialistCalculations)

    # 6. The tool_result the Executive's model received is the narrative alone.
    follow_up = exec_provider.calls[1]["messages"][-1]
    assert follow_up["role"] == "user"
    (tool_result,) = follow_up["content"]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu-cfo"
    assert type(tool_result["content"]) is str
    assert tool_result["content"] == outcome.narrative
    serialized_messages = json.dumps(exec_provider.calls[1]["messages"], default=repr)
    for field in ENGINE_ONLY_FIELDS:
        assert field not in serialized_messages, field
    for name in RICH_TYPE_NAMES:
        assert name not in serialized_messages, name

    # 7. Nothing yielded by the loop carries an engine record, and the audit row
    #    for the consultation stores the narrative alone.
    assert emitted, "the loop yielded nothing"
    for item in emitted:
        assert not isinstance(item, RoutedSpecialistOutput | SpecialistCalculations)
        assert not isinstance(item, CalculationResult)
    serialized_events = json.dumps([i for i in emitted if isinstance(i, dict)], default=repr)
    for field in ENGINE_ONLY_FIELDS:
        assert field not in serialized_events, field
    for name in RICH_TYPE_NAMES:
        assert name not in serialized_events, name
    rows = audit.query(event_type="specialist_consult", session_id=session_id)
    assert len(rows) == 1
    # ``query`` does not hydrate ``full_json`` (it selects the summary columns
    # only); ``get`` is the accessor that does, so the stored payload has to be
    # re-read by id to be inspected at all.
    stored = audit.get(rows[0].id)
    assert stored is not None and stored.full is not None
    assert stored.full["response"] == outcome.narrative
    serialized_audit = json.dumps(stored.full, default=repr)
    for field in ENGINE_ONLY_FIELDS:
        assert field not in serialized_audit, field
    for name in RICH_TYPE_NAMES:
        assert name not in serialized_audit, name

    # 8. The records are still there after the loop has finished — this is the
    #    child→parent transport, observed from the parent after the fact.
    assert calc_outcomes_out[0].calculations is calcs
    assert calc_outcomes_out[0].calculations.results[0].result_value == "11860000.00"
