"""CFO → gateway → engine, through the real ``route_parallel``, on a fixture.

Deterministic and offline: the provider is a stub returning a pinned
``emit_specialist_result`` payload. This is NOT a live-model trial and proves
nothing about whether a model will emit proposals; it proves that when one
does, every authority boundary holds.

The GreenHarvest figures live here and in ``test_calc_greenharvest_fixtures``
and nowhere in production code or any prompt.

Version 1 limitation, stated plainly: the seven operations verify arithmetic
and contradiction handling. Model-supplied operands are not independent
evidence — every record reports ``EVIDENCE_UNAVAILABLE``. Proposals cannot
reference each other, so row 6's 5,720,000 kg input is a model-supplied
literal that this fixture cross-checks against row 5's engine result; the
engine itself does not resolve one proposal from another. Pepper's 2,170 t is
supplied, not derived, in this trial. No claim moves to verified evidence
because of any of these calculations.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openexecutive.agents import finance as finance_module
from openexecutive.agents.finance import FinanceAgent
from openexecutive.audit.context import set_turn
from openexecutive.orchestrator.router import route_parallel
from openexecutive.specialists.calculation_gateway import mint_request_id
from openexecutive.specialists.result_contract import (
    EMIT_SPECIALIST_RESULT_TOOL,
    CalculationProvenance,
    SpecialistResult,
    emit_specialist_result_tool,
    parse_specialist_result,
)
from openexecutive.specialists.routed_output import (
    CALC_CLAIM_SET_UNSAFE,
    CALC_CONTEXT_UNAVAILABLE,
    CALC_DISABLED,
    CALC_GATEWAY_RAISED,
    CALC_NO_PROPOSALS,
    CALC_SKIPPED_STRUCTURE_LOST,
    CALC_TOO_MANY_PROPOSALS,
    RoutedSpecialistOutput,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "openexecutive"
TOOL_NAME = EMIT_SPECIALIST_RESULT_TOOL["name"]
FLAG = "CALC_CFO_STRUCTURED_ENABLED"
SESSION, TURN = "sess-greenharvest", "t-0123456789ab"

# Pre-3B2 schema fingerprints. DEFINITIONS (do not change to fit a value):
#   sha256 over the UTF-8 bytes of
#   json.dumps(tool, sort_keys=<flag>, separators=(",", ":"), ensure_ascii=True)
CANONICAL_SHA256 = "3e35cc89fa29e3e2231c650dbb6289b34fff9f19ec49a0f515bc7e1403a9df85"
INSERTION_SHA256 = "e09b4f8cbb52a3e714c508fb4f84749c828ea20b659b5f1a88978ea98eab3532"


def _sha(tool: dict[str, Any], *, sort_keys: bool) -> str:
    raw = json.dumps(tool, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# fixture payload
# ---------------------------------------------------------------------------

TND = "currency:TND"


def _operand(oid: str, value: str, unit: str, *, role: str = "input") -> dict[str, Any]:
    return {"operand_id": oid, "label": oid, "value": value, "unit": {"code": unit},
            "basis": "applicant_stated", "role": role}


def _proposal(operation: str, operands: list[dict[str, Any]], target: str, *,
              claim_ref: str, scale: int = 2, purpose: str = "GreenHarvest") -> dict[str, Any]:
    return {"operation": operation, "operands": operands, "target_unit": {"code": target},
            "scale": scale, "purpose": purpose, "claim_ref": claim_ref}


COST_STACK = ["15", "22", "8", "6", "10", "4"]

CLAIMS: list[dict[str, Any]] = [
    {"claim_id": "c1", "text": "The operating cost stack sums to 65% of revenue.",
     "claim_type": "derived_calculation", "attribution": "specialist_judgement",
     "calculation": {"inputs": COST_STACK, "method": "sum of six cost lines",
                     "model_stated_result": "65%"}},
    {"claim_id": "c2", "text": "Gross margin is above 60%.",
     "claim_type": "source_fact", "attribution": "applicant_asserted"},
    {"claim_id": "c3", "text": "Financing sources total TND 42,000,000 against capex "
                              "of TND 30,140,000.",
     "claim_type": "source_fact", "attribution": "applicant_asserted"},
    {"claim_id": "c4", "text": "Tomato production is 572 tonnes.",
     "claim_type": "source_fact", "attribution": "applicant_asserted"},
    {"claim_id": "c5", "text": "Pepper production is 217 tonnes.",
     "claim_type": "source_fact", "attribution": "applicant_asserted"},
]

# The seven operations. Each verified against the shipped engine before this
# file was written; the expected values below are the engine's, not ours.
PROPOSALS: list[dict[str, Any]] = [
    _proposal("sum_components",
              [_operand(f"c{i}", v, "pct") for i, v in enumerate(COST_STACK)],
              "pct", claim_ref="c1", scale=0),
    _proposal("subtract", [_operand("hundred", "100", "pct"), _operand("cost", "65", "pct")],
              "pct", claim_ref="c2", scale=0),
    _proposal("variance", [_operand("margin", "35", "pct"),
                           _operand("claimed", "60", "pct", role="stated_comparison")],
              "pct", claim_ref="c2"),
    _proposal("subtract", [_operand("sources", "42000000", TND),
                           _operand("capex", "30140000", TND)],
              TND, claim_ref="c3"),
    _proposal("multiply", [_operand("yield", "52", "kg_per_m2"), _operand("area", "11", "ha")],
              "kg", claim_ref="c4", scale=0),
    _proposal("variance", [_operand("produced", "5720000", "kg"),
                           _operand("stated", "572", "t", role="stated_comparison")],
              "t", claim_ref="c4"),
    _proposal("variance", [_operand("expected", "2170", "t"),
                           _operand("stated", "217", "t", role="stated_comparison")],
              "t", claim_ref="c5"),
]

# The narrative CONTRADICTS the engine on purpose (572 t, margin above 60%).
NARRATIVE = ("Tomato output is around 572 tonnes and pepper around 217 tonnes; "
             "gross margin stays above 60% on a 65% cost stack. Financing of "
             "TND 42,000,000 comfortably covers TND 30,140,000 of capex.")

EXPECTED = [  # (operation, result_value, result_unit, conflict)
    ("sum_components", "65", "pct", "NONE"),
    ("subtract", "35", "pct", "NONE"),
    ("variance", "-25.00", "pct", "CONFLICT_DETECTED"),
    ("subtract", "11860000.00", TND, "NONE"),
    ("multiply", "5720000", "kg", "NONE"),
    ("variance", "5148.00", "t", "ORDER_OF_MAGNITUDE"),
    ("variance", "1953.00", "t", "ORDER_OF_MAGNITUDE"),
]


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"narrative": NARRATIVE, "claims": CLAIMS,
                               "calculation_requests": PROPOSALS}
    payload.update(overrides)
    return payload


def _message(payload: Any, *text_blocks: str) -> SimpleNamespace:
    blocks: list[Any] = [SimpleNamespace(type="text", text=t) for t in text_blocks]
    blocks.append(SimpleNamespace(type="tool_use", id="tu1", name=TOOL_NAME, input=payload))
    return SimpleNamespace(content=blocks)


def _run_parallel(message: Any, *, bound: bool = True) -> tuple[RoutedSpecialistOutput, AsyncMock]:
    """Drive the REAL route_parallel → route_to_specialist_structured → CFO."""
    mock = AsyncMock(return_value=message)

    async def go() -> list[RoutedSpecialistOutput]:
        return await route_parallel(calls=[{"specialist": "cfo", "query": "Reconcile."}],
                                    retrieved_knowledge_map={})

    with patch.object(FinanceAgent, "analyze_with_tools", mock):
        if bound:
            with set_turn(session_id=SESSION, turn_id=TURN):
                out = asyncio.run(go())
        else:
            out = asyncio.run(go())
    assert len(out) == 1
    return out[0], mock


@pytest.fixture
def flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLAG, "true")


@pytest.fixture
def flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)


# ---------------------------------------------------------------------------
# G1 / G2: the flag at the wire
# ---------------------------------------------------------------------------


def test_off_schema_is_byte_identical_to_pre_3b2() -> None:
    tool = emit_specialist_result_tool()
    assert _sha(tool, sort_keys=True) == CANONICAL_SHA256
    assert _sha(tool, sort_keys=False) == INSERTION_SHA256
    assert _sha(emit_specialist_result_tool(include_calculation_requests=False),
                sort_keys=False) == INSERTION_SHA256
    assert "calculation_requests" not in json.dumps(tool)


def test_on_schema_adds_the_property_at_sorted_position_zero_only() -> None:
    on = emit_specialist_result_tool(include_calculation_requests=True)
    props = on["input_schema"]["properties"]
    assert list(props) == ["calculation_requests", "claims", "narrative"]
    assert on["input_schema"]["required"] == ["narrative"]
    # Everything except the added key is unchanged, and the template is untouched.
    off = emit_specialist_result_tool()
    on_minus = {**on, "input_schema": {**on["input_schema"],
                                       "properties": {k: v for k, v in props.items()
                                                      if k != "calculation_requests"}}}
    assert on_minus == off
    assert _sha(emit_specialist_result_tool(), sort_keys=False) == INSERTION_SHA256
    assert "calculation_requests" not in EMIT_SPECIALIST_RESULT_TOOL["input_schema"]["properties"]


def test_the_on_schema_never_exposes_parser_metadata() -> None:
    serialized = json.dumps(emit_specialist_result_tool(include_calculation_requests=True))
    for absent in ('"integrity"', '"degraded"', '"degraded_reason"', '"verified_result"',
                   '"verification_status"', '"request_id"', '"fingerprint"', '"authority"'):
        assert absent not in serialized, absent


def test_flag_off_sends_the_pre_3b2_schema_and_runs_no_gateway(flag_off: None) -> None:
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=AssertionError("gateway must not run"))):
        out, mock = _run_parallel(_message(_payload()))
    sent = mock.await_args.kwargs["tools"][0]
    assert _sha(sent, sort_keys=False) == INSERTION_SHA256
    assert out.calculations is None and out.frame is None
    assert out.diagnostics == (CALC_DISABLED,)
    # With the key unknown, the parse degrades exactly as it does today.
    assert out.specialist_result is not None
    assert out.specialist_result.integrity == "partial"
    assert out.specialist_result.calculation_requests == ()
    assert "1 unrecognized top-level payload key(s)" in (out.specialist_result.degraded_reason or "")


def test_flag_on_is_read_only_by_the_structured_path(flag_on: None) -> None:
    """``analyze()`` still sends the OFF schema even with the flag on."""
    agent = FinanceAgent()
    mock = AsyncMock(return_value=_message(_payload()))
    with (
        patch.object(FinanceAgent, "analyze_with_tools", mock),
        patch.object(finance_module, "execute_proposals",
                     MagicMock(side_effect=AssertionError("gateway must not run"))),
        set_turn(session_id=SESSION, turn_id=TURN),
    ):
        text = asyncio.run(agent.analyze(query="q"))
    assert _sha(mock.await_args.kwargs["tools"][0], sort_keys=False) == INSERTION_SHA256
    assert isinstance(text, str)


def test_no_literal_true_enables_the_flag_anywhere_in_production() -> None:
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        assert "include_calculation_requests=True" not in source, path
        assert "accept_calculation_requests=True" not in source, path


# ---------------------------------------------------------------------------
# G11: the GreenHarvest fixture, seven operations, through the real path
# ---------------------------------------------------------------------------


def test_greenharvest_seven_operations_through_route_parallel(flag_on: None) -> None:
    out, _ = _run_parallel(_message(_payload()))
    result = out.specialist_result
    assert result is not None and result.integrity == "intact" and not result.degraded
    assert out.diagnostics == ()
    assert out.frame is not None and out.frame.case_id == SESSION and out.frame.run_id == TURN

    calcs = out.calculations
    assert calcs is not None and calcs.specialist == "cfo" and calcs.dropped == ()
    assert len(calcs.requests) == len(calcs.results) == 7 == len(PROPOSALS)

    for record, (operation, value, unit, conflict) in zip(calcs.results, EXPECTED, strict=True):
        assert record.operation == operation
        assert record.arithmetic_status == "ARITHMETIC_VERIFIED", record.errors
        assert record.result_value == value
        assert record.result_unit is not None and record.result_unit.code == unit
        assert record.conflict == conflict
        assert record.authority.authority_version == "0.2.0-engine"
        assert record.fingerprint is not None and len(record.fingerprint) == 64
        assert record.computed_at == out.frame.computed_at

    # Independent Decimal cross-checks — the engine's numbers, recomputed here.
    assert Decimal(calcs.results[0].result_value or "") == sum(Decimal(v) for v in COST_STACK)
    assert Decimal(calcs.results[1].result_value or "") == Decimal(100) - Decimal(65)
    assert Decimal(calcs.results[2].absolute_difference or "") == Decimal(60) - Decimal(35)
    assert Decimal(calcs.results[3].result_value or "") == Decimal(42_000_000) - Decimal(30_140_000)
    assert Decimal(calcs.results[4].result_value or "") == Decimal(52) * Decimal(11) * Decimal(10_000)
    assert Decimal(calcs.results[5].ratio or "") == Decimal(10)
    assert Decimal(calcs.results[6].ratio or "") == Decimal(10)
    # Row 6's input is a model-supplied literal; the fixture cross-checks it
    # against row 5. The engine does not resolve one proposal from another.
    assert calcs.requests[5].operands[0].value == calcs.results[4].result_value


def test_the_gateway_minted_every_identity(flag_on: None) -> None:
    out, _ = _run_parallel(_message(_payload()))
    assert out.calculations is not None and out.specialist_result is not None
    for position, (request, proposal) in enumerate(
        zip(out.calculations.requests, out.specialist_result.calculation_requests, strict=True)
    ):
        content = json.dumps(proposal.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True)
        assert request.request_id == mint_request_id(
            case_id=SESSION, run_id=TURN, specialist="cfo", position=position,
            claim_ref=proposal.claim_ref, content=content,
        )
        assert request.correlation.specialist == "cfo"
        assert request.correlation.case_id == SESSION
        assert request.correlation.run_id == TURN
        assert request.correlation.claim_id == proposal.claim_ref


def test_no_evidence_is_verified_and_no_claim_moves(flag_on: None) -> None:
    out, _ = _run_parallel(_message(_payload()))
    assert out.calculations is not None and out.specialist_result is not None
    for record in out.calculations.results:
        assert record.evidence.status == "EVIDENCE_UNAVAILABLE"
        assert record.is_verified_evidence() is False
    for claim in out.specialist_result.claims:
        if claim.calculation is not None:
            assert claim.calculation.verification_status == "unverified"
            assert claim.calculation.verified_result is None
    with pytest.raises(ValueError):
        CalculationProvenance(method="m", verified_result="1")


def test_contradictory_narrative_is_returned_verbatim_beside_the_record(flag_on: None) -> None:
    out, mock = _run_parallel(_message(_payload()))
    assert out.narrative == NARRATIVE                       # byte-for-byte, unmodified
    assert "572" in out.narrative                           # the contradiction is still there
    assert out.calculations is not None
    assert out.calculations.results[5].result_value == "5148.00"
    assert out.calculations.results[5].conflict == "ORDER_OF_MAGNITUDE"
    mock.assert_awaited_once()                              # G6


def test_narrative_equals_the_legacy_rule_for_the_same_message(flag_on: None) -> None:
    """G10: same message through analyze() and analyze_structured() → same bytes."""
    message = _message(_payload(), "first text block")
    out, _ = _run_parallel(message)
    agent = FinanceAgent()
    with (
        patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)),
        set_turn(session_id=SESSION, turn_id=TURN),
    ):
        legacy = asyncio.run(agent.analyze(query="q"))
    # analyze() parses with the key UNKNOWN → degraded → first text block,
    # which is today's rule; the structured path with the flag on parses it
    # cleanly and returns the narrative. Both are the parser's own outputs.
    assert legacy == "first text block"
    assert out.narrative == NARRATIVE
    # And with the flag OFF the structured path is byte-identical to analyze().
    with patch.dict("os.environ", {FLAG: "false"}):
        off, _ = _run_parallel(message)
    assert off.narrative == legacy


# ---------------------------------------------------------------------------
# G3 / G4: isolation and fail-closed gates
# ---------------------------------------------------------------------------


def test_one_malformed_proposal_costs_nothing_else(flag_on: None) -> None:
    forged = {"operation": "add", "operands": [_operand("a", "1", "pct")], "purpose": "x",
              "target_unit": {"code": "pct"}, "request_id": "attacker", "fingerprint": "f" * 64,
              "computed_at": "1970-01-01T00:00:00Z", "authority": {"authority_id": "x"},
              "arithmetic_status": "ARITHMETIC_VERIFIED", "verified_result": "1"}
    out, _ = _run_parallel(_message(_payload(calculation_requests=[*PROPOSALS, forged, "junk"])))
    result = out.specialist_result
    assert result is not None and result.integrity == "partial" and result.degraded
    assert "2 calculation request(s) could not be read" in (result.degraded_reason or "")
    assert out.calculations is not None and len(out.calculations.results) == 7
    assert [r.result_value for r in out.calculations.results] == [e[1] for e in EXPECTED]


def test_a_claim_ref_to_a_claim_not_in_the_final_set_is_dropped(flag_on: None) -> None:
    proposals = [*PROPOSALS, _proposal("subtract", [_operand("a", "2", "pct"),
                                                    _operand("b", "1", "pct")],
                                       "pct", claim_ref="c9")]
    out, _ = _run_parallel(_message(_payload(calculation_requests=proposals)))
    assert out.calculations is not None
    assert len(out.calculations.results) == 7
    assert out.calculations.dropped == ("proposal_7_unknown_claim_ref",)


def test_a_duplicate_claim_id_loses_the_structure_and_skips_the_gateway(flag_on: None) -> None:
    claims = [*CLAIMS, {"claim_id": "c1", "text": "dup", "claim_type": "assessment"}]
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=AssertionError("gateway must not run"))):
        out, _ = _run_parallel(_message(_payload(claims=claims)))
    result = out.specialist_result
    assert result is not None and result.integrity == "lost"
    assert result.claims == () and result.calculation_requests == ()
    assert out.calculations is None and out.frame is None
    assert out.diagnostics == (CALC_SKIPPED_STRUCTURE_LOST,)


def test_a_claim_entry_that_is_not_an_object_keeps_the_survivors_authoritative(flag_on: None) -> None:
    claims: list[Any] = [*CLAIMS, "not an object"]
    out, _ = _run_parallel(_message(_payload(claims=claims)))
    result = out.specialist_result
    assert result is not None and result.integrity == "partial"
    assert [c.claim_id for c in result.claims] == ["c1", "c2", "c3", "c4", "c5"]
    assert out.calculations is not None and len(out.calculations.results) == 7


def test_an_unsafe_claim_id_refuses_the_whole_authorization_set(flag_on: None) -> None:
    claims = [*CLAIMS, {"claim_id": "c6\n", "text": "t", "claim_type": "assessment"}]
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=AssertionError("gateway must not run"))):
        out, _ = _run_parallel(_message(_payload(claims=claims)))
    assert out.diagnostics == (CALC_CLAIM_SET_UNSAFE,)
    assert out.calculations is None
    assert out.narrative == NARRATIVE


def test_no_bound_turn_fails_closed_with_no_minted_ids(flag_on: None) -> None:
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=AssertionError("gateway must not run"))):
        out, _ = _run_parallel(_message(_payload()), bound=False)
    assert out.diagnostics == (CALC_CONTEXT_UNAVAILABLE,)
    assert out.calculations is None and out.frame is None
    assert out.narrative == NARRATIVE


def test_an_unsafe_turn_id_fails_closed(flag_on: None) -> None:
    mock = AsyncMock(return_value=_message(_payload()))

    async def go() -> list[RoutedSpecialistOutput]:
        return await route_parallel(calls=[{"specialist": "cfo", "query": "q"}],
                                    retrieved_knowledge_map={})

    with (
        patch.object(FinanceAgent, "analyze_with_tools", mock),
        set_turn(session_id=SESSION, turn_id=" padded"),
    ):
        out = asyncio.run(go())[0]
    assert out.diagnostics == (CALC_CONTEXT_UNAVAILABLE,)


def test_too_many_proposals_is_a_declared_condition_not_a_crash(flag_on: None) -> None:
    """Security review: 33+ proposals used to surface as calc_gateway_raised."""
    from openexecutive.specialists.calculation_gateway import MAX_PROPOSALS_PER_CALL

    flood = [PROPOSALS[1]] * (MAX_PROPOSALS_PER_CALL + 1)
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=AssertionError("gateway must not run"))):
        out, _ = _run_parallel(_message(_payload(calculation_requests=flood)))
    assert out.diagnostics == (CALC_TOO_MANY_PROPOSALS,)
    assert out.calculations is None and out.frame is None       # gated before the frame
    assert out.specialist_result is not None
    assert len(out.specialist_result.calculation_requests) == MAX_PROPOSALS_PER_CALL + 1
    assert out.narrative == NARRATIVE
    # Exactly at the limit still executes.
    out, _ = _run_parallel(
        _message(_payload(calculation_requests=[PROPOSALS[1]] * MAX_PROPOSALS_PER_CALL))
    )
    assert out.calculations is not None
    assert len(out.calculations.results) == MAX_PROPOSALS_PER_CALL


def test_no_proposals_is_its_own_diagnostic(flag_on: None) -> None:
    out, _ = _run_parallel(_message(_payload(calculation_requests=[])))
    assert out.diagnostics == (CALC_NO_PROPOSALS,)
    assert out.calculations is None
    assert out.specialist_result is not None and out.specialist_result.integrity == "intact"


def test_a_raising_gateway_is_contained_as_a_diagnostic(flag_on: None) -> None:
    with patch.object(finance_module, "execute_proposals",
                      MagicMock(side_effect=RuntimeError("boom"))):
        out, _ = _run_parallel(_message(_payload()))
    assert out.diagnostics == (CALC_GATEWAY_RAISED,)
    assert out.calculations is None and out.frame is not None
    assert out.narrative == NARRATIVE


def test_the_diagnostic_log_line_is_a_literal(flag_on: None) -> None:
    """Asserted on the logger call itself, so no logging configuration another
    test leaves behind (handlers, propagation) can change the outcome."""
    warning = MagicMock()
    with (
        patch.object(finance_module, "execute_proposals",
                     MagicMock(side_effect=RuntimeError("SECRET-DETAIL\nforged line"))),
        patch.object(finance_module.logger, "warning", warning),
    ):
        _run_parallel(_message(_payload()))
    gateway_calls = [c for c in warning.call_args_list if "gateway" in str(c.args[0])]
    assert len(gateway_calls) == 1
    assert gateway_calls[0].args == ("cfo: calculation gateway raised",)
    assert "SECRET-DETAIL" not in str(gateway_calls[0])


# ---------------------------------------------------------------------------
# G5 / G6: analyze() never reaches the gateway; the provider is called once
# ---------------------------------------------------------------------------


def test_analyze_never_calls_the_gateway_even_with_proposals_and_flag(flag_on: None) -> None:
    agent = FinanceAgent()
    mock = AsyncMock(return_value=_message(_payload(), "legacy text"))
    with (
        patch.object(FinanceAgent, "analyze_with_tools", mock),
        patch.object(finance_module, "execute_proposals",
                     MagicMock(side_effect=AssertionError("gateway must not run"))),
        set_turn(session_id=SESSION, turn_id=TURN),
    ):
        text = asyncio.run(agent.analyze(query="q"))
    assert text == "legacy text"
    mock.assert_awaited_once()
    assert agent.last_result is not None
    assert agent.last_result.calculation_requests == ()


def test_analyze_structured_calls_the_provider_exactly_once(flag_on: None) -> None:
    _, mock = _run_parallel(_message(_payload()))
    mock.assert_awaited_once()


def test_execute_proposals_is_called_from_one_method_only() -> None:
    tree = ast.parse((PACKAGE_ROOT / "agents" / "finance.py").read_text(encoding="utf-8"))
    callers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                        and inner.func.id == "execute_proposals"):
                    callers.add(node.name)
    assert callers == {"_gated_calculations"}
    # …and _gated_calculations is called from analyze_structured only.
    users: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "_gated_calculations"):
                    users.add(node.name)
    assert users == {"analyze_structured"}


def test_finance_imports_the_gateway_and_never_the_calc_package() -> None:
    source = (PACKAGE_ROOT / "agents" / "finance.py").read_text(encoding="utf-8")
    assert "from openexecutive.specialists.calculation_gateway import" in source
    assert "openexecutive.calc" not in source


# ---------------------------------------------------------------------------
# ParseIntegrity is parser-authored, never model-selected
# ---------------------------------------------------------------------------


def test_the_parser_default_leaves_calculation_requests_unknown() -> None:
    """Every legacy caller omits the kwarg; for them the key must stay unknown."""
    result = parse_specialist_result(_message(_payload()), specialist="cfo")
    assert result.calculation_requests == ()
    assert result.integrity == "partial"
    assert "1 unrecognized top-level payload key(s)" in (result.degraded_reason or "")


@pytest.mark.parametrize("accept", [False, True])
def test_a_payload_cannot_select_intact(accept: bool) -> None:
    result = parse_specialist_result(
        _message({"narrative": "n", "claims": [], "integrity": "intact"}),
        specialist="cfo", accept_calculation_requests=accept,
    )
    assert result.integrity == "partial" and result.degraded
    assert "1 unrecognized top-level payload key(s)" in (result.degraded_reason or "")


def test_a_payload_cannot_smuggle_integrity_or_degraded_through_a_proposal() -> None:
    proposal = {**PROPOSALS[0], "integrity": "intact"}
    result = parse_specialist_result(
        _message({"narrative": "n", "claims": [], "calculation_requests": [proposal]}),
        specialist="cfo", accept_calculation_requests=True,
    )
    assert result.calculation_requests == ()
    assert "1 calculation request(s) could not be read" in (result.degraded_reason or "")


def test_integrity_derives_from_degraded_when_absent_and_is_checked_when_present() -> None:
    assert SpecialistResult(specialist="cfo", narrative="n").integrity == "intact"
    assert SpecialistResult(specialist="cfo", narrative="n", degraded=True,
                            degraded_reason="r").integrity == "partial"
    with pytest.raises(ValueError):
        SpecialistResult(specialist="cfo", narrative="n", integrity="lost")          # not degraded
    with pytest.raises(ValueError):
        SpecialistResult(specialist="cfo", narrative="n", degraded=True,
                         degraded_reason="r", integrity="intact")
    with pytest.raises(ValueError):
        SpecialistResult(specialist="cfo", narrative="n", degraded=True, degraded_reason="r",
                         integrity="lost",
                         claims=({"claim_id": "c1", "text": "t", "claim_type": "assessment"},))  # type: ignore[arg-type]


def test_a_deserialized_result_establishes_nothing() -> None:
    """Construction with integrity="intact" is valid and proves no provenance."""
    forged = SpecialistResult.model_validate(
        {"specialist": "cfo", "narrative": "n", "integrity": "intact"}
    )
    assert forged.integrity == "intact"   # constructible — which is exactly why the
    # gate keys on the in-process parse exit and the frame, never on this field alone.
