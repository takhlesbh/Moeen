"""CFO structured runtime wiring.

CFO is the only specialist routed through the structured contract. These tests
exist to prove two things at once: that the structured path works, and that
nothing downstream can tell the difference — ``route_to_specialist`` still
returns ``str``, the other specialists are untouched, and a degraded CFO call
returns exactly the bytes it returns today.
"""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from openexecutive.agents.finance import FinanceAgent
from openexecutive.specialists.result_contract import (
    EMIT_SPECIALIST_RESULT_TOOL,
    SpecialistResult,
    emit_specialist_result_tool,
)

TOOL_NAME = EMIT_SPECIALIST_RESULT_TOOL["name"]


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(payload: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="tu1", name=TOOL_NAME, input=payload)


def _message(*blocks: Any) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks))


def _run_cfo(message: Any, **kwargs: Any) -> tuple[str, AsyncMock]:
    """Drive CFO.analyze against a stubbed provider response."""
    agent = FinanceAgent()
    mock = AsyncMock(return_value=message)
    with patch.object(FinanceAgent, "analyze_with_tools", mock):
        out = asyncio.run(agent.analyze(query="What is our runway?", **kwargs))
    return out, mock


# ---------------------------------------------------------------------------
# 1. structured success
# ---------------------------------------------------------------------------


def test_cfo_valid_structured_result_returns_narrative() -> None:
    message = _message(_tool_use({
        "narrative": "Runway is 7.4 months at the current net burn.",
        "claims": [
            {"claim_id": "c1", "text": "Cash on hand is $3.03m",
             "claim_type": "source_fact", "attribution": "independent_evidence",
             "evidence": [{"kind": "document", "label": "[Q3-deck.pdf]",
                           "filename": "Q3-deck.pdf"}]},
            {"claim_id": "c2", "text": "Runway is 7.4 months",
             "claim_type": "derived_calculation",
             "calculation": {"inputs": ["cash $3.03m", "burn $410k/mo"],
                             "method": "cash / net burn",
                             "model_stated_result": "7.4 months"}},
        ],
    }))
    out, _ = _run_cfo(message)

    assert out == "Runway is 7.4 months at the current net burn."
    assert isinstance(out, str)


def test_cfo_retains_the_structured_result_internally() -> None:
    """Claims are parsed and kept, but only the narrative crosses the boundary."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact"}],
    }))
    agent = FinanceAgent()
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(agent.analyze(query="q"))

    assert out == "n"
    assert isinstance(agent.last_result, SpecialistResult)
    assert [c.claim_id for c in agent.last_result.claims] == ["c1"]
    assert agent.last_result.degraded is False


def test_cfo_passes_the_structured_tool() -> None:
    _, mock = _run_cfo(_message(_tool_use({"narrative": "n", "claims": []})))
    tools = mock.await_args.kwargs["tools"]
    assert [t["name"] for t in tools] == [TOOL_NAME]


# ---------------------------------------------------------------------------
# 2 & 7. prose-only degraded fallback — LEGACY semantics preserved
# ---------------------------------------------------------------------------


def test_cfo_prose_only_returns_the_text_unchanged() -> None:
    """The expected case on backends that drop tool_choice, e.g. Ollama."""
    prose = "Runway is tighter than the last board update implied."
    out, _ = _run_cfo(_message(_text(prose)))
    assert out == prose


def test_cfo_degraded_preserves_first_text_block_not_a_join() -> None:
    """The migration gate.

    ``BaseAgent.analyze`` returns the FIRST text block; the contract's parser
    joins all of them. CFO must keep the old rule, or every existing caller
    silently receives different text.
    """
    out, _ = _run_cfo(_message(_text("first"), _text("second")))
    assert out == "first"
    assert out != "first\n\nsecond"


def test_cfo_degraded_matches_base_agent_semantics_exactly() -> None:
    """Byte-identical to what agents/base.py:141-142 would have returned."""
    for blocks in (
        [_text("only")],
        [_text("first"), _text("second")],
        [],
        [SimpleNamespace(type="tool_use", id="x", name="other", input={})],
    ):
        message = _message(*blocks)
        legacy = next(
            (b.text for b in message.content if getattr(b, "type", "") == "text"), ""
        )
        out, _ = _run_cfo(message)
        assert out == legacy, blocks


def test_cfo_empty_response_returns_empty_string() -> None:
    out, _ = _run_cfo(_message())
    assert out == ""


def test_cfo_prose_only_records_a_degraded_result() -> None:
    agent = FinanceAgent()
    with patch.object(
        FinanceAgent, "analyze_with_tools", AsyncMock(return_value=_message(_text("p")))
    ):
        asyncio.run(agent.analyze(query="q"))
    assert agent.last_result is not None
    assert agent.last_result.degraded is True
    assert agent.last_result.claims == ()


# ---------------------------------------------------------------------------
# 3 & 10. malformed / multiple tool blocks
# ---------------------------------------------------------------------------


def test_cfo_malformed_tool_payload_degrades_to_legacy_text() -> None:
    out, _ = _run_cfo(_message(_text("fallback prose"), _tool_use("not an object")))
    assert out == "fallback prose"


def test_cfo_invalid_claim_degrades_and_keeps_usable_text() -> None:
    message = _message(_text("fallback prose"), _tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "not_a_type"}],
    }))
    out, _ = _run_cfo(message)
    assert out == "fallback prose"


def test_cfo_multiple_tool_blocks_stay_honest() -> None:
    """A second tool block degrades the result but keeps the first narrative.

    There is no text block here, so legacy would have been ``""`` — the
    narrative fallback applies and the caller gets the model's actual answer
    rather than nothing. The loss is still reported in ``degraded_reason``.
    """
    agent = FinanceAgent()
    message = _message(
        _tool_use({"narrative": "first", "claims": []}),
        _tool_use({"narrative": "second", "claims": []}),
    )
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(agent.analyze(query="q"))

    assert out == "first"
    assert agent.last_result is not None
    assert agent.last_result.degraded is True
    assert "2 emit_specialist_result blocks" in (agent.last_result.degraded_reason or "")


def test_degraded_with_no_text_block_falls_back_to_the_narrative() -> None:
    """Found by the live Qwen smoke, not by the mocks.

    Qwen reliably calls the tool and puts its whole answer in the payload,
    leaving no prose. When such a payload fails validation there is no text
    block, so returning the legacy ``""`` would hand the Executive an empty
    answer where today it receives prose — a regression created by wiring.

    Safe here specifically: with no text block there is no "first block" to
    preserve, so this can never be the forbidden join.
    """
    agent = FinanceAgent()
    message = _message(_tool_use({
        "narrative": "Your burn multiple is 1.41x, which is in the wasteful band.",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact",
                    "evidence": [{"kind": "not_a_kind", "label": "x"}]}],
    }))
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(agent.analyze(query="q"))

    assert agent.last_result is not None and agent.last_result.degraded is True
    assert out == "Your burn multiple is 1.41x, which is in the wasteful band."
    assert out != ""


def test_narrative_fallback_never_produces_a_join() -> None:
    """The fallback must not become a back door around legacy semantics."""
    agent = FinanceAgent()
    message = _message(
        _text("first"), _text("second"),
        _tool_use({"narrative": "n", "claims": [{"claim_id": "c1", "text": "t",
                                                 "claim_type": "bogus"}]}),
    )
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(agent.analyze(query="q"))

    assert out == "first"
    assert out != "first\n\nsecond"


def test_partial_degradation_still_matches_legacy_when_text_exists() -> None:
    """Same rule with prose present: the first text block, never the narrative."""
    agent = FinanceAgent()
    message = _message(
        _text("legacy prose"),
        _tool_use({"narrative": "structured", "claims": [], "Claims": []}),
    )
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(agent.analyze(query="q"))

    assert out == "legacy prose"
    assert agent.last_result is not None and agent.last_result.degraded is True


@pytest.mark.parametrize(
    "content", [42, None, "plain string", object()],
)
def test_cfo_never_raises_on_a_hostile_provider_response(content: Any) -> None:
    out, _ = _run_cfo(SimpleNamespace(content=content))
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# 4 & 5. calculation authority and provenance
# ---------------------------------------------------------------------------


def test_cfo_cannot_establish_a_verified_result() -> None:
    agent = FinanceAgent()
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "IRR is 24%", "claim_type": "derived_calculation",
            "calculation": {"inputs": ["cash flows"], "method": "IRR",
                            "model_stated_result": "24%",
                            "verified_result": "24%",
                            "verification_status": "verified"},
        }],
    }))
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        asyncio.run(agent.analyze(query="q"))

    calc = agent.last_result.claims[0].calculation  # type: ignore[union-attr]
    assert calc is not None
    assert calc.model_stated_result == "24%"
    assert calc.verified_result is None
    assert calc.verification_status == "unverified"


def test_cfo_invented_provenance_is_stripped() -> None:
    agent = FinanceAgent()
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "source_fact",
            "evidence": [{"kind": "document", "label": "[deck.pdf]",
                          "filename": "deck.pdf", "page": 12,
                          "url": "https://example.invalid", "sheet": "FY24",
                          "cell_range": "B2:B9", "retrieved_at": "2026-08-30",
                          "provenance_note": "confirmed on page 12"}],
        }],
    }))
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        asyncio.run(agent.analyze(query="q"))

    ref = agent.last_result.claims[0].evidence[0]  # type: ignore[union-attr]
    assert ref.filename == "deck.pdf"          # model-asserted, kept
    assert ref.page is None
    assert ref.url is None
    assert ref.sheet is None
    assert ref.cell_range is None
    assert ref.retrieved_at is None
    assert ref.provenance_note is None


# ---------------------------------------------------------------------------
# 8 & 9. blast radius — non-CFO specialists and the router
# ---------------------------------------------------------------------------


def test_non_cfo_specialists_still_use_base_analyze_with_no_tool() -> None:
    """The other nine specialists must not have moved."""
    from openexecutive.agents.base import BaseAgent
    from openexecutive.orchestrator.router import SPECIALIST_REGISTRY

    for name, agent in SPECIALIST_REGISTRY.items():
        if name == "cfo":
            assert type(agent).analyze is not BaseAgent.analyze
            continue
        assert type(agent).analyze is BaseAgent.analyze, name


def test_non_cfo_specialist_sends_no_structured_tool() -> None:
    from openexecutive.orchestrator.router import SPECIALIST_REGISTRY

    cso = SPECIALIST_REGISTRY["cso"]
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _message(_text("strategy answer"))

    with patch("openexecutive.agents.base.get_provider") as gp:
        gp.return_value = SimpleNamespace(messages_create=fake_create)
        out = asyncio.run(cso.analyze(query="q"))

    assert out == "strategy answer"
    assert "tools" not in captured


def test_route_to_specialist_cfo_still_returns_str() -> None:
    from openexecutive.orchestrator.router import route_to_specialist

    message = _message(_tool_use({"narrative": "structured answer", "claims": []}))
    with patch.object(FinanceAgent, "analyze_with_tools", AsyncMock(return_value=message)):
        out = asyncio.run(route_to_specialist(specialist_name="cfo", query="q"))

    assert isinstance(out, str)
    assert out == "structured answer"


def test_cfo_analyze_signature_is_substitutable_for_base_agent() -> None:
    """route_to_specialist and the sandbox call this by keyword.

    CFO may take parameters the base does not — ``retrieval_set`` is one — but
    only additively: every base parameter must keep its name, position, kind and
    default, and each extra must be keyword-only WITH a default. Together those
    make CFO callable exactly as any other specialist, which is what the ~90
    positional/keyword call sites actually depend on. Asserting the signatures
    are *identical* would be stricter than the property that matters and would
    block any per-agent capability from ever being added.
    """
    import inspect

    from openexecutive.agents.base import BaseAgent

    base = inspect.signature(BaseAgent.analyze).parameters
    cfo = inspect.signature(FinanceAgent.analyze).parameters

    # Base parameters come first, in order, unchanged.
    assert list(cfo)[: len(base)] == list(base)
    for name, param in base.items():
        assert cfo[name].kind == param.kind, name
        assert cfo[name].default == param.default, name

    for name in list(cfo)[len(base):]:
        extra = cfo[name]
        assert extra.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert extra.default is not inspect.Parameter.empty, name


def test_router_forwards_every_context_block_to_cfo() -> None:
    """A dropped block would silently strip RAG or memory from CFO's prompt."""
    from openexecutive.orchestrator.router import route_to_specialist

    captured: dict[str, Any] = {}

    async def fake_tools(self: Any, user_content: str, **kwargs: Any) -> Any:
        captured["user_content"] = user_content
        return _message(_text("ok"))

    with patch.object(FinanceAgent, "analyze_with_tools", fake_tools):
        asyncio.run(route_to_specialist(
            specialist_name="cfo", query="QUERY",
            context="CTX", retrieved_knowledge="KNOW",
            episodic_context="EPISODIC", failure_cases="FAILURES",
            department_memory="DEPT",
        ))

    content = captured["user_content"]
    for marker in ("QUERY", "CTX", "KNOW", "EPISODIC", "FAILURES", "DEPT"):
        assert marker in content, marker
    for tag in ("conversation_context", "relevant_knowledge", "past_decisions",
                "failure_cases", "department_memory"):
        assert f"<{tag}>" in content, tag


def test_prompt_override_replaces_the_prompt_and_never_appends() -> None:
    """The Council sandbox previews an edit; it must run exactly what was typed.

    ``analyze_with_tools`` only accepts an *addendum*, so appending would run the
    stock CFO prompt IN FRONT OF the operator's edit. An operator loosening a
    guardrail would then see the original constraints still holding, judge the
    edit safe, and save a prompt that runs without them — failure in the unsafe
    direction, on the one tool meant to catch exactly that.
    """
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _message(_text("ok"))

    agent = FinanceAgent()
    with patch("openexecutive.agents.base.get_provider") as gp:
        gp.return_value = SimpleNamespace(messages_create=fake_create)
        out = asyncio.run(
            agent.analyze(query="q", system_prompt_override="CUSTOM PROMPT")
        )

    system = captured["system"][0]["text"]
    assert system == "CUSTOM PROMPT"
    assert "Chief Financial Officer" not in system
    assert out == "ok"
    # Legacy path: no structured tool, and no structured result recorded.
    assert "tools" not in captured


def test_prompt_override_matches_base_agent_exactly() -> None:
    """Byte-for-byte the same system block BaseAgent.analyze would have sent."""
    from openexecutive.agents.base import BaseAgent
    from openexecutive.agents.strategy import StrategyAgent

    seen: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _message(_text("ok"))

    with patch("openexecutive.agents.base.get_provider") as gp:
        gp.return_value = SimpleNamespace(messages_create=fake_create)
        asyncio.run(FinanceAgent().analyze(query="q", system_prompt_override="P"))
        asyncio.run(StrategyAgent().analyze(query="q", system_prompt_override="P"))

    assert seen[0]["system"] == seen[1]["system"]
    assert isinstance(BaseAgent.analyze, type(FinanceAgent.analyze))


def test_empty_prompt_override_is_honoured_like_every_other_agent() -> None:
    """`""` clears the prompt for other agents; CFO must not treat it as unset."""
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _message(_text("ok"))

    with patch("openexecutive.agents.base.get_provider") as gp:
        gp.return_value = SimpleNamespace(messages_create=fake_create)
        asyncio.run(FinanceAgent().analyze(query="q", system_prompt_override=""))

    assert captured["system"][0]["text"] == ""


# ---------------------------------------------------------------------------
# output budget — must not be coupled to thinking representability
# ---------------------------------------------------------------------------


def test_cfo_keeps_its_16k_budget_where_thinking_is_unrepresentable() -> None:
    """The trap: analyze_with_tools only raises max_tokens inside `if use_deep`.

    Turning deep reasoning off to quiet the provider gate would also quarter
    CFO's answer budget on every local call.
    """
    from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

    local = OpenAICompatibleProvider(base_url="http://local/v1")
    with patch("openexecutive.providers.get_provider", return_value=local):
        _, mock = _run_cfo(_message(_text("p")))

    assert mock.await_args.kwargs["deep_reasoning_override"] is False
    assert mock.await_args.kwargs["max_tokens"] == 16000


def test_cfo_budget_matches_base_agent_on_anthropic() -> None:
    from openexecutive.providers.anthropic_provider import AnthropicProvider

    with patch("openexecutive.providers.get_provider", return_value=AnthropicProvider(api_key="k")):
        _, mock = _run_cfo(_message(_text("p")))
    assert mock.await_args.kwargs["max_tokens"] == 16000


def test_budget_follows_a_council_deep_reasoning_override() -> None:
    """A Council DB override turning deep reasoning off should drop the budget."""
    agent = FinanceAgent()
    mock = AsyncMock(return_value=_message(_text("p")))
    with patch.object(FinanceAgent, "analyze_with_tools", mock), patch.object(
        FinanceAgent, "effective_use_deep_reasoning", return_value=False
    ):
        asyncio.run(agent.analyze(query="q"))
    assert mock.await_args.kwargs["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# capability resolution — spec, not provider class
# ---------------------------------------------------------------------------


def test_thinking_support_is_read_from_the_provider_spec() -> None:
    """Duck-typed on the spec resolver, so a future capable backend just works."""
    from openexecutive.providers.feature_gate import FeatureSpec

    capable = SimpleNamespace(
        _resolve=lambda m: (m, FeatureSpec(supports_thinking=True))
    )
    with patch("openexecutive.providers.get_provider", return_value=capable):
        _, mock = _run_cfo(_message(_text("p")))
    assert mock.await_args.kwargs["deep_reasoning_override"] is None

    incapable = SimpleNamespace(
        _resolve=lambda m: (m, FeatureSpec(supports_thinking=False))
    )
    with patch("openexecutive.providers.get_provider", return_value=incapable):
        _, mock = _run_cfo(_message(_text("p")))
    assert mock.await_args.kwargs["deep_reasoning_override"] is False


def test_empty_model_override_resolves_consistently() -> None:
    """`or` would label the result with a different model than the request used."""
    agent = FinanceAgent()
    with patch.object(
        FinanceAgent, "analyze_with_tools",
        AsyncMock(return_value=_message(_tool_use({"narrative": "n", "claims": []}))),
    ):
        asyncio.run(agent.analyze(query="q", model_override=""))
    assert agent.last_result is not None
    assert agent.last_result.model == ""


# ---------------------------------------------------------------------------
# 11 & 12. tool factory independence
# ---------------------------------------------------------------------------


def test_tool_factory_returns_independent_copies() -> None:
    a, b = emit_specialist_result_tool(), emit_specialist_result_tool()
    assert a == b
    assert a is not b
    assert a["input_schema"] is not b["input_schema"]

    a["name"] = "hijacked"
    a["input_schema"]["properties"]["claims"]["items"]["properties"][
        "claim_type"]["enum"].append("POISON")

    assert b["name"] == TOOL_NAME
    enum = b["input_schema"]["properties"]["claims"]["items"]["properties"][
        "claim_type"]["enum"]
    assert "POISON" not in enum


def test_module_template_survives_a_full_cfo_call() -> None:
    before = copy.deepcopy(EMIT_SPECIALIST_RESULT_TOOL)
    out, mock = _run_cfo(_message(_tool_use({"narrative": "n", "claims": []})))

    # Mutating what the agent handed the provider must not touch the template.
    passed = mock.await_args.kwargs["tools"][0]
    passed["input_schema"]["properties"]["narrative"]["description"] = "hijacked"

    assert before == EMIT_SPECIALIST_RESULT_TOOL
    assert out == "n"


def test_tool_factory_output_is_json_serialisable() -> None:
    assert json.loads(json.dumps(emit_specialist_result_tool())) == (
        EMIT_SPECIALIST_RESULT_TOOL
    )


# ---------------------------------------------------------------------------
# deep-reasoning handling on the local path
# ---------------------------------------------------------------------------


def test_deep_reasoning_disabled_when_the_backend_cannot_represent_it() -> None:
    """Otherwise the gate strips thinking and WARN-logs on every CFO call."""
    from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

    local = OpenAICompatibleProvider(base_url="http://local/v1")
    with patch("openexecutive.providers.get_provider", return_value=local):
        _, mock = _run_cfo(_message(_text("p")))
    assert mock.await_args.kwargs["deep_reasoning_override"] is False


def test_deep_reasoning_left_alone_on_anthropic() -> None:
    """CFO keeps use_deep_reasoning=True where thinking is representable."""
    from openexecutive.providers.anthropic_provider import AnthropicProvider

    anthropic = AnthropicProvider(api_key="k")
    with patch("openexecutive.providers.get_provider", return_value=anthropic):
        _, mock = _run_cfo(_message(_text("p")))
    assert mock.await_args.kwargs["deep_reasoning_override"] is None


def test_explicit_deep_reasoning_override_always_wins() -> None:
    from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

    local = OpenAICompatibleProvider(base_url="http://local/v1")
    with patch("openexecutive.providers.get_provider", return_value=local):
        _, mock = _run_cfo(_message(_text("p")), deep_reasoning_override=True)
    assert mock.await_args.kwargs["deep_reasoning_override"] is True


def test_provider_lookup_failure_does_not_break_the_call() -> None:
    with patch("openexecutive.providers.get_provider", side_effect=RuntimeError("boom")):
        out, mock = _run_cfo(_message(_text("prose")))
    assert out == "prose"
    assert mock.await_args.kwargs["deep_reasoning_override"] is None
