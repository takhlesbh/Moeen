"""Capability-honesty and retry parity for the OpenAI-compatible provider path.

The invariant these tests pin: a feature the caller REQUESTED is either
translated, or explicitly surfaced as unsupported. It is never silently
discarded. Before this, ``thinking`` / ``output_config`` were accepted by the
Claude-via-OpenRouter feature spec and then dropped on the floor by
``to_openai_request``, so a deep-reasoning specialist degraded to a plain
completion with no error and no log line.

Retry parity is the second half: ``anthropic.AsyncAnthropic`` retries twice by
default, the OpenAI-compatible provider retried never, so reliability depended
on which backend a model happened to route to.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from openexecutive.providers.feature_gate import (
    FeatureSpec,
    UnsupportedFeatureError,
    apply_feature_gates,
    unsupported_requested,
)
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider
from openexecutive.providers.translator import (
    StreamAccumulator,
    from_openai_response,
    reasoning_text,
    to_openai_request,
)

_NON_CLAUDE = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
)

_TRANSLATOR_LOGGER = "openexecutive.providers.translator"
_PROVIDER_LOGGER = "openexecutive.providers.openai_compatible"


@contextmanager
def capture_warnings(logger_name: str) -> Iterator[list[str]]:
    """Collect WARNING messages from one logger.

    Deliberately not pytest's ``caplog``: ``api.main._configure_logging`` sets
    ``propagate = False`` on the ``openexecutive`` logger, so once any test in
    the session imports it, records never reach the root handler caplog
    installs — the assertions would silently pass on an empty list. Attaching
    a handler to the emitting logger is immune to that.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _deep_reasoning_kwargs() -> dict[str, Any]:
    """The exact shape ``BaseAgent.analyze`` builds for a deep specialist."""
    return {
        "system": [{"type": "text", "text": "CFO", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "runway?"}],
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }


# ---------------------------------------------------------------------------
# A. Deep-reasoning request contract
# ---------------------------------------------------------------------------


def test_unsupported_requested_names_thinking_when_spec_forbids_it() -> None:
    assert unsupported_requested(_NON_CLAUDE, _deep_reasoning_kwargs()) == [
        "cache_control",
        "thinking",
    ]


def test_unsupported_requested_reports_output_config_alone_as_thinking() -> None:
    """``output_config.effort`` is half the deep-reasoning contract; asking for
    it without ``thinking`` must still be reported."""
    kwargs = {"messages": [], "output_config": {"effort": "low"}}
    assert unsupported_requested(_NON_CLAUDE, kwargs) == ["thinking"]


def test_unsupported_requested_is_empty_when_nothing_was_asked_for() -> None:
    """No false positives: a plain request reports no losses."""
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    assert unsupported_requested(_NON_CLAUDE, kwargs) == []


def test_unsupported_requested_is_empty_for_a_fully_capable_spec() -> None:
    assert unsupported_requested(FeatureSpec(), _deep_reasoning_kwargs()) == []


def test_unsupported_requested_reports_tool_choice_without_tools() -> None:
    """Regression: ``apply_feature_gates`` pops BOTH ``tools`` and
    ``tool_choice``, so a request carrying only ``tool_choice`` used to be
    stripped while the reporter returned [] — a silent loss of exactly the
    kind this module exists to prevent."""
    no_tools = FeatureSpec(
        supports_cache_control=True,
        supports_thinking=True,
        supports_web_search=True,
        supports_tool_use=False,
    )
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "tool_choice": {"type": "any"}}
    assert unsupported_requested(no_tools, kwargs) == ["tool_use"]
    assert "tool_choice" not in apply_feature_gates(no_tools, kwargs)


def test_reporter_and_gate_agree_across_every_capability() -> None:
    """Property-style parity check: for each capability the gate can strip,
    the reporter must name it. Guards against a future carrier location being
    added to one side only."""
    cases: list[tuple[str, dict[str, Any]]] = [
        ("cache_control", {"system": [{"type": "text", "text": "s",
                                       "cache_control": {"type": "ephemeral"}}]}),
        ("cache_control", {"tools": [{"name": "t", "input_schema": {},
                                      "cache_control": {"type": "ephemeral"}}]}),
        ("thinking", {"thinking": {"type": "adaptive"}}),
        ("thinking", {"output_config": {"effort": "low"}}),
        ("tool_use", {"tools": [{"name": "t", "input_schema": {}}]}),
        ("tool_use", {"tool_choice": {"type": "any"}}),
        ("web_search", {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
    ]
    nothing = FeatureSpec(False, False, False, False)
    for capability, kwargs in cases:
        reported = unsupported_requested(nothing, dict(kwargs))
        assert capability in reported, f"{capability} stripped but not reported: {kwargs}"


def test_unsupported_requested_detects_web_search_and_tool_use() -> None:
    no_tools = FeatureSpec(
        supports_cache_control=True,
        supports_thinking=True,
        supports_web_search=False,
        supports_tool_use=False,
    )
    kwargs = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
    assert unsupported_requested(no_tools, kwargs) == ["tool_use", "web_search"]


def test_unsupported_requested_does_not_mutate_its_input() -> None:
    kwargs = _deep_reasoning_kwargs()
    unsupported_requested(_NON_CLAUDE, kwargs)
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_reporter_and_gate_agree_on_cache_control_in_message_blocks() -> None:
    """The reporter must find exactly what the gate strips — including the
    rolling user-turn cache marker, not just system/tool blocks."""
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ]
    }
    assert unsupported_requested(_NON_CLAUDE, kwargs) == ["cache_control"]
    gated = apply_feature_gates(_NON_CLAUDE, kwargs)
    assert "cache_control" not in gated["messages"][0]["content"][0]


def test_to_openai_request_refuses_unrepresentable_thinking() -> None:
    """The invariant guard: reaching translation with a field we cannot encode
    is an error, not a silent omission."""
    with pytest.raises(UnsupportedFeatureError) as exc:
        to_openai_request("some/model", _deep_reasoning_kwargs())
    assert "thinking" in str(exc.value)
    assert "output_config" in str(exc.value)
    assert exc.value.features == ["thinking", "output_config"]


def test_to_openai_request_accepts_a_properly_gated_request() -> None:
    """Post-gate, the same request translates cleanly — the guard only fires
    when the gate was bypassed."""
    gated = apply_feature_gates(_NON_CLAUDE, _deep_reasoning_kwargs())
    body = to_openai_request("some/model", gated)
    assert body["model"] == "some/model"
    assert "thinking" not in body
    assert "output_config" not in body


def _mock_ok_provider() -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"llama3.3": _NON_CLAUDE},
    )
    fake = MagicMock()
    fake.json.return_value = {
        "id": "x",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }
    fake.raise_for_status = MagicMock()
    provider._client.post = AsyncMock(return_value=fake)  # type: ignore[method-assign]
    return provider


def test_provider_logs_every_dropped_capability() -> None:
    """The end-to-end anti-silence guarantee, at the real call site."""
    provider = _mock_ok_provider()
    with capture_warnings(_PROVIDER_LOGGER) as messages:
        asyncio.run(provider.messages_create(model="llama3.3", **_deep_reasoning_kwargs()))

    warning = "\n".join(messages)
    assert "llama3.3" in warning
    assert "thinking" in warning
    assert "cache_control" in warning
    assert "requested but not delivered" in warning


def test_provider_stays_quiet_when_nothing_is_dropped() -> None:
    """No crying wolf — a request that asks for nothing unsupported logs nothing."""
    provider = _mock_ok_provider()
    with capture_warnings(_PROVIDER_LOGGER) as messages:
        asyncio.run(
            provider.messages_create(
                model="llama3.3",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        )
    assert [m for m in messages if "requested but not delivered" in m] == []


def test_claude_via_openrouter_spec_declares_thinking_unsupported() -> None:
    """Registry-level pin: the Claude-through-OpenRouter spec must not claim a
    capability the translator cannot encode, or the gate never fires and the
    loss goes back to being invisible."""
    from openexecutive.providers.registry import _CLAUDE_FEATURE_SPEC

    assert _CLAUDE_FEATURE_SPEC.supports_thinking is False
    # The three that DO survive translation stay on.
    assert _CLAUDE_FEATURE_SPEC.supports_cache_control is True
    assert _CLAUDE_FEATURE_SPEC.supports_web_search is True
    assert _CLAUDE_FEATURE_SPEC.supports_tool_use is True


# ---------------------------------------------------------------------------
# B. Reasoning in responses
# ---------------------------------------------------------------------------


def test_reasoning_text_reads_both_known_field_names() -> None:
    assert reasoning_text({"reasoning": "step 1"}) == "step 1"
    assert reasoning_text({"reasoning_content": "step 2"}) == "step 2"
    assert reasoning_text({"content": "just prose"}) == ""
    assert reasoning_text(None) == ""
    assert reasoning_text({"reasoning": 42}) == ""


def test_response_reasoning_is_reported_not_fabricated() -> None:
    """A reasoning field must NOT become an Anthropic thinking block — we never
    asked for one, and inventing it would dress prose up as model reasoning."""
    body = {
        "id": "x",
        "model": "m",
        "choices": [
            {
                "message": {"content": "answer", "reasoning": "hidden chain"},
                "finish_reason": "stop",
            }
        ],
    }
    with capture_warnings(_TRANSLATOR_LOGGER) as messages:
        msg = from_openai_response(body)

    assert [b.type for b in msg.content] == ["text"]
    assert msg.content[0].text == "answer"
    assert "reasoning content" in "\n".join(messages)


def test_response_without_reasoning_logs_nothing() -> None:
    body = {
        "id": "x",
        "model": "m",
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
    }
    with capture_warnings(_TRANSLATOR_LOGGER) as messages:
        from_openai_response(body)
    assert [m for m in messages if "reasoning content" in m] == []


def test_streamed_reasoning_deltas_are_reported_not_emitted() -> None:
    acc = StreamAccumulator()
    events = []
    events += acc.feed({"id": "x", "choices": [{"delta": {"reasoning": "thinking..."}}]})
    events += acc.feed({"choices": [{"delta": {"content": "hi"}}]})
    events += acc.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})

    # The reasoning delta produced no consumer-visible event.
    assert [e.delta.text for e in events if e.type == "content_block_delta"] == ["hi"]

    with capture_warnings(_TRANSLATOR_LOGGER) as messages:
        final = acc.finalize()
    assert [b.type for b in final.content] == ["text"]
    assert final.content[0].text == "hi"
    assert "reasoning content" in "\n".join(messages)


# ---------------------------------------------------------------------------
# C. Retry parity
# ---------------------------------------------------------------------------


def _provider_with_responses(*outcomes: Any, **kw: Any) -> tuple[OpenAICompatibleProvider, list[int]]:
    """Provider whose POST yields each outcome in turn. Exceptions are raised,
    anything else is returned. Second element counts the attempts."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"llama3.3": _NON_CLAUDE},
        retry_backoff_s=0.0,  # keep the suite fast; bounds are what we assert
        **kw,
    )
    calls = [0]

    async def _post(*_a: Any, **_k: Any) -> Any:
        outcome = outcomes[min(calls[0], len(outcomes) - 1)]
        calls[0] += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    provider._client.post = _post  # type: ignore[method-assign]
    return provider, calls


def _ok() -> MagicMock:
    fake = MagicMock()
    fake.json.return_value = {
        "id": "x",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }
    fake.raise_for_status = MagicMock()
    return fake


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    response = httpx.Response(code, text="boom", request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def _msg(**kw: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "hi"}], **kw}


@pytest.mark.parametrize("code", [408, 409, 429, 500, 502, 503, 504])
def test_retries_transient_status_codes_then_succeeds(code: int) -> None:
    provider, calls = _provider_with_responses(_status_error(code), _ok())
    result = asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert result.content[0].text == "ok"
    assert calls[0] == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_does_not_retry_client_errors(code: int) -> None:
    """Retrying a caller error just multiplies the bill; it cannot succeed."""
    provider, calls = _provider_with_responses(_status_error(code))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert exc.value.response.status_code == code
    assert calls[0] == 1


def test_retries_transport_errors() -> None:
    provider, calls = _provider_with_responses(httpx.ConnectError("refused"), _ok())
    result = asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert result.content[0].text == "ok"
    assert calls[0] == 2


def test_retry_count_is_bounded_and_error_surfaces() -> None:
    """Never infinite: 2 retries = 3 attempts total, matching the Anthropic
    SDK default, and the final failure is raised rather than swallowed."""
    provider, calls = _provider_with_responses(_status_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert calls[0] == 3


def test_max_retries_is_configurable_and_zero_means_one_attempt() -> None:
    provider, calls = _provider_with_responses(_status_error(503), max_retries=0)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert calls[0] == 1


def test_negative_max_retries_cannot_disable_the_bound() -> None:
    provider, calls = _provider_with_responses(_status_error(503), max_retries=-5)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert calls[0] == 1


def test_success_on_first_attempt_does_not_retry() -> None:
    provider, calls = _provider_with_responses(_ok())
    asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert calls[0] == 1


def test_does_not_retry_read_side_failures_that_already_billed() -> None:
    """A ReadTimeout means the request LANDED — the completion is already
    being generated and charged. Retrying would pay up to 3x for one logical
    call, so read-side failures propagate on the first attempt."""
    for exc in (
        httpx.ReadTimeout("slow"),
        httpx.ReadError("reset"),
        httpx.RemoteProtocolError("truncated"),
    ):
        provider, calls = _provider_with_responses(exc)
        with pytest.raises(type(exc)):
            asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
        assert calls[0] == 1, f"{type(exc).__name__} must not be retried"


def test_retries_connect_side_failures_that_never_landed() -> None:
    for exc in (httpx.ConnectError("refused"), httpx.ConnectTimeout("t"), httpx.PoolTimeout("p")):
        provider, calls = _provider_with_responses(exc, _ok())
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
        assert calls[0] == 2, f"{type(exc).__name__} should be retried"


def test_retry_delay_honours_server_retry_after() -> None:
    """Retrying faster than a rate limiter asked only deepens the limit."""
    from openexecutive.providers.openai_compatible import _retry_delay

    request = httpx.Request("POST", "http://x/v1/chat/completions")
    exc = httpx.HTTPStatusError(
        "429",
        request=request,
        response=httpx.Response(429, headers={"retry-after": "7"}, request=request),
    )
    # Jittered into [0.5x, 1.0x] of the requested delay — never sooner than
    # half of what the server asked, never longer than asked.
    assert 3.5 <= _retry_delay(1, exc, 0.5) <= 7.0


def test_retry_after_is_clamped_and_bad_values_ignored() -> None:
    from openexecutive.providers.openai_compatible import _MAX_RETRY_AFTER_S, _retry_delay

    request = httpx.Request("POST", "http://x/v1/chat/completions")

    def _exc(value: str) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError(
            "429",
            request=request,
            response=httpx.Response(429, headers={"retry-after": value}, request=request),
        )

    # Clamped to the cap, then jittered into [0.5x, 1.0x] of it.
    assert _MAX_RETRY_AFTER_S * 0.5 <= _retry_delay(1, _exc("99999"), 0.5) <= _MAX_RETRY_AFTER_S
    # Junk / HTTP-date / negative fall back to jittered backoff, never crash.
    for junk in ("Wed, 21 Oct 2015 07:28:00 GMT", "-5", "abc"):
        assert 0.0 <= _retry_delay(1, _exc(junk), 0.5) <= 0.5


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_retry_after_rejects_non_finite_values(value: str) -> None:
    """A backend-controlled ``Retry-After: nan`` must never reach asyncio.sleep.

    ``float("nan") < 0`` is False (every NaN comparison is), and
    ``min(nan, cap)`` is NaN — so a bare clamp let NaN through. On Python
    3.11/3.12 (what Docker/CI run) ``asyncio.sleep(nan)`` returns almost
    immediately, defeating the whole rate-limit backoff with one header; on
    3.13+ it raises a ValueError that ``_with_retry`` does not catch.
    """
    import math

    from openexecutive.providers.openai_compatible import _MAX_RETRY_AFTER_S, _retry_delay

    request = httpx.Request("POST", "http://x/v1/chat/completions")
    exc = httpx.HTTPStatusError(
        "429",
        request=request,
        response=httpx.Response(429, headers={"retry-after": value}, request=request),
    )
    delay = _retry_delay(1, exc, 0.5)
    assert math.isfinite(delay), f"non-finite delay from Retry-After: {value!r}"
    assert 0.0 <= delay <= max(_MAX_RETRY_AFTER_S, 0.5)


def test_retry_after_is_jittered_to_avoid_lockstep() -> None:
    """A fan-out that all receive the same Retry-After must not wake together."""
    from openexecutive.providers.openai_compatible import _retry_delay

    request = httpx.Request("POST", "http://x/v1/chat/completions")
    exc = httpx.HTTPStatusError(
        "429",
        request=request,
        response=httpx.Response(429, headers={"retry-after": "5"}, request=request),
    )
    samples = [_retry_delay(1, exc, 0.5) for _ in range(40)]
    assert len(set(samples)) > 1, "Retry-After delays identical — jitter missing"
    # Never sooner than half of what the server asked, never longer than asked.
    assert all(2.5 <= s <= 5.0 for s in samples)


def test_error_body_control_chars_are_neutralized() -> None:
    """An error body is backend-controlled; embedded newlines would let it
    forge extra log lines."""
    from openexecutive.providers.openai_compatible import _sanitize_for_log

    out = _sanitize_for_log("boom\nERROR forged log line\r\n\x00tail")
    assert "\n" not in out and "\r" not in out and "\x00" not in out
    assert "forged log line" in out  # content preserved, framing neutralized


def test_backoff_is_jittered_and_capped() -> None:
    """Without jitter a parallel specialist fan-out retries in lockstep and
    re-creates the burst that caused the 429."""
    from openexecutive.providers.openai_compatible import _MAX_RETRY_BACKOFF_S, _retry_delay

    exc = httpx.ConnectError("refused")
    samples = [_retry_delay(3, exc, 0.5) for _ in range(40)]
    assert len(set(samples)) > 1, "delays are identical — jitter missing"
    assert all(0.0 <= s <= _MAX_RETRY_BACKOFF_S for s in samples)
    # Even a huge attempt number stays under the cap.
    assert all(_retry_delay(50, exc, 0.5) <= _MAX_RETRY_BACKOFF_S for _ in range(10))


def test_error_body_is_logged_once_on_final_failure_only() -> None:
    """The body can echo request fragments, so a retried 503 must not print it
    on every attempt — nor at all when a later attempt succeeds."""
    from openexecutive.providers.openai_compatible import _ERROR_BODY_LOG_CHARS

    assert _ERROR_BODY_LOG_CHARS == 500  # pinned; used to truncate bodies

    # Succeeds on attempt 2 -> body never logged.
    provider, _ = _provider_with_responses(_status_error(503), _ok())
    with capture_warnings(_PROVIDER_LOGGER) as messages:
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert not any("boom" in m for m in messages), "body logged on a successful call"

    # Exhausts retries -> body logged exactly once.
    provider, _ = _provider_with_responses(_status_error(503))
    with capture_warnings(_PROVIDER_LOGGER) as messages, pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert sum("boom" in m for m in messages) == 1


def test_transport_failure_is_never_silent() -> None:
    """The final, non-retried transport error must still produce a log line."""
    provider, _ = _provider_with_responses(httpx.ReadTimeout("slow"))
    with capture_warnings(_PROVIDER_LOGGER) as messages, pytest.raises(httpx.ReadTimeout):
        asyncio.run(provider.messages_create(model="llama3.3", **_msg()))
    assert any("ReadTimeout" in m for m in messages)


def test_provider_forces_thinking_unsupported_on_every_spec() -> None:
    """``FeatureSpec()`` defaults ``supports_thinking=True``; a default-built
    spec reaching the provider would skip the gate and make the translator
    raise mid-turn. Normalizing at construction keeps that raise a true
    invariant guard rather than a latent 500."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"optimistic": FeatureSpec()},  # all four flags True
    )
    _, spec = provider._resolve("optimistic")
    assert spec.supports_thinking is False
    # The flags that DO survive translation are left alone.
    assert spec.supports_cache_control is True
    assert spec.supports_web_search is True

    # And the end-to-end path degrades honestly instead of raising.
    provider._client.post = AsyncMock(return_value=_ok())  # type: ignore[method-assign]
    with capture_warnings(_PROVIDER_LOGGER) as messages:
        asyncio.run(provider.messages_create(model="optimistic", **_deep_reasoning_kwargs()))
    assert any("thinking" in m for m in messages)


def test_failed_stream_open_never_orphans_a_connection() -> None:
    """Regression: the retry loop overwrote ``_response_cm`` with a fresh
    context, so a context left open by a failure was never exited — its
    connection never returned to the pool. Enough of those exhaust the pool
    and every later call blocks on PoolTimeout."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"llama3.3": _NON_CLAUDE},
        retry_backoff_s=0.0,
    )
    opened: list[int] = []
    closed: list[int] = []

    class _TrackingCM:
        def __init__(self, idx: int, mode: str) -> None:
            self._idx, self._mode = idx, mode

        async def __aenter__(self) -> Any:
            opened.append(self._idx)
            resp = MagicMock()
            if self._mode == "read_fails":
                # >=400, and reading the error body dies mid-read.
                resp.status_code = 503
                resp.aread = AsyncMock(side_effect=httpx.ReadError("cut"))
                return resp
            resp.status_code = 200

            async def _lines() -> Any:
                yield 'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}'
                yield "data: [DONE]"

            resp.aiter_lines = _lines
            return resp

        async def __aexit__(self, *a: Any) -> None:
            closed.append(self._idx)

    def _stream(*_a: Any, **_k: Any) -> Any:
        idx = len(opened) + 1
        return _TrackingCM(idx, "read_fails" if idx == 1 else "ok")

    provider._client.stream = _stream  # type: ignore[method-assign]

    async def _drive() -> str:
        async with provider.messages_stream(model="llama3.3", **_msg()) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()
        return final.content[0].text

    # A ReadError while buffering the error body is NOT retryable (read-side),
    # so this raises — but the first context must still be closed.
    with pytest.raises(httpx.ReadError):
        asyncio.run(_drive())
    assert opened == [1]
    assert closed == [1], f"orphaned context(s): {sorted(set(opened) - set(closed))}"


def test_stream_retry_closes_each_failed_attempt() -> None:
    """Same leak check on the path that DOES retry: a connect failure."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"llama3.3": _NON_CLAUDE},
        retry_backoff_s=0.0,
    )
    opened: list[int] = []
    closed: list[int] = []

    class _CM:
        def __init__(self, idx: int, fail: bool) -> None:
            self._idx, self._fail = idx, fail

        async def __aenter__(self) -> Any:
            if self._fail:
                # Fails before the context is considered entered.
                raise httpx.ConnectError("refused")
            opened.append(self._idx)
            resp = MagicMock()
            resp.status_code = 200

            async def _lines() -> Any:
                yield 'data: {"id":"x","choices":[{"delta":{"content":"ok"}}]}'
                yield "data: [DONE]"

            resp.aiter_lines = _lines
            return resp

        async def __aexit__(self, *a: Any) -> None:
            closed.append(self._idx)

    count = [0]

    def _stream(*_a: Any, **_k: Any) -> Any:
        count[0] += 1
        return _CM(count[0], fail=count[0] == 1)

    provider._client.stream = _stream  # type: ignore[method-assign]

    async def _drive() -> str:
        async with provider.messages_stream(model="llama3.3", **_msg()) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()
        return final.content[0].text

    assert asyncio.run(_drive()) == "ok"
    assert set(opened) == set(closed), f"orphaned: {sorted(set(opened) - set(closed))}"


def test_stream_retries_only_before_any_event_is_yielded() -> None:
    """Stream open is retried (nothing was emitted yet); mid-stream failures
    are not, because a half-delivered reply cannot be replayed."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"llama3.3": _NON_CLAUDE},
        retry_backoff_s=0.0,
    )
    calls = [0]

    class _FakeStreamCM:
        def __init__(self, fail: bool) -> None:
            self._fail = fail

        async def __aenter__(self) -> Any:
            if self._fail:
                raise httpx.ConnectError("refused")
            resp = MagicMock()
            resp.status_code = 200

            async def _lines() -> Any:
                yield 'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}'
                yield "data: [DONE]"

            resp.aiter_lines = _lines
            return resp

        async def __aexit__(self, *a: Any) -> None:
            return None

    def _stream(*_a: Any, **_k: Any) -> Any:
        calls[0] += 1
        return _FakeStreamCM(fail=calls[0] == 1)

    provider._client.stream = _stream  # type: ignore[method-assign]

    async def _drive() -> str:
        async with provider.messages_stream(model="llama3.3", **_msg()) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()
        return final.content[0].text

    assert asyncio.run(_drive()) == "hi"
    assert calls[0] == 2
